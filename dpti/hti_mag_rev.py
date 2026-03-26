#!/usr/bin/env python3
"""
Reverse HTI for magnetic spin systems (Plan 2 reversibility check).

Steps (Target → Reference):
  4. spin_spring_on   : H = H_deep + λ·H_spin_spring
  5. lattice_spring_on: H = H_deep + H_spin_spring + λ·H_latt_spring
  6. deep_off         : H = (1-λ)·H_deep + H_spin_spring + H_latt_spring

Integrand signs:
  spin_spring_on    : dH/dλ = +H_spin  → de = +all_esp / all_lambda
  lattice_spring_on : dH/dλ = +H_latt  → de = +all_es  / all_lambda
  deep_off          : dH/dλ = -H_deep  → de = -all_ed  / (1 - all_lambda)

Reversibility check:
  ΔF_fwd (from hti_mag) + ΔF_bwd (from this file) ≈ 0
"""

import glob
import json
import os
import shutil

import numpy as np
import scipy.constants as pc

from dpti.einstein import magnetic_frenkel
from dpti.lib.lammps import get_natoms, get_nspins
from dpti.lib.utils import (
    block_avg,
    create_path,
    get_first_matched_key_from_dict,
    get_task_file_abspath,
    integrate_range_hti,
    parse_seq,
    relative_link_file,
)
from dpti.hti_mag import _get_spring_lambda_mode


# ─────────────────────────────────────────────────────────────────────────────
# Force-field helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ff_spring_on(lamb, m_spring_k, var_spring, enabled=True):
    """Lattice spring coupling *growing* with lambda: k_eff = lambda * k."""
    ret = ""
    if not enabled:
        ret += "variable        l_spring equal 0.0\n"
        return ret
    ntypes = len(m_spring_k)
    for ii in range(ntypes):
        ret += f"group           type_{ii + 1} type {ii + 1}\n"
    for ii in range(ntypes):
        m_spring_const = m_spring_k[ii] * lamb if var_spring else m_spring_k[ii]
        ret += f"fix             l_spring_{ii + 1} type_{ii + 1} spring/self {m_spring_const:.10e}\n"
        ret += f"fix_modify      l_spring_{ii + 1} energy yes\n"
    sum_str = "f_l_spring_1"
    for ii in range(1, ntypes):
        sum_str += f"+f_l_spring_{ii + 1}"
    ret += f"variable        l_spring equal {sum_str}\n"
    return ret


def _ff_spring_spin_on(lamb, m_spring_spin_k, var_spring, enabled=True):
    """Spin spring coupling *growing* with lambda: k_eff = lambda * k."""
    ret = ""
    if not enabled:
        ret += "variable        l_spring_spin equal 0.0\n"
        return ret
    ntypes = len(m_spring_spin_k)
    for ii in range(ntypes):
        ret += f"group           type_{ii + 1} type {ii + 1}\n"
    for ii in range(ntypes):
        m_spring_const = m_spring_spin_k[ii] * lamb if var_spring else m_spring_spin_k[ii]
        ret += f"fix             l_spring_spin_{ii + 1} type_{ii + 1} spring/spin {m_spring_const:.10e}\n"
        ret += f"fix_modify      l_spring_spin_{ii + 1} energy yes\n"
    sum_str = "f_l_spring_spin_1"
    for ii in range(1, ntypes):
        sum_str += f"+f_l_spring_spin_{ii + 1}"
    ret += f"variable        l_spring_spin equal {sum_str}\n"
    return ret


def _ff_two_steps_rev(
    lamb,
    model,
    m_spring_k,
    m_spring_spin_k,
    step,
    append=None,
    if_meam=False,
    meam_model=None,
):
    """Force-field block for the three reverse steps."""
    ret = ""
    ret += "# --------------------- FORCE FIELDS ---------------------\n"

    # Deep potential pair style (always present, fully on except deep_off)
    if if_meam:
        raise NotImplementedError("meam not supported in hti_mag_rev")
    if append:
        ret += f"pair_style      deepspin {model:s} {append:s}\n"
    else:
        ret += f"pair_style      deepspin {model:s}\n"
    ret += "pair_coeff * *\n"

    if step == "spin_spring_on":
        # Springs: spin grows (λ·k_spin), lattice off
        ret += _ff_spring_on(lamb, m_spring_k, var_spring=False, enabled=False)
        ret += _ff_spring_spin_on(lamb, m_spring_spin_k, var_spring=True)
        # Deep: fully on, no adapt fix needed

    elif step == "lattice_spring_on":
        # Springs: lattice grows (λ·k_latt), spin fully on
        ret += _ff_spring_on(lamb, m_spring_k, var_spring=True)
        ret += _ff_spring_spin_on(lamb, m_spring_spin_k, var_spring=False)
        # Deep: fully on

    elif step == "deep_off":
        # Springs: both fully on (constant)
        ret += _ff_spring_on(lamb, m_spring_k, var_spring=False)
        ret += _ff_spring_spin_on(lamb, m_spring_spin_k, var_spring=False)
        # Deep: scales down as (1-λ)
        ret += "fix             l_deep all adapt 1 pair deepspin scale * * v_INV_LAMBDA\n"

    else:
        raise RuntimeError(f"unknown reverse step '{step}'")

    ret += "compute         e_deep all pe pair\n"
    ret += "compute         spin all property/atom sp spx spy spz fmx fmy fmz\n"
    return ret


# ─────────────────────────────────────────────────────────────────────────────
# LAMMPS input generator
# ─────────────────────────────────────────────────────────────────────────────

def _gen_lammps_input_rev(
    conf_file,
    mass_map,
    spin_mass,
    lamb,
    model,
    m_spring_k,
    m_spring_spin_k,
    nsteps,
    timestep,
    ens,
    temp,
    pres=1.0,
    tau_t=0.1,
    tau_p=0.5,
    thermo_freq=100,
    dump_freq=100,
    copies=None,
    crystal="frenkel",
    step="spin_spring_on",
    append=None,
    lattice_flag=1,
    spin_flag=1,
):
    ret = ""
    ret += "clear\n"
    ret += "# --------------------- VARIABLES-------------------------\n"
    ret += f"variable        NSTEPS          equal {nsteps}\n"
    ret += f"variable        THERMO_FREQ     equal {thermo_freq}\n"
    ret += f"variable        DUMP_FREQ       equal {dump_freq}\n"
    ret += f"variable        SP_MASS         equal {spin_mass:f}\n"
    ret += f"variable        TEMP            equal {temp:f}\n"
    ret += f"variable        PRES            equal {pres:f}\n"
    ret += f"variable        TAU_T           equal {tau_t:f}\n"
    ret += f"variable        TAU_P           equal {tau_p:f}\n"
    ret += f"variable        LAMBDA          equal {lamb:.10e}\n"
    ret += "variable        INV_LAMBDA      equal %.10e\n" % (1 - lamb)
    ret += "# ---------------------- INITIALIZAITION ------------------\n"
    ret += "units           metal\n"
    ret += "boundary        p p p\n"
    ret += "atom_style      spin\n"
    ret += "# --------------------- ATOM DEFINITION ------------------\n"
    ret += "box             tilt large\n"
    ret += f"read_data       {conf_file}\n"
    if copies is not None:
        ret += "replicate       %d %d %d\n" % (copies[0], copies[1], copies[2])
    ret += "change_box      all triclinic\n"
    for jj in range(len(mass_map)):
        ret += "mass            %d %f\n" % (jj + 1, mass_map[jj])

    ret += _ff_two_steps_rev(
        lamb,
        model,
        m_spring_k,
        m_spring_spin_k,
        step,
        append=append,
    )

    ret += "# --------------------- MD SETTINGS ----------------------\n"
    ret += "neighbor        1.0 bin\n"
    ret += f"timestep        {timestep}\n"
    ret += "thermo          ${THERMO_FREQ}\n"
    ret += "compute         allmsd all msd\n"
    ret += "thermo_style    custom step ke pe etotal enthalpy temp press vol v_l_spring c_e_deep v_l_spring_spin c_allmsd[*]\n"
    ret += "thermo_modify   format 9 %.16e\n"
    ret += "thermo_modify   format 10 %.16e\n"
    ret += "thermo_modify   format 11 %.16e\n"
    ret += f"dump            1 all custom ${{DUMP_FREQ}} dump.hti id type x y z vx vy vz c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]\n"

    if ens == "nvt":
        ret += "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T} mass ${SP_MASS} rand %d\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    elif ens == "nvt-langevin":
        ret += "fix             1 all nve/spin lattice_flag %d spin_flag %d\n" % (
            lattice_flag, spin_flag
        )
        if lattice_flag:
            ret += "fix             2 all langevin ${TEMP} ${TEMP} ${TAU_T} %d" % (
                np.random.default_rng().integers(1, 2**16)
            )
            if crystal == "frenkel":
                ret += " zero yes\n"
            else:
                ret += " zero no\n"
        if spin_flag:
            ret += "fix             3 all langevin/spin ${TEMP} ${TEMP} ${TAU_T} %d zero yes\n" % (
                np.random.default_rng().integers(1, 2**16)
            )
    else:
        raise RuntimeError(f"unknown ensemble '{ens}'")

    ret += "# --------------------- INITIALIZE -----------------------\n"
    if lattice_flag and spin_flag:
        ret += "velocity        all create ${TEMP} %d spin yes spmass ${SP_MASS}\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    elif lattice_flag:
        ret += "velocity        all create ${TEMP} %d\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    if crystal == "frenkel":
        ret += "fix             fc all recenter INIT INIT INIT\n"
        ret += "fix             fm all momentum 1 linear 1 1 1\n"
        ret += "velocity        all zero linear\n"
    elif crystal == "vega":
        ret += "group           first id 1\n"
        ret += "fix             fc first recenter INIT INIT INIT\n"
        ret += "fix             fm first momentum 1 linear 1 1 1\n"
        ret += "velocity        first zero linear\n"
    else:
        raise RuntimeError(f"unknown crystal '{crystal}'")

    ret += "# --------------------- RUN ------------------------------\n"
    ret += "run             ${NSTEPS}\n"
    ret += "write_data      out.lmp\n"
    return ret


# ─────────────────────────────────────────────────────────────────────────────
# Task creation
# ─────────────────────────────────────────────────────────────────────────────

# Lambda key names for each reverse step
_LAMBDA_KEYS_REV = {
    "spin_spring_on":    ["lambda_spin_spring_on"],
    "lattice_spring_on": ["lambda_lattice_spring_on", "lambda_spring_on"],
    "deep_off":          ["lambda_deep_off"],
}


def _make_tasks_rev(
    iter_name,
    jdata,
    ref,
    step,
    link=False,
):
    crystal   = jdata.get("crystal", "frenkel")
    protect_eps = jdata["protect_eps"]

    lambda_keys = _LAMBDA_KEYS_REV[step]
    all_lambda = parse_seq(
        get_first_matched_key_from_dict(jdata, lambda_keys)
    )
    if all_lambda[0] == 0:
        all_lambda[0] += protect_eps
    if all_lambda[-1] == 1:
        all_lambda[-1] -= protect_eps

    equi_conf = jdata["equi_conf"]
    equi_conf = os.path.abspath(equi_conf)
    model     = jdata["model"]
    model     = os.path.abspath(model)

    mass_map  = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    spin_mass = get_first_matched_key_from_dict(jdata, ["spin_mass_map", "sp_mass_map", "spin_mass"])
    spring_k      = jdata["spring_k"]
    spring_spin_k = get_first_matched_key_from_dict(
        jdata, ["spin_spring_k", "s_spring_k", "spring_k_spin", "spring_spin_k"]
    )
    m_spring_k      = [spring_k * m for m in mass_map]
    m_spring_spin_k = [spring_spin_k * m * spin_mass for m in mass_map]

    nsteps     = jdata["nsteps"]
    timestep   = get_first_matched_key_from_dict(jdata, ["timestep", "dt"])
    thermo_freq = get_first_matched_key_from_dict(jdata, ["thermo_freq", "stat_freq"])
    dump_freq   = get_first_matched_key_from_dict(
        jdata, ["dump_freq", "thermo_freq", "stat_freq"]
    )
    temp    = jdata["temp"]
    copies  = jdata.get("copies", None)
    langevin = jdata.get("langevin", True)
    append  = jdata.get("append", None)

    jdata["step"] = step
    create_path(iter_name)

    # Copy/link conf and model into iter_name
    copied_conf  = os.path.join(os.path.abspath(iter_name), "conf.lmp")
    linked_model = os.path.join(os.path.abspath(iter_name), "graph.pb")
    if not link:
        shutil.copyfile(equi_conf, copied_conf)
        shutil.copyfile(model,     linked_model)
    else:
        cwd = os.getcwd()
        os.chdir(iter_name)
        os.symlink(os.path.relpath(equi_conf), "conf.lmp")
        os.symlink(os.path.relpath(model),     "graph.pb")
        os.chdir(cwd)

    jdata_task = dict(jdata)
    jdata_task["equi_conf"] = "conf.lmp"
    jdata_task["model"]     = "graph.pb"
    cwd = os.getcwd()
    os.chdir(iter_name)
    with open("in.json", "w") as fp:
        json.dump(jdata_task, fp, indent=4)
    os.chdir(cwd)

    for idx, lamb in enumerate(all_lambda):
        work_path = os.path.join(iter_name, "task.%06d" % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.relpath(copied_conf),  "conf.lmp")
        os.symlink(os.path.relpath(linked_model), "graph.pb")

        ens = "nvt-langevin" if (langevin or idx == 0) else "nvt"

        lmp_str = _gen_lammps_input_rev(
            "conf.lmp",
            mass_map,
            spin_mass,
            lamb,
            "graph.pb",
            m_spring_k,
            m_spring_spin_k,
            nsteps,
            timestep,
            ens,
            temp,
            thermo_freq=thermo_freq,
            dump_freq=dump_freq,
            copies=copies,
            crystal=crystal,
            step=step,
            append=append,
        )
        with open("in.lammps", "w") as fp:
            fp.write(lmp_str)
        with open("lambda.out", "w") as fp:
            fp.write(str(lamb))
        os.chdir(cwd)


def make_tasks_rev(iter_name, jdata, ref="einstein"):
    """Create all three reverse-step task directories under iter_name."""
    equi_conf = os.path.abspath(jdata["equi_conf"])
    model     = os.path.abspath(jdata["model"])

    job_abs_dir  = create_path(iter_name)
    copied_conf  = os.path.join(os.path.abspath(iter_name), "conf.lmp")
    linked_model = os.path.join(os.path.abspath(iter_name), "graph.pb")
    shutil.copyfile(equi_conf, copied_conf)
    shutil.copyfile(model,     linked_model)
    jdata["equi_conf"] = "conf.lmp"
    jdata["model"]     = "graph.pb"

    cwd = os.getcwd()
    os.chdir(iter_name)
    with open("in.json", "w") as fp:
        json.dump(jdata, fp, indent=4)

    _make_tasks_rev("00.spin_spring_on",    jdata, ref, step="spin_spring_on",    link=True)
    _make_tasks_rev("01.lattice_spring_on", jdata, ref, step="lattice_spring_on", link=True)
    _make_tasks_rev("02.deep_off",          jdata, ref, step="deep_off",          link=True)
    os.chdir(cwd)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _post_tasks_rev(iter_name, jdata, natoms=None, scheme="s", step="spin_spring_on"):
    stat_skip  = jdata["stat_skip"]
    stat_bsize = jdata["stat_bsize"]
    all_tasks  = sorted(glob.glob(os.path.join(iter_name, "task.[0-9]*")))

    equi_conf = get_task_file_abspath(iter_name, jdata["equi_conf"])
    if natoms is None:
        natoms = get_natoms(equi_conf)
        nspins = get_nspins(equi_conf, natoms)
        if "copies" in jdata:
            natoms *= np.prod(jdata["copies"])
            nspins *= np.prod(jdata["copies"])

    all_lambda, all_es, all_esp, all_ed = [], [], [], []
    all_es_err, all_esp_err, all_ed_err = [], [], []
    all_etot, all_enthalpy, all_msd_xyz = [], [], []

    for ii in all_tasks:
        log_file = os.path.join(ii, "log.lammps")
        data = get_thermo(log_file)
        np.savetxt(os.path.join(ii, "data"), data, fmt="%.6e")
        sa,  se  = block_avg(data[:, 8],  skip=stat_skip, block_size=stat_bsize)
        da,  de  = block_avg(data[:, 9],  skip=stat_skip, block_size=stat_bsize)
        spa, spe = block_avg(data[:, 10], skip=stat_skip, block_size=stat_bsize)
        etot, _  = block_avg(data[:, 3],  skip=stat_skip, block_size=stat_bsize)
        enthalpy, _ = block_avg(data[:, 4], skip=stat_skip, block_size=stat_bsize)
        msd_xyz  = data[-1, -1]

        sa  /= natoms
        se  /= natoms
        spa /= natoms
        spe /= natoms
        da  /= natoms
        de  /= natoms

        lmda_name = os.path.join(ii, "lambda.out")
        ll = float(open(lmda_name).read())

        all_lambda.append(ll)
        all_es.append(sa);      all_es_err.append(se)
        all_esp.append(spa);    all_esp_err.append(spe)
        all_ed.append(da);      all_ed_err.append(de)
        all_etot.append(etot / natoms)
        all_enthalpy.append(enthalpy)
        all_msd_xyz.append(msd_xyz)

    all_lambda  = np.array(all_lambda)
    all_es      = np.array(all_es)
    all_esp     = np.array(all_esp)
    all_ed      = np.array(all_ed)
    all_es_err  = np.array(all_es_err)
    all_esp_err = np.array(all_esp_err)
    all_ed_err  = np.array(all_ed_err)

    # Integrand and error for each reverse step
    if step == "spin_spring_on":
        # dH/dλ = +H_spin_spring;  reported energy = λ·H_spin → divide by λ
        de_int   = all_esp / all_lambda
        all_err  = all_esp_err / all_lambda
    elif step == "lattice_spring_on":
        # dH/dλ = +H_latt_spring;  reported energy = λ·H_latt → divide by λ
        de_int   = all_es / all_lambda
        all_err  = all_es_err / all_lambda
    elif step == "deep_off":
        # dH/dλ = -H_deep;  reported energy = (1-λ)·H_deep → divide by (1-λ), negate
        de_int   = -all_ed / (1 - all_lambda)
        all_err  = all_ed_err / (1 - all_lambda)
    else:
        raise RuntimeError(f"unknown reverse step '{step}'")

    # Save hti.out
    all_print = np.array([
        all_lambda, de_int, all_err,
        all_ed, all_es, all_esp,
        all_ed_err, all_es_err, all_esp_err,
        all_etot, all_es, all_enthalpy, all_msd_xyz,
    ])
    np.savetxt(
        os.path.join(iter_name, "hti.out"),
        all_print.T,
        fmt="%.8e",
        header="lmbda dU dU_err Ud Us Usp Ud_err Us_err Usp_err etot spring_eng enthalpy msd_xyz",
    )

    diff_e, err, sys_err = integrate_range_hti(all_lambda, de_int, all_err, scheme=scheme)
    thermo_info = {
        "de":    diff_e,
        "err":   err,
        "sys_err": sys_err,
    }
    return diff_e, [err, sys_err], thermo_info


def post_tasks_rev(iter_name, jdata, natoms=None, scheme="s"):
    """Sum ΔF contributions from all three reverse steps."""
    steps = [
        ("spin_spring_on",    "00.spin_spring_on"),
        ("lattice_spring_on", "01.lattice_spring_on"),
        ("deep_off",          "02.deep_off"),
    ]
    total_de  = 0.0
    stt_err2  = 0.0
    sys_err   = 0.0
    thermo_info = {}

    for step, subdir in steps:
        subtask_name = os.path.join(iter_name, subdir)
        ei, erri, tinfo = _post_tasks_rev(
            subtask_name, jdata, natoms=natoms, scheme=scheme, step=step
        )
        print(f"# ΔF ({step:20s}): {ei:+20.12f}  stat {erri[0]:.3e}  sys {erri[1]:.3e}")
        total_de += ei
        stt_err2 += erri[0] ** 2
        sys_err  += erri[1]
        thermo_info[step] = tinfo

    err = [np.sqrt(stt_err2), sys_err]
    print(f"# ΔF_bwd (total)          : {total_de:+20.12f}  stat {err[0]:.3e}  sys {err[1]:.3e}")
    return total_de, err, thermo_info


# ─────────────────────────────────────────────────────────────────────────────
# Reversibility check
# ─────────────────────────────────────────────────────────────────────────────

def check_reversibility(fwd_job, bwd_job, jdata, natoms=None, scheme="s"):
    """
    Compare forward (hti_mag) and backward (hti_mag_rev) free energy differences.

    Imports post_tasks from hti_mag for the forward direction.

    Returns
    -------
    delta_fwd, delta_bwd, hysteresis
        hysteresis = |ΔF_fwd + ΔF_bwd|  (should be ~0 for reversible path)
    """
    from dpti.hti_mag import post_tasks as post_tasks_fwd

    print("=" * 60)
    print("Forward  (Reference → Target)")
    print("=" * 60)
    delta_fwd, err_fwd, _ = post_tasks_fwd(fwd_job, jdata, natoms=natoms, scheme=scheme)

    print()
    print("=" * 60)
    print("Backward (Target → Reference)")
    print("=" * 60)
    delta_bwd, err_bwd, _ = post_tasks_rev(bwd_job, jdata, natoms=natoms, scheme=scheme)

    hysteresis = abs(delta_fwd + delta_bwd)
    rel = hysteresis / abs(delta_fwd) * 100 if delta_fwd != 0 else float("inf")

    print()
    print("=" * 60)
    print("Reversibility Summary")
    print("=" * 60)
    print(f"  ΔF_fwd          = {delta_fwd:+.8f} eV/atom")
    print(f"  ΔF_bwd          = {delta_bwd:+.8f} eV/atom")
    print(f"  |ΔF_fwd+ΔF_bwd| = {hysteresis:.2e} eV/atom  ({rel:.2f}%)")
    if rel < 1.0:
        print("  Status: ✓ Reversible (η < 1%)")
    elif rel < 5.0:
        print("  Status: ~ Marginally reversible (1% < η < 5%)")
    else:
        print("  Status: ✗ Not reversible (η > 5%) — increase nsteps or λ density")

    return delta_fwd, delta_bwd, hysteresis
