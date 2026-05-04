#!/usr/bin/env python3

import glob
import json
import os
import shutil

import numpy as np
import scipy.constants as pc

import dpti.lib.lmp as lmp
from dpti import einstein, hti
from dpti.hti_liq import parse_lj_sigma_epsilon
from dpti.hti_mag import _get_spring_lambda_mode
from dpti.lib.lammps import get_thermo
from dpti.lib.utils import (
    block_avg,
    create_path,
    get_first_matched_key_from_dict,
    integrate_range_hti,
    parse_seq,
)


def make_iter_name(iter_index):
    return "task_hti." + ("%04d" % iter_index)


# ---------------------------------------------------------------------------
# Force-field fragment generators
# ---------------------------------------------------------------------------

def _lambda_factor(lamb, mode):
    if mode == "on":
        return lamb
    if mode == "off":
        return 1.0 - lamb
    return 1.0


def _ff_spin_mod(
    lamb,
    m_spring_spin_k,
    spin_ref_s0,
    spin_map,
    mode,
):
    """Return LAMMPS lines for the spin-modulus confining potential.

    spring/spin/mod k S0 is the only supported spin reference.  k is scaled by
    the lambda schedule; S0 remains fixed.

    mode: 'on'   -> factor = lambda        (soft_on / spin_mod_on)
          'off'  -> factor = 1 - lambda    (soft_off / spin_mod_off)
          'full' -> factor = 1             (deep_on)
    """
    ntypes = len(m_spring_spin_k)
    spin_types = [i for i in range(ntypes) if spin_map[i]]
    if not spin_types:
        return "variable        l_spring_spin equal 0.0\n"

    factor = _lambda_factor(lamb, mode)
    if factor <= 0.0:
        return "variable        l_spring_spin equal 0.0\n"

    ret = ""
    for ii in spin_types:
        ret += f"group           type_{ii + 1} type {ii + 1}\n"

    for ii in spin_types:
        k = m_spring_spin_k[ii] * factor
        s0 = spin_ref_s0[ii]
        ret += f"fix             l_spring_spin_{ii + 1} type_{ii + 1} spring/spin/mod {k:.10e} {s0:.10e}\n"
        ret += f"fix_modify      l_spring_spin_{ii + 1} energy yes\n"

    sum_str = "+".join(f"f_l_spring_spin_{ii + 1}" for ii in spin_types)
    ret += f"variable        l_spring_spin equal {sum_str}\n"
    return ret


def _ff_soft_on_mag(lamb, sparam, m_spring_spin_k, spin_ref_s0, spin_map, spring_lambda_mode):
    """Force field for soft_on step.

    spring_lambda_mode decides what is scaled with lambda here:
      joint/split  : LJ soft scales 0->1; spin_ref is full strength
      lattice_only : LJ soft only
      spin_only    : spin_mod only (no LJ soft)
    """
    has_lattice = spring_lambda_mode in ("joint", "split", "lattice_only")
    has_spin_on = spring_lambda_mode == "spin_only"
    has_spin_full = spring_lambda_mode in ("joint", "split")

    ret = "# --------------------- FORCE FIELDS ---------------------\n"
    if has_lattice:
        nn = sparam["n"]
        alpha_lj = sparam["alpha_lj"]
        rcut = sparam["rcut"]
        ret += f"pair_style      lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        ret = parse_lj_sigma_epsilon(ret, sparam, hybrid=False)
        ret += "fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes\n"
        ret += "compute         lj_pe all pair lj/cut/soft\n"

    if has_spin_on:
        ret += _ff_spin_mod(lamb, m_spring_spin_k, spin_ref_s0, spin_map, mode="on")
    elif has_spin_full:
        ret += _ff_spin_mod(lamb, m_spring_spin_k, spin_ref_s0, spin_map, mode="full")
    else:
        ret += "variable        l_spring_spin equal 0.0\n"

    if has_lattice:
        ret += "variable        e_diff equal c_lj_pe/v_LAMBDA\n"
    else:
        ret += "variable        e_diff equal v_l_spring_spin/v_LAMBDA\n"

    ret += "compute         spin all property/atom sp spx spy spz fmx fmy fmz\n"
    return ret


def _ff_deep_on_mag(lamb, sparam, model, m_spring_spin_k, spin_ref_s0, spin_map, spring_lambda_mode):
    """Force field for deep_on step.

    deepspin scales 0->1 via adapt/fep; LJ soft (if present) is full;
    spin_mod is full.  e_diff = deepspin contribution = c_e_diff_fep[1].
    """
    has_lattice = spring_lambda_mode in ("joint", "split", "lattice_only")
    has_spin = spring_lambda_mode != "lattice_only"

    ret = "# --------------------- FORCE FIELDS ---------------------\n"
    ret += "variable        ONE equal 1\n"
    if has_lattice:
        nn = sparam["n"]
        alpha_lj = sparam["alpha_lj"]
        rcut = sparam["rcut"]
        ret += f"pair_style      hybrid/overlay deepspin {model} lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        ret += "pair_coeff      * * deepspin\n"
        ret = parse_lj_sigma_epsilon(ret, sparam, hybrid=True)
    else:
        ret += f"pair_style      deepspin {model}\n"
        ret += "pair_coeff      * *\n"

    if has_spin:
        ret += _ff_spin_mod(lamb, m_spring_spin_k, spin_ref_s0, spin_map, mode="full")
    else:
        ret += "variable        l_spring_spin equal 0.0\n"

    ret += "fix             l_deep all adapt/fep 0 pair deepspin scale * * v_LAMBDA\n"
    ret += "compute         e_diff_fep all fep ${TEMP} pair deepspin scale * * v_ONE\n"
    ret += "variable        e_diff equal c_e_diff_fep[1]\n"
    ret += "compute         spin all property/atom sp spx spy spz fmx fmy fmz\n"
    return ret


def _ff_soft_off_mag(lamb, sparam, model, m_spring_spin_k, spin_ref_s0, spin_map, spring_lambda_mode):
    """Force field for soft_off step.

    spring_lambda_mode decides what is scaled with (1-lambda) here:
      joint        : LJ soft + spin_mod both scale 1->0
      split        : LJ soft scales 1->0; spin_ref remains full strength
      lattice_only : LJ soft only
      spin_only    : spin_mod only (no LJ soft)
    """
    has_lattice = spring_lambda_mode in ("joint", "split", "lattice_only")
    has_scaling_spin = spring_lambda_mode in ("joint", "spin_only")
    has_full_spin = spring_lambda_mode == "split"

    ret = "# --------------------- FORCE FIELDS ---------------------\n"
    ret += "variable        INV_LAMBDA equal 1-${LAMBDA}\n"
    if has_lattice:
        nn = sparam["n"]
        alpha_lj = sparam["alpha_lj"]
        rcut = sparam["rcut"]
        ret += f"pair_style      hybrid/overlay deepspin {model} lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        ret += "pair_coeff      * * deepspin\n"
        ret = parse_lj_sigma_epsilon(ret, sparam, hybrid=True)
        ret += "fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes\n"
        ret += "compute         lj_pe all pair lj/cut/soft\n"
    else:
        ret += f"pair_style      deepspin {model}\n"
        ret += "pair_coeff      * *\n"

    if has_scaling_spin:
        ret += _ff_spin_mod(lamb, m_spring_spin_k, spin_ref_s0, spin_map, mode="off")
    elif has_full_spin:
        ret += _ff_spin_mod(lamb, m_spring_spin_k, spin_ref_s0, spin_map, mode="full")
    else:
        ret += "variable        l_spring_spin equal 0.0\n"

    if has_lattice and has_scaling_spin:
        ret += "variable        e_diff equal -(c_lj_pe+v_l_spring_spin)/v_INV_LAMBDA\n"
    elif has_lattice:
        ret += "variable        e_diff equal -c_lj_pe/v_INV_LAMBDA\n"
    else:
        ret += "variable        e_diff equal -v_l_spring_spin/v_INV_LAMBDA\n"

    ret += "compute         spin all property/atom sp spx spy spz fmx fmy fmz\n"
    return ret


def _ff_spin_mod_on_mag(lamb, m_spring_spin_k, spin_ref_s0, spin_map):
    """Force field for spin_mod_on step (spin_only mode).

    No lattice interaction (U_lattice=0 throughout spin_only path);
    spin_mod scales 0->1.
    """
    ret = "# --------------------- FORCE FIELDS ---------------------\n"
    ret += _ff_spin_mod(lamb, m_spring_spin_k, spin_ref_s0, spin_map, mode="on")
    ret += "variable        e_diff equal v_l_spring_spin/v_LAMBDA\n"
    ret += "compute         spin all property/atom sp spx spy spz fmx fmy fmz\n"
    return ret


def _ff_spin_mod_off_mag(lamb, model, m_spring_spin_k, spin_ref_s0, spin_map):
    """Force field for spin_mod_off step (split mode only).

    deepspin is fully on (no LJ soft); spin_mod scales 1->0.
    """
    ret = "# --------------------- FORCE FIELDS ---------------------\n"
    ret += "variable        INV_LAMBDA equal 1-${LAMBDA}\n"
    ret += f"pair_style      deepspin {model}\n"
    ret += "pair_coeff      * *\n"
    ret += _ff_spin_mod(lamb, m_spring_spin_k, spin_ref_s0, spin_map, mode="off")
    ret += "variable        e_diff equal -v_l_spring_spin/v_INV_LAMBDA\n"
    ret += "compute         spin all property/atom sp spx spy spz fmx fmy fmz\n"
    return ret


# ---------------------------------------------------------------------------
# Full LAMMPS input generator
# ---------------------------------------------------------------------------

def _gen_lammps_input_mag_liq(
    step,
    conf_file,
    mass_map,
    spin_mass,
    lamb,
    soft_param,
    model,
    m_spring_spin_k,
    spin_ref_s0,
    spin_map,
    spring_lambda_mode,
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
    lattice_flag=1,
    spin_flag=1,
):
    ret = ""
    ret += "clear\n"
    ret += "# --------------------- VARIABLES-------------------------\n"
    ret += "variable        NSTEPS          equal %d\n" % nsteps
    ret += "variable        THERMO_FREQ     equal %d\n" % thermo_freq
    ret += "variable        DUMP_FREQ       equal %d\n" % dump_freq
    ret += f"variable        SP_MASS         equal {spin_mass:f}\n"
    ret += f"variable        TEMP            equal {temp:f}\n"
    ret += f"variable        PRES            equal {pres:f}\n"
    ret += f"variable        TAU_T           equal {tau_t:f}\n"
    ret += f"variable        TAU_P           equal {tau_p:f}\n"
    ret += f"variable        LAMBDA          equal {lamb:.10e}\n"
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

    if step == "soft_on":
        ret += _ff_soft_on_mag(lamb, soft_param, m_spring_spin_k, spin_ref_s0, spin_map, spring_lambda_mode)
    elif step == "deep_on":
        ret += _ff_deep_on_mag(lamb, soft_param, model, m_spring_spin_k, spin_ref_s0, spin_map, spring_lambda_mode)
    elif step == "soft_off":
        ret += _ff_soft_off_mag(lamb, soft_param, model, m_spring_spin_k, spin_ref_s0, spin_map, spring_lambda_mode)
    elif step == "spin_mod_on":
        ret += _ff_spin_mod_on_mag(lamb, m_spring_spin_k, spin_ref_s0, spin_map)
    elif step in ("spin_mod_off", "spin_ref_off"):
        ret += _ff_spin_mod_off_mag(lamb, model, m_spring_spin_k, spin_ref_s0, spin_map)
    else:
        raise RuntimeError(f"unknown step: {step}")

    ret += "# --------------------- MD SETTINGS ----------------------\n"
    ret += "neighbor        1.0 bin\n"
    ret += f"timestep        {timestep}\n"
    ret += "compute         allmsd all msd\n"
    ret += "compute         spinmsd all msd/spin\n"
    ret += "compute         spinmodmsd all msd/spin/mod\n"
    ret += "thermo          ${THERMO_FREQ}\n"
    # col 8: v_e_diff (integrand), col 9: v_l_spring_spin (monitor)
    # col 10-13: c_allmsd[1-4], col 14-17: c_spinmsd[1-4], col 18: c_spinmodmsd
    ret += "thermo_style    custom step ke pe etotal enthalpy temp press vol v_e_diff v_l_spring_spin c_allmsd[*] c_spinmsd[*] c_spinmodmsd\n"
    ret += "thermo_modify   format 9 %.16e\n"
    ret += "thermo_modify   format 10 %.16e\n"
    ret += "thermo_modify   format 19 %.16e\n"
    ret += "dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz c_spin[1] c_spin[2] c_spin[3] c_spin[4]\n"

    if ens == "nvt":
        ret += "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T} mass ${SP_MASS} rand %d\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    elif ens == "nvt-langevin":
        ret += "fix             1 all nve/spin lattice_flag %d spin_flag %d\n" % (
            lattice_flag, spin_flag
        )
        if lattice_flag:
            ret += "fix             2 all langevin ${TEMP} ${TEMP} ${TAU_T} %d zero no\n" % (
                np.random.default_rng().integers(1, 2**16)
            )
        if spin_flag:
            ret += "fix             3 all langevin/spin ${TEMP} ${TEMP} ${TAU_T} %d\n" % (
                np.random.default_rng().integers(1, 2**16)
            )
    elif ens in ("npt-iso", "npt"):
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P} mass ${SP_MASS} rand %d\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    elif ens == "nve":
        ret += "fix             1 all nve\n"
    else:
        raise RuntimeError(f"unknown ensemble: {ens}")

    ret += "fix             mzero all momentum 10 linear 1 1 1\n"
    ret += "# --------------------- INITIALIZE -----------------------\n"
    ret += "velocity        all create ${TEMP} %d spin yes spmass ${SP_MASS}\n" % (
        np.random.default_rng().integers(1, 2**16)
    )
    ret += "velocity        all zero linear\n"
    ret += "# --------------------- RUN ------------------------------\n"
    ret += "run             ${NSTEPS}\n"
    ret += "write_data      out.lmp\n"
    return ret


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def _get_subtask_dirs(spring_lambda_mode):
    """Return ordered list of (dir_prefix, step_name) for a given mode."""
    if spring_lambda_mode in ("joint", "lattice_only"):
        return [
            ("00.soft_on", "soft_on"),
            ("01.deep_on", "deep_on"),
            ("02.soft_off", "soft_off"),
        ]
    elif spring_lambda_mode == "spin_only":
        return [
            ("00.spin_mod_on", "spin_mod_on"),
            ("01.deep_on", "deep_on"),
            ("02.spin_mod_off", "spin_mod_off"),
        ]
    elif spring_lambda_mode == "split":
        return [
            ("00.soft_on", "soft_on"),
            ("01.deep_on", "deep_on"),
            ("02.soft_off", "soft_off"),
            ("03.spin_ref_off", "spin_ref_off"),
        ]
    else:
        raise RuntimeError(f"unknown spring_lambda_mode: {spring_lambda_mode}")


_STEP_TO_LAMBDA_KEY = {
    "soft_on": "lambda_soft_on",
    "deep_on": "lambda_deep_on",
    "soft_off": "lambda_soft_off",
    "spin_mod_on": "lambda_spin_mod_on",
    "spin_mod_off": "lambda_spin_mod_off",
    "spin_ref_off": "lambda_spin_ref_off",
}


def _expand_type_param(value, ntypes, name):
    if isinstance(value, (list, tuple)):
        if len(value) != ntypes:
            raise RuntimeError(f"{name} length {len(value)} does not match ntypes {ntypes}")
        return [float(v) for v in value]
    return [float(value) for _ in range(ntypes)]


def _get_spin_ref_params(jdata, mass_map, spin_mass, spin_map):
    """Return spring reference parameters as (k_by_type, s0_by_type)."""
    ntypes = len(mass_map)
    spin_ref = jdata.get("spin_ref")
    if spin_ref is not None:
        style = spin_ref.get("style", "spring")
        if str(style).lower() not in ("spring", "harmonic"):
            raise RuntimeError("spin_ref.style must be 'spring'")
        k_by_type = _expand_type_param(spin_ref.get("k", 0.0), ntypes, "spin_ref.k")
        s0_by_type = _expand_type_param(spin_ref.get("s0", 0.0), ntypes, "spin_ref.s0")
    else:
        style = jdata.get("spin_mod_style", "spring")
        if str(style).lower() not in ("spring", "harmonic"):
            raise RuntimeError("spin_mod_style must be 'spring'")
        spin_mod_k = jdata.get("spin_mod_k", 0.0)
        base_k = _expand_type_param(spin_mod_k, ntypes, "spin_mod_k")
        # Legacy spin_mod_k followed hti_mag and was scaled by atom mass and spin mass.
        k_by_type = [base_k[i] * mass_map[i] * spin_mass for i in range(ntypes)]
        s0_by_type = _expand_type_param(jdata.get("spin_mod_s0", 0.0), ntypes, "spin_mod_s0")

    k_by_type = [k_by_type[i] if spin_map[i] else 0.0 for i in range(ntypes)]
    return k_by_type, s0_by_type


def _make_tasks(iter_name, jdata, step):
    lambda_key = _STEP_TO_LAMBDA_KEY.get(step)
    if lambda_key is None:
        raise RuntimeError(f"unknown step: {step}")
    if lambda_key not in jdata and step == "spin_ref_off":
        lambda_key = "lambda_spin_mod_off"
    all_lambda = parse_seq(jdata[lambda_key], protect_eps=jdata.get("protect_eps"))

    equi_conf = jdata["equi_conf"]
    mass_map = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    spin_mass = get_first_matched_key_from_dict(jdata, ["spin_mass", "sp_mass"])
    spin_map = get_first_matched_key_from_dict(jdata, ["spin_map", "sp_map"])
    ntypes = len(mass_map)
    m_spring_spin_k, spin_ref_s0 = _get_spin_ref_params(
        jdata, mass_map, spin_mass, spin_map
    )

    model = jdata.get("model", None)
    model_name = os.path.basename(model) if model is not None else None
    soft_param = jdata.get("soft_param", {})
    nsteps = jdata["nsteps"]
    timestep = get_first_matched_key_from_dict(jdata, ["timestep", "dt"])
    thermo_freq = get_first_matched_key_from_dict(jdata, ["thermo_freq", "stat_freq"])
    dump_freq = get_first_matched_key_from_dict(
        jdata, ["dump_freq", "thermo_freq", "stat_freq"]
    )
    copies = jdata.get("copies", None)
    temp = jdata["temp"]
    spring_lambda_mode = _get_spring_lambda_mode(jdata)
    ens = jdata.get("ens", "nvt")

    if soft_param:
        soft_param = dict(soft_param)
        soft_param["element_num"] = ntypes
        if "sigma_oo" in soft_param:
            soft_param["sigma_0_0"] = soft_param["sigma_oo"]
            soft_param["sigma_0_1"] = soft_param["sigma_oh"]
            soft_param["sigma_1_1"] = soft_param["sigma_hh"]

    create_path(iter_name)
    cwd = os.getcwd()
    os.chdir(iter_name)
    os.symlink(os.path.join("..", "in.json"), "in.json")
    os.symlink(os.path.join("..", "conf.lmp"), "conf.lmp")
    os.symlink(os.path.join("..", model_name), model_name)
    os.chdir(cwd)

    for idx, ii in enumerate(all_lambda):
        work_path = os.path.join(iter_name, "task.%06d" % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.join("..", "conf.lmp"), "conf.lmp")
        os.symlink(os.path.join("..", model_name), model_name)
        lmp_str = _gen_lammps_input_mag_liq(
            step=step,
            conf_file="conf.lmp",
            mass_map=mass_map,
            spin_mass=spin_mass,
            lamb=ii,
            soft_param=soft_param,
            model=model_name,
            m_spring_spin_k=m_spring_spin_k,
            spin_ref_s0=spin_ref_s0,
            spin_map=spin_map,
            spring_lambda_mode=spring_lambda_mode,
            nsteps=nsteps,
            timestep=timestep,
            ens=ens,
            temp=temp,
            thermo_freq=thermo_freq,
            dump_freq=dump_freq,
            copies=copies,
        )
        with open("in.lammps", "w") as fp:
            fp.write(lmp_str)
        with open("lambda.out", "w") as fp:
            fp.write(str(ii))
        os.chdir(cwd)


def make_tasks(iter_name, jdata):
    equi_conf = os.path.abspath(jdata["equi_conf"])
    model = os.path.abspath(jdata["model"])
    spring_lambda_mode = _get_spring_lambda_mode(jdata)

    create_path(iter_name)
    copied_conf = os.path.join(os.path.abspath(iter_name), "conf.lmp")
    shutil.copyfile(equi_conf, copied_conf)
    jdata = dict(jdata)
    jdata["equi_conf"] = copied_conf
    model_name = os.path.basename(model)
    copied_model = os.path.join(os.path.abspath(iter_name), model_name)
    shutil.copyfile(model, copied_model)
    jdata["model"] = copied_model

    cwd = os.getcwd()
    os.chdir(iter_name)
    with open("in.json", "w") as fp:
        json.dump(jdata, fp, indent=4)
    os.chdir(cwd)

    for dir_prefix, step in _get_subtask_dirs(spring_lambda_mode):
        _make_tasks(os.path.join(iter_name, dir_prefix), jdata, step)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _compute_thermo(fname, natoms, stat_skip, stat_bsize):
    data = get_thermo(fname)
    ea, ee = block_avg(data[:, 3], skip=stat_skip, block_size=stat_bsize)
    ha, he = block_avg(data[:, 4], skip=stat_skip, block_size=stat_bsize)
    ta, te = block_avg(data[:, 5], skip=stat_skip, block_size=stat_bsize)
    pa, pe = block_avg(data[:, 6], skip=stat_skip, block_size=stat_bsize)
    va, ve = block_avg(data[:, 7], skip=stat_skip, block_size=stat_bsize)
    thermo_info = {
        "p": pa, "p_err": pe,
        "v": va / natoms, "v_err": ve / natoms,
        "e": ea / natoms, "e_err": ee / natoms,
        "h": ha / natoms, "h_err": he / natoms,
        "t": ta, "t_err": te,
    }
    unit_cvt = 1e5 * (1e-10 ** 3) / pc.electron_volt
    thermo_info["pv"] = pa * va * unit_cvt / natoms
    thermo_info["pv_err"] = pe * va * unit_cvt / natoms
    return thermo_info


def _post_tasks(iter_name, step, natoms):
    """Integrate one sub-step.  col 8 of thermo output is always v_e_diff."""
    jdata = json.load(open(os.path.join(iter_name, "in.json")))
    stat_skip = jdata["stat_skip"]
    stat_bsize = jdata["stat_bsize"]
    all_tasks = sorted(glob.glob(os.path.join(iter_name, "task.[0-9]*")))

    all_lambda, all_dp_a, all_dp_e, all_msd_xyz, all_msd_spin_mod = [], [], [], [], []

    for ii in all_tasks:
        log_name = os.path.join(ii, "log.lammps")
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, "data"), data, fmt="%.6e")
        dp_a, dp_e = block_avg(data[:, 8], skip=stat_skip, block_size=stat_bsize)
        # c_allmsd[4] is at col 13 (0-indexed):
        # step ke pe etotal enthalpy T P V e_diff spin_spr allmsd[1-4] spinmsd[1-4] spinmodmsd
        msd_xyz = data[-1, 13]
        msd_spin_mod = data[-1, 18]
        dp_a /= natoms
        dp_e /= natoms
        ll = float(open(os.path.join(ii, "lambda.out")).read())
        all_lambda.append(ll)
        all_dp_a.append(dp_a)
        all_dp_e.append(dp_e)
        all_msd_xyz.append(msd_xyz)
        all_msd_spin_mod.append(msd_spin_mod)

    all_lambda = np.array(all_lambda)
    all_dp_a = np.array(all_dp_a)
    all_dp_e = np.array(all_dp_e)
    all_msd_xyz = np.array(all_msd_xyz)
    all_msd_spin_mod = np.array(all_msd_spin_mod)

    np.savetxt(
        os.path.join(iter_name, "hti.out"),
        np.array([all_lambda, all_dp_a, all_dp_e, all_msd_xyz, all_msd_spin_mod]).T,
        fmt="%.8e",
        header="lmbda dU dU_err msd_xyz msd_spin_mod",
    )

    diff_e, err, sys_err = integrate_range_hti(all_lambda, all_dp_a, all_dp_e)
    thermo_info = _compute_thermo(
        os.path.join(all_tasks[-1], "log.lammps"), natoms, stat_skip, stat_bsize
    )
    return diff_e, [err, sys_err], thermo_info


def post_tasks(iter_name, natoms):
    jdata = json.load(open(os.path.join(iter_name, "in.json")))
    spring_lambda_mode = _get_spring_lambda_mode(jdata)
    subtasks = _get_subtask_dirs(spring_lambda_mode)

    with open(os.path.join(iter_name, "conf.lmp")) as fp:
        sys_data = lmp.to_system_data(fp.read().split("\n"))
    atom_numbs = list(sys_data["atom_numbs"])
    if "copies" in jdata:
        scale = int(np.prod(jdata["copies"]))
        atom_numbs = [n * scale for n in atom_numbs]

    mass_map = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    spin_mass = get_first_matched_key_from_dict(jdata, ["spin_mass", "sp_mass"])
    spin_map = get_first_matched_key_from_dict(jdata, ["spin_map", "sp_map"])
    spin_ref_style, spin_ref_k, spin_ref_s0 = _get_spin_ref_params(
        jdata, mass_map, spin_mass, spin_map
    )
    spin_ref = {"style": spin_ref_style, "k": spin_ref_k, "s0": spin_ref_s0}

    fe_lattice = einstein.ideal_gas_fe(iter_name)
    fe_spin_ref = einstein.spin_ref_fe_per_atom(
        jdata["temp"], atom_numbs, spin_map, spin_ref
    )
    total_de = 0.0
    stt_err2 = 0.0
    sys_err = 0.0
    tinfo = None
    errs = []
    stage_des = []

    for dir_prefix, step in subtasks:
        subtask_path = os.path.join(iter_name, dir_prefix)
        de, err, tinfo = _post_tasks(subtask_path, step, natoms)
        total_de += de
        stt_err2 += err[0] ** 2
        sys_err += err[1]
        errs.append(err)
        stage_des.append(de)

    step_names = [s for _, s in subtasks]
    print(f"# F_ideal_lattice [eV/atom]: {fe_lattice:20.12f}")
    print(f"# F_spin_ref      [eV/atom]: {fe_spin_ref:20.12f}")
    for step, de, err in zip(step_names, stage_des, errs):
        print(f"# dF {step:12s} [eV/atom]: {de:20.12f}  {err[0]:10.3e} {err[1]:10.3e}")
    print(f"# F_total         [eV/atom]: {fe_lattice + fe_spin_ref + total_de:20.12f}")
    print(f"# HTI mag-liq step errors [stt_err, sys_err]: " +
          " | ".join(f"{s}={e}" for s, e in zip(step_names, errs)))

    combined_err = [np.sqrt(stt_err2), sys_err]
    return fe_lattice + fe_spin_ref + total_de, combined_err, tinfo


def _print_thermo_info(info):
    ptr = "# thermodynamics (normalized by natoms)\n"
    ptr += "# E (err)  [eV]:  {:20.8f} {:20.8f}\n".format(info["e"], info["e_err"])
    ptr += "# H (err)  [eV]:  {:20.8f} {:20.8f}\n".format(info["h"], info["h_err"])
    ptr += "# T (err)   [K]:  {:20.8f} {:20.8f}\n".format(info["t"], info["t_err"])
    ptr += "# P (err) [bar]:  {:20.8f} {:20.8f}\n".format(info["p"], info["p_err"])
    ptr += "# V (err) [A^3]:  {:20.8f} {:20.8f}\n".format(info["v"], info["v_err"])
    ptr += "# PV(err)  [eV]:  {:20.8f} {:20.8f}".format(info["pv"], info["pv_err"])
    print(ptr)


def compute_task(
    job,
    free_energy_type="helmholtz",
    manual_pv=None,
    manual_pv_err=None,
    npt=None,
):
    jdata = json.load(open(os.path.join(job, "in.json")))
    with open(os.path.join(job, "conf.lmp")) as fp:
        sys_data = lmp.to_system_data(fp.read().split("\n"))
    natoms = sum(sys_data["atom_numbs"])
    if "copies" in jdata:
        natoms *= int(np.prod(jdata["copies"]))

    fe, fe_err, thermo_info = post_tasks(job, natoms)
    _print_thermo_info(thermo_info)

    print("# numb atoms: %d" % natoms)
    print_format = "%20.12f  %10.3e  %10.3e"
    pv = pv_err = None

    if free_energy_type == "helmholtz":
        e1 = fe
        e1_err = fe_err[0]
        print("# Helmholtz free ener per atom (err) [eV]:")
        print(print_format % (fe, fe_err[0], fe_err[1]))
    elif free_energy_type == "gibbs":
        if npt is not None:
            npt_in = json.load(open(os.path.join(npt, "jdata.json")))
            npt_info = json.load(open(os.path.join(npt, "result.json")))
            p = npt_in["pres"]
            v = npt_info["v"]
            v_err = npt_info["v_err"]
            unit_cvt = 1e5 * (1e-10 ** 3) / pc.electron_volt
            pv = p * v * unit_cvt
            pv_err = p * v_err * unit_cvt * np.sqrt(3)
            print(f"# use pv from npt task: pv = {pv:.6e} pv_err = {pv_err:.6e}")
        elif manual_pv is not None:
            print(f"# use manual_pv={manual_pv}")
            pv = manual_pv
            pv_err = manual_pv_err if manual_pv_err is not None else thermo_info["pv_err"]
        else:
            pv = thermo_info["pv"]
            pv_err = thermo_info["pv_err"]
        e1 = fe + pv
        e1_err = np.sqrt(fe_err[0] ** 2 + pv_err ** 2)
        print("# Gibbs free ener per atom (err) [eV]:")
        print(print_format % (e1, e1_err, fe_err[1]))
    else:
        raise RuntimeError(f"unknown free_energy_type: {free_energy_type}")

    info = thermo_info.copy()
    info["free_energy_type"] = free_energy_type
    info["pv"] = pv
    info["pv_err"] = pv_err
    info["e1"] = e1
    info["e1_err"] = e1_err
    with open(os.path.join(job, "result.json"), "w") as result:
        result.write(json.dumps(info))
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_module_subparsers(main_subparsers):
    module_parser = main_subparsers.add_parser(
        "hti_liq_mag",
        help="Hamiltonian thermodynamic integration for magnetic liquid",
    )
    module_subparsers = module_parser.add_subparsers(
        help="commands of HTI for magnetic liquid",
        dest="command",
        required=True,
    )

    parser_gen = module_subparsers.add_parser("gen", help="Generate a job")
    parser_gen.add_argument("PARAM", type=str, help="json parameter file")
    parser_gen.add_argument(
        "-o", "--output", type=str, default="new_job",
        help="output folder for the job",
    )
    parser_gen.set_defaults(func=handle_gen)

    parser_compute = module_subparsers.add_parser(
        "compute", help="Compute the result of a job"
    )
    parser_compute.add_argument("JOB", type=str, help="folder of the job")
    parser_compute.add_argument(
        "-t", "--type", type=str, default="helmholtz",
        choices=["helmholtz", "gibbs"],
        help="type of free energy",
    )
    parser_compute.add_argument(
        "-g", "--pv", type=float, default=None,
        help="press*vol override for Gibbs free energy",
    )
    parser_compute.add_argument(
        "-G", "--pv-err", type=float, default=None,
        help="press*vol error override",
    )
    parser_compute.add_argument(
        "--npt", type=str, default=None,
        help="directory of the npt task for PV",
    )
    parser_compute.set_defaults(func=handle_compute)

    parser_run = module_subparsers.add_parser("run", help="Run the job")
    parser_run.add_argument("JOB", type=str, help="folder of the job")
    parser_run.add_argument("machine", type=str, help="machine.json file")
    parser_run.add_argument("task_name", type=str, help="task name, e.g. 00, 01, 02")
    parser_run.add_argument("--no-dp", action="store_true")
    parser_run.set_defaults(func=handle_run)


def handle_gen(args):
    param_path = os.path.abspath(args.PARAM)
    param_dir = os.path.dirname(param_path)
    jdata = json.load(open(param_path))

    for key in ("equi_conf", "model"):
        if key in jdata and jdata[key] is not None and not os.path.isabs(jdata[key]):
            jdata[key] = os.path.normpath(os.path.join(param_dir, jdata[key]))

    make_tasks(args.output, jdata)


def handle_run(args):
    hti.run_task(args.JOB, args.machine, args.task_name, args.no_dp)


def handle_compute(args):
    compute_task(
        job=args.JOB,
        free_energy_type=args.type,
        manual_pv=args.pv,
        manual_pv_err=args.pv_err,
        npt=args.npt,
    )
