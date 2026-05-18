#!/usr/bin/env python3
"""
Thermodynamic integration for magnetic spin systems (DeepSpin potential).

Extends ti.py to support LAMMPS atom_style spin with:
  - nvt        : fix nvt/spin (built-in spin thermostat, mass + rand)
  - nvt-langevin: fix nve/spin + fix langevin + fix langevin/spin
  - npt        : fix npt/spin (iso, built-in spin thermostat)

Supported paths:
  - t      : temperature path  (NVT or NPT), integrand = E/T²  (or H/T² for NPT)
  - t-ginv : temperature path on 1/T grid
  - p      : pressure path     (NPT only),   integrand = V
"""

import glob
import json
import os
import shutil

import numpy as np
import scipy.constants as pc
from dpdispatcher import Machine, Resources, Submission, Task

from dpti.lib.lammps import get_natoms, get_nspins, get_thermo
from dpti.lib.utils import (
    block_avg,
    compute_nrefine,
    create_path,
    get_first_matched_key_from_dict,
    get_task_file_abspath,
    integrate_range,
    parse_seq,
    relative_link_file,
)


def make_iter_name(iter_index):
    return "task_ti_mag." + ("%04d" % iter_index)


def parse_seq_ginv(seq):
    tmp_seq = parse_seq(seq)
    t_begin = tmp_seq[0]
    t_end = tmp_seq[-1]
    ngrid = len(tmp_seq) - 1
    hh = (1 / t_end - 1 / t_begin) / ngrid
    inv_grid = np.arange(1 / t_begin, 1 / t_end + 0.5 * hh, hh)
    return 1.0 / inv_grid


def _get_thermo_labels(lmplog):
    with open(lmplog) as fp:
        for line in fp:
            labels = line.split()
            if labels and labels[0] == "Step":
                return labels
    return []


def _get_spin_kinetic_col(lmplog, data):
    labels = _get_thermo_labels(lmplog)
    for idx, label in enumerate(labels):
        if label.lower() in ("spinkineng", "ske"):
            return idx

    # New ti-mag logs have exactly one extra column inserted after Volume.
    if data.ndim == 2 and data.shape[1] >= 17:
        return 8
    return None


def _gen_lammps_input(
    conf_file,
    mass_map,
    spin_mass,
    model,
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
    custom_variables=None,
    append=None,
):
    """Generate LAMMPS input for a single spin TI task (no spring coupling)."""
    ret = ""
    ret += "clear\n"
    ret += "# --------------------- VARIABLES -------------------------\n"
    ret += "variable        NSTEPS          equal %d\n" % nsteps
    ret += "variable        THERMO_FREQ     equal %d\n" % thermo_freq
    ret += "variable        DUMP_FREQ       equal %d\n" % dump_freq
    ret += f"variable        SP_MASS         equal {spin_mass:.6f}\n"
    ret += f"variable        TEMP            equal {temp:f}\n"
    ret += f"variable        PRES            equal {pres:f}\n"
    ret += f"variable        TAU_T           equal {tau_t:f}\n"
    ret += f"variable        TAU_P           equal {tau_p:f}\n"
    if custom_variables is not None:
        for key, value in custom_variables.items():
            ret += f"variable        {key} equal {value}\n"
    ret += "# ---------------------- INITIALIZATION -------------------\n"
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
    ret += "# --------------------- FORCE FIELDS ---------------------\n"
    if append:
        ret += f"pair_style      deepspin {model:s} {append:s}\n"
    else:
        ret += f"pair_style      deepspin {model:s}\n"
    ret += "pair_coeff      * *\n"
    ret += "# --------------------- MD SETTINGS ----------------------\n"
    ret += "neighbor        1.0 bin\n"
    ret += f"timestep        {timestep}\n"
    ret += "thermo          ${THERMO_FREQ}\n"
    ret += "compute         spin all property/atom sp spx spy spz fmx fmy fmz\n"
    ret += "compute         allmsd all msd\n"
    ret += "compute         spinmsd all msd/spin\n"
    ret += "thermo_style    custom step ke pe etotal enthalpy temp press vol ske c_allmsd[*] c_spinmsd[*]\n"
    ret += "thermo_modify   format float %20.6f\n"
    ret += "dump            1 all custom ${DUMP_FREQ} traj.dump id type x y z vx vy vz c_spin[1] c_spin[2] c_spin[3] c_spin[4]\n"
    # ----- ensemble / thermostat -----
    if ens == "nvt":
        ret += "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T} mass ${SP_MASS} rand %d lattice %d spin %d\n" % (
            np.random.default_rng().integers(1, 2**16),
            lattice_flag,
            spin_flag
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
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P} mass ${SP_MASS} rand %d lattice %d spin %d\n" % (
            np.random.default_rng().integers(1, 2**16),
            lattice_flag,
            spin_flag
        )
    elif ens == "npt-aniso":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P} mass ${SP_MASS} rand %d lattice %d spin %d \n" % (
            np.random.default_rng().integers(1, 2**16),
            lattice_flag,
            spin_flag
        )
    elif ens == "npt-tri":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P} mass ${SP_MASS} rand %d lattice %d spin %d\n" % (
            np.random.default_rng().integers(1, 2**16),
            lattice_flag,
            spin_flag
        )
    elif ens == "nve":
        ret += "fix             1 all nve/spin lattice_flag %d spin_flag %d\n" % (
            lattice_flag, spin_flag
        )
    else:
        raise RuntimeError(f"unknown ensemble '{ens}'")
    ret += "fix             mzero all momentum 10 linear 1 1 1\n"
    ret += "# --------------------- INITIALIZE -----------------------\n"
    if lattice_flag and spin_flag:
        ret += "velocity        all create ${TEMP} %d spin yes spmass ${SP_MASS}\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    elif lattice_flag:
        ret += "velocity        all create ${TEMP} %d\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    ret += "velocity        all zero linear\n"
    ret += "# --------------------- RUN ------------------------------\n"
    ret += "run             ${NSTEPS}\n"
    ret += "write_data      final.lmp\n"
    return ret


def make_tasks(iter_name, jdata):
    """Create TI task directories for a magnetic spin system."""
    ti_settings = jdata.copy()

    equi_conf = os.path.abspath(jdata["equi_conf"])
    copies = jdata.get("copies", None)
    model = jdata["model"]
    append = jdata.get("append", None)
    custom_variables = jdata.get("custom_variables", None)

    mass_map = get_first_matched_key_from_dict(jdata, ["model_mass_map", "mass_map"])
    spin_mass = get_first_matched_key_from_dict(
        jdata, ["spin_mass_map", "sp_mass_map", "spin_mass"]
    )
    nsteps = jdata["nsteps"]
    timestep = get_first_matched_key_from_dict(jdata, ["timestep", "dt"])
    thermo_freq = get_first_matched_key_from_dict(jdata, ["thermo_freq", "stat_freq"])
    dump_freq = get_first_matched_key_from_dict(
        jdata, ["dump_freq", "thermo_freq", "stat_freq"]
    )
    ens = jdata["ens"]
    path = jdata["path"]
    lattice_flag = int(jdata.get("lattice_flag", 1))
    spin_flag = int(jdata.get("spin_flag", 1))

    if "nvt" in ens:
        if path == "t":
            temp_seq = get_first_matched_key_from_dict(jdata, ["temp_seq", "temps"])
            temp_list = parse_seq(temp_seq)
            tau_t = jdata["tau_t"]
            ntasks = len(temp_list)
        else:
            raise RuntimeError("supported path for nvt ensemble is 't'")
    elif "npt" in ens:
        if path == "t":
            temp_seq = get_first_matched_key_from_dict(jdata, ["temp_seq", "temps"])
            temp_list = parse_seq(temp_seq)
            pres = get_first_matched_key_from_dict(jdata, ["pres", "press"])
            ntasks = len(temp_list)
        elif path == "t-ginv":
            temp_seq = get_first_matched_key_from_dict(jdata, ["temp_seq", "temps"])
            temp_list = parse_seq_ginv(temp_seq)
            pres = get_first_matched_key_from_dict(jdata, ["pres", "press"])
            ntasks = len(temp_list)
        elif path == "p":
            temp = get_first_matched_key_from_dict(jdata, ["temp", "temps"])
            pres_seq = get_first_matched_key_from_dict(jdata, ["pres_seq", "press"])
            pres_list = parse_seq(pres_seq)
            ntasks = len(pres_list)
        else:
            raise RuntimeError("supported paths for npt ensemble are 't', 't-ginv', 'p'")
        tau_t = jdata["tau_t"]
        tau_p = jdata["tau_p"]
    else:
        raise RuntimeError(f"unsupported ensemble '{ens}'")

    job_abs_dir = create_path(iter_name)
    ti_settings["equi_conf"] = relative_link_file(equi_conf, job_abs_dir)
    if model:
        ti_settings["model"] = relative_link_file(model, job_abs_dir)

    with open(os.path.join(job_abs_dir, "ti_settings.json"), "w") as f:
        json.dump(ti_settings, f, indent=4)

    for ii in range(ntasks):
        task_dir = os.path.join(job_abs_dir, "task.%06d" % ii)
        task_abs_dir = create_path(task_dir)

        relative_link_file(equi_conf, task_abs_dir)
        task_model = model
        if model:
            relative_link_file(model, task_abs_dir)
            task_model = os.path.basename(model)

        if "nvt" in ens and path == "t":
            lmp_str = _gen_lammps_input(
                os.path.basename(equi_conf),
                mass_map,
                spin_mass,
                task_model,
                nsteps,
                timestep,
                ens,
                temp_list[ii],
                tau_t=tau_t,
                thermo_freq=thermo_freq,
                dump_freq=dump_freq,
                copies=copies,
                lattice_flag=lattice_flag,
                spin_flag=spin_flag,
                custom_variables=custom_variables,
                append=append,
            )
            thermo_out = temp_list[ii]
        elif "npt" in ens and path in ("t", "t-ginv"):
            lmp_str = _gen_lammps_input(
                os.path.basename(equi_conf),
                mass_map,
                spin_mass,
                task_model,
                nsteps,
                timestep,
                ens,
                temp_list[ii],
                pres,
                tau_t=tau_t,
                tau_p=tau_p,
                thermo_freq=thermo_freq,
                dump_freq=dump_freq,
                copies=copies,
                lattice_flag=lattice_flag,
                spin_flag=spin_flag,
                custom_variables=custom_variables,
                append=append,
            )
            thermo_out = temp_list[ii]
        elif "npt" in ens and path == "p":
            lmp_str = _gen_lammps_input(
                os.path.basename(equi_conf),
                mass_map,
                spin_mass,
                task_model,
                nsteps,
                timestep,
                ens,
                temp,
                pres_list[ii],
                tau_t=tau_t,
                tau_p=tau_p,
                thermo_freq=thermo_freq,
                dump_freq=dump_freq,
                copies=copies,
                lattice_flag=lattice_flag,
                spin_flag=spin_flag,
                custom_variables=custom_variables,
                append=append,
            )
            thermo_out = pres_list[ii]
        else:
            raise RuntimeError("invalid ens/path combination")

        with open(os.path.join(task_abs_dir, "thermo.out"), "w") as fp:
            fp.write(f"{thermo_out:f}")
        with open(os.path.join(task_abs_dir, "in.lammps"), "w") as fp:
            fp.write(lmp_str)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _compute_thermo(lmplog, natoms, stat_skip, stat_bsize):
    data = get_thermo(lmplog)
    ea, ee = block_avg(data[:, 3], skip=stat_skip, block_size=stat_bsize)
    ha, he = block_avg(data[:, 4], skip=stat_skip, block_size=stat_bsize)
    ta, te = block_avg(data[:, 5], skip=stat_skip, block_size=stat_bsize)
    pa, pe = block_avg(data[:, 6], skip=stat_skip, block_size=stat_bsize)
    va, ve = block_avg(data[:, 7], skip=stat_skip, block_size=stat_bsize)
    unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
    return {
        "p": pa, "p_err": pe,
        "v": va / natoms, "v_err": ve / natoms,
        "e": ea / natoms, "e_err": ee / natoms,
        "h": ha / natoms, "h_err": he / natoms,
        "t": ta, "t_err": te,
        "pv": pa * va * unit_cvt / natoms,
        "pv_err": pe * va * unit_cvt / natoms,
    }


def _print_thermo_info(info, more_head=""):
    ptr = f"# thermodynamics (normalized by natoms) {more_head}\n"
    ptr += "# E (err)  [eV]:  {:20.8f} {:20.8f}\n".format(info["e"], info["e_err"])
    ptr += "# H (err)  [eV]:  {:20.8f} {:20.8f}\n".format(info["h"], info["h_err"])
    ptr += "# T (err)   [K]:  {:20.8f} {:20.8f}\n".format(info["t"], info["t_err"])
    ptr += "# P (err) [bar]:  {:20.8f} {:20.8f}\n".format(info["p"], info["p_err"])
    ptr += "# V (err) [A^3]:  {:20.8f} {:20.8f}\n".format(info["v"], info["v_err"])
    ptr += "# PV(err)  [eV]:  {:20.8f} {:20.8f}".format(info["pv"], info["pv_err"])
    print(ptr)


def _thermo_inte(jdata, Eo, Eo_err, all_t, integrand, integrand_err, scheme="s"):
    path = jdata["path"]
    ens = jdata["ens"]
    all_temps, all_press, all_fe, all_fe_err, all_fe_sys_err = [], [], [], [], []

    if path in ("t", "t-ginv"):
        # Integrate in β = 1/T space for numerical stability.
        # Gibbs-Helmholtz: d(G/T)/d(1/T) = H
        # → G(T)/T = G(T₀)/T₀ + ∫_{1/T₀}^{1/T} H dβ
        #
        # The T-space formula G = T(G₀/T₀ - ∫H/T²dT) is mathematically equivalent
        # but amplifies integration errors by factor T — catastrophic for fine β spacing.
        T0 = all_t[0]
        H_intg = integrand * all_t ** 2        # recover H from H/T²
        H_intg_err = integrand_err * all_t ** 2
        beta = 1.0 / all_t                     # β grid (may be inc or dec)

        beta_out, inte, inte_e, stat_e = integrate_range(beta, H_intg, H_intg_err, scheme)
        T_out = 1.0 / beta_out

        for ii in range(len(beta_out)):
            T_ii  = T_out[ii]
            diff_e   = inte[ii]      # ∫_{β₀}^{β_ii} H dβ
            err      = stat_e[ii]
            sys_err  = inte_e[ii]

            # G(T_ii) = T_ii × (G(T₀)/T₀ + ∫_{β₀}^{β_ii} H dβ)
            e1 = (Eo / T0 + diff_e) * T_ii
            err = np.sqrt(np.square(Eo_err / T0) + np.square(err)) * T_ii
            sys_err *= T_ii
            all_temps.append(T_ii)
            if "npt" in ens:
                all_press.append(
                    get_first_matched_key_from_dict(jdata, ["pres", "press"])
                )
            all_fe.append(e1)
            all_fe_err.append(err)
            all_fe_sys_err.append(sys_err)
    else:
        # pressure path: integrate V dP, no amplification issue
        all_t_out, inte, inte_e, stat_e = integrate_range(
            all_t, integrand, integrand_err, scheme
        )
        for ii in range(len(all_t_out)):
            diff_e  = inte[ii]
            err     = stat_e[ii]
            sys_err = inte_e[ii]
            e1 = Eo + diff_e
            err = np.sqrt(np.square(Eo_err) + np.square(err))
            all_temps.append(get_first_matched_key_from_dict(jdata, ["temp", "temps"]))
            all_press.append(all_t_out[ii])
            all_fe.append(e1)
            all_fe_err.append(err)
            all_fe_sys_err.append(sys_err)

    return (
        np.asarray(all_temps),
        np.asarray(all_press),
        np.asarray(all_fe),
        np.asarray(all_fe_err),
        np.asarray(all_fe_sys_err),
    )


def post_tasks(
    iter_name, jdata, Eo, Eo_err=0, To=None, natoms=None, scheme="simpson", shift=0.0
):
    """Post-process TI tasks for a magnetic spin system.

    Thermo columns (with ske and spinmsd):
      0:Step 1:KinEng 2:PotEng 3:TotEng 4:Enthalpy 5:Temp 6:Press 7:Vol 8:SKE
      9-12: c_allmsd[1..4]   13-16: c_spinmsd[1..4]
    """
    equi_conf = get_task_file_abspath(iter_name, jdata["equi_conf"])
    if natoms is None:
        natoms = get_natoms(equi_conf)
        if "copies" in jdata:
            natoms *= np.prod(jdata["copies"])
    stat_skip = jdata["stat_skip"]
    stat_bsize = jdata["stat_bsize"]
    ens = jdata["ens"]
    path = jdata["path"]

    try:
        press = get_first_matched_key_from_dict(jdata, ["pres", "press"])
    except KeyError:
        press = None

    all_tasks = sorted(glob.glob(os.path.join(iter_name, "task.[0-9]*")))
    ntasks = len(all_tasks)

    include_spin_kinetic = bool(jdata.get("include_spin_kinetic", False))

    stat_col2 = None
    if "nvt" in ens and path == "t":
        stat_col = 3          # TotEng
        print("# TI in NVT along T path (magnetic spin)")
    elif "npt" in ens and path in ("t", "t-ginv"):
        stat_col = 3          # TotEng (enthalpy = E + PV computed below)
        stat_col2 = 7
        unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
        print("# TI in NPT along T path (magnetic spin)")
    elif "npt" in ens and path == "p":
        stat_col = 7          # Volume
        print("# TI in NPT along P path (magnetic spin)")
    else:
        raise RuntimeError("invalid ens/path setting")
    print(f"# natoms: {natoms}")

    all_t = []
    all_e = []
    all_e_err = []
    integrand = []
    integrand_err = []
    all_enthalpy = []
    all_msd_xyz = []
    all_msd_spin = []

    for ii in all_tasks:
        tt = float(open(os.path.join(ii, "thermo.out")).read())
        all_t.append(tt)

        log_name = os.path.join(ii, "log.lammps")
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, "data"), data, fmt="%20.6f")
        spin_kinetic_col = _get_spin_kinetic_col(log_name, data)

        if stat_col2 is not None:
            ea, ee = block_avg(
                data[:, stat_col] + press * data[:, stat_col2] * unit_cvt,
                skip=stat_skip,
                block_size=stat_bsize,
            )
        else:
            ea, ee = block_avg(data[:, stat_col], skip=stat_skip, block_size=stat_bsize)

        enthalpy, _ = block_avg(data[:, 4], skip=stat_skip, block_size=stat_bsize)

        if path in ("t", "t-ginv") and not include_spin_kinetic:
            if spin_kinetic_col is None:
                raise RuntimeError(
                    f"{log_name} does not contain a SpinKinEng/ske thermo column; "
                    "regenerate the ti-mag task with the updated input generator, "
                    "or set include_spin_kinetic=true for legacy logs."
                )
            ska, _ = block_avg(
                data[:, spin_kinetic_col],
                skip=stat_skip,
                block_size=stat_bsize,
            )
            ea -= ska
            enthalpy -= ska

        # COM correction: 3/2 kBT per atom for translational DoF
        if path in ("t", "t-ginv"):
            ea += 1.5 * pc.Boltzmann * tt / pc.electron_volt
        elif path == "p":
            ea += 1.5 * pc.Boltzmann * jdata["temp"] / pc.electron_volt

        ea -= shift
        ea /= natoms
        ee /= natoms

        # msd columns: c_allmsd[4] = col -5, c_spinmsd[4] = col -1
        msd_xyz  = data[-1, -5]
        msd_spin = data[-1, -1]

        all_e.append(ea)
        all_e_err.append(ee)
        all_enthalpy.append(enthalpy)
        all_msd_xyz.append(msd_xyz)
        all_msd_spin.append(msd_spin)

        if path in ("t", "t-ginv"):
            integrand.append(ea / (tt * tt))
            integrand_err.append(ee / (tt * tt))
        elif path == "p":
            unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
            integrand.append(ea * unit_cvt)
            integrand_err.append(ee * unit_cvt)

    all_print = np.array([
        all_t, integrand, all_e, all_e_err, all_enthalpy, all_msd_xyz, all_msd_spin
    ])
    np.savetxt(
        os.path.join(iter_name, "ti.out"),
        all_print.T,
        fmt="%.12e",
        header="t/p Integrand U/V U/V_err enthalpy msd_xyz msd_spin",
    )

    info0 = _compute_thermo(
        os.path.join(all_tasks[0], "log.lammps"), natoms, stat_skip, stat_bsize
    )
    info1 = _compute_thermo(
        os.path.join(all_tasks[-1], "log.lammps"), natoms, stat_skip, stat_bsize
    )
    _print_thermo_info(info0, "at start point")
    _print_thermo_info(info1, "at end point")

    # ── integrate ──
    if To is not None:
        index = all_t.index(To)
        all_t_1  = np.flip(all_t[:index + 1])
        intg_1   = np.flip(integrand[:index + 1])
        intge_1  = np.flip(integrand_err[:index + 1])
        all_t_2  = np.array(all_t[index:])
        intg_2   = np.array(integrand[index:])
        intge_2  = np.array(integrand_err[index:])

        temps1, press1, fe1, fe_err1, fe_sys_err1 = _thermo_inte(
            jdata, Eo, Eo_err, all_t_1, intg_1, intge_1, scheme
        )
        temps2, press2, fe2, fe_err2, fe_sys_err2 = _thermo_inte(
            jdata, Eo, Eo_err, all_t_2, intg_2, intge_2, scheme
        )
        all_temps = np.append(np.flip(temps1), temps2[1:])
        all_press = np.append(np.flip(press1), press2[1:])
        all_fe    = np.append(np.flip(fe1),    fe2[1:])
        all_fe_err     = np.append(np.flip(fe_err1),     fe_err2[1:])
        all_fe_sys_err = np.append(np.flip(fe_sys_err1), fe_sys_err2[1:])
    else:
        all_temps, all_press, all_fe, all_fe_err, all_fe_sys_err = _thermo_inte(
            jdata, Eo, Eo_err, np.array(all_t), np.array(integrand),
            np.array(integrand_err), scheme
        )

    # ── print results ──
    result = ""
    if "nvt" == ens:
        header = "#%8s  %20s  %9s  %9s  %9s" % (
            "T(ctrl)", "F", "stat_err", "inte_err", "err"
        )
        print(header)
        result += header + "\n"
        for ii in range(len(all_temps)):
            tot_err = np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]])
            line = (
                f"{all_temps[ii]:9.2f}  {all_fe[ii]:20.12f}  "
                f"{all_fe_err[ii]:9.2e}  {all_fe_sys_err[ii]:9.2e}  {tot_err:9.2e}"
            )
            print(line)
            result += line + "\n"
    elif "npt" in ens:
        header = "#%8s  %15s  %20s  %9s  %9s  %9s" % (
            "T(ctrl)", "P(ctrl)", "F", "stat_err", "inte_err", "err"
        )
        print(header)
        result += header + "\n"
        for ii in range(len(all_temps)):
            tot_err = np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]])
            line = (
                f"{all_temps[ii]:9.2f}  {all_press[ii]:15.8e}  {all_fe[ii]:20.12f}  "
                f"{all_fe_err[ii]:9.2e}  {all_fe_sys_err[ii]:9.2e}  {tot_err:9.2e}"
            )
            print(line)
            result += line + "\n"

    data_out = {
        "all_temps": all_temps.tolist(),
        "all_press": all_press.tolist(),
        "all_fe": all_fe.tolist(),
        "all_fe_stat_err": all_fe_err.tolist(),
        "all_fe_inte_err": all_fe_sys_err.tolist(),
        "all_fe_tot_err": [
            float(np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]]))
            for ii in range(len(all_temps))
        ],
    }
    info = {
        "start_point_info": info0,
        "end_point_info": info1,
        "data": data_out,
    }
    with open(os.path.join(iter_name, "../", "result"), "w") as f:
        f.write(result)
    with open(os.path.join(iter_name, "result.json"), "w") as f:
        json.dump(info, f)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Task helpers
# ─────────────────────────────────────────────────────────────────────────────

def refine_task(from_task, to_task, err):
    from_task = os.path.abspath(from_task)
    to_task = os.path.abspath(to_task)
    from_jdata = json.load(open(os.path.join(from_task, "ti_settings.json")))
    to_jdata = from_jdata.copy()
    path = from_jdata["path"]

    from_ti = os.path.join(from_task, "ti.out")
    if not os.path.isfile(from_ti):
        raise RuntimeError(
            f"cannot find {from_ti}, task must be computed before refining"
        )
    tmp_array = np.loadtxt(from_ti)
    all_t = tmp_array[:, 0]
    integrand = tmp_array[:, 1]
    ntask = all_t.size

    if path in ("t", "t-ginv"):
        interval_nrefine = compute_nrefine(all_t, integrand, err, all_t)
    elif path == "p":
        interval_nrefine = compute_nrefine(all_t, integrand, err)
    else:
        raise RuntimeError(f"unknown path '{path}'")

    refined_t = []
    back_map = []
    for ii in range(ntask - 1):
        refined_t.append(all_t[ii])
        back_map.append(ii)
        hh = (all_t[ii + 1] - all_t[ii]) / interval_nrefine[ii]
        for jj in range(1, interval_nrefine[ii]):
            refined_t.append(all_t[ii] + jj * hh)
            back_map.append(-1)
    refined_t.append(all_t[-1])
    back_map.append(ntask - 1)

    if to_jdata["path"] == "t-ginv":
        to_jdata["path"] = "t"
    if to_jdata["path"] == "t":
        to_jdata["temps"] = refined_t
    elif to_jdata["path"] == "p":
        to_jdata["press"] = refined_t
    else:
        raise RuntimeError(f"unknown path '{path}'")
    to_jdata["orig_task"] = from_task
    to_jdata["back_map"] = back_map
    to_jdata["refine_error"] = err
    to_jdata["equi_conf"] = get_task_file_abspath(from_task, from_jdata["equi_conf"])
    to_jdata["model"] = get_task_file_abspath(from_task, from_jdata["model"])

    make_tasks(to_task, to_jdata)

    from_task_list = sorted(glob.glob(os.path.join(from_task, "task.[0-9]*")))
    to_task_list = sorted(glob.glob(os.path.join(to_task, "task.[0-9]*")))
    assert len(from_task_list) == ntask
    assert len(to_task_list) == len(refined_t)

    for ii in range(len(to_task_list)):
        if back_map[ii] < 0:
            continue
        for jj in ["data", "log.lammps"]:
            shutil.copyfile(
                os.path.join(from_task_list[back_map[ii]], jj),
                os.path.join(to_task_list[ii], jj),
            )
        with open(os.path.join(to_task_list[ii], "from.dir"), "w") as fp:
            fp.write(from_task_list[back_map[ii]])


def compute_task(job, Eo, Eo_err, To, scheme="simpson"):
    with open(os.path.join(job, "ti_settings.json")) as f:
        jdata = json.load(f)
    return post_tasks(job, jdata, Eo=Eo, Eo_err=Eo_err, To=To, scheme=scheme)


def run_task(task_name, machine_file):
    task_dir_list = sorted(glob.glob(os.path.join(task_name, "task.*")))
    work_base_dir = os.getcwd()
    with open(machine_file) as f:
        mdata = json.load(f)
    machine = Machine.load_from_dict(mdata["machine"])
    resources = Resources.load_from_dict(mdata["resources"])

    submission = Submission(
        work_base=work_base_dir,
        resources=resources,
        machine=machine,
    )
    task_list = [
        Task(
            command=mdata["command"] + " -in in.lammps",
            task_work_path=ii,
            forward_files=["in.lammps", "*.lmp"],
            backward_files=["log*", "final.lmp", "traj.dump"],
        )
        for ii in task_dir_list
    ]
    submission.register_task_list(task_list=task_list)
    submission.run_submission()


# ─────────────────────────────────────────────────────────────────────────────
# CLI handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_gen(args):
    jdata = json.load(open(args.PARAM))
    make_tasks(args.output, jdata)


def handle_compute(args):
    job = args.JOB
    jdata = json.load(open(os.path.join(job, "ti_settings.json")))
    path = jdata["path"]

    Eo = args.Eo
    Eo_err = args.Eo_err if args.Eo_err is not None else 0.0
    To = args.To

    if args.hti is not None:
        if Eo is not None:
            raise ValueError(
                "Both --Eo and --hti are provided; --Eo will be overridden by "
                "the e1 value in the hti result.json file."
            )
        hti_result = json.load(open(os.path.join(args.hti, "result.json")))
        hti_in = json.load(open(os.path.join(args.hti, "in.json")))
        Eo = hti_result["e1"]
        Eo_err = hti_result.get("e1_err", 0.0)
        if To is None:
            if path in ("t", "t-ginv"):
                To = hti_in.get("temp")
                if To is None:
                    raise ValueError("Cannot find 'temp' in hti input json")
            elif path == "p":
                try:
                    To = get_first_matched_key_from_dict(hti_in, ["pres", "press"])
                except KeyError:
                    raise ValueError("Cannot find 'pres'/'press' in hti input json")

    if Eo is None:
        raise ValueError("Free energy of starting point must be supplied via -e/--Eo or -H/--hti")

    compute_task(job, Eo=Eo, Eo_err=Eo_err, To=To, scheme=args.scheme)


def handle_refine(args):
    refine_task(args.input, args.output, args.error)


def handle_run(args):
    run_task(args.JOB, args.machine)


# ─────────────────────────────────────────────────────────────────────────────
# Subparser registration
# ─────────────────────────────────────────────────────────────────────────────

def add_module_subparsers(main_subparsers):
    module_parser = main_subparsers.add_parser(
        "ti-mag",
        help="thermodynamic integration for magnetic spin systems",
    )
    module_subparsers = module_parser.add_subparsers(
        help="commands of TI for magnetic spin systems",
        dest="command",
        required=True,
    )

    # ── gen ──────────────────────────────────────────────────────────────────
    parser_gen = module_subparsers.add_parser("gen", help="generate task folders")
    parser_gen.add_argument("PARAM", type=str, help="JSON parameter file")
    parser_gen.add_argument(
        "-o", "--output",
        type=str,
        default="new_job",
        help="output folder for the job (default: new_job)",
    )
    parser_gen.set_defaults(func=handle_gen)

    # ── compute ───────────────────────────────────────────────────────────────
    parser_compute = module_subparsers.add_parser(
        "compute", help="post-process and integrate results"
    )
    parser_compute.add_argument("JOB", type=str, help="job folder")
    parser_compute.add_argument(
        "-e", "--Eo",
        type=float,
        default=None,
        help="free energy of the starting point (eV/atom)",
    )
    parser_compute.add_argument(
        "-E", "--Eo-err",
        type=float,
        default=None,
        help="statistical error of the starting free energy",
    )
    parser_compute.add_argument(
        "-t", "--To",
        type=float,
        default=None,
        help="starting thermodynamic coordinate (T in K or P in bar)",
    )
    parser_compute.add_argument(
        "-s", "--scheme",
        type=str,
        default="simpson",
        help="numerical integration scheme (default: simpson)",
    )
    parser_compute.add_argument(
        "-H", "--hti",
        type=str,
        default=None,
        help="HTI job folder; extracts Eo and To from its result.json / in.json",
    )
    parser_compute.set_defaults(func=handle_compute)

    # ── refine ────────────────────────────────────────────────────────────────
    parser_refine = module_subparsers.add_parser(
        "refine", help="refine the grid of a completed job"
    )
    parser_refine.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="input job folder",
    )
    parser_refine.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="output job folder",
    )
    parser_refine.add_argument(
        "-e", "--error",
        type=float,
        required=True,
        help="target integration error",
    )
    parser_refine.set_defaults(func=handle_refine)

    # ── run ───────────────────────────────────────────────────────────────────
    parser_run = module_subparsers.add_parser("run", help="submit tasks via dpdispatcher")
    parser_run.add_argument("JOB", type=str, help="job folder")
    parser_run.add_argument("machine", type=str, help="machine.json file")
    parser_run.set_defaults(func=handle_run)
