#!/usr/bin/env python3

import glob
import json
import os
import shutil

import numpy as np
import pymbar
import scipy.constants as pc

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from dpdispatcher import Machine, Resources, Submission, Task

from dpti.einstein import (
    free_energy,
    frenkel,
    magnetic_frenkel,
    magnetic_frenkel_modulus,
)
from dpti.lib.lammps import get_natoms, get_thermo, get_nspins

# from lib.utils import integrate_sys_err
from dpti.lib.utils import (
    block_avg,
    compute_nrefine,
    create_path,
    get_first_matched_key_from_dict,
    get_task_file_abspath,
    integrate_range_hti,
    parse_seq,
    relative_link_file,
)


def make_iter_name(iter_index):
    return "task_hti." + ("%04d" % iter_index)


def _get_spring_lambda_mode(jdata):
    spring_lambda_mode = jdata.get("spring_lambda_mode", "joint")
    aliases = {
        "same": "joint",
        "together": "joint",
        "joint": "joint",
        "separate": "split",
        "split": "split",
        "lattice_only": "lattice_only",
        "lattice-only": "lattice_only",
        "spin_only": "spin_only",
        "spin-only": "spin_only",
    }
    if spring_lambda_mode not in aliases:
        raise RuntimeError(
            "unknown spring_lambda_mode '{}', expected one of: {}".format(
                spring_lambda_mode,
                ", ".join(sorted(set(aliases.keys()))),
            )
        )
    return aliases[spring_lambda_mode]


def _get_spin_reference_mode(jdata):
    spin_reference = jdata.get("spin_reference")
    if spin_reference is None:
        if "spin_ref" in jdata:
            spin_reference = "modulus"
        elif any(
            key in jdata
            for key in [
                "spin_spring_k",
                "spring_spin_k",
                "s_spring_k",
                "spring_k_spin",
                "lambda_spin_spring_off",
                "spin_model",
                "spin_mu",
            ]
        ):
            spin_reference = "vector"
        else:
            spin_reference = "none"

    aliases = {
        "vector": "vector",
        "spin": "vector",
        "spring": "vector",
        "spin_spring": "vector",
        "spin-spring": "vector",
        "modulus": "modulus",
        "mod": "modulus",
        "spin_mod": "modulus",
        "spin-mod": "modulus",
        "spin_modulus": "modulus",
        "spin-modulus": "modulus",
        "none": "none",
        "no": "none",
        "false": "none",
    }
    key = str(spin_reference).lower()
    if key not in aliases:
        raise RuntimeError(
            "unknown spin_reference '{}', expected one of: {}".format(
                spin_reference,
                ", ".join(sorted(set(aliases.keys()))),
            )
        )
    return aliases[key]


def _expand_type_param(value, ntypes, name):
    if isinstance(value, (list, tuple)):
        if len(value) != ntypes:
            raise ValueError(f"{name} length {len(value)} does not match ntypes {ntypes}")
        return [float(v) for v in value]
    return [float(value) for _ in range(ntypes)]


def _get_spin_ref_params(jdata, mass_map, spin_mass, spin_map):
    ntypes = len(mass_map)
    spin_ref = jdata.get("spin_ref")
    if spin_ref is not None:
        style = spin_ref.get("style", "spring")
        k_by_type = _expand_type_param(spin_ref.get("k", 0.0), ntypes, "spin_ref.k")
        s0_by_type = _expand_type_param(spin_ref.get("s0", 0.0), ntypes, "spin_ref.s0")
    else:
        style = jdata.get("spin_mod_style", "spring")
        spin_mod_k = jdata.get("spin_mod_k", 0.0)
        base_k = _expand_type_param(spin_mod_k, ntypes, "spin_mod_k")
        k_by_type = [base_k[i] * mass_map[i] * spin_mass for i in range(ntypes)]
        s0_by_type = _expand_type_param(jdata.get("spin_mod_s0", 0.0), ntypes, "spin_mod_s0")

    style = str(style).lower()
    if style not in ("spring", "harmonic"):
        raise RuntimeError("hti_mag spin_reference='modulus' supports only spring/harmonic spin_ref")

    k_by_type = [k_by_type[i] if spin_map[i] else 0.0 for i in range(ntypes)]
    return style, k_by_type, s0_by_type


def _ff_lj_on(lamb, model, sparam):
    nn = sparam["n"]
    alpha_lj = sparam["alpha_lj"]
    rcut = sparam["rcut"]
    epsilon = sparam["epsilon"]
    # sigma = sparam['sigma']
    # sigma_oo = sparam['sigma_oo']
    # sigma_oh = sparam['sigma_oh']
    # sigma_hh = sparam['sigma_hh']
    activation = sparam["activation"]
    ret = ""
    ret += f"variable        EPSILON equal {epsilon:f}\n"
    ret += f"pair_style      lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"

    element_num = sparam.get("element_num", 1)
    sigma_key_index = filter(
        lambda t: t[0] <= t[1],
        ((i, j) for i in range(element_num) for j in range(element_num)),
    )
    for i, j in sigma_key_index:
        ret += "pair_coeff      {} {} ${{EPSILON}} {:f} {:f}\n".format(
            i + 1,
            j + 1,
            sparam["sigma_" + str(i) + "_" + str(j)],
            activation,
        )

    # ret += 'pair_coeff      * * ${EPSILON} %f %f\n' % (sigma, activation)
    # ret += 'pair_coeff      1 1 ${EPSILON} %f %f\n' % (sigma_oo, activation)
    # ret += 'pair_coeff      1 2 ${EPSILON} %f %f\n' % (sigma_oh, activation)
    # ret += 'pair_coeff      2 2 ${EPSILON} %f %f\n' % (sigma_hh, activation)
    ret += "fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes\n"
    ret += "compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON\n"
    return ret


def _ff_deep_on(lamb, model, sparam, if_meam=False, meam_model=None, append=None):
    nn = sparam["n"]
    alpha_lj = sparam["alpha_lj"]
    rcut = sparam["rcut"]
    epsilon = sparam["epsilon"]
    # sigma = sparam['sigma']
    # sigma_oo = sparam['sigma_oo']
    # sigma_oh = sparam['sigma_oh']
    # sigma_hh = sparam['sigma_hh']
    activation = sparam["activation"]
    ret = ""
    ret += f"variable        EPSILON equal {epsilon:f}\n"
    ret += "variable        ONE equal 1\n"
    # if if_meam:
    #     ret += 'pair_style      hybrid/overlay meam lj/cut/soft %f %f %f  \n' % (nn, alpha_lj, rcut)
    #     ret += 'pair_coeff      * * meam /home/fengbo/4_Sn/meam_files/library_18Metal.meam Sn /home/fengbo/4_Sn/meam_files/Sn_18Metal.meam Sn \n'
    if if_meam:
        ret += f"pair_style      hybrid/overlay meam lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        ret += f'pair_coeff      * * meam {meam_model["library"]} {meam_model["element"]} {meam_model["potential"]} {meam_model["element"]}\n'
    else:
        if append:
            ret += f"pair_style      hybrid/overlay deepmd {model:s} {append:s} lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        else:
            ret += f"pair_style      hybrid/overlay deepmd {model:s} lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        ret += "pair_coeff      * * deepmd\n"
    element_num = sparam.get("element_num", 1)
    sigma_key_index = filter(
        lambda t: t[0] <= t[1],
        ((i, j) for i in range(element_num) for j in range(element_num)),
    )
    for i, j in sigma_key_index:
        ret += "pair_coeff      {} {} lj/cut/soft ${{EPSILON}} {:f} {:f}\n".format(
            i + 1,
            j + 1,
            sparam["sigma_" + str(i) + "_" + str(j)],
            activation,
        )

    # ret += 'pair_coeff      * * lj/cut/soft ${EPSILON} %f %f\n' % (sigma, activation)
    # ret += 'pair_coeff      1 1 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oo, activation)
    # ret += 'pair_coeff      1 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oh, activation)
    # ret += 'pair_coeff      2 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_hh, activation)
    if if_meam:
        ret += "fix             tot_pot all adapt/fep 0 pair meam scale * * v_LAMBDA\n"
        ret += "compute         e_diff all fep ${TEMP} pair meam scale * * v_ONE\n"
    else:
        ret += (
            "fix             tot_pot all adapt/fep 0 pair deepmd scale * * v_LAMBDA\n"
        )
        ret += "compute         e_diff all fep ${TEMP} pair deepmd scale * * v_ONE\n"
    return ret


# def _ff_meam_on(lamb,
#                 model,
#                 sparam):
#     nn = sparam['n']
#     alpha_lj = sparam['alpha_lj']
#     rcut = sparam['rcut']
#     epsilon = sparam['epsilon']
# sigma = sparam['sigma']
# sigma_oo = sparam['sigma_oo']
# sigma_oh = sparam['sigma_oh']
# sigma_hh = sparam['sigma_hh']
#     activation = sparam['activation']
#     ret = ''
#     ret += 'variable        EPSILON equal %f\n' % epsilon
#     ret += 'variable        ONE equal 1\n'
#     ret += 'pair_style      hybrid/overlay meam lj/cut/soft %f %f %f  \n' % (nn, alpha_lj, rcut)
#     ret += 'pair_coeff      * * meam /home/fengbo/4_Sn/meam_files/library_18Metal.meam Sn /home/fengbo/4_Sn/meam_files/Sn_18Metal.meam Sn \n'

#     element_num=sparam.get('element_num', 1)
#     sigma_key_index = filter(lambda t:t[0] <= t[1], ((i,j) for i in range(element_num) for j in range(element_num)))
#     for (i, j) in sigma_key_index:
#         ret += 'pair_coeff      %s %s lj/cut/soft ${EPSILON} %f %f\n' % (i+1, j+1, sparam['sigma_'+str(i)+'_'+str(j)], activation)

# ret += 'pair_coeff      * * lj/cut/soft ${EPSILON} %f %f\n' % (sigma, activation)
# ret += 'pair_coeff      1 1 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oo, activation)
# ret += 'pair_coeff      1 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oh, activation)
# ret += 'pair_coeff      2 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_hh, activation)
#     if if_meam:
#         ret += 'pair_style      hybrid/overlay meam lj/cut/soft %f %f %f  \n' % (nn, alpha_lj, rcut)
#         ret += 'pair_coeff      * * meam /home/fengbo/4_Sn/meam_files/library_18Metal.meam Sn /home/fengbo/4_Sn/meam_files/Sn_18Metal.meam Sn \n'
#     else:
#         ret += 'fix             tot_pot all adapt/fep 0 pair meam scale * * v_LAMBDA\n'
#         ret += 'compute         e_diff all fep ${TEMP} pair meam scale * * v_ONE\n'
#     return ret


def _ff_lj_off(lamb, model, sparam, if_meam=False, meam_model=None, append=None):
    nn = sparam["n"]
    alpha_lj = sparam["alpha_lj"]
    rcut = sparam["rcut"]
    epsilon = sparam["epsilon"]
    # sigma = sparam['sigma']
    # sigma_oo = sparam['sigma_oo']
    # sigma_oh = sparam['sigma_oh']
    # sigma_hh = sparam['sigma_hh']
    activation = sparam["activation"]
    ret = ""
    ret += f"variable        EPSILON equal {epsilon:f}\n"
    ret += "variable        INV_EPSILON equal -${EPSILON}\n"
    # if if_meam:
    #     ret += 'pair_style      hybrid/overlay meam lj/cut/soft %f %f %f  \n'  % (nn, alpha_lj, rcut)
    #     ret += 'pair_coeff      * * meam /home/fengbo/4_Sn/meam_files/library_18Metal.meam Sn /home/fengbo/4_Sn/meam_files/Sn_18Metal.meam Sn\n'
    if if_meam:
        ret += f"pair_style      hybrid/overlay meam lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        ret += f'pair_coeff      * * meam {meam_model["library"]} {meam_model["element"]} {meam_model["potential"]} {meam_model["element"]}\n'
        # ret += f'pair_coeff      * * meam {meam_model[0]} {meam_model[2]} {meam_model[1]} {meam_model[2]}\n'
    else:
        if append:
            ret += f"pair_style      hybrid/overlay deepmd {model:s} {append:s} lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        else:
            ret += f"pair_style      hybrid/overlay deepmd {model:s} lj/cut/soft {nn:f} {alpha_lj:f} {rcut:f}\n"
        ret += "pair_coeff      * * deepmd\n"

    element_num = sparam.get("element_num", 1)
    sigma_key_index = filter(
        lambda t: t[0] <= t[1],
        ((i, j) for i in range(element_num) for j in range(element_num)),
    )
    for i, j in sigma_key_index:
        ret += "pair_coeff      {} {} lj/cut/soft ${{EPSILON}} {:f} {:f}\n".format(
            i + 1,
            j + 1,
            sparam["sigma_" + str(i) + "_" + str(j)],
            activation,
        )

    # ret += 'pair_coeff      * * lj/cut/soft ${EPSILON} %f %f\n' % (sigma, activation)
    # ret += 'pair_coeff      1 1 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oo, activation)
    # ret += 'pair_coeff      1 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oh, activation)
    # ret += 'pair_coeff      2 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_hh, activation)
    ret += "fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes\n"
    ret += "compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON\n"
    return ret


# def _ff_meam_lj_off(lamb,
#                model,
#                sparam) :
#     nn = sparam['n']
#     alpha_lj = sparam['alpha_lj']
#     rcut = sparam['rcut']
#     epsilon = sparam['epsilon']
# sigma = sparam['sigma']
# sigma_oo = sparam['sigma_oo']
# sigma_oh = sparam['sigma_oh']
# sigma_hh = sparam['sigma_hh']
#     activation = sparam['activation']
#     ret = ''
#     ret += 'variable        EPSILON equal %f\n' % epsilon
#     ret += 'variable        INV_EPSILON equal -${EPSILON}\n'
#     ret += 'pair_style      hybrid/overlay meam lj/cut/soft %f %f %f  \n'  % (nn, alpha_lj, rcut)
#     ret += 'pair_coeff      * * meam /home/fengbo/4_Sn/meam_files/library_18Metal.meam Sn /home/fengbo/4_Sn/meam_files/Sn_18Metal.meam Sn\n'

#     element_num=sparam.get('element_num', 1)
#     sigma_key_index = filter(lambda t:t[0] <= t[1], ((i,j) for i in range(element_num) for j in range(element_num)))
#     for (i, j) in sigma_key_index:
#         ret += 'pair_coeff      %s %s lj/cut/soft ${EPSILON} %f %f\n' % (i+1, j+1, sparam['sigma_'+str(i)+'_'+str(j)], activation)

# ret += 'pair_coeff      * * lj/cut/soft ${EPSILON} %f %f\n' % (sigma, activation)
# ret += 'pair_coeff      1 1 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oo, activation)
# ret += 'pair_coeff      1 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oh, activation)
# ret += 'pair_coeff      2 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_hh, activation)
#     ret += 'fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes\n'
#     ret += 'compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON\n'
#     return ret


def _ff_spring(lamb, m_spring_k, var_spring, enabled=True):
    ret = ""
    if not enabled:
        ret += "variable        l_spring equal 0.0\n"
        return ret
    ntypes = len(m_spring_k)
    for ii in range(ntypes):
        ret += f"group           type_{ii + 1} type {ii + 1}\n"
    for ii in range(ntypes):
        if var_spring:
            m_spring_const = m_spring_k[ii] * (1 - lamb)
        else:
            m_spring_const = m_spring_k[ii]
        ret += f"fix             l_spring_{ii + 1} type_{ii + 1} spring/self {m_spring_const:.10e}\n"
        ret += "fix_modify      l_spring_%s energy yes\n" % (ii + 1)
    sum_str = "f_l_spring_1"
    for ii in range(1, ntypes):
        sum_str += "+f_l_spring_%s" % (ii + 1)
    ret += f"variable        l_spring equal {sum_str}\n"
    return ret

def _ff_spring_spin(lamb, m_spring_spin_k, var_spring, enabled=True):
    ret = ""
    if not enabled:
        ret += "variable        l_spring_spin equal 0.0\n"
        return ret
    active_types = [ii for ii, value in enumerate(m_spring_spin_k) if value > 0.0]
    if not active_types:
        ret += "variable        l_spring_spin equal 0.0\n"
        return ret
    for ii in active_types:
        ret += f"group           type_{ii + 1} type {ii + 1}\n"
    for ii in active_types:
        if var_spring:
            m_spring_const = m_spring_spin_k[ii] * (1 - lamb)
        else:
            m_spring_const = m_spring_spin_k[ii]
        ret += f"fix             l_spring_spin_{ii + 1} type_{ii + 1} spring/spin {m_spring_const:.10e}\n"
        ret += "fix_modify      l_spring_spin_%s energy yes\n" % (ii + 1)
    sum_str = "+".join("f_l_spring_spin_%s" % (ii + 1) for ii in active_types)
    ret += f"variable        l_spring_spin equal {sum_str}\n"
    return ret


def _ff_spin_mod(lamb, m_spring_spin_k, spin_ref_s0, spin_map, mode, enabled=True):
    ret = ""
    if not enabled:
        ret += "variable        l_spring_spin equal 0.0\n"
        return ret

    if mode == "full":
        factor = 1.0
    elif mode == "off":
        factor = 1.0 - lamb
    else:
        raise RuntimeError("unknown spin_mod mode", mode)

    ntypes = len(m_spring_spin_k)
    spin_types = [ii for ii in range(ntypes) if spin_map[ii]]
    if not spin_types or factor <= 0.0:
        ret += "variable        l_spring_spin equal 0.0\n"
        return ret

    for ii in spin_types:
        ret += f"group           type_{ii + 1} type {ii + 1}\n"
    for ii in spin_types:
        m_spring_const = m_spring_spin_k[ii] * factor
        ret += f"fix             l_spring_spin_{ii + 1} type_{ii + 1} spring/spin/mod {m_spring_const:.10e} {spin_ref_s0[ii]:.10e}\n"
        ret += "fix_modify      l_spring_spin_%s energy yes\n" % (ii + 1)
    sum_str = "+".join("f_l_spring_spin_%s" % (ii + 1) for ii in spin_types)
    ret += f"variable        l_spring_spin equal {sum_str}\n"
    return ret


def _ff_soft_lj(
    lamb, model, m_spring_k, step, sparam, if_meam=False, meam_model=None, append=None
):
    ret = ""
    ret += "# --------------------- FORCE FIELDS ---------------------\n"
    if step == "lj_on":
        ret += _ff_lj_on(lamb, model, sparam)
        var_spring = False
    elif step == "deep_on":
        # ret += _ff_meam_on(lamb, model, sparam)
        ret += _ff_deep_on(
            lamb, model, sparam, if_meam=if_meam, meam_model=meam_model, append=append
        )
        var_spring = False
    elif step == "spring_off":
        # ret += _ff_meam_lj_off(lamb, model, sparam)
        ret += _ff_lj_off(
            lamb, model, sparam, if_meam=if_meam, meam_model=meam_model, append=append
        )
        var_spring = True
    else:
        raise RuntimeError("unkown step", step)

    ret += _ff_spring(lamb, m_spring_k, var_spring)

    return ret


def _ff_two_steps(
    lamb,
    model,
    m_spring_k,
    m_spring_spin_k,
    step,
    spin_ref_s0=None,
    spin_map=None,
    spin_reference="vector",
    append=None,
    if_meam=False,
    meam_model=None,
    spring_lambda_mode="joint",
    if_harmonic=False,
    m_target_spring_k=None,
    m_target_spring_spin_k=None,
    couple_c=0.0,
    zeeman_B=0.0,
    zeeman_dir=(0.0, 0.0, 1.0),
    magvol_lambda=0.0,
    magvol_v0=0.0,
    magvol_s0=0.0,
):
    ret = ""
    ret += "# --------------------- FORCE FIELDS ---------------------\n"
    if if_harmonic:
        # harmonic (Einstein crystal only): zero pair potential for validation
        ret += "pair_style      zero 10.0\n"
        ret += "pair_coeff      * *\n"
    elif if_meam:
        ret += "pair_style      meam\n"
        ret += f'pair_coeff      * * {meam_model["library"]} {meam_model["element"]} {meam_model["potential"]} {meam_model["element"]}\n'
    else:
        if append:
            ret += f"pair_style      deepspin {model:s} {append:s}\n"
        else:
            ret += f"pair_style      deepspin {model:s}\n"
        ret += "pair_coeff * *\n"

    # var_xxx allow xxx to change with time.
    if step == "both":
        var_lattice_spring = True
        var_spin_spring = True
    elif step == "deep_on":
        var_lattice_spring = False
        var_spin_spring = False
    elif step == "spring_off":
        var_lattice_spring = True
        var_spin_spring = spring_lambda_mode == "joint"
    elif step == "lattice_spring_off":
        var_lattice_spring = True
        var_spin_spring = False
    elif step == "spin_spring_off":
        var_lattice_spring = False
        var_spin_spring = True
    else:
        raise RuntimeError("unknown step", step)

    enable_lattice_spring = spring_lambda_mode != "spin_only"
    enable_spin_spring = spring_lambda_mode != "lattice_only"
    if step == "spin_spring_off":
        enable_lattice_spring = False
        enable_spin_spring = True
    if step == "lattice_spring_off":
        enable_lattice_spring = True
        enable_spin_spring = False

    if step == "both" or step == "deep_on":
        var_deep = True
    elif step == "spring_off" or step == "lattice_spring_off" or step == "spin_spring_off":
        var_deep = False
    else:
        raise RuntimeError("unknown step", step)
    # 1. joint spring_off lattice_flag = 1; spin_flag = 1
    #   var_lattice = True; var_spin = True;
    #   enable_lattice = True; enable_spin = True;
    # 2. seperate spring_off lattice_flag = 1; spin_flag = 1
    #   var_lattice = True; var_spin = False;
    #   enable_lattice = True; enable_spin = True;
    #   seperate spin_spring_off 
    #   var_lattice = False; var_spin = True;
    #   enable_lattice = False; enable_spin = True;
    # 3. only lattice_spring_off lattice_flag = 1; spin_flag = 0
    #  var_lattice = True; var_spin = False;
    #  enable_lattice = True; enable_spin = False;
    # 4. only spin_spring_off lattice_flag = 0; spin_flag = 1
    #  var_lattice = False; var_spin = True;
    #  enable_lattice = False; enable_spin = True;
    ret += _ff_spring(lamb, m_spring_k, var_lattice_spring, enabled=enable_lattice_spring)
    if spin_reference == "vector":
        ret += _ff_spring_spin(
            lamb,
            m_spring_spin_k,
            var_spin_spring,
            enabled=enable_spin_spring,
        )
    elif spin_reference == "modulus":
        if spin_ref_s0 is None or spin_map is None:
            raise RuntimeError("spin_reference='modulus' requires spin_ref_s0 and spin_map")
        spin_mod_mode = "off" if var_spin_spring else "full"
        ret += _ff_spin_mod(
            lamb,
            m_spring_spin_k,
            spin_ref_s0,
            spin_map,
            spin_mod_mode,
            enabled=enable_spin_spring,
        )
    elif spin_reference == "none":
        ret += "variable        l_spring_spin equal 0.0\n"
    else:
        raise RuntimeError("unknown spin_reference", spin_reference)

    if if_harmonic:
        # target lattice springs (k2): scale by lambda in deep_on, full strength otherwise
        ntypes = len(m_target_spring_k)
        for ii in range(ntypes):
            k2_const = m_target_spring_k[ii] * lamb if var_deep else m_target_spring_k[ii]
            ret += f"fix             l_k2_spring_{ii+1} type_{ii+1} spring/self {k2_const:.10e}\n"
            ret += f"fix_modify      l_k2_spring_{ii+1} energy yes\n"
        # target spin springs (k2_spin): scale by lambda in deep_on, full strength otherwise
        for ii in range(ntypes):
            k2_spin_const = m_target_spring_spin_k[ii] * lamb if var_deep else m_target_spring_spin_k[ii]
            ret += f"fix             l_k2_spring_spin_{ii+1} type_{ii+1} spring/spin {k2_spin_const:.10e}\n"
            ret += f"fix_modify      l_k2_spring_spin_{ii+1} energy yes\n"
        sum_terms = [
            f"f_l_k2_spring_{ii+1}+f_l_k2_spring_spin_{ii+1}" for ii in range(ntypes)
        ]
        # on-site lattice-spin coupling: part of the *target* system Hamiltonian.
        # scaled by lambda in deep_on (so its switch-on work is captured by the
        # deep_on integrand <e_k2_spring>/lambda), full strength during spring_off.
        # E_couple = c * sum_i (dr_i . ds_i); frozen-DOF modes give dr.ds=0 -> no effect.
        if couple_c:
            c_const = couple_c * lamb if var_deep else couple_c
            ret += f"fix             l_couple all couple/spin/lattice {c_const:.10e}\n"
            ret += "fix_modify      l_couple energy yes\n"
            sum_terms.append("f_l_couple")
        # external Zeeman field (eV/mu_B units), part of the *target* Hamiltonian.
        # scaled by lambda in deep_on (switch-on work captured by integrand), full otherwise.
        if zeeman_B:
            b_const = zeeman_B * lamb if var_deep else zeeman_B
            nx, ny, nz = zeeman_dir
            ret += f"fix             l_zeeman all zeeman/ev {b_const:.10e} {nx:.8f} {ny:.8f} {nz:.8f}\n"
            ret += "fix_modify      l_zeeman energy yes\n"
            sum_terms.append("f_l_zeeman")
        # magneto-volume coupling U_mv = lambda_mv*(V-V0)/V0 * sum_i(|s_i|^2 - s0^2),
        # part of the *target* Hamiltonian (spin-dependent even at fixed V). lambda-scaled
        # in deep_on (switch-on work captured by <e_k2_spring>/lambda), full in spring_off.
        # Note: the pure-volume EOS term U_eos(V) is a constant at fixed V (NVT HTI) and is
        # added analytically by the driver, so it is NOT injected here.
        if magvol_lambda:
            mv_const = magvol_lambda * lamb if var_deep else magvol_lambda
            ret += f"fix             l_magvol all magvol {mv_const:.10e} {magvol_v0:.10e} {magvol_s0:.10e}\n"
            ret += "fix_modify      l_magvol energy yes\n"
            sum_terms.append("f_l_magvol")
        sum_str = "+".join(sum_terms)
        ret += f"variable        e_k2_spring equal {sum_str}\n"
    else:
        if couple_c or magvol_lambda:
            raise RuntimeError(
                "couple_c/magvol are analytic-toy target terms: only supported "
                "with harmonic=true (their switch-on work is summed into "
                "e_k2_spring); with a real model they would be silently "
                "dropped from the deep_on integrand"
            )
        if var_deep:
            if if_meam:
                ret += "fix             l_deep all adapt 1 pair meam scale * * v_LAMBDA\n"
            else:
                ret += "fix             l_deep all adapt 1 pair deepspin scale * * v_LAMBDA\n"
        ret += "compute         e_deep all pe pair\n"
        # external Zeeman field (eV/mu_B) on the *target* Hamiltonian with a
        # real model: lambda-scaled in deep_on so its switch-on work enters
        # the integrand <(c_e_deep + f_l_zeeman)>/lambda, full strength during
        # spring_off (constant term, not part of dH/dlambda there).
        if zeeman_B:
            b_const = zeeman_B * lamb if var_deep else zeeman_B
            nx, ny, nz = zeeman_dir
            ret += f"fix             l_zeeman all zeeman/ev {b_const:.10e} {nx:.8f} {ny:.8f} {nz:.8f}\n"
            ret += "fix_modify      l_zeeman energy yes\n"
            ret += "variable        e_deep_tot equal c_e_deep+f_l_zeeman\n"
    ret += "compute         spin all property/atom sp spx spy spz fmx fmy fmz\n"
    return ret


def _gen_lammps_input(
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
    spin_ref_s0=None,
    spin_map=None,
    spin_reference="vector",
    pres=1.0,
    tau_t=0.1,
    tau_p=0.5,
    thermo_freq=100,
    dump_freq=100,
    copies=None,
    crystal="vega",
    sparam={},
    switch="one-step",
    step="both",
    if_meam=False,
    meam_model=None,
    custom_variables=None,
    append=None,
    spring_lambda_mode="joint",
    lattice_flag=1,
    spin_flag=1,
    if_harmonic=False,
    m_target_spring_k=None,
    m_target_spring_spin_k=None,
    couple_c=0.0,
    zeeman_B=0.0,
    zeeman_dir=(0.0, 0.0, 1.0),
    magvol_lambda=0.0,
    magvol_v0=0.0,
    magvol_s0=0.0,
):
    ret = ""
    ret += "clear\n"
    ret += "# --------------------- VARIABLES-------------------------\n"
    ret += "variable        NSTEPS          equal %d\n" % nsteps
    ret += "variable        THERMO_FREQ     equal %d\n" % thermo_freq
    ret += "variable        DUMP_FREQ       equal %d\n" % dump_freq
    ret += f"variable        SP_MASS            equal {spin_mass:f}\n"
    ret += f"variable        TEMP            equal {temp:f}\n"
    ret += f"variable        PRES            equal {pres:f}\n"
    ret += f"variable        TAU_T           equal {tau_t:f}\n"
    ret += f"variable        TAU_P           equal {tau_p:f}\n"
    ret += f"variable        LAMBDA          equal {lamb:.10e}\n"
    ret += "variable        INV_LAMBDA      equal %.10e\n" % (1 - lamb)
    if custom_variables is not None:
        for key, value in custom_variables.items():
            ret += f"variable {key} equal {value}\n"
    ret += "# ---------------------- INITIALIZAITION ------------------\n"
    ret += "units           metal\n"
    ret += "boundary        p p p\n"
    ret += "atom_style      spin\n"
    ret += "atom_modify     map yes\n"
    ret += "# --------------------- ATOM DEFINITION ------------------\n"
    ret += "box             tilt large\n"
    ret += f"read_data       {conf_file}\n"
    if copies is not None:
        ret += "replicate       %d %d %d\n" % (copies[0], copies[1], copies[2])
    ret += "change_box      all triclinic\n"
    for jj in range(len(mass_map)):
        ret += "mass            %d %f\n" % (jj + 1, mass_map[jj])

    # force field setting
    if switch == "one-step" or switch == "two-step":
        ret += _ff_two_steps(
            lamb,
            model,
            m_spring_k,
            m_spring_spin_k,
            step,
            spin_ref_s0=spin_ref_s0,
            spin_map=spin_map,
            spin_reference=spin_reference,
            append=append,
            if_meam=if_meam,
            meam_model=meam_model,
            spring_lambda_mode=spring_lambda_mode,
            if_harmonic=if_harmonic,
            m_target_spring_k=m_target_spring_k,
            m_target_spring_spin_k=m_target_spring_spin_k,
            couple_c=couple_c,
            zeeman_B=zeeman_B,
            zeeman_dir=zeeman_dir,
            magvol_lambda=magvol_lambda,
            magvol_v0=magvol_v0,
            magvol_s0=magvol_s0,
        )
    elif switch == "three-step":
        ret += _ff_soft_lj(
            lamb,
            model,
            m_spring_k,
            step,
            sparam,
            if_meam=if_meam,
            meam_model=meam_model,
            append=append,
        )
    else:
        raise RuntimeError("unknow switch", switch)

    ret += "# --------------------- MD SETTINGS ----------------------\n"
    ret += "neighbor        1.0 bin\n"
    ret += f"timestep        {timestep}\n"
    ret += "thermo          ${THERMO_FREQ}\n"
    ret += "compute         allmsd all msd\n"
    ret += "compute         spinmsd all msd/spin\n"
    spin_msd_col = "c_spinmsd[*]"
    if spin_reference == "modulus":
        ret += "compute         spinmodmsd all msd/spin/mod\n"
        spin_msd_col += " c_spinmodmsd"
    if if_harmonic:
        e_deep_col = "v_e_k2_spring"
    elif zeeman_B:
        e_deep_col = "v_e_deep_tot"
    else:
        e_deep_col = "c_e_deep"
    if 1 - lamb != 0:
        if not isinstance(m_spring_k, list):
            if switch == "three-step":
                ret += f"thermo_style    custom step ke pe etotal enthalpy temp press vol f_l_spring c_e_diff[1] f_l_spring_spin c_allmsd[*] {spin_msd_col}\n"
            else:
                ret += f"thermo_style    custom step ke pe etotal enthalpy temp press vol f_l_spring {e_deep_col} f_l_spring_spin c_allmsd[*] {spin_msd_col}\n"
        else:
            if switch == "three-step":
                ret += f"thermo_style    custom step ke pe etotal enthalpy temp press vol v_l_spring c_e_diff[1] v_l_spring_spin c_allmsd[*] {spin_msd_col}\n"
            else:
                ret += f"thermo_style    custom step ke pe etotal enthalpy temp press vol v_l_spring {e_deep_col} v_l_spring_spin c_allmsd[*] {spin_msd_col}\n"
    else:
        if switch == "three-step":
            ret += f"thermo_style    custom step ke pe etotal enthalpy temp press vol c_e_diff[1] c_e_diff[1] c_allmsd[*] {spin_msd_col}\n"
        else:
            ret += f"thermo_style    custom step ke pe etotal enthalpy temp press vol {e_deep_col} {e_deep_col} c_allmsd[*] {spin_msd_col}\n"
    ret += "thermo_modify   format 9 %.16e\n"
    ret += "thermo_modify   format 10 %.16e\n"
    ret += "thermo_modify   format 11 %.16e\n"
    ret += "dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]\n"
    if ens == "nvt":
        ret += "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T} mass ${SP_MASS} rand %d\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    elif ens == "nvt-langevin":
        ret += (
            "fix             1 all nve/spin lattice_flag %d spin_flag %d\n"
            % (lattice_flag, spin_flag)
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
            ret += "fix             3 all langevin/spin ${TEMP} ${TEMP} ${TAU_T} %d\n" % (
                np.random.default_rng().integers(1, 2**16)
            )
    elif ens == "npt-iso" or ens == "npt":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P} mass ${SP_MASS} rand %d\n"% (
            np.random.default_rng().integers(1, 2**16)
        )
    elif ens == "nve":
        ret += "fix             1 all nve\n"
    else:
        raise RuntimeError(f"unknow ensemble {ens}\n")

    ret += "# --------------------- INITIALIZE -----------------------\n"
    # 其实这里不用去额外关注. 因为 在nve中已经constrain了. 
    if lattice_flag and spin_flag:
        ret += "velocity        all create ${TEMP} %d spin yes spmass ${SP_MASS}\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    elif lattice_flag:
        ret += "velocity        all create ${TEMP} %d\n" % (
            np.random.default_rng().integers(1, 2**16)
        )
    elif spin_flag:
        # spin_only (lattice frozen): the spins still need an initial thermal velocity --
        # langevin/spin alone does not kick them off zero, so without this the spins stay
        # cold and the spin-spring switching work comes out ~0.
        ret += "velocity        all create ${TEMP} %d spin yes spmass ${SP_MASS}\n" % (
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
        raise RuntimeError("unknow crystal " + crystal)
    ret += "# --------------------- RUN ------------------------------\n"
    ret += "run             ${NSTEPS}\n"
    ret += "write_data      out.lmp\n"
    return ret



def make_tasks(iter_name, jdata, ref="einstein", switch="one-step", if_meam=None):
    if if_meam is None:
        if_meam = jdata.get("if_meam", False)
    if_harmonic = jdata.get("harmonic", False)
    equi_conf = os.path.abspath(jdata["equi_conf"])
    meam_model = jdata.get("meam_model", None)
    if not if_harmonic:
        model = os.path.abspath(jdata["model"])

    if if_meam is None:
        if_meam = jdata.get("if_meam", None)
    spring_lambda_mode = _get_spring_lambda_mode(jdata)

    if switch == "one-step":
        subtask_name = iter_name
        _make_tasks(
            subtask_name,
            jdata,
            ref,
            step="both",
            if_meam=if_meam,
            meam_model=meam_model,
            if_harmonic=if_harmonic,
        )
        if if_meam:
            relative_link_file(meam_model["library"], iter_name)
            relative_link_file(meam_model["potential"], iter_name)
        else:
            pass
    elif switch == "two-step" or switch == "three-step":
        job_abs_dir = create_path(iter_name)
        copied_conf = os.path.join(os.path.abspath(iter_name), "conf.lmp")
        shutil.copyfile(equi_conf, copied_conf)
        jdata["equi_conf"] = "conf.lmp"

        if if_meam:
            relative_link_file(meam_model["library"], job_abs_dir)
            relative_link_file(meam_model["potential"], job_abs_dir)

        if not if_harmonic:
            model_name = os.path.basename(model)
            linked_model = os.path.join(os.path.abspath(iter_name), model_name)
            shutil.copyfile(model, linked_model)
            jdata["model"] = model_name

        cwd = os.getcwd()
        os.chdir(iter_name)
        with open("in.json", "w") as fp:
            json.dump(jdata, fp, indent=4)
        if switch == "two-step":
            subtask_name = "00.deep_on"
            _make_tasks(
                subtask_name,
                jdata,
                ref,
                switch=switch,
                step="deep_on",
                link=True,
                if_meam=if_meam,
                meam_model=meam_model,
                if_harmonic=if_harmonic,
            )
            if spring_lambda_mode == "joint":
                subtask_name = "01.spring_off"
                _make_tasks(
                    subtask_name,
                    jdata,
                    ref,
                    switch=switch,
                    step="spring_off",
                    link=True,
                    if_meam=if_meam,
                    meam_model=meam_model,
                    if_harmonic=if_harmonic,
                )
            elif spring_lambda_mode == "split":
                subtask_name = "01.spring_off"
                _make_tasks(
                    subtask_name,
                    jdata,
                    ref,
                    switch=switch,
                    step="spring_off",
                    link=True,
                    if_meam=if_meam,
                    meam_model=meam_model,
                    if_harmonic=if_harmonic,
                )
                subtask_name = "02.spin_spring_off"
                _make_tasks(
                    subtask_name,
                    jdata,
                    ref,
                    switch=switch,
                    step="spin_spring_off",
                    link=True,
                    if_meam=if_meam,
                    meam_model=meam_model,
                    if_harmonic=if_harmonic,
                )
            elif spring_lambda_mode == "lattice_only":
                subtask_name = "01.lattice_spring_off"
                _make_tasks(
                    subtask_name,
                    jdata,
                    ref,
                    switch=switch,
                    step="lattice_spring_off",
                    link=True,
                    if_meam=if_meam,
                    meam_model=meam_model,
                    if_harmonic=if_harmonic,
                )
            elif spring_lambda_mode == "spin_only":
                subtask_name = "01.spin_spring_off"
                _make_tasks(
                    subtask_name,
                    jdata,
                    ref,
                    switch=switch,
                    step="spin_spring_off",
                    link=True,
                    if_meam=if_meam,
                    meam_model=meam_model,
                    if_harmonic=if_harmonic,
                )
            else:
                raise RuntimeError("unknown spring_lambda_mode", spring_lambda_mode)
        else:
            raise RuntimeError("unknow switch", switch)
        os.chdir(cwd)
    else:
        raise RuntimeError("unknow switch", switch)


def _make_tasks(
    iter_name,
    jdata,
    ref,
    switch="one-step",
    step="both",
    link=False,
    if_meam=False,
    meam_model=None,
    if_harmonic=False,
):
    if "crystal" not in jdata:
        print("do not find crystal in jdata, assume vega")
        jdata["crystal"] = "vega"

    crystal = jdata["crystal"]
    spring_lambda_mode = _get_spring_lambda_mode(jdata)
    protect_eps = jdata["protect_eps"]

    if switch == "one-step":
        if (
            jdata.get("lambda", None) is None
            and jdata.get("lambda_deep_on", None) is not None
        ):
            if jdata.get("lambda_lj_on", None) is not None:
                raise RuntimeError(
                    "It seems that you are using a json file for the three-step hti calculation. If that is the case, please set the correct switch option by using '-s three-step'. If you do want to use one-step switching, please use 'lambda' instead of 'lambda_lj_on', 'lambda_deep_on', and 'lambda_spring_off' in your json file. Check 'dpti hti gen -h' for more information."
                )
            elif jdata.get("lambda_lj_on", None) is None:
                raise RuntimeError(
                    "It seems that you are using a json file for the two-step hti calculation. If that is the case, please set the correct switch option by using '-s two-step'. If you do want to use one-step switching, please use 'lambda' instead of 'lambda_deep_on' and 'lambda_spring_off' in your json file. Check 'dpti hti gen -h' for more information."
                )
        all_lambda = parse_seq(jdata["lambda"])
    elif switch == "two-step" or switch == "three-step":
        if step == "deep_on":
            all_lambda = parse_seq(jdata["lambda_deep_on"])
        elif step == "spring_off":
            all_lambda = parse_seq(jdata["lambda_spring_off"])
        elif step == "lattice_spring_off":
            all_lambda = parse_seq(jdata["lambda_spring_off"])
        elif step == "spin_spring_off":
            all_lambda = parse_seq(
                get_first_matched_key_from_dict(
                    jdata,
                    ["lambda_spin_spring_off", "lambda_spring_off"],
                )
            )
        elif step == "lj_on":
            all_lambda = parse_seq(jdata["lambda_lj_on"])
        else:
            raise RuntimeError("unknown step", step)

    if all_lambda[0] == 0:
        all_lambda[0] += protect_eps
    if all_lambda[-1] == 1:
        all_lambda[-1] -= protect_eps

    equi_conf = jdata["equi_conf"]
    equi_conf = os.path.abspath(equi_conf)
    if not if_harmonic:
        model = jdata["model"]
        model = os.path.abspath(model)
    else:
        model = None
    # mass_map = jdata['mass_map']
    mass_map = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    spin_mass = get_first_matched_key_from_dict(jdata, ["spin_mass_map", "sp_mass_map", "spin_mass"])
    spin_reference = _get_spin_reference_mode(jdata)
    jdata["spin_reference"] = spin_reference
    spin_map = None
    spin_ref_s0 = None
    if spin_reference in ("vector", "modulus"):
        spin_map = get_first_matched_key_from_dict(jdata, ["spin_map", "sp_map"])

    nsteps = jdata["nsteps"]
    # timestep = jdata['timestep']
    timestep = get_first_matched_key_from_dict(jdata, ["timestep", "dt"])
    spring_k = jdata["spring_k"]
    spring_spin_k = None
    if spin_reference == "vector":
        spring_spin_k = get_first_matched_key_from_dict(
            jdata, ["spring_spin_k", "spin_spring_k", "s_spring_k", "spring_k_spin"]
        )
    custom_variables = jdata.get("custom_variables", None)
    append = jdata.get("append", None)

    if crystal == "frenkel":
        m_spring_k = []
        m_spring_spin_k = []
        # spring_k scalar -> shared Einstein frequency (k_i = spring_k * m_i);
        # spring_k list   -> per-type spring constants k_i [eV/A^2] taken as given.
        # einstein.frenkel() carries the matching generalised centre-of-mass correction.
        if isinstance(spring_k, list):
            assert len(spring_k) == len(mass_map)
            m_spring_k = list(spring_k)
        else:
            for ii in mass_map:
                m_spring_k.append(spring_k * ii)
        if spin_reference == "vector":
            m_spring_spin_k = [
                spring_spin_k * mass * spin_mass if spin_map[ii] else 0.0
                for ii, mass in enumerate(mass_map)
            ]
        elif spin_reference == "modulus":
            _, m_spring_spin_k, spin_ref_s0 = _get_spin_ref_params(
                jdata, mass_map, spin_mass, spin_map
            )
        else:
            m_spring_spin_k = [0.0 for _ in mass_map]
    if crystal == "vega":
        m_spring_k = []
        for ii in mass_map:
            m_spring_k.append(spring_k * ii)
        if spin_reference == "modulus":
            _, m_spring_spin_k, spin_ref_s0 = _get_spin_ref_params(
                jdata, mass_map, spin_mass, spin_map
            )
        else:
            m_spring_spin_k = [0.0 for _ in mass_map]
    m_target_spring_k = None
    m_target_spring_spin_k = None
    couple_c = jdata.get("couple_c", 0.0)
    zeeman_B = jdata.get("zeeman_B", 0.0)
    zeeman_dir = jdata.get("zeeman_dir", (0.0, 0.0, 1.0))
    magvol_lambda = jdata.get("magvol_lambda", 0.0)
    magvol_v0 = jdata.get("magvol_v0", 0.0)
    magvol_s0 = jdata.get("magvol_s0", 0.0)
    if if_harmonic:
        target_spring_k = jdata["target_spring_k"]
        target_spring_spin_k = jdata["target_spring_spin_k"]
        m_target_spring_k = [target_spring_k * ii for ii in mass_map]
        m_target_spring_spin_k = [target_spring_spin_k * ii * spin_mass for ii in mass_map]
    # thermo_freq = jdata['thermo_freq']
    thermo_freq = get_first_matched_key_from_dict(jdata, ["thermo_freq", "stat_freq"])
    dump_freq = get_first_matched_key_from_dict(
        jdata, ["dump_freq", "thermo_freq", "stat_freq"]
    )
    copies = None
    if "copies" in jdata:
        copies = jdata["copies"]
    temp = jdata["temp"]
    jdata["reference"] = ref
    jdata["switch"] = switch
    jdata["step"] = step

    create_path(iter_name)
    copied_conf = os.path.join(os.path.abspath(iter_name), "conf.lmp")
    if not link:
        shutil.copyfile(equi_conf, copied_conf)
    else:
        cwd = os.getcwd()
        os.chdir(iter_name)
        os.symlink(os.path.relpath(equi_conf), "conf.lmp")
        os.chdir(cwd)
    jdata["equi_conf"] = "conf.lmp"
    if not if_harmonic:
        model_name = os.path.basename(model)
        linked_model = os.path.join(os.path.abspath(iter_name), model_name)
        if not link:
            shutil.copyfile(model, linked_model)
        else:
            cwd = os.getcwd()
            os.chdir(iter_name)
            os.symlink(os.path.relpath(model), model_name)
            os.chdir(cwd)
        jdata["model"] = model_name
    else:
        model_name = None
        linked_model = None
    langevin = jdata.get("langevin", True)

    cwd = os.getcwd()
    os.chdir(iter_name)
    with open("in.json", "w") as fp:
        json.dump(jdata, fp, indent=4)
    os.chdir(cwd)

    for idx, ii in enumerate(all_lambda):
        work_path = os.path.join(iter_name, "task.%06d" % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.relpath(copied_conf), "conf.lmp")
        if not if_harmonic:
            os.symlink(os.path.relpath(linked_model), model_name)
        if if_meam:
            meam_library_basename = os.path.basename(meam_model["library"])
            meam_potential_basename = os.path.basename(meam_model["potential"])
            relative_link_file(os.path.join("../../", meam_library_basename), "./")
            relative_link_file(os.path.join("../../", meam_potential_basename), "./")
        ens = None
        if jdata.get("ens", False):
            ens = jdata.get("ens")
        if ens is not None and ens != "nvt" and ens != "nvt-langevin":
            raise RuntimeError(
                f"Unknow ensemble '{ens}': one should use the NVT ensemble in the HTI step. The only supported values for the 'ens' keyword are 'nvt' and 'nvt-langevin'."
            )
        if idx == 0:
            ens = "nvt-langevin"
        else:
            ens = "nvt"
        if langevin:
            ens = "nvt-langevin"

        lattice_flag = 1
        spin_flag = 1
        if spring_lambda_mode == "lattice_only":
            spin_flag = 0
        elif spring_lambda_mode == "spin_only":
            lattice_flag = 0

        if ref == "einstein":
            lmp_str = _gen_lammps_input(
                "conf.lmp",
                mass_map,
                spin_mass,
                ii,
                model_name,
                m_spring_k,
                m_spring_spin_k,
                nsteps,
                timestep,
                ens,
                temp,
                spin_ref_s0=spin_ref_s0,
                spin_map=spin_map,
                spin_reference=spin_reference,
                thermo_freq=thermo_freq,
                dump_freq=dump_freq,
                copies=copies,
                switch=switch,
                step=step,
                sparam={},
                crystal=crystal,
                if_meam=if_meam,
                meam_model=meam_model,
                custom_variables=custom_variables,
                append=append,
                spring_lambda_mode=spring_lambda_mode,
                lattice_flag=lattice_flag,
                spin_flag=spin_flag,
                if_harmonic=if_harmonic,
                m_target_spring_k=m_target_spring_k,
                m_target_spring_spin_k=m_target_spring_spin_k,
                couple_c=couple_c,
                zeeman_B=zeeman_B,
                zeeman_dir=zeeman_dir,
                magvol_lambda=magvol_lambda,
                magvol_v0=magvol_v0,
                magvol_s0=magvol_s0,
            )
        elif ref == "ideal":
            raise RuntimeError("choose hti_liq.py")
            # lmp_str \
            #     = _gen_lammps_input_ideal('conf.lmp',
            #                               model_mass_map,
            #                               ii,
            #                               'graph.pb',
            #                               nsteps,
            #                               dt,
            #                               ens,
            #                               temp,
            #                               prt_freq = stat_freq,
            #                               copies = copies,
            #                               if_meam = if_meam,
            #                               meam_model = meam_model)
        else:
            raise RuntimeError("unknow reference system type " + ref)
        with open("in.lammps", "w") as fp:
            fp.write(lmp_str)
        with open("lambda.out", "w") as fp:
            fp.write(str(ii))
        os.chdir(cwd)


def _compute_thermo(fname, natoms, stat_skip, stat_bsize):
    data = get_thermo(fname)
    ea, ee = block_avg(data[:, 3], skip=stat_skip, block_size=stat_bsize)
    ha, he = block_avg(data[:, 4], skip=stat_skip, block_size=stat_bsize)
    ta, te = block_avg(data[:, 5], skip=stat_skip, block_size=stat_bsize)
    pa, pe = block_avg(data[:, 6], skip=stat_skip, block_size=stat_bsize)
    va, ve = block_avg(data[:, 7], skip=stat_skip, block_size=stat_bsize)
    thermo_info = {}
    thermo_info["p"] = pa
    thermo_info["p_err"] = pe
    thermo_info["v"] = va / natoms
    thermo_info["v_err"] = ve / natoms
    thermo_info["e"] = ea / natoms
    thermo_info["e_err"] = ee / natoms
    thermo_info["h"] = ha / natoms
    thermo_info["h_err"] = he / natoms
    thermo_info["t"] = ta
    thermo_info["t_err"] = te
    unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
    thermo_info["pv"] = pa * va * unit_cvt / natoms
    thermo_info["pv_err"] = pe * va * unit_cvt / natoms
    return thermo_info


def post_tasks(iter_name, jdata, natoms=None, method="inte", scheme="s"):
    switch = "one-step"
    if os.path.isdir(os.path.join(iter_name, "00.deep_on")):
        switch = "two-step"
    if os.path.isdir(os.path.join(iter_name, "00.lj_on")):
        switch = "three-step"

    spring_lambda_mode = _get_spring_lambda_mode(jdata)

    if switch == "two-step":
        subtask_name = os.path.join(iter_name, "00.deep_on")
        if method == "inte":
            e0, err0, tinfo0 = _post_tasks(
                subtask_name,
                jdata,
                natoms=natoms,
                scheme=scheme,
                switch=switch,
                step="deep_on",
            )
        elif method == "mbar":
            e0, err0, tinfo0 = _post_tasks_mbar(
                subtask_name, jdata, natoms=natoms, switch=switch, step="deep_on"
            )
        else:
            raise RuntimeError("unknow method for integration")
        print(f"# fe of deep_on:    {e0:20.12f}  {err0[0]:10.3e} {err0[1]:10.3e}")
        sub_steps = []
        if spring_lambda_mode in ["joint", "split"]:
            sub_steps.append(("spring_off", os.path.join(iter_name, "01.spring_off")))
        elif spring_lambda_mode == "lattice_only":
            sub_steps.append(
                ("lattice_spring_off", os.path.join(iter_name, "01.lattice_spring_off"))
            )
        elif spring_lambda_mode == "spin_only":
            sub_steps.append(
                ("spin_spring_off", os.path.join(iter_name, "01.spin_spring_off"))
            )
        else:
            raise RuntimeError("unknown spring_lambda_mode", spring_lambda_mode)

        if spring_lambda_mode == "split":
            sub_steps.append(
                ("spin_spring_off", os.path.join(iter_name, "02.spin_spring_off"))
            )

        de = e0
        stt_err2 = np.square(err0[0])
        sys_err = err0[1]
        tinfo = tinfo0

        for sub_step, subtask_name in sub_steps:
            if method == "inte":
                ei, erri, tinfo = _post_tasks(
                    subtask_name,
                    jdata,
                    natoms=natoms,
                    scheme=scheme,
                    switch=switch,
                    step=sub_step,
                )
            elif method == "mbar":
                ei, erri, tinfo = _post_tasks_mbar(
                    subtask_name,
                    jdata,
                    natoms=natoms,
                    switch=switch,
                    step=sub_step,
                )
            else:
                raise RuntimeError("unknow method for integration")
            print(f"# fe of {sub_step}: {ei:20.12f}  {erri[0]:10.3e} {erri[1]:10.3e}")
            de += ei
            stt_err2 += np.square(erri[0])
            sys_err += erri[1]
        err = [np.sqrt(stt_err2), sys_err]
    else:
        if method == "inte":
            de, err, tinfo = _post_tasks(iter_name, jdata, natoms=natoms, scheme=scheme)
        elif method == "mbar":
            de, err, tinfo = _post_tasks_mbar(iter_name, jdata, natoms=natoms)
    return de, err, tinfo


def _post_tasks(
    iter_name, jdata, natoms=None, scheme="s", switch="one-step", step="both"
):
    stat_skip = jdata["stat_skip"]
    stat_bsize = jdata["stat_bsize"]
    all_tasks = glob.glob(os.path.join(iter_name, "task.[0-9]*"))
    all_tasks.sort()
    ntasks = len(all_tasks)
    equi_conf = get_task_file_abspath(iter_name, jdata["equi_conf"])
    assert os.path.isfile(equi_conf)
    if natoms is None:
        natoms = get_natoms(equi_conf)
        # nspins = get_nspins(equi_conf, natoms)
        if "copies" in jdata:
            natoms *= np.prod(jdata["copies"])
            # nspins *= np.prod(jdata["copies"])

    all_lambda = []
    all_es = []
    all_es_err = []
    all_esp = []
    all_esp_err = []
    all_ed = []
    all_ed_err = []

    all_etot = []
    all_etot_err = []
    all_enthalpy = []
    all_msd_xyz = []
    all_msd_spin = []
    spin_reference = _get_spin_reference_mode(jdata)

    for ii in all_tasks:
        log_name = os.path.join(ii, "log.lammps")
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, "data"), data, fmt="%.6e")
        sa, se = block_avg(data[:, 8], skip=stat_skip, block_size=stat_bsize)
        da, de = block_avg(data[:, 9], skip=stat_skip, block_size=stat_bsize)
        spa, spe = block_avg(data[:, 10], skip=stat_skip, block_size=stat_bsize)
        etot, etot_err = block_avg(data[:, 3], skip=stat_skip, block_size=stat_bsize)
        enthalpy, _ = block_avg(data[:, 4], skip=stat_skip, block_size=stat_bsize)
        msd_xyz = data[-1, 14]   # c_allmsd[4]
        if spin_reference == "modulus":
            msd_spin = data[-1, 19]  # c_spinmodmsd
        else:
            msd_spin = data[-1, 18]  # c_spinmsd[4]
        sa /= natoms
        se /= natoms
        spa /= natoms
        spe /= natoms
        da /= natoms
        de /= natoms
        lmda_name = os.path.join(ii, "lambda.out")
        ll = float(open(lmda_name).read())
        all_lambda.append(ll)
        all_es.append(sa)
        all_esp.append(spa)
        all_ed.append(da)
        all_es_err.append(se)
        all_esp_err.append(spe)
        all_ed_err.append(de)

        all_etot.append(etot / natoms)
        all_etot_err.append(etot_err)
        all_enthalpy.append(enthalpy)
        all_msd_xyz.append(msd_xyz)
        all_msd_spin.append(msd_spin)

    all_lambda = np.array(all_lambda)
    all_es = np.array(all_es)
    all_esp = np.array(all_esp)
    all_ed = np.array(all_ed)
    all_es_err = np.array(all_es_err)
    all_esp_err = np.array(all_esp_err)
    all_ed_err = np.array(all_ed_err)
    spring_lambda_mode = _get_spring_lambda_mode(jdata)
    if switch == "one-step" or switch == "two-step":
        if step == "both":
            de = all_ed / all_lambda - (all_es + all_esp) / (1 - all_lambda)
            all_err = np.sqrt(
                np.square(all_ed_err / all_lambda)
                + np.square(all_es_err / (1 - all_lambda))
                + np.square(all_esp_err / (1 - all_lambda))
            )
        elif step == "deep_on":
            de = all_ed / all_lambda
            all_err = all_ed_err / all_lambda
        elif step == "spring_off":
            if spring_lambda_mode == "split":
                de = -all_es / (1 - all_lambda)
                all_err = all_es_err / (1 - all_lambda)
            else:
                de = -(all_es + all_esp) / (1 - all_lambda)
                all_err = np.sqrt(
                    np.square(all_es_err / (1 - all_lambda))
                    + np.square(all_esp_err / (1 - all_lambda))
                )
        elif step == "lattice_spring_off":
            de = -all_es / (1 - all_lambda)
            all_err = all_es_err / (1 - all_lambda)
        elif step == "spin_spring_off":
            de = -all_esp / (1 - all_lambda)
            all_err = all_esp_err / (1 - all_lambda)
        else:
            raise RuntimeError("unknow step", step)
    # elif switch == "three-step": # 由于lj这块有问题 暂时不考虑three-step. 
    #     if step == "lj_on" or step == "deep_on":
    #         de = all_ed # 这里跟之前的不一样是因为在lammps in文件中的thermo形式与之前不同. 
    #         all_err = all_ed_err
    #     elif step == "spring_off":
    #         de = -all_es / (1 - all_lambda) + all_ed
    #         all_err = np.sqrt(
    #             np.square(all_es_err / (1 - all_lambda)) + np.square(all_ed_err)
    #         )
    #     else:
    #         raise RuntimeError("unknow step", step)
    else:
        raise RuntimeError("unknow switch", switch)
    # 在分步热力学积分中，系统的总势能 $U(\lambda)$ 通常这样定义：
    # 1. DeepMD On 阶段 ($\lambda: 0 \to 1$)：$$U(\lambda) = \lambda U_{DeepMD} + U_{spring}$$$\lambda = 0$ 
    # 只有弹簧势，DeepMD 势能为 0。$\lambda = 1$：DeepMD 势能全开，弹簧势也全开（即“相互作用的爱因斯坦晶体”）。
    # 被积函数：$\frac{\partial U}{\partial \lambda} = U_{DeepMD}$。 ==> $U_{DeepMD} = U(output) / \lambda $
    # 这就是为什么代码里写 de = all_ed / all_lambda。
    # (这里得到的自由能差为: A_Ein-sol - A_Ein-id)
    # 2. Spring Off 阶段 ($\lambda: 0 \to 1$) 注意，这里的 $\lambda$ 是积分变量。
    # 为了让弹簧从“全开”变为“关闭”，势能通常定义为：$$U(\lambda) = U_{DeepMD} + (1 - \lambda) U_{spring}$$$\lambda = 0$
    # 弹簧项系数为 1，全开。$\lambda = 1$：弹簧项系数为 0，关闭。被积函数：$\frac{\partial U}{\partial \lambda} = -U_{spring}$。 ==> $U_{spring} = -U(output) / (1 - \lambda) $
    # 这就是为什么代码里带有一个负号：de = -all_es / (1 - all_lambda)。
    all_print = []
    # all_print.append(np.arange(len(all_lambda)))
    all_print.append(all_lambda)
    all_print.append(de)
    all_print.append(all_err)
    all_print.append(all_ed / all_lambda)
    all_print.append(all_es / (1 - all_lambda))
    all_print.append(all_esp / (1 - all_lambda))
    all_print.append(all_ed_err / all_lambda)
    all_print.append(all_es_err / (1 - all_lambda))
    all_print.append(all_esp_err / (1 - all_lambda))
    all_print.append(all_etot)
    # all_print.append(all_etot_err)
    all_print.append(all_es)
    all_print.append(all_enthalpy)
    all_print.append(all_msd_xyz)
    all_print.append(np.array(all_msd_spin))
    all_print = np.array(all_print)
    np.savetxt(
        os.path.join(iter_name, "hti.out"),
        all_print.T,
        fmt="%.8e",
        header=(
            "lmbda dU dU_err Ud Us Usp Ud_err Us_err Usp_err etot "
            "spring_eng enthalpy msd_xyz "
            + ("msd_spin_mod" if spin_reference == "modulus" else "msd_spin")
        ),
    )

    diff_e, err, sys_err = integrate_range_hti(all_lambda, de, all_err, scheme=scheme)
    # new_lambda, i, i_e, s_e = integrate_range(all_lambda, de, all_err, scheme = scheme)
    # if new_lambda[-1] != all_lambda[-1] :
    #     if new_lambda[-1] == all_lambda[-2]:
    #         _, i1, i_e1, s_e1 = integrate_range(all_lambda[-2:], de[-2:], all_err[-2:], scheme='t')
    #         diff_e = i[-1] + i1[-1]
    #         err = np.linalg.norm([s_e[-1], s_e1[-1]])
    #         sys_err = i_e[-1] + i_e1[-1]
    #     else :
    #         raise RuntimeError("lambda does not match!")
    # else:
    #     diff_e = i[-1]
    #     err = s_e[-1]
    #     sys_err = i_e[-1]

    # diff_e, err = integrate(all_lambda, de, all_err)
    # sys_err = integrate_sys_err(all_lambda, de)

    path_endpnt = os.path.join(iter_name, "task.endpnt")
    if os.path.isdir(path_endpnt):
        print("# Found end point, compute thermo info from it")
        thermo_info = _compute_thermo(
            os.path.join(path_endpnt, "log.lammps"), natoms, stat_skip, stat_bsize
        )
    else:
        print("# Not found end point, compute thermo info from the last lambda")
        thermo_info = _compute_thermo(
            os.path.join(all_tasks[-1], "log.lammps"), natoms, stat_skip, stat_bsize
        )

    return diff_e, [err, sys_err], thermo_info


def _post_tasks_mbar(iter_name, jdata, natoms=None, switch="one-step", step="both"):
    stat_skip = jdata["stat_skip"]
    stat_bsize = jdata["stat_bsize"]
    all_tasks = glob.glob(os.path.join(iter_name, "task.[0-9]*"))
    all_tasks.sort()
    ntasks = len(all_tasks)
    equi_conf = jdata["equi_conf"]
    cwd = os.getcwd()
    os.chdir(iter_name)
    assert os.path.isfile(equi_conf)
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)
    temp = jdata["temp"]
    if natoms is None:
        natoms = get_natoms(equi_conf)
        if "copies" in jdata:
            natoms *= np.prod(jdata["copies"])
    print("# natoms: %d" % natoms)

    all_lambda = []
    for ii in all_tasks:
        lmda_name = os.path.join(ii, "lambda.out")
        ll = float(open(lmda_name).read())
        all_lambda.append(ll)
    all_lambda = np.array(all_lambda)
    nlambda = all_lambda.size

    ukn = np.array([])
    nk = []
    spring_lambda_mode = _get_spring_lambda_mode(jdata)
    kt_in_ev = pc.Boltzmann * temp / pc.electron_volt
    for idx, ii in enumerate(all_tasks):
        log_name = os.path.join(ii, "log.lammps")
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, "data"), data, fmt="%.6e")
        this_ed = data[:, 9] / kt_in_ev
        this_es = data[:, 8] / kt_in_ev
        this_esp = data[:, 10] / kt_in_ev
        this_ed = this_ed[stat_skip::1]
        this_es = this_es[stat_skip::1]
        this_esp = this_esp[stat_skip::1]
        nk.append(this_ed.size)
        if switch == "one-step" or switch == "two-step":
            if step == "both":
                ed = this_ed / all_lambda[idx]
                es = (this_es + this_esp) / (1 - all_lambda[idx])
                block_u = []
                for ll in all_lambda:
                    block_u.append(ed * ll + es * (1 - ll))
            elif step == "deep_on":
                ed = this_ed / all_lambda[idx]
                block_u = []
                for ll in all_lambda:
                    block_u.append(ed * ll)
            elif step == "spring_off":
                if spring_lambda_mode == "split":
                    es = this_es / (1 - all_lambda[idx])
                else:
                    es = (this_es + this_esp) / (1 - all_lambda[idx])
                block_u = []
                for ll in all_lambda:
                    block_u.append(es * (1 - ll))
            elif step == "lattice_spring_off":
                es = this_es / (1 - all_lambda[idx])
                block_u = []
                for ll in all_lambda:
                    block_u.append(es * (1 - ll))
            elif step == "spin_spring_off":
                es = this_esp / (1 - all_lambda[idx])
                block_u = []
                for ll in all_lambda:
                    block_u.append(es * (1 - ll))
            else:
                raise RuntimeError("unknow step", step)
        elif switch == "three-step":
            if step == "lj_on" or step == "deep_on":
                ed = this_ed
                block_u = []
                for ll in all_lambda:
                    block_u.append(ed * ll)
            elif step == "spring_off":
                ed = this_ed
                es = this_es / (1 - all_lambda[idx])
                block_u = []
                for ll in all_lambda:
                    block_u.append(ed * ll + es * (1 - ll))
            else:
                raise RuntimeError("unknow step", step)
        else:
            raise RuntimeError("unknow switch", switch)

        block_u = np.reshape(block_u, [nlambda, -1])
        if ukn.size == 0:
            ukn = block_u
        else:
            ukn = np.concatenate((ukn, block_u), axis=1)
    nk = np.array(nk)

    mbar = pymbar.MBAR(ukn, nk, initialize="BAR", relative_tolerance=1e-9)
    # Deltaf_ij, dDeltaf_ij, Theta_ij = mbar.getFreeEnergyDifferences()
    Deltaf_ij, dDeltaf_ij = mbar.getFreeEnergyDifferences()
    Deltaf_ij = Deltaf_ij / natoms
    dDeltaf_ij = dDeltaf_ij / natoms

    diff_e = Deltaf_ij[0, -1] * kt_in_ev
    err = dDeltaf_ij[0, -1] * kt_in_ev

    thermo_info = _compute_thermo(
        os.path.join(all_tasks[-1], "log.lammps"), natoms, stat_skip, stat_bsize
    )

    return diff_e, [err, 0], thermo_info


def print_thermo_info(info):
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
    method="inte",
    scheme="simpson",
    manual_pv=None,
    manual_pv_err=None,
    npt=None,
):
    jdata = json.load(open(os.path.join(job, "in.json")))
    if "reference" not in jdata:
        jdata["reference"] = "einstein"
    spin_reference = _get_spin_reference_mode(jdata)

    if jdata["crystal"] == "vega":
        e0 = free_energy(job)
    if jdata["crystal"] == "frenkel":
        if spin_reference == "vector":
            e0 = magnetic_frenkel(job)
        elif spin_reference == "modulus":
            e0 = magnetic_frenkel_modulus(job)
        elif spin_reference == "none":
            e0 = frenkel(job)
        else:
            raise RuntimeError("unknown spin_reference", spin_reference)
    de, de_err, thermo_info = post_tasks(job, jdata, method=method, scheme=scheme)
    # printing
    print_format = "%20.12f  %10.3e  %10.3e"
    print_thermo_info(thermo_info)

    info = thermo_info.copy()

    if jdata["reference"] == "einstein":
        if jdata["crystal"] == "frenkel" and spin_reference == "vector":
            print(f"# free ener of magnetic Frenkel Mole: {e0:20.8f}")
        elif jdata["crystal"] == "frenkel" and spin_reference == "modulus":
            print(f"# free ener of magnetic Frenkel modulus: {e0:20.8f}")
        else:
            print(f"# free ener of Einstein Mole: {e0:20.8f}")
    else:
        print(f"# free ener of ideal gas: {e0:20.8f}")

    print(
        ("# fe contrib due to integration " + print_format) % (de, de_err[0], de_err[1])
    )

    pv = None
    pv_err = None

    if free_energy_type == "helmholtz":
        e1 = e0 + de
        e1_err = de_err[0]
        print("# Helmholtz free ener per atom (stat_err inte_err) [eV]:")
        print(print_format % (e1, de_err[0], de_err[1]))
    elif free_energy_type == "gibbs":
        if npt is not None:
            npt_in = json.load(open(os.path.join(npt, "jdata.json")))
            npt_info = json.load(open(os.path.join(npt, "result.json")))
            p = npt_in["pres"]
            v = npt_info["v"]
            v_err = npt_info["v_err"]
            unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
            pv = p * v * unit_cvt
            pv_err = p * v_err * unit_cvt * np.sqrt(3)
            print(f"# use pv from npt task: pv = {pv:.6e} pv_err = {pv_err:.6e}")
        elif npt is None and manual_pv is None:
            pv = thermo_info["pv"]
        elif npt is None and manual_pv is not None:
            print(f"# use manual_pv={manual_pv}")
            pv = manual_pv
        if npt is None and manual_pv_err is None:
            pv_err = thermo_info["pv_err"]
        elif npt is None and manual_pv_err is not None:
            print(f"# use manual_pv_err={manual_pv_err}")
            pv_err = manual_pv_err
        e1 = e0 + de + pv
        e1_err = np.sqrt(de_err[0] ** 2 + pv_err**2)
        print("# Gibbs free ener per atom (stat_err inte_err) [eV]:")
        print(print_format % (e1, e1_err, de_err[1]))
    else:
        raise RuntimeError("unknown free energy type")

    info["free_energy_type"] = free_energy_type
    info["e0"] = e0
    info["pv"] = pv
    info["pv_err"] = pv_err
    info["de"] = de
    info["de_err"] = de_err
    info["e1"] = e1
    info["e1_err"] = e1_err
    with open(os.path.join(job, "result.json"), "w") as result:
        result.write(json.dumps(info))
    return info


def hti_phase_trans_analyze(job, jdata=None):
    if_phase_trans = False

    logfile0 = glob.glob(os.path.join(job, "00*", "hti.out"))[0]
    logfile1 = glob.glob(os.path.join(job, "01*", "hti.out"))[0]
    logfile2 = glob.glob(os.path.join(job, "02*", "hti.out"))[0]

    log0 = np.loadtxt(logfile0)
    log1 = np.loadtxt(logfile1)
    log2 = np.loadtxt(logfile2)
    print(logfile0, log0)
    print(logfile1, log1)
    print(logfile2, log2)

    msd0 = list(log0[:, -1])
    msd1 = list(log1[:, -1])
    msd2 = list(log2[:, -1])

    msd_all = []
    msd_all.extend(msd0)
    msd_all.extend(msd1)
    msd_all.extend(msd2)

    msd_min = min(msd_all)
    msd_max = max(msd_all)

    print("# dpti hti 00 log0")
    print(log0)
    print("# dpti hti 01 log1")
    print(log1)
    print("# dpti hti 02 log2")
    print(log2)

    if msd_min < 20 and msd_max > 100:
        if_phase_trans = True

    return if_phase_trans


def run_task(task_dir, machine_file, task_name, no_dp=False):
    if task_name == "00" or task_name == "01" or task_name == "02":
        job_work_dir_ = glob.glob(os.path.join(task_dir, task_name + "*"))
        assert (
            len(job_work_dir_) == 1
        ), f"The task_name you entered is {task_name}. It indicates that you want to run tasks for step {task_name} of the two-step or three-step HTI. Please make sure that there is one and only one {task_name}.* directory in the hti task directory."
        job_work_dir = job_work_dir_[0]
    elif task_name == "one-step":
        job_work_dir = task_dir
    task_dir_list = glob.glob(os.path.join(job_work_dir, "task*"))
    task_dir_list = sorted(task_dir_list)
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

    command = (
        f"{mdata['command']} -i in.lammps"
        if no_dp
        else f"ln -s ../../../graph.pb graph.pb; {mdata['command']} -i in.lammps"
    )
    task_list = [
        Task(
            command=command,
            task_work_path=ii,
            forward_files=["in.lammps", "conf.lmp"],
            backward_files=["log*", "dump.hti", "out.lmp"],
        )
        for ii in task_dir_list
    ]
    if not no_dp:
        submission.forward_common_files = ["graph.pb"]

    submission.register_task_list(task_list=task_list)
    submission.run_submission()


def add_module_subparsers(main_subparsers):
    module_parser = main_subparsers.add_parser(
        "hti_mag", help="Hamiltonian thermodynamic integration for atomic solid"
    )
    module_subparsers = module_parser.add_subparsers(
        help="commands of Hamiltonian thermodynamic integration for atomic solid",
        dest="command",
        required=True,
    )

    parser_gen = module_subparsers.add_parser("gen", help="generate a job")
    parser_gen.add_argument("PARAM", type=str, help="json parameter file")
    parser_gen.add_argument(
        "-o",
        "--output",
        type=str,
        default="new_job",
        help="the output folder for the job",
    )
    parser_gen.add_argument(
        "-s",
        "--switch",
        type=str,
        default="one-step",
        choices=[
            "one-step",
            "two-step",
            "three-step",
        ],
        help="one-step: switching on DP and switching off spring simultanenously.\
                            two-step: 1 switching on DP, 2 switching off spring.\
                            three-step: 1 switching on soft LJ, 2 switching on DP, 3 switching off spring and soft LJ.",
    )
    parser_gen.add_argument(
        "-z", "--meam", help="whether use meam instead of dp", action="store_true"
    )
    parser_gen.set_defaults(func=handle_gen)

    parser_compute = module_subparsers.add_parser(
        "compute", help="Compute the result of a job"
    )
    parser_compute.add_argument("JOB", type=str, help="folder of the job")
    parser_compute.add_argument(
        "-t",
        "--type",
        type=str,
        default="helmholtz",
        choices=["helmholtz", "gibbs"],
        help="the type of free energy",
    )
    parser_compute.add_argument(
        "-m",
        "--inte-method",
        type=str,
        default="inte",
        choices=["inte", "mbar"],
        help="the method of thermodynamic integration",
    )
    parser_compute.add_argument(
        "-s",
        "--scheme",
        type=str,
        default="simpson",
        help="the numeric integration scheme",
    )
    parser_compute.add_argument(
        "-g",
        "--pv",
        type=float,
        default=None,
        help="press*vol value override to calculate Gibbs free energy",
    )
    parser_compute.add_argument(
        "-G", "--pv-err", type=float, default=None, help="press*vol error"
    )
    parser_compute.add_argument(
        "--npt",
        type=str,
        default=None,
        help="directory of the npt task; will use PV from npt result, where P is the control variable and V varies.",
    )
    parser_compute.set_defaults(func=handle_compute)

    parser_run = module_subparsers.add_parser("run", help="run the job")
    parser_run.add_argument("JOB", type=str, help="folder of the job")
    parser_run.add_argument("machine", type=str, help="machine.json file for the job")
    parser_run.add_argument(
        "task_name",
        type=str,
        help="task name, can be one-step, 00, 01, or 02. The task names 00, 01, and 02 are used for the two-step or three-step HTI.",
    )
    parser_run.add_argument(
        "--no-dp", action="store_true", help="whether to use Deep Potential or not"
    )
    parser_run.set_defaults(func=handle_run)


def handle_gen(args):
    jdata = json.load(open(args.PARAM))
    if "crystal" in jdata and jdata["crystal"] == "frenkel":
        print("# gen task with Frenkel's Einstein crystal")
    else:
        print("# gen task with Vega's Einstein molecule")
    print("output:", args.output)
    make_tasks(
        args.output, jdata, ref="einstein", switch=args.switch, if_meam=args.meam
    )


def handle_compute(args):
    compute_task(
        job=args.JOB,
        free_energy_type=args.type,
        method=args.inte_method,
        scheme=args.scheme,
        manual_pv=args.pv,
        manual_pv_err=args.pv_err,
        npt=args.npt,
    )


def handle_run(args):
    run_task(args.JOB, args.machine, args.task_name, args.no_dp)
