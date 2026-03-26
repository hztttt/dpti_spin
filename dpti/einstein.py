#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import scipy.constants as pc

# from . import lib
from dpti.lib import lmp
from dpti.lib.utils import get_first_matched_key_from_dict

# from lib import lmp


def compute_lambda(temp, mass):
    # De-Broglie Wavelength
    # $$\Lambda = \sqrt{\frac{h^2} {2\pi m k_B T}}$$
    ret = (
        2.0
        * np.pi
        * mass
        * (1e-3 / pc.Avogadro)
        * pc.Boltzmann
        * temp
        / (pc.Planck * pc.Planck)
    )
    # print('!!', mass, 1./np.sqrt(ret))
    return 1.0 / np.sqrt(ret)


def compute_spring(temp, spring_k):
    # $\text{Lambda\_s} = \sqrt{\frac{E}{2 \pi k_B T}}$ 
    ret = (0.5 * spring_k * pc.electron_volt / (pc.angstrom * pc.angstrom)) / (
        pc.Boltzmann * temp * np.pi
    )
    return np.sqrt(ret)


def compute_spin_spring(temp, spin_spring_k):
    ret = (0.5 * spin_spring_k * pc.electron_volt) / (pc.Boltzmann * temp * np.pi)
    return np.sqrt(ret)


def compute_spin_lambda(temp, spin_mu, h_s=pc.Planck):
    ret = 2.0 * np.pi * spin_mu * (1e-3 / pc.Avogadro) * pc.Boltzmann * temp / (h_s * h_s)
    return 1.0 / np.sqrt(ret)


def ideal_gas_fe(job):
    with open(os.path.join(job, "in.json")) as f:
        jdata = json.load(f)
    equi_conf = jdata["equi_conf"]
    cwd = os.getcwd()
    os.chdir(job)
    assert os.path.isfile(equi_conf), equi_conf
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)
    temp = jdata["temp"]
    # mass_map = jdata['mass_map']
    mass_map = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    if "copies" in jdata:
        ncopies = np.prod(jdata["copies"])
    else:
        ncopies = 1

    with open(equi_conf) as f:
        sys_data = lmp.to_system_data(f.read().split("\n"))
    vol = np.linalg.det(sys_data["cell"])
    natoms = [ii * ncopies for ii in sys_data["atom_numbs"]]
    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]
    fe = 0
    for idx, ii in enumerate(natoms):
        # kinetic contrib
        # print('```', idx, ii)
        if ii > 0:
            rho = ii / (vol * (pc.angstrom**3))
            fe += ii * np.log(rho * (Lambda_k[idx] ** 3))
            fe -= ii
            fe += 0.5 * np.log(2.0 * np.pi * ii)
    fe *= pc.Boltzmann * temp / pc.electron_volt
    fe /= np.sum(natoms)
    return fe


def free_energy(job):
    # vega style free energy
    # print(898, job, os.getcwd())
    with open(os.path.join(job, "in.json")) as f:
        jdata = json.load(f)
    # jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
    equi_conf = jdata["equi_conf"]
    cwd = os.getcwd()
    os.chdir(job)
    assert os.path.isfile(equi_conf)
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)
    temp = jdata["temp"]
    # mass_map = jdata['mass_map']
    mass_map = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    spring_k = jdata["spring_k"]
    if not isinstance(spring_k, list):
        m_spring_k = []
        for ii in mass_map:
            m_spring_k.append(spring_k * ii)
        # spring_k = spring_k_1
    assert len(mass_map) == len(m_spring_k)
    if "copies" in jdata:
        ncopies = np.prod(jdata["copies"])
    else:
        ncopies = 1

    with open(equi_conf) as f:
        sys_data = lmp.to_system_data(f.read().split("\n"))

    # sys_data = lmp.to_system_data(open(equi_conf).read().split('\n'))
    vol = np.linalg.det(sys_data["cell"])
    natoms = [ii * ncopies for ii in sys_data["atom_numbs"]]

    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]
    Lambda_s = [compute_spring(temp, ii) for ii in m_spring_k]
    # print(np.log(Lambda_k), np.log(Lambda_s))

    with open(equi_conf) as fp:
        lines = list(fp)
    for idx, ii in enumerate(lines):
        if "Atoms" in ii:
            break
    first_type = int(lines[idx + 2].split()[1]) - 1
    # print('# fixed atom of type %d ' % first_type)

    fe = 0
    fact = pc.Boltzmann * temp / pc.electron_volt / np.sum(natoms)
    for idx, ii in enumerate(natoms):
        # kinetic contrib
        # print(idx)
        fe += 3 * ii * np.log(Lambda_k[idx])
        # print(3 * ii * np.log(Lambda_k[idx]) * fact)
        if idx == first_type:
            fe += 3 * (ii - 1) * np.log(Lambda_s[idx])
            fe += np.log(ii / (vol * (pc.angstrom**3)))
            # print(3.0 * (ii-1) * np.log(Lambda_s[idx]) * fact)
            # print(np.log(ii / (vol * (pc.angstrom**3))) * fact)
        else:
            fe += 3 * ii * np.log(Lambda_s[idx])
            # print(3.0 * ii * np.log(Lambda_s[idx]) * fact)
    fe *= pc.Boltzmann * temp / pc.electron_volt
    fe /= np.sum(natoms)
    return fe


def frenkel(job):
    with open(os.path.join(job, "in.json")) as f:
        jdata = json.load(f)
    # jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
    equi_conf = jdata["equi_conf"]
    cwd = os.getcwd()
    os.chdir(job)
    assert os.path.isfile(equi_conf)
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)
    temp = jdata["temp"]
    # mass_map = jdata['mass_map']
    mass_map = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    s_spring_k = jdata["spring_k"]
    spring_k = jdata["spring_k"]
    assert not isinstance(spring_k, list)
    if not isinstance(spring_k, list):
        m_spring_k = []
        for ii in mass_map:
            m_spring_k.append(spring_k * ii)
        # spring_k = spring_k_1
    if "copies" in jdata:
        ncopies = np.prod(jdata["copies"])
    else:
        ncopies = 1
    with open(equi_conf) as f:
        sys_data = lmp.to_system_data(f.read().split("\n"))

    # sys_data = lmp.to_system_data(open(equi_conf).read().split('\n'))
    vol = np.linalg.det(sys_data["cell"])
    natoms = [ii * ncopies for ii in sys_data["atom_numbs"]]

    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]
    Lambda_s = [compute_spring(temp, ii) for ii in m_spring_k]
    s_Lambda_s = compute_spring(temp, s_spring_k) # s_spring_k 和 spring_k的区别? 

    fe = 0
    sum_m = 0
    fact = pc.Boltzmann * temp / pc.electron_volt / np.sum(natoms)
    for idx, ii in enumerate(natoms):
        fe += 3.0 * ii * np.log(Lambda_k[idx]) # -ln[(1/\Lambda)^(3N)] = 3N ln(\Lambda)
        fe += 3.0 * ii * np.log(Lambda_s[idx]) # -ln(\frac{\pi}{\beta E}^{3(N)/2}) = 3N/2 ln(\frac{\beta E}{\pi})
        # print(idx)
        # print(3.0 * ii * np.log(Lambda_k[idx]) * fact)
        # print(3.0 * ii * np.log(Lambda_s[idx]) * fact)
        sum_m += mass_map[idx] * ii # 为什么这里是原子质量乘以原子数目? 公式对应ln(N^(3/2)) 是因为在de Broglie wavelength引入了质量?
    fe -= 3.0 * np.log(s_Lambda_s)
    fe -= 1.5 * np.log(sum_m)
    # fe += 2.0 * np.log(np.sum(natoms)/3.0)
    # print('# FS corr (does not apply)', 2.0 * np.log(np.sum(natoms)/3.0) *pc.Boltzmann * temp / pc.electron_volt / np.sum(natoms) * 3.0)
    # print((3.0 * np.log(s_Lambda_s) + 1.5 * np.log(sum_m)) * fact)
    fe += np.log(np.sum(natoms) / (vol * (pc.angstrom**3)))
    # print(np.log(np.sum(natoms) / (vol * (pc.angstrom**3))) * fact, np.log(np.sum(natoms) / 3.0 / (vol * (pc.angstrom**3))) * fact)

    fe *= pc.Boltzmann * temp / pc.electron_volt
    fe /= np.sum(natoms)
    # total_atom_num = np.sum(natoms)
    # print('### debug:', fe, Lambda_k, Lambda_s, np.log(Lambda_k[0]), np.log(Lambda_s[0]))
    # print('### debug:average U', 3 * pc.Boltzmann * temp)
    # print('### debug:Helmholtz free energy', fe)
    # print('### debug:', (3*total_atom_num - 0.5 - 3*total_atom_num*np.log(Lambda_k[0]) - 3*(natoms[0]-1)*np.log(Lambda_s[0]) + np.log((vol * (pc.angstrom**3))) + 0.5*np.log(total_atom_num)) )
    return fe


def magnetic_frenkel(job):
    with open(os.path.join(job, "in.json")) as f:
        jdata = json.load(f)

    equi_conf = jdata["equi_conf"]
    cwd = os.getcwd()
    os.chdir(job)
    assert os.path.isfile(equi_conf)
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)

    temp = jdata["temp"]
    mass_map = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    spin_mass = get_first_matched_key_from_dict(jdata, ["spin_mass_map", "sp_mass_map", "spin_mass"])
    spin_map = get_first_matched_key_from_dict(jdata, ["spin_map", "sp_map"])

    spring_k = jdata["spring_k"]
    assert not isinstance(spring_k, list)
    m_spring_k = [spring_k * ii for ii in mass_map]
    spring_spin_k = get_first_matched_key_from_dict(
        jdata, ["spin_spring_k", "s_spring_k", "spring_k_spin", "spring_spin_k"]
    )
    if spring_spin_k is None:
        raise ValueError(
            "magnetic_frenkel requires explicit `spin_spring_k` (or `s_spring_k`/`spring_k_spin`) in in.json"
        )
    m_spring_spin_k = [spring_spin_k * ii * spin_mass for ii in mass_map]

    spin_model = str(jdata.get("spin_model", "tspin")).lower()
    if spin_model not in ["tspin", "llg"]:
        raise ValueError("spin_model must be either 'tspin' or 'llg'")

    include_spin_kinetic = bool(jdata.get("include_spin_kinetic", True))
    h_s = float(jdata.get("h_s", pc.Planck))

    if "copies" in jdata:
        ncopies = np.prod(jdata["copies"])
    else:
        ncopies = 1

    with open(equi_conf) as f:
        sys_data = lmp.to_system_data(f.read().split("\n"))

    vol = np.linalg.det(sys_data["cell"])
    natoms = [ii * ncopies for ii in sys_data["atom_numbs"]]
    total_atoms = np.sum(natoms)

    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]
    Lambda_E = [compute_spring(temp, ii) for ii in m_spring_k]
    Lambda_E_cm = compute_spring(temp, spring_k)
    Lambda_S_kin = [compute_spin_lambda(temp, ii * spin_mass) for ii in mass_map]
    Lambda_S_ref = [compute_spin_spring(temp, ii) for ii in m_spring_spin_k]

    fe = 0.0
    sfe = 0.0
    sum_m = 0.0

    for idx, ii in enumerate(natoms):
        fe += 3.0 * ii * np.log(Lambda_k[idx])
        fe += 3.0 * ii * np.log(Lambda_E[idx])
        sum_m += mass_map[idx] * ii
        if spin_map[idx]:
             # magentic contributions.
             sfe += 3.0 * ii * np.log(Lambda_S_ref[idx])
             sfe += 3.0 * ii * np.log(Lambda_S_kin[idx])

    fe -= 3.0 * np.log(Lambda_E_cm)
    fe -= 1.5 * np.log(sum_m)
    fe += np.log(total_atoms / (vol * (pc.angstrom**3)))

    # if spin_model == "tspin":
    #     fe += 3.0 * total_atoms * np.log(Lambda_S_ref)
    # else:
    #     kbt_ev = pc.Boltzmann * temp / pc.electron_volt
    #     beta_ev = 1.0 / kbt_ev
    #     if spin_spring_k <= 0.0:
    #         raise ValueError("spin_spring_k must be positive for llg model")
    #     spin_factor = (2.0 * np.pi * kbt_ev / spin_spring_k) * (
    #         1.0 - np.exp(-2.0 * beta_ev * spin_spring_k)
    #     )
    #     fe -= total_atoms * np.log(spin_factor)

    if include_spin_kinetic:
        if spin_mass is None:
            raise ValueError(
                "include_spin_kinetic=True requires `spin_mass` in in.json"
            )
        # Lambda_S_kin = compute_spin_lambda(temp, float(spin_mass), h_s)
        # Lambda_S_kin = [compute_spin_lambda(temp, ii * spin_mass) for ii in mass_map]
        # fe += 3.0 * total_atoms * np.log(Lambda_S_kin)

    fe *= pc.Boltzmann * temp / pc.electron_volt
    fe /= total_atoms
    sfe *= pc.Boltzmann * temp / pc.electron_volt
    sfe /= total_atoms
    fe += sfe
    return fe

def _main():
    parser = argparse.ArgumentParser(
        description="Compute free energy of Einstein molecule"
    )
    parser.add_argument("PARAM", type=str, help="json parameter file")
    args = parser.parse_args()

    jdata = json.load(open(args.PARAM))
    fe = free_energy(jdata)
    print("# free energy of Einstein molecule in eV:")
    print(fe)


if __name__ == "__main__":
    _main()
