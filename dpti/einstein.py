#!/usr/bin/env python3

import argparse
import json
import math
import os

import numpy as np
import scipy.constants as pc
from scipy.integrate import quad

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


def _expand_type_param(value, ntypes, name):
    if isinstance(value, (list, tuple)):
        if len(value) != ntypes:
            raise ValueError(f"{name} length {len(value)} does not match ntypes {ntypes}")
        return [float(v) for v in value]
    return [float(value) for _ in range(ntypes)]


def spin_ref_partition(k, s0, beta):
    """Configurational partition function for 1/2 k (|S|-S0)^2."""
    if k <= 0.0:
        raise ValueError("spin_ref.k must be positive for harmonic spin reference")
    a = 0.5 * beta * k
    x = math.sqrt(a) * s0
    erf_term = 1.0 + math.erf(x)
    exp_term = math.exp(-a * s0 * s0)
    return (
        (math.pi ** 1.5) / (a ** 1.5) * (1.0 + 2.0 * a * s0 * s0) * erf_term
        + 2.0 * math.pi * s0 / a * exp_term
    )


def spin_ref_mean_energy(k, s0, beta):
    z_spin = spin_ref_partition(k, s0, beta)

    def integrand(s):
        u = 0.5 * k * (s - s0) ** 2
        return u * s * s * np.exp(-0.5 * beta * k * (s - s0) ** 2)

    val, _ = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1.0e-10, limit=200)
    return 4.0 * np.pi * val / z_spin


def spin_ref_fe_per_atom(temp, atom_numbs, spin_map, spin_ref):
    style = str(spin_ref.get("style", "spring")).lower()
    if style not in ("spring", "harmonic"):
        return 0.0

    ntypes = len(atom_numbs)
    k_by_type = _expand_type_param(spin_ref.get("k", 0.0), ntypes, "spin_ref.k")
    s0_by_type = _expand_type_param(spin_ref.get("s0", 0.0), ntypes, "spin_ref.s0")
    natoms = sum(atom_numbs)
    if natoms <= 0:
        return 0.0

    kbt = pc.Boltzmann / pc.electron_volt * temp
    beta = 1.0 / kbt
    fe = 0.0
    for count, is_spin, k, s0 in zip(atom_numbs, spin_map, k_by_type, s0_by_type):
        if not is_spin or count == 0:
            continue
        z_spin = spin_ref_partition(k, s0, beta)
        fe += (count / natoms) * (-kbt * np.log(z_spin))
    return float(fe)


def spin_ref_analytic_per_atom(temp, atom_numbs, spin_map, spin_ref):
    style = str(spin_ref.get("style", "spring")).lower()
    if style not in ("spring", "harmonic"):
        raise ValueError("spin_ref analytic values support only harmonic spin_ref style")

    ntypes = len(atom_numbs)
    k_by_type = _expand_type_param(spin_ref.get("k", 0.0), ntypes, "spin_ref.k")
    s0_by_type = _expand_type_param(spin_ref.get("s0", 0.0), ntypes, "spin_ref.s0")
    natoms = sum(atom_numbs)
    if natoms <= 0:
        return {"free_energy": 0.0, "mean_energy": 0.0, "per_type": []}

    kbt = pc.Boltzmann / pc.electron_volt * temp
    beta = 1.0 / kbt
    fe = 0.0
    u = 0.0
    per_type = []
    for ii, (count, is_spin, k, s0) in enumerate(zip(atom_numbs, spin_map, k_by_type, s0_by_type)):
        if not is_spin or count == 0:
            continue
        z_spin = spin_ref_partition(k, s0, beta)
        fe_type = -kbt * np.log(z_spin)
        u_type = spin_ref_mean_energy(k, s0, beta)
        weight = count / natoms
        fe += weight * fe_type
        u += weight * u_type
        per_type.append({
            "type": int(ii + 1),
            "count": int(count),
            "k": float(k),
            "s0": float(s0),
            "free_energy": float(fe_type),
            "mean_energy": float(u_type),
        })
    return {"free_energy": float(fe), "mean_energy": float(u), "per_type": per_type}


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
    spring_k = jdata["spring_k"]
    # spring_k SCALAR  -> shared Einstein frequency: k_i = spring_k * m_i  (omega^2 = spring_k)
    # spring_k LIST    -> per-type spring constants k_i [eV/A^2] given directly.  Needed when
    #   one species sits in a much stiffer well than the rest: in bcc Fe+C the interstitial C
    #   is clamped by six Fe neighbours, so its measured <dr^2> is only 1.2x that of Fe, not
    #   the 4.65x a shared omega would impose on a 12 amu atom.  A shared omega therefore
    #   hands C a reference well ~3.2x too wide, which costs reference/target overlap.
    if isinstance(spring_k, list):
        assert len(spring_k) == len(mass_map)
        m_spring_k = list(spring_k)
    else:
        m_spring_k = [spring_k * ii for ii in mass_map]
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

    fe = 0
    sum_m = 0
    inv_k_m2 = 0.0
    for idx, ii in enumerate(natoms):
        fe += 3.0 * ii * np.log(Lambda_k[idx]) # -ln[(1/\Lambda)^(3N)] = 3N ln(\Lambda)
        fe += 3.0 * ii * np.log(Lambda_s[idx]) # -ln(\frac{\pi}{\beta E}^{3(N)/2}) = 3N/2 ln(\frac{\beta E}{\pi})
        sum_m += mass_map[idx] * ii
        inv_k_m2 += ii * mass_map[idx] ** 2 / m_spring_k[idx]

    # Centre-of-mass correction for the Frenkel molecule.  Integrating the Einstein
    # Gaussian against delta^3(R_CM) gives, for ARBITRARY per-type spring constants k_i,
    #     F_CM / kT = -3 ln Lambda_s(K_CM),   K_CM = M^2 / sum_i n_i m_i^2 / k_i
    # with M = sum_i n_i m_i.  The previous code hard-coded the special case k_i = m_i w^2,
    # for which sum_i n_i m_i^2/k_i = M/w^2 and hence K_CM = M w^2 -- exactly the old
    # `-3 ln Lambda_s(w^2) - 1.5 ln M`.  The general form is therefore an exact
    # generalisation, not an approximation, and reduces to the old one bit-for-bit.
    k_cm = sum_m**2 / inv_k_m2
    fe -= 3.0 * np.log(compute_spring(temp, k_cm))
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

    include_spin_kinetic = bool(jdata.get("include_spin_kinetic", False))
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
            # magnetic contributions
            sfe += 3.0 * ii * np.log(Lambda_S_ref[idx])
            if include_spin_kinetic:
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

    fe *= pc.Boltzmann * temp / pc.electron_volt
    fe /= total_atoms
    sfe *= pc.Boltzmann * temp / pc.electron_volt
    sfe /= total_atoms
    fe += sfe
    return fe


def magnetic_frenkel_modulus(job):
    """Frenkel lattice reference plus harmonic spin-modulus reference."""
    with open(os.path.join(job, "in.json")) as f:
        jdata = json.load(f)

    spin_ref = jdata.get("spin_ref")
    if spin_ref is None:
        raise ValueError("magnetic_frenkel_modulus requires `spin_ref` in in.json")

    spin_ref_style = str(spin_ref.get("style", "spring")).lower()
    if spin_ref_style not in ("spring", "harmonic"):
        raise ValueError(
            "magnetic_frenkel_modulus supports only harmonic/spring spin_ref style"
        )

    equi_conf = jdata["equi_conf"]
    cwd = os.getcwd()
    os.chdir(job)
    assert os.path.isfile(equi_conf)
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)

    spin_map = get_first_matched_key_from_dict(jdata, ["spin_map", "sp_map"])
    if "copies" in jdata:
        ncopies = np.prod(jdata["copies"])
    else:
        ncopies = 1

    with open(equi_conf) as f:
        sys_data = lmp.to_system_data(f.read().split("\n"))

    atom_numbs = [ii * ncopies for ii in sys_data["atom_numbs"]]
    spin_fe = spin_ref_fe_per_atom(jdata["temp"], atom_numbs, spin_map, spin_ref)
    return frenkel(job) + spin_fe

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
