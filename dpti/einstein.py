#!/usr/bin/env python3

import argparse
import json
import math
import os

import numpy as np
import scipy.constants as pc
from scipy.integrate import quad
from scipy.special import ive

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


def spin_ref_landau_partition(k, s0, beta):
    """ln I0 for the Landau modulus well, plus the (z, a, b) it was built from.

    The well is written k (|S|^2 - S0^2)^2 rather than Ma-Dudarev's A S^2 + B S^4
    (Phys. Rev. B 96, 094418 (2017), Eq. 12).  Same two-parameter family --
    A = -2 k S0^2, B = k, minimum at S0 -- but this form has U(S0) = 0 EXACTLY, the
    same zero convention as the harmonic modulus spring 1/2 k (|S| - S0)^2.  A
    reference whose minimum sat at -B S0^4 would need that offset carried consistently
    through F_ref *and* the lambda path, and getting it wrong cancels in every closure
    and reversibility check (memory: project_decompose_reference_bookkeeping).
    Choosing the zero away removes the trap, and is what makes the absolute free
    energies of the two wells directly comparable.

    Closed form.  With b = beta k, a = -2 beta k S0^2, z = a^2/(8b) = beta k S0^4 / 2:

        Z  = 4 pi exp(-beta k S0^4) * I2,   I2 = -dI0/da,
        I0 = int_0^inf exp(-b s^4 - a s^2) ds
           = (pi/4) sqrt(|a|/2b) e^z [ I_{-1/4}(z) + I_{1/4}(z) ]      (a < 0 always)

    Bessel, not erf: the quartic is non-elementary even on the full line, whereas the
    harmonic case is an ordinary Gaussian made non-elementary only by the s >= 0 cut.

    I_nu(z) overflows past z ~ 700, so this uses the exponentially scaled
    ive(nu, z) = exp(-z) I_nu(z) -- which turns the e^z of the closed form into e^{2z}.
    Returns ln I0, never I0.
    """
    if k <= 0.0:
        raise ValueError("spin_ref.k must be positive for the landau spin reference")
    if s0 <= 0.0:
        raise ValueError("spin_ref.s0 must be positive for the landau spin reference")
    b = beta * k
    a = -2.0 * beta * k * s0 * s0
    z = a * a / (8.0 * b)
    # ive(nu, z) = exp(-z) I_nu(z), so the e^z in the closed form BECOMES e^{2z} once
    # the scaled Bessel is substituted.  Writing +z here is the single mistake the
    # cross-check below exists to catch -- and it did catch it, first try.
    ln_i0 = math.log((math.pi / 4.0) * math.sqrt(abs(a) / (2.0 * b))
                     * (ive(-0.25, z) + ive(0.25, z))) + 2.0 * z
    return ln_i0, z, a, b


def _spin_ref_landau_smax(k, s0, beta):
    """|S| beyond which beta*U > 200 -- the integrand is dead there in double precision."""
    return s0 + (200.0 / (beta * k)) ** 0.25 + 1.0


def spin_ref_landau_lnz(k, s0, beta):
    """ln of Z = 4 pi int_0^inf s^2 exp(-beta k (s^2-s0^2)^2) ds, by direct quadrature.

    PRIMARY path.  The integrand is bounded by s^2 (the well minimum is exactly 0) and
    dies like exp(-beta k s^4), so quadrature on a finite window is unconditionally
    stable: no overflow, no cancellation, no large-argument Bessel.  The Bessel form
    above is carried only as an INDEPENDENT check -- a closed form that is silently
    wrong would shift every absolute free energy by the same amount, and no closure,
    reversibility or path-independence test in this pipeline would see it.
    """
    smax = _spin_ref_landau_smax(k, s0, beta)
    val, _ = quad(lambda s: s * s * np.exp(-beta * k * (s * s - s0 * s0) ** 2),
                  0.0, smax, points=[s0], epsabs=0.0, epsrel=1.0e-12, limit=400)
    return math.log(4.0 * math.pi * val)


def spin_ref_landau_mean_energy(k, s0, beta):
    """<U> of the Landau modulus reference, for the lambda = 0 reference-state gate.

    One confined degree of freedom (the modulus; the two angular ones are free), so
    this must come out near 1/2 kT, a little below it because the quartic wall is
    stiffer than harmonic.  That is the check the gate actually applies.
    """
    smax = _spin_ref_landau_smax(k, s0, beta)
    def w(s):
        return np.exp(-beta * k * (s * s - s0 * s0) ** 2)
    num, _ = quad(lambda s: k * (s * s - s0 * s0) ** 2 * s * s * w(s), 0.0, smax,
                  points=[s0], epsabs=0.0, epsrel=1.0e-10, limit=400)
    den, _ = quad(lambda s: s * s * w(s), 0.0, smax,
                  points=[s0], epsabs=0.0, epsrel=1.0e-10, limit=400)
    return num / den


def _spin_ref_style(spin_ref):
    """-> "harmonic" | "landau" | None (None meaning: not a modulus reference at all)."""
    style = str(spin_ref.get("style", "spring")).lower()
    if style in ("spring", "harmonic"):
        return "harmonic"
    if style == "landau":
        return "landau"
    return None


def _spin_ref_fe_one_type(style, k, s0, beta, kbt):
    """-kT ln Z for ONE spin type, for whichever modulus reference is in force."""
    if style == "harmonic":
        return -kbt * np.log(spin_ref_partition(k, s0, beta))

    ln_z = spin_ref_landau_lnz(k, s0, beta)
    # Cross-check: I2 = -dI0/da = -I0 * d(ln I0)/da, the log-derivative by a central
    # difference on the analytic ln I0, with a relative step so it is scale-free.
    # ln Z = ln(4 pi I2) - beta k s0^4, the last term being the exp(-beta k S0^4) that
    # shifts the well minimum to zero.
    #
    # Differentiate ln I0, NOT I0: ln I0 ~ a^2/(4b) is nearly quadratic in a, so its
    # central difference is almost exact, whereas I0 ~ e^{2z} has an O((z da/|a|)^2)
    # truncation error that grows with the well stiffness and tripped this guard at
    # k = 0.05, s0 = 2.46, 200 K (rel 2.7e-7) on a reference that was in fact correct.
    a0 = -2.0 * beta * k * s0 * s0
    da = 1.0e-5 * abs(a0)
    b = beta * k

    def _ln_i0(a_val):
        z = a_val * a_val / (8.0 * b)
        return math.log((math.pi / 4.0) * math.sqrt(abs(a_val) / (2.0 * b))
                        * (ive(-0.25, z) + ive(0.25, z))) + 2.0 * z

    dln_i0 = (_ln_i0(a0 + da) - _ln_i0(a0 - da)) / (2.0 * da)
    ln_z_bessel = (math.log(4.0 * math.pi) + _ln_i0(a0) + math.log(-dln_i0)
                   - beta * k * s0 ** 4)
    if not math.isclose(ln_z, ln_z_bessel, rel_tol=1.0e-7, abs_tol=1.0e-9):
        raise RuntimeError(
            f"landau spin reference: quadrature ln Z = {ln_z!r} disagrees with the "
            f"Bessel closed form {ln_z_bessel!r} (k={k}, s0={s0}, beta={beta}); "
            "refusing to return a reference free energy only one of the two supports")
    return -kbt * ln_z


def spin_ref_fe_per_atom(temp, atom_numbs, spin_map, spin_ref):
    style = _spin_ref_style(spin_ref)
    if style is None:
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
        fe += (count / natoms) * _spin_ref_fe_one_type(style, k, s0, beta, kbt)
    return float(fe)


def spin_ref_analytic_per_atom(temp, atom_numbs, spin_map, spin_ref):
    style = _spin_ref_style(spin_ref)
    if style is None:
        raise ValueError("spin_ref analytic values support only harmonic/landau style")

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
        fe_type = _spin_ref_fe_one_type(style, k, s0, beta, kbt)
        u_type = (spin_ref_mean_energy(k, s0, beta) if style == "harmonic"
                  else spin_ref_landau_mean_energy(k, s0, beta))
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

    temp = jdata["temp"]
    mass_map = get_first_matched_key_from_dict(jdata, ["mass_map", "model_mass_map"])
    spin_mass = get_first_matched_key_from_dict(jdata, ["spin_mass_map", "sp_mass_map", "spin_mass"])
    spin_map = get_first_matched_key_from_dict(jdata, ["spin_map", "sp_map"])

    equi_conf = jdata["equi_conf"]
    cwd = os.getcwd()
    os.chdir(job)
    assert os.path.isfile(equi_conf)
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)

    spring_spin_k = get_first_matched_key_from_dict(
        jdata, ["spin_spring_k", "s_spring_k", "spring_k_spin", "spring_spin_k"]
    )
    if spring_spin_k is None:
        raise ValueError(
            "magnetic_frenkel requires explicit `spin_spring_k` (or `s_spring_k`/`spring_k_spin`) in in.json"
        )
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

    natoms = [ii * ncopies for ii in sys_data["atom_numbs"]]
    total_atoms = np.sum(natoms)

    Lambda_S_kin = [compute_spin_lambda(temp, ii * spin_mass) for ii in mass_map]
    m_spring_spin_k = [spring_spin_k * ii * spin_mass for ii in mass_map]
    Lambda_S_ref = [compute_spin_spring(temp, ii) for ii in m_spring_spin_k]

    sfe = 0.0

    for idx, ii in enumerate(natoms):
        if spin_map[idx]:
            # magnetic contributions
            sfe += 3.0 * ii * np.log(Lambda_S_ref[idx])
            if include_spin_kinetic:
                sfe += 3.0 * ii * np.log(Lambda_S_kin[idx])

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

    sfe *= pc.Boltzmann * temp / pc.electron_volt
    sfe /= total_atoms
    # Reuse the lattice-only implementation so scalar shared-frequency and
    # per-type Einstein springs receive exactly the same generalized CM correction.
    return frenkel(job) + sfe


def magnetic_frenkel_modulus(job):
    """Frenkel lattice reference plus harmonic spin-modulus reference."""
    with open(os.path.join(job, "in.json")) as f:
        jdata = json.load(f)

    spin_ref = jdata.get("spin_ref")
    if spin_ref is None:
        raise ValueError("magnetic_frenkel_modulus requires `spin_ref` in in.json")

    if _spin_ref_style(spin_ref) is None:
        raise ValueError(
            "magnetic_frenkel_modulus supports only harmonic/spring or landau spin_ref style"
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
