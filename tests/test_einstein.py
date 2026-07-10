import os
import unittest

import numpy as np
import scipy.constants as pc

# from numpy.testing import assert_almost_equal
from dpti.einstein import (
    compute_spin_lambda,
    compute_spin_spring,
    free_energy,
    frenkel,
    ideal_gas_fe,
    magnetic_frenkel,
    magnetic_frenkel_modulus,
    spin_ref_analytic_per_atom,
    spin_ref_fe_per_atom,
    spin_ref_mean_energy,
    spin_ref_partition,
)
from scipy.integrate import quad

lambda_seq = [
    "0.00:0.05:0.010",
    "0.05:0.15:0.020",
    "0.15:0.35:0.040",
    "0.35:1.00:0.065",
    "1",
]


_HTI_TEST_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hti_test_files")


class TestEinstein(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_frenkel(self):
        fe1 = -0.14061204010964043
        fe2 = frenkel(os.path.join(_HTI_TEST_FILES, "frenkel"))
        self.assertAlmostEqual(fe1, fe2)

    def test_vega(self):
        fe1 = -0.13882760104909486
        fe2 = free_energy(os.path.join(_HTI_TEST_FILES, "vega"))
        self.assertAlmostEqual(fe1, fe2)

    def test_ideal(self):
        fe1 = -1.8983591660560315
        fe2 = ideal_gas_fe(os.path.join(_HTI_TEST_FILES, "ideal"))
        # print('ideal_gas fe', fe2)
        self.assertAlmostEqual(fe1, fe2)

        # print(ideal_gas_fe)
        # pass
        # print(fe2)
        # self.assertAlmostEqual(fe1, fe2)
        # assert_almost_equal(array1, array2, decimal=10)

    # def test_protect_eps(self):
    #     array1 = parse_seq(lambda_seq, protect_eps=1e-6)
    #     assert_almost_equal(array1, array2, decimal=10)

    # def test_no_posi_args(self):
    #     with self.assertRaises(TypeError):
    #         parse_seq(lambda_seq, 1e-6)
    # assert_almost_equal(array1, array2, decimal=10)


# class TestBlockAvg(unittest.TestCase):
#     def setUp(self):
#         self.maxDiff = None

#     def test_normal(self):
#         avg1 = 7.158014302
#         err1 = 0.262593991
#         data_file = 'lammps_test_files/get_thermo.data'
#         data_array = np.loadtxt(data_file)
#         avg2, err2 = block_avg(data_array[:,1])
#         self.assertAlmostEqual(avg1, avg2, places=8)
#         self.assertAlmostEqual(err1, err2, places=8)

# class TestIntegrateRangeHti(unittest.TestCase):
#     def setUp(self):
#         self.maxDiff = None
#         # lamb

#     def test_lamb_array_odd(self):
#         result1 = -0.07001571298782591
#         stt_err1 = 2.1394708051996743e-05
#         sys_err2 = 2.0797767427780528e-07
#         data = np.loadtxt('hti_test_files/odd.hti.out')
#         lamb_array = data[:,0]
#         dU_array = data[:,1]
#         dU_err_array = data[:,2]
#         result2, stt_err2, sys_err2 = integrate_range_hti(lamb_array, dU_array, dU_err_array)
#         self.assertAlmostEqual(result1, result2, places=8)
#         self.assertAlmostEqual(stt_err1, stt_err2, places=8)
#         self.assertAlmostEqual(sys_err2, sys_err2, places=8)

#     def test_lamb_array_even(self):
#         result1 = -35.48046669098458
#         stt_err1 = 0.0001625198805022198
#         sys_err2 = 8.812949063852216e-07
#         data = np.loadtxt('hti_test_files/even.hti.out')
#         lamb_array = data[:,0]
#         dU_array = data[:,1]
#         dU_err_array = data[:,2]
#         result2, stt_err2, sys_err2 = integrate_range_hti(lamb_array, dU_array, dU_err_array)
#         self.assertAlmostEqual(result1, result2, places=8)
#         self.assertAlmostEqual(stt_err1, stt_err2, places=8)
#         self.assertAlmostEqual(sys_err2, sys_err2, places=8)


class TestComputeSpinSpring(unittest.TestCase):
    def test_value_at_200K(self):
        temp, k = 200.0, 0.12
        expected = np.sqrt(0.5 * k * pc.electron_volt / (pc.Boltzmann * temp * np.pi))
        self.assertAlmostEqual(compute_spin_spring(temp, k), expected, places=12)

    def test_scales_with_sqrt_k(self):
        temp = 300.0
        v1 = compute_spin_spring(temp, 0.10)
        v2 = compute_spin_spring(temp, 0.40)
        self.assertAlmostEqual(v2 / v1, 2.0, places=12)


class TestComputeSpinLambda(unittest.TestCase):
    def test_value_at_200K(self):
        temp, mu = 200.0, 1.0
        ret = 2.0 * np.pi * mu * (1e-3 / pc.Avogadro) * pc.Boltzmann * temp / (pc.Planck ** 2)
        expected = 1.0 / np.sqrt(ret)
        self.assertAlmostEqual(compute_spin_lambda(temp, mu), expected, places=20)

    def test_scales_inversely_with_sqrt_T(self):
        mu = 1.0
        v1 = compute_spin_lambda(100.0, mu)
        v2 = compute_spin_lambda(400.0, mu)
        self.assertAlmostEqual(v2 / v1, 0.5, places=12)


class TestSpinRefFreeEnergy(unittest.TestCase):
    def test_partition_matches_direct_quadrature(self):
        temp = 2000.0
        k = 1.0
        s0 = 2.5
        beta = 1.0 / (pc.Boltzmann / pc.electron_volt * temp)

        expected, _ = quad(
            lambda s: 4.0 * np.pi * s * s * np.exp(-0.5 * beta * k * (s - s0) ** 2),
            0.0,
            np.inf,
            epsabs=0.0,
            epsrel=1.0e-11,
            limit=200,
        )
        self.assertAlmostEqual(spin_ref_partition(k, s0, beta), expected, places=10)

    def test_partition_s0_zero_limit(self):
        temp = 500.0
        k = 0.7
        beta = 1.0 / (pc.Boltzmann / pc.electron_volt * temp)
        a = 0.5 * beta * k
        expected = np.pi ** 1.5 / a ** 1.5
        self.assertAlmostEqual(spin_ref_partition(k, 0.0, beta), expected, places=12)

    def test_mean_energy_matches_direct_quadrature(self):
        temp = 2000.0
        k = 0.7
        s0 = 2.0
        beta = 1.0 / (pc.Boltzmann / pc.electron_volt * temp)
        z_spin = spin_ref_partition(k, s0, beta)
        expected, _ = quad(
            lambda s: (
                4.0
                * np.pi
                * 0.5
                * k
                * (s - s0) ** 2
                * s
                * s
                * np.exp(-0.5 * beta * k * (s - s0) ** 2)
            ),
            0.0,
            np.inf,
            epsabs=0.0,
            epsrel=1.0e-11,
            limit=200,
        )
        self.assertAlmostEqual(spin_ref_mean_energy(k, s0, beta), expected / z_spin, places=11)

    def test_fe_per_atom_weighted_by_spin_types(self):
        temp = 2000.0
        atom_numbs = [2, 1, 3]
        spin_map = [1, 0, 1]
        spin_ref = {
            "style": "spring",
            "k": [0.7, 99.0, 1.0],
            "s0": [2.0, 99.0, 2.5],
        }
        kbt = pc.Boltzmann / pc.electron_volt * temp
        beta = 1.0 / kbt
        f0 = -kbt * np.log(spin_ref_partition(0.7, 2.0, beta))
        f2 = -kbt * np.log(spin_ref_partition(1.0, 2.5, beta))
        expected = (2.0 * f0 + 3.0 * f2) / 6.0
        self.assertAlmostEqual(
            spin_ref_fe_per_atom(temp, atom_numbs, spin_map, spin_ref),
            expected,
            places=12,
        )

    def test_analytic_per_atom_reports_mean_energy(self):
        analytic = spin_ref_analytic_per_atom(
            2000.0,
            [4],
            [1],
            {"style": "spring", "k": 1.0, "s0": 2.5},
        )
        self.assertEqual(len(analytic["per_type"]), 1)
        self.assertAlmostEqual(analytic["free_energy"], -0.7636030451901273)
        self.assertAlmostEqual(analytic["mean_energy"], analytic["per_type"][0]["mean_energy"])


_MAG_FRENKEL_DIR = os.path.join(_HTI_TEST_FILES, "magnetic_frenkel")
_CONF_LMP = os.path.join(_HTI_TEST_FILES, "frenkel", "conf.lmp")


class TestMagneticFrenkel(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def _make_tmp_job(self, extra_keys=None, remove_keys=None):
        """Return (tmpdir, cleanup_fn) with a valid in.json using absolute equi_conf."""
        import json
        import shutil
        import tempfile

        with open(os.path.join(_MAG_FRENKEL_DIR, "in.json")) as f:
            jdata = json.load(f)
        jdata["equi_conf"] = _CONF_LMP
        if remove_keys:
            for k in remove_keys:
                jdata.pop(k, None)
        if extra_keys:
            jdata.update(extra_keys)
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "in.json"), "w") as f:
            json.dump(jdata, f)
        return tmp, lambda: shutil.rmtree(tmp)

    def test_known_value(self):
        fe = magnetic_frenkel(_MAG_FRENKEL_DIR)
        self.assertAlmostEqual(fe, 0.07583270037843304)

    def test_include_spin_kinetic_lowers_fe(self):
        fe_no_kin = magnetic_frenkel(_MAG_FRENKEL_DIR)
        tmp, cleanup = self._make_tmp_job(extra_keys={"include_spin_kinetic": True})
        try:
            fe_kin = magnetic_frenkel(tmp)
        finally:
            cleanup()
        # spin kinetic de Broglie term ln(Λ_S_kin) is very negative → sfe more negative
        self.assertGreater(fe_no_kin, fe_kin)

    def test_spring_spin_k_alias_s_spring_k(self):
        fe_ref = magnetic_frenkel(_MAG_FRENKEL_DIR)
        tmp, cleanup = self._make_tmp_job(
            remove_keys=["spring_spin_k"],
            extra_keys={"s_spring_k": 0.12},
        )
        try:
            fe_alias = magnetic_frenkel(tmp)
        finally:
            cleanup()
        self.assertAlmostEqual(fe_ref, fe_alias, places=12)

    def test_spring_spin_k_alias_spring_k_spin(self):
        fe_ref = magnetic_frenkel(_MAG_FRENKEL_DIR)
        tmp, cleanup = self._make_tmp_job(
            remove_keys=["spring_spin_k"],
            extra_keys={"spring_k_spin": 0.12},
        )
        try:
            fe_alias = magnetic_frenkel(tmp)
        finally:
            cleanup()
        self.assertAlmostEqual(fe_ref, fe_alias, places=12)

    def test_missing_spring_spin_k_raises(self):
        # get_first_matched_key_from_dict raises KeyError when no key found;
        # the ValueError guard in magnetic_frenkel covers the None case.
        tmp, cleanup = self._make_tmp_job(
            remove_keys=["spring_spin_k", "s_spring_k", "spring_k_spin", "spin_spring_k"]
        )
        try:
            with self.assertRaises((ValueError, KeyError)):
                magnetic_frenkel(tmp)
        finally:
            cleanup()

    def test_invalid_spin_model_raises(self):
        tmp, cleanup = self._make_tmp_job(extra_keys={"spin_model": "bad_model"})
        try:
            with self.assertRaises(ValueError):
                magnetic_frenkel(tmp)
        finally:
            cleanup()

    def test_magnetic_frenkel_modulus_equals_lattice_plus_spin_ref_scalar(self):
        spin_ref = {"style": "spring", "k": 0.7, "s0": 2.0}
        tmp, cleanup = self._make_tmp_job(
            remove_keys=["spring_spin_k", "s_spring_k", "spring_k_spin", "spin_spring_k"],
            extra_keys={"spin_reference": "modulus", "spin_ref": spin_ref},
        )
        try:
            atom_numbs = [144]
            expected = frenkel(tmp) + spin_ref_fe_per_atom(
                400.0, atom_numbs, [True], spin_ref
            )
            self.assertAlmostEqual(magnetic_frenkel_modulus(tmp), expected, places=12)
        finally:
            cleanup()

    def test_magnetic_frenkel_modulus_accepts_type_lists(self):
        spin_ref = {"style": "spring", "k": [0.7], "s0": [2.0]}
        tmp, cleanup = self._make_tmp_job(
            remove_keys=["spring_spin_k", "s_spring_k", "spring_k_spin", "spin_spring_k"],
            extra_keys={"spin_reference": "modulus", "spin_ref": spin_ref},
        )
        try:
            atom_numbs = [144]
            expected = frenkel(tmp) + spin_ref_fe_per_atom(
                400.0, atom_numbs, [True], spin_ref
            )
            self.assertAlmostEqual(magnetic_frenkel_modulus(tmp), expected, places=12)
        finally:
            cleanup()

    def test_magnetic_frenkel_modulus_rejects_non_harmonic_style(self):
        tmp, cleanup = self._make_tmp_job(
            remove_keys=["spring_spin_k", "s_spring_k", "spring_k_spin", "spin_spring_k"],
            extra_keys={
                "spin_reference": "modulus",
                "spin_ref": {"style": "lj_core", "k": 0.7, "s0": 2.0},
            },
        )
        try:
            with self.assertRaises(ValueError):
                magnetic_frenkel_modulus(tmp)
        finally:
            cleanup()


if __name__ == "__main__":
    unittest.main()
