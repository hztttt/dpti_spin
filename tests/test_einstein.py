import os
import unittest

import numpy as np
import scipy.constants as pc
from scipy.integrate import quad

# from numpy.testing import assert_almost_equal
from dpti.einstein import (
    free_energy,
    frenkel,
    ideal_gas_fe,
    spin_ref_fe_per_atom,
    spin_ref_partition,
)

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

if __name__ == "__main__":
    unittest.main()
