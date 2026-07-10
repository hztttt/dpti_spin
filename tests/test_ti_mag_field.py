import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dpti.ti_mag import (  # noqa: E402
    LAMMPS_G_FACTOR,
    LAMMPS_MUB_EV_PER_T,
    _field_unit_params,
    _thermo_inte,
    make_tasks,
)


def _write_spin_conf(path):
    with open(path, "w") as fp:
        fp.write(
            """LAMMPS data file for ti_mag field path test

1 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi
0.0 0.0 0.0 xy xz yz

Atoms # spin

1 1 5.0 5.0 5.0 1.0 0.0 0.0 2.0
"""
        )


class TestTiMagFieldPath(unittest.TestCase):
    def test_field_unit_evuB_converts_to_lammps_tesla(self):
        field = 0.02
        lmp_field, deriv_scale = _field_unit_params(field, "eV/uB")
        self.assertAlmostEqual(
            lmp_field,
            field / (LAMMPS_G_FACTOR * LAMMPS_MUB_EV_PER_T),
            places=12,
        )
        self.assertEqual(deriv_scale, 1.0)

    def test_field_unit_tesla_conjugate_scale(self):
        lmp_field, deriv_scale = _field_unit_params(10.0, "tesla")
        self.assertEqual(lmp_field, 10.0)
        self.assertAlmostEqual(
            deriv_scale,
            LAMMPS_G_FACTOR * LAMMPS_MUB_EV_PER_T,
            places=15,
        )

    def test_make_tasks_generates_field_path_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf = os.path.join(tmp, "conf.lmp")
            _write_spin_conf(conf)
            job = os.path.join(tmp, "field_job")
            make_tasks(
                job,
                {
                    "equi_conf": conf,
                    "mass_map": [1.0],
                    "spin_mass": 1.0,
                    "ens": "nvt-langevin",
                    "path": "b",
                    "field_seq": [0.0, 0.02],
                    "field_unit": "eV/uB",
                    "field_direction": [0.0, 0.0, 2.0],
                    "temp": 300.0,
                    "tau_t": 0.1,
                    "nsteps": 10,
                    "timestep": 0.001,
                    "thermo_freq": 1,
                    "dump_freq": 10,
                    "stat_skip": 0,
                    "stat_bsize": 1,
                    "harmonic": True,
                    "spring_k": 1.0,
                    "spring_spin_k": 1.0,
                },
            )

            with open(os.path.join(job, "task.000001", "in.lammps")) as fp:
                lammps_input = fp.read()

            self.assertIn("fix             zee all precession/spin zeeman", lammps_input)
            self.assertIn("fix_modify      zee energy yes", lammps_input)
            self.assertIn("variable        field_integrand equal -${FIELD_DERIV_SCALE}*c_field_moment", lammps_input)
            self.assertIn("f_zee v_field_integrand c_field_moment", lammps_input)
            self.assertIn("variable        FIELD_NZ        equal 1.0000000000000000e+00", lammps_input)

    def test_field_path_integrates_conjugate_variable(self):
        fields = np.array([0.0, 0.5, 1.0])
        integrand = -2.0 * fields
        errs = np.zeros_like(fields)
        temps, press, out_fields, fe, fe_err, fe_sys_err = _thermo_inte(
            {"path": "b", "ens": "nvt", "temp": 300.0},
            Eo=1.0,
            Eo_err=0.0,
            all_t=fields,
            integrand=integrand,
            integrand_err=errs,
            scheme="trapezoidal",
        )
        np.testing.assert_allclose(temps, [300.0, 300.0, 300.0])
        self.assertEqual(press.size, 0)
        np.testing.assert_allclose(out_fields.astype(float), fields)
        np.testing.assert_allclose(fe, [1.0, 0.75, 0.0])
        np.testing.assert_allclose(fe_err, 0.0)
        np.testing.assert_allclose(fe_sys_err, 0.0)


if __name__ == "__main__":
    unittest.main()
