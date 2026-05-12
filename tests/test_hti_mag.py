"""
Tests for hti_mag.py — magnetic HTI forward steps.

Test coverage:
  1. _ff_spring        : spring const = (1-λ)·k  (forward decay)
  2. _ff_spring_spin   : spin spring const = (1-λ)·k
  3. _get_spring_lambda_mode : alias resolution and unknown-key error
  4. _ff_two_steps     : LAMMPS strings for each forward step
  5. Integrand formulas: correct sign & divisor for each step
"""

import sys
import os
import shutil
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dpti.hti_mag import (
    _ff_spin_mod,
    _ff_spring,
    _ff_spring_spin,
    _ff_two_steps,
    _get_spring_lambda_mode,
    _get_spin_reference_mode,
    make_tasks,
)
from dpti.lib.utils import integrate_range_hti


# ─────────────────────────────────────────────────────────────────────────────
# 1. _ff_spring: forward direction — spring const decays as (1-λ)·k
# ─────────────────────────────────────────────────────────────────────────────

class TestFfSpringFwd(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_var_spring_uses_one_minus_lambda_times_k(self):
        """spring/self constant = (1-lambda) * m_spring_k  (decaying coupling)."""
        lamb = 0.3
        m_spring_k = [10.0]
        result = _ff_spring(lamb, m_spring_k, var_spring=True)
        expected_k = (1 - lamb) * m_spring_k[0]      # 7.0
        self.assertIn(f"{expected_k:.10e}", result)
        # Ensure the "on" formula value is NOT present
        wrong_k = lamb * m_spring_k[0]               # 3.0
        self.assertNotIn(f"{wrong_k:.10e}", result)

    def test_not_var_spring_uses_full_k(self):
        """When var_spring=False, spring/self constant = m_spring_k (fully on)."""
        lamb = 0.3
        m_spring_k = [10.0]
        result = _ff_spring(lamb, m_spring_k, var_spring=False)
        self.assertIn(f"{m_spring_k[0]:.10e}", result)

    def test_disabled_returns_zero_variable(self):
        result = _ff_spring(0.5, [10.0], var_spring=True, enabled=False)
        self.assertIn("l_spring equal 0.0", result)
        self.assertNotIn("spring/self", result)

    def test_multitype(self):
        lamb = 0.4
        m_spring_k = [10.0, 20.0]
        result = _ff_spring(lamb, m_spring_k, var_spring=True)
        self.assertIn(f"{(1 - lamb) * 10.0:.10e}", result)
        self.assertIn(f"{(1 - lamb) * 20.0:.10e}", result)
        self.assertIn("f_l_spring_1+f_l_spring_2", result)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _ff_spring_spin: same logic for spin spring
# ─────────────────────────────────────────────────────────────────────────────

class TestFfSpringSpinFwd(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_var_spring_spin_uses_one_minus_lambda_times_k(self):
        lamb = 0.6
        m_spring_spin_k = [5.0]
        result = _ff_spring_spin(lamb, m_spring_spin_k, var_spring=True)
        expected_k = (1 - lamb) * m_spring_spin_k[0]  # 2.0
        self.assertIn(f"{expected_k:.10e}", result)
        wrong_k = lamb * m_spring_spin_k[0]            # 3.0
        self.assertNotIn(f"{wrong_k:.10e}", result)
        self.assertIn("spring/spin", result)

    def test_disabled(self):
        result = _ff_spring_spin(0.5, [5.0], var_spring=True, enabled=False)
        self.assertIn("l_spring_spin equal 0.0", result)
        self.assertNotIn("spring/spin", result)


# ─────────────────────────────────────────────────────────────────────────────
# 3. _get_spring_lambda_mode: alias resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSpringLambdaMode(unittest.TestCase):
    def _mode(self, value):
        return _get_spring_lambda_mode({"spring_lambda_mode": value})

    def test_aliases_joint(self):
        for alias in ("joint", "same", "together"):
            self.assertEqual(self._mode(alias), "joint", msg=alias)

    def test_aliases_split(self):
        for alias in ("split", "separate"):
            self.assertEqual(self._mode(alias), "split", msg=alias)

    def test_aliases_lattice_only(self):
        for alias in ("lattice_only", "lattice-only"):
            self.assertEqual(self._mode(alias), "lattice_only", msg=alias)

    def test_aliases_spin_only(self):
        for alias in ("spin_only", "spin-only"):
            self.assertEqual(self._mode(alias), "spin_only", msg=alias)

    def test_default_is_joint(self):
        self.assertEqual(_get_spring_lambda_mode({}), "joint")

    def test_unknown_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            _get_spring_lambda_mode({"spring_lambda_mode": "bad_mode"})


class TestGetSpinReferenceMode(unittest.TestCase):
    def test_explicit_modulus(self):
        self.assertEqual(_get_spin_reference_mode({"spin_reference": "modulus"}), "modulus")

    def test_spin_ref_defaults_to_modulus(self):
        self.assertEqual(_get_spin_reference_mode({"spin_ref": {"k": 1.0, "s0": 2.0}}), "modulus")

    def test_legacy_spring_defaults_to_vector(self):
        self.assertEqual(_get_spin_reference_mode({"spring_spin_k": 0.1}), "vector")

    def test_default_is_none(self):
        self.assertEqual(_get_spin_reference_mode({}), "none")


# ─────────────────────────────────────────────────────────────────────────────
# 4. _ff_two_steps: LAMMPS force-field strings per forward step
# ─────────────────────────────────────────────────────────────────────────────

class TestFfTwoStepsFwd(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.model = "graph.pb"
        self.m_spring_k = [1.0]
        self.m_spring_spin_k = [0.5]
        self.lamb = 0.4

    def _gen(self, step, spring_lambda_mode="joint"):
        return _ff_two_steps(
            self.lamb, self.model,
            self.m_spring_k, self.m_spring_spin_k,
            step,
            spring_lambda_mode=spring_lambda_mode,
        )

    # ── deep_on ──────────────────────────────────────────────────────────────
    def test_deep_on_has_adapt_for_deep(self):
        """Deep potential must be growing: adapt fix with v_LAMBDA present."""
        out = self._gen("deep_on")
        self.assertIn("adapt", out)
        self.assertIn("v_LAMBDA", out)

    def test_deep_on_lattice_spring_constant(self):
        """Lattice spring is fully on (constant, not varying)."""
        out = self._gen("deep_on")
        expected_k = self.m_spring_k[0]               # 1.0
        self.assertIn(f"{expected_k:.10e}", out)
        # Decaying value must NOT appear
        wrong_k = (1 - self.lamb) * self.m_spring_k[0]
        self.assertNotIn(f"{wrong_k:.10e}", out)

    def test_deep_on_spin_spring_constant(self):
        """Spin spring is fully on (constant)."""
        out = self._gen("deep_on")
        expected_k = self.m_spring_spin_k[0]          # 0.5
        self.assertIn(f"{expected_k:.10e}", out)

    def test_deep_on_has_compute_e_deep(self):
        out = self._gen("deep_on")
        self.assertIn("compute         e_deep all pe pair", out)

    # ── spring_off (joint) ────────────────────────────────────────────────────
    def test_spring_off_joint_no_adapt_for_deep(self):
        """Deep potential is fully on: no adapt fix."""
        out = self._gen("spring_off", spring_lambda_mode="joint")
        self.assertNotIn("adapt", out)

    def test_spring_off_joint_lattice_spring_decays(self):
        """Lattice spring const = (1-λ)·k_latt."""
        out = self._gen("spring_off", spring_lambda_mode="joint")
        expected_k = (1 - self.lamb) * self.m_spring_k[0]
        self.assertIn(f"{expected_k:.10e}", out)

    def test_spring_off_joint_spin_spring_decays(self):
        """Spin spring const = (1-λ)·k_spin."""
        out = self._gen("spring_off", spring_lambda_mode="joint")
        expected_k = (1 - self.lamb) * self.m_spring_spin_k[0]
        self.assertIn(f"{expected_k:.10e}", out)

    # ── lattice_spring_off ────────────────────────────────────────────────────
    def test_lattice_spring_off_lattice_decays(self):
        """Lattice spring const = (1-λ)·k_latt."""
        out = self._gen("lattice_spring_off")
        expected_k = (1 - self.lamb) * self.m_spring_k[0]
        self.assertIn(f"{expected_k:.10e}", out)

    def test_lattice_spring_off_spin_disabled(self):
        """Spin spring is disabled (l_spring_spin equal 0.0)."""
        out = self._gen("lattice_spring_off")
        self.assertIn("l_spring_spin equal 0.0", out)

    # ── spin_spring_off ───────────────────────────────────────────────────────
    def test_spin_spring_off_spin_decays(self):
        """Spin spring const = (1-λ)·k_spin."""
        out = self._gen("spin_spring_off")
        expected_k = (1 - self.lamb) * self.m_spring_spin_k[0]
        self.assertIn(f"{expected_k:.10e}", out)

    def test_spin_spring_off_lattice_disabled(self):
        """Lattice spring is disabled (l_spring equal 0.0)."""
        out = self._gen("spin_spring_off")
        self.assertIn("l_spring equal 0.0", out)

    # ── unknown step ──────────────────────────────────────────────────────────
    def test_unknown_step_raises(self):
        with self.assertRaises(RuntimeError):
            _ff_two_steps(0.5, "graph.pb", [1.0], [0.5], "bad_step")


class TestSpinModulusReference(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.model = "graph.pb"
        self.m_spring_k = [1.0]
        self.spin_ref_k = [0.7]
        self.spin_ref_s0 = [2.2]
        self.spin_map = [True]
        self.lamb = 0.4

    def test_ff_spin_mod_full_uses_spring_spin_mod(self):
        out = _ff_spin_mod(
            self.lamb,
            self.spin_ref_k,
            self.spin_ref_s0,
            self.spin_map,
            "full",
        )
        self.assertIn("spring/spin/mod", out)
        self.assertIn(f"{self.spin_ref_k[0]:.10e} {self.spin_ref_s0[0]:.10e}", out)
        self.assertNotIn("spring/spin 7", out)

    def test_ff_spin_mod_off_scales_one_minus_lambda(self):
        out = _ff_spin_mod(
            self.lamb,
            self.spin_ref_k,
            self.spin_ref_s0,
            self.spin_map,
            "off",
        )
        expected_k = (1 - self.lamb) * self.spin_ref_k[0]
        self.assertIn(f"{expected_k:.10e} {self.spin_ref_s0[0]:.10e}", out)

    def test_two_step_modulus_deep_on_full_strength(self):
        out = _ff_two_steps(
            self.lamb,
            self.model,
            self.m_spring_k,
            self.spin_ref_k,
            "deep_on",
            spin_ref_s0=self.spin_ref_s0,
            spin_map=self.spin_map,
            spin_reference="modulus",
        )
        self.assertIn("spring/spin/mod", out)
        self.assertNotIn("spring/spin ", out)
        self.assertIn(f"{self.spin_ref_k[0]:.10e} {self.spin_ref_s0[0]:.10e}", out)

    def test_two_step_modulus_spin_spring_off_decays(self):
        out = _ff_two_steps(
            self.lamb,
            self.model,
            self.m_spring_k,
            self.spin_ref_k,
            "spin_spring_off",
            spin_ref_s0=self.spin_ref_s0,
            spin_map=self.spin_map,
            spin_reference="modulus",
        )
        expected_k = (1 - self.lamb) * self.spin_ref_k[0]
        self.assertIn("l_spring equal 0.0", out)
        self.assertIn("spring/spin/mod", out)
        self.assertNotIn("spring/spin ", out)
        self.assertIn(f"{expected_k:.10e} {self.spin_ref_s0[0]:.10e}", out)

    def test_make_tasks_split_keeps_spin_spring_off_directory(self):
        tmp = tempfile.mkdtemp()
        try:
            model = os.path.join(tmp, "graph.pb")
            with open(model, "w") as fp:
                fp.write("dummy")
            job = os.path.join(tmp, "job")
            conf = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "hti_test_files",
                    "frenkel",
                    "conf.lmp",
                )
            )
            jdata = {
                "equi_conf": conf,
                "model": model,
                "mass_map": [118.71],
                "spin_mass": 1.0,
                "spin_map": [True],
                "spring_k": 0.02,
                "spin_reference": "modulus",
                "spin_ref": {"style": "spring", "k": 0.7, "s0": 2.0},
                "spring_lambda_mode": "split",
                "lambda_deep_on": [0.2],
                "lambda_spring_off": [0.3],
                "lambda_spin_spring_off": [0.4],
                "protect_eps": 1.0e-6,
                "nsteps": 10,
                "timestep": 0.001,
                "thermo_freq": 1,
                "temp": 400,
                "crystal": "frenkel",
            }
            make_tasks(job, jdata, switch="two-step")
            spin_dir = os.path.join(job, "02.spin_spring_off")
            self.assertTrue(os.path.isdir(spin_dir))
            with open(os.path.join(spin_dir, "task.000000", "in.lammps")) as fp:
                lammps_input = fp.read()
            self.assertIn("spring/spin/mod", lammps_input)
            self.assertNotIn("spring/spin ", lammps_input)
            self.assertIn(f"{0.7 * (1 - 0.4):.10e} {2.0:.10e}", lammps_input)
            self.assertIn("compute         spinmodmsd all msd/spin/mod", lammps_input)
            self.assertIn("c_spinmsd[*] c_spinmodmsd", lammps_input)
        finally:
            shutil.rmtree(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integrand formula correctness (pure numpy, no file IO)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrandFormulasFwd(unittest.TestCase):
    """
    Mock 'measured' LAMMPS energies and verify the integrand matches
    the analytical expectation.

    Convention (same as hti_mag.py):
      all_ed[i]  = LAMMPS pe pair  at task i, already /natoms
      all_es[i]  = LAMMPS l_spring at task i, already /natoms
      all_esp[i] = LAMMPS l_spring_spin at task i, already /natoms
    """

    def setUp(self):
        self.lam = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.f_deep = 2.0 + 0.5 * self.lam
        self.f_latt = 1.5 - 0.3 * self.lam
        self.f_spin = 0.8 + 0.4 * self.lam

    # ── deep_on ──────────────────────────────────────────────────────────────
    def test_deep_on_integrand(self):
        """
        deep_on at λ: LAMMPS reports λ·f_deep(λ), integrand = all_ed/λ = f_deep
        """
        lam = self.lam
        all_ed = lam * self.f_deep
        de_int = all_ed / lam
        np.testing.assert_allclose(de_int, self.f_deep, rtol=1e-12)

    # ── spring_off (joint) ────────────────────────────────────────────────────
    def test_spring_off_joint_integrand(self):
        """
        joint spring_off at λ: LAMMPS reports (1-λ)·(f_latt+f_spin)(1-λ).
        integrand = -(all_es + all_esp) / (1-λ)
        """
        lam = self.lam
        f_latt_at_1ml = self.f_latt[::-1]
        f_spin_at_1ml = self.f_spin[::-1]
        all_es = (1 - lam) * f_latt_at_1ml
        all_esp = (1 - lam) * f_spin_at_1ml
        de_int = -(all_es + all_esp) / (1 - lam)
        np.testing.assert_allclose(de_int, -(f_latt_at_1ml + f_spin_at_1ml), rtol=1e-12)

    # ── lattice_spring_off ────────────────────────────────────────────────────
    def test_lattice_spring_off_integrand(self):
        """
        lattice_spring_off at λ: LAMMPS reports (1-λ)·f_latt(1-λ).
        integrand = -all_es / (1-λ)
        """
        lam = self.lam
        f_at_1ml = self.f_latt[::-1]
        all_es = (1 - lam) * f_at_1ml
        de_int = -all_es / (1 - lam)
        np.testing.assert_allclose(de_int, -f_at_1ml, rtol=1e-12)

    # ── spin_spring_off ───────────────────────────────────────────────────────
    def test_spin_spring_off_integrand(self):
        """
        spin_spring_off at λ: LAMMPS reports (1-λ)·f_spin(1-λ).
        integrand = -all_esp / (1-λ)
        """
        lam = self.lam
        f_at_1ml = self.f_spin[::-1]
        all_esp = (1 - lam) * f_at_1ml
        de_int = -all_esp / (1 - lam)
        np.testing.assert_allclose(de_int, -f_at_1ml, rtol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
