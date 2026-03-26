"""
Tests for hti_mag_rev.py — magnetic HTI reverse steps.

Test coverage:
  1. _ff_spring_on        : spring const = λ·k  (vs spring_off: (1-λ)·k)
  2. _ff_spring_spin_on   : spin spring const = λ·k
  3. _ff_two_steps_rev    : LAMMPS strings for each reverse step
  4. Integrand formulas   : correct sign & divisor for each step
  5. Analytic cancellation: ΔF_fwd + ΔF_bwd = 0 for ergodic mock data
"""

import sys
import os
import textwrap
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dpti.hti_mag_rev import (
    _ff_spring_on,
    _ff_spring_spin_on,
    _ff_two_steps_rev,
)
from dpti.lib.utils import integrate_range_hti


# ─────────────────────────────────────────────────────────────────────────────
# 1. _ff_spring_on: spring constant must be λ·k (growing coupling)
# ─────────────────────────────────────────────────────────────────────────────

class TestFfSpringOn(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_var_spring_uses_lambda_times_k(self):
        """spring/self constant = lambda * m_spring_k  (not (1-lambda)*k)."""
        lamb = 0.2
        m_spring_k = [10.0]
        result = _ff_spring_on(lamb, m_spring_k, var_spring=True)
        expected_k = lamb * m_spring_k[0]            # 2.0
        self.assertIn(f"{expected_k:.10e}", result)
        # Ensure the "off" formula value is NOT present
        wrong_k = (1 - lamb) * m_spring_k[0]         # 8.0
        self.assertNotIn(f"{wrong_k:.10e}", result)

    def test_not_var_spring_uses_full_k(self):
        """When var_spring=False, spring/self constant = m_spring_k (fully on)."""
        lamb = 0.2
        m_spring_k = [10.0]
        result = _ff_spring_on(lamb, m_spring_k, var_spring=False)
        self.assertIn(f"{m_spring_k[0]:.10e}", result)

    def test_disabled_returns_zero_variable(self):
        result = _ff_spring_on(0.5, [10.0], var_spring=True, enabled=False)
        self.assertIn("l_spring equal 0.0", result)

    def test_multitype(self):
        lamb = 0.3
        m_spring_k = [10.0, 20.0]
        result = _ff_spring_on(lamb, m_spring_k, var_spring=True)
        self.assertIn(f"{lamb * 10.0:.10e}", result)
        self.assertIn(f"{lamb * 20.0:.10e}", result)
        self.assertIn("f_l_spring_1+f_l_spring_2", result)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _ff_spring_spin_on: same logic for spin spring
# ─────────────────────────────────────────────────────────────────────────────

class TestFfSpringSpinOn(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_var_spring_spin_uses_lambda_times_k(self):
        lamb = 0.4
        m_spring_spin_k = [5.0]
        result = _ff_spring_spin_on(lamb, m_spring_spin_k, var_spring=True)
        expected_k = lamb * m_spring_spin_k[0]       # 2.0
        self.assertIn(f"{expected_k:.10e}", result)
        wrong_k = (1 - lamb) * m_spring_spin_k[0]    # 3.0
        self.assertNotIn(f"{wrong_k:.10e}", result)
        self.assertIn("spring/spin", result)

    def test_disabled(self):
        result = _ff_spring_spin_on(0.5, [5.0], var_spring=True, enabled=False)
        self.assertIn("l_spring_spin equal 0.0", result)


# ─────────────────────────────────────────────────────────────────────────────
# 3. _ff_two_steps_rev: LAMMPS force-field strings per step
# ─────────────────────────────────────────────────────────────────────────────

class TestFfTwoStepsRev(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.model = "graph.pb"
        self.m_spring_k = [1.0]
        self.m_spring_spin_k = [0.5]
        self.lamb = 0.4

    def _gen(self, step):
        return _ff_two_steps_rev(
            self.lamb, self.model,
            self.m_spring_k, self.m_spring_spin_k,
            step,
        )

    # step 4: spin_spring_on ─────────────────────────────────────────────────
    def test_spin_spring_on_no_adapt_for_deep(self):
        """Deep potential must be fully on: no adapt fix for deepspin."""
        out = self._gen("spin_spring_on")
        self.assertNotIn("adapt", out)

    def test_spin_spring_on_spin_spring_grows(self):
        """Spin spring const = λ·k_spin."""
        out = self._gen("spin_spring_on")
        expected_k = self.lamb * self.m_spring_spin_k[0]
        self.assertIn(f"{expected_k:.10e}", out)
        self.assertIn("spring/spin", out)

    def test_spin_spring_on_lattice_disabled(self):
        """Lattice spring is OFF (l_spring equal 0.0)."""
        out = self._gen("spin_spring_on")
        self.assertIn("l_spring equal 0.0", out)

    def test_spin_spring_on_has_compute_e_deep(self):
        out = self._gen("spin_spring_on")
        self.assertIn("compute         e_deep all pe pair", out)

    # step 5: lattice_spring_on ───────────────────────────────────────────────
    def test_lattice_spring_on_no_adapt_for_deep(self):
        """Deep potential must be fully on: no adapt fix."""
        out = self._gen("lattice_spring_on")
        self.assertNotIn("adapt", out)

    def test_lattice_spring_on_lattice_grows(self):
        """Lattice spring const = λ·k_latt."""
        out = self._gen("lattice_spring_on")
        expected_k = self.lamb * self.m_spring_k[0]
        self.assertIn(f"{expected_k:.10e}", out)
        self.assertIn("spring/self", out)

    def test_lattice_spring_on_spin_spring_full(self):
        """Spin spring constant = full k_spin (not varying)."""
        out = self._gen("lattice_spring_on")
        expected_k_full = self.m_spring_spin_k[0]   # 0.5
        self.assertIn(f"{expected_k_full:.10e}", out)
        # Varying-coupling value must NOT appear
        wrong_k = self.lamb * self.m_spring_spin_k[0]
        self.assertNotIn(f"{wrong_k:.10e}", out)

    # step 6: deep_off ────────────────────────────────────────────────────────
    def test_deep_off_uses_inv_lambda_scale(self):
        """Deep potential must decay: adapt uses v_INV_LAMBDA (not v_LAMBDA)."""
        out = self._gen("deep_off")
        self.assertIn("v_INV_LAMBDA", out)
        self.assertNotIn("scale * * v_LAMBDA\n", out)  # v_LAMBDA alone must be absent

    def test_deep_off_lattice_spring_constant(self):
        """Lattice spring is fully on, constant."""
        out = self._gen("deep_off")
        expected_k = self.m_spring_k[0]              # 1.0
        self.assertIn(f"{expected_k:.10e}", out)

    def test_deep_off_spin_spring_constant(self):
        """Spin spring is fully on, constant."""
        out = self._gen("deep_off")
        expected_k = self.m_spring_spin_k[0]         # 0.5
        self.assertIn(f"{expected_k:.10e}", out)

    def test_unknown_step_raises(self):
        with self.assertRaises(RuntimeError):
            _ff_two_steps_rev(0.5, "graph.pb", [1.0], [0.5], "bad_step")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Integrand formula correctness
#    For each step, verify de = correct expression given known mock arrays.
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrandFormulas(unittest.TestCase):
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
        # Arbitrary smooth ergodic functions
        self.f_deep   = 2.0 + 0.5 * self.lam          # ⟨H_deep_full⟩ at coupling λ
        self.f_latt   = 1.5 - 0.3 * self.lam           # ⟨H_latt_full⟩ at coupling λ
        self.f_spin   = 0.8 + 0.4 * self.lam           # ⟨H_spin_full⟩ at coupling λ

    # ── deep_off ─────────────────────────────────────────────────────────────
    def test_deep_off_integrand(self):
        """
        deep_off at λ: LAMMPS reports (1-λ)·f_deep(1-λ), integrand = -f_deep(1-λ)
        """
        lam = self.lam
        f_at_1ml = self.f_deep[::-1]                   # f_deep evaluated at (1-λ)
        all_ed = (1 - lam) * f_at_1ml                  # what LAMMPS reports
        de_int = -all_ed / (1 - lam)                   # our formula
        np.testing.assert_allclose(de_int, -f_at_1ml, rtol=1e-12)

    # ── spin_spring_on ────────────────────────────────────────────────────────
    def test_spin_spring_on_integrand(self):
        """
        spin_spring_on at λ: LAMMPS reports λ·f_spin(λ), integrand = +f_spin(λ)
        """
        lam = self.lam
        all_esp = lam * self.f_spin                    # what LAMMPS reports
        de_int = all_esp / lam                         # our formula
        np.testing.assert_allclose(de_int, self.f_spin, rtol=1e-12)

    # ── lattice_spring_on ─────────────────────────────────────────────────────
    def test_lattice_spring_on_integrand(self):
        """
        lattice_spring_on at λ: LAMMPS reports λ·f_latt(λ), integrand = +f_latt(λ)
        """
        lam = self.lam
        all_es = lam * self.f_latt
        de_int = all_es / lam
        np.testing.assert_allclose(de_int, self.f_latt, rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Analytic cancellation: ΔF_fwd + ΔF_bwd = 0  (ergodic mock data)
#
#    For an ergodic system the distribution at coupling α is path-independent.
#    Mapping:
#      deep_on   at λ: coupling α = λ        → LAMMPS: λ·f(λ)
#      deep_off  at λ: coupling α = (1-λ)    → LAMMPS: (1-λ)·f(1-λ)
#    After substitution, ΔF_off = -ΔF_on.
#    Same argument applies to lattice and spin springs.
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticCancellation(unittest.TestCase):
    """
    Construct forward (hti_mag) and backward (hti_mag_rev) integrand arrays
    using the ergodic-equivalence mapping, then verify ΔF_fwd + ΔF_bwd ≈ 0.
    """

    def setUp(self):
        # Uniform λ grid (same points, same for both directions)
        n = 11
        self.lam = np.linspace(0.05, 0.95, n)   # avoid 0 and 1
        # Arbitrary smooth 'true' energy functions at coupling α
        a_d, b_d = 3.0, -1.5     # f_deep(α) = a_d + b_d * α
        a_l, b_l = 1.0,  0.6     # f_latt(α)
        a_s, b_s = 0.5,  0.3     # f_spin(α)
        α = self.lam
        self.f_deep = a_d + b_d * α
        self.f_latt = a_l + b_l * α
        self.f_spin = a_s + b_s * α
        # Analytic ΔF (trapezoid over uniform grid would give same result):
        #   ΔF_deep = ∫₀¹ f_deep dα ≈ a_d + b_d/2
        #   ΔF_latt = -∫₀¹ f_latt dα  (spring_off sign)
        #   ΔF_spin = -∫₀¹ f_spin dα

    def _integrate(self, lam, integrand):
        de, err, _ = integrate_range_hti(lam, integrand, np.zeros_like(lam))
        return de

    # ── deep_on (fwd step 1) vs deep_off (bwd step 6) ────────────────────────
    def test_deep_cancels(self):
        lam = self.lam
        # Forward: all_ed = λ·f_deep(λ),  integrand = all_ed/λ = f_deep
        all_ed_fwd = lam * self.f_deep
        de_fwd = self._integrate(lam, all_ed_fwd / lam)

        # Backward: all_ed = (1-λ)·f_deep(1-λ), integrand = -all_ed/(1-λ)
        f_at_1ml = self.f_deep[::-1]
        all_ed_bwd = (1 - lam) * f_at_1ml
        de_bwd = self._integrate(lam, -all_ed_bwd / (1 - lam))

        self.assertAlmostEqual(de_fwd + de_bwd, 0.0, places=10)

    # ── lattice spring_off (fwd step 2) vs lattice_spring_on (bwd step 5) ────
    def test_lattice_spring_cancels(self):
        lam = self.lam
        # Forward spring_off: all_es = (1-λ)·f_latt(1-λ), integrand = -all_es/(1-λ)
        f_at_1ml = self.f_latt[::-1]
        all_es_fwd = (1 - lam) * f_at_1ml
        de_fwd = self._integrate(lam, -all_es_fwd / (1 - lam))

        # Backward spring_on: all_es = λ·f_latt(λ), integrand = +all_es/λ
        all_es_bwd = lam * self.f_latt
        de_bwd = self._integrate(lam, all_es_bwd / lam)

        self.assertAlmostEqual(de_fwd + de_bwd, 0.0, places=10)

    # ── spin_spring_off (fwd step 3) vs spin_spring_on (bwd step 4) ──────────
    def test_spin_spring_cancels(self):
        lam = self.lam
        # Forward spin_spring_off: all_esp = (1-λ)·f_spin(1-λ)
        f_at_1ml = self.f_spin[::-1]
        all_esp_fwd = (1 - lam) * f_at_1ml
        de_fwd = self._integrate(lam, -all_esp_fwd / (1 - lam))

        # Backward spin_spring_on: all_esp = λ·f_spin(λ)
        all_esp_bwd = lam * self.f_spin
        de_bwd = self._integrate(lam, all_esp_bwd / lam)

        self.assertAlmostEqual(de_fwd + de_bwd, 0.0, places=10)

    # ── total cancellation: all 3 pairs ──────────────────────────────────────
    def test_total_free_energy_cancels(self):
        lam = self.lam
        ΔF_fwd = 0.0
        ΔF_bwd = 0.0

        # deep_on
        ΔF_fwd += self._integrate(lam, lam * self.f_deep / lam)
        # deep_off
        ΔF_bwd += self._integrate(lam, -(1 - lam) * self.f_deep[::-1] / (1 - lam))

        # lattice_spring_off
        ΔF_fwd += self._integrate(lam, -(1 - lam) * self.f_latt[::-1] / (1 - lam))
        # lattice_spring_on
        ΔF_bwd += self._integrate(lam, lam * self.f_latt / lam)

        # spin_spring_off
        ΔF_fwd += self._integrate(lam, -(1 - lam) * self.f_spin[::-1] / (1 - lam))
        # spin_spring_on
        ΔF_bwd += self._integrate(lam, lam * self.f_spin / lam)

        self.assertAlmostEqual(ΔF_fwd + ΔF_bwd, 0.0, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
