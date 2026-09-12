####################################################################################################
#                                     test_gnl_field.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-12                                                                              #
#                                                                                                  #
# Purpose: The per-harmonic-term GNL basis/coefficient decomposition and GradientNonlinearityPhase #
#          - parity against gradunwarp's own aggregate output, stochastic reproducibility, and     #
#          numerical stability through order 9. Needs gradunwarp (skipped entirely otherwise).      #
#                                                                                                  #
####################################################################################################

"""
Tests for augmentrum.physics.gnl_field.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

try:
    import gradunwarp   # noqa: F401
    GRADUNWARP_AVAILABLE = True
except ImportError:
    GRADUNWARP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRADUNWARP_AVAILABLE, reason="gradunwarp not installed")

if GRADUNWARP_AVAILABLE:
    from augmentrum.physics.gnl_field import (
        term_index, basis_matrix, coefficient_matrix, sample_coefficients,
        perturb_coefficients_by_order, GradientNonlinearityPhase,
    )
    from augmentrum.physics.gradient_nonlinearity import load_coefficients


#**************************************************************************************************#
#                                       Class TestTermIndex                                         #
#**************************************************************************************************#
class TestTermIndex:
    """(max_order + 1)**2 real terms; no degenerate m=0 sine term."""

    @pytest.mark.parametrize("order", [0, 1, 3, 9])
    def test_term_count_matches_formula(self, order):
        terms = term_index(order)
        assert len(terms) == (order + 1) ** 2

    def test_no_m_zero_beta_term(self):
        assert all(not is_beta for (n, m, is_beta) in term_index(5) if m == 0)


#**************************************************************************************************#
#                                Class TestParityWithGradunwarp                                     #
#**************************************************************************************************#
class TestParityWithGradunwarp:
    """Requirement 4: coefficient loading + Phi@C field reconstruction must
    match gradunwarp's own aggregate eval_spherical_harmonics exactly."""

    def test_reconstructed_displacement_matches_aggregate(self):
        import gradunwarp.core.unwarp_resample as ur
        from gradunwarp.core.utils import CoordsVector as CV

        coeffs = load_coefficients()   # warns; the bundled dummy fixture
        rng = np.random.default_rng(0)
        pts_m = rng.normal(scale=0.08, size=(12, 3))

        Phi, terms = basis_matrix(pts_m, coeffs.R0_m, max_order=9)
        C = coefficient_matrix(coeffs, terms)
        reconstructed_m = Phi @ C   # (N, 3) x,y,z displacement, meters

        pts_mm = pts_m * 1000.0
        dv, _ = ur.eval_spherical_harmonics(
            coeffs, 'siemens', CV(x=pts_mm[:, 0].copy(), y=pts_mm[:, 1].copy(), z=pts_mm[:, 2].copy()))
        expected_m = np.stack([dv.x, dv.y, dv.z], axis=-1) / 1000.0

        np.testing.assert_allclose(reconstructed_m, expected_m, atol=1e-9)


#**************************************************************************************************#
#                                Class TestSingleKnownHarmonic                                      #
#**************************************************************************************************#
class TestSingleKnownHarmonic:
    """Requirement 3: a single nonzero harmonic's forward-encoded phase must
    match an independently hand-computed -2*pi*k.Delta_r(r) for that term."""

    def test_single_term_matches_hand_computation(self):
        from gradunwarp.core.coeffs import Coeffs

        n = 4
        zeros = lambda: np.zeros((n, n))
        alpha_z = zeros()
        alpha_z[3, 0] = 0.02   # one nonzero term: (n=3, m=0, alpha, axis=z)
        coeffs = Coeffs(alpha_x=zeros(), alpha_y=zeros(), alpha_z=alpha_z,
                       beta_x=zeros(), beta_y=zeros(), beta_z=zeros(), R0_m=0.25)

        rng = np.random.default_rng(1)
        positions_m = rng.normal(scale=0.06, size=(8, 3))
        T = 10
        k_traj = rng.normal(scale=15.0, size=(3, T))

        Phi, terms = basis_matrix(positions_m, coeffs.R0_m, max_order=3)
        term_idx = terms.index((3, 0, False))
        # Hand computation: only the z-axis column contributes, only at this one term.
        delta_r_z = alpha_z[3, 0] * Phi[:, term_idx]                    # (N,) meters
        expected_phase = -2.0 * np.pi * np.outer(delta_r_z, k_traj[2])  # (N, T)

        import torch
        term = GradientNonlinearityPhase(coeffs, k_traj, max_order=3)
        positions_t = torch.tensor(positions_m, dtype=torch.float32)
        t = torch.arange(T, dtype=torch.float32) * 1e-5
        phase = term.phase(positions_t, t).numpy()

        np.testing.assert_allclose(phase, expected_phase, atol=1e-4, rtol=1e-4)


#**************************************************************************************************#
#                                  Class TestStochasticGeneration                                   #
#**************************************************************************************************#
class TestStochasticGeneration:
    """Requirements 5, 6, 7: seeded reproducibility, seed variation, order-9 stability."""

    def test_fixed_seed_reproducible(self):
        c1 = sample_coefficients(max_order=9, seed=42)
        c2 = sample_coefficients(max_order=9, seed=42)
        for field in ('alpha_x', 'alpha_y', 'alpha_z', 'beta_x', 'beta_y', 'beta_z'):
            np.testing.assert_array_equal(getattr(c1, field), getattr(c2, field))

    def test_different_seeds_differ_but_decay_with_order(self):
        c1 = sample_coefficients(max_order=9, seed=1, base_std=1.0, decay=0.5,
                                include_linear_terms=True)
        c2 = sample_coefficients(max_order=9, seed=2, base_std=1.0, decay=0.5,
                                include_linear_terms=True)
        assert not np.array_equal(c1.alpha_x, c2.alpha_x)

        # Magnitude should decay with order (statistically, averaged over both draws).
        combined = np.abs(c1.alpha_z) + np.abs(c2.alpha_z)
        low_order = combined[0:2, :].mean()
        high_order = combined[7:9, :].mean()
        assert high_order < low_order

    def test_include_linear_terms_false_zeroes_order_0_and_1(self):
        c = sample_coefficients(max_order=5, seed=0, include_linear_terms=False)
        for field in ('alpha_x', 'alpha_y', 'alpha_z', 'beta_x', 'beta_y', 'beta_z'):
            arr = getattr(c, field)
            assert np.all(arr[0] == 0.0) and np.all(arr[1] == 0.0)

    def test_order_9_is_finite_and_stable(self):
        c = sample_coefficients(max_order=9, seed=0, include_linear_terms=True)
        rng = np.random.default_rng(3)
        positions_m = rng.normal(scale=0.1, size=(30, 3))
        Phi, terms = basis_matrix(positions_m, c.R0_m, max_order=9)
        C = coefficient_matrix(c, terms)
        field = Phi @ C
        assert np.all(np.isfinite(field))
        assert len(terms) == 100


#**************************************************************************************************#
#                              Class TestPerturbCoefficientsByOrder                                  #
#**************************************************************************************************#
class TestPerturbCoefficientsByOrder:
    """measured_stochastic: only requested orders change; reproducible by seed."""

    def test_only_requested_orders_change(self):
        base = load_coefficients()
        perturbed = perturb_coefficients_by_order(base, std_by_order={5: 1e-4}, seed=0)

        # Order 5 changed (or stayed zero if the base file has no order-5 term
        # and noise happened to be negligible - use a large std to guarantee change).
        perturbed_loud = perturb_coefficients_by_order(base, std_by_order={5: 10.0}, seed=0)
        assert not np.array_equal(perturbed_loud.alpha_z[5], base.alpha_z[5])

        # Orders not mentioned must be untouched exactly.
        for order in (0, 1, 2, 3, 4, 6, 7, 8, 9):
            if order < base.alpha_z.shape[0]:
                np.testing.assert_array_equal(perturbed.alpha_z[order], base.alpha_z[order])

    def test_seeded_reproducibility(self):
        base = load_coefficients()
        p1 = perturb_coefficients_by_order(base, std_by_order={3: 1e-3}, seed=7)
        p2 = perturb_coefficients_by_order(base, std_by_order={3: 1e-3}, seed=7)
        np.testing.assert_array_equal(p1.alpha_z, p2.alpha_z)
