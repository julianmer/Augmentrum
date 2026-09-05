####################################################################################################
#                                 test_concomitant_field.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-05                                                                              #
#                                                                                                  #
# Purpose: Dependency-free checks of the concomitant-field formula against its well-known special  #
#          case and basic symmetries - no GIRF-Sim/.seq file needed.                                #
#                                                                                                  #
####################################################################################################

"""
Tests for augmentrum.physics.concomitant_field.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.physics.concomitant_field import (
    ConcomitantFieldPhase,
    concomitant_field_basis,
    concomitant_field_coefficients,
)

DT = 1e-5
T = 200
B0 = 3.0


#**************************************************************************************************#
#                                    Class TestGzOnlySpecialCase                                    #
#**************************************************************************************************#
#                                                                                                  #
# Gx=Gy=0 must reduce to the textbook Gz^2*(x^2+y^2)/(8*B0) result.                                 #
#                                                                                                  #
#**************************************************************************************************#
class TestGzOnlySpecialCase:
    """Gx=Gy=0 must reduce to the textbook Gz^2*(x^2+y^2)/(8*B0) result."""

    def test_z_only_gradient_matches_textbook_formula(self):
        gz_amplitude = 0.02   # T/m
        g = np.zeros((3, T))
        g[2] = gz_amplitude

        coeffs = concomitant_field_coefficients(g, DT, B0)
        positions = np.array([[0.05, -0.03, 0.0], [0.1, 0.1, 0.0]])
        basis = concomitant_field_basis(positions)
        phase = basis @ coeffs   # (N, T)

        gamma = 42.57747892e6
        # cumsum(...)[n] already includes one full dt of accumulation, i.e. it is
        # the phase at t=(n+1)*dt, not n*dt - matching GIRFGlobalPhase's own
        # "cumsum(c0)*dt" convention elsewhere in this codebase.
        t = (np.arange(T) + 1) * DT
        expected_freq = gamma * gz_amplitude ** 2 / (8.0 * B0)   # Hz, constant in time
        expected_phase = 2.0 * np.pi * expected_freq * t
        r2 = positions[:, 0] ** 2 + positions[:, 1] ** 2

        np.testing.assert_allclose(phase, expected_phase[None, :] * r2[:, None], rtol=1e-10)

    def test_no_gradient_gives_zero_phase(self):
        g = np.zeros((3, T))
        coeffs = concomitant_field_coefficients(g, DT, B0)
        assert np.allclose(coeffs, 0.0)


#**************************************************************************************************#
#                                    Class TestCrossTermSymmetry                                    #
#**************************************************************************************************#
#                                                                                                  #
# Swapping Gx<->Gy and x<->y must swap the xz/yz cross terms and leave z2/r2 untouched.             #
#                                                                                                  #
#**************************************************************************************************#
class TestCrossTermSymmetry:
    """Swapping Gx<->Gy and x<->y must swap the xz/yz cross terms and leave z2/r2 untouched."""

    def test_swapping_x_and_y_axes_swaps_cross_terms(self):
        rng = np.random.default_rng(0)
        g = rng.normal(scale=0.01, size=(3, T))
        coeffs = concomitant_field_coefficients(g, DT, B0)   # (4, T): z2, r2, xz, yz

        g_swapped = g[[1, 0, 2]]   # Gx<->Gy
        coeffs_swapped = concomitant_field_coefficients(g_swapped, DT, B0)

        np.testing.assert_allclose(coeffs_swapped[0], coeffs[0])   # z2 unchanged
        np.testing.assert_allclose(coeffs_swapped[1], coeffs[1])   # r2 (x^2+y^2) unchanged
        np.testing.assert_allclose(coeffs_swapped[2], coeffs[3])   # xz <-> yz
        np.testing.assert_allclose(coeffs_swapped[3], coeffs[2])


#**************************************************************************************************#
#                                     Class TestConcomitantPhase                                    #
#**************************************************************************************************#
#                                                                                                  #
# The PhaseTerm-shaped wrapper: shape, dtype, and consistency with the free functions.              #
#                                                                                                  #
#**************************************************************************************************#
class TestConcomitantPhase:
    """The PhaseTerm-shaped wrapper: shape, dtype, and consistency with the free functions."""

    def test_matches_the_free_functions(self):
        rng = np.random.default_rng(1)
        g = rng.normal(scale=0.015, size=(3, T))
        positions = rng.normal(scale=0.08, size=(5, 3))
        t = np.arange(T) * DT

        term = ConcomitantFieldPhase(gradients_t_per_m=g, dt=DT, b0_tesla=B0)
        phase = term.phase(positions, t)

        expected = concomitant_field_basis(positions) @ concomitant_field_coefficients(g, DT, B0)
        np.testing.assert_allclose(phase, expected)
        assert phase.shape == (5, T)

    def test_rejects_mismatched_time_axis(self):
        g = np.zeros((3, T))
        term = ConcomitantFieldPhase(gradients_t_per_m=g, dt=DT, b0_tesla=B0)
        with pytest.raises(ValueError):
            term.phase(np.zeros((2, 3)), np.arange(T + 1) * DT)


#**************************************************************************************************#
#                                      Class TestInputValidation                                    #
#**************************************************************************************************#
class TestInputValidation:
    """Bad shapes/values are rejected rather than silently mishandled."""

    def test_rejects_non_positive_b0(self):
        with pytest.raises(ValueError, match="b0_tesla"):
            concomitant_field_coefficients(np.zeros((3, T)), DT, 0.0)

    def test_rejects_wrong_gradient_shape(self):
        with pytest.raises(ValueError, match="gradients_t_per_m"):
            concomitant_field_coefficients(np.zeros((2, T)), DT, B0)

    def test_rejects_wrong_position_shape(self):
        with pytest.raises(ValueError, match="positions"):
            concomitant_field_basis(np.zeros((5, 2)))
