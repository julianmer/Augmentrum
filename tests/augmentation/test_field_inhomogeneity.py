####################################################################################################
#                                test_field_inhomogeneity.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-04                                                                              #
#                                                                                                  #
# Purpose: FieldInhomogeneity's B1+ complex modulation and segmented B0 off-resonance model,       #
#          against the identity case, analytic single-voxel spectral shifts, and a fine-segment    #
#          reference.                                                                               #
#                                                                                                  #
####################################################################################################

"""
Tests for FieldInhomogeneity (B1+ transmit-field and segmented B0 off-resonance).
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.augmentation.field_inhomogeneity import FieldInhomogeneity

B, X, Y, Z, T = 1, 32, 32, 1, 64
DWELL = 1.0 / 2000.0   # 2 kHz spectral width


def _disc_phantom(batch=B, decay_hz=15.0):
    """A disc, decaying FID per voxel - enough spatial structure to segment."""
    yy, xx = np.mgrid[0:X, 0:Y]
    disc = ((xx - X / 2) ** 2 + (yy - Y / 2) ** 2 < (X / 3) ** 2).astype(np.float32)
    t = np.arange(T) * DWELL
    fid = np.exp(-decay_hz * t).astype(np.complex64)
    vol = np.zeros((batch, X, Y, Z, T), np.complex64)
    vol[:] = disc[None, :, :, None, None] * fid[None, None, None, None, :]
    return vol


def _run(vol, **kwargs):
    # A Cartesian trajectory makes the gridding NUFFT round trip essentially
    # exact (its samples land on-grid), which isolates the field effects under
    # test from the gridding NUFFT's own approximation error. Tests that care
    # about that error instead (e.g. segment-count convergence) are unaffected
    # by the choice, since the same trajectory is shared by every case compared.
    kwargs.setdefault('trajectory', 'cartesian_2d')
    kwargs.setdefault('traj_seed', 0)
    field = FieldInhomogeneity(**kwargs)
    out, water = field.process_tensor(vol, sw_hz=1.0 / DWELL)
    return np.asarray(out), field


#**************************************************************************************************#
#                                     Class TestIdentityWhenNone                                   #
#**************************************************************************************************#
#                                                                                                  #
# Requirement 1: b1_map=None, b0_map=None -> existing behavior, unchanged.                        #
#                                                                                                  #
#**************************************************************************************************#
class TestIdentityWhenNone:
    """Requirement 1: b1_map=None, b0_map=None -> existing behavior, unchanged."""

    def test_no_maps_is_bitwise_identity(self):
        vol = _disc_phantom()
        out, field = _run(vol.copy())
        assert out is vol or np.array_equal(out, vol)
        assert field.DOMAIN is None

    def test_no_nufft_is_run_without_a_b0_map(self):
        """B1+ alone never touches the trajectory machinery."""
        b1 = (2.0 * np.ones((X, Y, Z), np.complex64))
        out, field = _run(_disc_phantom(), b1_map=b1)
        assert field.last_meta_ is None
        assert field.last_b0_segments_ is None


#**************************************************************************************************#
#                                       Class TestB1Magnitude                                       #
#**************************************************************************************************#
#                                                                                                  #
# Requirement 2: a magnitude-only B1+ map scales the image spatially.                              #
#                                                                                                  #
#**************************************************************************************************#
class TestB1Magnitude:
    """Requirement 2: a magnitude-only B1+ map scales the image spatially."""

    def test_magnitude_scales_each_half_differently(self):
        vol = _disc_phantom()
        b1 = np.ones((X, Y, Z), np.float32)
        b1[:, :Y // 2, :] = 0.5   # left half at half sensitivity
        b1[:, Y // 2:, :] = 1.5   # right half boosted

        out, _ = _run(vol.copy(), b1_map=b1)

        left = np.abs(out[0, :, :Y // 2]).mean()
        right = np.abs(out[0, :, Y // 2:]).mean()
        orig_left = np.abs(vol[0, :, :Y // 2]).mean()
        orig_right = np.abs(vol[0, :, Y // 2:]).mean()

        assert left == pytest.approx(0.5 * orig_left, rel=1e-4)
        assert right == pytest.approx(1.5 * orig_right, rel=1e-4)


#**************************************************************************************************#
#                                       Class TestB1Phase                                           #
#**************************************************************************************************#
#                                                                                                  #
# Requirement 3: a constant B1+ phase is a global complex phase.                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestB1Phase:
    """Requirement 3: a constant B1+ phase is a global complex phase."""

    def test_constant_phase_rotates_every_voxel_alike(self):
        vol = _disc_phantom()
        phi = 0.7
        b1 = (np.ones((X, Y, Z), np.float32) * np.exp(1j * phi)).astype(np.complex64)

        out, _ = _run(vol.copy(), b1_map=b1)

        np.testing.assert_allclose(out, vol * np.exp(1j * phi), atol=1e-4, rtol=1e-4)


#**************************************************************************************************#
#                                    Class TestConstantB0Shift                                      #
#**************************************************************************************************#
#                                                                                                  #
# Requirement 4: a spatially constant B0 map gives a known spectral frequency shift.               #
#                                                                                                  #
#**************************************************************************************************#
class TestConstantB0Shift:
    """Requirement 4: a spatially constant B0 map gives a known spectral frequency shift."""

    def test_constant_offset_shifts_every_voxel_the_same_way(self):
        vol = _disc_phantom()
        shift_hz = 12.0
        b0 = np.full((X, Y, Z), shift_hz, dtype=np.float64)

        out, _ = _run(vol.copy(), b0_map=b0, b0_n_segments=4)

        t = np.arange(T) * DWELL
        expected = vol * np.exp(-1j * 2.0 * np.pi * shift_hz * t)[None, None, None, None, :]

        # Approximate (gridding NUFFT round trip), not exact - but a Cartesian
        # trajectory makes that approximation error negligible here.
        num = np.linalg.norm(out - expected)
        den = np.linalg.norm(expected)
        assert num / den < 0.01, f"relative error {num / den:.4f} too high for a uniform shift"

    def test_zero_b0_reduces_to_ordinary_forward_nufft(self):
        """Requirement 7: a zero B0 map is the plain NUFFT round trip."""
        vol = _disc_phantom()
        b0 = np.zeros((X, Y, Z), dtype=np.float64)

        out_seg, _ = _run(vol.copy(), b0_map=b0, b0_n_segments=8)
        out_plain, _ = _run(vol.copy(), b0_map=b0, b0_n_segments=1)

        np.testing.assert_allclose(out_seg, out_plain, atol=1e-5, rtol=1e-5)


#**************************************************************************************************#
#                                   Class TestSpatiallyVaryingB0                                    #
#**************************************************************************************************#
#                                                                                                  #
# Requirement 5: different voxels acquire different frequency shifts.                              #
#                                                                                                  #
#**************************************************************************************************#
class TestSpatiallyVaryingB0:
    """Requirement 5: different voxels acquire different frequency shifts."""

    def test_two_halves_diverge_from_a_single_uniform_shift(self):
        vol = _disc_phantom(decay_hz=2.0)   # slow decay -> B0 dephasing dominates
        b0 = np.zeros((X, Y, Z), dtype=np.float64)
        b0[:, :Y // 2, :] = -30.0
        b0[:, Y // 2:, :] = 30.0

        out_varying, _ = _run(vol.copy(), b0_map=b0, b0_n_segments=8)
        out_uniform, _ = _run(vol.copy(), b0_map=np.zeros_like(b0), b0_n_segments=8)

        left = out_varying[0, :, :Y // 2]
        right = out_varying[0, :, Y // 2:]
        left_ref = out_uniform[0, :, :Y // 2]
        right_ref = out_uniform[0, :, Y // 2:]

        # Each half now differs from its own (unshifted) reconstruction, and the
        # two halves' induced phase evolutions differ from one another.
        assert np.linalg.norm(left - left_ref) > 1e-3
        assert np.linalg.norm(right - right_ref) > 1e-3
        assert np.linalg.norm(left - right) > np.linalg.norm(left_ref - right_ref)


#**************************************************************************************************#
#                                  Class TestSegmentCountConvergence                                #
#**************************************************************************************************#
#                                                                                                  #
# Requirement 6: more segments converge toward a fine-resolution B0 reference.                     #
#                                                                                                  #
#**************************************************************************************************#
class TestSegmentCountConvergence:
    """Requirement 6: more segments converge toward a fine-resolution B0 reference."""

    def test_error_against_fine_reference_shrinks(self):
        vol = _disc_phantom(decay_hz=2.0)
        yy, xx = np.mgrid[0:X, 0:Y]
        b0 = (60.0 * (xx / X - 0.5)).astype(np.float64)[:, :, None]   # smooth gradient

        reference, _ = _run(vol.copy(), b0_map=b0, b0_n_segments=64)

        errors = []
        for q in (2, 4, 8, 16):
            out, _ = _run(vol.copy(), b0_map=b0, b0_n_segments=q)
            errors.append(float(np.linalg.norm(out - reference)))

        assert errors[-1] <= errors[0], f"error did not shrink with more segments: {errors}"


#**************************************************************************************************#
#                                        Class TestComposition                                      #
#**************************************************************************************************#
#                                                                                                  #
# Requirement 8: B1+ and B0 compose in the documented order.                                       #
#                                                                                                  #
#**************************************************************************************************#
class TestComposition:
    """Requirement 8: B1+ and B0 compose in the documented order I -> B1+ -> B0 -> NUFFT."""

    def test_combined_matches_applying_each_stage_by_hand(self):
        vol = _disc_phantom(decay_hz=2.0)
        b1 = np.ones((X, Y, Z), np.float32)
        b1[:, Y // 2:, :] = 1.4
        shift_hz = 10.0
        b0 = np.full((X, Y, Z), shift_hz, dtype=np.float64)

        combined, _ = _run(vol.copy(), b1_map=b1, b0_map=b0, b0_n_segments=4)

        # Reference: B1+ applied by hand, then the same B0-only model.
        vol_b1 = vol * b1[None, :, :, :, None]
        b1_then_b0, _ = _run(vol_b1.copy(), b0_map=b0, b0_n_segments=4)

        np.testing.assert_allclose(combined, b1_then_b0, atol=1e-5, rtol=1e-5)


#**************************************************************************************************#
#                                    Class TestSegmentDiagnostics                                   #
#**************************************************************************************************#
#                                                                                                  #
# Provenance / the error-budget helper from the class docstring.                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestSegmentDiagnostics:
    """Provenance / the error-budget helper from the class docstring."""

    def test_max_phase_error_scales_with_bin_width_and_time(self):
        small = FieldInhomogeneity.max_phase_error_rad(delta_f_hz=1.0, t_max_s=0.01)
        large = FieldInhomogeneity.max_phase_error_rad(delta_f_hz=10.0, t_max_s=0.01)
        assert large == pytest.approx(10.0 * small)

    def test_segments_are_recorded_after_a_b0_call(self):
        vol = _disc_phantom()
        b0 = np.linspace(-20, 20, X * Y).reshape(X, Y, 1)
        _, field = _run(vol, b0_map=b0, b0_n_segments=6)

        assert field.last_b0_segments_ is not None
        assert len(field.last_b0_segments_) == B
        assert len(field.last_b0_segments_[0]['f_q_hz']) <= 6


#**************************************************************************************************#
#                                   Class TestInvalidConfiguration                                  #
#**************************************************************************************************#
#                                                                                                  #
# Constructor-time validation.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestInvalidConfiguration:
    """Constructor-time validation."""

    def test_rejects_unknown_b1_mode(self):
        with pytest.raises(ValueError, match="b1_mode"):
            FieldInhomogeneity(b1_mode="not_a_mode")

    def test_rejects_unknown_segment_strategy(self):
        with pytest.raises(ValueError, match="b0_segment_strategy"):
            FieldInhomogeneity(b0_segment_strategy="not_a_strategy")

    def test_rejects_mismatched_map_shape(self):
        vol = _disc_phantom()
        bad_b1 = np.ones((X + 1, Y, Z), np.complex64)
        with pytest.raises(ValueError, match="b1_map"):
            _run(vol, b1_map=bad_b1)
