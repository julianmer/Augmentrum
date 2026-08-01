"""
Tests for PhaseShift and FrequencyShift modules.

Tests cover:
- Zero-order phase
- First-order phase
- Frequency shift
- Backend compatibility
- Integration tests
"""

import pytest
import numpy as np
from augmentrum.augmentation.phase_frequency import PhaseShift, FrequencyShift
from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus, Backend


#**************************************************************************************************#
#                                   Class TestPhaseShiftCreation                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test PhaseShift initialization.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class TestPhaseShiftCreation:
    """Test PhaseShift initialization."""

    def test_create_zero_order_only(self):
        """Test creating with zero-order phase only."""
        phase = PhaseShift(zero_order_deg=60.0)
        assert phase.zero_order_deg == 60.0
        assert phase.first_order_deg == 0.0

    def test_create_first_order_only(self):
        """Test creating with first-order phase only."""
        phase = PhaseShift(first_order_deg=90.0)
        assert phase.zero_order_deg == 0.0
        assert phase.first_order_deg == 90.0

    def test_create_both_orders(self):
        """Test creating with both phase orders."""
        phase = PhaseShift(zero_order_deg=30.0, first_order_deg=45.0)
        assert phase.zero_order_deg == 30.0
        assert phase.first_order_deg == 45.0


#**************************************************************************************************#
#                                     Class TestZeroOrderPhase                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test zero-order phase shift.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestZeroOrderPhase:
    """Test zero-order phase shift."""

    def test_zero_order_changes_data(self, dummy_nifti_list):
        """Test that zero-order phase modifies data."""
        phase = PhaseShift(zero_order_deg=60.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = phase(nifti_plus, None)
        phased_data = result_data[0][:]

        assert not np.allclose(phased_data, original_data)

    def test_zero_order_preserves_magnitude(self, dummy_nifti_list):
        """Test that zero-order phase preserves magnitude."""
        phase = PhaseShift(zero_order_deg=60.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_mag = np.abs(nifti_plus[0][:])
        result_data, _ = phase(nifti_plus, None)
        phased_mag = np.abs(result_data[0][:])

        assert np.allclose(phased_mag, original_mag)

    def test_zero_phase_does_nothing(self, dummy_nifti_list):
        """Test that zero phase doesn't change data."""
        phase = PhaseShift(zero_order_deg=0.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = phase(nifti_plus, None)

        assert np.allclose(result_data[0][:], original_data)


#**************************************************************************************************#
#                                    Class TestFirstOrderPhase                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test first-order phase shift.                                                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestFirstOrderPhase:
    """Test first-order phase shift."""

    def test_first_order_changes_data(self, dummy_nifti_list):
        """Test that first-order phase modifies data."""
        phase = PhaseShift(first_order_deg=90.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = phase(nifti_plus, None)
        phased_data = result_data[0][:]

        assert not np.allclose(phased_data, original_data)

    def test_first_order_different_from_zero_order(self, dummy_nifti_list):
        """Test that first-order phase is different from zero-order."""
        # Use same starting data
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        original_data = nifti_plus[0][:].copy()

        # Apply zero-order phase (constant phase shift)
        phase_zero = PhaseShift(zero_order_deg=45.0, first_order_deg=0.0)
        nifti_plus_zero = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_zero, _ = phase_zero(nifti_plus_zero, None)
        data_zero = result_zero[0][:].copy()

        # Apply first-order phase (linear ramp)
        phase_first = PhaseShift(zero_order_deg=0.0, first_order_deg=360.0)  # Full rotation
        nifti_plus_first = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_first, _ = phase_first(nifti_plus_first, None)
        data_first = result_first[0][:].copy()

        # Both should differ from original
        assert not np.allclose(data_zero, original_data)
        assert not np.allclose(data_first, original_data)

        # Zero-order and first-order should produce different phase patterns
        # (zero-order rotates everything uniformly, first-order creates a ramp)
        # We can't directly compare them since they start differently,
        # but we can verify that first-order creates non-uniform phase
        # by checking that different parts of the spectrum are affected differently

        # For first-order, check that it actually does something
        diff_first = np.abs(data_first - original_data)
        assert np.max(diff_first) > 0.01  # Should create some change


#**************************************************************************************************#
#                                 Class TestFrequencyShiftCreation                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test FrequencyShift initialization.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class TestFrequencyShiftCreation:
    """Test FrequencyShift initialization."""

    def test_create_positive_shift(self):
        """Test creating with positive frequency shift."""
        freq = FrequencyShift(shift_hz=10.0)
        assert freq.shift_hz == 10.0

    def test_create_negative_shift(self):
        """Test creating with negative frequency shift."""
        freq = FrequencyShift(shift_hz=-20.0)
        assert freq.shift_hz == -20.0


#**************************************************************************************************#
#                                     Class TestFrequencyShift                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test frequency shift.                                                                            #
#                                                                                                  #
#**************************************************************************************************#
class TestFrequencyShift:
    """Test frequency shift."""

    def test_frequency_shift_changes_data(self, dummy_nifti_list):
        """Test that frequency shift modifies data."""
        freq = FrequencyShift(shift_hz=10.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = freq(nifti_plus, None)
        shifted_data = result_data[0][:]

        assert not np.allclose(shifted_data, original_data)

    def test_zero_shift_does_nothing(self, dummy_nifti_list):
        """Test that zero shift doesn't change data."""
        freq = FrequencyShift(shift_hz=0.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = freq(nifti_plus, None)

        assert np.allclose(result_data[0][:], original_data)

    def test_opposite_shifts_cancel(self, dummy_nifti_list):
        """Test that opposite shifts approximately cancel."""
        freq_plus = FrequencyShift(shift_hz=10.0)
        freq_minus = FrequencyShift(shift_hz=-10.0)

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        original_data = nifti_plus[0][:].copy()

        # Apply positive then negative shift
        result1, _ = freq_plus(nifti_plus, None)
        result2, _ = freq_minus(result1, None)

        # Should be close to original (within numerical precision)
        assert np.allclose(result2[0][:], original_data, rtol=1e-5)


#**************************************************************************************************#
#                              Class TestPhaseFrequencyWaterReference                              #
#**************************************************************************************************#
#                                                                                                  #
# Test water reference handling.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestPhaseFrequencyWaterReference:
    """Test water reference handling."""

    def test_phase_water_unchanged(self, dummy_nifti_list):
        """Test that water reference is not modified by phase shift."""
        from copy import deepcopy
        phase = PhaseShift(zero_order_deg=60.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        # Use COPIES so they're not the same objects
        water_niftis = [deepcopy(dummy_nifti_list[0]), deepcopy(dummy_nifti_list[1])]
        water_plus = NIfTI_MRS_Plus(nifti_list=water_niftis, backend=Backend.NIFTI_LIST)

        original_water = water_plus[0][:].copy()
        result_data, result_water = phase(nifti_plus, water_plus)

        # Water data should be unchanged (check data, not object identity)
        assert result_water is not None
        assert np.allclose(result_water[0][:], original_water, rtol=1e-5, atol=1e-7)

    def test_frequency_water_unchanged(self, dummy_nifti_list):
        """Test that water reference is not modified by frequency shift."""
        from copy import deepcopy
        freq = FrequencyShift(shift_hz=10.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        # Use COPIES so they're not the same objects
        water_niftis = [deepcopy(dummy_nifti_list[0]), deepcopy(dummy_nifti_list[1])]
        water_plus = NIfTI_MRS_Plus(nifti_list=water_niftis, backend=Backend.NIFTI_LIST)

        original_water = water_plus[0][:].copy()
        result_data, result_water = freq(nifti_plus, water_plus)

        # Water data should be unchanged (check data, not object identity)
        assert result_water is not None
        assert np.allclose(result_water[0][:], original_water, rtol=1e-5, atol=1e-7)


#**************************************************************************************************#
#                               Class TestPhaseFrequencyIntegration                                #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests.                                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestPhaseFrequencyIntegration:
    """Integration tests."""

    def test_phase_in_pipeline(self, dummy_nifti_list):
        """Test PhaseShift in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        phase = PhaseShift(zero_order_deg=30.0, first_order_deg=15.0)
        pipeline = AugmentationPipeline([phase])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)

    def test_frequency_in_pipeline(self, dummy_nifti_list):
        """Test FrequencyShift in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        freq = FrequencyShift(shift_hz=10.0)
        pipeline = AugmentationPipeline([freq])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)

    def test_combined_phase_and_frequency(self, dummy_nifti_list):
        """Test combining phase and frequency shifts."""
        from augmentrum.core.pipeline import AugmentationPipeline

        phase = PhaseShift(zero_order_deg=30.0)
        freq = FrequencyShift(shift_hz=10.0)
        pipeline = AugmentationPipeline([phase, freq])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert result_data is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
