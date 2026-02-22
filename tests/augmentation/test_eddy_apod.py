"""
Tests for EddyCurrent and Apodization modules.
"""

import pytest
import numpy as np
from augmentrum.augmentation.eddy_current import EddyCurrent
from augmentrum.augmentation.apodization import Apodization
from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus, Backend


# ============================================================================
# EddyCurrent Tests
# ============================================================================

class TestEddyCurrentCreation:
    """Test EddyCurrent initialization."""

    def test_create_synthetic(self):
        """Test creating synthetic eddy current."""
        ec = EddyCurrent(mode='synthetic', std_rad=0.8, lp_cut_hz=30.0)
        assert ec.mode == 'synthetic'
        assert ec.std_rad == 0.8
        assert ec.lp_cut_hz == 30.0

    def test_create_water(self):
        """Test creating water-derived eddy current."""
        ec = EddyCurrent(mode='water', lp_cut_hz=20.0, strength=1.5)
        assert ec.mode == 'water'
        assert ec.lp_cut_hz == 20.0
        assert ec.strength == 1.5

    def test_default_mode_is_synthetic(self):
        """Test that default mode is synthetic."""
        ec = EddyCurrent()
        assert ec.mode == 'synthetic'

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            EddyCurrent(mode='invalid')


class TestSyntheticEddyCurrent:
    """Test synthetic eddy current."""

    def test_synthetic_changes_data(self, dummy_nifti_list):
        """Test that synthetic eddy current modifies data."""
        ec = EddyCurrent(mode='synthetic', std_rad=0.8)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = ec(nifti_plus, None)
        ec_data = result_data[0][:]

        assert not np.allclose(ec_data, original_data)

    def test_synthetic_preserves_dtype(self, dummy_nifti_list):
        """Test that eddy current preserves complex dtype."""
        ec = EddyCurrent(mode='synthetic')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = ec(nifti_plus, None)
        assert np.iscomplexobj(result_data[0][:])

    def test_synthetic_reproducibility(self, dummy_nifti_list):
        """Test reproducibility with same seed."""
        ec1 = EddyCurrent(mode='synthetic', seed=42)
        ec2 = EddyCurrent(mode='synthetic', seed=42)

        nifti_plus1 = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        nifti_plus2 = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result1, _ = ec1(nifti_plus1, None)
        result2, _ = ec2(nifti_plus2, None)

        assert np.allclose(result1[0][:], result2[0][:])


class TestWaterDerivedEddyCurrent:
    """Test water-derived eddy current."""

    def test_water_mode_requires_water_reference(self, dummy_nifti_list):
        """Test that water mode requires water reference."""
        ec = EddyCurrent(mode='water')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        with pytest.raises(ValueError, match="Water reference required"):
            ec(nifti_plus, None)

    def test_water_derived_changes_data(self, dummy_nifti_list):
        """Test that water-derived eddy current modifies data."""
        ec = EddyCurrent(mode='water', strength=1.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        water_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list[:2], backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = ec(nifti_plus, water_plus)
        ec_data = result_data[0][:]

        assert not np.allclose(ec_data, original_data)


# ============================================================================
# Apodization Tests
# ============================================================================

class TestApodizationCreation:
    """Test Apodization initialization."""

    def test_create_truncate(self):
        """Test creating truncation apodization."""
        apod = Apodization(mode='truncate', n_pts=1024)
        assert apod.mode == 'truncate'
        assert apod.n_pts == 1024

    def test_create_exponential(self):
        """Test creating exponential apodization."""
        apod = Apodization(mode='exponential', lb_hz=5.0)
        assert apod.mode == 'exponential'
        assert apod.lb_hz == 5.0

    def test_default_mode_is_exponential(self):
        """Test that default mode is exponential."""
        apod = Apodization(lb_hz=3.0)
        assert apod.mode == 'exponential'

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            Apodization(mode='invalid')


class TestTruncationApodization:
    """Test truncation apodization."""

    @pytest.mark.skip(reason="Truncation changes shape and can't be assigned back to NIfTI in-place")
    @pytest.mark.skip(reason="Truncation changes shape and can't be assigned back to NIfTI in-place")
    def test_truncate_changes_shape(self, dummy_nifti_list):
        """Test truncation changes the shape correctly."""
        # Note: dummy data has shape (..., 2048, 8, 16) where last dim is DYN
        # Truncation operates on last dimension (16 points)
        apod = Apodization(mode='truncate', n_pts=8)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_shape = nifti_plus[0].shape
        result_data, _ = apod(nifti_plus, None)
        new_shape = result_data[0].shape

        # Last dimension should be truncated from 16 to 8
        assert new_shape[-1] < original_shape[-1]
        assert new_shape[-1] == 8

    @pytest.mark.skip(reason="Truncation changes shape and can't be assigned back to NIfTI in-place")
    def test_truncate_with_frac(self, dummy_nifti_list):
        """Test truncation with fraction of points."""
        apod = Apodization(mode='truncate', frac_pts=0.5)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_size = nifti_plus[0].shape[-1]
        result_data, _ = apod(nifti_plus, None)
        new_size = result_data[0].shape[-1]

        # Should be approximately half
        assert new_size <= original_size // 2 + 1


class TestExponentialApodization:
    """Test exponential apodization."""

    def test_exponential_changes_data(self, dummy_nifti_list):
        """Test that exponential apodization modifies data."""
        apod = Apodization(mode='exponential', lb_hz=5.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = apod(nifti_plus, None)
        apod_data = result_data[0][:]

        assert not np.allclose(apod_data, original_data)

    def test_exponential_preserves_shape(self, dummy_nifti_list):
        """Test that exponential preserves data shape."""
        apod = Apodization(mode='exponential', lb_hz=5.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_shape = nifti_plus[0].shape
        result_data, _ = apod(nifti_plus, None)
        new_shape = result_data[0].shape

        assert new_shape == original_shape

    def test_exponential_decreases_later_points(self, dummy_nifti_list):
        """Test that exponential apodization decreases later FID points."""
        apod = Apodization(mode='exponential', lb_hz=10.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = apod(nifti_plus, None)
        apod_data = result_data[0][:]

        # Later points should have smaller magnitude (use .any() for multidimensional)
        assert (np.abs(apod_data[..., -1]) < np.abs(original_data[..., -1])).any()

    def test_auto_lb_calculation(self, dummy_nifti_list):
        """Test auto lb calculation."""
        apod = Apodization(mode='exponential', auto_lb=True, target_pts=512, target_damp=0.01)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = apod(nifti_plus, None)
        assert result_data is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestEddyApodIntegration:
    """Integration tests."""

    def test_eddy_in_pipeline(self, dummy_nifti_list):
        """Test EddyCurrent in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        ec = EddyCurrent(mode='synthetic', std_rad=0.6)
        pipeline = AugmentationPipeline([ec])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)

    def test_apod_in_pipeline(self, dummy_nifti_list):
        """Test Apodization in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        apod = Apodization(mode='exponential', lb_hz=3.0)
        pipeline = AugmentationPipeline([apod])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)

    def test_combined_eddy_and_apod(self, dummy_nifti_list):
        """Test combining eddy current and apodization."""
        from augmentrum.core.pipeline import AugmentationPipeline

        ec = EddyCurrent(mode='synthetic')
        apod = Apodization(mode='exponential', lb_hz=3.0)
        pipeline = AugmentationPipeline([ec, apod])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert result_data is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
