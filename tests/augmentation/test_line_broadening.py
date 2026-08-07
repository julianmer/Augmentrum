"""
Tests for Line Broadening augmentation module.

Tests cover:
- Lorentzian broadening
- Gaussian broadening
- Voigt broadening
- Backend compatibility
- Integration tests
"""

import pytest
import numpy as np
from augmentrum.augmentation.line_broadening import LineBroadening
from augmentrum.core import NIfTI_MRS_Plus, Backend


#**************************************************************************************************#
#                                 Class TestLineBroadeningCreation                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test LineBroadening initialization.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class TestLineBroadeningCreation:
    """Test LineBroadening initialization."""

    def test_create_lorentzian(self):
        """Test creating Lorentzian broadening."""
        broadening = LineBroadening(lb_hz=5.0, mode='lorentzian')

        assert broadening.lb_hz == 5.0
        assert broadening.gb_hz == 0.0
        assert broadening.mode == 'lorentzian'

    def test_create_gaussian(self):
        """Test creating Gaussian broadening."""
        broadening = LineBroadening(gb_hz=3.0, mode='gaussian')

        assert broadening.lb_hz == 0.0
        assert broadening.gb_hz == 3.0
        assert broadening.mode == 'gaussian'

    def test_create_voigt(self):
        """Test creating Voigt broadening."""
        broadening = LineBroadening(lb_hz=5.0, gb_hz=3.0, mode='voigt')

        assert broadening.lb_hz == 5.0
        assert broadening.gb_hz == 3.0
        assert broadening.mode == 'voigt'

    def test_default_mode_is_voigt(self):
        """Test that default mode is voigt."""
        broadening = LineBroadening(lb_hz=5.0, gb_hz=3.0)

        assert broadening.mode == 'voigt'

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            LineBroadening(lb_hz=5.0, mode='invalid')

    def test_supports_all_backends(self):
        """Test that LineBroadening supports all backends."""
        broadening = LineBroadening(lb_hz=5.0)

        assert broadening.SUPPORTED_BACKENDS == tuple(Backend)
        assert broadening.supports_backend(Backend.NIFTI_LIST)
        assert broadening.supports_backend(Backend.NUMPY)


#**************************************************************************************************#
#                                  Class TestLorentzianBroadening                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test Lorentzian broadening.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class TestLorentzianBroadening:
    """Test Lorentzian broadening."""

    def test_lorentzian_changes_data(self, dummy_nifti_list):
        """Test that Lorentzian broadening modifies data."""
        broadening = LineBroadening(lb_hz=10.0, mode='lorentzian')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get original data
        original_data = nifti_plus[0][:].copy()

        # Apply broadening
        result_data, _ = broadening(nifti_plus, None)

        # Get broadened data
        broadened_data = result_data[0][:]

        # Data should have changed
        assert not np.allclose(broadened_data, original_data)

    def test_lorentzian_zero_does_nothing(self, dummy_nifti_list):
        """Test that zero Lorentzian broadening doesn't change data."""
        broadening = LineBroadening(lb_hz=0.0, mode='lorentzian')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get original data
        original_data = nifti_plus[0][:].copy()

        # Apply broadening
        result_data, _ = broadening(nifti_plus, None)

        # Data should be unchanged
        assert np.allclose(result_data[0][:], original_data)

    def test_lorentzian_decreases_later_points(self, dummy_nifti_list):
        """Test that Lorentzian broadening decreases later FID points."""
        broadening = LineBroadening(lb_hz=50.0, mode='lorentzian')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get original data
        original_data = nifti_plus[0][:].copy()

        # Apply broadening
        result_data, _ = broadening(nifti_plus, None)
        broadened_data = result_data[0][:]

        # Later points should have smaller magnitude (use .any() since it's multidimensional)
        assert (np.abs(broadened_data[..., -1]) < np.abs(original_data[..., -1])).any()


#**************************************************************************************************#
#                                   Class TestGaussianBroadening                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test Gaussian broadening.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianBroadening:
    """Test Gaussian broadening."""

    def test_gaussian_changes_data(self, dummy_nifti_list):
        """Test that Gaussian broadening modifies data."""
        broadening = LineBroadening(gb_hz=10.0, mode='gaussian')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get original data
        original_data = nifti_plus[0][:].copy()

        # Apply broadening
        result_data, _ = broadening(nifti_plus, None)

        # Data should have changed
        assert not np.allclose(result_data[0][:], original_data)

    def test_gaussian_zero_does_nothing(self, dummy_nifti_list):
        """Test that zero Gaussian broadening doesn't change data."""
        broadening = LineBroadening(gb_hz=0.0, mode='gaussian')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get original data
        original_data = nifti_plus[0][:].copy()

        # Apply broadening
        result_data, _ = broadening(nifti_plus, None)

        # Data should be unchanged
        assert np.allclose(result_data[0][:], original_data)


#**************************************************************************************************#
#                                    Class TestVoigtBroadening                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test Voigt broadening.                                                                           #
#                                                                                                  #
#**************************************************************************************************#
class TestVoigtBroadening:
    """Test Voigt broadening."""

    def test_voigt_changes_data(self, dummy_nifti_list):
        """Test that Voigt broadening modifies data."""
        broadening = LineBroadening(lb_hz=10.0, gb_hz=5.0, mode='voigt')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get original data
        original_data = nifti_plus[0][:].copy()

        # Apply broadening
        result_data, _ = broadening(nifti_plus, None)

        # Data should have changed
        assert not np.allclose(result_data[0][:], original_data)

    def test_voigt_with_both_zero_does_nothing(self, dummy_nifti_list):
        """Test that zero Voigt broadening doesn't change data."""
        broadening = LineBroadening(lb_hz=0.0, gb_hz=0.0, mode='voigt')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get original data
        original_data = nifti_plus[0][:].copy()

        # Apply broadening
        result_data, _ = broadening(nifti_plus, None)

        # Data should be unchanged
        assert np.allclose(result_data[0][:], original_data)


#**************************************************************************************************#
#                             Class TestLineBroadeningMultipleSubjects                             #
#**************************************************************************************************#
#                                                                                                  #
# Test processing multiple subjects.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestLineBroadeningMultipleSubjects:
    """Test processing multiple subjects."""

    def test_processes_all_subjects(self, dummy_nifti_list):
        """Test that all subjects are processed."""
        broadening = LineBroadening(lb_hz=10.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = broadening(nifti_plus, None)

        # Should have same number of subjects
        assert len(result_data) == len(dummy_nifti_list)

    def test_same_broadening_per_subject(self, dummy_nifti_list):
        """Test that same broadening is applied to each subject."""
        broadening = LineBroadening(lb_hz=10.0, mode='lorentzian')

        # Create identical data for all subjects
        nifti_list_identical = []
        template_data = dummy_nifti_list[0][:].copy()
        for nifti in dummy_nifti_list:
            nifti[:] = template_data
            nifti_list_identical.append(nifti)

        nifti_plus = NIfTI_MRS_Plus(nifti_list=nifti_list_identical, backend=Backend.NIFTI_LIST)
        result_data, _ = broadening(nifti_plus, None)

        # All subjects should have same result
        data0 = result_data[0][:]
        data1 = result_data[1][:]

        assert np.allclose(data0, data1)


#**************************************************************************************************#
#                                  Class TestLineBroadeningWater                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test water reference handling.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestLineBroadeningWater:
    """Test water reference handling."""

    def test_water_unchanged(self, dummy_nifti_list):
        """Test that water reference is not modified."""
        from copy import deepcopy
        broadening = LineBroadening(lb_hz=10.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        # Use COPIES so they're not the same objects
        water_niftis = [deepcopy(dummy_nifti_list[0]), deepcopy(dummy_nifti_list[1])]
        water_plus = NIfTI_MRS_Plus(nifti_list=water_niftis, backend=Backend.NIFTI_LIST)

        # Get original water
        original_water = water_plus[0][:].copy()

        # Apply broadening
        result_data, result_water = broadening(nifti_plus, water_plus)

        # Water data should be unchanged (check data, not object identity)
        assert result_water is not None
        assert np.allclose(result_water[0][:], original_water, rtol=1e-5, atol=1e-7)


#**************************************************************************************************#
#                                 Class TestLineBroadeningLogging                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test automatic logging/provenance.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestLineBroadeningLogging:
    """Test automatic logging/provenance."""

    def test_logging_when_not_volatile(self, dummy_nifti_list):
        """Test that metadata is logged when volatile=False."""
        broadening = LineBroadening(lb_hz=10.0, gb_hz=5.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=False)

        result_data, _ = broadening(nifti_plus, None)

        # Check metadata was updated
        assert 'common_provenance' in result_data.metadata_common
        assert len(result_data.metadata_common['common_provenance']) > 0

    def test_no_logging_when_volatile(self, dummy_nifti_list):
        """Test that metadata is not logged when volatile=True."""
        broadening = LineBroadening(lb_hz=10.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=True)

        result_data, _ = broadening(nifti_plus, None)

        # Metadata should still be empty
        assert result_data.metadata_common == {}


#**************************************************************************************************#
#                                Class TestLineBroadeningEdgeCases                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test edge cases.                                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class TestLineBroadeningEdgeCases:
    """Test edge cases."""

    def test_with_single_subject(self, dummy_nifti_mrs):
        """Test with single subject."""
        broadening = LineBroadening(lb_hz=10.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[dummy_nifti_mrs], backend=Backend.NIFTI_LIST)

        result_data, _ = broadening(nifti_plus, None)

        assert len(result_data) == 1

    def test_preserves_complex_dtype(self, dummy_nifti_list):
        """Test that complex dtype is preserved."""
        broadening = LineBroadening(lb_hz=10.0, gb_hz=5.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = broadening(nifti_plus, None)

        # Data should still be complex
        assert np.iscomplexobj(result_data[0][:])


#**************************************************************************************************#
#                               Class TestLineBroadeningIntegration                                #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests.                                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestLineBroadeningIntegration:
    """Integration tests."""

    def test_in_pipeline(self, dummy_nifti_list):
        """Test LineBroadening in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        broadening = LineBroadening(lb_hz=10.0, gb_hz=5.0)
        pipeline = AugmentationPipeline([broadening])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)

    def test_chained_with_other_modules(self, dummy_nifti_list):
        """Test chaining LineBroadening with other modules."""
        from augmentrum.core.pipeline import AugmentationPipeline
        from augmentrum.augmentation.gaussian_noise import GaussianNoise

        broadening = LineBroadening(lb_hz=10.0)
        noise = GaussianNoise(sigma_frac=0.02)
        pipeline = AugmentationPipeline([broadening, noise])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert result_data is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
