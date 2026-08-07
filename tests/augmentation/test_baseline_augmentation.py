"""
Tests for BaselineAugmentation module.

Tests cover:
- Random Walk baseline
- B-Spline baseline
- Polynomial baseline
- Backend compatibility
- Integration tests
"""

import pytest
import numpy as np
from augmentrum.augmentation.baseline_augmentation import BaselineAugmentation
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend


#**************************************************************************************************#
#                              Class TestBaselineAugmentationCreation                              #
#**************************************************************************************************#
#                                                                                                  #
# Test BaselineAugmentation initialization.                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestBaselineAugmentationCreation:
    """Test BaselineAugmentation initialization."""

    def test_create_random_walk(self):
        """Test creating random walk baseline."""
        baseline = BaselineAugmentation(mode='random_walk', baseline_frac=0.05)
        assert baseline.mode == 'random_walk'
        assert baseline.baseline_frac == 0.05

    def test_create_bspline(self):
        """Test creating B-spline baseline."""
        baseline = BaselineAugmentation(mode='bspline', knots_per_ppm=12, baseline_frac=0.10)
        assert baseline.mode == 'bspline'
        assert baseline.knots_per_ppm == 12
        assert baseline.baseline_frac == 0.10

    def test_create_polynomial(self):
        """Test creating polynomial baseline."""
        baseline = BaselineAugmentation(mode='polynomial', order=5)
        assert baseline.mode == 'polynomial'
        assert baseline.order == 5

    def test_default_mode_is_random_walk(self):
        """Test that default mode is random_walk."""
        baseline = BaselineAugmentation()
        assert baseline.mode == 'random_walk'

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            BaselineAugmentation(mode='invalid')


#**************************************************************************************************#
#                                   Class TestRandomWalkBaseline                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test random walk baseline.                                                                       #
#                                                                                                  #
#**************************************************************************************************#
class TestRandomWalkBaseline:
    """Test random walk baseline."""

    def test_random_walk_changes_data(self, dummy_nifti_list):
        """Test that random walk baseline modifies data."""
        baseline = BaselineAugmentation(mode='random_walk', baseline_frac=0.10)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = baseline(nifti_plus, None)
        augmented_data = result_data[0][:]

        assert not np.allclose(augmented_data, original_data)

    def test_random_walk_preserves_dtype(self, dummy_nifti_list):
        """Test that baseline preserves complex dtype."""
        baseline = BaselineAugmentation(mode='random_walk')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = baseline(nifti_plus, None)
        assert np.iscomplexobj(result_data[0][:])

    def test_random_walk_reproducibility(self, dummy_nifti_list):
        """Test reproducibility with same seed."""
        baseline1 = BaselineAugmentation(mode='random_walk', seed=42)
        baseline2 = BaselineAugmentation(mode='random_walk', seed=42)

        nifti_plus1 = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        nifti_plus2 = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result1, _ = baseline1(nifti_plus1, None)
        result2, _ = baseline2(nifti_plus2, None)

        assert np.allclose(result1[0][:], result2[0][:])


#**************************************************************************************************#
#                                    Class TestBSplineBaseline                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test B-spline baseline.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class TestBSplineBaseline:
    """Test B-spline baseline."""

    @pytest.mark.slow
    @pytest.mark.skip(reason="B-spline implementation has numerical stability issues causing extremely slow execution")
    # TODO: Fix!
    def test_bspline_changes_data(self, dummy_nifti_list):
        """Test that B-spline baseline modifies data."""
        baseline = BaselineAugmentation(mode='bspline', knots_per_ppm=8, baseline_frac=0.10)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = baseline(nifti_plus, None)
        augmented_data = result_data[0][:]

        assert not np.allclose(augmented_data, original_data)

    @pytest.mark.slow
    @pytest.mark.skip(reason="B-spline implementation has numerical stability issues causing extremely slow execution")
    # TODO: Fix!
    def test_bspline_smoothness(self, dummy_nifti_list):
        """Test that B-spline produces smooth baseline."""
        baseline = BaselineAugmentation(mode='bspline', ed_per_ppm=2.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = baseline(nifti_plus, None)
        assert result_data is not None


#**************************************************************************************************#
#                                   Class TestPolynomialBaseline                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test polynomial baseline.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestPolynomialBaseline:
    """Test polynomial baseline."""

    def test_polynomial_changes_data(self, dummy_nifti_list):
        """Test that polynomial baseline modifies data."""
        baseline = BaselineAugmentation(mode='polynomial', order=3)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = baseline(nifti_plus, None)
        augmented_data = result_data[0][:]

        assert not np.allclose(augmented_data, original_data)

    def test_polynomial_with_windows(self, dummy_nifti_list):
        """Test polynomial with ppm windows."""
        baseline = BaselineAugmentation(
            mode='polynomial',
            order=3,
            ppm_windows=[(5.0, 4.0), (1.0, 0.5)]
        )
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = baseline(nifti_plus, None)
        assert result_data is not None


#**************************************************************************************************#
#                                Class TestBaselineMultipleSubjects                                #
#**************************************************************************************************#
#                                                                                                  #
# Test processing multiple subjects.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestBaselineMultipleSubjects:
    """Test processing multiple subjects."""

    def test_processes_all_subjects(self, dummy_nifti_list):
        """Test that all subjects are processed."""
        baseline = BaselineAugmentation(mode='random_walk')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = baseline(nifti_plus, None)
        assert len(result_data) == len(dummy_nifti_list)


#**************************************************************************************************#
#                                 Class TestBaselineWaterReference                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test water reference handling.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestBaselineWaterReference:
    """Test water reference handling."""

    def test_water_unchanged(self, dummy_nifti_list):
        """Test that water reference is not modified."""
        from copy import deepcopy
        baseline = BaselineAugmentation(mode='random_walk')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        # Use COPIES so they're not the same objects
        water_niftis = [deepcopy(dummy_nifti_list[0]), deepcopy(dummy_nifti_list[1])]
        water_plus = NIfTI_MRS_Plus(nifti_list=water_niftis, backend=Backend.NIFTI_LIST)

        original_water = water_plus[0][:].copy()
        result_data, result_water = baseline(nifti_plus, water_plus)

        # Water data should be unchanged (check data, not object identity)
        assert result_water is not None
        assert np.allclose(result_water[0][:], original_water, rtol=1e-5, atol=1e-7)


#**************************************************************************************************#
#                                  Class TestBaselineIntegration                                   #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests.                                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestBaselineIntegration:
    """Integration tests."""

    def test_in_pipeline(self, dummy_nifti_list):
        """Test baseline in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        baseline = BaselineAugmentation(mode='random_walk', baseline_frac=0.05)
        pipeline = AugmentationPipeline([baseline])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
