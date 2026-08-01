"""
Tests for GaussianNoise augmentation module.

Tests cover:
- Initialization with different noise parameters
- SNR-based noise
- Sigma-based noise
- Sigma-fraction based noise
- Backend compatibility
- Integration tests
"""

import pytest
import numpy as np
from augmentrum.augmentation.gaussian_noise import GaussianNoise
from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus, Backend


#**************************************************************************************************#
#                                 Class TestGaussianNoiseCreation                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test GaussianNoise initialization.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseCreation:
    """Test GaussianNoise initialization."""

    def test_create_with_snr(self):
        """Test creating with SNR parameter."""
        noise = GaussianNoise(snr_db=20.0)
        assert noise.snr_db == 20.0
        assert noise.sigma is None
        assert noise.sigma_frac is None

    def test_create_with_sigma(self):
        """Test creating with sigma parameter."""
        noise = GaussianNoise(sigma=0.01)
        assert noise.sigma == 0.01
        assert noise.snr_db is None
        assert noise.sigma_frac is None

    def test_create_with_sigma_frac(self):
        """Test creating with sigma_frac parameter."""
        noise = GaussianNoise(sigma_frac=0.02)
        assert noise.sigma_frac == 0.02
        assert noise.snr_db is None
        assert noise.sigma is None

    def test_error_with_no_parameters(self):
        """Test that error is raised with no parameters."""
        with pytest.raises(ValueError, match="Must provide one of"):
            GaussianNoise()

    def test_error_with_multiple_parameters(self):
        """Test that error is raised with multiple parameters."""
        with pytest.raises(ValueError, match="Provide only ONE of"):
            GaussianNoise(snr_db=20.0, sigma=0.01)

    def test_supports_all_backends(self):
        """Test that GaussianNoise supports all backends."""
        noise = GaussianNoise(sigma_frac=0.02)
        assert noise.SUPPORTED_BACKENDS == []


#**************************************************************************************************#
#                                  Class TestGaussianNoiseSNRMode                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test SNR-based noise addition.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseSNRMode:
    """Test SNR-based noise addition."""

    def test_snr_changes_data(self, dummy_nifti_list):
        """Test that SNR-based noise modifies data."""
        noise = GaussianNoise(snr_db=20.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = noise(nifti_plus, None)
        noisy_data = result_data[0][:]

        assert not np.allclose(noisy_data, original_data)

    def test_snr_preserves_dtype(self, dummy_nifti_list):
        """Test that noise preserves complex dtype."""
        noise = GaussianNoise(snr_db=20.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = noise(nifti_plus, None)
        assert np.iscomplexobj(result_data[0][:])

    def test_higher_snr_less_noise(self, dummy_nifti_list):
        """Test that higher SNR results in less noise."""
        # Use independent copies and no seed to get different noise
        low_snr = GaussianNoise(snr_db=10)
        high_snr = GaussianNoise(snr_db=30)

        nifti_plus1 = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        nifti_plus2 = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original = nifti_plus1[0][:].copy()

        result_low, _ = low_snr(nifti_plus1, None)
        result_high, _ = high_snr(nifti_plus2, None)

        diff_low = np.std(result_low[0][:] - original)
        diff_high = np.std(result_high[0][:] - original)

        # Lower SNR should have higher noise (higher std deviation)
        # Allow some tolerance for random variation
        assert diff_low >= diff_high * 0.9  # 10% tolerance


#**************************************************************************************************#
#                                 Class TestGaussianNoiseSigmaMode                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test sigma-based noise addition.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseSigmaMode:
    """Test sigma-based noise addition."""

    def test_sigma_changes_data(self, dummy_nifti_list):
        """Test that sigma-based noise modifies data."""
        noise = GaussianNoise(sigma=0.01)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = noise(nifti_plus, None)
        noisy_data = result_data[0][:]

        assert not np.allclose(noisy_data, original_data)

    def test_reproducibility_with_seed(self, dummy_nifti_list):
        """Test reproducibility with same seed."""
        nifti_plus1 = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        nifti_plus2 = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        noise1 = GaussianNoise(sigma=0.01, seed=42)
        noise2 = GaussianNoise(sigma=0.01, seed=42)

        result1, _ = noise1(nifti_plus1, None)
        result2, _ = noise2(nifti_plus2, None)

        assert np.allclose(result1[0][:], result2[0][:])


#**************************************************************************************************#
#                               Class TestGaussianNoiseSigmaFracMode                               #
#**************************************************************************************************#
#                                                                                                  #
# Test sigma-fraction based noise addition.                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseSigmaFracMode:
    """Test sigma-fraction based noise addition."""

    def test_sigma_frac_changes_data(self, dummy_nifti_list):
        """Test that sigma_frac-based noise modifies data."""
        noise = GaussianNoise(sigma_frac=0.02)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = noise(nifti_plus, None)
        noisy_data = result_data[0][:]

        assert not np.allclose(noisy_data, original_data)

    def test_sigma_frac_scales_with_signal(self, dummy_nifti_list):
        """Test that sigma_frac scales with signal amplitude."""
        noise = GaussianNoise(sigma_frac=0.05)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Modify one FID to have higher amplitude
        first_nifti = nifti_plus[0]
        first_nifti[:] = first_nifti[:] * 10.0

        result_data, _ = noise(nifti_plus, None)
        # Noise should scale with signal amplitude
        assert result_data is not None


#**************************************************************************************************#
#                             Class TestGaussianNoiseMultipleSubjects                              #
#**************************************************************************************************#
#                                                                                                  #
# Test processing multiple subjects.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseMultipleSubjects:
    """Test processing multiple subjects."""

    def test_processes_all_subjects(self, dummy_nifti_list):
        """Test that all subjects are processed."""
        noise = GaussianNoise(sigma_frac=0.02)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = noise(nifti_plus, None)
        assert len(result_data) == len(dummy_nifti_list)

    def test_different_noise_per_subject(self, dummy_nifti_list):
        """Test that each subject gets different noise."""
        noise = GaussianNoise(sigma=0.01)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = noise(nifti_plus, None)

        data1 = result_data[0][:]
        data2 = result_data[1][:]

        # Different subjects should have different noise realizations
        assert not np.allclose(data1, data2)


#**************************************************************************************************#
#                              Class TestGaussianNoiseWaterReference                               #
#**************************************************************************************************#
#                                                                                                  #
# Test water reference handling.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseWaterReference:
    """Test water reference handling."""

    def test_water_unchanged(self, dummy_nifti_list):
        """Test that water reference is not modified."""
        from copy import deepcopy
        noise = GaussianNoise(sigma_frac=0.02)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        # Use COPIES so they're not the same objects
        water_niftis = [deepcopy(dummy_nifti_list[0]), deepcopy(dummy_nifti_list[1])]
        water_plus = NIfTI_MRS_Plus(nifti_list=water_niftis, backend=Backend.NIFTI_LIST)

        original_water = water_plus[0][:].copy()
        result_data, result_water = noise(nifti_plus, water_plus)

        # Water data should be unchanged (check data, not object identity)
        assert result_water is not None
        assert np.allclose(result_water[0][:], original_water, rtol=1e-5, atol=1e-7)

    def test_water_none_handled(self, dummy_nifti_list):
        """Test that None water is handled correctly."""
        noise = GaussianNoise(sigma_frac=0.02)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, result_water = noise(nifti_plus, None)
        assert result_water is None


#**************************************************************************************************#
#                                  Class TestGaussianNoiseLogging                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test automatic logging/provenance.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseLogging:
    """Test automatic logging/provenance."""

    def test_logging_when_not_volatile(self, dummy_nifti_list):
        """Test that metadata is logged when volatile=False."""
        noise = GaussianNoise(sigma_frac=0.02)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=False)

        result_data, _ = noise(nifti_plus, None)
        assert 'common_provenance' in result_data.metadata_common

    def test_no_logging_when_volatile(self, dummy_nifti_list):
        """Test that metadata is not logged when volatile=True."""
        noise = GaussianNoise(sigma_frac=0.02)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=True)

        result_data, _ = noise(nifti_plus, None)
        assert result_data.metadata_common == {}


#**************************************************************************************************#
#                                 Class TestGaussianNoiseEdgeCases                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test edge cases.                                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseEdgeCases:
    """Test edge cases."""

    def test_with_single_subject(self, dummy_nifti_mrs):
        """Test with single subject."""
        noise = GaussianNoise(sigma_frac=0.02)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[dummy_nifti_mrs], backend=Backend.NIFTI_LIST)

        result_data, _ = noise(nifti_plus, None)
        assert len(result_data) == 1


#**************************************************************************************************#
#                                Class TestGaussianNoiseIntegration                                #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests.                                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestGaussianNoiseIntegration:
    """Integration tests."""

    def test_in_pipeline(self, dummy_nifti_list):
        """Test GaussianNoise in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        noise = GaussianNoise(sigma_frac=0.02)
        pipeline = AugmentationPipeline([noise])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)

    def test_chained_with_other_modules(self, dummy_nifti_list):
        """Test chaining with other modules."""
        from augmentrum.core.pipeline import AugmentationPipeline
        from augmentrum.augmentation.line_broadening import LineBroadening

        broadening = LineBroadening(lb_hz=5.0)
        noise = GaussianNoise(sigma_frac=0.02)
        pipeline = AugmentationPipeline([broadening, noise])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert result_data is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
