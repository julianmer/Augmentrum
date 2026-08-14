"""
Tests for ResidualWater, SpuriousEchoes, ArtificialPeaks modules.
"""

import pytest
import numpy as np
from augmentrum.augmentation.residual_water import ResidualWater
from augmentrum.augmentation.spurious_echoes import SpuriousEchoes
from augmentrum.augmentation.artificial_peaks import ArtificialPeaks
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend


#**************************************************************************************************#
#                                 Class TestResidualWaterCreation                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test ResidualWater initialization.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestResidualWaterCreation:
    """Test ResidualWater initialization."""

    def test_create_default(self):
        """Test creating with default parameters."""
        water = ResidualWater()
        assert water.center_ppm == 4.7
        assert water.phase_deg == 0.0
        assert water.amplitude_scale == 0.1

    def test_create_custom_peaks(self):
        """Test creating with custom peaks."""
        peaks = ((0.0, 0.25, 1.0), (0.15, 0.20, 0.5))
        water = ResidualWater(peaks=peaks, phase_deg=10.0)
        assert water.peaks == peaks
        assert water.phase_deg == 10.0


#**************************************************************************************************#
#                                     Class TestResidualWater                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test residual water addition.                                                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestResidualWater:
    """Test residual water addition."""

    def test_water_changes_data(self, dummy_nifti_list):
        """Test that water peaks modify data."""
        water = ResidualWater(amplitude_scale=0.2)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = water(nifti_plus, None)
        water_data = result_data[0][:]

        assert not np.allclose(water_data, original_data)

    def test_water_preserves_dtype(self, dummy_nifti_list):
        """Test that water preserves complex dtype."""
        water = ResidualWater()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = water(nifti_plus, None)
        assert np.iscomplexobj(result_data[0][:])


#**************************************************************************************************#
#                                 Class TestSpuriousEchoesCreation                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test SpuriousEchoes initialization.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class TestSpuriousEchoesCreation:
    """Test SpuriousEchoes initialization."""

    def test_create_default(self):
        """Test creating with default parameters."""
        echoes = SpuriousEchoes()
        assert len(echoes.echoes) == 1
        assert echoes.global_phase_deg == 0.0

    def test_create_multiple_echoes(self):
        """Test creating with multiple echoes."""
        echo_list = [(0.1, 0.3, 0.0, 5.0, 0.0), (0.2, 0.15, 10.0, 3.0, 2.0)]
        echoes = SpuriousEchoes(echoes=echo_list, global_phase_deg=5.0)
        assert len(echoes.echoes) == 2
        assert echoes.global_phase_deg == 5.0


#**************************************************************************************************#
#                                     Class TestSpuriousEchoes                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test spurious echoes addition.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestSpuriousEchoes:
    """Test spurious echoes addition."""

    def test_echoes_change_data(self, dummy_nifti_list):
        """Test that echoes modify data."""
        echoes = SpuriousEchoes(echoes=[(0.1, 0.2, 0.0, 5.0, 0.0)])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = echoes(nifti_plus, None)
        echo_data = result_data[0][:]

        assert not np.allclose(echo_data, original_data)

    def test_echoes_preserve_dtype(self, dummy_nifti_list):
        """Test that echoes preserve complex dtype."""
        echoes = SpuriousEchoes()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = echoes(nifti_plus, None)
        assert np.iscomplexobj(result_data[0][:])


#**************************************************************************************************#
#                                Class TestArtificialPeaksCreation                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test ArtificialPeaks initialization.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class TestArtificialPeaksCreation:
    """Test ArtificialPeaks initialization."""

    def test_create_default(self):
        """Test creating with default parameters."""
        peaks = ArtificialPeaks()
        assert len(peaks.peaks) == 1
        assert peaks.ref_ppm == 4.7
        assert peaks.amp_mode == 'real'

    def test_create_multiple_peaks(self):
        """Test creating with multiple peaks."""
        peak_list = [
            {'ppm': 3.0, 'amp': 0.1, 'phase_deg': 0.0, 'lb_hz': 5.0, 'gb_hz': 0.0},
            {'ppm': 2.5, 'amp': 0.05, 'phase_deg': 45.0, 'lb_hz': 3.0, 'gb_hz': 2.0}
        ]
        peaks = ArtificialPeaks(peaks=peak_list)
        assert len(peaks.peaks) == 2


#**************************************************************************************************#
#                                    Class TestArtificialPeaks                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test artificial peaks addition.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class TestArtificialPeaks:
    """Test artificial peaks addition."""

    def test_peaks_change_data(self, dummy_nifti_list):
        """Test that artificial peaks modify data."""
        peaks = ArtificialPeaks(peaks=[
            {'ppm': 3.0, 'amp': 0.1, 'phase_deg': 0.0, 'lb_hz': 5.0, 'gb_hz': 0.0}
        ])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = peaks(nifti_plus, None)
        peak_data = result_data[0][:]

        assert not np.allclose(peak_data, original_data)

    def test_peaks_preserve_dtype(self, dummy_nifti_list):
        """Test that peaks preserve complex dtype."""
        peaks = ArtificialPeaks()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = peaks(nifti_plus, None)
        assert np.iscomplexobj(result_data[0][:])

    def test_voigt_peaks(self, dummy_nifti_list):
        """Test Voigt-shaped peaks (both lb_hz and gb_hz > 0)."""
        peaks = ArtificialPeaks(peaks=[
            {'ppm': 3.0, 'amp': 0.1, 'phase_deg': 0.0, 'lb_hz': 5.0, 'gb_hz': 3.0}
        ])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = peaks(nifti_plus, None)
        assert result_data is not None


#**************************************************************************************************#
#                                  Class TestArtifactsIntegration                                  #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests for artifact modules.                                                          #
#                                                                                                  #
#**************************************************************************************************#
class TestArtifactsIntegration:
    """Integration tests for artifact modules."""

    def test_all_artifacts_in_pipeline(self, dummy_nifti_list):
        """Test all artifact modules in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        water = ResidualWater(amplitude_scale=0.1)
        echoes = SpuriousEchoes(echoes=[(0.1, 0.2, 0.0, 5.0, 0.0)])
        peaks = ArtificialPeaks(peaks=[
            {'ppm': 3.0, 'amp': 0.05, 'phase_deg': 0.0, 'lb_hz': 5.0, 'gb_hz': 0.0}
        ])

        pipeline = AugmentationPipeline([water, echoes, peaks])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#**************************************************************************************************#
#                                    Class TestTurcoWater                                          #
#**************************************************************************************************#
#                                                                                                  #
# The seven-Lorentzian residual water of Turco et al. (WaterFit), as a model preset.               #
#                                                                                                  #
#**************************************************************************************************#
class TestTurcoWater:
    """model='turco' selects the WaterFit peak set; explicit peaks still win."""

    def test_turco_selects_the_seven_seeds(self):
        water = ResidualWater(model='turco')

        assert water.peaks == ResidualWater.TURCO_PEAKS
        assert len(water.peaks) == 7
        # the seeds sit at 4.7 + delta for delta in +-{0, 0.05, 0.10, 0.15}
        offsets = sorted(round(p[0], 2) for p in water.peaks)
        assert offsets == [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

    def test_default_model_is_unchanged(self):
        assert ResidualWater().peaks == ResidualWater.LOBE_PEAKS

    def test_explicit_peaks_beat_the_model(self):
        peaks = ((0.0, 0.3, 1.0),)
        assert ResidualWater(model='turco', peaks=peaks).peaks == peaks

    def test_an_unknown_model_is_refused(self):
        with pytest.raises(ValueError, match="model must be"):
            ResidualWater(model='hlsvd')

    def test_a_per_peak_phase_equals_the_global_phase_for_one_peak(self):
        """With a single lobe the two phase routes must be the same rotation."""
        ppm = np.linspace(3.0, 6.0, 512)

        per_peak = ResidualWater._water_lobe_profile(
            ppm, peaks=((0.0, 0.2, 1.0, 35.0),), phase_deg=0.0)
        global_ph = ResidualWater._water_lobe_profile(
            ppm, peaks=((0.0, 0.2, 1.0),), phase_deg=35.0)

        assert np.allclose(per_peak, global_ph, atol=1e-12)

    def test_turco_water_stays_in_the_water_region(self):
        """Seven merged Lorentzians must still be a water hump, not a baseline."""
        ppm = np.linspace(0.0, 9.4, 2048)
        profile = ResidualWater._water_lobe_profile(
            ppm, peaks=ResidualWater.TURCO_PEAKS)

        magnitude = np.abs(profile)
        inside = (ppm > 4.3) & (ppm < 5.1)
        assert magnitude[inside].max() == pytest.approx(1.0, abs=1e-9)
        assert magnitude[~inside].max() < 0.2
