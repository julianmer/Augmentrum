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
        assert water.center_ppm is None, "None means the nucleus' reference (4.65 ppm for 1H)"
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
        assert peaks.ref_ppm is None, "None means the nucleus' reference (4.65 ppm for 1H)"
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
        """
        Seven merged Lorentzians must still be a water hump, not a baseline.

        Judged on the absorption part: the lobes are causal, so they carry the
        dispersive tails a water residual has, which fall off as 1/distance
        and are not what makes a hump a baseline.
        """
        ppm = ppm_axis(N_PTS, SW_HZ, SF_MHZ, '1H')
        profile = ResidualWater._water_lobe_profile(
            ppm, peaks=ResidualWater.TURCO_PEAKS)

        inside = (ppm > 4.3) & (ppm < 5.1)
        assert np.abs(profile[inside]).max() == pytest.approx(1.0, abs=1e-9)
        assert np.abs(np.real(profile[~inside])).max() < 0.2


#*************#
#   helpers   #
#*************#
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl_mrs.core import MRS
from augmentrum.core.pipeline import AugmentationPipeline
from augmentrum.processing.utils import ppm_axis, ppm_reference

N_PTS, SW_HZ, SF_MHZ = 2048, 4000.0, 123.26


def _lorentzian_fid(ppm=2.01, lb_hz=6.0, nucleus='1H', n=N_PTS, sw=SW_HZ, sf=SF_MHZ):
    """A single resonance at *ppm* on the FSL-MRS axis, so there is a peak to scale by."""
    t = np.arange(n) / sw
    f0 = (ppm - ppm_reference(nucleus)) * sf
    return (np.exp(2j * np.pi * f0 * t) * np.exp(-np.pi * lb_hz * t)).astype(np.complex64)


def _batch(fid, n_subjects, backend=Backend.NUMPY, nucleus='1H', sw=SW_HZ, sf=SF_MHZ):
    """*n_subjects* copies of one FID, so any difference between samples is the module's."""
    niftis = [gen_nifti_mrs(fid.reshape(1, 1, 1, -1).copy(), 1 / sw, sf, nucleus=nucleus)
              for _ in range(n_subjects)]
    return NIfTI_MRS_Plus(niftis, backend=backend, volatile=True)


def _values(plus):
    return np.asarray(plus.get_data(Backend.NUMPY))


def _fsl_spectrum(fid, nucleus='1H', sw=SW_HZ, sf=SF_MHZ):
    """The spectrum and ppm axis exactly as FSL-MRS shows them."""
    mrs = MRS(FID=np.asarray(fid).squeeze(), cf=sf, bw=sw, nucleus=nucleus)
    return mrs.getAxes(), mrs.get_spec()


def _fsl_bin(sw=SW_HZ, sf=SF_MHZ, n=N_PTS):
    return sw / (n - 1) / sf


#**************************************************************************************************#
#                                     Class TestPpmReference                                       #
#**************************************************************************************************#
#                                                                                                  #
# One ppm axis, the FSL-MRS / NIfTI-MRS one, for every module that places a feature.               #
#                                                                                                  #
#**************************************************************************************************#
class TestPpmReference:
    """User ppm values mean what an FSL-MRS plot shows."""

    def test_reference_follows_the_nucleus(self):
        assert ppm_reference('1H') == 4.65
        assert ppm_reference('2H') == 4.65
        assert ppm_reference('31P') == 0.0
        assert ppm_reference('13C') == 0.0
        assert ppm_reference(None) == 4.65

    def test_unknown_nucleus_warns_and_uses_zero(self):
        with pytest.warns(UserWarning, match="19F"):
            assert ppm_reference('19F') == 0.0

    def test_ppm_axis_labels_every_bin_as_fsl_mrs_does(self):
        """
        The modules work on fftshift(ifft(fid)), FSL-MRS shows fftshift(fft(fid)):
        bin j of one is bin (-j) mod n of the other. Every bin but the Nyquist
        one must carry exactly the ppm FSL-MRS prints for it.
        """
        fsl = MRS(FID=np.ones(N_PTS, complex), cf=SF_MHZ, bw=SW_HZ, nucleus='1H').getAxes()
        axis = ppm_axis(N_PTS, SW_HZ, SF_MHZ, '1H')

        j = np.arange(1, N_PTS)
        assert np.allclose(axis[j], fsl[(-j) % N_PTS], atol=1e-9)
        assert np.all(np.diff(axis) < 0), "descending, with the Nyquist alias on top"
        assert np.isclose(axis[0], axis[1] + _fsl_bin()), "Nyquist bin carries its +sw/2 alias"

    def test_ppm_axis_is_referenced_per_nucleus(self):
        proton = ppm_axis(N_PTS, SW_HZ, SF_MHZ, '1H')
        phosphorus = ppm_axis(N_PTS, SW_HZ, 51.7, '31P')
        assert np.isclose(proton[N_PTS // 2] - 4.65, (phosphorus[N_PTS // 2]) * 51.7 / SF_MHZ)

    def test_a_peak_requested_at_1_30_ppm_lands_there_on_the_fsl_axis(self):
        """The symptom: with a 4.7 ppm reference the peak landed at 1.25 ppm."""
        peaks = ArtificialPeaks(peaks=[{'ppm': 1.30, 'amp': 0.5, 'lb_hz': 3.0}])
        data = _batch(_lorentzian_fid(), 1)
        added = _values(peaks(data)[0])[0, 0, 0, 0] - _values(data)[0, 0, 0, 0]

        axis, spectrum = _fsl_spectrum(added)
        assert abs(axis[np.argmax(np.real(spectrum))] - 1.30) <= _fsl_bin()

    def test_31p_places_peaks_from_a_zero_reference(self):
        """On 31P the carrier is 0 ppm, so 1.30 ppm is 1.30 ppm above it."""
        sf = 51.7
        fid = _lorentzian_fid(ppm=-2.5, nucleus='31P', sf=sf)
        peaks = ArtificialPeaks(peaks=[{'ppm': 1.30, 'amp': 0.5, 'lb_hz': 3.0}])
        data = _batch(fid, 1, nucleus='31P', sf=sf)
        added = _values(peaks(data)[0])[0, 0, 0, 0] - _values(data)[0, 0, 0, 0]

        axis, spectrum = _fsl_spectrum(added, nucleus='31P', sf=sf)
        assert abs(axis[np.argmax(np.real(spectrum))] - 1.30) <= _fsl_bin(sf=sf)

    def test_an_explicit_ref_ppm_still_overrides(self):
        """ref_ppm=4.7 is the legacy convention: the peak then sits 0.05 ppm low on FSL's axis."""
        peaks = ArtificialPeaks(peaks=[{'ppm': 1.30, 'amp': 0.5, 'lb_hz': 3.0}], ref_ppm=4.7)
        data = _batch(_lorentzian_fid(), 1)
        added = _values(peaks(data)[0])[0, 0, 0, 0] - _values(data)[0, 0, 0, 0]

        axis, spectrum = _fsl_spectrum(added)
        assert abs(axis[np.argmax(np.real(spectrum))] - 1.25) <= _fsl_bin()

    def test_residual_water_sits_on_the_water(self):
        """The default water is at 4.65 ppm on the FSL axis, not 0.05 ppm off it."""
        data = _batch(_lorentzian_fid(), 1)
        added = _values(ResidualWater(amplitude_scale=0.5)(data)[0])[0, 0, 0, 0] \
            - _values(data)[0, 0, 0, 0]

        axis, spectrum = _fsl_spectrum(added)
        assert abs(axis[np.argmax(np.abs(spectrum))] - 4.65) <= _fsl_bin()


#**************************************************************************************************#
#                                   Class TestPerSampleProfiles                                    #
#**************************************************************************************************#
#                                                                                                  #
# A batch carries a spread of artefacts, not one artefact copied onto every sample.                #
#                                                                                                  #
#**************************************************************************************************#
class TestPerSampleProfiles:
    """Ranged parameters are drawn per sample; fixed ones give identical samples."""

    @staticmethod
    def _added(module, n_subjects=4, backend=Backend.NUMPY, **user_kwargs):
        data = _batch(_lorentzian_fid(), n_subjects, backend=backend)
        pipe = AugmentationPipeline([module], user_kwargs=user_kwargs)
        out, _ = pipe(data, None, batch_params=pipe.sample_batch_parameters(n_subjects))
        return (_values(out) - _values(data))[:, 0, 0, 0, :]

    @staticmethod
    def _all_differ(added):
        return all(not np.allclose(added[i], added[j], atol=1e-6)
                   for i in range(len(added)) for j in range(i + 1, len(added)))

    def test_residual_water_varies_per_sample_in_a_pipeline(self):
        added = self._added(ResidualWater(), amplitude_scale=(0.05, 0.3),
                            center_ppm=(4.55, 4.75), phase_deg=(-30, 30))
        assert self._all_differ(added)

    def test_residual_water_declares_its_per_sample_parameters(self):
        assert set(ResidualWater.PER_SAMPLE_PARAMS) == {'amplitude_scale', 'phase_deg',
                                                        'center_ppm'}

    def test_fixed_residual_water_is_the_same_on_every_sample(self):
        added = self._added(ResidualWater(amplitude_scale=0.2))
        assert all(np.allclose(added[0], row, atol=1e-6) for row in added)

    def test_artificial_peaks_draw_a_peak_per_sample(self):
        """The default is a random lipid-like peak: every sample gets its own."""
        added = self._added(ArtificialPeaks(seed=0))
        assert self._all_differ(added)

    def test_fixed_artificial_peaks_are_the_same_on_every_sample(self):
        peaks = [{'ppm': 3.0, 'amp': 0.1, 'lb_hz': 5.0}]
        added = self._added(ArtificialPeaks(peaks=peaks, seed=0))
        assert all(np.allclose(added[0], row, atol=1e-6) for row in added)

    def test_the_default_peak_is_a_random_lipid(self):
        default = ArtificialPeaks().peaks[0]
        assert default['ppm'] == (0.9, 1.6)
        assert default['amp'] == (0.05, 0.3)
        assert default['lb_hz'] == (5.0, 20.0)

    def test_ranges_stay_inside_their_bounds_and_scalars_repeat(self):
        module = ArtificialPeaks(peaks=[{'ppm': (0.8, 1.6), 'amp': (0.05, 0.3), 'lb_hz': 7.0,
                                         'gb_hz': (0.0, 4.0), 'phase_deg': (-30, 30)}], seed=1)
        (drawn,) = module._draw(256)
        assert drawn['ppm'].min() >= 0.8 and drawn['ppm'].max() <= 1.6
        assert drawn['amp'].min() >= 0.05 and drawn['amp'].max() <= 0.3
        assert drawn['phase_deg'].min() >= -30 and drawn['phase_deg'].max() <= 30
        assert np.all(drawn['lb_hz'] == 7.0)
        assert len(np.unique(drawn['ppm'])) == 256, "a range is drawn, not repeated"

    def test_the_amplitude_alias_is_accepted(self):
        module = ArtificialPeaks(peaks=[{'ppm': 3.0, 'amplitude': 0.2, 'lb_hz': 5.0}])
        assert module.peaks[0]['amp'] == 0.2

    def test_spurious_echoes_draw_an_echo_per_sample(self):
        echoes = [{'delay_s': (0.02, 0.2), 'amp': (0.1, 0.3), 'phase_deg': (-90, 90)}]
        added = self._added(SpuriousEchoes(echoes=echoes, seed=0))
        assert self._all_differ(added)

    def test_a_sample_shares_its_profile_across_coils(self):
        """Per sample, not per trace: every coil of a sample sees the same peak."""
        fid = _lorentzian_fid()
        volume = np.repeat(fid.reshape(1, 1, 1, -1)[..., None], 3, axis=-1)
        niftis = []
        for _ in range(2):
            nifti = gen_nifti_mrs(volume.copy(), 1 / SW_HZ, SF_MHZ)
            nifti.set_dim_tag(4, 'DIM_COIL')
            niftis.append(nifti)
        data = NIfTI_MRS_Plus(niftis, backend=Backend.NUMPY, volatile=True)

        added = _values(ArtificialPeaks(seed=0)(data)[0]) - _values(data)
        assert np.allclose(added[..., 0], added[..., 2], atol=1e-6)
        assert not np.allclose(added[0], added[1], atol=1e-6)


#**************************************************************************************************#
#                                   Class TestListEqualsTensor                                     #
#**************************************************************************************************#
#                                                                                                  #
# The NIfTI-list path and the tensor path are one computation written twice.                       #
#                                                                                                  #
#**************************************************************************************************#
class TestListEqualsTensor:
    """Same seed, same numbers, whichever path and backend runs it."""

    @staticmethod
    def _spectra(data):
        return np.fft.fftshift(np.fft.ifft(_values(data), axis=-1), axes=-1)

    @pytest.mark.parametrize("make", [
        lambda: ArtificialPeaks(seed=5),
        lambda: ResidualWater(amplitude_scale=0.2, phase_deg=20.0),
    ], ids=['ArtificialPeaks', 'ResidualWater'])
    def test_frequency_modules_agree_on_both_paths(self, make):
        data = _batch(_lorentzian_fid(), 3, backend=Backend.NIFTI_LIST)
        listed, _ = make().process_nifti_list([n.copy() for n in data.list()])
        from_list = np.stack([n[:] for n in listed])

        spectra, _ = make().process_tensor(self._spectra(data), sw_hz=SW_HZ, sf_mhz=SF_MHZ,
                                           nucleus='1H')
        from_tensor = np.fft.fft(np.fft.ifftshift(spectra, axes=-1), axis=-1)

        assert np.allclose(from_list, from_tensor, atol=1e-5)

    def test_per_sample_vectors_reach_the_list_path(self):
        module = ResidualWater()
        module.amplitude_scale = np.array([0.1, 0.2, 0.3])
        data = _batch(_lorentzian_fid(), 3, backend=Backend.NIFTI_LIST)
        listed, _ = module.process_nifti_list([n.copy() for n in data.list()])
        added = np.stack([n[:] for n in listed]) - _values(data)
        peak = np.abs(np.fft.ifft(added, axis=-1)).max(axis=-1).ravel()

        assert np.allclose(peak / peak[0], [1.0, 2.0, 3.0], rtol=1e-3)

    @pytest.mark.parametrize("make", [
        lambda: SpuriousEchoes(echoes=[{'delay_s': (0.02, 0.2), 'amp': (0.1, 0.3),
                                        'phase_deg': (-90, 90), 'freq_hz': (-20, 20)}], seed=2),
        lambda: SpuriousEchoes(mode='hybrid', echoes=[{'tau': (0.01, 0.05), 'alpha': 0.3,
                                                       'T2': 0.01, 'df_hz': 30.0}], seed=2),
        lambda: SpuriousEchoes(mode='echo', echoes=[{'alpha': 0.3, 't_echo': (0.05, 0.2),
                                                     'T2': 0.03}], seed=2),
        lambda: ArtificialPeaks(seed=2),
        lambda: ResidualWater(model='turco', amplitude_scale=0.3),
    ], ids=['replica', 'hybrid', 'echo', 'ArtificialPeaks', 'ResidualWater'])
    def test_nifti_list_numpy_and_torch_give_the_same_batch(self, make):
        pytest.importorskip('torch')
        outputs = [_values(make()(_batch(_lorentzian_fid(), 3, backend=backend))[0])
                   for backend in (Backend.NIFTI_LIST, Backend.NUMPY, Backend.PYTORCH)]
        assert np.allclose(outputs[0], outputs[1], atol=1e-5)
        assert np.allclose(outputs[1], outputs[2], atol=1e-5)


#**************************************************************************************************#
#                                       Class TestReplica                                          #
#**************************************************************************************************#
#                                                                                                  #
# 'replica' is a delayed copy of the FID - a ghost - not a rescaled copy of the spectrum.          #
#                                                                                                  #
#**************************************************************************************************#
class TestReplica:
    """A replica at tau modulates the spectrum with period 1/tau Hz."""

    TAU, AMP = 0.05, 0.3

    @staticmethod
    def _fid(n=2048, sw=2000.0):
        t = np.arange(n) / sw
        return np.exp(-t * 15.0).astype(np.complex128)[None, :]

    def _spectra(self, sw=2000.0):
        fid = self._fid(sw=sw)
        out, _ = SpuriousEchoes(mode='replica', echoes=[{'delay_s': self.TAU, 'amp': self.AMP,
                                                         'decay_hz': 0.0}]).process_tensor(
            fid, sw_hz=sw)
        freq = np.fft.fftshift(np.fft.fftfreq(fid.shape[-1], 1 / sw))
        return freq, np.fft.fftshift(np.fft.fft(fid[0])), np.fft.fftshift(np.fft.fft(out[0]))

    def test_replica_modulates_the_spectrum_with_period_one_over_tau(self):
        freq, before, after = self._spectra()
        ratio = np.abs(after) / np.abs(before)
        for f_hz in (0.0, 10.0, 20.0, 30.0, 40.0):
            expected = abs(1 + self.AMP * np.exp(-2j * np.pi * f_hz * self.TAU))
            assert np.isclose(ratio[np.argmin(np.abs(freq - f_hz))], expected, atol=0.02), \
                f"ratio at {f_hz} Hz should be {expected:.3f}"

    def test_the_difference_is_not_the_spectrum(self):
        """The old model gave "fid + fid * envelope": a phased copy of the whole spectrum."""
        _, before, after = self._spectra()
        corr = abs(np.corrcoef(np.real(after - before), np.real(before))[0, 1])
        assert corr < 0.6, f"the replica's delta correlates {corr:.3f} with the spectrum"

    def test_the_ghost_starts_at_the_delay_with_the_fid_s_first_point(self):
        sw, n = 2000.0, 1024
        fid = self._fid(n=n, sw=sw)
        out, _ = SpuriousEchoes(mode='replica',
                                echoes=[{'delay_s': 0.1, 'amp': 0.3, 'phase_deg': 90.0,
                                         'decay_hz': 0.0}]).process_tensor(fid, sw_hz=sw)
        ghost = np.asarray(out)[0] - fid[0]
        shift = int(round(0.1 * sw))

        assert np.allclose(ghost[:shift], 0.0)
        assert np.isclose(ghost[shift], 0.3 * np.exp(1j * np.pi / 2) * fid[0, 0])
        assert np.allclose(ghost[shift:], 0.3j * fid[0, :n - shift])

    def test_the_default_replica_is_a_real_echo(self):
        """The default echo (0.1 s) must be a shifted copy, not a step envelope."""
        sw = 2000.0
        fid = self._fid(n=1024, sw=sw)
        out, _ = SpuriousEchoes(mode='replica').process_tensor(fid, sw_hz=sw)
        ghost = np.asarray(out)[0] - fid[0]
        assert np.allclose(ghost[:200], 0.0)
        assert not np.allclose(ghost[200:], 0.0)

    def test_hybrid_is_unchanged_and_matches_its_numpy_path(self):
        fid = self._fid(n=1024, sw=2000.0)
        echo = [{'tau': 0.02, 'alpha': 0.3, 'phase_deg': 25.0, 't_echo': 0.05, 'T2': 0.03,
                 'df_hz': 35.0}]
        module = SpuriousEchoes(mode='hybrid', echoes=echo)
        tensor_out, _ = module.process_tensor(fid, sw_hz=2000.0)

        t = np.arange(1024) / 2000.0
        delayed = np.zeros(1024, complex)
        delayed[40:] = fid[0, :-40]
        mod = np.exp(1j * (2 * np.pi * 35.0 * (t - 0.02) + np.deg2rad(25.0)))
        ghost = 0.3 * np.max(np.abs(fid)) * (delayed / np.max(np.abs(fid))) \
            * np.exp(-np.abs(t - 0.05) / 0.03) * mod
        assert np.allclose(np.asarray(tensor_out)[0], fid[0] + ghost, atol=1e-6)
        assert np.allclose(module._add_echoes(fid[0], 2000.0), fid[0] + ghost, atol=1e-6)

    def test_echo_ranges_stay_inside_their_bounds(self):
        module = SpuriousEchoes(mode='replica',
                                echoes=[{'delay_s': (0.05, 0.1), 'amp': (0.1, 0.2),
                                         'decay_hz': 4.0}], seed=3)
        (drawn,) = module._draw(128, 2000.0)
        assert drawn['delay_s'].min() >= 0.05 - 1e-9 and drawn['delay_s'].max() <= 0.1 + 1e-9
        assert drawn['amp'].min() >= 0.1 and drawn['amp'].max() <= 0.2
        assert np.all(drawn['decay_hz'] == 4.0)
        assert np.all(drawn['shift'] == np.round(drawn['delay_s'] * 2000.0))


#**************************************************************************************************#
#                                   Class TestLocalizedEcho                                        #
#**************************************************************************************************#
#                                                                                                  #
# The default: SMART MRS's localized echo, drawn per sample, placed on the ppm axis.               #
#                                                                                                  #
#**************************************************************************************************#
class TestLocalizedEcho:
    """The localized echo is the default, placed in ppm, and can hit a few transients."""

    @staticmethod
    def _run(module, plus, water=None):
        pipe = AugmentationPipeline([module])
        out, _ = pipe(plus, water, batch_params=pipe.sample_batch_parameters(len(plus)))[:2]
        return _values(out)

    def test_the_default_is_a_random_localized_echo(self):
        module = SpuriousEchoes(seed=0)
        assert module.mode == 'echo'
        (drawn,) = module._draw(256, SW_HZ, n_points=N_PTS, sf_mhz=SF_MHZ, nucleus='1H')
        t_acq = (N_PTS - 1) / SW_HZ
        assert 0.1 * t_acq - 1e-9 <= drawn['t_echo'].min() <= drawn['t_echo'].max() <= 0.9 * t_acq
        assert 0.01 <= drawn['T2'].min() <= drawn['T2'].max() <= 0.05
        ppm = drawn['freq_hz'] / SF_MHZ + ppm_reference('1H')
        assert 0.0 <= ppm.min() <= ppm.max() <= 8.0
        assert 0.02 <= drawn['amp'].min() <= drawn['amp'].max() <= 0.2
        assert np.unique(drawn['t_echo']).size == 256, "every sample draws its own echo"

    def test_the_registry_name_gives_the_localized_echo(self):
        from augmentrum import Augmentrum
        module, fixed = Augmentrum.resolve_module('spurious_echoes')
        assert module(**fixed).mode == 'echo'

    def test_legacy_tuples_still_mean_replicas(self):
        assert SpuriousEchoes(echoes=[(0.1, 0.2, 0.0, 5.0, 0.0)]).mode == 'replica'
        assert SpuriousEchoes(mode='replica').echoes == [SpuriousEchoes.REPLICA_DEFAULT]

    @pytest.mark.parametrize("ppm", [0.9, 1.3, 3.5, 6.0])
    def test_an_echo_requested_in_ppm_peaks_there_on_the_fsl_axis(self, ppm):
        plus = _batch(_lorentzian_fid(ppm=2.01), 1)
        before = _values(plus)[0, 0, 0, 0]
        echo = SpuriousEchoes(echoes=[{'ppm': ppm, 't_echo': 0.2, 'T2': 0.05, 'amp': 0.5}])
        added = self._run(echo, plus)[0, 0, 0, 0] - before
        axis, spec = _fsl_spectrum(added)
        assert abs(axis[np.argmax(np.abs(spec))] - ppm) <= 1.5 * _fsl_bin()

    def test_the_echo_is_centred_at_t_echo_and_follows_eq_1(self):
        plus = _batch(_lorentzian_fid(), 1)
        before = _values(plus)[0, 0, 0, 0]
        echo = SpuriousEchoes(echoes=[{'ppm': 1.3, 't_echo': 0.25, 'T2': 0.03, 'amp': 0.1,
                                       'phase_deg': 40.0}])
        added = self._run(echo, plus)[0, 0, 0, 0] - before
        t = np.arange(N_PTS) / SW_HZ
        f = (1.3 - 4.65) * SF_MHZ
        expected = (0.1 * np.abs(before).max() * np.exp(-np.abs(t - 0.25) / 0.03)
                    * np.exp(1j * (2 * np.pi * f * t + np.deg2rad(40.0))))
        assert np.allclose(added, expected, atol=1e-6 * np.abs(expected).max())

    def test_the_phase_turns_the_echo_and_leaves_its_size(self):
        """SMART MRS's code adds the phase outside the exponent, where it scales the echo."""
        plus = _batch(_lorentzian_fid(), 1)
        before = _values(plus)[0, 0, 0, 0]
        added = {}
        for phase in (0.0, 180.0):
            echo = SpuriousEchoes(echoes=[{'ppm': 1.3, 't_echo': 0.2, 'amp': 0.1,
                                           'phase_deg': phase}])
            added[phase] = self._run(echo, _batch(_lorentzian_fid(), 1))[0, 0, 0, 0] - before
        assert np.allclose(added[180.0], -added[0.0], atol=1e-6)

    def test_31p_places_the_echo_from_a_zero_reference(self):
        plus = _batch(_lorentzian_fid(ppm=0.0, nucleus='31P', sf=49.9), 1, nucleus='31P',
                      sf=49.9)
        before = _values(plus)[0, 0, 0, 0]
        echo = SpuriousEchoes(echoes=[{'ppm': -10.0, 't_echo': 0.2, 'T2': 0.05, 'amp': 0.5}])
        added = self._run(echo, plus)[0, 0, 0, 0] - before
        axis, spec = _fsl_spectrum(added, nucleus='31P', sf=49.9)
        assert abs(axis[np.argmax(np.abs(spec))] + 10.0) <= 1.5 * _fsl_bin(sf=49.9)

    @pytest.mark.parametrize("kwargs, message", [
        ({'echoes': [{'ppm': 1.0, 'freq_hz': 10.0}]}, "not both"),
        ({'mode': 'replica', 'echoes': [{'ppm': 1.0}]}, "freq_hz"),
        ({'mode': 'replica', 'echoes': [{'t_echo_frac': 0.5}]}, "replica"),
        ({'transient_fraction': 0.0}, "transient_fraction"),
    ])
    def test_inconsistent_settings_are_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SpuriousEchoes(**kwargs)

    @staticmethod
    def _transients(n_dyn=8, n_coils=2, subjects=3, backend=Backend.NUMPY):
        """Uncombined data with coils and transients, one object per subject."""
        rng = np.random.default_rng(0)
        niftis = []
        for _ in range(subjects):
            data = (rng.standard_normal((1, 1, 1, 256, n_coils, n_dyn))
                    + 1j * rng.standard_normal((1, 1, 1, 256, n_coils, n_dyn)))
            nifti = gen_nifti_mrs(data.astype(np.complex64), 1 / 2000.0, 123.26,
                                  dim_tags=['DIM_COIL', 'DIM_DYN', None])
            niftis.append(nifti)
        return niftis

    @pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.NIFTI_LIST])
    def test_a_fraction_of_the_transients_carries_the_echo(self, backend):
        niftis = self._transients()
        before = np.stack([n[:] for n in niftis])
        plus = NIfTI_MRS_Plus([n.copy() for n in niftis], backend=backend, volatile=True)
        module = SpuriousEchoes(transient_fraction=0.25, seed=0)
        pipe = AugmentationPipeline([module])
        out, _ = pipe(plus, None, batch_params=pipe.sample_batch_parameters(3))[:2]
        after = np.stack([n[:] for n in out.list()])

        changed = ~np.all(np.isclose(after, before), axis=(1, 2, 3, 4))    # (subject, coil, dyn)
        for subject in changed:
            hit = np.flatnonzero(subject.any(axis=0))
            assert hit.size == 2, "a quarter of 8 transients"
            assert np.all(subject[:, hit]), "every coil of a hit transient carries it"
        assert not np.array_equal(changed[0], changed[1]) or not np.array_equal(changed[1],
                                                                                changed[2])

    def test_the_transient_fraction_draws_equally_on_both_engines(self):
        niftis = self._transients()
        results = []
        for backend in (Backend.NUMPY, Backend.NIFTI_LIST):
            plus = NIfTI_MRS_Plus([n.copy() for n in niftis], backend=backend, volatile=True)
            pipe = AugmentationPipeline([SpuriousEchoes(transient_fraction=0.5, seed=7)])
            out, _ = pipe(plus, None, batch_params=pipe.sample_batch_parameters(3))[:2]
            results.append(np.stack([n[:] for n in out.list()]))
        assert np.allclose(results[0], results[1], atol=1e-5)

    def test_without_transients_the_fraction_warns_and_hits_everything(self):
        plus = _batch(_lorentzian_fid(), 2)
        before = _values(plus)
        with pytest.warns(UserWarning, match="no transient"):
            after = self._run(SpuriousEchoes(transient_fraction=0.5, seed=0), plus)
        assert not np.any(np.all(np.isclose(after, before), axis=-1))


#**************************************************************************************************#
#                                 Class TestSeededReproducibility                                  #
#**************************************************************************************************#
#                                                                                                  #
# A seed reproduces the batch on every backend, and the backends agree.                            #
#                                                                                                  #
#**************************************************************************************************#
class TestSeededReproducibility:
    """The per-sample draws come from one NumPy generator per call, whatever the backend."""

    @pytest.mark.parametrize("make", [
        lambda seed: ArtificialPeaks(seed=seed),
        lambda seed: SpuriousEchoes(echoes=[{'delay_s': (0.02, 0.2), 'amp': (0.1, 0.3),
                                             'phase_deg': (-90, 90)}], seed=seed),
    ], ids=['ArtificialPeaks', 'SpuriousEchoes'])
    def test_seeded_runs_reproduce_on_numpy_and_torch(self, make):
        pytest.importorskip('torch')
        first = _values(make(11)(_batch(_lorentzian_fid(), 3, backend=Backend.NUMPY))[0])
        again = _values(make(11)(_batch(_lorentzian_fid(), 3, backend=Backend.NUMPY))[0])
        torch_out = _values(make(11)(_batch(_lorentzian_fid(), 3, backend=Backend.PYTORCH))[0])
        other = _values(make(12)(_batch(_lorentzian_fid(), 3, backend=Backend.NUMPY))[0])

        assert np.allclose(first, again)
        assert np.allclose(first, torch_out, atol=1e-5)
        assert not np.allclose(first, other, atol=1e-6)

    def test_consecutive_batches_differ_under_a_seed(self):
        module = ArtificialPeaks(seed=11)
        first = _values(module(_batch(_lorentzian_fid(), 2))[0])
        second = _values(module(_batch(_lorentzian_fid(), 2))[0])
        assert not np.allclose(first, second, atol=1e-6)


#**************************************************************************************************#
#                                       Class TestCausality                                        #
#**************************************************************************************************#
#                                                                                                  #
# What a module adds is a signal: it starts at the first point of the FID and decays.              #
#                                                                                                  #
#**************************************************************************************************#
class TestCausality:
    """
    A resonance is a decaying complex exponential in the FID, so nothing of it
    belongs at the end of the acquisition. A lineshape drawn on the axis as a
    real curve has a two-sided FID instead, half of it wrapped to the end,
    which is where it used to ring after zero-filling or truncation.
    """

    @staticmethod
    def _added(module, **user_kwargs):
        """What *module* adds to one spectrum, as the time-domain output of a pipeline."""
        data = _batch(_lorentzian_fid(), 1)
        pipe = AugmentationPipeline([module], user_kwargs=user_kwargs)
        out, _ = pipe(data, None, batch_params=pipe.sample_batch_parameters(1))
        return (_values(out) - _values(data))[0, 0, 0, 0]

    @staticmethod
    def _second_half(fid):
        """The fraction of the FID's energy in its second half: ~0 causal, 0.5 white."""
        fid = np.asarray(fid).ravel()
        return float(np.sum(np.abs(fid[fid.size // 2:]) ** 2) / np.sum(np.abs(fid) ** 2))

    @pytest.mark.parametrize("make", [
        lambda: ArtificialPeaks(peaks=[{'ppm': 1.3, 'amp': 0.3, 'lb_hz': 10.0}]),
        lambda: ArtificialPeaks(peaks=[{'ppm': 1.3, 'amp': 0.3, 'gb_hz': 10.0}]),
        lambda: ArtificialPeaks(peaks=[{'ppm': 1.3, 'amp': 0.3, 'lb_hz': 6.0, 'gb_hz': 8.0,
                                        'phase_deg': 40.0}]),
        lambda: ArtificialPeaks(seed=0),
        lambda: ResidualWater(amplitude_scale=1.0),
        lambda: ResidualWater(model='turco', amplitude_scale=1.0, phase_deg=30.0),
    ], ids=['lorentzian', 'gaussian', 'voigt', 'default_peak', 'water_lobes', 'water_turco'])
    def test_what_is_added_is_causal(self, make):
        assert self._second_half(self._added(make())) <= 0.02

    def test_the_fid_of_a_lorentzian_peak_decays_as_a_signal_model_says(self):
        """The added FID is "A exp(2 pi i f t) exp(-pi lb t)": its envelope is exp(-pi lb t)."""
        lb_hz = 10.0
        fid = self._added(ArtificialPeaks(peaks=[{'ppm': 1.3, 'amp': 0.3, 'lb_hz': lb_hz}]))
        t = np.arange(N_PTS) / SW_HZ
        envelope = np.abs(fid[:N_PTS // 2]) / np.abs(fid[0])
        assert np.allclose(envelope, np.exp(-np.pi * lb_hz * t[:N_PTS // 2]), atol=1e-3)

    @pytest.mark.parametrize("peak, fwhm_hz", [
        ({'ppm': 1.3, 'amp': 0.5, 'lb_hz': 12.0}, 12.0),
        ({'ppm': 1.3, 'amp': 0.5, 'gb_hz': 12.0}, 12.0),
    ], ids=['lorentzian', 'gaussian'])
    def test_the_width_is_the_fwhm_on_the_fsl_axis(self, peak, fwhm_hz):
        """lb_hz and gb_hz are the FWHM of the absorption line in Hz, to a bin."""
        axis, spectrum = _fsl_spectrum(self._added(ArtificialPeaks(peaks=[peak])))
        real = np.real(spectrum)
        above = np.flatnonzero(real >= 0.5 * real.max())
        fwhm = (abs(axis[above[-1]] - axis[above[0]]) + _fsl_bin()) * SF_MHZ
        assert abs(fwhm - fwhm_hz) <= _fsl_bin() * SF_MHZ

    def test_amplitude_is_still_the_fraction_of_the_peak(self):
        """A causal peak at amp 0.5 still adds a real height of half the spectrum's peak."""
        data = _batch(_lorentzian_fid(), 1)
        peaks = ArtificialPeaks(peaks=[{'ppm': 1.3, 'amp': 0.5, 'lb_hz': 10.0}])
        spectra = [np.fft.fftshift(np.fft.ifft(_values(x)[0, 0, 0, 0]))
                   for x in (data, peaks(data)[0])]
        added = spectra[1] - spectra[0]
        assert np.max(np.real(added)) / np.max(np.real(spectra[0])) == pytest.approx(0.5, rel=1e-3)

    def test_lobes_are_weighed_by_area_so_a_narrower_lobe_stands_taller(self):
        """rel_amp is the lobe's amplitude in the FID, as in the WaterFit model."""
        ppm = ppm_axis(N_PTS, SW_HZ, SF_MHZ, '1H')
        both = ResidualWater._water_lobe_profile(ppm, peaks=((0.0, 0.1, 1.0), (0.5, 0.2, 1.0)))

        at = lambda x: np.real(both)[np.argmin(np.abs(ppm - (4.65 + x)))]
        assert at(0.0) == pytest.approx(2.0 * at(0.5), rel=0.05)
