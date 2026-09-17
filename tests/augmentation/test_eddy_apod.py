"""
Tests for EddyCurrent and Apodization modules.
"""

import pytest
import numpy as np
from augmentrum.augmentation.eddy_current import EddyCurrent
from augmentrum.augmentation.apodization import Apodization
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend


#**************************************************************************************************#
#                                  Class TestEddyCurrentCreation                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test EddyCurrent initialization.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                  Class TestSyntheticEddyCurrent                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test synthetic eddy current.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                Class TestWaterDerivedEddyCurrent                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test water-derived eddy current.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                  Class TestApodizationCreation                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test Apodization initialization.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                 Class TestTruncationApodization                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test truncation apodization.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestTruncationApodization:
    """Test truncation apodization."""

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

    def test_truncate_with_frac(self, dummy_nifti_list):
        """Test truncation with fraction of points."""
        apod = Apodization(mode='truncate', frac_pts=0.5)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_size = nifti_plus[0].shape[-1]
        result_data, _ = apod(nifti_plus, None)
        new_size = result_data[0].shape[-1]

        # Should be approximately half
        assert new_size <= original_size // 2 + 1


#**************************************************************************************************#
#                                 Class TestExponentialApodization                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test exponential apodization.                                                                    #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                  Class TestEddyApodIntegration                                   #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests.                                                                               #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                   Class TestApodizationWindows                                   #
#**************************************************************************************************#
#                                                                                                  #
# The gaussian and hamming windows next to the original truncate/exponential.                      #
#                                                                                                  #
#**************************************************************************************************#
class TestApodizationWindows:
    """The window set: every mode is a multiply by w(t), except truncation."""

    @staticmethod
    def _fid(n=512, sw_hz=2000.0):
        t = np.arange(n) / sw_hz
        return (np.exp(-15.0 * t) * np.exp(1j * 2 * np.pi * 40.0 * t)).astype(
            np.complex64)[None, :]

    def test_gaussian_window_matches_the_formula(self):
        from augmentrum.augmentation import Apodization

        fid, sw = self._fid(), 2000.0
        out, _ = Apodization(mode='gaussian', gb_hz=8.0).process_tensor(fid, sw_hz=sw)

        t = np.arange(fid.shape[-1]) / sw
        expected = fid * np.exp(-((np.pi * 8.0 * t) ** 2) / (4 * np.log(2)))
        assert np.allclose(out, expected, atol=1e-6)

    def test_gaussian_needs_a_width(self):
        from augmentrum.augmentation import Apodization

        with pytest.raises(ValueError, match="gb_hz"):
            Apodization(mode='gaussian').process_tensor(self._fid(), sw_hz=2000.0)

    def test_hamming_window_decays_from_one(self):
        from augmentrum.augmentation import Apodization

        fid = self._fid()
        out, _ = Apodization(mode='hamming').process_tensor(fid, sw_hz=2000.0)

        ratio = np.abs(np.asarray(out)[0]) / np.abs(fid[0])
        assert ratio[0] == pytest.approx(1.0, abs=1e-6)
        assert ratio[-1] == pytest.approx(0.08, abs=1e-6)
        assert np.all(np.diff(ratio) <= 1e-6), "hamming half-window must decay"

    def test_unknown_mode_is_refused(self):
        from augmentrum.augmentation import Apodization

        with pytest.raises(ValueError, match="mode must be"):
            Apodization(mode='hann')


#**************************************************************************************************#
#                                     Class TestEchoMode                                           #
#**************************************************************************************************#
#                                                                                                  #
# The standalone localized echo of Berrington 2021 / SMART MRS.                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestEchoMode:
    """mode='echo': an independent additive echo, not a replica of the FID."""

    @staticmethod
    def _fid(n=1024, sw_hz=2000.0):
        t = np.arange(n) / sw_hz
        return (np.exp(-15.0 * t) * np.exp(1j * 2 * np.pi * 40.0 * t)).astype(
            np.complex64)[None, :]

    def test_echo_matches_the_smart_formula(self):
        from augmentrum.augmentation import SpuriousEchoes

        fid, sw = self._fid(), 2000.0
        echo = {'alpha': 0.3, 't_echo': 0.2, 'T2': 0.05,
                'freq_hz': 60.0, 'phase_deg': 30.0}
        out, _ = SpuriousEchoes(mode='echo', echoes=[echo]).process_tensor(
            fid, sw_hz=sw)

        t = np.arange(fid.shape[-1]) / sw
        ghost = (0.3 * np.max(np.abs(fid))
                 * np.exp(-np.abs(t - 0.2) / 0.05)
                 * np.exp(1j * (2 * np.pi * 60.0 * t + np.deg2rad(30.0))))
        assert np.allclose(np.asarray(out)[0], fid[0] + ghost, atol=1e-5)

    def test_echo_is_independent_of_the_fid_shape(self):
        """Unlike replica/hybrid, zeroing the late FID must not zero the echo."""
        from augmentrum.augmentation import SpuriousEchoes

        fid = self._fid()
        gated = fid.copy()
        gated[..., 200:] = 0.0

        echo = [{'alpha': 0.3, 't_echo': 0.4, 'T2': 0.03}]
        out, _ = SpuriousEchoes(mode='echo', echoes=echo).process_tensor(
            gated, sw_hz=2000.0)

        # The echo is centered at 0.4 s = sample 800, far beyond the gate
        assert np.max(np.abs(np.asarray(out)[0, 700:900])) > 0.0

    def test_tensor_and_list_paths_agree(self):
        from augmentrum.augmentation import SpuriousEchoes

        fid = self._fid()
        echo = [{'alpha': 0.2, 't_echo': 0.1, 'T2': 0.04, 'freq_hz': -25.0}]

        module = SpuriousEchoes(mode='echo', echoes=echo)
        tensor_out, _ = module.process_tensor(fid, sw_hz=2000.0)
        numpy_out = module._add_echoes(fid[0], 2000.0)

        assert np.allclose(np.asarray(tensor_out)[0], numpy_out, atol=1e-5)


#*************#
#   helpers   #
#*************#
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

N_PTS, SW_HZ, SF_MHZ = 2048, 4000.0, 123.26


def _water(coefficient, n=N_PTS, sw=SW_HZ, t2=0.06, tau=0.03, offset_hz=5.0):
    """
    An uncorrected water: a decaying eddy-current phase on top of a frequency
    offset and a constant phase, decaying into a noise floor like a real one.
    """
    t = np.arange(n) / sw
    rng = np.random.default_rng(int(abs(coefficient) * 1000))
    phase = coefficient * np.exp(-t / tau) + 2 * np.pi * offset_hz * t + 0.4
    fid = np.exp(-t / t2) * np.exp(1j * phase)
    return fid + 1e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


LIBRARY = [_water(c) for c in (0.6, -0.9, 1.3, -0.4)]


def _ones(n_subjects, backend=Backend.NUMPY, n=N_PTS, sw=SW_HZ):
    """Constant FIDs: whatever phase comes out is the trajectory that was applied."""
    niftis = [gen_nifti_mrs(np.ones((1, 1, 1, n), np.complex64), 1 / sw, SF_MHZ)
              for _ in range(n_subjects)]
    return NIfTI_MRS_Plus(niftis, backend=backend, volatile=True)


def _phases(plus):
    values = np.asarray(plus.get_data(Backend.NUMPY))
    return np.unwrap(np.angle(values), axis=-1)[:, 0, 0, 0, :]


#**************************************************************************************************#
#                                     Class TestEddySource                                         #
#**************************************************************************************************#
#                                                                                                  #
# A library of uncorrected waters, one trajectory drawn per sample.                                #
#                                                                                                  #
#**************************************************************************************************#
class TestEddySource:
    """mode='water' with a source library."""

    def test_a_source_implies_water_mode_and_refuses_synthetic(self):
        assert EddyCurrent(source=LIBRARY).mode == 'water'
        assert EddyCurrent().mode == 'synthetic'
        with pytest.raises(ValueError, match="source"):
            EddyCurrent(mode='synthetic', source=LIBRARY)
        with pytest.raises(ValueError, match="empty"):
            EddyCurrent(source=[])

    def test_draws_differ_within_a_batch(self):
        module = EddyCurrent(source=LIBRARY, seed=0)
        phases = _phases(module(_ones(8))[0])
        distinct = {tuple(np.round(row[::64], 4)) for row in phases}
        assert len(distinct) >= 2, "every sample got the same trajectory"
        assert len(distinct) <= len(LIBRARY)

    def test_the_library_is_what_is_drawn(self):
        module = EddyCurrent(source=LIBRARY, seed=0)
        library = module._trajectories(N_PTS, SW_HZ)
        for row in _phases(module(_ones(6))[0]):
            assert any(np.allclose(row, entry, atol=1e-5) for entry in library)

    def test_same_seed_reproduces_and_seeds_differ(self):
        first = _phases(EddyCurrent(source=LIBRARY, seed=4)(_ones(6))[0])
        again = _phases(EddyCurrent(source=LIBRARY, seed=4)(_ones(6))[0])
        other = _phases(EddyCurrent(source=LIBRARY, seed=5)(_ones(6))[0])
        assert np.allclose(first, again)
        assert not np.allclose(first, other, atol=1e-6)

    def test_numpy_torch_and_nifti_list_agree(self):
        pytest.importorskip('torch')
        outputs = [_phases(EddyCurrent(source=LIBRARY, seed=4)(_ones(5, backend=backend))[0])
                   for backend in (Backend.NUMPY, Backend.PYTORCH, Backend.NIFTI_LIST)]
        assert np.allclose(outputs[0], outputs[1], atol=1e-5)
        assert np.allclose(outputs[0], outputs[2], atol=1e-5)

    def test_without_a_source_the_call_time_water_is_used(self):
        module = EddyCurrent(mode='water')
        water = NIfTI_MRS_Plus(
            [gen_nifti_mrs(LIBRARY[2].reshape(1, 1, 1, -1).astype(np.complex64), 1 / SW_HZ,
                           SF_MHZ)], backend=Backend.NUMPY, volatile=True)
        phase = _phases(module(_ones(1), water)[0])[0]
        expected = module._ec_phase_from_water(LIBRARY[2].astype(np.complex64), SW_HZ)
        assert np.allclose(phase, expected, atol=1e-5)
        assert np.abs(phase).max() > 0.1, "an uncorrected water must leave a trajectory"

    def test_a_library_of_one_equals_the_call_time_water(self):
        water = NIfTI_MRS_Plus(
            [gen_nifti_mrs(LIBRARY[0].reshape(1, 1, 1, -1).astype(np.complex64), 1 / SW_HZ,
                           SF_MHZ)], backend=Backend.NUMPY, volatile=True)
        at_call = _phases(EddyCurrent(mode='water')(_ones(2), water)[0])
        from_source = _phases(EddyCurrent(source=[LIBRARY[0].astype(np.complex64)])(_ones(2))[0])
        assert np.allclose(at_call, from_source, atol=1e-5)

    def test_water_mode_without_source_still_needs_water(self):
        with pytest.raises(ValueError, match="Water reference required"):
            EddyCurrent(mode='water')(_ones(1), None)

    def test_trajectories_start_at_zero_and_hold_after_the_water_decays(self):
        """
        Unwrapping a decayed water random-walks through noise; the phase is
        read while the water is above the floor and held from there.
        """
        module = EddyCurrent(source=LIBRARY)
        (phase,) = module._trajectories(N_PTS, SW_HZ)[:1]
        magnitude = np.abs(LIBRARY[0])
        last = int(np.flatnonzero(magnitude < module.WATER_FLOOR * magnitude.max())[0])

        assert phase[0] == 0.0
        assert np.allclose(phase[last + 200:], phase[-1], atol=1e-3)
        assert np.abs(phase[:last]).max() > 0.1

    def test_a_corrected_water_leaves_almost_nothing(self):
        """The symptom: after ECC the water's phase is flat, so 'water' mode did nothing."""
        t = np.arange(N_PTS) / SW_HZ
        corrected = np.exp(-t / 0.06) * np.exp(1j * (2 * np.pi * 5.0 * t + 0.4))
        phase = EddyCurrent(source=[corrected])._trajectories(N_PTS, SW_HZ)[0]
        assert np.abs(phase).max() < 1e-3

    def test_sources_are_resampled_to_the_data_grid(self):
        short = [_water(0.6, n=1024, sw=2000.0)]
        nifti_source = [gen_nifti_mrs(short[0].reshape(1, 1, 1, -1), 1 / 2000.0, SF_MHZ)]

        module = EddyCurrent(source=nifti_source)
        (resampled,) = module._trajectories(N_PTS, SW_HZ)
        (own_grid,) = module._trajectories(1024, 2000.0)
        assert resampled.shape == (N_PTS,)
        assert np.allclose(resampled, np.interp(np.arange(N_PTS) / SW_HZ,
                                                np.arange(1024) / 2000.0, own_grid,
                                                right=own_grid[-1]))

        # a bare array is taken to share the data's dwell time
        bare = EddyCurrent(source=short)._trajectories(1024, 2000.0)[0]
        assert np.allclose(bare, own_grid)

    def test_strength_is_read_per_sample(self):
        module = EddyCurrent(source=[LIBRARY[0]], seed=0)
        module.strength = np.array([0.5, 1.0, 2.0])
        phases = _phases(module(_ones(3))[0])
        assert np.allclose(phases[1], 2 * phases[0], atol=1e-5)
        assert np.allclose(phases[2], 4 * phases[0], atol=1e-5)
        assert 'strength' in EddyCurrent.PER_SAMPLE_PARAMS

    def test_a_multicoil_data_batch_gets_one_trajectory_per_sample(self):
        volume = np.ones((1, 1, 1, 512, 3), np.complex64)
        niftis = []
        for _ in range(2):
            nifti = gen_nifti_mrs(volume.copy(), 1 / SW_HZ, SF_MHZ)
            nifti.set_dim_tag(4, 'DIM_COIL')
            niftis.append(nifti)
        data = NIfTI_MRS_Plus(niftis, backend=Backend.NUMPY, volatile=True)
        out = np.asarray(EddyCurrent(source=LIBRARY, seed=0)(data)[0].get_data(Backend.NUMPY))
        assert out.shape == (2, 1, 1, 1, 512, 3)
        assert np.allclose(out[..., 0], out[..., 2])


#**************************************************************************************************#
#                                Class TestSyntheticTrajectory                                     #
#**************************************************************************************************#
#                                                                                                  #
# The synthetic trajectory is a stationary low-passed process anchored at zero.                    #
#                                                                                                  #
#**************************************************************************************************#
class TestSyntheticTrajectory:
    """mode='synthetic' after the fix."""

    def test_phase_starts_at_zero(self):
        phases = _phases(EddyCurrent(seed=0)(_ones(4))[0])
        assert np.allclose(phases[:, 0], 0.0, atol=1e-6)

    def test_no_startup_transient(self):
        """
        Filtering exactly N points let filtfilt start from the first raw noise
        sample, a 0.6 rad transient decaying over the first 100 ms; the
        trajectory's early excursion must be of the low-passed process' size.
        """
        module = EddyCurrent(seed=0)
        rows = np.stack([module._synth_ec_phase(N_PTS, SW_HZ, np.random.default_rng(k))
                         for k in range(40)])
        early = np.abs(rows[:, :400]).max(axis=1).mean()
        assert early < 0.3, f"{early:.2f} rad within 100 ms is a filter transient"
        assert rows.std(axis=1).mean() < 0.15

    def test_a_batch_of_trajectories_is_the_draws_one_by_one(self):
        """The batch takes the same noise in the same order and gives the same bits."""
        module = EddyCurrent(seed=0)
        batch = module._synth_ec_phases(5, N_PTS, SW_HZ, np.random.default_rng(4))
        rng = np.random.default_rng(4)
        rows = np.stack([module._synth_ec_phase(N_PTS, SW_HZ, rng) for _ in range(5)])
        assert np.array_equal(batch, rows)

    def test_list_and_tensor_paths_agree(self):
        first = _phases(EddyCurrent(seed=3)(_ones(3, backend=Backend.NUMPY))[0])
        listed = _phases(EddyCurrent(seed=3)(_ones(3, backend=Backend.NIFTI_LIST))[0])
        assert np.allclose(first, listed, atol=1e-5)
        assert not np.allclose(first[0], first[1], atol=1e-6), "one trajectory per sample"
