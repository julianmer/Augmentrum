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
        from augmentrum.augmentation.noise import Noise

        broadening = LineBroadening(lb_hz=10.0)
        noise = Noise(sigma_frac=0.02)
        pipeline = AugmentationPipeline([broadening, noise])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert result_data is not None



#**************************************************************************************************#
#                                  Class TestLorentzianNarrowing                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test Lorentzian narrowing (a negative lb_hz).                                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestLorentzianNarrowing:
    """A negative lb_hz multiplies the FID by exp(+pi |lb| t), warns once, and a cap limits it."""

    SW = 1000.0
    N = 512

    def _t(self):
        return np.arange(self.N) / self.SW

    def test_negative_lb_narrows_exactly(self):
        """lb_hz = -2 multiplies the FID by exp(+2 pi t)."""
        fid = np.ones((1, self.N), complex)
        broadening = LineBroadening(lb_hz=-2.0, mode='lorentzian')
        with pytest.warns(UserWarning, match='narrows the lines'):
            out, _ = broadening.process_tensor(fid, sw_hz=self.SW)
        assert np.allclose(out[0], np.exp(2 * np.pi * self._t()))

    def test_all_negative_per_sample_widths_narrow(self):
        """A batch whose widths are all negative is narrowed, not passed through."""
        fid = np.ones((3, self.N), complex)
        lb = np.array([-0.5, -1.0, -2.0])
        broadening = LineBroadening(lb_hz=lb, mode='voigt')
        with pytest.warns(UserWarning):
            out, _ = broadening.process_tensor(fid, sw_hz=self.SW)
        assert np.allclose(out, np.exp(-np.pi * lb[:, None] * self._t()[None]))

    def test_mixed_per_sample_widths(self):
        """Narrowing, nothing and broadening side by side in one batch."""
        fid = np.ones((3, self.N), complex)
        lb = np.array([-1.0, 0.0, 2.0])
        broadening = LineBroadening(lb_hz=lb, gb_hz=0.0, mode='voigt')
        with pytest.warns(UserWarning):
            out, _ = broadening.process_tensor(fid, sw_hz=self.SW)
        assert np.allclose(out, np.exp(-np.pi * lb[:, None] * self._t()[None]))

    def test_narrowing_undoes_broadening(self):
        """Broadening by 3 Hz and then narrowing by 3 Hz returns the original FID."""
        rng = np.random.default_rng(0)
        fid = rng.standard_normal((2, self.N)) + 1j * rng.standard_normal((2, self.N))
        wide, _ = LineBroadening(lb_hz=3.0, mode='lorentzian').process_tensor(fid, sw_hz=self.SW)
        with pytest.warns(UserWarning):
            back, _ = LineBroadening(lb_hz=-3.0, mode='lorentzian').process_tensor(
                wide, sw_hz=self.SW)
        assert np.allclose(back, fid)

    def test_cap_stops_the_rise(self):
        """With narrow_cap_s the envelope is exp(+pi |lb| min(t, cap))."""
        fid = np.ones((1, self.N), complex)
        broadening = LineBroadening(lb_hz=-1.5, mode='lorentzian', narrow_cap_s=0.1)
        with pytest.warns(UserWarning, match='capped at 0.1 s'):
            out, _ = broadening.process_tensor(fid, sw_hz=self.SW)
        assert np.allclose(out[0], np.exp(1.5 * np.pi * np.minimum(self._t(), 0.1)))

    def test_cap_leaves_broadening_alone(self):
        """The cap only acts on narrowing; positive widths decay over the whole FID."""
        fid = np.ones((2, self.N), complex)
        lb = np.array([-1.0, 2.0])
        broadening = LineBroadening(lb_hz=lb, mode='lorentzian', narrow_cap_s=0.1)
        with pytest.warns(UserWarning):
            out, _ = broadening.process_tensor(fid, sw_hz=self.SW)
        t = self._t()
        assert np.allclose(out[0], np.exp(np.pi * np.minimum(t, 0.1)))
        assert np.allclose(out[1], np.exp(-2 * np.pi * t))

    def test_positive_widths_unchanged(self):
        """Non-negative widths give exactly the envelope they always did, without a warning."""
        import warnings as w
        fid = np.ones((2, self.N), complex)
        lb, gb = np.array([0.0, 2.5]), np.array([1.0, 0.0])
        with w.catch_warnings():
            w.simplefilter('error')
            out, _ = LineBroadening(lb_hz=lb, gb_hz=gb, mode='voigt').process_tensor(
                fid, sw_hz=self.SW)
        t = self._t()[None]
        expected = np.exp(-np.pi * lb[:, None] * t) * np.exp(-(np.pi * gb[:, None] * t) ** 2
                                                              / (4 * np.log(2)))
        assert np.array_equal(out, fid * expected.astype(out.real.dtype))

    def test_warns_once_per_module(self):
        """The warning comes on the first narrowing call only."""
        import warnings as w
        fid = np.ones((1, self.N), complex)
        broadening = LineBroadening(lb_hz=-1.0, mode='lorentzian')
        with w.catch_warnings(record=True) as caught:
            w.simplefilter('always')
            broadening.process_tensor(fid, sw_hz=self.SW)
            broadening.process_tensor(fid, sw_hz=self.SW)
        assert sum('narrows the lines' in str(c.message) for c in caught) == 1

    def test_negative_gaussian_raises(self):
        """A Gaussian cannot be narrowed by the envelope."""
        fid = np.ones((1, self.N), complex)
        with pytest.raises(ValueError, match='gb_hz must be >= 0'):
            LineBroadening(gb_hz=-1.0, mode='gaussian').process_tensor(fid, sw_hz=self.SW)
        with pytest.raises(ValueError, match='gb_hz must be >= 0'):
            LineBroadening(lb_hz=1.0, gb_hz=np.array([1.0, -0.5]), mode='voigt').process_tensor(
                np.ones((2, self.N), complex), sw_hz=self.SW)

    def test_invalid_cap_raises(self):
        with pytest.raises(ValueError, match='narrow_cap_s'):
            LineBroadening(lb_hz=-1.0, narrow_cap_s=0.0)

    def test_nifti_list_path_narrows(self, dummy_nifti_list):
        """On the NIfTI-list backend, narrowing by 5 Hz undoes broadening by 5 Hz."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        original = nifti_plus[0][:].copy()
        wide, _ = LineBroadening(lb_hz=5.0, mode='lorentzian')(nifti_plus, None)
        assert not np.allclose(wide[0][:], original)
        with pytest.warns(UserWarning):
            back, _ = LineBroadening(lb_hz=-5.0, mode='lorentzian')(wide, None)
        assert np.allclose(back[0][:], original)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_torch_per_sample(self, device):
        """Per-sample narrowing on a torch tensor stays on its device and matches NumPy."""
        torch = pytest.importorskip('torch')
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('no CUDA')
        lb = np.array([-1.0, 1.0])
        fid = torch.ones((2, self.N), dtype=torch.complex64, device=device)
        broadening = LineBroadening(lb_hz=lb, mode='lorentzian', narrow_cap_s=0.2)
        with pytest.warns(UserWarning):
            out, _ = broadening.process_tensor(fid, sw_hz=self.SW)
        assert out.device.type == device
        t = self._t()
        expected = np.stack([np.exp(np.pi * np.minimum(t, 0.2)), np.exp(-np.pi * t)])
        assert np.allclose(out.cpu().numpy(), expected, rtol=1e-5)


#**************************************************************************************************#
#                                     Class TestLineshapeKernel                                    #
#**************************************************************************************************#
#                                                                                                  #
# Test the optional lineshape kernel (a B0 distribution convolved into every line).                #
#                                                                                                  #
#**************************************************************************************************#
class TestLineshapeKernel:
    """kernel=None changes nothing; a kernel multiplies the FID by its characteristic function."""

    SW = 1000.0
    N = 512

    def _t(self):
        return np.arange(self.N) / self.SW

    def _fid(self, batch=3, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((batch, self.N)) + 1j * rng.standard_normal((batch, self.N))

    def test_off_by_default(self):
        """Without a kernel the output equals the plain Voigt."""
        fid = self._fid()
        a, _ = LineBroadening(lb_hz=2.0, gb_hz=1.0).process_tensor(fid, sw_hz=self.SW)
        b = fid * np.exp(-np.pi * 2.0 * self._t() - (np.pi * 1.0 * self._t()) ** 2
                         / (4 * np.log(2)))
        assert np.array_equal(a, LineBroadening(lb_hz=2.0, gb_hz=1.0, kernel=None)
                              .process_tensor(fid, sw_hz=self.SW)[0])
        assert np.allclose(a, b)

    def test_single_point_kernel_is_identity(self):
        fid = self._fid()
        out, _ = LineBroadening(kernel=[3.0], kernel_step_hz=1.0).process_tensor(fid, sw_hz=self.SW)
        assert np.allclose(out, fid)

    def test_given_kernel_is_a_convolution(self):
        """Two equal points 4 Hz apart: the mean of the FID shifted by -2 and +2 Hz."""
        fid = self._fid()
        out, _ = LineBroadening(kernel=[1.0, 1.0], kernel_step_hz=4.0).process_tensor(
            fid, sw_hz=self.SW)
        t = self._t()
        expected = fid * 0.5 * (np.exp(-2j * np.pi * 2.0 * t) + np.exp(2j * np.pi * 2.0 * t))
        assert np.allclose(out, expected)

    def test_given_kernel_convolves_the_spectrum(self):
        """The spectrum of the output is the spectrum convolved with the kernel (whole bins)."""
        fid = self._fid(batch=1)[0]
        step = self.SW / self.N                           # one bin
        w = np.array([0.2, 1.0, 0.0, 0.5])
        out, _ = LineBroadening(kernel=w, kernel_step_hz=step).process_tensor(
            fid[None], sw_hz=self.SW)
        spec = np.fft.fft(fid)
        # offsets -1.5 .. +1.5 bins; half a bin more makes them the whole bins -1 .. 2
        half = np.exp(2j * np.pi * 0.5 * step * self._t())
        conv = sum(wk * np.roll(np.fft.fft(fid), k - 1) for k, wk in enumerate(w / w.sum()))
        assert np.allclose(np.fft.fft(out[0] * half), conv)

    def test_one_component_no_spread_is_gaussian(self):
        """A random kernel with one component and zero spread is Gaussian broadening."""
        fid = self._fid()
        a, _ = LineBroadening(kernel='random', kernel_components=1, kernel_spread_hz=0.0,
                              kernel_width_hz=2.5, seed=1).process_tensor(fid, sw_hz=self.SW)
        b, _ = LineBroadening(gb_hz=2.5, mode='gaussian').process_tensor(fid, sw_hz=self.SW)
        assert np.allclose(a, b)

    def test_random_kernel_keeps_area_and_centre(self):
        """Unit area (the first FID point is kept) and zero mean offset (no net shift)."""
        fid = np.ones((64, self.N), complex)
        out, _ = LineBroadening(kernel='random', seed=3).process_tensor(fid, sw_hz=self.SW)
        assert np.allclose(out[:, 0], 1.0)
        # the phase of the second point is 2 pi t1 (mean offset) + O(t1^3 skewness): ~0 Hz
        slope = np.angle(out[:, 1]) * self.SW / (2 * np.pi)
        assert np.all(np.abs(slope) < 1e-4)
        assert np.all(np.abs(out) <= 1.0 + 1e-12)          # a kernel only broadens

    def test_random_kernels_differ_per_sample_and_replay_with_seed(self):
        fid = np.ones((8, self.N), complex)
        a, _ = LineBroadening(kernel='random', seed=5).process_tensor(fid, sw_hz=self.SW)
        b, _ = LineBroadening(kernel='random', seed=5).process_tensor(fid, sw_hz=self.SW)
        assert np.array_equal(a, b)
        assert not np.allclose(a[0], a[1])

    def test_kernel_on_top_of_voigt(self):
        fid = self._fid()
        voigt, _ = LineBroadening(lb_hz=1.0, gb_hz=2.0).process_tensor(fid, sw_hz=self.SW)
        both, _ = LineBroadening(lb_hz=1.0, gb_hz=2.0, kernel=[1.0, 1.0],
                                 kernel_step_hz=2.0).process_tensor(fid, sw_hz=self.SW)
        assert np.allclose(both, voigt * np.cos(2 * np.pi * 1.0 * self._t()))

    @pytest.mark.parametrize('kernel, step', [([-1.0, 2.0], 1.0), ([0.0, 0.0], 1.0),
                                              ([1.0, 1.0], None), ('box', 1.0)])
    def test_invalid_kernels_raise(self, kernel, step):
        with pytest.raises(ValueError):
            LineBroadening(kernel=kernel, kernel_step_hz=step)

    def test_nifti_list_path(self, dummy_nifti_list):
        """One kernel per subject on the NIfTI-list backend; replays with the seed."""
        outs = []
        for _ in range(2):
            nifti_plus = NIfTI_MRS_Plus(nifti_list=[n.copy() if hasattr(n, 'copy') else n
                                                    for n in dummy_nifti_list],
                                        backend=Backend.NIFTI_LIST)
            original = nifti_plus[0][:].copy()
            out, _ = LineBroadening(kernel='random', seed=7)(nifti_plus, None)
            assert not np.allclose(out[0][:], original)
            outs.append(out[0][:].copy())
        assert np.allclose(outs[0], outs[1])

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_torch_matches_numpy(self, device):
        torch = pytest.importorskip('torch')
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('no CUDA')
        fid = self._fid(batch=4)
        a, _ = LineBroadening(lb_hz=1.0, kernel='random', seed=2).process_tensor(
            fid, sw_hz=self.SW)
        b, _ = LineBroadening(lb_hz=1.0, kernel='random', seed=2).process_tensor(
            torch.as_tensor(fid, device=device), sw_hz=self.SW)
        assert b.device.type == device
        assert np.allclose(b.cpu().numpy(), a)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
