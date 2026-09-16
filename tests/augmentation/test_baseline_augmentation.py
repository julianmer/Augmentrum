"""
Tests for BaselineAugmentation module.

Tests cover:
- Random Walk baseline
- B-Spline baseline
- Polynomial baseline
- Backend compatibility
- Integration tests
- The two processing paths agreeing, and the native math agreeing across backends
- Per-sample parameters, the ppm axis, and the cached operators
- Every baseline being the spectrum of a causal signal
"""

import pytest
import numpy as np
from augmentrum.augmentation.baseline_augmentation import BaselineAugmentation
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from nifti_mrs_plus.core import DataState


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

    def test_bspline_changes_data(self, dummy_nifti_list):
        """Test that B-spline baseline modifies data."""
        baseline = BaselineAugmentation(mode='bspline', knots_per_ppm=8, baseline_frac=0.10)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = baseline(nifti_plus, None)
        augmented_data = result_data[0][:]

        assert not np.allclose(augmented_data, original_data)

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


#***********************#
#   the size it asked   #
#***********************#
def test_every_mode_responds_to_baseline_frac():
    """
    All three modes are documented to scale by it, so all three must.

    The polynomial mode used to ignore it entirely: it fitted the spectrum and
    added the raw fit, which came out about the size of the signal itself. That
    is invisible unless the added amount is measured against the peak. The
    parameter is defined on the real part - the real baseline against the real
    peak - so that is what is measured; the imaginary part of a complex
    baseline is free to be larger.
    """
    plus = _batch(1, n_pts=512)
    spectrum = _spectrum(plus)
    peak = np.abs(spectrum.real).max()

    for mode in ('random_walk', 'bspline', 'polynomial'):
        added = []
        for frac in (0.05, 0.20):
            out, _ = BaselineAugmentation(mode=mode, baseline_frac=frac, seed=1)(plus)
            added.append(np.abs((_spectrum(out) - spectrum).real).max() / peak)

        assert added[0] <= 0.05 + 1e-6, f"{mode} added {added[0]:.3f} of the peak for frac=0.05"
        assert added[1] > added[0], f"{mode} ignored baseline_frac"


#***************#
#   fixtures    #
#***************#
N_PTS, SW_HZ, SF_MHZ = 1024, 2000.0, 123.0


def _fid(n_pts=N_PTS, seed=0, dtype=np.complex128):
    """A phased three-peak FID with a little noise, so real peaks are real peaks."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_pts) / SW_HZ
    fid = sum(np.exp(2j * np.pi * (ppm - 4.65) * SF_MHZ * t) * np.exp(-t / 0.08)
              for ppm in (2.01, 3.03, 3.2))
    fid = fid + 0.02 * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
    return fid.astype(dtype)


def _batch(n_subjects=2, n_pts=N_PTS, backend=Backend.NUMPY, dtype=np.complex128,
           coils=None):
    """A volatile batch of single-voxel subjects on *backend*."""
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    objs = []
    for i in range(n_subjects):
        fid = _fid(n_pts, seed=i, dtype=dtype).reshape(1, 1, 1, n_pts)
        if coils:
            fid = np.stack([fid * (c + 1) for c in range(coils)], axis=-1)
        nifti = gen_nifti_mrs(fid, 1 / SW_HZ, SF_MHZ)
        if coils:
            nifti.set_dim_tag(4, 'DIM_COIL')
        objs.append(nifti)
    return NIfTI_MRS_Plus(nifti_list=objs, backend=backend, volatile=True)


def _spectrum(plus):
    """The package's spectra, "(batch, ..., n_points)", as NumPy."""
    from augmentrum.processing import DomainTransform
    out, _ = DomainTransform(spectral='frequency')(plus)
    return np.asarray(out.get_data(Backend.NUMPY))


def _in_frequency_domain(plus, backend):
    """
    The batch moved to the frequency domain and handed over on *backend*.

    A time-domain batch never reaches "process_nifti_list": the base class
    moves it through a tensor backend first. A list batch already in the
    frequency state is what does, so that is what the list path is tested on.
    """
    from augmentrum.processing import DomainTransform
    moved, _ = DomainTransform(spectral='frequency')(plus)
    return NIfTI_MRS_Plus(nifti_list=moved.list(), backend=backend, volatile=True,
                          state=moved.state)


def _fsl_view(fid):
    """FSL-MRS's axis and spectrum for one FID: what a user sees the data on."""
    from fsl_mrs.core import MRS
    mrs = MRS(FID=np.asarray(fid).ravel(), cf=SF_MHZ, bw=SW_HZ, nucleus='1H')
    return mrs.getAxes(), mrs.get_spec()


def _shared_draws(module, seed=0):
    """
    Replace the module's backend-native draw with one NumPy stream.

    "SeedGenerator" streams differ between frameworks by design, so the only
    way to compare the native math across backends is to hand both sides the
    same numbers and let each form its baseline from them.
    """
    from nifti_mrs_plus import ops
    stream = np.random.default_rng(seed)

    def draw(shape, like):
        return ops.match_backend(stream.standard_normal(tuple(shape)), like)

    module._draw = draw


#**************************************************************************************************#
#                                  Class TestPolynomialModel                                       #
#**************************************************************************************************#
#                                                                                                  #
# The polynomial mode is a random polynomial of the stated order, confined to its windows.         #
#                                                                                                  #
#**************************************************************************************************#
class TestPolynomialModel:
    """The polynomial mode is a random polynomial of the stated order."""

    @pytest.mark.parametrize("order", [2, 5])
    def test_order_k_needs_degree_k(self, order):
        """
        A degree-k fit reproduces the real baseline to rounding; degree k-1 cannot.

        The unit fit axis runs from -1 to 1 with ppm over the window, as
        FSL-MRS defines it, so a plain polynomial fit in that variable must
        recover the real part exactly. The imaginary part is not a polynomial:
        it is the real part's Hilbert transform.
        """
        from scipy.signal import hilbert

        plus = _batch(1)
        before = _spectrum(plus)[0].ravel()
        out, _ = BaselineAugmentation(mode='polynomial', order=order, seed=3)(plus)
        added = _spectrum(out)[0].ravel() - before

        from augmentrum.processing.utils import ppm_axis
        ppm = ppm_axis(added.size, SW_HZ, SF_MHZ, '1H')
        x = 2.0 * (ppm - ppm.min()) / (ppm.max() - ppm.min()) - 1.0
        residual = {}
        for degree in (order - 1, order):
            coeffs = np.polynomial.polynomial.polyfit(x, added.real, degree)
            fit = np.polynomial.polynomial.polyval(x, coeffs)
            residual[degree] = np.abs(added.real - fit).max() / np.abs(added.real).max()

        assert residual[order] < 1e-10, f"degree {order} left {residual[order]:.1e}"
        assert residual[order - 1] > 1e-3, f"degree {order - 1} fitted an order-{order} baseline"
        assert np.allclose(added, hilbert(added.real), atol=1e-12 * np.abs(added).max())

    def test_scaled_and_dc_free(self):
        """The real baseline peaks at exactly baseline_frac of the real peak, averaging zero."""
        plus = _batch(2)
        before = _spectrum(plus)
        out, _ = BaselineAugmentation(mode='polynomial', baseline_frac=0.07, seed=5)(plus)
        added = _spectrum(out) - before

        for b in range(2):
            ratio = np.abs(added[b].real).max() / np.abs(before[b].real).max()
            assert abs(ratio - 0.07) < 1e-9
            assert abs(added[b].real.mean()) < 1e-9 * np.abs(added[b].real).max()
            assert abs(added[b].imag.mean()) < 1e-9 * np.abs(added[b].imag).max()

    def test_confined_to_windows_on_the_fsl_axis(self):
        """
        Outside the windows the real baseline is zero, judged on FSL-MRS's own axis.

        The windows are given in either order, as a user would write them. The
        imaginary part is the Hilbert transform of the windowed curve and
        reaches beyond the windows, as the dispersion of anything confined
        must; only the absorption part is confined.
        """
        windows = [(4.0, 3.5), (1.8, 0.8)]
        plus = _batch(1)
        fid_before = plus.list()[0][:].copy()
        out, _ = BaselineAugmentation(mode='polynomial', order=3, ppm_windows=windows,
                                      baseline_frac=0.1, seed=2)(plus)
        axis, spec_before = _fsl_view(fid_before)
        _, spec_after = _fsl_view(out.list()[0][:])
        delta = np.abs(np.real(spec_after - spec_before))
        dispersion = np.abs(np.imag(spec_after - spec_before))

        inside = np.zeros(axis.size, bool)
        for a, b in windows:
            inside |= (axis >= min(a, b)) & (axis <= max(a, b))

        assert delta[~inside].max() < 1e-9 * delta.max(), "baseline leaked outside its windows"
        assert np.mean(delta[inside] > 1e-6 * delta.max()) > 0.9, "windows barely touched"
        assert dispersion[~inside].max() > 1e-3 * delta.max(), "dispersion reaches beyond"

    def test_window_too_narrow_raises(self):
        """A window with fewer points than the order can hold is a mistake, not a fit."""
        plus = _batch(1)
        module = BaselineAugmentation(mode='polynomial', order=6, ppm_windows=[(3.00, 3.01)])
        with pytest.raises(ValueError, match="too few"):
            module(plus)

    def test_default_order_is_fsl_mrs_default(self):
        """FSL-MRS fits a second-order baseline by default; the default here mirrors it."""
        assert BaselineAugmentation(mode='polynomial').order == 2


#**************************************************************************************************#
#                                    Class TestBSplineModel                                        #
#**************************************************************************************************#
#                                                                                                  #
# The B-spline mode is a smooth curve, exactly scaled, with its operator built once per axis.     #
#                                                                                                  #
#**************************************************************************************************#
class TestBSplineModel:
    """The B-spline mode is a smooth curve, exactly scaled, built once per axis."""

    def test_scaled_and_dc_free(self):
        """The real baseline peaks at exactly baseline_frac of the real peak, averaging zero."""
        plus = _batch(2)
        before = _spectrum(plus)
        out, _ = BaselineAugmentation(mode='bspline', baseline_frac=0.07, seed=5)(plus)
        added = _spectrum(out) - before

        for b in range(2):
            ratio = np.abs(added[b].real).max() / np.abs(before[b].real).max()
            assert abs(ratio - 0.07) < 1e-9
            assert abs(added[b].real.mean()) < 1e-9 * np.abs(added[b].real).max()

    def test_smooth(self):
        """A penalised spline is far smoother than the white noise it is made from."""
        plus = _batch(1)
        before = _spectrum(plus)[0].ravel()
        out, _ = BaselineAugmentation(mode='bspline', seed=5)(plus)
        added = (_spectrum(out)[0].ravel() - before).real

        # Bin 0 is the Nyquist bin and holds the far end of the axis, so the
        # one step allowed to jump is the one out of it. A curve smooth at the
        # ppm scale (some sixty bins) has a relative second difference of
        # order (2 pi / 60)^2; white noise of the same size has one near 2.
        roughness = np.abs(np.diff(added[1:], 2)).max() / np.abs(added).max()
        assert roughness < 0.05

    def test_operator_cached_per_axis(self):
        """One build serves every batch on the same axis; another axis gets its own."""
        module = BaselineAugmentation(mode='bspline', seed=0)
        module(_batch(1))
        assert len(module._operators) == 1
        first = next(iter(module._operators.values()))

        module(_batch(3))
        assert next(iter(module._operators.values())) is first

        module(_batch(1, n_pts=512))
        assert len(module._operators) == 2

    def test_lambda_meets_target_degrees_of_freedom(self):
        """The penalty weight is solved for, not picked off a coarse grid."""
        module = BaselineAugmentation(mode='bspline', ed_per_ppm=3.0)
        n = 64
        rng = np.random.default_rng(0)
        basis = rng.standard_normal((400, n))
        btb, diff = basis.T @ basis, BaselineAugmentation._diff_matrix_2(n)
        dtd, ridge = diff.T @ diff, 1e-10 * np.eye(n)

        lam = module._lambda_for(btb, dtd, ridge, target=12.0)
        ed = np.trace(np.linalg.solve(btb + lam * dtd + ridge, btb))
        assert abs(ed - 12.0) < 1e-2


#**************************************************************************************************#
#                                      Class TestTwoPaths                                          #
#**************************************************************************************************#
#                                                                                                  #
# The NIfTI-list path and the tensor path are one computation.                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestTwoPaths:
    """The NIfTI-list path and the tensor path are one computation."""

    @pytest.mark.parametrize("mode", ['random_walk', 'bspline', 'polynomial'])
    @pytest.mark.parametrize("coils", [None, 3])
    def test_list_equals_numpy(self, mode, coils):
        """Same seed, same numbers, on a list backend and on NumPy - to 1e-10."""
        results = []
        for backend in (Backend.NIFTI_LIST, Backend.NUMPY):
            plus = _in_frequency_domain(_batch(2, coils=coils), backend)
            out, _ = BaselineAugmentation(mode=mode, baseline_frac=0.1, phase_deg=15.0,
                                          seed=9)(plus)
            results.append(np.stack([o[:] for o in out.list()]))

        scale = np.abs(results[0]).max()
        assert np.abs(results[0] - results[1]).max() < 1e-10 * scale

    @pytest.mark.parametrize("mode", ['random_walk', 'bspline', 'polynomial'])
    def test_torch_equals_numpy_given_the_same_draws(self, mode):
        """
        The native math agrees across backends to single precision.

        The draws themselves are per-backend by design, so both modules are
        handed one stream; the random walk draws in NumPy either way.
        """
        torch = pytest.importorskip("torch")
        added = []
        for backend in (Backend.NUMPY, Backend.PYTORCH):
            plus = _batch(2, backend=backend, dtype=np.complex64)
            before = np.stack([o[:] for o in plus.list()])
            module = BaselineAugmentation(mode=mode, baseline_frac=0.1, phase_deg=15.0, seed=9)
            _shared_draws(module)
            out, _ = module(plus)
            added.append(np.stack([o[:] for o in out.list()]) - before)

        scale = np.abs(added[0]).max()
        assert np.abs(added[0] - added[1]).max() < 1e-4 * scale

    def test_ragged_list_goes_one_by_one(self):
        """Objects that cannot be stacked are still all processed, each at its own fraction."""
        objs = [_in_frequency_domain(_batch(1, n_pts=n), Backend.NIFTI_LIST).list()[0]
                for n in (512, 1024)]
        before = [o[:].copy() for o in objs]
        plus = NIfTI_MRS_Plus(nifti_list=objs, backend=Backend.NIFTI_LIST, volatile=True,
                              state=DataState(spectral='frequency'))

        fracs = [0.05, 0.1]
        out, _ = BaselineAugmentation(mode='polynomial', baseline_frac=fracs, seed=1)(plus)
        for o, b, frac in zip(out.list(), before, fracs):
            assert o[:].shape == b.shape
            ratio = np.abs((o[:] - b).real).max() / np.abs(b.real).max()
            assert abs(ratio - frac) < 1e-9

    @pytest.mark.parametrize("mode", ['random_walk', 'bspline', 'polynomial'])
    def test_gradient_is_the_identity(self, mode):
        """
        Adding a baseline leaves d(output)/d(input) = 1.

        The baseline's size is set against the data's peak, but that scale is
        detached: a loss must not be able to shrink a nuisance by way of the
        data it was measured on.
        """
        torch = pytest.importorskip("torch")
        plus = _batch(2, backend=Backend.PYTORCH, dtype=np.complex64)
        leaf = plus.get_data(Backend.PYTORCH).clone().requires_grad_(True)
        plus.set_data(leaf, Backend.PYTORCH)

        out, _ = BaselineAugmentation(mode=mode, baseline_frac=0.1, phase_deg=10.0, seed=1)(plus)
        result = out.get_data(Backend.PYTORCH)
        assert result.grad_fn is not None
        torch.real(result).sum().backward()

        assert torch.allclose(leaf.grad, torch.ones_like(leaf.grad), atol=1e-5)


#**************************************************************************************************#
#                                     Class TestPerSample                                          #
#**************************************************************************************************#
#                                                                                                  #
# A batch can carry one amplitude and one phase per sample.                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestPerSample:
    """A batch can carry one amplitude and one phase per sample."""

    def test_declares_per_sample_params(self):
        assert set(BaselineAugmentation.PER_SAMPLE_PARAMS) == {'baseline_frac', 'phase_deg'}

    @pytest.mark.parametrize("mode", ['bspline', 'polynomial'])
    def test_each_sample_gets_its_own_fraction(self, mode):
        plus = _batch(3)
        before = _spectrum(plus)
        fracs = np.array([0.02, 0.10, 0.06])
        out, _ = BaselineAugmentation(mode=mode, baseline_frac=fracs, seed=4)(plus)
        added = _spectrum(out) - before

        for b in range(3):
            ratio = np.abs(added[b].real).max() / np.abs(before[b].real).max()
            assert abs(ratio - fracs[b]) < 1e-9

    def test_each_sample_gets_its_own_phase(self):
        """A quarter turn on one sample multiplies that sample's baseline by i, only that one."""
        added = {}
        for phase in (0.0, 90.0, np.array([0.0, 90.0])):
            plus = _batch(2)
            before = _spectrum(plus)
            out, _ = BaselineAugmentation(mode='bspline', phase_deg=phase, seed=4)(plus)
            added[np.ndim(phase) or float(phase)] = _spectrum(out) - before

        assert np.allclose(added[90.0], 1j * added[0.0], atol=1e-12 * np.abs(added[0.0]).max())
        assert np.allclose(added[1][0], added[0.0][0])
        assert np.allclose(added[1][1], added[90.0][1])

    def test_pipeline_spreads_a_range_over_the_batch(self):
        """A ranged baseline_frac arrives as one value per sample, not one per batch."""
        from augmentrum.core.pipeline import AugmentationPipeline

        module = BaselineAugmentation(mode='polynomial', seed=0)
        pipeline = AugmentationPipeline([module], user_kwargs={'baseline_frac': (0.01, 0.2)})
        params = pipeline.sample_batch_parameters(4)
        assert np.shape(params[0]['baseline_frac']) == (4,)

        plus = _batch(4)
        before = _spectrum(plus)
        out, _ = pipeline(plus, None, batch_params=params)
        added = _spectrum(out) - before

        ratios = [np.abs(added[b].real).max() / np.abs(before[b].real).max() for b in range(4)]
        assert np.allclose(ratios, params[0]['baseline_frac'], rtol=1e-6)


#**************************************************************************************************#
#                                      Class TestPpmAxis                                           #
#**************************************************************************************************#
#                                                                                                  #
# The axis the baseline is placed on is FSL-MRS's, bin for bin.                                    #
#**************************************************************************************************#
class TestPpmAxis:
    """The axis the baseline is placed on is FSL-MRS's, bin for bin."""

    def test_reference_by_nucleus(self):
        from augmentrum.processing.utils import ppm_reference
        assert ppm_reference('1H') == 4.65
        assert ppm_reference('2H') == 4.65
        assert ppm_reference('31P') == 0.0
        assert ppm_reference('13C') == 0.0
        with pytest.warns(UserWarning, match="No ppm reference"):
            assert ppm_reference('23Na') == 0.0

    @pytest.mark.parametrize("bin_index", [1, 300, 512, 900, 1023])
    def test_axis_matches_fsl_bin_for_bin(self, bin_index):
        """
        A spike in one bin of the package's spectrum shows up at exactly the
        ppm "ppm_axis" claims for that bin on "MRS.getAxes()".
        """
        from augmentrum.processing.utils import ppm_axis
        n = 1024
        spectrum = np.zeros(n, complex)
        spectrum[bin_index] = 1.0
        fid = np.fft.fft(np.fft.ifftshift(spectrum))

        axis, spec = _fsl_view(fid)
        claimed = ppm_axis(n, SW_HZ, SF_MHZ, '1H')[bin_index]
        assert axis[np.argmax(np.abs(spec))] == pytest.approx(claimed, abs=1e-12)

    def test_axis_is_fsl_axis_reversed(self):
        """The same values FSL-MRS reports, in the package's bin order."""
        from augmentrum.processing.utils import ppm_axis, ppm_shift_axis
        fsl = ppm_shift_axis(64, SW_HZ, SF_MHZ)
        ours = ppm_axis(64, SW_HZ, SF_MHZ)
        assert np.allclose(ours[1:], fsl[::-1][:-1])   # bin j here is FSL bin (-j) mod n
        assert np.isclose(ours[0], fsl[-1] + (fsl[1] - fsl[0]))   # Nyquist: its +sw/2 alias
        assert np.all(np.diff(ours) < 0)              # so the axis stays monotonic

    def test_ref_ppm_override_equals_the_nucleus_reference(self):
        """Saying 4.65 explicitly is the same as letting 1H imply it."""
        results = []
        for ref in (None, 4.65):
            plus = _batch(1)
            module = BaselineAugmentation(mode='polynomial', ppm_windows=[(1.0, 2.0)],
                                          ref_ppm=ref, seed=6)
            out, _ = module(plus)
            results.append(out.list()[0][:])
        assert np.allclose(results[0], results[1])

    def test_ref_ppm_override_moves_the_window(self):
        """A different reference puts the same window on different bins."""
        results = []
        for ref in (4.65, 3.65):
            plus = _batch(1)
            _, before = _fsl_view(plus.list()[0][:])
            module = BaselineAugmentation(mode='polynomial', ppm_windows=[(1.0, 2.0)],
                                          ref_ppm=ref, seed=6)
            out, _ = module(plus)
            axis, spec = _fsl_view(out.list()[0][:])
            # the real part: the dispersion of a confined curve reaches beyond it
            touched = np.abs(np.real(spec - before)) > 1e-9 * np.abs(np.real(spec - before)).max()
            results.append((axis[touched].min(), axis[touched].max()))

        assert results[0] == pytest.approx((1.0, 2.0), abs=0.02)
        assert results[1] == pytest.approx((2.0, 3.0), abs=0.02)


#**************************************************************************************************#
#                                  Class TestReproducibility                                       #
#**************************************************************************************************#
class TestReproducibility:
    """A seed fixes the baseline; no seed gives a fresh one."""

    @pytest.mark.parametrize("mode", ['random_walk', 'bspline', 'polynomial'])
    def test_same_seed_same_baseline(self, mode):
        outs = [BaselineAugmentation(mode=mode, seed=42)(_batch(2))[0].list()[1][:]
                for _ in range(2)]
        assert np.array_equal(outs[0], outs[1])

    @pytest.mark.parametrize("mode", ['random_walk', 'bspline', 'polynomial'])
    def test_different_seed_different_baseline(self, mode):
        outs = [BaselineAugmentation(mode=mode, seed=seed)(_batch(1))[0].list()[0][:]
                for seed in (1, 2)]
        assert not np.allclose(outs[0], outs[1])


#**************************************************************************************************#
#                                      Class TestCausality                                         #
#**************************************************************************************************#
#                                                                                                  #
# A baseline is the spectrum of causal broad signals, so its FID starts at the first point.        #
#                                                                                                  #
#**************************************************************************************************#
class TestCausality:
    """
    The imaginary part of a baseline follows from its real part (Kramers-
    Kronig). Drawn independently, or left at zero, the baseline is a spectrum
    whose FID is two-sided, half of it wrapped to the end of the acquisition.
    """

    @staticmethod
    def _second_half(fid):
        """The fraction of the FID's energy in its second half: ~0 causal, 0.5 white."""
        fid = np.asarray(fid).ravel()
        return float(np.sum(np.abs(fid[fid.size // 2:]) ** 2) / np.sum(np.abs(fid) ** 2))

    @pytest.mark.parametrize("n_pts", [1024, 1001])
    def test_the_analytic_signal_is_scipy_s_hilbert(self, n_pts):
        """The one-sided step in the transform, even and odd lengths alike."""
        from scipy.signal import hilbert
        curve = np.random.default_rng(0).standard_normal((3, n_pts))
        like = np.zeros(1, complex)
        assert np.allclose(BaselineAugmentation._analytic(curve, like), hilbert(curve, axis=-1),
                           atol=1e-12)

    def test_the_analytic_signal_is_native_on_torch(self):
        torch = pytest.importorskip("torch")
        from scipy.signal import hilbert
        curve = np.random.default_rng(0).standard_normal((2, 512)).astype(np.float32)
        like = torch.zeros(1, dtype=torch.complex64)
        out = BaselineAugmentation._analytic(torch.as_tensor(curve), like)
        assert isinstance(out, torch.Tensor) and out.dtype == torch.complex64
        assert np.allclose(out.numpy(), hilbert(curve, axis=-1), atol=1e-5)

    @pytest.mark.parametrize("kwargs", [
        dict(mode='random_walk'),
        dict(mode='bspline'),
        dict(mode='polynomial'),
        dict(mode='polynomial', order=4, ppm_windows=[(0.2, 4.2)]),
    ], ids=['random_walk', 'bspline', 'polynomial', 'windowed'])
    def test_what_is_added_is_causal(self, kwargs):
        """Measured on the time-domain output of a pipeline, for a single spectrum."""
        from augmentrum.core.pipeline import AugmentationPipeline

        plus = _batch(1)
        before = np.asarray(plus.get_data(Backend.NUMPY))
        pipe = AugmentationPipeline([BaselineAugmentation(seed=2, phase_deg=20.0, **kwargs)])
        out, _ = pipe(plus, None, batch_params=pipe.sample_batch_parameters(1))
        added = np.asarray(out.get_data(Backend.NUMPY)) - before
        assert self._second_half(added[0, 0, 0, 0]) <= 0.02

    @pytest.mark.parametrize("mode", ['bspline', 'polynomial'])
    def test_the_imaginary_part_is_the_hilbert_transform_of_the_real_part(self, mode):
        from scipy.signal import hilbert
        plus = _batch(2)
        before = _spectrum(plus)
        out, _ = BaselineAugmentation(mode=mode, seed=6)(plus)
        added = _spectrum(out) - before
        for b in range(2):
            row = added[b].ravel()
            assert np.allclose(row, hilbert(row.real), atol=1e-12 * np.abs(row).max())
