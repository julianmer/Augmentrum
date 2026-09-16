####################################################################################################
#                                     baseline_augmentation.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (jlamaste@gmail.com)                                                     #
#          K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Baseline augmentation - supports random walk, B-spline, and polynomial baselines        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List, Tuple
from scipy.interpolate import BSpline
from scipy.signal import convolve, hilbert

from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from augmentrum.processing.utils import ppm_axis, ppm_reference
from nifti_mrs_plus import Backend, ops


#**************************************************************************************************#
#                                    Class BaselineAugmentation                                    #
#**************************************************************************************************#
#                                                                                                  #
# Add baseline distortions to MRS data.                                                            #
#                                                                                                  #
#**************************************************************************************************#
class BaselineAugmentation(BaseModule):
    """
    Add baseline distortions to MRS data.

    Supports three modes:
    - 'random_walk': Bounded random walk (default)
    - 'bspline': Cubic B-spline baseline with P-spline smoothing
    - 'polynomial': Random polynomial over the fit window, on the regressors
      FSL-MRS parameterises its baseline with

    Every mode draws one real smooth curve per trace and makes it complex by
    its Hilbert transform, then scales it against the trace's own real peak
    and rotates it by "phase_deg". A baseline is the spectrum of broad,
    fast-decaying signals - macromolecules, lipids, residual water, eddy
    currents - and every one of those is causal: it starts at the first point
    of the FID. The spectrum of a causal signal is analytic, so its imaginary
    part is fixed by its real part (Kramers-Kronig). That is why the imaginary
    coefficient set of FSL-MRS's fit model, free there because the fit only
    has to follow the data, is not drawn independently here: an independent
    imaginary curve, or a real one on its own, is a spectrum whose FID is
    two-sided, half of it wrapped to the end of the acquisition, where it
    rings once the FID is zero-filled or truncated.

    The random draws happen on the data's backend and the baseline is formed
    there too, Hilbert transform included, so a torch batch keeps its device
    and its autograd graph; only the data-independent operators (a spline
    basis, a polynomial basis, the smoothing operator) are built in NumPy,
    once per spectral axis, and cached.

    Parameters
    ----------
    mode : str
        Baseline mode: 'random_walk', 'bspline', or 'polynomial' (default: 'random_walk')
    baseline_frac : float
        Amplitude of the baseline as a fraction of the real spectrum's peak
        (default: 0.05 = 5%). For 'bspline' and 'polynomial' the real part of
        the baseline peaks at exactly this fraction; for 'random_walk' the walk
        is bounded to +-bounds_amp times it.
    phase_deg : float
        Phase rotation applied to the baseline, in degrees (default: 0.0)

    **Random Walk Parameters:**
    step_sd : float
        Step size standard deviation for random walk (default: 0.02)
    bounds_amp : float
        The walk reflects off +-bounds_amp, in units of baseline_frac times
        the real peak (default: 1.0)
    smooth_pts : int
        Moving average window size (default: 101)

    **B-Spline Parameters:**
    knots_per_ppm : int
        Number of B-spline knots per ppm (default: 12)
    ed_per_ppm : float
        Target effective degrees of freedom per ppm (default: 3.0)

    **Polynomial Parameters:**
    order : int
        Polynomial order (default: 2, FSL-MRS's default baseline order)
    ppm_windows : list of tuples or None
        Windows the baseline is confined to, as [(ppm_a, ppm_b), ...] in
        either order; each carries its own polynomial and the real baseline is
        zero elsewhere (default: None = the whole axis). The imaginary part is
        the Hilbert transform of the windowed curve, and reaches a little
        beyond the windows, as the dispersion of anything confined must.

    ref_ppm : float or None
        The ppm at the carrier. None takes it from the data's nucleus by the
        FSL-MRS convention, 4.65 ppm for 1H (default: None)
    seed : int or None
        Random seed for reproducibility (default: None)

    Examples
    --------
    >>> # Random walk baseline (default)
    >>> baseline = BaselineAugmentation(mode='random_walk', baseline_frac=0.05)
    >>> result_data, _ = baseline(nifti_plus, None)

    >>> # B-spline baseline
    >>> baseline = BaselineAugmentation(mode='bspline', baseline_frac=0.10, knots_per_ppm=12)
    >>> result_data, _ = baseline(nifti_plus, None)

    >>> # Polynomial baseline over the fit window only
    >>> baseline = BaselineAugmentation(mode='polynomial', baseline_frac=0.08, order=4,
    ...                                 ppm_windows=[(0.2, 4.2)])
    >>> result_data, _ = baseline(nifti_plus, None)
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # A baseline is a feature of a spectrum, so that is where it is built.
    DOMAIN = Domain(spectral='frequency')

    # The amplitude and the phase broadcast, so a batch can carry a spread of each.
    PER_SAMPLE_PARAMS = ('baseline_frac', 'phase_deg')

    MODES = ('random_walk', 'bspline', 'polynomial')

    def __init__(self, mode: str = 'random_walk', baseline_frac: float = 0.05,
                 # Random walk params
                 step_sd: float = 0.02, bounds_amp: float = 1.0, smooth_pts: int = 101,
                 # B-spline params
                 knots_per_ppm: int = 12, ed_per_ppm: float = 3.0, phase_deg: float = 0.0,
                 # Polynomial params
                 order: int = 2, ppm_windows: Optional[List[Tuple[float, float]]] = None,
                 # General
                 ref_ppm: Optional[float] = None, seed: Optional[int] = None):
        """Initialize baseline augmentation module."""
        super().__init__()

        if mode not in self.MODES:
            raise ValueError(
                f"mode must be 'random_walk', 'bspline', or 'polynomial', got '{mode}'")
        if int(order) < 1:
            raise ValueError(f"order must be at least 1, got {order}")

        self.mode = mode
        self.baseline_frac = baseline_frac
        self.phase_deg = phase_deg
        self.seed = seed

        # Random walk params
        self.step_sd = step_sd
        self.bounds_amp = bounds_amp
        self.smooth_pts = smooth_pts

        # B-spline params
        self.knots_per_ppm = knots_per_ppm
        self.ed_per_ppm = ed_per_ppm

        # Polynomial params
        self.order = order
        self.ppm_windows = ppm_windows

        self.ref_ppm = ref_ppm

        # Data-independent operators, keyed by the spectral axis and the
        # parameters they were built from. One build serves every batch.
        self._operators = {}

    #*****************#
    #   entry points  #
    #*****************#
    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add a baseline to each NIFTI_MRS object.

        The objects hold spectra when this runs: the module declares the
        frequency domain, and the base class moves a batch there before
        dispatching - through a tensor backend, so a list of FIDs never lands
        here. What does is a list already in the frequency state.

        The objects are stacked into one batch and handed to the routine the
        tensor path uses, so the list and the tensor backends draw the same
        random numbers from the same seed and come out equal. Objects that
        differ in shape or axis cannot be stacked and go one at a time, each a
        batch of one with the per-sample parameters read at its index.

        Args:
            data_list: List of NIFTI_MRS objects, in the frequency domain
            water_list: Optional list of water reference NIFTI_MRS objects (unchanged)
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, water_list)
        """
        specs, axes = [], []
        for nifti in data_list:
            spec = np.moveaxis(nifti[:], 3, -1)     # spectral axis last, like the tensor path
            specs.append(spec)
            axes.append((spec.shape, 1.0 / nifti.dwelltime,
                         float(nifti.spectrometer_frequency[0]), str(nifti.nucleus[0])))

        if len(set(axes)) == 1:
            shape, sw_hz, sf_mhz, nucleus = axes[0]
            ppm = self._ppm(shape[-1], sw_hz, sf_mhz, nucleus)
            out = self._add_baseline(np.stack(specs), ppm, self.baseline_frac, self.phase_deg)
            for nifti, spec in zip(data_list, out):
                nifti[:] = np.moveaxis(spec, -1, 3)
        else:
            for i, (nifti, spec, (shape, sw_hz, sf_mhz, nucleus)) in enumerate(
                    zip(data_list, specs, axes)):
                ppm = self._ppm(shape[-1], sw_hz, sf_mhz, nucleus)
                out = self._add_baseline(spec[None], ppm,
                                         self.sample_of(self.baseline_frac, i),
                                         self.sample_of(self.phase_deg, i))
                nifti[:] = np.moveaxis(out[0], -1, 3)

        return data_list, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add a baseline to tensor/array data (**any backend, natively**).

        Args:
            data_array: Input spectra of shape "(batch, ..., n_points)"
            water_array: Optional water reference tensor (unchanged)
            backend: Backend enum (unused - ops dispatch on the tensor)
            **kwargs: Must contain "'sw_hz'" and "'sf_mhz'"; "'nucleus'" fixes
                the ppm reference and defaults to 1H

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        sf_mhz = kwargs.get('sf_mhz')
        if sw_hz is None or sf_mhz is None:
            raise ValueError(
                "BaselineAugmentation.process_tensor requires 'sw_hz' and 'sf_mhz' in kwargs")

        ppm = self._ppm(ops.shape(data_array)[-1], sw_hz, sf_mhz, kwargs.get('nucleus', '1H'))
        return self._add_baseline(data_array, ppm, self.baseline_frac, self.phase_deg), water_array

    #***************#
    #   the batch   #
    #***************#
    def _add_baseline(self, spec, ppm, baseline_frac, phase_deg):
        """
        Add this module's baseline to a batch of spectra, on their own backend.

        Every trace - each voxel and transient behind a batch entry - gets its
        own baseline, scaled against its own real peak.

        Args:
            spec: "(batch, ..., n_points)" complex spectra, any backend.
            ppm: The ppm of every bin.
            baseline_frac: A scalar, or one value per sample.
            phase_deg: A scalar, or one value per sample.

        Returns:
            The spectra with a baseline added, same shape and backend.
        """
        shape = ops.shape(spec)
        batch, n_pts = shape[0], shape[-1]
        flat = ops.reshape(spec, (-1, n_pts))
        traces = ops.shape(flat)[0]
        magnitude = ops.abs(ops.real(flat))

        if self.mode == 'polynomial':
            unit = self._analytic(self._polynomial(magnitude, ppm), flat)
        elif self.mode == 'bspline':
            unit = self._analytic(self._bspline(magnitude, ppm), flat)
        else:
            unit = self._random_walk(flat)

        # Amplitude relative to each trace's own real peak. The peak is detached:
        # how large a nuisance is belongs to the perturbation, and a loss must
        # not be able to shrink it by way of the data.
        peak = ops.detach(self._peak(magnitude))
        if self.mode == 'random_walk':
            scale = peak                              # the walk is bounded already
        else:
            scale = peak / self._peak(ops.abs(ops.real(unit)))
        scale = scale * self._column(baseline_frac, batch, traces, peak)

        baseline = ops.cast_like(unit * ops.cast_like(scale, unit), flat)

        if np.any(np.asarray(phase_deg) != 0):
            rotation = np.exp(1j * np.deg2rad(np.asarray(phase_deg, dtype=np.float64)))
            baseline = baseline * self._column(rotation, batch, traces, baseline)

        return ops.reshape(flat + baseline, shape)

    #*****************#
    #   the modes     #
    #*****************#
    def _polynomial(self, like, ppm):
        """
        A random real polynomial per trace: one coefficient set over the
        orthogonal basis FSL-MRS parameterises its baseline with.
        """
        basis = self._polynomial_basis(ppm)
        traces = ops.shape(like)[0]
        basis_t = ops.match_backend(np.ascontiguousarray(basis.T), like)
        return ops.matmul(self._draw((traces, basis.shape[1]), like), basis_t)

    def _bspline(self, like, ppm):
        """
        A smooth random real curve per trace: white noise put through the
        P-spline smoother, "a = A^-1 B^T z", then read out on the centred basis.
        """
        basis_c, smoother = self._bspline_operator(ppm)
        traces, n_pts = ops.shape(like)
        noise = self._draw((traces, n_pts), like)
        coeffs = ops.matmul(noise, ops.match_backend(np.ascontiguousarray(smoother.T), like))
        return ops.matmul(coeffs, ops.match_backend(np.ascontiguousarray(basis_c.T), like))

    def _random_walk(self, like):
        """
        A bounded, smoothed random walk per trace, made complex by its Hilbert
        transform. Generated in NumPy - each step reflects off the bound the
        previous one reached, which no backend vectorises - then promoted once;
        scipy's "hilbert" is the construction "_analytic" repeats natively.
        """
        traces, n_pts = ops.shape(like)
        rng = self.rng.numpy_rng()
        bound = float(self.bounds_amp)

        steps = rng.normal(0.0, float(self.step_sd), size=(traces, n_pts))
        walk = np.empty((traces, n_pts))
        level = np.zeros(traces)
        for i in range(n_pts):
            level = level + steps[:, i]
            level = np.where(level < -bound, -2.0 * bound - level, level)
            level = np.where(level > bound, 2.0 * bound - level, level)
            walk[:, i] = level

        width = int(self.smooth_pts)
        if width > 1:
            walk = convolve(walk, np.ones((1, width)) / float(width), mode='same')

        return ops.match_backend(hilbert(walk, axis=-1), like)

    #*************************#
    #   the analytic signal   #
    #*************************#
    @staticmethod
    def _analytic(curve, like):
        """
        The analytic signal "r + i H(r)" of each real curve *r*, along the spectral axis.

        Built as "scipy.signal.hilbert" builds it - the transform of *r* kept
        on one side, doubled, and transformed back - but on the curve's own
        backend, so a torch baseline stays on its device and in its graph.

        Args:
            curve: "(traces, n_points)" real curves, any backend.
            like: A complex tensor lending its dtype.

        Returns:
            "(traces, n_points)" complex, whose real part is *curve*.
        """
        n_pts = ops.shape(curve)[-1]
        one_sided = np.zeros(n_pts)
        one_sided[0] = 1.0
        if n_pts % 2 == 0:
            one_sided[1:n_pts // 2] = 2.0
            one_sided[n_pts // 2] = 1.0
        else:
            one_sided[1:(n_pts + 1) // 2] = 2.0

        transform = ops.fft(ops.cast_like(curve, like))
        return ops.ifft(transform * ops.match_backend(one_sided, transform))

    #*******************#
    #   the operators   #
    #*******************#
    def _polynomial_basis(self, ppm):
        """
        Orthogonal polynomial basis over the fit windows, cached per axis.

        Mirrors FSL-MRS's baseline regressors: over each window a unit axis
        "x = linspace(-1, 1)", running with ppm, carries the polynomials of
        degree 1 to "order", each orthogonalised against the lower ones on the
        window's own points. The constant is left out, and that is what
        removes the DC component - every remaining column is orthogonal to it,
        so it averages to zero over its window. Legendre polynomials go into
        the orthogonalisation rather than monomials so that high orders stay
        well-conditioned, and each column is scaled to unit peak so that every
        degree weighs the same before the random draw.

        Returns:
            "(n_points, order * n_windows)" basis, zero outside the windows.
        """
        order = int(self.order)
        windows = None if self.ppm_windows is None else tuple(
            (float(a), float(b)) for a, b in self.ppm_windows)
        key = ('polynomial', self._axis_key(ppm), order, windows)
        if key in self._operators:
            return self._operators[key]

        n_pts = ppm.size
        blocks = []
        for lo, hi in (windows or [(ppm.min(), ppm.max())]):
            lo, hi = min(lo, hi), max(lo, hi)
            idx = np.flatnonzero((ppm >= lo) & (ppm <= hi))
            if idx.size < order + 2:
                raise ValueError(
                    f"ppm window ({lo}, {hi}) holds {idx.size} points, too few for a "
                    f"polynomial of order {order} on an axis from {ppm.min():.2f} to "
                    f"{ppm.max():.2f} ppm.")
            x = np.linspace(-1.0, 1.0, idx.size)
            q, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, order))
            q = q[:, 1:] / np.abs(q[:, 1:]).max(axis=0)
            block = np.zeros((n_pts, order))
            block[idx[np.argsort(ppm[idx])]] = q     # x rises with ppm, whatever the bin order
            blocks.append(block)

        basis = np.concatenate(blocks, axis=1)
        self._operators[key] = basis
        return basis

    def _bspline_operator(self, ppm):
        """
        The centred basis and the smoothing operator for this axis, cached.

        Everything here depends only on the axis and the module parameters, not
        on any spectrum, so one basis build and one lambda search serve every
        batch. The basis columns are centred so that any curve read out on
        them averages to zero - the DC removal, done once instead of per trace.

        Returns:
            "(basis_c, smoother)": "(n_points, n_b)" and "(n_b, n_points)", so that
            a baseline is "z @ smoother.T @ basis_c.T" for white noise "z".
        """
        key = ('bspline', self._axis_key(ppm), int(self.knots_per_ppm), float(self.ed_per_ppm))
        if key in self._operators:
            return self._operators[key]

        # ---- Cubic B-spline basis over the ppm range ----
        ppm_min, ppm_max = float(ppm.min()), float(ppm.max())
        span_ppm = ppm_max - ppm_min
        n_knots = max(8, int(np.ceil(span_ppm * self.knots_per_ppm)))
        knots = np.linspace(ppm_min, ppm_max, n_knots)
        basis = self._cubic_bspline_basis(ppm, knots, degree=3)
        n_b = basis.shape[1]

        # ---- P-spline penalty (2nd difference on the coefficients) ----
        diff = self._diff_matrix_2(n_b)
        btb = basis.T @ basis
        dtd = diff.T @ diff
        ridge = 1e-10 * np.eye(n_b)

        # ---- Lambda for the wanted effective degrees of freedom ----
        # At least 2: a straight line is what the penalty leaves at infinity.
        target = float(np.clip(self.ed_per_ppm * span_ppm, 2.0, n_b))
        lam = self._lambda_for(btb, dtd, ridge, target)

        smoother = np.linalg.solve(btb + lam * dtd + ridge, basis.T)
        basis_c = basis - basis.mean(axis=0)

        self._operators[key] = (basis_c, smoother)
        return basis_c, smoother

    @staticmethod
    def _lambda_for(btb, dtd, ridge, target, halvings=50):
        """
        The penalty weight whose effective degrees of freedom meet *target*.

        ED(lambda) = tr((B'B + lambda D'D)^-1 B'B) falls monotonically from the
        number of basis functions to the penalty's null space, so a bisection
        on log(lambda) finds the target to any precision. It runs a fixed
        number of halvings over a 20-decade bracket, so it cannot hang, and
        stops early once ED is within a hundredth of the target.
        """
        def ed(log_lam):
            return float(np.trace(np.linalg.solve(btb + 10.0 ** log_lam * dtd + ridge, btb)))

        lo, hi = -10.0, 10.0
        for _ in range(halvings):
            mid = 0.5 * (lo + hi)
            value = ed(mid)
            if abs(value - target) < 1e-2:
                return 10.0 ** mid
            if value > target:
                lo = mid
            else:
                hi = mid
        return 10.0 ** (0.5 * (lo + hi))

    @staticmethod
    def _cubic_bspline_basis(x, knots, degree=3):
        """Return B-spline basis (design) matrix for given x and knot vector."""
        # Open uniform knot vector with clamped ends for cubic splines
        k = degree
        t0, t1 = knots[0], knots[-1]
        t = np.r_[t0 * np.ones(k), knots, t1 * np.ones(k)]
        n_b = len(t) - (k + 1)
        basis = np.empty((x.size, n_b))
        # coefficient vectors are unit vectors to evaluate each basis function
        for j in range(n_b):
            c = np.zeros(n_b)
            c[j] = 1.0
            basis[:, j] = BSpline(t, c, k, extrapolate=True)(x)
        return basis

    @staticmethod
    def _diff_matrix_2(n):
        """Second-order finite-difference matrix (n-2 rows by n cols)."""
        diff = np.zeros((n - 2, n))
        r = np.arange(n - 2)
        diff[r, r] = 1.0
        diff[r, r + 1] = -2.0
        diff[r, r + 2] = 1.0
        return diff

    #***************#
    #   utilities   #
    #***************#
    def _ppm(self, n_pts, sw_hz, sf_mhz, nucleus):
        """The ppm of every bin, referenced by nucleus unless "ref_ppm" says otherwise."""
        if self.ref_ppm is None:
            return ppm_axis(n_pts, sw_hz, sf_mhz, nucleus)
        # An explicit reference is the same bins shifted. Built against a known
        # nucleus so an unknown one does not warn about a reference it was given.
        return ppm_axis(n_pts, sw_hz, sf_mhz, '1H') + (float(self.ref_ppm) - ppm_reference('1H'))

    @staticmethod
    def _axis_key(ppm):
        """What identifies a linear axis: its length and two of its points."""
        return (int(ppm.size), float(ppm[1]), float(ppm[-1]))

    def _draw(self, shape, like):
        """Standard normal draws on *like*'s backend and device, in its precision."""
        return self.rng.normal(tuple(shape), like=like, dtype=self._float_name(like))

    @staticmethod
    def _float_name(x):
        """The dtype of *x* by name ('float32', 'float64'), as SeedGenerator wants it."""
        return str(getattr(x.dtype, 'name', x.dtype)).split('.')[-1]

    @staticmethod
    def _peak(magnitude):
        """The largest value in each row, with one standing in for an all-zero row."""
        peak = ops.amax(magnitude, axis=-1, keepdims=True)
        return ops.where(peak > 0, peak, ops.cast_like(peak * 0.0 + 1.0, peak))

    @staticmethod
    def _column(value, batch, traces, like):
        """
        A parameter as a "(traces, 1)" column on *like*'s backend.

        A scalar becomes one row that broadcasts. A per-sample vector is read
        modulo its length, as "sample_of" does, and each sample's value is
        repeated over the traces behind it - the voxels or transients of that
        batch entry.
        """
        arr = np.asarray(value)
        if arr.ndim == 0:
            column = arr.reshape(1, 1)
        else:
            per_sample = arr.reshape(-1)[np.arange(batch) % arr.size]
            column = np.repeat(per_sample, traces // batch).reshape(traces, 1)
        return ops.match_backend(column, like)
