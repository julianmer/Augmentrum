####################################################################################################
#                                  baseline_augmentation.py                                       #
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

import numpy as np
from typing import Optional, List, Tuple
from scipy.signal import convolve
from scipy.interpolate import BSpline
from augmentrum.core.base_module import BaseModule


# ============================================================================
# Helper Functions (from original augment_mrs.py lines 329-366)
# ============================================================================

def moving_average(x, w):
    """Apply moving average smoothing."""
    if w <= 1:
        return x
    kernel = np.ones(int(w), float) / float(w)
    return convolve(x, kernel, mode="same")


def bounded_random_walk(n, *, step_sd=1e-3, lo=-1.0, hi=1.0, seed=None):
    """
    Reflecting bounded random walk of length n (real-valued).

    Args:
        n: Number of points
        step_sd: Standard deviation of step size
        lo: Lower bound
        hi: Upper bound
        seed: Random seed

    Returns:
        Array of random walk values
    """
    rng = np.random.default_rng(seed)
    y = np.empty(n, float)
    v = 0.0
    for i in range(n):
        v += rng.normal(0.0, step_sd)
        # Reflect at the bounds
        if v < lo:
            v = lo + (lo - v)
        if v > hi:
            v = hi - (v - hi)
        y[i] = v
    return y


def make_rw_baseline_freq(n_points, *, step_sd=1e-3, bounds_amp=1.0,
                          smooth_pts=101, seed=123):
    """
    Create a smooth, bounded RW baseline (real) in frequency domain.

    Args:
        n_points: Number of spectral points
        step_sd: Step size standard deviation (smaller = smoother)
        bounds_amp: Amplitude bounds [-bounds_amp, +bounds_amp]
        smooth_pts: Moving average window size (bigger = smoother)
        seed: Random seed

    Returns:
        Baseline array (fftshifted order)
    """
    raw = bounded_random_walk(n_points, step_sd=step_sd,
                              lo=-bounds_amp, hi=bounds_amp, seed=seed)
    base = moving_average(raw, smooth_pts)
    return base


def scale_baseline_to_spec(baseline, spec, frac=0.05):
    """
    Scale baseline to a fraction of the spectrum's real peak (visibility).

    Args:
        baseline: Raw baseline array
        spec: Complex spectrum
        frac: Fraction of spectrum peak (e.g., 0.05 = 5%)

    Returns:
        Scaled baseline
    """
    scale = frac * np.max(np.abs(np.real(spec)))
    return baseline * scale


# B-spline helpers (from augment_mrs.py lines 688-713)
def _cubic_bspline_basis(x, knots, degree=3):
    """Return B-spline basis (design) matrix for given x and knot vector."""
    # Open uniform knot vector with clamped ends for cubic splines
    k = degree
    # Pad end knots (k+1 repeats) for clamping
    t0, t1 = knots[0], knots[-1]
    t = np.r_[t0*np.ones(k), knots, t1*np.ones(k)]
    n_b = len(t) - (k + 1)  # number of basis funcs
    B = np.empty((x.size, n_b))
    # coefficient vectors are unit vectors to evaluate each basis function
    for j in range(n_b):
        c = np.zeros(n_b)
        c[j] = 1.0
        B[:, j] = BSpline(t, c, k, extrapolate=True)(x)
    return B


def _diff_matrix_2(n):
    """Second-order finite-difference matrix (n-2 rows by n cols)."""
    D = np.zeros((n-2, n))
    r = np.arange(n-2)
    D[r,   r  ] = 1.0
    D[r, r+1] = -2.0
    D[r, r+2] = 1.0
    return D


# Polynomial helpers (from augment_mrs.py lines 633-685)
def auto_baseline_windows(ppm, margin_ppm=0.5):
    """
    Suggests two baseline windows near the edges of the spectral range.
    """
    ppm_min, ppm_max = ppm.min(), ppm.max()

    # Ensure the windows are returned in an intuitive, increasing (or decreasing) order
    # based on the natural flow of the ppm axis.
    if ppm_min < ppm_max:
        # Example: [0 ppm, 10 ppm] -> [0, 0.5] and [9.5, 10]
        return [
            (ppm_min, ppm_min + margin_ppm),  # lower edge window (e.g., ~0.0 to 0.5 ppm)
            (ppm_max - margin_ppm, ppm_max)   # upper edge window (e.g., ~9.5 to 10.0 ppm)
        ]
    else:
        # Example: [10 ppm, 0 ppm] -> [10, 9.5] and [0.5, 0]
        return [
            (ppm_min, ppm_min - margin_ppm),  # lower edge window (e.g., ~10.0 to 9.5 ppm)
            (ppm_max + margin_ppm, ppm_max)   # upper edge window (e.g., ~0.5 to 0.0 ppm)
        ]


def calculate_ppm_and_mask(fid, sw_hz, sf_hz, ref_ppm, ppm_windows):
    """Internal function to transform FID, calculate PPM axis, and create the baseline mask."""
    spec = np.fft.fftshift(np.fft.ifft(fid))
    N = spec.size

    # Larmor frequency in Hz (sf_hz is usually in MHz, hence 1e6)
    sf_hz_abs = sf_hz * 1e6

    # Calculate frequency offset (in Hz), centered at 0
    freq_offset_hz = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/sw_hz))

    # Calculate ppm axis: ref_ppm - (f - f_ref) / sf_hz
    ppm = ref_ppm - freq_offset_hz / sf_hz_abs
    ppm = np.asarray(ppm)

    mask_baseline = np.zeros_like(ppm, dtype=bool)

    if ppm_windows is None:
        ppm_windows = auto_baseline_windows(ppm)

    # Iterate through all defined ppm windows
    for (p1, p2) in ppm_windows:
        # Determine the low and high ppm values for masking, regardless of p1/p2 order
        p_low = min(p1, p2)
        p_high = max(p1, p2)

        region_mask = (ppm >= p_low) & (ppm <= p_high)
        mask_baseline = np.logical_or(mask_baseline, region_mask)

    return spec, N, mask_baseline


# ============================================================================
# BaselineAugmentation Module
# ============================================================================

class BaselineAugmentation(BaseModule):
    """
    Add baseline distortions to MRS data.

    Supports three modes:
    - 'random_walk': Bounded random walk with Hilbert transform (default)
    - 'bspline': Cubic B-spline baseline with P-spline smoothing
    - 'polynomial': Polynomial baseline fitted to edge regions

    Parameters
    ----------
    mode : str
        Baseline mode: 'random_walk', 'bspline', or 'polynomial' (default: 'random_walk')
    baseline_frac : float
        Fraction of spectrum peak for baseline amplitude (default: 0.05 = 5%)

    **Random Walk Parameters:**
    step_sd : float
        Step size standard deviation for random walk (default: 0.02)
    bounds_amp : float
        Amplitude bounds for random walk (default: 1.0)
    smooth_pts : int
        Moving average window size (default: 101)

    **B-Spline Parameters:**
    knots_per_ppm : int
        Number of B-spline knots per ppm (default: 12)
    ed_per_ppm : float
        Target effective degrees of freedom per ppm (default: 3.0)
    phase_deg : float
        Baseline phase rotation in degrees (default: 0.0)

    **Polynomial Parameters:**
    order : int
        Polynomial order (default: 3)
    ppm_windows : list of tuples or None
        Baseline fitting windows as [(ppm_min1, ppm_max1), ...] (default: None = auto)

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

    >>> # Polynomial baseline
    >>> baseline = BaselineAugmentation(mode='polynomial', baseline_frac=0.08, order=5)
    >>> result_data, _ = baseline(nifti_plus, None)
    """

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self, mode: str = 'random_walk', baseline_frac: float = 0.05,
                 # Random walk params
                 step_sd: float = 0.02, bounds_amp: float = 1.0, smooth_pts: int = 101,
                 # B-spline params
                 knots_per_ppm: int = 12, ed_per_ppm: float = 3.0, phase_deg: float = 0.0,
                 # Polynomial params
                 order: int = 3, ppm_windows: Optional[List[Tuple[float, float]]] = None,
                 # General
                 seed: Optional[int] = None):
        """Initialize baseline augmentation module."""
        super().__init__(mode=mode, baseline_frac=baseline_frac, step_sd=step_sd,
                        bounds_amp=bounds_amp, smooth_pts=smooth_pts,
                        knots_per_ppm=knots_per_ppm, ed_per_ppm=ed_per_ppm,
                        phase_deg=phase_deg, order=order, ppm_windows=ppm_windows, seed=seed)

        if mode not in ['random_walk', 'bspline', 'polynomial']:
            raise ValueError(f"mode must be 'random_walk', 'bspline', or 'polynomial', got '{mode}'")

        self.mode = mode
        self.baseline_frac = baseline_frac
        self.seed = seed

        # Random walk params
        self.step_sd = step_sd
        self.bounds_amp = bounds_amp
        self.smooth_pts = smooth_pts

        # B-spline params
        self.knots_per_ppm = knots_per_ppm
        self.ed_per_ppm = ed_per_ppm
        self.phase_deg = phase_deg

        # Polynomial params
        self.order = order
        self.ppm_windows = ppm_windows

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add baseline to list of NIFTI_MRS objects.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects (unchanged)
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, water_list)
        """
        processed_data = []

        for nifti in data_list:
            # Get FID data
            fid = nifti[:]

            # Get parameters
            sw_hz = 1.0 / nifti.dwelltime
            sf_mhz = nifti.spectrometer_frequency[0]

            # Add baseline based on mode
            if self.mode == 'random_walk':
                fid_with_baseline = self._add_random_walk_baseline(fid, sw_hz, sf_mhz)
            elif self.mode == 'bspline':
                fid_with_baseline = self._add_bspline_baseline(fid, sw_hz, sf_mhz)
            elif self.mode == 'polynomial':
                fid_with_baseline = self._add_polynomial_baseline(fid, sw_hz, sf_mhz)
            else:
                raise ValueError(f"Invalid baseline mode: {self.mode}")

            # Update NIFTI_MRS data
            nifti[:] = fid_with_baseline
            processed_data.append(nifti)

        return processed_data, water_list

    def _add_random_walk_baseline(self, fid: np.ndarray, sw_hz: float, sf_mhz: float,
                                   ref_ppm: float = 4.7) -> np.ndarray:
        """Add random walk baseline with Hilbert transform for complex baseline (from baseline_augmentation_BROKEN.py)."""
        from scipy.signal import hilbert

        original_shape = fid.shape
        N = original_shape[-1]
        fid_2d = fid.reshape(-1, N)
        result = np.zeros_like(fid_2d)

        for i in range(fid_2d.shape[0]):
            fid_1d = fid_2d[i]
            spec = np.fft.fftshift(np.fft.ifft(fid_1d))  # MRS convention: ifft for FID -> spectrum

            # Get spectrum peak for scaling
            spec_max = np.max(np.abs(np.real(spec)))

            # Generate normalized random walk (bounded to [-bounds_amp, +bounds_amp])
            base_rw_real = self._make_rw_baseline(N)

            # Apply Hilbert transform to create complex baseline
            # This makes the baseline causal and adds imaginary component
            base_rw_complex = hilbert(base_rw_real)

            # Scale to spectrum: first to bounds_amp * spec_max, then by baseline_frac
            # This makes bounds_amp relative to the spectrum, and baseline_frac controls the final amplitude
            baseline = base_rw_complex * self.bounds_amp * spec_max * self.baseline_frac

            spec_with_base = spec + baseline
            fid_augmented = np.fft.fft(np.fft.ifftshift(spec_with_base))  # MRS convention: fft for spectrum -> FID
            result[i] = fid_augmented

        return result.reshape(original_shape)

    def _make_rw_baseline(self, n_points: int) -> np.ndarray:
        """Create smooth, bounded random walk baseline."""
        raw = self._bounded_random_walk(n_points)
        baseline = moving_average(raw, self.smooth_pts)
        return baseline

    def _bounded_random_walk(self, n: int) -> np.ndarray:
        """Generate bounded random walk."""
        rng = np.random.default_rng(self.seed)
        y = np.empty(n, dtype=float)
        v = 0.0

        for i in range(n):
            v += rng.normal(0.0, self.step_sd)
            if v < -self.bounds_amp:
                v = -self.bounds_amp + (-self.bounds_amp - v)
            if v > self.bounds_amp:
                v = self.bounds_amp - (v - self.bounds_amp)
            y[i] = v

        return y

    def _add_bspline_baseline(self, fid: np.ndarray, sw_hz: float, sf_mhz: float,
                              ref_ppm: float = 4.7) -> np.ndarray:
        """Add B-spline baseline to FID (uses exact code from augment_mrs.py)."""
        original_shape = fid.shape
        N = original_shape[-1]
        fid_2d = fid.reshape(-1, N)
        result = np.zeros_like(fid_2d)

        for i in range(fid_2d.shape[0]):
            fid_1d = fid_2d[i]

            # ---- Go to (shifted) frequency domain (EXACT from augment_mrs.py) ----
            spec = np.fft.fftshift(np.fft.ifft(fid_1d))
            dt = 1.0 / float(sw_hz)
            freq_hz = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
            ppm = ref_ppm - freq_hz / float(sf_mhz)    # descending axis (e.g., 6→0)

            # ---- Build cubic B-spline basis over ppm domain ----
            ppm_min, ppm_max = ppm.min(), ppm.max()
            span_ppm = float(ppm_max - ppm_min)
            n_knots = max(8, int(np.ceil(span_ppm * self.knots_per_ppm)))  # interior knots
            knots = np.linspace(ppm_min, ppm_max, n_knots)
            B = _cubic_bspline_basis(ppm, knots, degree=3)            # [N x n_b]
            n_b = B.shape[1]

            # ---- P-spline penalty (2nd-diff on coefficients) ----
            D = _diff_matrix_2(n_b)

            # ---- Choose lambda from desired ED/ppm (rough heuristic) ----
            target_ED = max(2.0, self.ed_per_ppm * span_ppm)  # lower bound 2 for straight line with 2 dof
            # λ grid (logspace) – wide range covers most cases
            lam_grid = 10.0 ** np.linspace(-6, 8, 30)
            BtB = B.T @ B
            DtD = D.T @ D
            best = (None, None, None)  # (lam, ED, Ainv)
            for lam in lam_grid:
                A = BtB + lam * DtD
                # Use trace trick: tr(S) = tr(B A^{-1} B^T) = tr(A^{-1} BtB)
                Ainv = np.linalg.pinv(A)
                ED = np.trace(Ainv @ BtB)
                if best[0] is None or abs(ED - target_ED) < abs(best[1] - target_ED):
                    best = (lam, ED, Ainv)
            lam_star, ED_star, Ainv = best

            # ---- Draw smooth random coefficients consistent with the penalty ----
            I = np.eye(n_b)
            A = BtB + lam_star * DtD + 1e-10 * I   # small ridge for stability
            Ainv = np.linalg.pinv(A)

            # z must be length N to match B.T @ z
            rng = np.random.default_rng(self.seed)
            z = rng.normal(size=N)                 # N-length white noise
            a = Ainv @ (B.T @ z)                   # n_b-length smooth coefficient vector

            # ---- Make baseline and normalize/scale/phase ----
            baseline = B @ a
            # Remove DC so it doesn't just offset the spectrum
            baseline = baseline - baseline.mean()
            # Scale relative to your spectrum amplitude
            peak = np.max(np.abs(np.real(spec))) or 1.0
            bas_peak = np.max(np.abs(np.real(baseline))) or 1.0
            baseline = baseline * (self.baseline_frac * peak / bas_peak)
            if self.phase_deg:
                baseline = baseline * np.exp(1j * np.deg2rad(self.phase_deg))

            # ---- Add to spectrum and return to FID ----
            spec_aug = spec + baseline
            fid_with_baseline = np.fft.fft(np.fft.ifftshift(spec_aug))
            result[i] = fid_with_baseline

        return result.reshape(original_shape)

    def _add_polynomial_baseline(self, fid: np.ndarray, sw_hz: float, sf_mhz: float,
                                  ref_ppm: float = 4.7) -> np.ndarray:
        """Add polynomial baseline to FID (uses exact code from augment_mrs.py baseline_add_poly)."""
        original_shape = fid.shape
        N = original_shape[-1]
        fid_2d = fid.reshape(-1, N)
        result = np.zeros_like(fid_2d)

        for i in range(fid_2d.shape[0]):
            fid_1d = fid_2d[i]

            # calculate_ppm_and_mask uses ifft internally (correct for FID→spectrum)
            spec, N_pts, mask_baseline = calculate_ppm_and_mask(fid_1d, sw_hz, sf_mhz, ref_ppm, self.ppm_windows)

            idx = np.flatnonzero(mask_baseline)
            x = np.arange(N_pts)

            # Instead of random noise, we generate a synthetic baseline
            # by fitting polynomial to random values at baseline points
            rng = np.random.default_rng(self.seed)
            noise_scale = self.baseline_frac * np.max(np.abs(np.real(spec)))
            noise_real = rng.normal(0, noise_scale, len(idx))
            noise_imag = rng.normal(0, noise_scale, len(idx))

            # Fit polynomial to the noisy baseline points
            cr = np.polyfit(idx, noise_real, self.order)
            ci = np.polyfit(idx, noise_imag, self.order)
            fit = np.polyval(cr, x) + 1j * np.polyval(ci, x)

            # Add baseline to spectrum and convert back using fft (spectrum→FID, MRS convention)
            spec_corr = spec + fit
            fid_with_baseline = np.fft.fft(np.fft.ifftshift(spec_corr))
            result[i] = fid_with_baseline

        return result.reshape(original_shape)















