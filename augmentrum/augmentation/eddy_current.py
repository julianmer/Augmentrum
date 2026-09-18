####################################################################################################
#                                      eddy_current.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Implements water-derived and synthetic eddy current phase distortions for MRS data.     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List, Tuple
from scipy.signal import butter, filtfilt
from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from augmentrum.processing.utils import batch_profile, to_backend
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from nifti_mrs_plus import ops
from nifti_mrs_plus.ops import to_numpy


#**************************************************************************************************#
#                                        Class EddyCurrent                                         #
#**************************************************************************************************#
#                                                                                                  #
# Add eddy current phase distortions to MRS data.                                                  #
#                                                                                                  #
#**************************************************************************************************#
class EddyCurrent(BaseModule):
    """
    Add eddy current phase distortions to MRS data.

    An eddy current leaves a time-varying phase on the FID, "exp(i·φ(t))",
    which distorts the lineshape. The trajectory always starts at zero and
    carries no linear ramp (with "remove_linear"), so it is a pure lineshape
    distortion: zero-order phase and frequency offsets are other modules' job.

    Supports two modes:
    - 'synthetic': Generate a synthetic eddy current phase trajectory
    - 'water': Take the trajectory from a water reference, either one passed
      at call time or a library of them given as *source*

    Parameters
    ----------
    mode : str, optional
        'synthetic' or 'water'. None (default) means 'water' when a *source* is
        given and 'synthetic' otherwise.

    **Synthetic Mode Parameters:**
    std_rad : float
        Standard deviation of the white phase noise that is low-pass filtered
        into the trajectory (default: 0.6). The trajectory itself is much
        smaller: filtering keeps a fraction of about "2·lp_cut_hz/sw_hz" of
        the noise power, so its spread is of order
        "std_rad·sqrt(2·lp_cut_hz/sw_hz)" - about 0.07 rad at the defaults
        on a 4 kHz acquisition, up to twice that away from t = 0 because the
        trajectory is anchored there.
    lp_cut_hz : float
        Low-pass filter cutoff frequency in Hz (default: 25.0)

    **Water Mode Parameters:**
    source : sequence, optional
        Water reference FIDs to build a trajectory library from: 1-D complex
        arrays (taken to share the data's dwell time), "NIFTI_MRS" objects,
        or a "NIfTI_MRS_Plus". Their phase trajectories are extracted once
        (unwrap, held where the water has decayed into noise, linear trend
        removed, low-pass filtered, anchored at zero) and each sample draws
        one from the library with this module's seeded generator. The
        library should hold water that is *not* eddy-current corrected: a
        corrected water has an almost flat phase and does nothing. Without a
        *source*, the water passed at call time is used, one per sample.
        A multi-transient or multi-coil water is averaged; coil-combine first.
        Trajectories are resampled onto the data's time grid when its length
        or dwell time differs.
    lp_cut_hz : float
        Low-pass filter cutoff for water-derived phase (default: 25.0)

    **Common Parameters:**
    strength : float
        Strength multiplier for the eddy current effect (default: 1.0). Drawn
        per sample when a pipeline gives it as a range.
    remove_linear : bool
        Remove linear trend from phase (default: True)
    seed : int, optional
        Seed for the synthetic draws and the library draws

    Examples
    --------
    >>> # Synthetic eddy current
    >>> ec = EddyCurrent(mode='synthetic', std_rad=0.8, lp_cut_hz=30.0)
    >>> result_data, _ = ec(nifti_plus, None)
    >>>
    >>> # Water-derived eddy current from the water passed at call time
    >>> ec = EddyCurrent(mode='water', lp_cut_hz=20.0, strength=1.0)
    >>> result_data, result_water = ec(nifti_plus, water_plus)
    >>>
    >>> # A library of uncorrected waters; every sample draws its own
    >>> ec = EddyCurrent(source=[water_fid_a, water_fid_b], strength=(0.5, 1.5), seed=0)
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # A time-varying phase is applied sample by sample along the FID.
    DOMAIN = Domain(spectral='time')

    # The phasor is built per sample, so the strength can differ per sample.
    PER_SAMPLE_PARAMS = ('strength',)

    #: Fraction of a water FID's peak magnitude below which its phase is not
    #: read. A decayed water is noise, and unwrapping noise random-walks by
    #: tens of radians; the trajectory is held from the last reliable point.
    WATER_FLOOR = 0.02

    def __init__(self, mode: Optional[str] = None,
                 std_rad: float = 0.6, lp_cut_hz: float = 25.0,
                 strength: float = 1.0, remove_linear: bool = True,
                 source=None, seed: Optional[int] = None):
        """Initialize eddy current module."""
        super().__init__()

        if mode is None:
            mode = 'water' if source is not None else 'synthetic'
        self.mode = mode.lower()
        self.std_rad = std_rad
        self.lp_cut_hz = lp_cut_hz
        self.strength = strength
        self.remove_linear = remove_linear
        self.seed = seed

        if self.mode not in ['synthetic', 'water']:
            raise ValueError(f"mode must be 'synthetic' or 'water', got '{mode}'")
        if source is not None and self.mode != 'water':
            raise ValueError("A trajectory 'source' only makes sense in mode='water'.")

        # The library holds raw FIDs; trajectories are extracted per data grid
        # on first use, since the low-pass and the resampling need the grid.
        self._library: List[Tuple[np.ndarray, Optional[float]]] = self._collect(source)
        self._trajectory_cache = {}

    #*****************#
    #   the library   #
    #*****************#
    @staticmethod
    def _water_rows(water, batched: bool) -> np.ndarray:
        """
        Water FIDs as rows "(n_water, T)", or one water as "(T,)".

        A batched water keeps the NIfTI layout "(batch, X, Y, Z, T, ...)", so
        the spectral axis is index 4 once there are more than five axes; a
        single NIfTI array has it at index 3. Anything else is taken to have
        time last. Extra transients or coils are averaged.
        """
        w = np.asarray(water)
        if batched:
            if w.ndim == 1:
                w = w[None]
            if w.ndim >= 5:
                w = np.moveaxis(w, 4, -1)
            return w.reshape(w.shape[0], -1, w.shape[-1]).mean(axis=1)
        if w.ndim >= 4:
            w = np.moveaxis(w, 3, -1)
        return w.reshape(-1, w.shape[-1]).mean(axis=0)

    def _collect(self, source) -> List[Tuple[np.ndarray, Optional[float]]]:
        """The library as (FID, dwell-derived bandwidth or None) pairs."""
        if source is None:
            return []
        if isinstance(source, NIfTI_MRS_Plus):
            source = source.list()

        from fsl_mrs.core.nifti_mrs import NIFTI_MRS

        entries = []
        for item in source:
            if isinstance(item, NIFTI_MRS):
                entries.append((self._water_rows(item[:], batched=False),
                                1.0 / float(item.dwelltime)))
            else:
                entries.append((self._water_rows(np.asarray(item), batched=False), None))
        if not entries:
            raise ValueError("EddyCurrent 'source' is empty; give it at least one water FID.")
        return entries

    def _trajectories(self, n_points: int, sw_hz: float) -> np.ndarray:
        """The library's trajectories on the data's grid, "(n_water, n_points)"."""
        key = (int(n_points), float(sw_hz))
        if key not in self._trajectory_cache:
            rows = []
            for fid, sw_src in self._library:
                sw_src = float(sw_hz) if sw_src is None else sw_src
                phi = self._ec_phase_from_water(fid, sw_src)
                rows.append(self._resample(phi, sw_src, n_points, sw_hz))
            self._trajectory_cache[key] = np.stack(rows)
        return self._trajectory_cache[key]

    @staticmethod
    def _resample(phi: np.ndarray, sw_src: float, n_points: int, sw_hz: float) -> np.ndarray:
        """A trajectory on the data's time grid; held flat beyond its own."""
        if phi.size == n_points and float(sw_src) == float(sw_hz):
            return phi
        t_src = np.arange(phi.size, dtype=float) / float(sw_src)
        t = np.arange(n_points, dtype=float) / float(sw_hz)
        return np.interp(t, t_src, phi, left=phi[0], right=phi[-1])

    #*******************#
    #   trajectories   #
    #*******************#
    def _low_pass(self, phi: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        The trajectory below the cutoff, zero-phase so nothing is delayed.

        Trajectories may be stacked as rows; each is filtered on its own. The
        filter is a constant of the cutoff and the bandwidth, designed once.
        """
        if self.lp_cut_hz is None or self.lp_cut_hz <= 0:
            return phi
        nyq = 0.5 * float(sw_hz)
        Wn = min(max(self.lp_cut_hz / nyq, 1e-6), 0.999999)
        designs = self.__dict__.setdefault('_filters', {})
        if Wn not in designs:
            designs[Wn] = butter(2, Wn, btype='low')
        b, a = designs[Wn]
        return filtfilt(b, a, phi, axis=-1)

    def _ec_phase_from_water(self, fid_water: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Extract the eddy current phase trajectory from a water reference.

        The phase is read while the water is above "WATER_FLOOR" of its peak
        and held constant after, since a decayed water is noise and its
        unwrapped phase a random walk. The linear trend (zero-order phase and
        frequency offset) is fitted over that reliable part and removed, the
        result low-pass filtered and anchored at φ(0) = 0.

        Args:
            fid_water: Water reference FID
            sw_hz: Spectral width in Hz

        Returns:
            Phase values in radians, one per point of the water
        """
        w = np.asarray(fid_water).ravel()
        w = w.astype(np.result_type(w.dtype, np.complex64), copy=False)     # its own precision
        n = w.size
        t = np.arange(n, dtype=float) / float(sw_hz)
        mag = np.abs(w)
        phi = np.unwrap(np.angle(w))

        # Reliable from the start until the water first drops under the floor
        below = np.flatnonzero(mag < self.WATER_FLOOR * mag.max())
        last = int(below[0]) if below.size else n
        last = min(max(last, 2), n)

        if self.remove_linear:
            A = np.c_[np.ones(last), t[:last]]
            k0, k1 = np.linalg.lstsq(A, phi[:last], rcond=None)[0]
            phi = phi - (k0 + k1 * t)
        phi[last:] = phi[last - 1]

        phi = self._low_pass(phi, sw_hz)
        return phi - phi[0]

    def _synth_ec_phase(self, N: int, sw_hz: float, rng: np.random.Generator) -> np.ndarray:
        """
        Synthesize an eddy current phase trajectory.

        White phase noise of "std_rad", low-pass filtered, detrended and
        anchored at φ(0) = 0: an eddy current builds up from the excitation,
        it does not arrive with a phase already on it.

        The noise is drawn longer than the FID and cropped after filtering.
        "filtfilt" starts its state from the first sample as if the input had
        been constant before it, so filtering exactly N points leaves a
        transient of the raw noise's size (0.6 rad, ten times the trajectory)
        decaying over the first tens of milliseconds - which is what used to
        pass for the eddy current.

        Args:
            N: Number of points
            sw_hz: Spectral width in Hz
            rng: Generator to draw the noise from

        Returns:
            Phase values in radians
        """
        t = np.arange(N, dtype=float) / float(sw_hz)
        pad = 0
        if self.lp_cut_hz is not None and self.lp_cut_hz > 0:
            pad = int(np.ceil(3.0 * float(sw_hz) / float(self.lp_cut_hz)))
        noise = rng.normal(scale=self.std_rad, size=N + 2 * pad)
        phi = self._low_pass(noise, sw_hz)[pad:pad + N]

        if self.remove_linear:
            A = np.c_[np.ones(N), t]
            k0, k1 = np.linalg.lstsq(A, phi, rcond=None)[0]
            phi = phi - (k0 + k1 * t)

        return phi - phi[0]

    def _synth_ec_phases(self, batch: int, N: int, sw_hz: float,
                         rng: np.random.Generator) -> np.ndarray:
        """
        "_synth_ec_phase" for *batch* samples at once, "(batch, N)", bit for bit.

        The noise of all samples is one draw of the same numbers in the same
        order, and filtering, detrending and anchoring work row by row; only
        the least-squares fit stays a call per sample, since LAPACK solves
        several right-hand sides with a different rounding than one.
        """
        t = np.arange(N, dtype=float) / float(sw_hz)
        pad = 0
        if self.lp_cut_hz is not None and self.lp_cut_hz > 0:
            pad = int(np.ceil(3.0 * float(sw_hz) / float(self.lp_cut_hz)))
        noise = rng.normal(scale=self.std_rad, size=(batch, N + 2 * pad))
        phi = self._low_pass(noise, sw_hz)[:, pad:pad + N]

        if self.remove_linear:
            A = np.c_[np.ones(N), t]
            k = np.array([np.linalg.lstsq(A, row, rcond=None)[0] for row in phi])
            phi = phi - (k[:, :1] + k[:, 1:] * t)

        return phi - phi[:, :1]

    def _phases(self, batch: int, n_points: int, sw_hz: float, rng: np.random.Generator,
                water_of=None) -> np.ndarray:
        """
        One trajectory per sample, "(batch, n_points)".

        Draws in one fixed order from *rng* so the tensor and NIfTI-list paths
        produce the same batch for the same seed.

        Args:
            batch: Number of samples.
            n_points: Length of the data's FID.
            sw_hz: The data's spectral width.
            rng: This call's generator.
            water_of: Callable giving sample i's water at call time as
                "(fid_1d, sw_hz)"; used in water mode without a library.
        """
        if self.mode == 'water':
            if self._library:
                library = self._trajectories(n_points, sw_hz)
                draws = rng.integers(len(library), size=batch)
                return library[draws]
            if water_of is None:
                raise ValueError("Water reference required for 'water' mode eddy current")
            rows = []
            for i in range(batch):
                fid, sw_water = water_of(i)
                rows.append(self._resample(self._ec_phase_from_water(fid, sw_water),
                                           sw_water, n_points, sw_hz))
            return np.stack(rows)

        return self._synth_ec_phases(batch, n_points, sw_hz, rng)

    def _phasors(self, phases: np.ndarray) -> np.ndarray:
        """"exp(i·strength·φ)" per sample, the strength read per sample."""
        strength = np.array([float(self.sample_of(self.strength, i))
                             for i in range(phases.shape[0])])
        return np.exp(1j * strength[:, None] * phases)

    #**********************#
    #   processing paths   #
    #**********************#
    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add eddy current distortion to list of NIFTI_MRS objects.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects
                (required for 'water' mode without a source library)
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, processed_water_list)
        """
        if self.mode == 'water' and water_list is None and not self._library:
            raise ValueError("Water reference required for 'water' mode eddy current")

        def water_of(index):
            water = water_list[min(index, len(water_list) - 1)]
            return self._water_rows(water[:], batched=False), 1.0 / water.dwelltime

        # Drawn for the whole batch at once, in the order the tensor path draws,
        # on the first subject's grid; a subject on another grid is resampled.
        first = data_list[0]
        n_first, sw_first = first.shape[3], 1.0 / first.dwelltime
        phases = self._phases(len(data_list), n_first, sw_first, self.rng.numpy_rng(),
                              water_of=water_of if water_list is not None else None)

        processed_data = []
        for i, nifti in enumerate(data_list):
            fid = nifti[:]
            sw_hz = 1.0 / nifti.dwelltime

            # The spectral axis is index 3 of a NIfTI-MRS array; bring it last
            # so a coil or average axis behind it is not mistaken for it.
            moved = fid.ndim > 4
            work = np.moveaxis(fid, 3, -1) if moved else fid

            phase = self._resample(phases[i], sw_first, work.shape[-1], sw_hz)
            out = work * np.exp(1j * float(self.sample_of(self.strength, i)) * phase)

            nifti[:] = np.moveaxis(out, -1, 3) if moved else out
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add eddy current distortion to tensor/array data (**any backend**).

        The phase trajectories are generated in NumPy (SciPy filters), one
        per sample, then applied as a complex phasor multiplication which
        stays in the native backend — "data * to_backend(phasor, data)".

        Args:
            data_array: Input tensor of shape "(batch, ..., n_points)"
            water_array: Optional water reference tensor, "(batch, X, Y, Z, T, ...)"
            backend: Backend enum (unused)
            **kwargs: Must contain "'sw_hz'" (spectral width in Hz)

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        if sw_hz is None:
            raise ValueError("EddyCurrent.process_tensor requires 'sw_hz' in kwargs")

        if self.mode == 'water' and water_array is None and not self._library:
            raise ValueError("Water reference required for 'water' mode eddy current")

        shape = ops.shape(data_array)
        ndim = len(shape)
        n_points = int(shape[-1])
        batch = int(shape[0]) if ndim > 1 else 1

        # Only a water-mode draw without a library reads the water: fetching it
        # from a device otherwise costs a wait for nothing.
        water_of = None
        if water_array is not None and self.mode == 'water' and not self._library:
            rows = self._water_rows(to_numpy(water_array), batched=True)
            # The water shares the data's dwell time; only its length may differ
            water_of = lambda i: (rows[min(i, rows.shape[0] - 1)], float(sw_hz))

        phases = self._phases(batch, n_points, float(sw_hz), self.rng.numpy_rng(),
                              water_of=water_of)
        phasor = batch_profile(self._phasors(phases), ndim)

        # Apply phasor: backend-native multiply (preserves gradients for data)
        return data_array * ops.cast_like(to_backend(phasor, data_array), data_array), \
            water_array
