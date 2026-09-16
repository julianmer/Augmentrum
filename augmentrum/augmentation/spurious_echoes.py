####################################################################################################
#                                    spurious_echoes.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Adds delayed echo replicas (ghosting artifacts) to MRS data.                            #
#          Supports a localized echo, a delayed replica of the FID, and a hybrid of the two        #
#          (Kyathanahally et al. 2021, Berrington et al. 2021, Bugler et al. 2025).                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List, Dict
from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from augmentrum.processing.utils import batch_profile
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from nifti_mrs_plus import ops
from nifti_mrs_plus.ops import match_backend


#**************************************************************************************************#
#                                       Class SpuriousEchoes                                       #
#**************************************************************************************************#
#                                                                                                  #
# Add spurious echo artifacts to MRS data.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class SpuriousEchoes(BaseModule):
    """
    Add spurious echo artifacts to MRS data.

    Simulates ghosting artifacts by adding delayed, attenuated copies
    of the FID signal at specified time delays.

    Supports three modes:

    - 'echo': The localized echo of Berrington et al. 2021 / SMART MRS
      (Bugler et al. 2025): an independent additive signal
      "A·exp(-|t-t_echo|/T2)·exp(i(2πf·t+φ))", scaled by max |FID|.
    - 'replica' (default): A delayed replica of the FID itself,
      "amp·e^{iφ}·fid(t-τ)·exp(-π·decay·(t-τ))·exp(i2πf(t-τ))" for t ≥ τ:
      the FID shifted by τ, attenuated and frequency-shifted. In the
      spectrum this is the whole spectrum again, weighted by a modulation of
      period 1/τ Hz - the classic ghost, not a rescaled copy of the spectrum.
    - 'hybrid': The delayed replica shaped by the localized envelope of the
      'echo' model (Kyathanahally/Bugler model).

    Parameters
    ----------
    echoes : list of dicts or tuples
        Each echo specifies:
        - delay_s / tau: Echo delay time in seconds
        - amp / alpha: Relative amplitude (replica: of the FID copy; echo and
          hybrid: of max |FID|)
        - phase_deg: Additional phase for this echo (degrees)
        - decay_hz: Exponential decay of the replica in Hz (replica mode)
        - freq_hz / df_hz: Frequency offset of the echo (Hz)

        For echo and hybrid mode, additional per-echo keys:
        - t_echo: Echo center time in seconds (defaults to tau)
        - T2: Envelope time constant in seconds (defaults to 0.04)
        - gaussian_env: Use Gaussian envelope (bool, default False)

        Any numeric key may be a "(low, high)" range instead of a number.
        Ranges are drawn uniformly, once per sample, from this module's
        seeded generator; numbers stay fixed for every sample.

    mode : str
        'echo', 'replica' (default), or 'hybrid'.
    global_phase_deg : float
        Global phase offset for all echoes (default: 0.0)
    alpha_reference : str
        How amplitude is referenced in hybrid mode:
        'max' (default) = fraction of max(|FID|),
        'tau' = fraction of |FID| at delay time
    seed : int, optional
        Seed for the per-sample draws.

    Examples
    --------
    >>> # Replica mode (simple)
    >>> se = SpuriousEchoes(
    ...     echoes=[{'delay_s': 0.1, 'amp': 0.3, 'phase_deg': 0,
    ...              'decay_hz': 5.0, 'freq_hz': 0.0}]
    ... )
    >>> result_data, _ = se(nifti_plus, None)

    >>> # A different replica per sample: delay, amplitude and phase drawn
    >>> se = SpuriousEchoes(
    ...     echoes=[{'delay_s': (0.05, 0.2), 'amp': (0.05, 0.3),
    ...              'phase_deg': (-180, 180), 'decay_hz': (2, 10)}], seed=0
    ... )

    >>> # Hybrid mode (matches legacy add_spurious_echo_artifact)
    >>> se = SpuriousEchoes(
    ...     mode='hybrid',
    ...     echoes=[{'tau': 0.18, 'alpha': 0.32, 'phase_deg': 25,
    ...              't_echo': 0.26, 'T2': 0.03, 'df_hz': 35.0}]
    ... )
    >>> result_data, _ = se(nifti_plus, None)
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # An echo is a delayed copy of the FID.
    DOMAIN = Domain(spectral='time')

    #: Aliases accepted in an echo dict, mapped onto the canonical key.
    ALIASES = {'tau': 'delay_s', 'alpha': 'amp', 'amplitude': 'amp',
               'decay': 'decay_hz', 'df_hz': 'freq_hz'}

    #: Numeric keys that may be given as (low, high) ranges, with the default
    #: used when absent; None means "depends on the mode" (see _defaults).
    NUMERIC_KEYS = {'delay_s': None, 'amp': None, 'phase_deg': 0.0, 'decay_hz': 5.0,
                    'freq_hz': 0.0, 'T2': 0.04, 't_echo': None}

    def __init__(self, echoes=None,
                 mode: str = 'replica',
                 global_phase_deg: float = 0.0,
                 alpha_reference: str = 'max',
                 seed: Optional[int] = None):
        """Initialize spurious echoes module."""
        if echoes is None:
            echoes = [{'delay_s': 0.1, 'amp': 0.2, 'phase_deg': 0.0,
                       'decay_hz': 5.0, 'freq_hz': 0.0}]

        super().__init__()

        self.mode = mode.lower()
        if self.mode not in ('echo', 'replica', 'hybrid'):
            raise ValueError(f"mode must be 'echo', 'replica' or 'hybrid', got '{mode}'")

        self.global_phase_deg = global_phase_deg
        self.alpha_reference = alpha_reference

        # Normalize echoes to a list of dicts with canonical keys
        self.echoes = []
        for echo in echoes:
            if isinstance(echo, dict):
                self.echoes.append({self.ALIASES.get(k, k): v for k, v in echo.items()})
            elif isinstance(echo, (tuple, list)):
                # Legacy tuple: (delay_s, amp, phase_deg, decay_hz, freq_hz)
                self.echoes.append({
                    'delay_s': echo[0],
                    'amp': echo[1],
                    'phase_deg': echo[2] if len(echo) > 2 else 0.0,
                    'decay_hz': echo[3] if len(echo) > 3 else 5.0,
                    'freq_hz': echo[4] if len(echo) > 4 else 0.0,
                })
            else:
                raise ValueError(f"Echo must be a dict or tuple, got {type(echo)}")

    #***************#
    #   the draws   #
    #***************#
    def _defaults(self) -> Dict[str, float]:
        """The mode's defaults for the keys whose meaning depends on it."""
        if self.mode == 'replica':
            return {'delay_s': 0.1, 'amp': 0.1}
        return {'delay_s': 0.18, 'amp': 0.05}

    def _draw(self, batch: int, sw_hz: float) -> List[Dict[str, np.ndarray]]:
        """
        This batch's echo parameters, one "(batch,)" vector per key per echo.

        Ranges are drawn from a single generator taken from the module's seed
        stream, so the same seed gives the same echoes on every backend and on
        both processing paths; fixed values are repeated. Delays are resolved
        to whole samples here, so the envelope starts exactly where the shifted
        copy does.
        """
        rng = self.rng.numpy_rng()
        defaults = self._defaults()
        table = []
        for echo in self.echoes:
            drawn = {}
            for key, default in self.NUMERIC_KEYS.items():
                value = echo.get(key, defaults.get(key, default))
                if value is None:
                    continue
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    drawn[key] = rng.uniform(float(value[0]), float(value[1]), size=batch)
                else:
                    drawn[key] = np.full(batch, float(value))

            # Delay in whole samples: 'delay_pts' directly, or the delay in
            # seconds (legacy: a value above 1 is already in points)
            if 'delay_pts' in echo:
                drawn['shift'] = np.full(batch, int(echo['delay_pts']))
            else:
                delay = drawn['delay_s']
                delay = np.where(delay > 1.0, delay / float(sw_hz), delay)
                drawn['shift'] = np.round(delay * float(sw_hz)).astype(int)
            drawn['delay_s'] = drawn['shift'] / float(sw_hz)
            if 't_echo' not in drawn:
                drawn['t_echo'] = drawn['delay_s']
            drawn['gaussian_env'] = bool(echo.get('gaussian_env', False))
            table.append(drawn)
        return table

    @staticmethod
    def _at(drawn: Dict, index: int) -> Dict:
        """Sample *index*'s scalar parameters out of a drawn echo."""
        return {k: (v if isinstance(v, bool) else v[index]) for k, v in drawn.items()}

    #******************#
    #   the profiles   #
    #******************#
    def _replica_envelope(self, echo: Dict, t: np.ndarray) -> np.ndarray:
        """
        What multiplies the delayed FID copy in replica mode: zero before the
        delay, then an attenuated, phased, frequency-shifted decay from it.
        """
        td = np.maximum(t - echo['delay_s'], 0.0)
        step = (t >= echo['delay_s']).astype(np.float64)
        phase = np.deg2rad(self.global_phase_deg) + np.deg2rad(echo['phase_deg'])
        return (echo['amp'] * np.exp(1j * phase)
                * np.exp(-np.pi * echo['decay_hz'] * td)
                * np.exp(1j * 2.0 * np.pi * echo['freq_hz'] * td) * step)

    def _localized_envelope(self, echo: Dict, t: np.ndarray) -> np.ndarray:
        """The envelope of Berrington et al. 2021, peaking at "t_echo"."""
        if echo['gaussian_env']:
            return np.exp(-((t - echo['t_echo']) ** 2) / (2.0 * echo['T2'] ** 2))
        return np.exp(-np.abs(t - echo['t_echo']) / echo['T2'])

    def _echo_profile(self, echo: Dict, t: np.ndarray) -> np.ndarray:
        """
        One localized echo at unit amplitude reference.

        The model of Berrington et al. 2021 as packaged by SMART MRS
        (Bugler et al. 2025): an envelope peaking at "t_echo", carrying its
        own frequency and phase, independent of the FID it is added to. The
        caller multiplies by its amplitude reference (max |FID|).
        """
        phase = np.deg2rad(self.global_phase_deg) + np.deg2rad(echo['phase_deg'])
        return (echo['amp'] * self._localized_envelope(echo, t)
                * np.exp(1j * (2.0 * np.pi * echo['freq_hz'] * t + phase)))

    def _hybrid_modulation(self, echo: Dict, t: np.ndarray) -> np.ndarray:
        """What multiplies the (normalized) delayed copy in hybrid mode."""
        phase = np.deg2rad(self.global_phase_deg) + np.deg2rad(echo['phase_deg'])
        mod = np.exp(1j * (2.0 * np.pi * echo['freq_hz'] * (t - echo['delay_s']) + phase))
        if echo['T2'] >= 1e4:
            # No envelope: a plain shift and add
            return mod
        return self._localized_envelope(echo, t) * mod

    @staticmethod
    def _delayed(data_array, shifts: np.ndarray):
        """
        The FID shifted right by each sample's own number of points.

        One shift when the batch shares it; otherwise each sample is shifted
        on its own and the batch reassembled, still on the data's backend.
        """
        shifts = np.asarray(shifts, dtype=int)
        if len(ops.shape(data_array)) <= 1 or np.all(shifts == shifts[0]):
            return ops.shift_right(data_array, int(shifts[0]))
        return ops.concatenate([ops.shift_right(data_array[i:i + 1], int(s))
                                for i, s in enumerate(shifts)], axis=0)

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add spurious echoes to list of NIFTI_MRS objects.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, processed_water_list)
        """
        processed_data = []
        table = None

        for i, nifti in enumerate(data_list):
            fid = nifti[:]
            sw_hz = 1.0 / nifti.dwelltime
            if table is None:
                table = self._draw(len(data_list), sw_hz)

            # The spectral axis is index 3 of a NIfTI-MRS array; bring it last
            # so a coil or average axis behind it is not mistaken for it.
            moved = fid.ndim > 4
            work = np.moveaxis(fid, 3, -1) if moved else fid

            out = self._add_echoes(work, sw_hz, [self._at(d, i) for d in table])
            nifti[:] = np.moveaxis(out, -1, 3) if moved else out
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add spurious echo artifacts to tensor/array data (**any backend**).

        **Echo mode**: an independent additive echo, purely time-based apart
        from one amplitude reference per FID.

        **Replica mode**: a delayed copy of the FID (via "ops.shift_right",
        on the data's own backend) times a time-only envelope.

        **Hybrid mode**: the delayed copy, scaled by an amplitude taken from
        the data and shaped by the localized envelope.

        Envelopes are built in NumPy, one per sample, and promoted once; the
        delay and the amplitude run on the data's backend, so nothing is
        converted and gradients survive.

        Args:
            data_array: Input tensor of shape "(batch, ..., n_points)"
            water_array: Optional water reference (unchanged)
            backend: Backend enum (unused)
            **kwargs: Must contain "'sw_hz'" (spectral width in Hz)

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        if sw_hz is None:
            raise ValueError("SpuriousEchoes.process_tensor requires 'sw_hz' in kwargs")

        shape = ops.shape(data_array)
        ndim = len(shape)
        n_points = int(shape[-1])
        batch = int(shape[0]) if ndim > 1 else 1
        t = np.arange(n_points, dtype=np.float64) / float(sw_hz)

        table = self._draw(batch, float(sw_hz))
        ghost_total = None

        for drawn in table:
            per_sample = [self._at(drawn, i) for i in range(batch)]

            if self.mode == 'echo':
                profile = np.stack([self._echo_profile(e, t) for e in per_sample])
                max_abs = ops.amax(ops.abs(data_array), axis=-1, keepdims=True)
                ghost = ops.cast_like(max_abs, data_array) * ops.cast_like(
                    match_backend(batch_profile(profile, ndim), data_array), data_array)

            elif self.mode == 'replica':
                envelope = np.stack([self._replica_envelope(e, t) for e in per_sample])
                delayed = self._delayed(data_array, drawn['shift'])
                ghost = delayed * ops.cast_like(
                    match_backend(batch_profile(envelope, ndim), data_array), data_array)

            else:  # hybrid: the delayed copy under the localized envelope
                modulation = np.stack([self._hybrid_modulation(e, t) for e in per_sample])
                delayed = self._delayed(data_array, drawn['shift'])
                mod = ops.cast_like(match_backend(batch_profile(modulation, ndim), data_array),
                                    data_array)

                alpha = batch_profile(drawn['amp'][:, None], ndim)
                if np.all(np.asarray(drawn['T2']) >= 1e4):
                    ghost = delayed * mod * ops.cast_like(
                        match_backend(alpha.astype(np.complex128), data_array), data_array)
                else:
                    max_abs = ops.amax(ops.abs(data_array), axis=-1, keepdims=True) + 1e-30
                    if self.alpha_reference == 'tau':
                        amp_ref = self._at_delay(data_array, drawn['shift'])
                    else:  # 'max'
                        amp_ref = max_abs
                    scale = ops.cast_like(match_backend(alpha, amp_ref), amp_ref) * amp_ref
                    ghost = (ops.cast_like(scale, data_array)
                             * (delayed / ops.cast_like(max_abs, delayed)) * mod)

            ghost_total = ghost if ghost_total is None else ghost_total + ghost

        if ghost_total is None:
            return data_array, water_array
        return data_array + ghost_total, water_array

    @staticmethod
    def _at_delay(data_array, shifts: np.ndarray):
        """|FID| at each sample's own delay, "(batch, ..., 1)", for alpha_reference='tau'."""
        n_points = int(ops.shape(data_array)[-1])
        idx = np.minimum(np.asarray(shifts, dtype=int), n_points - 1)
        if len(ops.shape(data_array)) <= 1 or np.all(idx == idx[0]):
            return ops.abs(data_array[..., int(idx[0]):int(idx[0]) + 1])
        return ops.abs(ops.concatenate(
            [data_array[i:i + 1, ..., int(k):int(k) + 1] for i, k in enumerate(idx)], axis=0))

    #********************#
    #   the numpy path   #
    #********************#
    def _add_echoes(self, fid: np.ndarray, sw_hz: float,
                    echoes: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Add spurious echoes to one subject's FIDs in NumPy.

        Args:
            fid: FID data with the spectral points last
            sw_hz: Spectral width in Hz
            echoes: This subject's echoes, scalar parameters per echo; None
                draws them for a single subject

        Returns:
            FID with echoes
        """
        if echoes is None:
            echoes = [self._at(drawn, 0) for drawn in self._draw(1, sw_hz)]

        original_shape = fid.shape
        n_points = original_shape[-1]

        fid_2d = np.asarray(fid, dtype=np.complex128).reshape(-1, n_points)
        result = np.zeros_like(fid_2d)

        for i in range(fid_2d.shape[0]):
            fid_1d = fid_2d[i]
            if self.mode == 'hybrid':
                result[i] = self._add_echoes_hybrid_1d(fid_1d, sw_hz, echoes)
            elif self.mode == 'echo':
                result[i] = self._add_echoes_echo_1d(fid_1d, sw_hz, echoes)
            else:
                result[i] = self._add_echoes_replica_1d(fid_1d, sw_hz, echoes)

        return result.reshape(original_shape)

    @staticmethod
    def _shifted_1d(f: np.ndarray, shift: int) -> np.ndarray:
        """"f(t - tau)": zero before the delay, the FID from it on."""
        delayed = np.zeros_like(f)
        if 0 < shift < f.size:
            delayed[shift:] = f[:f.size - shift]
        elif shift <= 0:
            delayed[:] = f
        return delayed

    def _add_echoes_echo_1d(self, fid: np.ndarray, sw_hz: float, echoes: List[Dict]) -> np.ndarray:
        """Add independent localized echoes to a 1D FID."""
        f = np.asarray(fid, dtype=np.complex128)
        t = np.arange(f.size, dtype=float) / float(sw_hz)

        amp_ref = np.max(np.abs(f))
        out = f.copy()
        for echo in echoes:
            out += amp_ref * self._echo_profile(echo, t)
        return out

    def _add_echoes_replica_1d(self, fid: np.ndarray, sw_hz: float,
                               echoes: List[Dict]) -> np.ndarray:
        """
        Add echoes to 1D FID using the replica model.

        The ghost is the FID delayed by tau, attenuated, phased and
        frequency-shifted from the delay on - the same arithmetic as the
        tensor path, in NumPy.
        """
        f = np.asarray(fid, dtype=np.complex128)
        t = np.arange(f.size, dtype=float) / float(sw_hz)

        out = f.copy()
        for echo in echoes:
            out += self._shifted_1d(f, int(echo['shift'])) * self._replica_envelope(echo, t)
        return out

    def _add_echoes_hybrid_1d(self, fid: np.ndarray, sw_hz: float,
                              echoes: List[Dict]) -> np.ndarray:
        """
        Add echoes to 1D FID using the hybrid model.

        Physically shifts the FID in time, then applies the localized envelope
        of Berrington et al. 2021 / SMART MRS (Bugler et al. 2025) and a
        frequency modulation — our extension of that echo model to a replica.
        Matches the legacy add_spurious_echo_artifact.
        """
        f = np.asarray(fid, dtype=np.complex128)
        n = f.size
        t = np.arange(n, dtype=float) / float(sw_hz)

        out = f.copy()
        for echo in echoes:
            shift = int(echo['shift'])
            delayed = self._shifted_1d(f, shift)
            mod = self._hybrid_modulation(echo, t)

            if echo['T2'] >= 1e4:
                # No envelope — simple: ghost = alpha * delayed_fid * mod
                ghost = echo['amp'] * delayed * mod
            else:
                if self.alpha_reference == 'tau':
                    amp_ref = np.abs(f[min(shift, n - 1)])
                else:  # 'max'
                    amp_ref = np.max(np.abs(f)) + 1e-30
                ghost = echo['amp'] * amp_ref * (delayed / (np.max(np.abs(f)) + 1e-30)) * mod

            out += ghost

        return out
