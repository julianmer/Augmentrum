####################################################################################################
#                                    spurious_echoes.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Adds spurious echoes (out-of-voxel signal refocused late) to MRS data.                  #
#          Supports a localized echo, a delayed replica of the FID, and a hybrid of the two        #
#          (Kyathanahally et al. 2021, Berrington et al. 2021, Bugler et al. 2025).                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import warnings

import numpy as np
from typing import Optional, List, Dict
from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from augmentrum.processing.utils import (batch_profile, device_axis, device_values, on_cuda,
                                         ppm_reference, to_backend)
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from nifti_mrs_plus import ops


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

    - 'echo' (default): The localized echo of Berrington et al. 2021 / SMART MRS
      (Bugler et al. 2025, Eq. 1): an independent additive signal
      "A·exp(-|t-t_echo|/T2)·exp(i(2πf·t+φ))", scaled by max |FID| - signal from
      outside the voxel, refocused late by imperfect crushing. The phase sits
      inside the exponent as in the paper; SMART MRS's code adds it outside,
      where it scales the amplitude instead.
    - 'replica': A delayed replica of the FID itself,
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
        - t_echo_frac: Echo center as a fraction of the acquisition time,
          instead of t_echo
        - T2: Envelope time constant in seconds (defaults to 0.04)
        - gaussian_env: Use Gaussian envelope (bool, default False)

        For echo mode only:
        - ppm: Where the echo sits, on the FSL-MRS / NIfTI-MRS axis of the
          data's nucleus (4.65 ppm at the carrier for 1H), instead of freq_hz

        Without *echoes*, echo mode draws one echo per sample from the ranges
        SMART MRS uses - t_echo_frac 0.1-0.9, ppm 0-8, any phase - with T2 of
        10-50 ms (SMART's code draws 10-50 against a time axis in seconds, which
        leaves the echo undamped) and an amplitude of 2-20 % of max |FID|.
        Replica and hybrid mode default to one fixed replica at 0.1 s.

        Any numeric key may be a "(low, high)" range instead of a number.
        Ranges are drawn uniformly, once per sample, from this module's
        seeded generator; numbers stay fixed for every sample.

    mode : str, optional
        'echo', 'replica', or 'hybrid'. None (default) is 'echo', or 'replica'
        when *echoes* are given as legacy tuples, which describe replicas.
    global_phase_deg : float
        Global phase offset for all echoes (default: 0.0)
    alpha_reference : str
        How amplitude is referenced in hybrid mode:
        'max' (default) = fraction of max(|FID|),
        'tau' = fraction of |FID| at delay time
    transient_fraction : float
        Share of the transients (DIM_DYN) that carry the echo, drawn per sample;
        1.0 (default) puts it in all of them. As SMART MRS does, an echo can
        then hit a few transients of a scan before they are averaged. Data
        without a transient axis gets the echo whole, with a warning.
    seed : int, optional
        Seed for the per-sample draws.

    Examples
    --------
    >>> # A random localized echo per sample, in a quarter of the transients
    >>> se = SpuriousEchoes(transient_fraction=0.25, seed=0)

    >>> # A localized echo placed on the ppm axis
    >>> se = SpuriousEchoes(echoes=[{'ppm': 1.3, 't_echo': 0.25, 'T2': 0.03,
    ...                              'amp': 0.1, 'phase_deg': (-180, 180)}])

    >>> # Replica mode (simple)
    >>> se = SpuriousEchoes(
    ...     mode='replica',
    ...     echoes=[{'delay_s': 0.1, 'amp': 0.3, 'phase_deg': 0,
    ...              'decay_hz': 5.0, 'freq_hz': 0.0}]
    ... )
    >>> result_data, _ = se(nifti_plus, None)

    >>> # A different replica per sample: delay, amplitude and phase drawn
    >>> se = SpuriousEchoes(
    ...     mode='replica',
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
                    'freq_hz': 0.0, 'T2': 0.04, 't_echo': None, 't_echo_frac': None,
                    'ppm': None}

    #: The echo drawn when none is given in echo mode (see the class docstring).
    ECHO_DEFAULT = {'t_echo_frac': (0.1, 0.9), 'T2': (0.01, 0.05), 'ppm': (0.0, 8.0),
                    'phase_deg': (0.0, 360.0), 'amp': (0.02, 0.2)}

    #: The replica used when none is given in replica or hybrid mode.
    REPLICA_DEFAULT = {'delay_s': 0.1, 'amp': 0.2, 'phase_deg': 0.0,
                       'decay_hz': 5.0, 'freq_hz': 0.0}

    def __init__(self, echoes=None,
                 mode: Optional[str] = None,
                 global_phase_deg: float = 0.0,
                 alpha_reference: str = 'max',
                 transient_fraction: float = 1.0,
                 seed: Optional[int] = None):
        """Initialize spurious echoes module."""
        super().__init__()

        if mode is None:
            # Legacy tuples are (delay_s, amp, phase_deg, decay_hz, freq_hz): replicas.
            legacy = echoes is not None and any(isinstance(e, (tuple, list)) for e in echoes)
            mode = 'replica' if legacy else 'echo'
        self.mode = mode.lower()
        if self.mode not in ('echo', 'replica', 'hybrid'):
            raise ValueError(f"mode must be 'echo', 'replica' or 'hybrid', got '{mode}'")

        if echoes is None:
            echoes = [dict(self.ECHO_DEFAULT if self.mode == 'echo' else self.REPLICA_DEFAULT)]

        if not 0.0 < float(transient_fraction) <= 1.0:
            raise ValueError(f"transient_fraction must be in (0, 1], got {transient_fraction}")

        self.global_phase_deg = global_phase_deg
        self.alpha_reference = alpha_reference
        self.transient_fraction = float(transient_fraction)

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

        for echo in self.echoes:
            if 'ppm' in echo and 'freq_hz' in echo:
                raise ValueError("An echo takes 'ppm' or 'freq_hz', not both.")
            if 'ppm' in echo and self.mode != 'echo':
                raise ValueError(f"'ppm' places a localized echo; in {self.mode} mode the "
                                 f"frequency is an offset of the copy, give 'freq_hz'.")
            if 't_echo_frac' in echo and self.mode == 'replica':
                raise ValueError("'t_echo_frac' has no meaning in replica mode.")

    #***************#
    #   the draws   #
    #***************#
    def _defaults(self) -> Dict[str, float]:
        """The mode's defaults for the keys whose meaning depends on it."""
        if self.mode == 'replica':
            return {'delay_s': 0.1, 'amp': 0.1}
        return {'delay_s': 0.18, 'amp': 0.05}

    def _draw(self, batch: int, sw_hz: float, n_points: Optional[int] = None,
              sf_mhz: Optional[float] = None,
              nucleus: Optional[str] = None) -> List[Dict[str, np.ndarray]]:
        """
        This batch's echo parameters, one "(batch,)" vector per key per echo.

        Ranges are drawn from a single generator taken from the module's seed
        stream, so the same seed gives the same echoes on every backend and on
        both processing paths; fixed values are repeated. Delays are resolved
        to whole samples here, so the envelope starts exactly where the shifted
        copy does. A position in ppm becomes a frequency on the data's axis,
        "f = (ppm - reference) · sf", and a fractional echo time becomes seconds
        of this acquisition.

        Args:
            batch: Number of samples.
            sw_hz: Spectral width in Hz.
            n_points: Points per FID, for 't_echo_frac'.
            sf_mhz: Spectrometer frequency in MHz, for 'ppm'.
            nucleus: NIfTI-MRS nucleus, for the ppm reference (1H when None).
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

            frac = drawn.pop('t_echo_frac', None)
            if frac is not None and 't_echo' not in drawn:
                if n_points is None:
                    raise ValueError("'t_echo_frac' needs the number of points per FID.")
                drawn['t_echo'] = frac * (int(n_points) - 1) / float(sw_hz)
            if 't_echo' not in drawn:
                drawn['t_echo'] = drawn['delay_s']

            ppm = drawn.pop('ppm', None)
            if ppm is not None:
                if sf_mhz is None:
                    raise ValueError("'ppm' needs the spectrometer frequency of the data.")
                drawn['freq_hz'] = (ppm - ppm_reference(nucleus)) * float(sf_mhz)
            drawn['gaussian_env'] = bool(echo.get('gaussian_env', False))
            table.append(drawn)
        return table

    def _transient_mask(self, batch: int, n_dyn: int) -> np.ndarray:
        """
        Which transients carry the echo: "(batch, n_dyn)", ones and zeros.

        Each sample hits its own randomly chosen round(fraction · n_dyn)
        transients, at least one.
        """
        rng = self.rng.numpy_rng()
        n_hit = max(1, int(round(self.transient_fraction * n_dyn)))
        mask = np.zeros((batch, n_dyn))
        for b in range(batch):
            mask[b, rng.choice(n_dyn, size=n_hit, replace=False)] = 1.0
        return mask

    def _dyn_axis(self, tags, rank: int, offset: int) -> Optional[int]:
        """
        The transient axis of an array with the spectral points last, or None.

        *offset* is where the higher dimensions start (4 with a batch axis in
        front, 3 for one NIfTI object); tags past the array's rank name axes a
        singleton squeeze already removed.
        """
        if self.transient_fraction >= 1.0:
            return None
        tags = [t for t in (tags or []) if t][:max(0, rank - offset - 1)]
        if 'DIM_DYN' not in tags:
            warnings.warn("SpuriousEchoes: transient_fraction is set but the data has no "
                          "transient (DIM_DYN) axis; the echo is added to every trace.")
            return None
        return offset + tags.index('DIM_DYN')

    @staticmethod
    def _at(drawn: Dict, index: int) -> Dict:
        """Sample *index*'s scalar parameters out of a drawn echo."""
        return {k: (v if isinstance(v, bool) else v[index]) for k, v in drawn.items()}

    @staticmethod
    def _columns(drawn: Dict, batch: int) -> Dict:
        """
        A drawn echo as "(batch, 1)" columns, so a profile built from it
        broadcasts to one row per sample - element by element the arithmetic
        of "_at"'s scalars, and so the same numbers.
        """
        return {k: (v if isinstance(v, bool) else np.asarray(v)[:batch, None])
                for k, v in drawn.items()}

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

    def _echo_profile_device(self, echo: Dict, n_points: int, sw_hz: float, like):
        """"_echo_profile" in float64 on *like*'s CUDA device, from the host-drawn columns."""
        import torch
        t = device_axis(('time', n_points, sw_hz),
                        lambda: np.arange(n_points, dtype=np.float64) / sw_hz, like)
        phase = np.deg2rad(self.global_phase_deg) + np.deg2rad(echo['phase_deg'])
        t_echo, T2 = device_values(echo['t_echo'], like), device_values(echo['T2'], like)
        d = t - t_echo
        if echo['gaussian_env']:
            envelope = torch.exp(-(d * d) / (2.0 * (T2 * T2)))
        else:
            envelope = torch.exp(-torch.abs(d) / T2)
        angle = 2.0 * np.pi * device_values(echo['freq_hz'], like) * t + device_values(phase, like)
        return (device_values(echo['amp'], like) * envelope
                * torch.polar(torch.ones_like(angle), angle))

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
        table, masks = None, None

        for i, nifti in enumerate(data_list):
            fid = nifti[:]
            sw_hz = 1.0 / nifti.dwelltime

            # The spectral axis is index 3 of a NIfTI-MRS array; bring it last
            # so a coil or average axis behind it is not mistaken for it.
            moved = fid.ndim > 4
            work = np.moveaxis(fid, 3, -1) if moved else fid
            dyn_axis = self._dyn_axis(getattr(nifti, 'dim_tags', None), work.ndim, 3)

            if table is None:
                # the whole batch is drawn first, in the tensor path's order
                table = self._draw(len(data_list), sw_hz, n_points=work.shape[-1],
                                   sf_mhz=nifti.spectrometer_frequency[0],
                                   nucleus=self._nucleus_of(nifti))
                if dyn_axis is not None:
                    masks = self._transient_mask(len(data_list), work.shape[dyn_axis])

            weights = None
            if masks is not None and dyn_axis is not None:
                view = [1] * work.ndim
                view[dyn_axis] = work.shape[dyn_axis]
                flags = np.broadcast_to(masks[i].reshape(view), work.shape)
                weights = flags.reshape(-1, work.shape[-1])[:, 0]

            out = self._add_echoes(work, sw_hz, [self._at(d, i) for d in table],
                                   weights=weights)
            nifti[:] = np.moveaxis(out, -1, 3) if moved else out
            processed_data.append(nifti)

        return processed_data, water_list

    @staticmethod
    def _nucleus_of(nifti) -> Optional[str]:
        """A NIfTI-MRS object's nucleus as one string, or None."""
        nucleus = getattr(nifti, 'nucleus', None)
        if isinstance(nucleus, (list, tuple)):
            nucleus = nucleus[0] if nucleus else None
        return None if nucleus is None else str(nucleus)

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
            **kwargs: Must contain "'sw_hz'" (spectral width in Hz); 'sf_mhz' and
                'nucleus' place a ppm, 'dim_tags' finds the transients.

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

        table = self._draw(batch, float(sw_hz), n_points=n_points,
                           sf_mhz=kwargs.get('sf_mhz'), nucleus=kwargs.get('nucleus'))
        dyn_axis = self._dyn_axis(kwargs.get('dim_tags'), ndim, 4)
        mask = None
        if dyn_axis is not None:
            view = [1] * ndim
            view[0], view[dyn_axis] = batch, int(shape[dyn_axis])
            mask = self._transient_mask(batch, int(shape[dyn_axis])).reshape(view)
        ghost_total = None

        for drawn in table:
            if self.mode == 'echo':
                if on_cuda(data_array) and ndim > 1:
                    profile = self._echo_profile_device(self._columns(drawn, batch), n_points,
                                                        float(sw_hz), data_array)
                    profile = profile.reshape((batch,) + (1,) * (ndim - 2) + (n_points,))
                else:
                    profile = batch_profile(self._echo_profile(self._columns(drawn, batch), t),
                                            ndim)
                max_abs = ops.amax(ops.abs(data_array), axis=-1, keepdims=True)
                ghost = ops.cast_like(max_abs, data_array) * ops.cast_like(
                    to_backend(profile, data_array), data_array)

            elif self.mode == 'replica':
                envelope = self._replica_envelope(self._columns(drawn, batch), t)
                delayed = self._delayed(data_array, drawn['shift'])
                ghost = delayed * ops.cast_like(
                    to_backend(batch_profile(envelope, ndim), data_array), data_array)

            else:  # hybrid: the delayed copy under the localized envelope
                per_sample = [self._at(drawn, i) for i in range(batch)]
                modulation = np.stack([self._hybrid_modulation(e, t) for e in per_sample])
                delayed = self._delayed(data_array, drawn['shift'])
                mod = ops.cast_like(to_backend(batch_profile(modulation, ndim), data_array),
                                    data_array)

                alpha = batch_profile(drawn['amp'][:, None], ndim)
                if np.all(np.asarray(drawn['T2']) >= 1e4):
                    ghost = delayed * mod * ops.cast_like(
                        to_backend(alpha.astype(np.complex128), data_array), data_array)
                else:
                    max_abs = ops.amax(ops.abs(data_array), axis=-1, keepdims=True) + 1e-30
                    if self.alpha_reference == 'tau':
                        amp_ref = self._at_delay(data_array, drawn['shift'])
                    else:  # 'max'
                        amp_ref = max_abs
                    scale = ops.cast_like(to_backend(alpha, amp_ref), amp_ref) * amp_ref
                    ghost = (ops.cast_like(scale, data_array)
                             * (delayed / ops.cast_like(max_abs, delayed)) * mod)

            ghost_total = ghost if ghost_total is None else ghost_total + ghost

        if ghost_total is None:
            return data_array, water_array
        if mask is not None:
            ghost_total = ghost_total * ops.cast_like(to_backend(mask, data_array),
                                                      data_array)
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
                    echoes: Optional[List[Dict]] = None, weights: Optional[np.ndarray] = None,
                    sf_mhz: Optional[float] = None, nucleus: Optional[str] = None) -> np.ndarray:
        """
        Add spurious echoes to one subject's FIDs in NumPy.

        Args:
            fid: FID data with the spectral points last
            sw_hz: Spectral width in Hz
            echoes: This subject's echoes, scalar parameters per echo; None
                draws them for a single subject
            weights: Per-trace 0/1 flags (traces in C order), which traces
                carry the echo; None puts it in all of them
            sf_mhz: Spectrometer frequency in MHz, when drawing a ppm here
            nucleus: NIfTI-MRS nucleus, when drawing a ppm here

        Returns:
            FID with echoes
        """
        if echoes is None:
            echoes = [self._at(drawn, 0) for drawn in
                      self._draw(1, sw_hz, n_points=fid.shape[-1], sf_mhz=sf_mhz,
                                 nucleus=nucleus)]

        original_shape = fid.shape
        n_points = original_shape[-1]

        fid_2d = np.asarray(fid, dtype=np.complex128).reshape(-1, n_points)
        result = np.zeros_like(fid_2d)

        for i in range(fid_2d.shape[0]):
            fid_1d = fid_2d[i]
            if weights is not None and not weights[i]:
                result[i] = fid_1d
                continue
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
