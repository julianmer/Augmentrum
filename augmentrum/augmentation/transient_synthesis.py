####################################################################################################
#                                    transient_synthesis.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-14                                                                              #
#                                                                                                  #
# Purpose: Synthesizes a train of transients from a single (averaged) FID, with the correlated     #
#          scan-time structure real transients have: drift, respiration, coupled phase, motion     #
#          events - rather than independent per-transient jitter.                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional

from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from nifti_mrs_plus import Backend
from nifti_mrs_plus.ops import match_backend, cast_like


__all__ = ['TransientSynthesizer']


#**************************************************************************************************#
#                                    Class TransientSynthesizer                                    #
#**************************************************************************************************#
#                                                                                                  #
# Synthesizes N transients from one FID, with the correlated structure of a real scan.             #
#                                                                                                  #
#**************************************************************************************************#
class TransientSynthesizer(BaseModule):
    """
    Synthesizes N transients from one FID, with the correlated structure of a real scan.

    A transient train is not independent draws: the frequency walks (gradient
    heating drift ~0.3 Hz/min at 3 T, Hui et al. 2021; short-term wander;
    breathing at ~0.3 Hz, van de Moortele et al. 2002), the phase partly
    follows the frequency (Near et al. 2015), and motion arrives as sparse
    events that kick frequency, phase, amplitude, and lineshape at once and
    can leave a persistent step behind (Rowland et al. 2017, Hess et al.
    2011). Simulators so far draw these independently per transient (ISBI
    2023 challenge; SMART MRS uses linear drift only) — this module samples
    the correlated processes instead.

    The transient index becomes a new trailing DIM_DYN axis, so the result
    feeds straight into "AverageSampler" and "RawProcessor". Thermal noise is
    deliberately not added here: pair with the "Noise" module, which knows
    about coil covariance and spatial profiles.

    Every rate/level argument takes a scalar (fixed), a "(min, max)" tuple
    (drawn once per scan), or None for the literature-derived default. One
    scan is drawn per batch element.

    Args:
        n_transients: How many transients to synthesize (the DIM_DYN length).
        tr_s: Repetition time in seconds — the clock the processes run on.
        drift_hz_per_min: Linear frequency drift rate. None samples the 3 T
            envelope: lognormal around 0.3 Hz/min, random sign.
        ar_tau_s: Correlation time of the stochastic frequency wander.
        ar_sigma_hz: Stationary spread of that wander.
        resp_amp_hz: Respiratory frequency modulation amplitude. None samples
            |N(0.25, 0.15)| Hz, capped at 1 Hz.
        resp_freq_hz: Breathing rate in Hz. None samples N(0.28, 0.05),
            clipped to [0.15, 0.4].
        phase_coupling_deg_per_hz: Slow phase drift tracking the frequency.
        phase_jitter_deg: Independent per-transient phase jitter (receiver).
        amp_jitter_frac: Independent per-transient amplitude jitter.
        broaden_hz: Extra lineshape broadening reached by the end of the scan
            (slow shim degradation), drawn from this range.
        events_per_min: Rate of the motion-event process. Events last 1-6
            transients; 30% are disruptive (amplitude loss and 5-20 Hz
            broadening on top of the frequency/phase kick), and half of all
            events leave a persistent frequency step behind.
        seed: Fixes the whole synthesis. Draws still vary from call to call.

    Examples:
        >>> import numpy as np
        >>> fid = np.ones((1, 1, 1, 1, 512), np.complex64)
        >>> train, _ = TransientSynthesizer(n_transients=8, seed=0).process_tensor(
        ...     fid, sw_hz=2000.0)
        >>> train.shape
        (1, 1, 1, 1, 512, 8)
    """

    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    # The corruptions are phases and envelopes wound along the FID.
    DOMAIN = Domain(spectral='time')

    # The transient train is a new acquisition axis only this module knows.
    ADDS_DIM_TAGS = ('DIM_DYN',)

    def __init__(self, n_transients: int = 32, tr_s: float = 2.0,
                 drift_hz_per_min=None,
                 ar_tau_s=(20.0, 120.0),
                 ar_sigma_hz=(0.05, 0.3),
                 resp_amp_hz=None,
                 resp_freq_hz=None,
                 phase_coupling_deg_per_hz=(0.0, 5.0),
                 phase_jitter_deg=(0.5, 3.0),
                 amp_jitter_frac=(0.005, 0.02),
                 broaden_hz=(0.0, 2.0),
                 events_per_min=(0.1, 0.5),
                 seed: Optional[int] = None):
        super().__init__()

        if int(n_transients) < 1:
            raise ValueError(f"n_transients must be at least 1, got {n_transients!r}")

        self.n_transients = int(n_transients)
        self.tr_s = float(tr_s)
        self.drift_hz_per_min = drift_hz_per_min
        self.ar_tau_s = ar_tau_s
        self.ar_sigma_hz = ar_sigma_hz
        self.resp_amp_hz = resp_amp_hz
        self.resp_freq_hz = resp_freq_hz
        self.phase_coupling_deg_per_hz = phase_coupling_deg_per_hz
        self.phase_jitter_deg = phase_jitter_deg
        self.amp_jitter_frac = amp_jitter_frac
        self.broaden_hz = broaden_hz
        self.events_per_min = events_per_min
        self.seed = seed

    #*****************#
    #   scan draws    #
    #*****************#
    @staticmethod
    def _level(value, rng, default):
        """A scalar as given, a uniform draw from a range, or the default."""
        if value is None:
            return float(default(rng))
        if isinstance(value, tuple):
            return float(rng.uniform(*value))
        return float(value)

    def _scan_parameters(self, rng):
        """One scan's worth of process parameters, from the 3 T envelope."""
        drift = self._level(
            self.drift_hz_per_min, rng,
            lambda r: r.choice([-1.0, 1.0]) * np.clip(
                np.exp(np.log(0.3) + 0.7 * r.standard_normal()), 0.05, 1.5))
        resp_amp = self._level(
            self.resp_amp_hz, rng,
            lambda r: min(abs(0.25 + 0.15 * r.standard_normal()), 1.0))
        resp_freq = self._level(
            self.resp_freq_hz, rng,
            lambda r: np.clip(0.28 + 0.05 * r.standard_normal(), 0.15, 0.4))

        return {
            'drift_hz_per_s': drift / 60.0,
            'ar_tau_s': self._level(self.ar_tau_s, rng, None),
            'ar_sigma_hz': self._level(self.ar_sigma_hz, rng, None),
            'resp_amp_hz': resp_amp,
            'resp_freq_hz': resp_freq,
            'resp_phase': rng.uniform(0.0, 2.0 * np.pi),
            'coupling_deg_per_hz': self._level(self.phase_coupling_deg_per_hz, rng, None),
            'phase_jitter_deg': self._level(self.phase_jitter_deg, rng, None),
            'amp_jitter_frac': self._level(self.amp_jitter_frac, rng, None),
            'broaden_hz': self._level(self.broaden_hz, rng, None),
            'events_per_min': self._level(self.events_per_min, rng, None),
        }

    def _transient_tracks(self, rng):
        """
        The four per-transient tracks: frequency (Hz), phase (rad),
        amplitude (fraction), extra broadening (Hz FWHM).
        """
        n = self.n_transients
        p = self._scan_parameters(rng)
        t = np.arange(n, dtype=np.float64) * self.tr_s
        scan_s = max(t[-1], self.tr_s)

        # Frequency: linear drift + AR(1) wander + breathing.
        rho = np.exp(-self.tr_s / max(p['ar_tau_s'], 1e-6))
        wander = np.zeros(n)
        step_sd = p['ar_sigma_hz'] * np.sqrt(max(1.0 - rho ** 2, 1e-12))
        for i in range(1, n):
            wander[i] = rho * wander[i - 1] + step_sd * rng.standard_normal()
        freq = (p['drift_hz_per_s'] * t + wander
                + p['resp_amp_hz'] * np.sin(2.0 * np.pi * p['resp_freq_hz'] * t
                                            + p['resp_phase']))

        # Phase: coupled to the frequency excursion, plus receiver jitter.
        phase = (np.deg2rad(p['coupling_deg_per_hz']) * (freq - freq.mean())
                 + np.deg2rad(p['phase_jitter_deg']) * rng.standard_normal(n))

        # Amplitude and lineshape: small wander, growing shim degradation.
        amp = 1.0 + p['amp_jitter_frac'] * rng.standard_normal(n)
        broaden = np.clip(p['broaden_hz'] * t / scan_s, 0.0, None)

        # Motion events: sparse, clustered, occasionally persistent.
        n_events = rng.poisson(p['events_per_min'] * scan_s / 60.0)
        for _ in range(int(n_events)):
            start = int(rng.integers(0, n))
            stop = min(n, start + 1 + int(rng.geometric(0.5)))
            hit = slice(start, min(stop, start + 6))
            disruptive = rng.random() < 0.3

            freq[hit] += 5.0 * rng.standard_normal()
            phase[hit] += np.deg2rad(15.0) * rng.standard_normal()
            if disruptive:
                amp[hit] *= 1.0 - rng.uniform(0.05, 0.5)
                broaden[hit] += rng.uniform(5.0, 20.0)
            if rng.random() < 0.5:
                freq[hit.stop:] += 3.0 * rng.standard_normal()
                broaden[hit.stop:] += rng.uniform(0.5, 3.0)

        return freq, phase, amp, broaden

    #*****************#
    #   synthesis     #
    #*****************#
    def _train_factor(self, n_pts: int, sw_hz: float, rng) -> np.ndarray:
        """
        The complex "(n_pts, n_transients)" factor that turns one FID into a
        train: "a_n * exp(-pi * L_n * t) * exp(i * (2 pi f_n t + phi_n))".
        """
        freq, phase, amp, broaden = self._transient_tracks(rng)
        t = (np.arange(n_pts, dtype=np.float64) / float(sw_hz))[:, None]

        return (amp[None, :]
                * np.exp(-np.pi * broaden[None, :] * t)
                * np.exp(1j * (2.0 * np.pi * freq[None, :] * t + phase[None, :]))
                ).astype(np.complex64)

    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.NUMPY, **kwargs):
        """
        Expand a batch of FIDs into transient trains.

        Args:
            data_array: "(batch, ..., n_points)" complex FIDs, without a
                transient axis — averaging first is what this module inverts.
            water_array: Passed through unchanged.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: Must contain "'sw_hz'"; reads "dim_tags" to refuse data
                that already carries DIM_DYN.

        Returns:
            "(data with a trailing transient axis, water_unchanged)".
        """
        sw_hz = kwargs.get('sw_hz')
        if sw_hz is None:
            raise ValueError("TransientSynthesizer.process_tensor requires 'sw_hz' in kwargs")
        if 'DIM_DYN' in (kwargs.get('dim_tags') or ()):
            raise ValueError(
                "The data already carries transients (DIM_DYN). Synthesizing a "
                "train on top of one would nest acquisition axes — average "
                "first, or use AverageSampler on the real transients instead.")

        n_pts = int(data_array.shape[-1])
        n_batch = int(data_array.shape[0])
        rng = self.rng.numpy_rng()

        # One scan realization per batch element: (batch, 1, ..., 1, T, N).
        factor = np.stack([self._train_factor(n_pts, sw_hz, rng)
                           for _ in range(n_batch)])
        factor = factor.reshape((n_batch,) + (1,) * (data_array.ndim - 2)
                                + factor.shape[1:])

        train = data_array[..., None] * cast_like(
            match_backend(factor, data_array), data_array)
        return train, water_array

    def _spectral_axis_back(self, data_array):
        """
        Undo the spectral relocation around the axis this module appended.

        The base hook restores the caller's layout by rank, but the train
        grew the rank by one: T returns to its slot, the axes behind it keep
        their order, and the new transient axis stays last — which is where
        "ADDS_DIM_TAGS" says it lives.
        """
        from nifti_mrs_plus import ops

        rank = self._axis_rank + 1
        order = (list(range(self.SPECTRAL_AXIS)) + [rank - 2]
                 + list(range(self.SPECTRAL_AXIS, rank - 2)) + [rank - 1])
        return ops.transpose(data_array, order)
