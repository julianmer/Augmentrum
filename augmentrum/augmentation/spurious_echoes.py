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
#          Supports simple replica model and hybrid model (Kyathanahally et al. 2021,              #
#          Bugler et al. 2025).                                                                    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List, Tuple, Union
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from augmentrum.utils.tensor_ops import to_numpy, match_backend


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

    Supports two modes:
    - 'replica': Simple multiplicative echo — multiplies FID by a decaying
      envelope after a delay (classic replica model).
    - 'hybrid': Physically shifts the FID in time, then shapes with a
      localized envelope (Kyathanahally/Bugler model). Produces more
      realistic ghosting artifacts.

    Parameters
    ----------
    echoes : list of dicts or tuples
        Each echo specifies:
        - delay_s / tau: Echo delay time in seconds
        - amp / alpha: Relative amplitude (vs max |FID|)
        - phase_deg: Additional phase for this echo (degrees)
        - decay_hz: Exponential decay in Hz (replica mode)
        - freq_hz / df_hz: Frequency offset of echo (Hz)

        For hybrid mode, additional per-echo keys:
        - t_echo: Echo center time in seconds (defaults to tau)
        - T2: Envelope time constant in seconds (defaults to 0.04)
        - gaussian_env: Use Gaussian envelope (bool, default False)

    mode : str
        'replica' (default) or 'hybrid'
    global_phase_deg : float
        Global phase offset for all echoes (default: 0.0)
    alpha_reference : str
        How amplitude is referenced in hybrid mode:
        'max' (default) = fraction of max(|FID|),
        'tau' = fraction of |FID| at delay time

    Examples
    --------
    >>> # Replica mode (simple)
    >>> se = SpuriousEchoes(
    ...     echoes=[{'delay_s': 0.1, 'amp': 0.3, 'phase_deg': 0,
    ...              'decay_hz': 5.0, 'freq_hz': 0.0}]
    ... )
    >>> result_data, _ = se(nifti_plus, None)

    >>> # Hybrid mode (matches legacy add_spurious_echo_artifact)
    >>> se = SpuriousEchoes(
    ...     mode='hybrid',
    ...     echoes=[{'tau': 0.18, 'alpha': 0.32, 'phase_deg': 25,
    ...              't_echo': 0.26, 'T2': 0.03, 'df_hz': 35.0}]
    ... )
    >>> result_data, _ = se(nifti_plus, None)
    """

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self, echoes=None,
                 mode: str = 'replica',
                 global_phase_deg: float = 0.0,
                 alpha_reference: str = 'max'):
        """Initialize spurious echoes module."""
        if echoes is None:
            echoes = [{'delay_s': 0.1, 'amp': 0.2, 'phase_deg': 0.0,
                       'decay_hz': 5.0, 'freq_hz': 0.0}]

        super().__init__(echoes=echoes, mode=mode,
                         global_phase_deg=global_phase_deg,
                         alpha_reference=alpha_reference)

        self.mode = mode.lower()
        if self.mode not in ('replica', 'hybrid'):
            raise ValueError(f"mode must be 'replica' or 'hybrid', got '{mode}'")

        self.global_phase_deg = global_phase_deg
        self.alpha_reference = alpha_reference

        # Normalize echoes to a list of dicts
        self.echoes = []
        for echo in echoes:
            if isinstance(echo, dict):
                self.echoes.append(dict(echo))
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

        for nifti in data_list:
            # Get FID data
            fid = nifti[:]

            # Get spectral width
            sw_hz = 1.0 / nifti.dwelltime

            # Add echoes
            fid_with_echoes = self._add_echoes(fid, sw_hz)

            # Update NIFTI_MRS data
            nifti[:] = fid_with_echoes
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add spurious echo artifacts to tensor/array data (**any backend**).

        **Replica mode**: echo envelopes are data-independent (pure numpy)
        and applied via ``data + data * match_backend(envelope, data)`` —
        fully backend-native multiply/add.

        **Hybrid mode**: echo amplitude is referenced to ``max|FID|`` which
        requires a scalar pull to numpy; the additive ghost is then applied
        via ``data + match_backend(ghost_additive, data)``.

        Args:
            data_array: Input tensor of shape ``(batch, ..., n_points)``
            water_array: Optional water reference (unchanged)
            backend: Backend enum (unused)
            **kwargs: Must contain ``'sw_hz'`` (spectral width in Hz)

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        if sw_hz is None:
            raise ValueError("SpuriousEchoes.process_tensor requires 'sw_hz' in kwargs")

        original_shape = data_array.shape
        N = original_shape[-1]

        if self.mode == 'replica':
            # Envelope is purely time-based — no data dependency at all
            t = np.arange(N, dtype=np.float64) / float(sw_hz)
            t_shape = [1] * (len(original_shape) - 1) + [N]
            t = t.reshape(t_shape)

            gphi = np.deg2rad(self.global_phase_deg)
            echo_sum = np.zeros(t_shape, dtype=np.complex128)

            for echo in self.echoes:
                delay_s = echo.get('delay_s', echo.get('tau', 0.1))
                amp = echo.get('amp', echo.get('alpha', echo.get('amplitude', 0.1)))
                phase_deg = echo.get('phase_deg', 0.0)
                decay_hz = echo.get('decay_hz', echo.get('decay', 5.0))
                freq_hz = echo.get('freq_hz', echo.get('df_hz', 0.0))

                if delay_s > 1.0:
                    delay_s = delay_s / float(sw_hz)

                u = (t >= delay_s).astype(np.float64)
                td = (t - delay_s) * u
                env = (amp
                       * np.exp(1j * (gphi + np.deg2rad(phase_deg)))
                       * np.exp(-np.pi * float(decay_hz) * td)
                       * np.exp(1j * 2.0 * np.pi * float(freq_hz) * td))
                echo_sum += env

            # Apply: fid_out = fid + fid * sum_of_envelopes  (backend-native)
            return data_array + data_array * match_backend(echo_sum, data_array), water_array

        else:  # hybrid mode — amplitude references max(|fid|): pull per-FID to numpy
            n_batch = int(np.prod(original_shape[:-1]))
            data_np = to_numpy(data_array).reshape(n_batch, N)
            ghost_np = np.zeros_like(data_np)

            for i in range(n_batch):
                fid_1d = data_np[i]
                result = self._add_echoes_hybrid_1d(fid_1d, sw_hz)
                ghost_np[i] = result - fid_1d  # additive contribution only

            ghost_np = ghost_np.reshape(original_shape)
            return data_array + match_backend(ghost_np, data_array), water_array

    def _add_echoes(self, fid: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Add spurious echoes to FID data.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz

        Returns:
            FID with echoes added
        """
        # Work with last dimension (FID points)
        original_shape = fid.shape
        N = original_shape[-1]

        # Reshape to 2D for processing
        fid_2d = fid.reshape(-1, N)
        result = np.zeros_like(fid_2d)

        # Process each FID
        for i in range(fid_2d.shape[0]):
            fid_1d = fid_2d[i]
            if self.mode == 'hybrid':
                result[i] = self._add_echoes_hybrid_1d(fid_1d, sw_hz)
            else:
                result[i] = self._add_echoes_replica_1d(fid_1d, sw_hz)

        # Reshape back to original shape
        return result.reshape(original_shape)

    # ── Replica mode (original simple model) ──────────────────────────────

    def _add_echoes_replica_1d(self, fid: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Add echoes to 1D FID using replica model.

        The echo is the original FID multiplied by a decaying, frequency-shifted
        envelope that starts at the delay time.

        Args:
            fid: 1D FID data
            sw_hz: Spectral width in Hz

        Returns:
            FID with echoes
        """
        f = np.asarray(fid, dtype=np.complex128)
        n = f.size
        t = np.arange(n, dtype=float) / float(sw_hz)

        out = f.copy()
        gphi = np.deg2rad(self.global_phase_deg)

        for echo in self.echoes:
            delay_val = echo.get('delay_s', echo.get('tau', 0.1))
            amp = echo.get('amp', echo.get('alpha', echo.get('amplitude', 0.1)))
            phase_deg = echo.get('phase_deg', 0.0)
            decay_hz = echo.get('decay_hz', echo.get('decay', 5.0))
            freq_hz = echo.get('freq_hz', echo.get('df_hz', 0.0))

            # delay_val might be in seconds or points - convert to seconds
            if delay_val > 1.0:
                delay_s = delay_val / float(sw_hz)
            else:
                delay_s = delay_val

            # Step function: 1 after delay, 0 before
            u = (t >= delay_s).astype(float)
            td = (t - delay_s) * u

            # Echo signal
            echo_env = (
                amp
                * np.exp(1j * (gphi + np.deg2rad(phase_deg)))
                * np.exp(-np.pi * float(decay_hz) * td)
                * np.exp(1j * 2.0 * np.pi * float(freq_hz) * td)
            )

            # Add delayed replica
            out += f * echo_env

        return out

    # ── Hybrid mode (Kyathanahally / Bugler model) ───────────────────────

    def _add_echoes_hybrid_1d(self, fid: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Add echoes to 1D FID using hybrid model.

        Physically shifts the FID in time, then applies a localized envelope
        and frequency modulation. Matches the legacy add_spurious_echo_artifact
        (Kyathanahally et al. 2021, Bugler et al. 2025).

        Args:
            fid: 1D FID data
            sw_hz: Spectral width in Hz

        Returns:
            FID with echoes
        """
        f = np.asarray(fid, dtype=np.complex128)
        n = f.size
        dt = 1.0 / float(sw_hz)
        t = np.arange(n, dtype=float) * dt

        out = f.copy()
        gphi = np.deg2rad(self.global_phase_deg)

        for echo in self.echoes:
            # Extract parameters (support both naming conventions)
            alpha = echo.get('alpha', echo.get('amp', echo.get('amplitude', 0.05)))
            phase_deg = echo.get('phase_deg', 0.0)
            T2 = echo.get('T2', 0.04)
            df_hz = echo.get('df_hz', echo.get('freq_hz', 0.0))
            gaussian_env = echo.get('gaussian_env', False)

            # Determine shift in samples — support delay_pts directly
            if 'delay_pts' in echo:
                shift_pts = int(echo['delay_pts'])
                tau = shift_pts * dt
            else:
                tau = echo.get('tau', echo.get('delay_s', 0.18))
                shift_pts = int(round(tau / dt))

            t_echo = echo.get('t_echo', tau)

            # Delayed (physically shifted) copy of the FID
            delayed_fid = np.zeros_like(f)
            if shift_pts < n:
                delayed_fid[shift_pts:] = f[:n - shift_pts]

            # Phase + frequency modulation
            phase_rad = gphi + np.deg2rad(phase_deg)
            mod = np.exp(1j * (2.0 * np.pi * df_hz * (t - tau) + phase_rad))

            # Build ghost: handle no-envelope case (T2 >= 1e4) for exact
            # match with simple shift-and-add legacy model
            if T2 >= 1e4:
                # No envelope — simple: ghost = alpha * delayed_fid * mod
                ghost = alpha * delayed_fid * mod
            else:
                # Full hybrid: normalized FID × envelope × amplitude
                if self.alpha_reference == 'tau':
                    tau_idx = min(shift_pts, n - 1)
                    Ag = alpha * np.abs(f[tau_idx])
                else:  # 'max'
                    Ag = alpha * (np.max(np.abs(f)) + 1e-30)

                delayed_norm = delayed_fid / (np.max(np.abs(f)) + 1e-30)

                if gaussian_env:
                    env = np.exp(-((t - t_echo) ** 2) / (2.0 * T2 ** 2))
                else:
                    env = np.exp(-np.abs(t - t_echo) / T2)

                ghost = Ag * delayed_norm * env * mod

            out += ghost

        return out
