####################################################################################################
#                                      apodization.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Implements FID truncation and exponential weighting (apodization) for MRS data.         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List
from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from nifti_mrs_plus.ops import match_backend


#**************************************************************************************************#
#                                        Class Apodization                                         #
#**************************************************************************************************#
#                                                                                                  #
# Apply apodization (truncation or exponential weighting) to MRS data.                             #
#                                                                                                  #
#**************************************************************************************************#
class Apodization(BaseModule):
    """
    Apply apodization (truncation or exponential weighting) to MRS data.

    Supports two modes:
    - 'truncate': Cut FID after specified number of points
    - 'exponential': Apply exponential decay (Lorentzian broadening)

    Parameters
    ----------
    mode : str
        Apodization mode: 'truncate', 'exponential', 'gaussian', or 'hamming'
        (default: 'exponential')
    n_pts : int, optional
        Number of points to keep (for truncation mode)
    frac_pts : float, optional
        Fraction of points to keep (for truncation mode), range [0, 1]
    lb_hz : float, optional
        Lorentzian broadening in Hz (for exponential mode)
    gb_hz : float, optional
        Gaussian broadening FWHM in Hz (for gaussian mode)
    auto_lb : bool
        Auto-calculate lb_hz to preserve signal (default: False)
    target_damp : float
        Target damping factor for auto_lb (default: 0.01)
    target_pts : int, optional
        Target point where signal becomes mostly noise (for auto_lb)

    Examples
    --------
    >>> # Truncation mode
    >>> apod = Apodization(mode='truncate', n_pts=1024)
    >>>
    >>> # Exponential mode with manual lb
    >>> apod = Apodization(mode='exponential', lb_hz=5.0)
    >>>
    >>> # Gaussian window
    >>> apod = Apodization(mode='gaussian', gb_hz=8.0)
    >>>
    >>> # Hamming window over the acquisition
    >>> apod = Apodization(mode='hamming')
    >>>
    >>> # Exponential mode with auto lb
    >>> apod = Apodization(mode='exponential', auto_lb=True, target_pts=1024)
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # A window and a truncation both act on the FID.
    DOMAIN = Domain(spectral='time')

    MODES = ('truncate', 'exponential', 'gaussian', 'hamming')

    def __init__(self, mode: str = 'exponential',
                 n_pts: Optional[int] = None,
                 frac_pts: Optional[float] = None,
                 lb_hz: Optional[float] = None,
                 gb_hz: Optional[float] = None,
                 auto_lb: bool = False,
                 target_damp: float = 0.01,
                 target_pts: Optional[int] = None):
        """Initialize apodization module."""
        super().__init__()

        self.mode = mode.lower()
        self.n_pts = n_pts
        self.frac_pts = frac_pts
        self.lb_hz = lb_hz
        self.gb_hz = gb_hz
        self.auto_lb = auto_lb
        self.target_damp = target_damp
        self.target_pts = target_pts

        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Apply apodization to list of NIFTI_MRS objects.

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

            # Get spectral width (for exponential mode)
            sw_hz = 1.0 / nifti.dwelltime

            # Apply apodization
            if self.mode == 'truncate':
                fid_apod = self._apply_truncation(fid)
            else:  # a weighting window
                fid_apod = self._apply_window(fid, sw_hz)

            # Update NIFTI_MRS data
            if fid_apod.shape[-1] != fid.shape[-1]:
                # Shape changed (truncation) — must build a new NIFTI_MRS object
                spec_freq = nifti.spectrometer_frequency[0]
                nucleus = nifti.nucleus[0] if nifti.nucleus else '1H'
                dim_tags = list(nifti.dim_tags) if hasattr(nifti, 'dim_tags') else [None, None, None]
                affine = nifti.getAffine('voxel', 'world') if hasattr(nifti, 'getAffine') else None

                new_nifti = gen_nifti_mrs(
                    data=fid_apod,
                    dwelltime=nifti.dwelltime,
                    spec_freq=spec_freq,
                    nucleus=nucleus,
                    dim_tags=dim_tags,
                    affine=affine,
                )
                # Copy across any extra header fields
                if hasattr(nifti, 'hdr_ext'):
                    for key in nifti.hdr_ext:
                        if key not in ('SpectrometerFrequency', 'ResonantNucleus',
                                       'dim_5', 'dim_6', 'dim_7'):
                            try:
                                new_nifti.add_hdr_field(key, nifti.hdr_ext[key])
                            except Exception:
                                pass
                nifti = new_nifti
            else:
                nifti[:] = fid_apod
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Apply apodization to tensor/array data (**any backend**).

        **Truncation mode**: slices the last axis with "data[..., :n_keep]"
        — works natively in NumPy, PyTorch, JAX, and TensorFlow without any
        conversion.

        **Exponential mode**: the decay envelope is a numpy array (no data
        dependency) multiplied via "data * match_backend(envelope, data)"
        — fully backend-native multiply, gradients preserved.

        Args:
            data_array: Input tensor of shape "(batch, ..., n_points)"
            water_array: Optional water reference (unchanged)
            backend: Backend enum (unused)
            **kwargs: "'sw_hz'" required for exponential mode

        Returns:
            Tuple of (processed_data, water_array)
        """
        if self.mode == 'truncate':
            N = data_array.shape[-1]
            if self.n_pts is not None:
                n_keep = int(np.clip(self.n_pts, 1, N))
            elif self.frac_pts is not None:
                if not (0 < self.frac_pts <= 1):
                    raise ValueError("frac_pts must be in range (0, 1]")
                n_keep = int(np.ceil(self.frac_pts * N))
            else:
                raise ValueError("Must provide n_pts or frac_pts for truncation mode")
            # Native slice — valid in numpy, torch, jax, tf
            return data_array[..., :n_keep], water_array

        else:  # a weighting window
            sw_hz = kwargs.get('sw_hz')
            if sw_hz is None:
                raise ValueError(f"Apodization.process_tensor requires 'sw_hz' for {self.mode} mode")

            N = data_array.shape[-1]
            envelope = self._envelope(N, sw_hz)
            if envelope is None:
                return data_array, water_array

            envelope = envelope.reshape([1] * (len(data_array.shape) - 1) + [N])
            # Backend-native multiply (envelope is data-independent numpy array)
            return data_array * match_backend(envelope, data_array), water_array

    def _apply_truncation(self, fid: np.ndarray) -> np.ndarray:
        """
        Apply truncation apodization.

        Args:
            fid: Input FID data

        Returns:
            Truncated FID
        """
        N = fid.shape[-1]

        # Determine number of points to keep
        if self.n_pts is not None:
            data_keep = int(np.clip(self.n_pts, 1, N))
        elif self.frac_pts is not None:
            if not (0 < self.frac_pts <= 1):
                raise ValueError("frac_pts must be in range (0, 1]")
            data_keep = int(np.ceil(self.frac_pts * N))
        else:
            raise ValueError("Must provide n_pts or frac_pts for truncation mode")

        # Truncate along last dimension
        return fid[..., :data_keep]

    def _apply_window(self, fid: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Multiply the FID by this mode's weighting window.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz

        Returns:
            Apodized FID
        """
        N = fid.shape[-1]
        envelope = self._envelope(N, sw_hz)
        if envelope is None:
            return fid

        return fid * envelope.reshape(tuple([1] * (fid.ndim - 1) + [N]))

    def _envelope(self, n_pts: int, sw_hz: float) -> Optional[np.ndarray]:
        """
        This mode's weighting window "w(t)" over *n_pts* samples.

        Every window is data-independent, so both processing paths build it
        here once, in NumPy, and multiply it in natively.

        Returns:
            The "(n_pts,)" window, or None when the weight would be identity.
        """
        t = np.arange(n_pts, dtype=np.float64) / float(sw_hz)

        if self.mode == 'exponential':
            if self.auto_lb:
                if self.target_pts is None:
                    raise ValueError("Must provide target_pts when auto_lb=True")
                lb = self._calc_lb(None, sw_hz, self.target_pts, self.target_damp)
            elif self.lb_hz is not None:
                lb = self.lb_hz
            else:
                raise ValueError("Must provide lb_hz or set auto_lb=True for exponential mode")
            return None if lb <= 0 else np.exp(-np.pi * float(lb) * t)

        if self.mode == 'gaussian':
            if self.gb_hz is None:
                raise ValueError("Must provide gb_hz for gaussian mode")
            if self.gb_hz <= 0:
                return None
            return np.exp(-((np.pi * float(self.gb_hz) * t) ** 2) / (4.0 * np.log(2.0)))

        # 'hamming' — the decaying half of the window over the acquisition,
        # 1 at t=0 falling to 0.08 at the last sample. No parameter to it.
        n = np.arange(n_pts, dtype=np.float64)
        return 0.54 + 0.46 * np.cos(np.pi * n / max(n_pts - 1, 1))

    @staticmethod
    def _calc_lb(fid, sw_hz: float, desired_npts: int, target_damp: float = 0.01) -> float:
        """
        Calculate Lorentzian broadening to achieve target damping.

        Args:
            fid: FID data
            sw_hz: Spectral width in Hz
            desired_npts: Point where signal is mostly noise
            target_damp: Target damping factor (default: 0.01 = 1% of signal)

        Returns:
            Lorentzian broadening in Hz
        """
        return (-sw_hz / (np.pi * desired_npts)) * np.log(target_damp)
