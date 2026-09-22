####################################################################################################
#                                     line_broadening.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Implements Lorentzian, Gaussian, and Voigt line broadening for MRS data, and           #
#          Lorentzian narrowing for negative widths.                                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math
import warnings
import numpy as np
from typing import Optional, List

from augmentrum.core.base_module import BaseModule
from augmentrum.core import precision as prec
from augmentrum.processing.domain import Domain
from augmentrum.processing.utils import to_backend
from nifti_mrs_plus import Backend, ops


#**************************************************************************************************#
#                                       Class LineBroadening                                       #
#**************************************************************************************************#
#                                                                                                  #
# Apply line broadening to MRS data.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class LineBroadening(BaseModule):
    """
    Apply line broadening to MRS data.

    Supports Lorentzian (exponential), Gaussian, and Voigt (combined) broadening.
    A negative Lorentzian width narrows the lines instead (see Notes).

    **Backend-agnostic**: all "process_tensor" operations go through
    "nifti_mrs_plus.ops", which dispatches on the tensor it is given, so they
    work transparently with NumPy arrays, PyTorch tensors, JAX arrays and
    TensorFlow tensors without losing gradients or leaving the device.

    Parameters
    ----------
    lb_hz : float
        Lorentzian broadening Full Width at Half Maximum (FWHM) in Hz. Negative
        values narrow the lines by that much (see Notes).
    gb_hz : float
        Gaussian broadening FWHM in Hz, >= 0
    mode : str
        Broadening type: 'lorentzian', 'gaussian', or 'voigt' (default)
    narrow_cap_s : float, optional
        For a negative lb_hz only: the time in seconds after which the rising
        narrowing envelope stops growing. It limits how much late-FID noise is
        amplified, at the cost of a small change to the line shape. None
        (default) narrows exactly.

    Notes
    -----
    A negative lb_hz multiplies the FID by exp(+pi |lb_hz| t): the lines get
    narrower by |lb_hz| Hz (a longer T2*), exactly for the Lorentzian part of
    every line. This comes at a cost, and a UserWarning says so the first time
    a module narrows:

    - The noise is amplified along with the signal, most at the end of the FID.
      Over a 0.512 s acquisition, narrowing by 1 Hz raises the noise standard
      deviation 2.7-fold, by 1.5 Hz 5.1-fold; capping the envelope at 0.2 s
      (narrow_cap_s=0.2) brings these to 1.7 and 2.3. On the 90 COWS scans
      (3 T, NAA FWHM 4.8 Hz) 1 Hz of narrowing keeps 44 % of the NAA
      SNR, 69 % with that cap, which moves the NAA line by less than 1 % of
      its peak.
    - It is only physical while |lb_hz| stays below the narrowest Lorentzian
      width in the data. Beyond that a line's FID grows with time and its
      spectrum is no longer a line an MR scanner could measure.
    - A Gaussian cannot be narrowed this way (its FID would grow as exp(+t^2)),
      so a negative gb_hz raises a ValueError.

    Examples
    --------
    >>> # Lorentzian broadening only
    >>> broadening = LineBroadening(lb_hz=5.0, mode='lorentzian')
    >>>
    >>> # Gaussian broadening only
    >>> broadening = LineBroadening(gb_hz=3.0, mode='gaussian')
    >>>
    >>> # Voigt (combined) broadening
    >>> broadening = LineBroadening(lb_hz=5.0, gb_hz=3.0, mode='voigt')
    >>>
    >>> # Lorentzian narrowing by up to 1 Hz or broadening by up to 3 Hz, the
    >>> # narrowing envelope capped at 0.2 s
    >>> broadening = LineBroadening(lb_hz=(-1.0, 3.0), mode='lorentzian', narrow_cap_s=0.2)
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # Acts on every coil and transient alike, so per-sample masks pass through.
    MASKS = 'pass'

    # Broadening multiplies the FID by a decay. The same operation in a
    # spectrum would be a convolution, not a multiply.
    DOMAIN = Domain(spectral='time')

    # The envelope broadcasts, so a batch can carry one width per sample.
    PER_SAMPLE_PARAMS = ('lb_hz', 'gb_hz')

    def __init__(self, lb_hz: float = 0.0, gb_hz: float = 0.0, mode: str = 'voigt',
                 narrow_cap_s: Optional[float] = None):
        """Initialize line broadening module."""
        super().__init__()

        self.lb_hz = lb_hz
        self.gb_hz = gb_hz
        self.mode = mode.lower()
        self.narrow_cap_s = narrow_cap_s
        self._warned_narrowing = False

        if self.mode not in ['lorentzian', 'gaussian', 'voigt']:
            raise ValueError(f"mode must be 'lorentzian', 'gaussian', or 'voigt', got '{mode}'")
        if narrow_cap_s is not None and not narrow_cap_s > 0:
            raise ValueError(f'narrow_cap_s must be > 0 seconds or None, got {narrow_cap_s}')

    def _check_widths(self, lb_hz, gb_hz):
        """Reject a negative Gaussian width; warn once when a Lorentzian width narrows."""
        if self.mode != 'lorentzian' and np.any(np.asarray(gb_hz) < 0):
            raise ValueError(
                f'gb_hz must be >= 0, got {gb_hz}: a Gaussian line cannot be narrowed by '
                'the FID envelope (it would grow as exp(+t^2))')
        if (self.mode != 'gaussian' and not self._warned_narrowing
                and np.any(np.asarray(lb_hz) < 0)):
            self._warned_narrowing = True
            cap = ('' if self.narrow_cap_s is None
                   else f', here capped at {self.narrow_cap_s:g} s by narrow_cap_s')
            warnings.warn(
                'LineBroadening: a negative lb_hz narrows the lines by multiplying the FID by '
                'exp(+pi |lb_hz| t). This also amplifies the noise, most at the end of the FID '
                '(over a 0.5 s acquisition, 1 Hz of narrowing raises the noise SD about '
                f'2.7-fold{cap}), and it is only physical while |lb_hz| stays below the '
                'narrowest Lorentzian width in the data: beyond that the FID grows with time.',
                UserWarning, stacklevel=3)

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Apply line broadening to list of NIFTI_MRS objects.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, processed_water_list)
        """
        processed_data = []

        for i, nifti in enumerate(data_list):
            # Get spectral width from NIFTI_MRS
            sw_hz = 1.0 / nifti.dwelltime

            # Get FID data
            fid = nifti[:]

            # This subject's widths, out of a per-sample vector or the scalar
            lb_hz = self.sample_of(self.lb_hz, i)
            gb_hz = self.sample_of(self.gb_hz, i)
            self._check_widths(lb_hz, gb_hz)

            # Apply broadening based on mode
            cap = self.narrow_cap_s
            if self.mode == 'lorentzian':
                fid_broadened = self._apply_lorentzian(fid, sw_hz, lb_hz, cap)
            elif self.mode == 'gaussian':
                fid_broadened = self._apply_gaussian(fid, sw_hz, gb_hz)
            else:  # voigt
                fid_broadened = self._apply_voigt(fid, sw_hz, lb_hz, gb_hz, cap)

            # Update NIFTI_MRS data
            nifti[:] = fid_broadened
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Apply line broadening to tensor/array data (**any backend**).

        Works identically with NumPy, PyTorch, JAX, or TensorFlow tensors.
        Gradients are preserved for differentiable training pipelines.

        Args:
            data_array: Input tensor of shape "(batch, ..., n_points)"
            water_array: Optional water reference tensor (unchanged)
            backend: Backend enum (unused — ops dispatch on the tensor)
            **kwargs: Must contain "'sw_hz'" (spectral width in Hz)

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        if sw_hz is None:
            raise ValueError("LineBroadening.process_tensor requires 'sw_hz' in kwargs")

        self._check_widths(self.lb_hz, self.gb_hz)
        cap = self.narrow_cap_s
        if self.mode == 'lorentzian':
            result = self._apply_lorentzian(data_array, sw_hz, self.lb_hz, cap)
        elif self.mode == 'gaussian':
            result = self._apply_gaussian(data_array, sw_hz, self.gb_hz)
        else:  # voigt
            result = self._apply_voigt(data_array, sw_hz, self.lb_hz, self.gb_hz, cap)

        return result, water_array

    #**********************#
    #   envelope helpers   #
    #**********************#
    # The envelope is built on the FID's own backend, so a torch FID keeps its
    # device and its autograd graph and a JAX FID stays traceable. `ops`
    # dispatches on the tensor it is handed rather than on a global setting.

    @staticmethod
    def _width(value, t, ndim):
        """A width in Hz as something *t* can be multiplied by: a plain float,
        or a per-sample column promoted onto *t*'s backend."""
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return float(arr)
        return to_backend(arr.reshape((-1,) + (1,) * (ndim - 1)), t)

    @staticmethod
    def _make_time_envelope(fid, sw_hz, lb_hz, gb_hz, narrow_cap_s=None):
        """
        Build the broadening envelope on *fid*'s backend, shaped to broadcast.

        Positive Lorentzian widths decay as exp(-pi lb t), negative ones rise as
        exp(+pi |lb| t'), where t' stops at *narrow_cap_s* when it is given. The
        two parts are separate factors, so non-negative widths give exactly the
        envelope they always did.
        """
        n_pts = fid.shape[-1]
        ndim = len(fid.shape)
        t = ops.arange_like(fid, n_pts, dtype=prec.real_name(fid)) / float(sw_hz)

        # Reshape t for broadcasting: (1, 1, ..., N)
        t = ops.reshape(t, [1] * (ndim - 1) + [n_pts])

        env = None
        lb_all = np.asarray(lb_hz, dtype=np.float64)
        narrowing = np.any(lb_all < 0)
        if np.any(lb_all > 0):
            # without narrowing the widths go in as given: the same envelope as ever
            decay = np.maximum(lb_all, 0.0) if narrowing else lb_hz
            lb = LineBroadening._width(decay, t, ndim)
            env = ops.exp(-math.pi * lb * t)
        if narrowing:
            narrow = LineBroadening._width(np.minimum(lb_all, 0.0), t, ndim)
            t_rise = t if narrow_cap_s is None else ops.clip(t, 0.0, float(narrow_cap_s))
            rise = ops.exp(-math.pi * narrow * t_rise)
            env = rise if env is None else env * rise
        if np.any(np.asarray(gb_hz) > 0):
            gb = LineBroadening._width(gb_hz, t, ndim)
            gauss = ops.exp(-((math.pi * gb * t) ** 2) / (4 * math.log(2)))
            env = gauss if env is None else env * gauss
        return env

    @staticmethod
    def _apply_lorentzian(fid, sw_hz, lb_hz, narrow_cap_s=None):
        """Apply Lorentzian (exponential) line broadening, or narrowing for lb_hz < 0."""
        if not np.any(np.asarray(lb_hz) != 0):
            return fid
        envelope = LineBroadening._make_time_envelope(fid, sw_hz, lb_hz, 0.0, narrow_cap_s)
        return fid * ops.cast_like(envelope, fid)

    @staticmethod
    def _apply_gaussian(fid, sw_hz, gb_hz):
        """Apply Gaussian line broadening."""
        if not np.any(np.asarray(gb_hz) > 0):
            return fid
        envelope = LineBroadening._make_time_envelope(fid, sw_hz, 0.0, gb_hz)
        return fid * ops.cast_like(envelope, fid)

    @staticmethod
    def _apply_voigt(fid, sw_hz, lb_hz, gb_hz, narrow_cap_s=None):
        """Apply Voigt (Lorentzian × Gaussian) line broadening; lb_hz < 0 narrows."""
        if not (np.any(np.asarray(lb_hz) != 0) or np.any(np.asarray(gb_hz) > 0)):
            return fid
        envelope = LineBroadening._make_time_envelope(fid, sw_hz, lb_hz, gb_hz, narrow_cap_s)
        return fid * ops.cast_like(envelope, fid)
