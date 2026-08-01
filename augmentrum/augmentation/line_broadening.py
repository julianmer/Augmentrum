####################################################################################################
#                                     line_broadening.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Implements Lorentzian, Gaussian, and Voigt line broadening for MRS data.                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math
import numpy as np
from typing import Optional, List

from augmentrum.core.base_module import BaseModule
from augmentrum.utils.tensor_ops import match_backend


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

    **Backend-agnostic** (zea pattern): all ``process_tensor`` operations use
    ``keras.ops`` so they work transparently with NumPy arrays, PyTorch tensors,
    JAX arrays, and TensorFlow tensors without losing gradients.

    Parameters
    ----------
    lb_hz : float
        Lorentzian broadening Full Width at Half Maximum (FWHM) in Hz
    gb_hz : float
        Gaussian broadening FWHM in Hz
    mode : str
        Broadening type: 'lorentzian', 'gaussian', or 'voigt' (default)

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
    """

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self, lb_hz: float = 0.0, gb_hz: float = 0.0, mode: str = 'voigt'):
        """Initialize line broadening module."""
        super().__init__(lb_hz=lb_hz, gb_hz=gb_hz, mode=mode)

        self.lb_hz = lb_hz
        self.gb_hz = gb_hz
        self.mode = mode.lower()

        if self.mode not in ['lorentzian', 'gaussian', 'voigt']:
            raise ValueError(f"mode must be 'lorentzian', 'gaussian', or 'voigt', got '{mode}'")

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

        for nifti in data_list:
            # Get spectral width from NIFTI_MRS
            sw_hz = 1.0 / nifti.dwelltime

            # Get FID data
            fid = nifti[:]

            # Apply broadening based on mode
            if self.mode == 'lorentzian':
                fid_broadened = self._apply_lorentzian(fid, sw_hz, self.lb_hz)
            elif self.mode == 'gaussian':
                fid_broadened = self._apply_gaussian(fid, sw_hz, self.gb_hz)
            else:  # voigt
                fid_broadened = self._apply_voigt(fid, sw_hz, self.lb_hz, self.gb_hz)

            # Update NIFTI_MRS data
            nifti[:] = fid_broadened
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Apply line broadening to tensor/array data (**any backend**).

        Uses ``keras.ops`` — works identically with NumPy, PyTorch, JAX, or
        TensorFlow tensors.  Gradients are preserved for differentiable
        training pipelines.

        Args:
            data_array: Input tensor of shape ``(batch, ..., n_points)``
            water_array: Optional water reference tensor (unchanged)
            backend: Backend enum (unused — ops dispatch automatically)
            **kwargs: Must contain ``'sw_hz'`` (spectral width in Hz)

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        if sw_hz is None:
            raise ValueError("LineBroadening.process_tensor requires 'sw_hz' in kwargs")

        if self.mode == 'lorentzian':
            result = self._apply_lorentzian(data_array, sw_hz, self.lb_hz)
        elif self.mode == 'gaussian':
            result = self._apply_gaussian(data_array, sw_hz, self.gb_hz)
        else:  # voigt
            result = self._apply_voigt(data_array, sw_hz, self.lb_hz, self.gb_hz)

        return result, water_array

    #*************************************************************#
    #   envelope helpers — fully backend-agnostic via keras.ops   #
    #*************************************************************#
    # The time axis `t` is a plain numpy array (no gradients needed for
    # coordinates).  The envelope is also numpy.  The final multiplication
    # `fid * envelope` is handled by the framework's own __mul__, which
    # auto-promotes the numpy envelope to the correct backend and keeps
    # the fid tensor on its original device / graph.

    @staticmethod
    def _make_time_envelope(fid, sw_hz, lb_hz, gb_hz):
        """Build the broadening envelope (numpy) shaped for broadcasting."""
        N = fid.shape[-1]
        t = np.arange(N, dtype=np.float64) / float(sw_hz)

        # Reshape t for broadcasting: (1, 1, ..., N)
        t = t.reshape([1] * (len(fid.shape) - 1) + [N])

        env = np.ones_like(t)
        if lb_hz > 0:
            env *= np.exp(-math.pi * lb_hz * t)
        if gb_hz > 0:
            env *= np.exp(-((math.pi * gb_hz * t) ** 2) / (4 * math.log(2)))
        return env

    @staticmethod
    def _apply_lorentzian(fid, sw_hz, lb_hz):
        """Apply Lorentzian (exponential) line broadening."""
        if lb_hz <= 0:
            return fid
        envelope = LineBroadening._make_time_envelope(fid, sw_hz, lb_hz, 0.0)
        return fid * match_backend(envelope, fid)

    @staticmethod
    def _apply_gaussian(fid, sw_hz, gb_hz):
        """Apply Gaussian line broadening."""
        if gb_hz <= 0:
            return fid
        envelope = LineBroadening._make_time_envelope(fid, sw_hz, 0.0, gb_hz)
        return fid * match_backend(envelope, fid)

    @staticmethod
    def _apply_voigt(fid, sw_hz, lb_hz, gb_hz):
        """Apply Voigt (Lorentzian × Gaussian) line broadening."""
        if lb_hz <= 0 and gb_hz <= 0:
            return fid
        envelope = LineBroadening._make_time_envelope(fid, sw_hz, lb_hz, gb_hz)
        return fid * match_backend(envelope, fid)
