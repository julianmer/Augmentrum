####################################################################################################
#                                    phase_frequency.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Implements zero-order phase, first-order phase, and frequency shift augmentations.      #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math
import numpy as np
from typing import Optional, List

from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from nifti_mrs_plus import Backend
from nifti_mrs_plus.ops import fft, ifft, fftshift, ifftshift, match_backend


#**************************************************************************************************#
#                                         Class PhaseShift                                         #
#**************************************************************************************************#
#                                                                                                  #
# Apply phase shifts to MRS data.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class PhaseShift(BaseModule):
    """
    Apply phase shifts to MRS data.

    Supports:
    - Zero-order phase: Constant phase shift across entire spectrum
    - First-order phase: Linear phase ramp across spectrum

    Parameters
    ----------
    zero_order_deg : float
        Zero-order phase shift in degrees (default: 0.0)
    first_order_deg : float
        First-order phase shift in degrees (default: 0.0)
        Applied as linear ramp from left to right edge

    Examples
    --------
    >>> # Zero-order phase only
    >>> phase = PhaseShift(zero_order_deg=60.0)
    >>>
    >>> # First-order phase only
    >>> phase = PhaseShift(first_order_deg=90.0)
    >>>
    >>> # Both
    >>> phase = PhaseShift(zero_order_deg=30.0, first_order_deg=45.0)
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    def __init__(self, zero_order_deg: float = 0.0, first_order_deg: float = 0.0):
        """Initialize phase shift module."""
        super().__init__()

        self.zero_order_deg = zero_order_deg
        self.first_order_deg = first_order_deg

    @property
    def DOMAIN(self):
        """
        A first-order shift is a ramp across the spectrum, so it needs one; a
        zero-order shift is a constant factor and works anywhere, so asking for
        a domain it does not need would force a transform for nothing.

        A property rather than a value set at construction, because the pipeline
        samples "first_order_deg" ranges per batch and injects them onto the
        instance — the domain has to follow the value that will actually run,
        not the constructor default.
        """
        if np.any(np.asarray(self.first_order_deg) != 0.0):
            return Domain(spectral='frequency')
        return None

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Apply phase shift to list of NIFTI_MRS objects.

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

            # Get spectral width (for first-order phase)
            sw_hz = 1.0 / nifti.dwelltime

            # Apply phase shifts
            fid_phased = self._apply_phase(fid, sw_hz)

            # Update NIFTI_MRS data
            nifti[:] = fid_phased
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Apply phase shift to tensor/array data (**any backend**).

        Uses "keras.ops" — works with NumPy, PyTorch, JAX, or TensorFlow.
        Gradients are preserved.

        Args:
            data_array: Input tensor of shape "(batch, ..., n_points)"
            water_array: Optional water reference tensor (unchanged)
            backend: Backend enum (unused — ops dispatch automatically)
            **kwargs: Must contain "'sw_hz'" (spectral width in Hz)

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz', 1.0)
        result = self._apply_phase(data_array, sw_hz)
        return result, water_array

    def _apply_phase(self, fid, sw_hz: float):
        """
        Apply phase shifts to FID data (any backend tensor).

        Zero-order phase is a scalar complex multiplication (fully vectorized).
        First-order phase requires FFT → ramp → IFFT (uses tensor_ops).
        """
        # Zero-order: fully vectorized scalar multiply
        if self.zero_order_deg != 0.0:
            phi = math.radians(self.zero_order_deg)
            # complex scalar * any-backend tensor — works everywhere
            phase_factor = np.exp(-1j * phi)
            fid = fid * match_backend(np.array(phase_factor), fid)

        # First-order: needs spectral domain
        if self.first_order_deg != 0.0:
            fid = self._first_order_phase(fid, self.first_order_deg)

        return fid

    @staticmethod
    def _zero_order_phase(fid, phase_deg: float):
        """Apply zero-order phase shift (any backend tensor)."""
        phi_rad = math.radians(phase_deg)
        factor = np.array(np.exp(-1j * phi_rad))
        return fid * match_backend(factor, fid)

    @staticmethod
    def _first_order_phase(fid, phc1_deg: float):
        """
        Apply first-order phase shift (any backend tensor).

        Uses backend-agnostic FFT from tensor_ops.
        The linear phase ramp is a numpy array; multiplication with the
        spectrum tensor auto-promotes to the correct backend.
        """
        # The data is already a spectrum: a first-order shift declares the
        # frequency domain, so the module is put there before it runs.
        spec = fid
        N = fid.shape[-1]

        # Linear ramp (numpy — no gradients needed for coordinates)
        u = np.linspace(0.0, 1.0, N, dtype=np.float64)
        ramp_shape = [1] * (len(fid.shape) - 1) + [N]
        ramp = np.exp(1j * np.deg2rad(phc1_deg * u)).reshape(ramp_shape)

        # Apply the ramp; the caller puts the data back where it was
        return spec * match_backend(ramp, spec)


#**************************************************************************************************#
#                                       Class FrequencyShift                                       #
#**************************************************************************************************#
#                                                                                                  #
# Apply frequency shift to MRS data.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class FrequencyShift(BaseModule):
    """
    Apply frequency shift to MRS data.

    Shifts the entire spectrum by a specified frequency offset in Hz.
    Typical range: [-40, 40] Hz, safe mode: [-20, 20] Hz.

    Parameters
    ----------
    shift_hz : float
        Frequency shift in Hz (positive = upfield shift)

    Examples
    --------
    >>> # Shift by +10 Hz
    >>> freq_shift = FrequencyShift(shift_hz=10.0)
    >>>
    >>> # Shift by -20 Hz
    >>> freq_shift = FrequencyShift(shift_hz=-20.0)
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # A frequency shift is applied as a phase that winds along the FID.
    DOMAIN = Domain(spectral='time')

    def __init__(self, shift_hz: float = 0.0):
        """Initialize frequency shift module."""
        super().__init__()

        self.shift_hz = shift_hz

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Apply frequency shift to list of NIFTI_MRS objects.

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

            # Apply frequency shift
            fid_shifted = self._apply_shift(fid, sw_hz)

            # Update NIFTI_MRS data
            nifti[:] = fid_shifted
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Apply frequency shift to tensor/array data (**any backend**).

        Uses "keras.ops" — works with NumPy, PyTorch, JAX, or TensorFlow.

        Args:
            data_array: Input tensor of shape "(batch, ..., n_points)"
            water_array: Optional water reference tensor (unchanged)
            backend: Backend enum (unused — ops dispatch automatically)
            **kwargs: Must contain "'sw_hz'" (spectral width in Hz)

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        if sw_hz is None:
            raise ValueError("FrequencyShift.process_tensor requires 'sw_hz' in kwargs")

        result = self._apply_shift(data_array, sw_hz)
        return result, water_array

    def _apply_shift(self, fid, sw_hz: float):
        """
        Apply frequency shift to FID data (any backend tensor).

        The shift phasor is a numpy array; multiplication with the FID tensor
        auto-promotes to the correct backend.
        """
        if self.shift_hz == 0.0:
            return fid

        N = fid.shape[-1]

        # Time axis (numpy — no gradients needed for coordinates)
        t = np.arange(N, dtype=np.float64) / float(sw_hz)
        t = t.reshape([1] * (len(fid.shape) - 1) + [N])

        # Shift phasor (numpy complex)
        shift_factor = np.exp(1j * 2.0 * math.pi * float(self.shift_hz) * t)

        # Multiply: convert phasor to same backend, preserves gradients
        return fid * match_backend(shift_factor, fid)
