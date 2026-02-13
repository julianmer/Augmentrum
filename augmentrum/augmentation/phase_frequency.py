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

import numpy as np
from typing import Optional, List
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus


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

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self, zero_order_deg: float = 0.0, first_order_deg: float = 0.0):
        """Initialize phase shift module."""
        super().__init__(zero_order_deg=zero_order_deg, first_order_deg=first_order_deg)

        self.zero_order_deg = zero_order_deg
        self.first_order_deg = first_order_deg

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

    def _apply_phase(self, fid: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Apply phase shifts to FID data.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz

        Returns:
            Phase-shifted FID
        """
        # Work with original shape
        original_shape = fid.shape
        fid_flat = fid.reshape(-1, fid.shape[-1])
        result = np.zeros_like(fid_flat)

        # Process each FID
        for i in range(fid_flat.shape[0]):
            fid_1d = fid_flat[i]

            # Apply zero-order phase
            if self.zero_order_deg != 0.0:
                fid_1d = self._zero_order_phase(fid_1d, self.zero_order_deg)

            # Apply first-order phase
            if self.first_order_deg != 0.0:
                fid_1d = self._first_order_phase(fid_1d, self.first_order_deg)

            result[i] = fid_1d

        # Reshape back to original shape
        return result.reshape(original_shape)

    @staticmethod
    def _zero_order_phase(fid: np.ndarray, phase_deg: float) -> np.ndarray:
        """
        Apply zero-order phase shift.

        Args:
            fid: Input FID
            phase_deg: Phase in degrees

        Returns:
            Phase-shifted FID
        """
        phi_rad = np.deg2rad(phase_deg)
        rotation_factor = np.exp(-1j * phi_rad)
        return fid * rotation_factor

    @staticmethod
    def _first_order_phase(fid: np.ndarray, phc1_deg: float) -> np.ndarray:
        """
        Apply first-order phase shift.

        Applies linear phase ramp across spectrum.
        Pivot is at left edge.

        Args:
            fid: Input FID
            phc1_deg: First-order phase in degrees

        Returns:
            Phase-shifted FID
        """
        # FFT to spectrum
        spec = np.fft.fftshift(np.fft.ifft(fid))
        N = spec.size

        # Linear ramp from 0 to 1 (left-edge pivot)
        u = np.linspace(0.0, 1.0, N)
        phi_deg = phc1_deg * u
        spec_ph = spec * np.exp(1j * np.deg2rad(phi_deg))

        # Back to FID
        return np.fft.fft(np.fft.ifftshift(spec_ph))


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

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self, shift_hz: float = 0.0):
        """Initialize frequency shift module."""
        super().__init__(shift_hz=shift_hz)

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

    def _apply_shift(self, fid: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Apply frequency shift to FID data.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz

        Returns:
            Frequency-shifted FID
        """
        if self.shift_hz == 0.0:
            return fid

        # Work with original shape
        original_shape = fid.shape
        N = original_shape[-1]

        # Time axis for last dimension
        t = np.arange(N, dtype=float) / float(sw_hz)

        # Reshape t to broadcast correctly
        t_shape = tuple([1] * (fid.ndim - 1) + [N])
        t = t.reshape(t_shape)

        # Apply frequency shift: multiply by exp(+i*2π*delta_f*t)
        shift_factor = np.exp(1j * 2.0 * np.pi * float(self.shift_hz) * t)

        return fid * shift_factor
