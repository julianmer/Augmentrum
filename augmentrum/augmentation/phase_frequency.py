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
from augmentrum.processing.utils import (device_axis, device_kernels, device_values, on_cuda,
                                         to_backend)
from nifti_mrs_plus import Backend
from nifti_mrs_plus.ops import fft, ifft, fftshift, ifftshift


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
        First-order phase shift in degrees (default: 0.0): a linear ramp
        across the spectral width, from -first_order_deg/2 at its lower edge
        to +first_order_deg/2 at its upper one, through zero at the centre
        (the reference frequency), as FSL-MRS' first-order phase pivots

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

    # Acts on every coil and transient alike, so per-sample masks pass through.
    MASKS = 'pass'

    # A constant rotation and a ramp both broadcast, so a batch can carry one
    # of each per sample; any non-zero ramp puts the batch in its spectrum.
    PER_SAMPLE_PARAMS = ('zero_order_deg', 'first_order_deg')

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
            i = len(processed_data)
            fid_phased = self._apply_phase(fid, sw_hz, self.sample_of(self.zero_order_deg, i),
                                           self.sample_of(self.first_order_deg, i))

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
        result = self._apply_phase(data_array, sw_hz, self.zero_order_deg, self.first_order_deg)
        return result, water_array

    def _apply_phase(self, fid, sw_hz: float, zero_order_deg, first_order_deg):
        """
        Apply phase shifts to FID data (any backend tensor).

        Zero-order phase is a constant complex multiplication (fully
        vectorized); a "(batch,)" *zero_order_deg* rotates each sample by its
        own phase, and a "(batch,)" *first_order_deg* gives each its own ramp.
        First-order phase requires FFT → ramp → IFFT (tensor_ops).
        """
        zero = np.any(np.asarray(zero_order_deg) != 0.0)
        first = np.any(np.asarray(first_order_deg) != 0.0)
        kernels = device_kernels(fid) if len(fid.shape) > 1 and (zero or first) else None
        if kernels is not None:
            # both shifts in one launch, each rounded as below
            params, u, consts = self._kernel_inputs(fid, zero_order_deg, first_order_deg)
            return kernels.phase(fid, params, u, consts, zero, first)

        # Zero-order: fully vectorized constant multiply
        if zero:
            phi = np.deg2rad(self.per_sample(np.asarray(zero_order_deg, dtype=np.float64),
                                             len(fid.shape)))
            # complex factor * any-backend tensor — works everywhere
            phase_factor = np.exp(-1j * phi)
            fid = fid * to_backend(np.asarray(phase_factor), fid)

        # First-order: needs spectral domain
        if first:
            fid = self._first_order_phase(fid, first_order_deg)

        return fid

    def _kernel_inputs(self, fid, zero_order_deg, first_order_deg):
        """
        The inputs of "augmentrum.core.kernels.phase" on *fid*'s device, as "_apply_phase" and
        "_first_order_phase" form them: per sample (3, B) float64 in one upload (the first-order
        phase in degrees, the zero-order factor's real and imaginary parts), the centred ramp,
        and pi / 180.
        """
        batch, n = fid.shape[0], fid.shape[-1]
        phi = np.deg2rad(np.broadcast_to(np.asarray(zero_order_deg, dtype=np.float64), (batch,)))
        factor = np.exp(-1j * phi)
        params = np.stack([np.broadcast_to(np.asarray(first_order_deg, dtype=np.float64),
                                           (batch,)), factor.real, factor.imag])
        u = device_axis(('centred_ramp', n), lambda: self._centred_ramp(n), fid)
        degree = device_axis(('degree',), lambda: [np.pi / 180.0], fid)
        return device_values(params, fid), u, degree

    @staticmethod
    def _zero_order_phase(fid, phase_deg: float):
        """Apply zero-order phase shift (any backend tensor)."""
        phi_rad = math.radians(phase_deg)
        factor = np.array(np.exp(-1j * phi_rad))
        return fid * to_backend(factor, fid)

    @staticmethod
    def _centred_ramp(n):
        """The ramp's unit axis on the fftshifted spectrum: -1/2 at the first bin, 0 at the centre."""
        return (np.arange(n, dtype=np.float64) - n // 2) / n

    @classmethod
    def _first_order_phase(cls, fid, phc1_deg):
        """
        Apply first-order phase shift (any backend tensor).

        Uses backend-agnostic FFT from tensor_ops.
        The linear phase ramp is a numpy array; multiplication with the
        spectrum tensor auto-promotes to the correct backend. A "(batch,)"
        *phc1_deg* gives every sample its own ramp.
        """
        # The data is already a spectrum: a first-order shift declares the
        # frequency domain, so the module is put there before it runs.
        spec = fid
        N = fid.shape[-1]
        ndim = len(fid.shape)
        ramp_shape = [1] * (ndim - 1) + [N]

        if on_cuda(spec):
            # the same float64 ramp (deg2rad is a multiply by pi / 180), on the device
            import torch
            u = device_axis(('centred_ramp', N), lambda: cls._centred_ramp(N),
                            spec).reshape(ramp_shape)
            phc1 = np.asarray(phc1_deg, dtype=np.float64)
            phc1 = (float(phc1) if phc1.ndim == 0
                    else device_values(phc1, spec).reshape((-1,) + (1,) * (ndim - 1)))
            angle = (phc1 * u) * (np.pi / 180.0)
            ramp = torch.polar(torch.ones_like(angle), angle)
            return spec * ramp.to(spec.dtype)

        # Linear ramp (numpy — no gradients needed for coordinates)
        u = cls._centred_ramp(N).reshape(ramp_shape)
        ramp = np.exp(1j * np.deg2rad(BaseModule.per_sample(phc1_deg, ndim) * u))

        # Apply the ramp; the caller puts the data back where it was
        return spec * to_backend(ramp, spec)


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

    # Acts on every coil and transient alike, so per-sample masks pass through.
    MASKS = 'pass'

    # A frequency shift is applied as a phase that winds along the FID.
    DOMAIN = Domain(spectral='time')

    # The phasor broadcasts, so a batch can carry one shift per sample.
    PER_SAMPLE_PARAMS = ('shift_hz',)

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
            fid_shifted = self._apply_shift(fid, sw_hz,
                                            self.sample_of(self.shift_hz, len(processed_data)))

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

        result = self._apply_shift(data_array, sw_hz, self.shift_hz)
        return result, water_array

    def _apply_shift(self, fid, sw_hz: float, shift_hz):
        """
        Apply frequency shift to FID data (any backend tensor).

        The shift phasor is a numpy array; multiplication with the FID tensor
        auto-promotes to the correct backend. A "(batch,)" *shift_hz* gives
        each sample its own shift.
        """
        if not np.any(np.asarray(shift_hz) != 0.0):
            return fid

        N = fid.shape[-1]
        shape = [1] * (len(fid.shape) - 1) + [N]
        shift = self.per_sample(np.asarray(shift_hz, dtype=np.float64), len(fid.shape))

        kernels = device_kernels(fid) if len(fid.shape) > 1 else None
        if kernels is not None:
            # the same phase, (2 pi shift) t, its phasor and the product in one launch
            t = device_axis(('time', N, float(sw_hz)),
                            lambda: np.arange(N, dtype=np.float64) / float(sw_hz), fid)
            turn = 2.0 * math.pi * np.broadcast_to(np.asarray(shift_hz, dtype=np.float64),
                                                   (fid.shape[0],))
            unit = device_axis(('unit',), lambda: [1.0], fid)
            return kernels.phase(fid, device_values(turn[None], fid), t, unit, False, True)

        if on_cuda(fid):
            # the same float64 phase, formed in the same order, on the device
            import torch
            t = device_axis(('time', N, float(sw_hz)),
                            lambda: np.arange(N, dtype=np.float64) / float(sw_hz), fid)
            phase = (2.0 * math.pi * device_values(shift, fid)) * t.reshape(shape)
            return fid * torch.polar(torch.ones_like(phase), phase).to(fid.dtype)

        # Time axis (numpy — no gradients needed for coordinates)
        t = np.arange(N, dtype=np.float64) / float(sw_hz)
        t = t.reshape(shape)

        # Shift phasor (numpy complex), one row per sample for a vector shift
        shift_factor = np.exp(1j * 2.0 * math.pi * shift * t)

        # Multiply: convert phasor to same backend, preserves gradients
        return fid * to_backend(shift_factor, fid)
