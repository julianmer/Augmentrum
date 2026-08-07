####################################################################################################
#                                     residual_water.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (jlamaste@gmail.com)                                                     #
#          K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Simulates imperfect water suppression by adding Lorentzian water lobes around           #
#          4.7 ppm in the frequency domain.                                                        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List
from augmentrum.core.base_module import BaseModule
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from nifti_mrs_plus import ops
from nifti_mrs_plus.ops import fft, ifft, fftshift, ifftshift, match_backend


#**************************************************************************************************#
#                                       Class ResidualWater                                        #
#**************************************************************************************************#
#                                                                                                  #
# Add residual water peaks to MRS data.                                                            #
#                                                                                                  #
#**************************************************************************************************#
class ResidualWater(BaseModule):
    """
    Add residual water peaks to MRS data.

    Adds Lorentzian-shaped water peaks around 4.7 ppm to simulate imperfect
    water suppression.

    Parameters
    ----------
    center_ppm : float
        Center position of water peak in ppm (default: 4.7)
    peaks : tuple of tuples
        Each tuple is (delta_ppm, FWHM_ppm, rel_amp)
        - delta_ppm: offset from center in ppm
        - FWHM_ppm: Full Width at Half Maximum in ppm
        - rel_amp: relative amplitude
        Default: Three lobes at 0.0, +0.12, -0.15 ppm
    phase_deg : float
        Phase of water peaks in degrees (default: 0.0)
    amplitude_scale : float
        Scale factor for water amplitude relative to spectrum (default: 0.1 = 10%)

    Examples
    --------
    >>> # Default water (10% amplitude, 3 lobes)
    >>> water = ResidualWater()
    >>> result_data, _ = water(nifti_plus, None)

    >>> # Subtle water (5%)
    >>> water = ResidualWater(amplitude_scale=0.05)

    >>> # Strong water (30%)
    >>> water = ResidualWater(amplitude_scale=0.30)

    >>> # Custom peaks
    >>> water = ResidualWater(
    ...     peaks=((0.0, 0.25, 1.0), (0.10, 0.20, 0.5)),
    ...     phase_deg=15.0
    ... )
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    def __init__(self, center_ppm: float = 4.7,
                 peaks: tuple = ((0.0, 0.20, 1.0),
                                (0.12, 0.18, 0.4),
                                (-0.15, 0.25, 0.3)),
                 phase_deg: float = 0.0,
                 amplitude_scale: float = 0.1):
        """Initialize residual water module."""
        super().__init__()

        self.center_ppm = center_ppm
        self.peaks = peaks
        self.phase_deg = phase_deg
        self.amplitude_scale = amplitude_scale

    #*****************#
    #   water lobes   #
    #*****************#

    @staticmethod
    def _water_lobe_profile(ppm_axis, *, center_ppm=4.7,
                            peaks=((0.0, 0.20, 1.0),   # (delta_ppm, FWHM_ppm, rel_amp)
                                   (+0.12, 0.18, 0.4),
                                   (-0.15, 0.25, 0.3)),
                            phase_deg=0.0):
        """
        The complex water lobe profile at unit amplitude.

        Depends only on the ppm axis, so it is identical for every FID in a
        batch and is built once. Multiply by a per-FID amplitude to get the
        additive water contribution.

        Args:
            ppm_axis: PPM axis
            center_ppm: Center position of water peak (default: 4.7 ppm)
            peaks: Tuple of (delta_ppm, FWHM_ppm, rel_amp) for each lobe
            phase_deg: Phase of water peaks in degrees

        Returns:
            Complex profile with the same length as ppm_axis, peak magnitude 1
        """
        ppm = np.asarray(ppm_axis, float)
        phi = np.deg2rad(phase_deg)
        w = np.zeros_like(ppm, float)

        # Simple Lorentzians
        for dppm, fwhm_ppm, rel_amp in peaks:
            x = ppm - (center_ppm + dppm)
            hw = 0.5 * fwhm_ppm
            w += rel_amp * (hw / (x**2 + hw**2)) / np.pi

        # Normalize lobes to ~unit max
        w /= np.max(w) if np.max(w) > 0 else 1.0
        return w * (np.cos(phi) + 1j * np.sin(phi))

    @staticmethod
    def _add_water_lobes(spec, ppm_axis, *, center_ppm=4.7,
                         peaks=((0.0, 0.20, 1.0),   # (delta_ppm, FWHM_ppm, rel_amp)
                                (+0.12, 0.18, 0.4),
                                (-0.15, 0.25, 0.3)),
                         phase_deg=0.0, amplitude_scale=0.1):
        """
        Add Lorentzian residual water lobes to a complex spectrum.

        Args:
            spec: Complex spectrum
            ppm_axis: PPM axis
            center_ppm: Center position of water peak (default: 4.7 ppm)
            peaks: Tuple of (delta_ppm, FWHM_ppm, rel_amp) for each lobe
            phase_deg: Phase of water peaks in degrees
            amplitude_scale: Scale factor relative to spectrum max (default: 0.1 = 10%)

        Returns:
            Spectrum with water peaks added
        """
        s = np.asarray(spec, dtype=complex)
        profile = ResidualWater._water_lobe_profile(
            ppm_axis, center_ppm=center_ppm, peaks=peaks, phase_deg=phase_deg)
        water_amp = amplitude_scale * np.max(np.abs(np.real(s)))

        return s + water_amp * profile

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add residual water to list of NIFTI_MRS objects.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects (unchanged)
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, water_list)
        """
        processed_data = []

        for nifti in data_list:
            # Get FID data
            fid = nifti[:]

            # Get parameters
            sw_hz = 1.0 / nifti.dwelltime
            sf_mhz = nifti.spectrometer_frequency[0]

            # Add water to FID
            fid_with_water = self._add_water_to_fid(fid, sw_hz, sf_mhz)

            # Update NIFTI_MRS data
            nifti[:] = fid_with_water
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add residual water peaks to tensor/array data (**any backend**).

        FFT/IFFT are done **natively** via "tensor_ops".  Water peak shapes
        and amplitude scaling are computed in NumPy per-FID, then promoted
        to the target backend via "match_backend".

        Args:
            data_array: Input tensor of shape "(batch, ..., n_points)"
            water_array: Optional water reference (unchanged)
            backend: Backend enum (unused)
            **kwargs: Must contain "'sw_hz'" and "'sf_mhz'"

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        sf_mhz = kwargs.get('sf_mhz')
        if sw_hz is None or sf_mhz is None:
            raise ValueError("ResidualWater.process_tensor requires 'sw_hz' and 'sf_mhz' in kwargs")

        N = data_array.shape[-1]
        ref_ppm = 4.7

        # 1. Spectrum (backend-native FFT)
        spec = fftshift(ifft(data_array))

        # 2. ppm axis (numpy)
        dt = 1.0 / float(sw_hz)
        freq_hz = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
        ppm = ref_ppm - freq_hz / float(sf_mhz)

        # 3. The lobe profile depends only on the ppm axis, so it is the same
        #    for every FID and is built once, in NumPy, at unit amplitude.
        unit_lobes = self._water_lobe_profile(
            ppm,
            center_ppm=self.center_ppm,
            peaks=self.peaks,
            phase_deg=self.phase_deg,
        )

        # 4. The only data-dependent term is one amplitude per FID, taken on the
        #    data's own backend so the spectrum is never converted.
        water_amp = self.amplitude_scale * ops.amax(
            ops.abs(ops.real(spec)), axis=-1, keepdims=True)

        water_add = ops.cast_like(match_backend(unit_lobes, spec), spec) \
            * ops.cast_like(water_amp, spec)

        # 5. Add water in spectral domain (backend-native)
        spec_aug = spec + water_add

        # 6. Back to FID (backend-native IFFT)
        return fft(ifftshift(spec_aug)), water_array

    def _add_water_to_fid(self, fid: np.ndarray, sw_hz: float, sf_mhz: float,
                          ref_ppm: float = 4.7) -> np.ndarray:
        """
        Add water peaks to FID data.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz
            sf_mhz: Spectrometer frequency in MHz
            ref_ppm: Reference ppm (default: 4.7)

        Returns:
            FID with water peaks added
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

            # 1) IFFT to frequency domain (FID -> spectrum) - MRS convention
            spec = np.fft.fftshift(np.fft.ifft(fid_1d))

            # 2) Create ppm axis
            dt = 1.0 / float(sw_hz)
            freq_hz = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
            ppm = ref_ppm - freq_hz / sf_mhz

            # 3) Add water peaks in frequency domain
            spec_with_water = self._add_water_lobes(
                spec, ppm,
                center_ppm=self.center_ppm,
                peaks=self.peaks,
                phase_deg=self.phase_deg,
                amplitude_scale=self.amplitude_scale
            )

            # 4) FFT back to time domain (spectrum -> FID) - MRS convention
            fid_with_water = np.fft.fft(np.fft.ifftshift(spec_with_water))
            result[i] = fid_with_water

        # Reshape back to original shape
        return result.reshape(original_shape)
