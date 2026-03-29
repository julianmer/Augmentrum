####################################################################################################
#                                   artificial_peaks.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Adds artificial contaminant peaks at specified frequencies in MRS data.                 #
#                                                                                                  #
####################################################################################################

import numpy as np
from typing import Optional, List, Dict
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from augmentrum.utils.tensor_ops import fft, ifft, fftshift, ifftshift, to_numpy, match_backend


class ArtificialPeaks(BaseModule):
    """
    Add artificial contaminant peaks to MRS data.

    Adds Lorentzian, Gaussian, or Voigt peaks at specified ppm positions
    in the frequency domain to simulate contamination or additional metabolites.

    Parameters
    ----------
    peaks : list of dicts
        Each dict contains:
        - ppm: Peak position in ppm
        - amp: Amplitude as fraction of main spectrum peak (0.0-1.0)
        - phase_deg: Complex phase in degrees
        - lb_hz: Lorentzian FWHM in Hz (0 for none)
        - gb_hz: Gaussian FWHM in Hz (0 for none)
        If both lb_hz and gb_hz > 0, creates Voigt peak
    ref_ppm : float
        Reference ppm for axis calculation (default: 4.7)
    amp_mode : str
        How to calculate reference amplitude: 'real' or 'abs' (default: 'real')

    Examples
    --------
    >>> # Add single Lorentzian peak at 3.0 ppm
    >>> peaks = ArtificialPeaks(peaks=[
    ...     {'ppm': 3.0, 'amp': 0.1, 'phase_deg': 0.0, 'lb_hz': 5.0, 'gb_hz': 0.0}
    ... ])
    >>> result_data, _ = peaks(nifti_plus, None)
    """

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self, peaks: List[Dict] = None, ref_ppm: float = 4.7, amp_mode: str = 'real'):
        """Initialize artificial peaks module."""
        if peaks is None:
            peaks = [{'ppm': 3.0, 'amp': 0.05, 'phase_deg': 0.0, 'lb_hz': 5.0, 'gb_hz': 0.0}]

        super().__init__(peaks=peaks, ref_ppm=ref_ppm, amp_mode=amp_mode)

        self.peaks = peaks
        self.ref_ppm = ref_ppm
        self.amp_mode = amp_mode

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add artificial peaks to list of NIFTI_MRS objects.

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

            # Get parameters
            sw_hz = 1.0 / nifti.dwelltime
            sf_mhz = nifti.spectrometer_frequency[0]

            # Add peaks
            fid_with_peaks = self._add_peaks(fid, sw_hz, sf_mhz)

            # Update NIFTI_MRS data
            nifti[:] = fid_with_peaks
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add artificial peaks to tensor/array data (**any backend**).

        FFT/IFFT are done **natively** via ``tensor_ops`` (preserves gradients).
        Peak shapes and amplitude references are computed in NumPy then
        promoted to the target backend with ``match_backend``.

        Args:
            data_array: Input tensor of shape ``(batch, ..., n_points)``
            water_array: Optional water reference (unchanged)
            backend: Backend enum (unused)
            **kwargs: Must contain ``'sw_hz'`` and ``'sf_mhz'``

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        sf_mhz = kwargs.get('sf_mhz')
        if sw_hz is None or sf_mhz is None:
            raise ValueError("ArtificialPeaks.process_tensor requires 'sw_hz' and 'sf_mhz' in kwargs")

        N = data_array.shape[-1]
        n_batch = int(np.prod(data_array.shape[:-1]))

        # 1. Spectrum (backend-native FFT — preserves gradients)
        spec = fftshift(ifft(data_array))  # shape = original_shape

        # 2. ppm axis (numpy — coordinate, no gradients needed)
        dt = 1.0 / float(sw_hz)
        freq_hz = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
        ppm = self.ref_ppm - freq_hz / float(sf_mhz)

        # 3. Amplitude reference and peak shapes (numpy, per-FID scalar pull)
        spec_np = to_numpy(spec).reshape(n_batch, N)
        contam_np = np.zeros((n_batch, N), dtype=np.complex128)

        for i in range(n_batch):
            if self.amp_mode == 'abs':
                peak_ref = float(np.max(np.abs(spec_np[i]))) or 1.0
            else:
                peak_ref = float(np.max(np.abs(spec_np[i].real))) or 1.0

            contam = np.zeros(N, dtype=np.complex128)
            for p in self.peaks:
                ppm0 = float(p['ppm'])
                amp_frac = float(p.get('amp', p.get('amplitude', 0.05)))
                phase_deg = float(p.get('phase_deg', 0.0))
                lb_hz = float(p.get('lb_hz', 0.0))
                gb_hz = float(p.get('gb_hz', 0.0))

                if lb_hz > 0 and gb_hz > 0:
                    shape = self._voigt(ppm, ppm0, lb_hz, gb_hz, sf_mhz)
                elif lb_hz > 0:
                    shape = self._lorentzian(ppm, ppm0, lb_hz, sf_mhz)
                elif gb_hz > 0:
                    shape = self._gaussian(ppm, ppm0, gb_hz, sf_mhz)
                else:
                    continue

                contam += (amp_frac * peak_ref) * shape * np.exp(1j * np.deg2rad(phase_deg))

            contam_np[i] = contam

        contam_np = contam_np.reshape(data_array.shape)

        # 4. Add contamination in spectral domain (backend-native)
        spec_aug = spec + match_backend(contam_np, spec)

        # 5. Back to FID (backend-native IFFT)
        return fft(ifftshift(spec_aug)), water_array

    def _add_peaks(self, fid: np.ndarray, sw_hz: float, sf_mhz: float) -> np.ndarray:
        """
        Add artificial peaks to FID data.

        Args:
            fid: Input FID data (shape: x, y, z, points, [coils], [averages], ...)
            sw_hz: Spectral width in Hz
            sf_mhz: Spectrometer frequency in MHz

        Returns:
            FID with peaks added
        """
        # In MRS data, dimension 3 (index 3) is typically the FID points
        # Shape is typically: (x, y, z, points, coils, averages, ...)
        original_shape = fid.shape

        # Find the FID points dimension (should be dim 3, index 3)
        # For standard MRS: shape is (1, 1, 1, points, coils, averages)
        if len(original_shape) >= 4:
            # FID points are at index 3 (4th dimension)
            n_points = original_shape[3]

            # Move FID dimension to the end for easier processing
            # From: (x, y, z, points, coils, avg) -> (x, y, z, coils, avg, points)
            fid_moved = np.moveaxis(fid, 3, -1)
            moved_shape = fid_moved.shape

            # Reshape to 2D: (all_other_dims, points)
            fid_2d = fid_moved.reshape(-1, n_points)
            result = np.zeros_like(fid_2d)

            # Process each FID
            for i in range(fid_2d.shape[0]):
                fid_1d = fid_2d[i]
                fid_with_peaks = self._add_peaks_1d(fid_1d, sw_hz, sf_mhz)
                result[i] = fid_with_peaks

            # Reshape back and move FID dimension back to position 3
            result_moved = result.reshape(moved_shape)
            result_final = np.moveaxis(result_moved, -1, 3)

            return result_final
        else:
            # Simple 1D case or legacy format - last dimension is FID points
            N = original_shape[-1]
            fid_2d = fid.reshape(-1, N)
            result = np.zeros_like(fid_2d)

            for i in range(fid_2d.shape[0]):
                fid_1d = fid_2d[i]
                fid_with_peaks = self._add_peaks_1d(fid_1d, sw_hz, sf_mhz)
                result[i] = fid_with_peaks

            return result.reshape(original_shape)

    def _add_peaks_1d(self, fid: np.ndarray, sw_hz: float, sf_mhz: float) -> np.ndarray:
        """
        Add peaks to 1D FID.

        Args:
            fid: 1D FID data
            sw_hz: Spectral width in Hz
            sf_mhz: Spectrometer frequency in MHz

        Returns:
            FID with peaks
        """
        # IFFT to frequency domain (correct MRS convention)
        spec = np.fft.fftshift(np.fft.ifft(fid))

        # Create ppm axis
        n = fid.size
        dt = 1.0 / float(sw_hz)
        freq_hz = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
        # Legacy formula: ppm_offset = freq_hz_offset / larmor_freq_MHz
        ppm = self.ref_ppm - freq_hz / float(sf_mhz)

        # Calculate reference amplitude
        if self.amp_mode == 'abs':
            peak_ref = np.max(np.abs(spec)) or 1.0
        else:
            peak_ref = np.max(np.abs(spec.real)) or 1.0

        # Generate contamination
        contam = np.zeros_like(spec, dtype=np.complex128)

        for p in self.peaks:
            ppm0 = float(p['ppm'])
            # Support both 'amp' and 'amplitude' keys
            amp_frac = float(p.get('amp', p.get('amplitude', 0.05)))
            phase_deg = float(p.get('phase_deg', 0.0))
            lb_hz = float(p.get('lb_hz', 0.0))
            gb_hz = float(p.get('gb_hz', 0.0))

            # Generate peak shape
            if lb_hz > 0 and gb_hz > 0:
                shape = self._voigt(ppm, ppm0, lb_hz, gb_hz, sf_mhz)
            elif lb_hz > 0:
                shape = self._lorentzian(ppm, ppm0, lb_hz, sf_mhz)
            elif gb_hz > 0:
                shape = self._gaussian(ppm, ppm0, gb_hz, sf_mhz)
            else:
                continue  # No width, skip

            # Add to contamination
            phase = np.exp(1j * np.deg2rad(phase_deg))
            peak_contam = (amp_frac * peak_ref) * shape * phase
            contam += peak_contam

        # Add contamination to spectrum
        spec_aug = spec + contam

        # Back to time domain (FFT to invert IFFT)
        return np.fft.fft(np.fft.ifftshift(spec_aug))

    @staticmethod
    def _lorentzian(ppm: np.ndarray, ppm0: float, fwhm_hz: float, sf_mhz: float) -> np.ndarray:
        """Generate Lorentzian peak shape."""
        if fwhm_hz <= 0:
            return np.zeros_like(ppm)

        # Convert Hz to ppm: divide by sf_mhz (which is in MHz)
        # This gives: Hz / MHz = ppm (since 1 MHz = 1e6 Hz, the ratio is in ppm)
        fwhm_ppm = fwhm_hz / float(sf_mhz)
        x = ppm - ppm0
        hw = 0.5 * fwhm_ppm
        L = (hw / (x**2 + hw**2)) / np.pi
        return L / (L.max() if L.max() > 0 else 1.0)

    @staticmethod
    def _gaussian(ppm: np.ndarray, ppm0: float, fwhm_hz: float, sf_mhz: float) -> np.ndarray:
        """Generate Gaussian peak shape."""
        if fwhm_hz <= 0:
            return np.zeros_like(ppm)

        # Convert Hz to ppm: divide by sf_mhz (which is in MHz)
        fwhm_ppm = fwhm_hz / float(sf_mhz)
        x = ppm - ppm0
        G = np.exp(-4.0 * np.log(2.0) * (x**2) / (fwhm_ppm**2))
        return G

    @staticmethod
    def _voigt(ppm: np.ndarray, ppm0: float, lb_hz: float, gb_hz: float, sf_mhz: float) -> np.ndarray:
        """Generate Voigt peak shape (product of Lorentzian and Gaussian)."""
        if lb_hz <= 0 and gb_hz <= 0:
            return np.zeros_like(ppm)

        # Don't call the normalized functions - build them here
        if lb_hz > 0:
            fwhm_ppm_l = lb_hz / float(sf_mhz)
            x = ppm - ppm0
            hw = 0.5 * fwhm_ppm_l
            L = (hw / (x**2 + hw**2)) / np.pi
        else:
            L = 1.0

        if gb_hz > 0:
            fwhm_ppm_g = gb_hz / float(sf_mhz)
            x = ppm - ppm0
            G = np.exp(-4.0 * np.log(2.0) * (x**2) / (fwhm_ppm_g**2))
        else:
            G = 1.0

        V = L * G
        vmax = V.max() if isinstance(V, np.ndarray) else V
        return V / (vmax if vmax > 0 else 1.0)
