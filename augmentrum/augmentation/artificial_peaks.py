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

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List, Dict
from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from augmentrum.processing.utils import ppm_axis, ppm_reference, batch_profile
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from nifti_mrs_plus import ops
from nifti_mrs_plus.ops import match_backend


#**************************************************************************************************#
#                                      Class ArtificialPeaks                                       #
#**************************************************************************************************#
#                                                                                                  #
# Add artificial contaminant peaks to MRS data.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class ArtificialPeaks(BaseModule):
    """
    Add artificial contaminant peaks to MRS data.

    Adds Lorentzian, Gaussian, or Voigt peaks at specified ppm positions
    in the frequency domain to simulate contamination or additional metabolites.
    All ppm values are on the FSL-MRS / NIfTI-MRS axis (protons referenced to
    4.65 ppm), so a peak requested at 1.30 ppm shows at 1.30 on an FSL-MRS plot.

    Parameters
    ----------
    peaks : list of dicts
        Each dict contains:
        - ppm: Peak position in ppm
        - amp: Amplitude as fraction of main spectrum peak (0.0-1.0)
        - phase_deg: Complex phase in degrees
        - lb_hz: Lorentzian FWHM in Hz (0 for none)
        - gb_hz: Gaussian FWHM in Hz (0 for none)
        If both lb_hz and gb_hz > 0, creates Voigt peak.
        Any of these may be a "(low, high)" range instead of a number; ranges
        are drawn uniformly, once per sample, from this module's seeded
        generator, while numbers stay fixed for every sample. The default is
        one randomised lipid/contaminant peak: ppm in (0.9, 1.6), amp in
        (0.05, 0.3), lb_hz in (5, 20) - a fixed peak on top of a metabolite is
        no augmentation.
    ref_ppm : float, optional
        Reference ppm of the carrier (0 Hz). None (default) is the nucleus'
        reference from "ppm_reference" (4.65 ppm for 1H); set it only to place
        peaks on another convention.
    amp_mode : str
        How to calculate reference amplitude: 'real' or 'abs' (default: 'real')
    seed : int, optional
        Seed for the per-sample draws.

    Examples
    --------
    >>> # A random lipid-like peak per sample (the default)
    >>> peaks = ArtificialPeaks()
    >>> result_data, _ = peaks(nifti_plus, None)

    >>> # Add a fixed Lorentzian peak at 3.0 ppm
    >>> peaks = ArtificialPeaks(peaks=[
    ...     {'ppm': 3.0, 'amp': 0.1, 'phase_deg': 0.0, 'lb_hz': 5.0, 'gb_hz': 0.0}
    ... ])

    >>> # Ranges: position, width and phase drawn per sample
    >>> peaks = ArtificialPeaks(peaks=[
    ...     {'ppm': (0.8, 1.6), 'amp': (0.05, 0.3), 'lb_hz': (5, 20),
    ...      'phase_deg': (-30, 30)}
    ... ], seed=0)
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # A peak is a feature of a spectrum, so that is where it is added.
    DOMAIN = Domain(spectral='frequency')

    #: The keys of a peak dict that may be given as (low, high) ranges, with
    #: their defaults when absent.
    PEAK_KEYS = {'ppm': None, 'amp': 0.05, 'phase_deg': 0.0, 'lb_hz': 0.0, 'gb_hz': 0.0}

    #: The default: a lipid/contaminant peak drawn afresh for every sample.
    DEFAULT_PEAKS = ({'ppm': (0.9, 1.6), 'amp': (0.05, 0.3), 'phase_deg': 0.0,
                      'lb_hz': (5.0, 20.0), 'gb_hz': 0.0},)

    def __init__(self, peaks: List[Dict] = None, ref_ppm: Optional[float] = None,
                 amp_mode: str = 'real', seed: Optional[int] = None):
        """Initialize artificial peaks module."""
        if peaks is None:
            peaks = [dict(p) for p in self.DEFAULT_PEAKS]

        super().__init__()

        self.peaks = [self._normalize(p) for p in peaks]
        self.ref_ppm = ref_ppm
        self.amp_mode = amp_mode

    @classmethod
    def _normalize(cls, peak: Dict) -> Dict:
        """One peak dict with canonical keys ('amplitude' is an alias of 'amp')."""
        if 'ppm' not in peak:
            raise ValueError(f"A peak needs a 'ppm'; got {peak!r}")
        out = dict(peak)
        if 'amp' not in out and 'amplitude' in out:
            out['amp'] = out.pop('amplitude')
        for key, default in cls.PEAK_KEYS.items():
            if key not in out and default is not None:
                out[key] = default
        return out

    #****************#
    #   the draws   #
    #****************#
    def _draw(self, batch: int) -> List[Dict[str, np.ndarray]]:
        """
        This batch's peak parameters, one "(batch,)" vector per key per peak.

        Ranges are drawn from a single generator taken from the module's seed
        stream, so the same seed gives the same peaks on every backend and on
        both processing paths; fixed values are repeated.
        """
        rng = self.rng.numpy_rng()
        table = []
        for peak in self.peaks:
            drawn = {}
            for key in self.PEAK_KEYS:
                value = peak[key]
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    drawn[key] = rng.uniform(float(value[0]), float(value[1]), size=batch)
                else:
                    drawn[key] = np.full(batch, float(value))
            table.append(drawn)
        return table

    def _profile(self, index: int, ppm: np.ndarray, table: List[Dict], sf_mhz: float) -> np.ndarray:
        """Sample *index*'s contamination on *ppm*, in units of its amplitude reference."""
        contam = np.zeros(ppm.shape, dtype=np.complex128)
        for drawn in table:
            ppm0 = float(drawn['ppm'][index])
            amp_frac = float(drawn['amp'][index])
            phase_deg = float(drawn['phase_deg'][index])
            lb_hz = float(drawn['lb_hz'][index])
            gb_hz = float(drawn['gb_hz'][index])

            if lb_hz > 0 and gb_hz > 0:
                shape = self._voigt(ppm, ppm0, lb_hz, gb_hz, sf_mhz)
            elif lb_hz > 0:
                shape = self._lorentzian(ppm, ppm0, lb_hz, sf_mhz)
            elif gb_hz > 0:
                shape = self._gaussian(ppm, ppm0, gb_hz, sf_mhz)
            else:
                continue  # No width, skip

            contam += amp_frac * shape * np.exp(1j * np.deg2rad(phase_deg))
        return contam

    def _axis(self, n_points: int, sw_hz: float, sf_mhz: float, nucleus) -> np.ndarray:
        """The ppm axis, on the nucleus' reference unless "ref_ppm" overrides it."""
        ppm = ppm_axis(n_points, sw_hz, sf_mhz, nucleus)
        if self.ref_ppm is not None:
            ppm = ppm - ppm_reference(nucleus) + float(self.ref_ppm)
        return ppm

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add artificial peaks to list of NIFTI_MRS objects.

        The NIfTI-MRS format stores FIDs, so each subject is taken to a
        spectrum here and back; a pipeline never reaches this method in the
        wrong domain, since the declared frequency DOMAIN governs its plan.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, processed_water_list)
        """
        table = self._draw(len(data_list))
        processed_data = []

        for i, nifti in enumerate(data_list):
            fid = nifti[:]
            sw_hz = 1.0 / nifti.dwelltime
            sf_mhz = nifti.spectrometer_frequency[0]
            nucleus = nifti.nucleus[0] if nifti.nucleus else '1H'

            # The spectral axis is index 3 of a NIfTI-MRS array; bring it last
            # so a coil or average axis behind it is not mistaken for it.
            moved = fid.ndim > 4
            work = np.moveaxis(fid, 3, -1) if moved else fid
            n_points = work.shape[-1]

            spec = np.fft.fftshift(np.fft.ifft(work, axis=-1), axes=-1)
            profile = self._profile(i, self._axis(n_points, sw_hz, sf_mhz, nucleus),
                                    table, sf_mhz)
            magnitude = np.abs(spec) if self.amp_mode == 'abs' else np.abs(np.real(spec))
            peak_ref = np.max(magnitude, axis=-1, keepdims=True)
            peak_ref = np.where(peak_ref > 0, peak_ref, 1.0)
            spec = spec + peak_ref * profile

            out = np.fft.fft(np.fft.ifftshift(spec, axes=-1), axis=-1)
            nifti[:] = np.moveaxis(out, -1, 3) if moved else out
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add artificial peaks to tensor/array data (**any backend**).

        Everything touching the data runs on the data's own backend, so
        gradients and device placement survive. Only the peak shapes are built
        in NumPy: they depend on the ppm axis and the drawn parameters alone,
        one profile per sample, promoted once with "match_backend".

        Args:
            data_array: Input spectra of shape "(batch, ..., n_points)"
            water_array: Optional water reference (unchanged)
            backend: Backend enum (unused)
            **kwargs: Must contain "'sw_hz'" and "'sf_mhz'"; "'nucleus'" sets
                the ppm reference (1H when absent)

        Returns:
            Tuple of (processed_data, water_array)
        """
        sw_hz = kwargs.get('sw_hz')
        sf_mhz = kwargs.get('sf_mhz')
        if sw_hz is None or sf_mhz is None:
            raise ValueError("ArtificialPeaks.process_tensor requires 'sw_hz' and 'sf_mhz' in kwargs")
        nucleus = kwargs.get('nucleus', '1H')

        spec = data_array
        shape = ops.shape(spec)
        ndim = len(shape)
        n_points = int(shape[-1])
        batch = int(shape[0]) if ndim > 1 else 1

        # 1. ppm axis and one contamination profile per sample (NumPy)
        ppm = self._axis(n_points, float(sw_hz), float(sf_mhz), nucleus)
        table = self._draw(batch)
        unit_contam = np.stack([self._profile(i, ppm, table, float(sf_mhz))
                                for i in range(batch)])
        unit_contam = batch_profile(unit_contam, ndim)

        # 2. One amplitude per FID, on the data's own backend
        magnitude = ops.abs(spec if self.amp_mode == 'abs' else ops.real(spec))
        peak_ref = ops.amax(magnitude, axis=-1, keepdims=True)
        peak_ref = ops.where(peak_ref > 0, peak_ref, ops.cast_like(peak_ref * 0.0 + 1.0, peak_ref))

        contam = ops.cast_like(match_backend(unit_contam, spec), spec) \
            * ops.cast_like(peak_ref, spec)

        # 3. Add contamination in the spectral domain (backend-native)
        return spec + contam, water_array

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
