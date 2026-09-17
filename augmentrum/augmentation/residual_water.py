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
# Purpose: Simulates imperfect water suppression by adding causal Lorentzian water lobes around    #
#          the water resonance.                                                                    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List
from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from augmentrum.processing.utils import (ppm_axis, ppm_reference, batch_profile,
                                         per_sample_factor, causal_lineshape, to_backend)
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from nifti_mrs_plus import ops


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

    Adds Lorentzian-shaped water peaks around the water resonance to simulate
    imperfect water suppression. Each lobe is a decaying complex exponential
    in the FID, as residual water is, transformed to the spectrum
    ("causal_lineshape"): a Lorentzian drawn on the axis would be real, and a
    real spectrum has a two-sided FID whose second half wraps to the end of
    the acquisition and rings once the FID is zero-filled or truncated. All
    ppm values are on the FSL-MRS / NIfTI-MRS axis (protons referenced to
    4.65 ppm), so a lobe placed here sits where an FSL-MRS plot shows it.

    Parameters
    ----------
    center_ppm : float, optional
        Center position of the water peak in ppm. None (default) is the
        nucleus' reference from "ppm_reference" - 4.65 ppm for 1H - which is
        where the water sits on the FSL-MRS axis.
    peaks : tuple of tuples, optional
        Each tuple is (delta_ppm, FWHM_ppm, rel_amp) or
        (delta_ppm, FWHM_ppm, rel_amp, phase_deg)
        - delta_ppm: offset from center in ppm
        - FWHM_ppm: Full Width at Half Maximum in ppm
        - rel_amp: relative amplitude of the lobe in the FID - its area in the
          spectrum, as a signal model weights it - so a narrower lobe stands
          taller
        - phase_deg: per-peak phase in degrees, optional
        None (default) uses the peaks of the chosen *model*.
    phase_deg : float
        Global phase of the water profile in degrees (default: 0.0)
    amplitude_scale : float
        Scale factor for water amplitude relative to spectrum (default: 0.1 = 10%)
    model : str
        Which peak set None *peaks* means: 'lobes' (default) is three lobes at
        0.0, +0.12, -0.15 ppm; 'turco' is the seven-Lorentzian model of
        Turco et al. (WaterFit), seeded at the water and +-0.05, +-0.10,
        +-0.15 ppm around it. Explicit *peaks* always win over the model.

    In a pipeline, "amplitude_scale", "phase_deg" and "center_ppm" given as
    ranges are drawn once per sample, so a batch carries a spread of residual
    waters rather than one.

    Examples
    --------
    >>> # Default water (10% amplitude, 3 lobes)
    >>> water = ResidualWater()
    >>> result_data, _ = water(nifti_plus, None)

    >>> # Subtle water (5%)
    >>> water = ResidualWater(amplitude_scale=0.05)

    >>> # Turco et al. seven-Lorentzian residual water
    >>> water = ResidualWater(model='turco')

    >>> # Custom peaks, with a per-peak phase on the second lobe
    >>> water = ResidualWater(
    ...     peaks=((0.0, 0.25, 1.0), (0.10, 0.20, 0.5, 30.0)),
    ...     phase_deg=15.0
    ... )
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # Water is a peak at a ppm position, which only exists in a spectrum.
    DOMAIN = Domain(spectral='frequency')

    # The profile is built per sample, so each of these can differ per sample.
    PER_SAMPLE_PARAMS = ('amplitude_scale', 'phase_deg', 'center_ppm')

    MODELS = ('lobes', 'turco')

    #: Three asymmetric lobes — the house model of imperfect suppression.
    LOBE_PEAKS = ((0.0, 0.20, 1.0),
                  (0.12, 0.18, 0.4),
                  (-0.15, 0.25, 0.3))

    #: Seven Lorentzians at the WaterFit seeds of Turco et al. (MRM 2026),
    #: offsets from the water; 0.24 ppm FWHM is their 30 Hz init damping at 3 T.
    TURCO_PEAKS = ((0.0, 0.24, 1.0),
                   (0.05, 0.24, 1.0),
                   (-0.05, 0.24, 1.0),
                   (0.10, 0.24, 1.0),
                   (-0.10, 0.24, 1.0),
                   (0.15, 0.24, 1.0),
                   (-0.15, 0.24, 1.0))

    def __init__(self, center_ppm: Optional[float] = None,
                 peaks: Optional[tuple] = None,
                 phase_deg: float = 0.0,
                 amplitude_scale: float = 0.1,
                 model: str = 'lobes'):
        """Initialize residual water module."""
        super().__init__()

        if model not in self.MODELS:
            raise ValueError(f"model must be one of {self.MODELS}, got {model!r}")

        self.center_ppm = center_ppm
        self.model = model
        self.peaks = peaks if peaks is not None else (
            self.TURCO_PEAKS if model == 'turco' else self.LOBE_PEAKS)
        self.phase_deg = phase_deg
        self.amplitude_scale = amplitude_scale

    #*****************#
    #   water lobes   #
    #*****************#
    @staticmethod
    def _water_lobe_profile(ppm_axis, *, center_ppm=None,
                            peaks=((0.0, 0.20, 1.0),   # (delta_ppm, FWHM_ppm, rel_amp)
                                   (+0.12, 0.18, 0.4),
                                   (-0.15, 0.25, 0.3)),
                            phase_deg=0.0):
        """
        The complex water lobe profile at unit amplitude.

        Depends only on the ppm axis and the lobe parameters, so it is built
        in NumPy and multiplied by a per-FID amplitude afterwards. Every lobe
        is causal, and the sum of causal lobes is causal.

        Args:
            ppm_axis: PPM axis, the bins of "fftshift(ifft(fid))"
            center_ppm: Center position of water peak; None is the proton
                reference (4.65 ppm)
            peaks: Tuple of (delta_ppm, FWHM_ppm, rel_amp[, phase_deg]) per lobe
            phase_deg: Global phase of the water profile in degrees

        Returns:
            Complex profile with the same length as ppm_axis, peak magnitude 1
        """
        lobes = ResidualWater._water_lobes(ppm_axis, center_ppm=center_ppm, peaks=peaks)
        return lobes * np.exp(1j * np.deg2rad(phase_deg))

    @staticmethod
    def _water_lobes(ppm_axis, *, center_ppm=None,
                     peaks=((0.0, 0.20, 1.0), (+0.12, 0.18, 0.4), (-0.15, 0.25, 0.3))):
        """
        "_water_lobe_profile" before its global phase: the part that does not
        depend on the phase, so a batch of phases can share one build.
        """
        ppm = np.asarray(ppm_axis, float)
        if center_ppm is None:
            center_ppm = ppm_reference('1H')
        w = np.zeros_like(ppm, complex)

        # Causal Lorentzians of unit area, each with its own optional phase: a
        # lineshape of unit peak becomes one of unit area through the height
        # 2 / (pi FWHM) a unit-area Lorentzian has, so rel_amp weighs the lobe
        # as its amplitude in the FID.
        for peak in peaks:
            dppm, fwhm_ppm, rel_amp = peak[:3]
            peak_phi = np.deg2rad(peak[3]) if len(peak) > 3 else 0.0
            lobe = causal_lineshape(ppm, center_ppm + dppm, lorentz_ppm=fwhm_ppm)
            w = w + rel_amp * np.exp(1j * peak_phi) * lobe * 2.0 / (np.pi * fwhm_ppm)

        # Normalize lobes to ~unit max
        peak_mag = np.max(np.abs(w))
        return w / (peak_mag if peak_mag > 0 else 1.0)

    def _profile(self, index: int, ppm: np.ndarray, nucleus) -> np.ndarray:
        """
        Sample *index*'s unit lobe profile on *ppm*.

        Reads this sample's entry out of whatever the pipeline set - a scalar
        for the batch or a vector with one value per sample - so the tensor and
        NIfTI-list paths build the same profile for the same sample.
        """
        center = self.sample_of(self.center_ppm, index)
        return self._water_lobe_profile(
            ppm,
            center_ppm=ppm_reference(nucleus) if center is None else float(center),
            peaks=self.peaks,
            phase_deg=float(self.sample_of(self.phase_deg, index)),
        )

    def _profiles(self, batch: int, ppm: np.ndarray, nucleus) -> np.ndarray:
        """
        One unit profile per sample, "(batch, N)".

        The lobes depend on the axis and the sample's centre alone, so samples -
        and batches - that share them share one build; the global phase, drawn
        per sample, is applied afterwards, as "_water_lobe_profile" applies it.
        """
        cache = self.__dict__.setdefault('_lobe_cache', {})
        axis = (ppm.size, hash(ppm.tobytes()), nucleus, repr(self.peaks))
        rows = []
        for i in range(batch):
            center = self.sample_of(self.center_ppm, i)
            center = ppm_reference(nucleus) if center is None else float(center)
            key = axis + (center,)
            if key not in cache:
                if len(cache) >= 256:
                    cache.clear()
                cache[key] = self._water_lobes(ppm, center_ppm=center, peaks=self.peaks)
            rows.append(cache[key])
        phases = np.array([float(self.sample_of(self.phase_deg, i)) for i in range(batch)])
        return np.stack(rows) * np.exp(1j * np.deg2rad(phases))[:, None]

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add residual water to list of NIFTI_MRS objects.

        The NIfTI-MRS format stores FIDs, so each subject is taken to a
        spectrum here and back; a pipeline never reaches this method in the
        wrong domain, since the declared frequency DOMAIN governs its plan.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects (unchanged)
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, water_list)
        """
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
            profile = self._profile(i, ppm_axis(n_points, sw_hz, sf_mhz, nucleus), nucleus)
            scale = float(self.sample_of(self.amplitude_scale, i))
            water_amp = scale * np.max(np.abs(np.real(spec)), axis=-1, keepdims=True)
            spec = spec + water_amp * profile

            out = np.fft.fft(np.fft.ifftshift(spec, axes=-1), axis=-1)
            nifti[:] = np.moveaxis(out, -1, 3) if moved else out
            processed_data.append(nifti)

        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add residual water peaks to tensor/array data (**any backend**).

        The lobe profiles depend only on the FID grid and the parameters, so
        they are built in NumPy, one per sample, and promoted once with
        "match_backend". The only data-dependent term is one amplitude per
        FID, taken on the data's own backend so the spectrum is never
        converted.

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
            raise ValueError("ResidualWater.process_tensor requires 'sw_hz' and 'sf_mhz' in kwargs")
        nucleus = kwargs.get('nucleus', '1H')

        spec = data_array
        shape = ops.shape(spec)
        ndim = len(shape)
        n_points = int(shape[-1])
        batch = int(shape[0]) if ndim > 1 else 1

        # 1. ppm axis and one unit profile per sample (NumPy, coordinates only)
        ppm = ppm_axis(n_points, float(sw_hz), float(sf_mhz), nucleus)
        unit_lobes = batch_profile(self._profiles(batch, ppm, nucleus), ndim)

        # 2. One amplitude per FID, on the data's own backend
        amp_ref = ops.amax(ops.abs(ops.real(spec)), axis=-1, keepdims=True)
        water_amp = per_sample_factor(self.amplitude_scale, ndim, amp_ref) * amp_ref

        water_add = ops.cast_like(to_backend(unit_lobes, spec), spec) \
            * ops.cast_like(water_amp, spec)

        # 3. Add water in the spectral domain (backend-native)
        return spec + water_add, water_array
