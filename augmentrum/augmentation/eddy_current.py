####################################################################################################
#                                      eddy_current.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Implements water-derived and synthetic eddy current phase distortions for MRS data.     #
#                                                                                                  #
####################################################################################################

import numpy as np
from typing import Optional, List
from scipy.signal import butter, filtfilt
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus


class EddyCurrent(BaseModule):
    """
    Add eddy current phase distortions to MRS data.

    Supports two modes:
    - 'synthetic': Generate synthetic eddy current phase trajectory
    - 'water': Derive eddy current phase from water reference

    Parameters
    ----------
    mode : str
        Eddy current mode: 'synthetic' or 'water' (default: 'synthetic')

    **Synthetic Mode Parameters:**
    std_rad : float
        Standard deviation of phase noise in radians (default: 0.6)
    lp_cut_hz : float
        Low-pass filter cutoff frequency in Hz (default: 25.0)
    seed : int, optional
        Random seed for reproducibility

    **Common Parameters:**
    strength : float
        Strength multiplier for eddy current effect (default: 1.0)
    remove_linear : bool
        Remove linear trend from phase (default: True)

    **Water Mode Parameters:**
    Uses water reference from water_list parameter in __call__
    lp_cut_hz : float
        Low-pass filter cutoff for water-derived phase (default: 20.0)

    Examples
    --------
    >>> # Synthetic eddy current
    >>> ec = EddyCurrent(mode='synthetic', std_rad=0.8, lp_cut_hz=30.0)
    >>> result_data, _ = ec(nifti_plus, None)
    >>>
    >>> # Water-derived eddy current
    >>> ec = EddyCurrent(mode='water', lp_cut_hz=20.0, strength=1.0)
    >>> result_data, result_water = ec(nifti_plus, water_plus)
    """

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self, mode: str = 'synthetic',
                 std_rad: float = 0.6, lp_cut_hz: float = 25.0,
                 strength: float = 1.0, remove_linear: bool = True, seed: Optional[int] = None):
        """Initialize eddy current module."""
        super().__init__(mode=mode, std_rad=std_rad, lp_cut_hz=lp_cut_hz,
                        strength=strength, remove_linear=remove_linear, seed=seed)

        self.mode = mode.lower()
        self.std_rad = std_rad
        self.lp_cut_hz = lp_cut_hz
        self.strength = strength
        self.remove_linear = remove_linear
        self.seed = seed

        if self.mode not in ['synthetic', 'water']:
            raise ValueError(f"mode must be 'synthetic' or 'water', got '{mode}'")

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add eddy current distortion to list of NIFTI_MRS objects.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects (required for 'water' mode)
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, processed_water_list)
        """
        if self.mode == 'water' and water_list is None:
            raise ValueError("Water reference required for 'water' mode eddy current")

        processed_data = []

        for idx, nifti in enumerate(data_list):
            # Get FID data
            fid = nifti[:]

            # Get spectral width
            sw_hz = 1.0 / nifti.dwelltime

            # Get water reference if available
            water_fid = None
            if water_list is not None and idx < len(water_list):
                water_fid = water_list[idx][:]

            # Apply eddy current
            fid_with_ec = self._add_eddy_current(fid, sw_hz, water_fid)

            # Update NIFTI_MRS data
            nifti[:] = fid_with_ec
            processed_data.append(nifti)

        return processed_data, water_list

    def _add_eddy_current(self, fid: np.ndarray, sw_hz: float, water_fid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Add eddy current to FID data.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz
            water_fid: Optional water reference FID (for 'water' mode)

        Returns:
            FID with eddy current
        """
        # Work with last dimension (FID points)
        original_shape = fid.shape
        N = original_shape[-1]

        # Reshape to 2D for processing
        fid_2d = fid.reshape(-1, N)
        result = np.zeros_like(fid_2d)

        # If water mode, also reshape water FID
        if self.mode == 'water' and water_fid is not None:
            water_2d = water_fid.reshape(-1, N)
        else:
            water_2d = None

        # Process each FID
        for i in range(fid_2d.shape[0]):
            fid_1d = fid_2d[i]

            # Generate phase distortion based on mode
            if self.mode == 'water' and water_2d is not None:
                # Extract phase from water reference
                water_1d = water_2d[min(i, water_2d.shape[0] - 1)]  # Handle different sizes
                phi_ec = self._ec_phase_from_water(water_1d, sw_hz)
            else:
                # Generate synthetic phase
                phi_ec = self._synth_ec_phase(N, sw_hz)

            # Apply phase distortion
            fid_with_ec = fid_1d * np.exp(1j * self.strength * phi_ec)
            result[i] = fid_with_ec

        # Reshape back to original shape
        return result.reshape(original_shape)

    def _ec_phase_from_water(self, fid_water: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Extract eddy current phase trajectory from water reference.

        Args:
            fid_water: Water reference FID
            sw_hz: Spectral width in Hz

        Returns:
            Phase values in radians
        """
        fid_water = np.asarray(fid_water)
        N = fid_water.size
        t = np.arange(N, dtype=float) / float(sw_hz)

        # Unwrap phase
        phi = np.unwrap(np.angle(fid_water))

        # Apply low-pass filter
        if self.lp_cut_hz is not None and self.lp_cut_hz > 0:
            nyq = 0.5 * float(sw_hz)
            Wn = min(max(self.lp_cut_hz / nyq, 1e-6), 0.999999)
            b, a = butter(2, Wn, btype='low')
            phi = filtfilt(b, a, phi)

        # Remove linear trend if requested
        if self.remove_linear:
            A = np.c_[np.ones(N), t]
            k0, k1 = np.linalg.lstsq(A, phi, rcond=None)[0]
            phi = phi - (k0 + k1 * t)

        return phi

    def _synth_ec_phase(self, N: int, sw_hz: float) -> np.ndarray:
        """
        Synthesize eddy current phase trajectory.

        Args:
            N: Number of points
            sw_hz: Spectral width in Hz

        Returns:
            Phase values in radians
        """
        rng = np.random.default_rng(self.seed)
        t = np.arange(N, dtype=float) / float(sw_hz)

        # Generate white noise
        w = rng.normal(scale=self.std_rad, size=N)

        # Apply low-pass filter
        nyq = 0.5 * float(sw_hz)
        Wn = min(max(self.lp_cut_hz / nyq, 1e-6), 0.999999)
        b, a = butter(2, Wn, btype='low')
        phi = filtfilt(b, a, w)

        # Remove linear trend if requested
        if self.remove_linear:
            A = np.c_[np.ones(N), t]
            k0, k1 = np.linalg.lstsq(A, phi, rcond=None)[0]
            phi = phi - (k0 + k1 * t)

        return phi
