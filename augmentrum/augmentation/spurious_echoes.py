####################################################################################################
#                                    spurious_echoes.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Adds delayed echo replicas (ghosting artifacts) to MRS data.                            #
#                                                                                                  #
####################################################################################################

import numpy as np
from typing import Optional, List, Tuple
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus


class SpuriousEchoes(BaseModule):
    """
    Add spurious echo artifacts to MRS data.

    Simulates ghosting artifacts by adding delayed, attenuated copies
    of the FID signal at specified time delays.

    Parameters
    ----------
    echoes : list of tuples
        Each tuple is (delay_s, amp, phase_deg, decay_hz, freq_hz)
        - delay_s: Echo start time in seconds
        - amp: Relative amplitude (vs original FID)
        - phase_deg: Additional phase for this echo (degrees)
        - decay_hz: Exponential decay (linewidth in Hz)
        - freq_hz: Frequency offset of echo (Hz)
    global_phase_deg : float
        Global phase offset for all echoes (default: 0.0)

    Examples
    --------
    >>> # Add single echo at 0.1s with 30% amplitude
    >>> echoes = SpuriousEchoes(
    ...     echoes=[(0.1, 0.3, 0.0, 5.0, 0.0)],
    ...     global_phase_deg=0.0
    ... )
    >>> result_data, _ = echoes(nifti_plus, None)
    """

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self, echoes: List[Tuple[float, float, float, float, float]] = None,
                 global_phase_deg: float = 0.0):
        """Initialize spurious echoes module."""
        if echoes is None:
            echoes = [(0.1, 0.2, 0.0, 5.0, 0.0)]  # Default: one echo

        super().__init__(echoes=echoes, global_phase_deg=global_phase_deg)

        # Convert dict format to tuple format if needed
        processed_echoes = []
        for echo in echoes:
            if isinstance(echo, dict):
                # Convert dict to tuple: (delay_s, amp, phase_deg, decay_hz, freq_hz)
                # Support both 'delay_pts' (will be converted) and 'delay_s'
                delay_pts = echo.get('delay_pts', None)
                delay_s = echo.get('delay_s', None)
                if delay_s is None and delay_pts is not None:
                    delay_s = delay_pts  # Will be converted to seconds in _add_echoes_1d
                elif delay_s is None:
                    delay_s = 0.1  # Default

                amp = echo.get('amplitude', echo.get('amp', 0.1))
                phase_deg = echo.get('phase_deg', 0.0)
                decay_hz = echo.get('decay', echo.get('decay_hz', 5.0))
                freq_hz = echo.get('freq_hz', 0.0)

                processed_echoes.append((delay_s, amp, phase_deg, decay_hz, freq_hz))
            else:
                # Already a tuple
                processed_echoes.append(echo)

        self.echoes = processed_echoes
        self.global_phase_deg = global_phase_deg

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add spurious echoes to list of NIFTI_MRS objects.

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

            # Add echoes
            fid_with_echoes = self._add_echoes(fid, sw_hz)

            # Update NIFTI_MRS data
            nifti[:] = fid_with_echoes
            processed_data.append(nifti)

        return processed_data, water_list

    def _add_echoes(self, fid: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Add spurious echoes to FID data.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz

        Returns:
            FID with echoes added
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

            # Add echoes
            fid_with_echoes = self._add_echoes_1d(fid_1d, sw_hz)
            result[i] = fid_with_echoes

        # Reshape back to original shape
        return result.reshape(original_shape)

    def _add_echoes_1d(self, fid: np.ndarray, sw_hz: float) -> np.ndarray:
        """
        Add echoes to 1D FID.

        Args:
            fid: 1D FID data
            sw_hz: Spectral width in Hz

        Returns:
            FID with echoes
        """
        f = np.asarray(fid, dtype=np.complex128)
        n = f.size
        t = np.arange(n, dtype=float) / float(sw_hz)

        out = f.copy()
        gphi = np.deg2rad(self.global_phase_deg)

        for (delay_val, amp, phase_deg, decay_hz, freq_hz) in self.echoes:
            # delay_val might be in seconds or points - convert to seconds
            # If delay_val > 1.0, assume it's in points, otherwise in seconds
            if delay_val > 1.0:
                # Assume it's in points, convert to seconds
                delay_s = delay_val / float(sw_hz)
            else:
                # Already in seconds
                delay_s = delay_val

            # Step function: 1 after delay, 0 before
            u = (t >= delay_s).astype(float)
            td = (t - delay_s) * u

            # Echo signal
            echo = (
                amp
                * np.exp(1j * (gphi + np.deg2rad(phase_deg)))
                * np.exp(-np.pi * float(decay_hz) * td)
                * np.exp(1j * 2.0 * np.pi * float(freq_hz) * td)
            )

            # Add delayed replica
            out += f * echo

        return out
