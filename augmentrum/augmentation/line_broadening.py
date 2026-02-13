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

import numpy as np
from typing import Optional, List
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus


class LineBroadening(BaseModule):
    """
    Apply line broadening to MRS data.

    Supports Lorentzian (exponential), Gaussian, and Voigt (combined) broadening.
    Works with any backend - operates in time domain on FID data.

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

    @staticmethod
    def _apply_lorentzian(fid: np.ndarray, sw_hz: float, lb_hz: float) -> np.ndarray:
        """
        Apply Lorentzian (exponential) line broadening.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz
            lb_hz: Lorentzian FWHM in Hz

        Returns:
            Broadened FID
        """
        if lb_hz <= 0:
            return fid

        # Get time points for the last dimension (FID points)
        N = fid.shape[-1]
        t = np.arange(N) / sw_hz

        # Reshape t to broadcast correctly
        t_shape = tuple([1] * (fid.ndim - 1) + [N])
        t = t.reshape(t_shape)

        # Apply exponential decay
        envelope = np.exp(-np.pi * lb_hz * t)
        return fid * envelope

    @staticmethod
    def _apply_gaussian(fid: np.ndarray, sw_hz: float, gb_hz: float) -> np.ndarray:
        """
        Apply Gaussian line broadening.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz
            gb_hz: Gaussian FWHM in Hz

        Returns:
            Broadened FID
        """
        if gb_hz <= 0:
            return fid

        # Get time points for the last dimension
        N = fid.shape[-1]
        t = np.arange(N) / sw_hz

        # Reshape t to broadcast correctly
        t_shape = tuple([1] * (fid.ndim - 1) + [N])
        t = t.reshape(t_shape)

        # Apply Gaussian decay
        envelope = np.exp(-((np.pi * gb_hz * t) ** 2) / (4 * np.log(2)))
        return fid * envelope

    @staticmethod
    def _apply_voigt(fid: np.ndarray, sw_hz: float, lb_hz: float, gb_hz: float) -> np.ndarray:
        """
        Apply Voigt (Lorentzian × Gaussian) line broadening.

        Args:
            fid: Input FID data
            sw_hz: Spectral width in Hz
            lb_hz: Lorentzian FWHM in Hz
            gb_hz: Gaussian FWHM in Hz

        Returns:
            Broadened FID
        """
        if lb_hz <= 0 and gb_hz <= 0:
            return fid

        # Get time points for the last dimension
        N = fid.shape[-1]
        t = np.arange(N) / sw_hz

        # Reshape t to broadcast correctly
        t_shape = tuple([1] * (fid.ndim - 1) + [N])
        t = t.reshape(t_shape)

        # Apply combined envelope
        envelope = (np.exp(-np.pi * lb_hz * t) *
                   np.exp(-((np.pi * gb_hz * t) ** 2) / (4 * np.log(2))))
        return fid * envelope
