####################################################################################################
#                                      gaussian_noise.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Implements uncorrelated complex Gaussian noise (AWGN) for MRS data augmentation.        #
#                                                                                                  #
####################################################################################################

import numpy as np
from typing import Optional, List
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus


class GaussianNoise(BaseModule):
    """
    Add uncorrelated complex Gaussian noise (AWGN) to MRS data.

    Adds white Gaussian noise to both real and imaginary components
    of the complex FID signal.

    Parameters
    ----------
    snr: float, optional
        Signal-to-noise ratio (unitless, signal_power / noise_power)
    snr_db : float, optional
        Signal-to-noise ratio in dB (signal_power / noise_power)
    sigma : float, optional
        Standard deviation per dimension (real and imaginary)
    sigma_frac : float, optional
        Sigma as fraction of max|FID| (alternative to sigma)
    seed : int, optional
        Random seed for reproducibility

    Notes
    -----
    Provide EITHER snr_db, sigma, OR sigma_frac (not multiple).
    If snr_db is used, sigma is computed from mean(|fid|^2).

    Examples
    --------
    >>> # Using SNR
    >>> noise = GaussianNoise(snr=20.0)
    >>>
    >>> # Using SNR in dB
    >>> noise = GaussianNoise(snr_db=10.0)
    >>>
    >>> # Using sigma directly
    >>> noise = GaussianNoise(sigma=0.01)
    >>>
    >>> # Using sigma as fraction
    >>> noise = GaussianNoise(sigma_frac=0.02)
    """

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self,
                 snr: Optional[float] = None,
                 snr_db: Optional[float] = None,
                 sigma: Optional[float] = None,
                 sigma_frac: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Initialize Gaussian noise module.

        Args:
            snr_db: SNR in dB
            sigma: Absolute noise level
            sigma_frac: Noise as fraction of signal max
            seed: Random seed
        """
        super().__init__(snr=snr, snr_db=snr_db, sigma=sigma, sigma_frac=sigma_frac, seed=seed)

        self.snr = snr
        self.snr_db = snr_db
        self.sigma = sigma
        self.sigma_frac = sigma_frac
        self.seed = seed

        # Validate parameters
        params_provided = sum([snr is not None, snr_db is not None,
                               sigma is not None, sigma_frac is not None])
        if params_provided == 0:
            raise ValueError("Must provide one of: snr, snr_db, sigma, or sigma_frac")
        if params_provided > 1:
            raise ValueError("Provide only ONE of: snr, snr_db, sigma, or sigma_frac")

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """
        Add Gaussian noise to list of NIFTI_MRS objects.

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

            # Add noise
            fid_noisy = self._add_noise(fid)

            # Update NIFTI_MRS data
            nifti[:] = fid_noisy
            processed_data.append(nifti)

        return processed_data, water_list

    def _add_noise(self, fid: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to FID data.

        Args:
            fid: Input FID data

        Returns:
            FID with noise added
        """
        rng = np.random.default_rng(self.seed)

        fid = np.asarray(fid)
        original_shape = fid.shape
        fid_flat = fid.reshape(-1, original_shape[-1])

        out_dtype = np.result_type(fid.dtype, np.complex64)
        fid_flat = fid_flat.astype(out_dtype, copy=False)

        if self.sigma is not None:
            scale = float(self.sigma)  # scalar -> broadcast
        elif self.sigma_frac is not None:
            peak = np.max(np.abs(fid_flat), axis=-1)
            peak = np.where(peak > 0, peak, 1.0)
            scale = (self.sigma_frac * peak)[:, None]  # per row
        else:
            sig_pow = np.mean(np.abs(fid_flat) ** 2, axis=-1) + 1e-16
            if self.snr is not None:
                noise_pow = sig_pow / float(self.snr)
            else:  # self.snr_db is not None
                noise_pow = sig_pow / (10.0 ** (float(self.snr_db) / 10.0))

            # complex noise: power split equally over Re/Im
            scale = np.sqrt(noise_pow / 2.0)[:, None]

        noise = (rng.normal(0.0, 1.0, size=fid_flat.shape) +
                 1j * rng.normal(0.0, 1.0, size=fid_flat.shape)) * scale

        return (fid_flat + noise).reshape(original_shape)
