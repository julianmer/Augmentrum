####################################################################################################
#                                    amplitude_scaling.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kcigwe@bu.edu)                                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-08                                                                              #
#                                                                                                  #
# Purpose: Amplitude scaling for MRS data augmentation.                                            #
#          Note: True metabolite-specific scaling requires basis functions (simulated data only).  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import List, Optional, Union, Tuple
import numpy as np
from fsl_mrs.core import NIFTI_MRS

from augmentrum.core.base_module import BaseModule
from nifti_mrs_plus import Backend
from nifti_mrs_plus.ops import match_backend


#**************************************************************************************************#
#                                      Class AmplitudeScaling                                      #
#**************************************************************************************************#
#                                                                                                  #
# Apply amplitude scaling to MRS data.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class AmplitudeScaling(BaseModule):
    """
    Apply amplitude scaling to MRS data.

    **Note:** This performs global amplitude scaling. True metabolite-specific scaling
    requires access to individual basis functions and is only applicable to simulated data.

    For realistic metabolite-specific scaling:
    - Fit in-vivo data to a basis set
    - Scale individual metabolite components
    - Reconstruct the FID

    This module provides global scaling for demonstration/preprocessing purposes.

    Parameters:
        scale_factor: Scaling factor to apply. Can be:
                     - Single float: Fixed scaling
                     - Tuple (min, max): Random uniform sampling from range
                     Default: 1.0 (no scaling)
        distribution: Distribution for random sampling ('uniform', 'normal')
                     Only used if scale_factor is a tuple
        seed: Random seed for reproducibility

    Examples:
        >>> # Fixed scaling
        >>> scaler = AmplitudeScaling(scale_factor=0.8)
        >>>
        >>> # Random scaling in range
        >>> scaler = AmplitudeScaling(scale_factor=(0.7, 1.3))
        >>>
        >>> # Normal distribution around 1.0
        >>> scaler = AmplitudeScaling(scale_factor=(0.9, 1.1), distribution='normal')
    """

    SUPPORTED_BACKENDS = [Backend.NIFTI_LIST, Backend.NUMPY, Backend.PYTORCH,
                         Backend.TENSORFLOW, Backend.JAX, Backend.KERAS]

    def __init__(
        self,
        scale_factor: Union[float, Tuple[float, float]] = 1.0,
        distribution: str = 'uniform',
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize AmplitudeScaling module."""
        super().__init__()

        self.scale_factor = scale_factor
        self.distribution = distribution
        self.seed = seed

        # Validate distribution
        if distribution not in ['uniform', 'normal']:
            raise ValueError(f"distribution must be 'uniform' or 'normal', got {distribution}")

    def _get_scale_factor(self) -> float:
        """Get scale factor (sample if range is provided)."""
        if isinstance(self.scale_factor, (tuple, list)):
            min_scale, max_scale = self.scale_factor

            if self.distribution == 'uniform':
                return self.rng.numpy_rng().uniform(min_scale, max_scale)
            elif self.distribution == 'normal':
                # Use range as mean ± std
                mean = (min_scale + max_scale) / 2.0
                std = (max_scale - min_scale) / 4.0  # ~95% within range
                return self.rng.numpy_rng().normal(mean, std)
        else:
            return float(self.scale_factor)

    def process_nifti_list(
        self,
        data_list: List[NIFTI_MRS],
        water_list: Optional[List[NIFTI_MRS]] = None,
        **kwargs
    ) -> Tuple[List[NIFTI_MRS], Optional[List[NIFTI_MRS]]]:
        """
        Process list of NIFTI_MRS objects.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects

        Returns:
            Tuple of (scaled_data_list, water_list_unchanged)
        """
        scaled_data = []

        for nifti in data_list:
            # Get scale factor (sample if random)
            scale = self._get_scale_factor()

            # Get data
            fid = nifti[:].copy()

            # Apply scaling
            fid_scaled = fid * scale

            # Create new NIFTI_MRS with scaled data
            nifti_scaled = nifti.copy()
            nifti_scaled[:] = fid_scaled

            scaled_data.append(nifti_scaled)

        # Water references unchanged
        return scaled_data, water_list

    def process_tensor(
        self,
        data,
        water=None,
        **kwargs
    ) -> Tuple:
        """
        Process tensor data.

        Args:
            data: Input tensor (batch, ...)
            water: Optional water reference tensor

        Returns:
            Tuple of (scaled_data, water_unchanged)
        """
        # Get scale factor for each sample in batch
        batch_size = data.shape[0]
        scales = np.array([self._get_scale_factor() for _ in range(batch_size)])

        # Reshape scales for broadcasting
        scale_shape = [batch_size] + [1] * (data.ndim - 1)
        scales = scales.reshape(scale_shape)

        # Apply scaling
        data_scaled = data * match_backend(scales, data)

        return data_scaled, water

    def __repr__(self):
        return (f"AmplitudeScaling(scale_factor={self.scale_factor}, "
                f"distribution='{self.distribution}')")
