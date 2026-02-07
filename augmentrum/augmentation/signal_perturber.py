####################################################################################################
#                                   signal_perturber.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Implements NoisePerturber for signal-level augmentations of MRS data, including         #
#          noise injection, frequency shifts, phase errors, and spectrum misalignments.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

# own
from augmentrum.core.base_module import BaseModule
from augmentrum.core import Backend


#**************************************************************************************************#
#                                      Class NoisePerturber                                        #
#**************************************************************************************************#
#                                                                                                  #
# Implements signal-level augmentations of MRS data, including noise injection, frequency shifts,  #
# phase errors, and spectral misalignments (for edited spectra).                                   #
#                                                                                                  #
#**************************************************************************************************#
class NoisePerturber(BaseModule):
    """
    Implements signal-level augmentations of MRS data.

    Adds noise injection, frequency shifts, phase errors, and spectral misalignments.
    Operates on NIfTI list by default, but can work with any backend.

    Logging is automatic via BaseModule (only if not volatile).
    """

    # Supports all backends
    SUPPORTED_BACKENDS = []  # Empty = supports all

    def __init__(self, amp_mean=0.0, amp_var_low=0.0, amp_var_high=0.0,
                 phase_low=0.0, phase_high=0.0, freq_low=0.0, freq_high=0.0, misalign=False):
        """
        Initializes the NoisePerturber with specified parameters.

        Args:
            amp_mean (float): Mean of amplitude noise.
            amp_var_low (float): Lower bound of amplitude noise variance.
            amp_var_high (float): Upper bound of amplitude noise variance.
            phase_low (float): Lower bound of phase noise (radians).
            phase_high (float): Upper bound of phase noise (radians).
            freq_low (float): Lower bound of frequency shift (Hz).
            freq_high (float): Upper bound of frequency shift (Hz).
            misalign (bool): Whether to apply spectral misalignment (for edited spectra).
        """
        super().__init__(amp_mean=amp_mean, amp_var_low=amp_var_low, amp_var_high=amp_var_high,
                        phase_low=phase_low, phase_high=phase_high,
                        freq_low=freq_low, freq_high=freq_high, misalign=misalign)

        self.amp_mean = amp_mean
        self.amp_var_low = amp_var_low
        self.amp_var_high = amp_var_high
        self.phase_low = phase_low
        self.phase_high = phase_high
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.misalign = misalign

    def process_nifti_list(self, data_list, water_list=None, **kwargs):
        """
        Process each NIfTI-MRS object individually.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water NIFTI_MRS objects
            **kwargs: Additional arguments

        Returns:
            Tuple of (processed_data_list, processed_water_list)
        """
        processed_data = []

        for nifti in data_list:
            # Apply perturbations to this NIfTI object
            processed_nifti = self._apply_perturbations(nifti)
            processed_data.append(processed_nifti)

        # Water typically doesn't get perturbed
        return processed_data, water_list

    def _apply_perturbations(self, data):
        """
        Applies signal perturbations to a single NIfTI-MRS object.

        Args:
            data: Input MRS data (NIFTI_MRS object)

        Returns:
            Perturbed data
        """
        # amplitude noise
        if self.amp_var_high != 0 or self.amp_var_low != 0 or self.amp_mean != 0:
            amp_var = np.random.uniform(self.amp_var_low, self.amp_var_high, size=data.shape)
            amp_noise = (np.random.normal(self.amp_mean, np.sqrt(amp_var), size=data.shape) +
                        1j * np.random.normal(self.amp_mean, np.sqrt(amp_var), size=data.shape))
            data[:] += amp_noise

        # phase noise
        if self.phase_high != 0 or self.phase_low != 0:
            phase = np.random.uniform(self.phase_low, self.phase_high,
                                     size=data.shape[:3] + (1,) + data.shape[4:])
            data[:] *= np.exp(1j * phase)

        # frequency shift
        if self.freq_high != 0 or self.freq_low != 0:
            time = np.arange(data.shape[-1]) * data.dwelltime
            freq_noise = np.random.uniform(self.freq_low, self.freq_high,
                                          size=data.shape[:3] + (1,) + data.shape[4:])
            data[:] *= np.exp(1j * time * freq_noise * 2 * np.pi)

        if self.misalign:
            # TODO: implement misalignment for edited spectra
            raise NotImplementedError("Misalignment not implemented yet.")

        return data
