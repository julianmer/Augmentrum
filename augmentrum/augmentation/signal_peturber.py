####################################################################################################
#                                   signal_perturber.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Implements SignalPerturber for signal-level augmentations of MRS data, including        #
#          noise injection, frequency shifts, phase errors, and spectrum misalignments.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np


#**************************************************************************************************#
#                                      Class SignalPerturber                                       #
#**************************************************************************************************#
#                                                                                                  #
# Implements signal-level augmentations of MRS data, including noise injection, frequency shifts,  #
# phase errors, and spectral misalignments (for edited spectra).                                   #
#                                                                                                  #
#**************************************************************************************************#
class SignalPerturber:
    def __init__(self, amp_mean=0.0, amp_var_low=0.0, amp_var_high=0.0,
                 phase_low=0.0, phase_high=0.0, freq_low=0.0, freq_high=0.0, misalign=False):
        self.amp_mean = amp_mean
        self.amp_var_low = amp_var_low
        self.amp_var_high = amp_var_high
        self.phase_low = phase_low
        self.phase_high = phase_high
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.misalign = misalign

    def __call__(self, data, water=None, **kwargs):
        """
        Applies signal perturbations to the MRS data.

        Args:
            data: Input MRS data (NiftiMRS object).
            water: Optional water reference data (NiftiMRS object).
            **kwargs: Additional arguments.
        """

        # amplitude noise
        if self.amp_var_high > self.amp_var_low:
            amp_var = np.random.uniform(self.amp_var_low, self.amp_var_high, size=data.shape)
            amp_noise = (np.random.normal(self.amp_mean, np.sqrt(amp_var), size=data.shape) +
                         1j * np.random.normal(self.amp_mean, np.sqrt(amp_var), size=data.shape))
            data[:] += amp_noise

        # phase noise
        if self.phase_high > self.phase_low:
            phase = np.random.uniform(self.phase_low, self.phase_high,
                                      size=data.shape[:3] + (1,) + data.shape[4:])
            data[:] *= np.exp(1j * phase)

        # frequency shift
        if self.freq_high > self.freq_low:
            time = np.arange(data.shape[3]) * data.dwelltime
            freq_noise = np.random.uniform(self.freq_low, self.freq_high,
                                           size=data.shape[:3] + (1,) + data.shape[4:])
            data[:] *= np.exp(1j * time * freq_noise * 2 * np.pi)

        if self.misalign:
            # TODO: implement misalignment for edited spectra
            raise NotImplementedError("Misalignment not implemented yet.")

        return data, water