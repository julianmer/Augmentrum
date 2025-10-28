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

# own
from augmentrum.processing.utils import update_processing_prov


#**************************************************************************************************#
#                                      Class SignalPerturber                                       #
#**************************************************************************************************#
#                                                                                                  #
# Abstract base class for signal-level augmentations of MRS data.                                  #
#                                                                                                  #
#**************************************************************************************************#
class SignalPerturber:

    def __init__(self, **kwargs):
        """
        Initializes the SignalPerturber.

        Args:
            **kwargs: Additional arguments.
        """
        self.kwargs = kwargs

    def __call__(self, data, water=None, **kwargs):
        """
        Applies signal perturbations to the MRS data.

        If the update_processing_prov function is not implemented in the subclass,
        a generic message is added to the processing provenance.

        Args:
            data: Input MRS data (NiftiMRS object).
            water: Optional water reference data (NiftiMRS object).
            **kwargs: Additional arguments.
        """
        # if update processing provenance is not implemented, add generic message
        if not hasattr(self, 'update_processing_prov'):
            processing_info = f'{__name__}.{self.__class__.__name__}, '
            processing_info += f'**kwargs: {self.kwargs}'
            update_processing_prov(data, 'Generic SignalPerturber', processing_info)

        else:
            self.update_processing_prov(data, water, **kwargs)

        return self.forward(data, water, **kwargs)

    def forward(self, data, water=None, **kwargs):
        raise NotImplementedError("SignalPerturber is an abstract base class.")


#**************************************************************************************************#
#                                      Class NoisePerturber                                        #
#**************************************************************************************************#
#                                                                                                  #
# Implements signal-level augmentations of MRS data, including noise injection, frequency shifts,  #
# phase errors, and spectral misalignments (for edited spectra).                                   #
#                                                                                                  #
#**************************************************************************************************#
class NoisePerturber(SignalPerturber):
    """
    Implements signal-level augmentations of MRS data, including noise injection, frequency shifts,
    phase errors, and spectral misalignments (for edited spectra).

    TODO: replace with better implementation
    """
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

    def forward(self, data, water=None, **kwargs):
        """
        Applies signal perturbations to the MRS data.

        Args:
            data: Input MRS data (NiftiMRS object).
            water: Optional water reference data (NiftiMRS object).
            **kwargs: Additional arguments.
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
            time = np.arange(data.shape[3]) * data.dwelltime
            freq_noise = np.random.uniform(self.freq_low, self.freq_high,
                                           size=data.shape[:3] + (1,) + data.shape[4:])
            data[:] *= np.exp(1j * time * freq_noise * 2 * np.pi)

        if self.misalign:
            # TODO: implement misalignment for edited spectra
            raise NotImplementedError("Misalignment not implemented yet.")

        # TODO: update processing provenance with specific perturbations applied

        return data, water

    # def update_processing_prov(self, data, water=None, **kwargs):
    #     """
    #     Placeholder for updating processing provenance. Makes sure the generic message is not added.
    #     While the actual update is handled in the forward method to add the specific perturbations applied.
    #     """
    #     pass
