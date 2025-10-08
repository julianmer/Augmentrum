####################################################################################################
#                                      raw_processor.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Implements for fast processing of selected coil/average subsets.                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

# own
from augmentrum.processing.utils import safe_squeeze


#**************************************************************************************************#
#                                       Class RawProcessor                                         #
#**************************************************************************************************#
#                                                                                                  #
# Processes raw MRS data with steps like coil combination, alignment, outlier removal,             #
# averaging, eddy current correction, truncation, water removal, frequency shifting, and phase     #
# correction.                                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class RawProcessor:
    """
    Processes raw MRS data with steps like coil combination, alignment, outlier removal,
    averaging, eddy current correction, truncation, water removal, frequency shifting, and phase
    correction.
    """

    def __init__(self, conj=True, coil=True, align=True, remove_outliers=True, average=True,
                 ecc=True, truncate=False, remove_water=False, shift_ref=True, phase_correct=True):
        """
        Initializes the processor with specified steps.

        Args:
            conj (bool): Whether to conjugate the data.
            coil (bool): Whether to perform coil combination.
            align (bool): Whether to align dynamics.
            remove_outliers (bool): Whether to remove outlier averages.
            average (bool): Whether to average dynamics.
            ecc (bool): Whether to perform eddy current correction.
            truncate (bool): Whether to truncate the FID.
            remove_water (bool): Whether to remove residual water peak.
            shift_ref (bool): Whether to shift spectrum to reference peak.
            phase_correct (bool): Whether to perform phase correction.
        """
        self.conj = conj
        self.coil = coil
        self.align = align  # after water removal the slowest step
        self.remove_outliers = remove_outliers
        self.average = average
        self.ecc = ecc
        self.truncate = truncate
        self.remove_water = remove_water   # very slow at the moment
        self.shift_ref = shift_ref
        self.phase_correct = phase_correct

        self.coil_method = 'adaptive'  # 'fsl-mrs' or 'adaptive'

    def __call__(self, data_met, data_wat=None, report=None, **kwargs):
        """
        Processes the MRS data with the specified steps.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            report: Optional report object for logging processing steps.
            **kwargs: Additional arguments.

        Returns:
            Processed metabolite and water MRS data (NiftiMRS objects).
        """
        if self.conj: # conjugate if needed
            data_met = proc.conjugate(data_met)
            data_wat = proc.conjugate(data_wat)

        if (self.coil and 'DIM_COIL' in getattr(data_met, 'dim_tags', [])
            and data_met.shape[data_met.dim_position('DIM_COIL')] > 1): # coil combination
            if self.coil_method == 'fsl-mrs':
                avg_ref = proc.average(data_wat, 'DIM_DYN')
                noise, covariance, no_prewhiten = self._estimate_noise_cov(data_met)
                data_met = proc.coilcombine(data_met, reference=avg_ref, report=report, noise=noise,
                                            covariance=covariance, no_prewhiten=no_prewhiten)
                data_wat = proc.coilcombine(data_wat, reference=avg_ref, noise=noise,
                                            covariance=covariance, no_prewhiten=no_prewhiten)
            elif self.coil_method == 'adaptive':
                from augmentrum.processing.utils import own_nifti_coil_combination_adaptive
                data_met, data_wat = own_nifti_coil_combination_adaptive(data_met, data_wat, report=report)
            else:
                raise ValueError(f"Unknown coil combination method: {self.coil_method}")

        if (self.align and 'DIM_DYN' in getattr(data_met, 'dim_tags', [])
            and data_met.shape[data_met.dim_position('DIM_DYN')] > 1):
            # squeeze coil dim if still present
            if 'DIM_COIL' in data_met.dim_tags: data_met = data_met.copy(remove_dim='DIM_COIL')
            if 'DIM_COIL' in data_wat.dim_tags: data_wat = data_wat.copy(remove_dim='DIM_COIL')

            data_met = proc.align(data_met, 'DIM_DYN', ppmlim=(0.2, 4.2), report=report)  # align phases
            data_wat = proc.align(data_wat, 'DIM_DYN', ppmlim=(0, 8))

        if (self.remove_outliers and 'DIM_DYN' in getattr(data_met, 'dim_tags', [])
            and data_met.shape[data_met.dim_position('DIM_DYN')] > 1):
            data_met, _ = proc.remove_unlike(data_met, report=report)  # remove outlier avergaes

        if self.average and 'DIM_DYN' in getattr(data_met, 'dim_tags', []):
            if data_met.shape[data_met.dim_position('DIM_DYN')] > 1:
                data_met = proc.average(data_met, 'DIM_DYN', report=report)  # combine averages
            if data_wat.shape[data_wat.dim_position('DIM_DYN')] > 1:
                data_wat = proc.average(data_wat, 'DIM_DYN')

        if 'DIM_DYN' in data_met.dim_tags: data_met = safe_squeeze(data_met)
        if 'DIM_DYN' in data_wat.dim_tags: data_wat = safe_squeeze(data_wat)

        if self.ecc:
            data_met = proc.ecc(data_met, data_wat, report=report)  # eddy current correction
            data_wat = proc.ecc(data_wat, data_wat)

        if self.truncate:
            data_met = proc.truncate_or_pad(data_met, -1, 'first', report=report)   # truncation
            data_wat = proc.truncate_or_pad(data_wat, -1, 'first')

        if self.remove_water:
            data_met = proc.remove_peaks(data_met, [-0.15, -0.15], limit_units='ppm',
                                         report=report)  # remove residual water

        if self.shift_ref:
            data_met = proc.shift_to_reference(data_met, 3.027, (2.9, 3.1), report=report)  # shift to ref

        if self.phase_correct:
            data_met = proc.phase_correct(data_met, (2.9, 3.1), report=report)  # phase corretion
            data_wat = proc.phase_correct(data_wat, (4.55, 4.7), hlsvd=False)

        return data_met, data_wat

    def _estimate_noise_cov(self, data, noise=None, no_prewhiten=False):
        """
        Estimates noise covariance matrix for coil combination.
        """
        from fsl_mrs.utils.preproc.combine import estimate_noise_cov, CovarianceEstimationError

        stacked_data = [dd for dd, _ in
                        data.iterate_over_dims(dim='DIM_COIL', iterate_over_space=True, reduce_dim_index=True)]
        try:
            covariance = estimate_noise_cov(np.asarray(stacked_data))
        except CovarianceEstimationError:
            no_prewhiten = True
            covariance = None
        return noise, covariance, no_prewhiten