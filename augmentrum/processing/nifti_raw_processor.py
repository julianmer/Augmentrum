####################################################################################################
#                                      raw_processor.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Implements NIfTI-based processing for raw MRS data using FSL-MRS functions              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

# own
from augmentrum.processing.utils import safe_squeeze
from augmentrum.core.base_module import BaseModule
from augmentrum.core import Backend


#**************************************************************************************************#
#                                     Class NIfTI_RawProcessor                                     #
#**************************************************************************************************#
#                                                                                                  #
# Processes raw MRS data using FSL-MRS functions on NIfTI-MRS objects.                             #
#                                                                                                  #
#**************************************************************************************************#
class NIfTI_RawProcessor(BaseModule):
    """
    Processes raw MRS data using FSL-MRS functions on NIfTI-MRS objects.

    This module operates on the NIFTI_LIST backend, processing each NIFTI_MRS
    object individually using FSL-MRS processing functions.

    Logging is automatic via BaseModule (only if not volatile).
    """

    SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

    def __init__(self, conj=True, coil=True, align=True, remove_outliers=True, average=True,
                 ecc=True, truncate=False, remove_water=False, shift_ref=True, phase_correct=True,
                 coil_method='fsl-mrs', registration_method='fsl-mrs', remove_method='fsl-mrs',
                 average_method='fsl-mrs', ecc_method='own', water_removal_method='fsl-mrs',
                 shift_ref_method='fsl-mrs', phase_correct_method='fsl-mrs', **kwargs):
        """
        Initializes the processor with specified steps. Make sure custom/added steps use some
        form of update processing provenance, for clarification see:
        augmentrum.processing.utils.update_processing_prov,
        fsl_mrs.utils.preproc.nifti_mrs_proc.update_processing_prov

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

            coil_method (str): Coil combination method ('fsl-mrs' or 'adaptive').
            registration_method (str): Registration method.
            remove_method (str): Outlier removal method.
            average_method (str): Averaging method.
            ecc_method (str): Eddy current correction method ('fsl-mrs' or 'own').
            water_removal_method (str): Water removal method.
            shift_ref_method (str): Frequency shifting method.
            phase_correct_method (str): Phase correction method.
        """
        super().__init__()

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

        self.coil_method = coil_method
        self.registration_method = registration_method
        self.remove_method = remove_method
        self.average_method = average_method
        self.ecc_method = ecc_method
        self.water_removal_method = water_removal_method
        self.shift_ref_method = shift_ref_method
        self.phase_correct_method = phase_correct_method

    def process_nifti_list(self, data_list, water_list=None, report=None, **kwargs):
        """
        Processes lists of NIfTI-MRS data with the specified steps.

        Each NIFTI_MRS object in the list is processed individually using
        FSL-MRS processing functions.

        Args:
            data_list: List of metabolite MRS data (NIFTI_MRS objects).
            water_list: List of water reference MRS data (NIFTI_MRS objects), optional.
            report: Optional report object for logging processing steps.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (processed_data_list, processed_water_list)
        """
        processed_data = []
        processed_water = []

        for i, data_met in enumerate(data_list):
            data_wat = water_list[i] if water_list is not None else None

            # Process this subject
            proc_met, proc_wat = self._process_single(data_met, data_wat, report=report, **kwargs)

            processed_data.append(proc_met)
            if water_list is not None:
                processed_water.append(proc_wat if proc_wat is not None else data_wat)

        return processed_data, (processed_water if water_list is not None else None)

    def _process_single(self, data_met, data_wat=None, report=None, **kwargs):
        """
        Processes a single subject's MRS data with the specified steps.

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
            data_wat = proc.conjugate(data_wat) if data_wat is not None else None

        if self.coil: # coil combination
            data_met, data_wat = self.coil_combine(data_met, data_wat,
                                                   method=self.coil_method, report=report)

        if self.align:  # registration
            data_met, data_wat = self.registration(data_met, data_wat,
                                                   method=self.registration_method, report=report)

        if self.remove_outliers:  # remove outlier averages
            data_met, data_wat = self.remove_unlike(data_met, data_wat,
                                                      method=self.remove_method, report=report)

        if self.average:  # averaging
            data_met, data_wat = self.combine_averages(data_met, data_wat,
                                                       method=self.average_method, report=report)

        if 'DIM_DYN' in data_met.dim_tags or 'DIM_COIL' in data_met.dim_tags:
            data_met = safe_squeeze(data_met)
        if data_wat is not None and ('DIM_DYN' in data_wat.dim_tags or 'DIM_COIL' in data_wat.dim_tags):
            data_wat = safe_squeeze(data_wat)

        if self.ecc:  # eddy current correction
            data_met, data_wat = self.eddy_current_correction(data_met, data_wat,
                                                              method=self.ecc_method, report=report)

        if self.truncate:  # truncation or zero-filling
            data_met = proc.truncate_or_pad(data_met, -1, 'first', report=report)   # truncation
            data_wat = proc.truncate_or_pad(data_wat, -1, 'first') if data_wat is not None else None

        if self.remove_water:   # unsuppressed water removal
            data_met, data_wat = self.water_removal(data_met, data_wat,
                                                    method=self.water_removal_method, report=report)

        if self.shift_ref:  # frequency shift to reference
            data_met, data_wat = self.shift_to_reference(data_met, data_wat,
                                                         method=self.shift_ref_method, report=report)

        if self.phase_correct:   # phase correction
            data_met, data_wat = self.phase_correction(data_met, data_wat,
                                                       method=self.phase_correct_method, report=report)

        return data_met, data_wat

    def coil_combine(self, data_met, data_wat=None, method='fsl-mrs', report=None):
        """
        Performs coil combination on the MRS data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Coil combination method ('fsl-mrs' or 'adaptive').
            report: Optional report object for logging processing steps.

        Returns:
            Coil combined metabolite and water MRS data (NiftiMRS objects).
        """
        if 'DIM_COIL' in getattr(data_met, 'dim_tags', []) and data_met.shape[data_met.dim_position('DIM_COIL')] > 1:
            if method == 'fsl-mrs':
                if data_wat is not None and 'DIM_DYN' in getattr(data_wat, 'dim_tags', []):
                    avg_ref = proc.average(data_wat, 'DIM_DYN')
                else:
                    avg_ref = data_wat
                noise, covariance, no_prewhiten = self._estimate_noise_cov(data_met)
                data_met = proc.coilcombine(data_met, reference=avg_ref, report=report, noise=noise,
                                            covariance=covariance, no_prewhiten=no_prewhiten)
                data_wat = proc.coilcombine(data_wat, reference=avg_ref, noise=noise,
                                            covariance=covariance, no_prewhiten=no_prewhiten) if data_wat is not None else None
            elif method == 'adaptive':
                from augmentrum.processing.utils import own_nifti_coil_combination_adaptive
                data_met, data_wat = own_nifti_coil_combination_adaptive(data_met, data_wat, report=report)
            else:
                raise ValueError(f"Unknown coil combination method: {method}")
        return data_met, data_wat

    def registration(self, data_met, data_wat=None, method='fsl-mrs', report=None):
        """
        Performs registration on the MRS data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Registration method ('fsl-mrs', ...).
            report: Optional report object for logging processing steps.

        Returns:
            Registered metabolite and water MRS data (NiftiMRS objects).
        """
        if method == 'fsl-mrs':
            if 'DIM_DYN' in getattr(data_met, 'dim_tags', []) and data_met.shape[data_met.dim_position('DIM_DYN')] > 1:
                # squeeze coil dim if still present
                if 'DIM_COIL' in data_met.dim_tags:
                    data_met = data_met.copy(remove_dim='DIM_COIL')
                data_met = proc.align(data_met, 'DIM_DYN', ppmlim=(0.2, 4.2), report=report)
            if data_wat is not None and 'DIM_DYN' in getattr(data_wat, 'dim_tags', []):
                if data_wat is not None and 'DIM_COIL' in data_wat.dim_tags:
                    data_wat = data_wat.copy(remove_dim='DIM_COIL')
                data_wat = proc.align(data_wat, 'DIM_DYN', ppmlim=(0, 8))
        else:
            raise ValueError(f"Unknown registration method: {method}")
        return data_met, data_wat

    def remove_unlike(self, data_met, data_wat=None, method='fsl-mrs', report=None):
        """
        Removes outlier averages from the MRS data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Outlier removal method ('fsl-mrs', ...).
            report: Optional report object for logging processing steps.

        Returns:
            MRS data with outliers removed (NiftiMRS objects).
        """
        if 'DIM_DYN' in getattr(data_met, 'dim_tags', []) and data_met.shape[data_met.dim_position('DIM_DYN')] > 1:
            if method == 'fsl-mrs':
                data_met, _ = proc.remove_unlike(data_met, report=report)  # remove outlier averages
            else:
                raise ValueError(f"Unknown outlier removal method: {method}")
        return data_met, data_wat

    def combine_averages(self, data_met, data_wat=None, method='fsl-mrs', report=None):
        """
        Combines averages in the MRS data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Averaging method ('fsl-mrs', ...).
            report: Optional report object for logging processing steps.

        Returns:
            Averaged metabolite and water MRS data (NiftiMRS objects).
        """
        if 'DIM_DYN' in getattr(data_met, 'dim_tags', []):
            if data_met.shape[data_met.dim_position('DIM_DYN')] > 1:
                data_met = proc.average(data_met, 'DIM_DYN', report=report)  # combine averages
        if data_wat is not None and 'DIM_DYN' in getattr(data_wat, 'dim_tags', []):
            if data_wat is not None and data_wat.shape[data_wat.dim_position('DIM_DYN')] > 1:
                data_wat = proc.average(data_wat, 'DIM_DYN')
        return data_met, data_wat

    def eddy_current_correction(self, data_met, data_wat=None, method='own', report=None):
        """
        Performs eddy current correction on the MRS data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Eddy current correction method ('fsl-mrs' or 'own').
            report: Optional report object for logging processing steps.

        Returns:
            None. The metabolite data is modified in place.
        """
        if self.ecc_method == 'fsl-mrs':
            data_met = proc.ecc(data_met, data_wat if data_wat is not None else data_met,
                                report=report)  # eddy current correction
            data_wat = proc.ecc(data_wat, data_wat) if data_wat is not None else None
        elif self.ecc_method == 'own':
            from augmentrum.processing.utils import own_nifti_ecc
            data_met = own_nifti_ecc(data_met, data_wat if data_wat is not None else data_met, report=report)
            data_wat = own_nifti_ecc(data_wat, data_wat) if data_wat is not None else None
        else:
            raise ValueError(f"Unknown ECC method: {self.ecc_method}")
        return data_met, data_wat

    def water_removal(self, data_met, data_wat=None, method='fsl-mrs', report=None):
        """
        Removes residual water peak from the MRS data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Water removal method ('fsl-mrs', ...).
            report: Optional report object for logging processing steps.

        Returns:
            MRS data with water peak removed (NiftiMRS object).
        """
        if method == 'fsl-mrs':
            data_met = proc.remove_peaks(data_met, [-0.15, 0.15], limit_units='ppm',
                                         report=report)  # remove residual water
        else:
            raise ValueError(f"Unknown water removal method: {method}")
        return data_met, data_wat

    def shift_to_reference(self, data_met, data_wat=None, method='fsl-mrs', report=None):
        """
        Shifts the MRS data to a reference peak.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Frequency shifting method ('fsl-mrs', ...).
            report: Optional report object for logging processing steps.

        Returns:
            Frequency shifted metabolite MRS data (NiftiMRS object).
        """
        if method == 'fsl-mrs':
            data_met = proc.shift_to_reference(data_met, 3.027, (2.9, 3.1), report=report)  # shift to ref
        else:
            raise ValueError(f"Unknown frequency shifting method: {method}")
        return data_met, data_wat

    def phase_correction(self, data_met, data_wat=None, method='fsl-mrs', report=None):
        """
        Performs phase correction on the MRS data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Phase correction method ('fsl-mrs', ...).
            report: Optional report object for logging processing steps.

        Returns:
            Phase corrected metabolite and water MRS data (NiftiMRS objects).
        """
        if method == 'fsl-mrs':
            data_met = proc.phase_correct(data_met, (2.9, 3.1), report=report)  # phase corretion
            data_wat = proc.phase_correct(data_wat, (4.55, 4.7), hlsvd=False) if data_wat is not None else None
        else:
            raise ValueError(f"Unknown phase correction method: {method}")
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