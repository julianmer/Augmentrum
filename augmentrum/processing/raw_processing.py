####################################################################################################
#                                       raw_processing.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Processing of raw (uncombined, unaveraged) MRS data. NIfTI_RawProcessor wraps FSL-MRS   #
#          functions on NIfTI-MRS objects; RawProcessor is its tensor twin, running the same       #
#          pipeline batched on any array backend with gradients through the signal path.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

from nifti_mrs_plus import ops

# own
from augmentrum.processing.utils import (safe_squeeze, fid_to_spec,
                                         ppm_shift_axis, ppm_window, move_axis)
from augmentrum.processing.domain import Domain
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
    DOMAIN = Domain(spectral='time')

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


#**************************************************************************************************#
#                                        Class RawProcessor                                        #
#**************************************************************************************************#
#                                                                                                  #
# Tensor twin of NIfTI_RawProcessor: the same raw-data pipeline, batched on any array backend.     #
#                                                                                                  #
#**************************************************************************************************#
class RawProcessor(BaseModule):
    """
    Tensor twin of NIfTI_RawProcessor: the same raw-data pipeline, batched on any array backend.

    Every step splits into a detached estimate (mirroring the FSL-MRS formulas)
    and a differentiable application (backend-native multiply and sum), so
    results track the NIfTI path closely while gradients flow through the
    signal path on torch / jax / tensorflow, and the whole batch is processed
    in vectorized sweeps instead of per-FID Python loops.

    Deliberate differences from the NIfTI path:
      * outlier removal keeps the dynamic axis and hands averaging a
        zero-weight mask — identical results after averaging, which is why it
        requires ``average=True``;
      * alignment with registration_method='own' swaps the per-transient
        Powell search for a closed-form phase and a vectorized pattern descent
        over the frequency shift — equal objective values, but on noisy data
        the solvers settle in micro-minima a fraction of a Hz apart (the
        default 'fsl-mrs' method runs the same Powell search as FSL-MRS and
        matches it);
      * water removal takes the top Hankel components from a truncated
        Lanczos SVD (hlsvdpropy's sparse path) instead of a dense
        decomposition, so it matches to floating-point tolerance rather than
        bit-exactly.

    The water reference is assumed to share the metabolite data's dimension
    layout, and ppm referencing assumes 1H data.
    """

    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)
    DOMAIN = Domain(spectral='time')

    _dropped_tags = frozenset()

    def __init__(self, conj=True, coil=True, align=True, remove_outliers=True, average=True,
                 ecc=True, truncate=False, remove_water=False, shift_ref=True, phase_correct=True,
                 coil_method='fsl-mrs', registration_method='fsl-mrs', remove_method='fsl-mrs',
                 average_method='fsl-mrs', ecc_method='own', water_removal_method='fsl-mrs',
                 shift_ref_method='fsl-mrs', phase_correct_method='fsl-mrs', **kwargs):
        """
        Initializes the processor; flags and defaults match NIfTI_RawProcessor,
        so the two are drop-in interchangeable in a pipeline.

        Args:
            conj (bool): Whether to conjugate the data.
            coil (bool): Whether to perform coil combination.
            align (bool): Whether to align dynamics.
            remove_outliers (bool): Whether to mask outlier averages.
            average (bool): Whether to average dynamics.
            ecc (bool): Whether to perform eddy current correction.
            truncate (bool): Whether to truncate the FID.
            remove_water (bool): Whether to remove residual water peak.
            shift_ref (bool): Whether to shift spectrum to reference peak.
            phase_correct (bool): Whether to perform phase correction.

            coil_method (str): Coil combination method ('fsl-mrs').
            registration_method (str): Registration method ('fsl-mrs' for the
                FSL-MRS Powell search, 'own' for the fast vectorized search).
            remove_method (str): Outlier removal method ('fsl-mrs').
            average_method (str): Averaging method ('fsl-mrs').
            ecc_method (str): Eddy current correction method ('fsl-mrs' or 'own').
            water_removal_method (str): Water removal method ('fsl-mrs').
            shift_ref_method (str): Frequency shifting method ('fsl-mrs').
            phase_correct_method (str): Phase correction method ('fsl-mrs').
        """
        super().__init__()

        self.conj = conj
        self.coil = coil
        self.align = align
        self.remove_outliers = remove_outliers
        self.average = average
        self.ecc = ecc
        self.truncate = truncate
        self.remove_water = remove_water
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

    def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
        """
        Runs the processing pipeline on a batched tensor, spectral axis last.

        Args:
            data_array: Metabolite data, (batch, X, Y, Z, higher dims..., T).
            water_array: Water reference in the untransposed NIfTI layout
                (batch, X, Y, Z, T, higher dims...), optional.
            backend: The array backend the tensors live on.
            **kwargs: Injected metadata; sw_hz, sf_mhz and dim_tags are used.

        Returns:
            Tuple of processed (data_array, water_array), collapsed dimensions
            removed (the water is returned in its untransposed layout).
        """
        sw_hz = kwargs.get('sw_hz')
        sf_mhz = kwargs.get('sf_mhz')
        if sw_hz is None or sf_mhz is None:
            raise ValueError("RawProcessor needs 'sw_hz' and 'sf_mhz' — provide them or process "
                             "data with header metadata attached.")

        met, wat = data_array, water_array
        rank = len(ops.shape(met))
        tags = [t for t in (kwargs.get('dim_tags') or []) if t][:max(0, rank - 5)]
        self._dropped_tags = set()

        if wat is not None and len(ops.shape(wat)) > 5:
            wat = move_axis(wat, self.SPECTRAL_AXIS, -1)

        if self.conj:
            met = ops.complex_from(ops.real(met), -ops.imag(met))
            wat = ops.complex_from(ops.real(wat), -ops.imag(wat)) if wat is not None else None

        if self.coil:
            met, wat, tags = self.coil_combine(met, wat, tags)

        if self.align:
            met, wat, tags = self.registration(met, wat, tags, sw_hz, sf_mhz)

        mask = self.outlier_mask(met, tags) if self.remove_outliers else None

        if self.average:
            met, wat, tags = self.combine_averages(met, wat, tags, mask)

        if 'DIM_DYN' in tags or 'DIM_COIL' in tags:
            met, wat, tags = self._squeeze_singletons(met, wat, tags)

        if self.ecc:
            met, wat = self.eddy_current_correction(met, wat)

        if self.truncate:
            met = met[..., 1:]
            wat = wat[..., 1:] if wat is not None else None

        if self.remove_water:
            met = self.water_removal(met, sw_hz, sf_mhz)

        if self.shift_ref:
            met = self.shift_to_reference(met, sw_hz, sf_mhz)

        if self.phase_correct:
            met, wat = self.phase_correction(met, wat, sw_hz, sf_mhz)

        if wat is not None and len(ops.shape(wat)) > 5:
            wat = move_axis(wat, -1, self.SPECTRAL_AXIS)
        return met, wat

    def coil_combine(self, met, wat, tags):
        """
        wSVD coil combination; weights from the water reference when present.

        Estimates per-subject noise covariance from the last tenth of every
        FID (prewhitening is silently disabled per subject when there are too
        few samples, as in FSL-MRS), derives the wSVD weights, and combines both
        metabolite and water data as a weighted sum over the coil axis.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor in the same layout, or None.
            tags: Higher-dimension tags, mutated in place.

        Returns:
            Tuple of (met, wat, tags) with the coil dimension collapsed.
        """
        if 'DIM_COIL' not in tags:
            return met, wat, tags
        coil_axis = 4 + tags.index('DIM_COIL')
        n_coil = ops.shape(met)[coil_axis]
        if n_coil <= 1:
            return met, wat, tags
        if self.coil_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor coil combination method: {self.coil_method}")

        n_time = ops.shape(met)[-1]
        n_batch = ops.shape(met)[0]

        # per-subject noise covariance and whitening, from the FID tails
        noise = ops.to_numpy(met[..., int(0.9 * n_time):])
        noise = np.moveaxis(noise, coil_axis, -1).reshape(n_batch, -1, n_coil)
        eye = np.eye(n_coil, dtype=np.complex128)
        cov = np.empty((n_batch, n_coil, n_coil), dtype=np.complex128)
        white = np.empty_like(cov)
        white_inv = np.empty_like(cov)
        for b, samples in enumerate(noise):
            if samples.shape[0] < 10 * n_coil:
                cov[b], white[b], white_inv[b] = eye, eye, eye      # prewhitening disabled
            else:
                cov[b] = np.cov(samples, rowvar=False)
                d, v = np.linalg.eigh(cov[b], UPLO='U')
                white[b] = v @ np.diag(1 / np.sqrt(d))
                white_inv[b] = np.linalg.inv(white[b])

        met_tc = move_axis(met, coil_axis, -1)                     # (..., T, C)
        wat_tc = move_axis(wat, coil_axis, -1) if wat is not None else None
        others = [t for t in tags if t != 'DIM_COIL']

        if wat_tc is not None:
            ref_tc = wat_tc
            if 'DIM_DYN' in others:
                ref_tc = ops.mean(ref_tc, axis=4 + others.index('DIM_DYN'), keepdims=True)
            source_tc, with_reference = ref_tc, True
        else:
            source_tc, with_reference = met_tc, False

        lead = len(ops.shape(source_tc)) - 2
        shape = (n_batch,) + (1,) * (lead - 1) + (n_coil, n_coil)
        source = ops.to_numpy(source_tc).astype(np.complex128)
        _, _, vh = np.linalg.svd(source @ white.reshape(shape), full_matrices=False)
        weights = self._wsvd_weights(vh[..., 0, :], white.reshape(shape),
                                     white_inv.reshape(shape), cov.reshape(shape),
                                     with_reference)

        met = ops.sum(met_tc * ops.match_backend(weights[..., None, :], met_tc), axis=-1)
        if wat_tc is not None:
            wat = ops.sum(wat_tc * ops.match_backend(weights[..., None, :], wat_tc), axis=-1)

        tags.remove('DIM_COIL')
        self._dropped_tags.add('DIM_COIL')
        return met, wat, tags

    def registration(self, met, wat, tags, sw_hz, sf_mhz):
        """
        Aligns dynamics in phase and frequency (spectral registration).

        A leftover coil dimension is reduced to its first element the way the
        FSL-MRS path's copy(remove_dim='DIM_COIL') does. The metabolite data
        aligns within (0.2, 4.2) ppm, the water within (0, 8) ppm. Method
        'fsl-mrs' reproduces the FSL-MRS Powell search per transient; 'own'
        is the vectorized pattern search (much faster, results equal in
        objective value but not identical on noisy data).

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor in the same layout, or None.
            tags: Higher-dimension tags, mutated in place.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.

        Returns:
            Tuple of (met, wat, tags).
        """
        if self.registration_method not in ('fsl-mrs', 'own'):
            raise ValueError(f"Unknown tensor registration method: {self.registration_method}")
        if 'DIM_DYN' not in tags:
            return met, wat, tags
        if ops.shape(met)[4 + tags.index('DIM_DYN')] <= 1:
            return met, wat, tags

        if 'DIM_COIL' in tags:
            coil_axis = 4 + tags.index('DIM_COIL')
            met = met[(slice(None),) * coil_axis + (0,)]
            wat = wat[(slice(None),) * coil_axis + (0,)] if wat is not None else None
            tags.remove('DIM_COIL')
            self._dropped_tags.add('DIM_COIL')

        dyn_axis = 4 + tags.index('DIM_DYN')
        met = self._apply_alignment(met, dyn_axis, sw_hz, sf_mhz, (0.2, 4.2))
        if wat is not None and ops.shape(wat)[dyn_axis] > 1:
            wat = self._apply_alignment(wat, dyn_axis, sw_hz, sf_mhz, (0, 8))
        return met, wat, tags

    def _apply_alignment(self, data, dyn_axis, sw_hz, sf_mhz, ppmlim):
        """
        Estimates alignment parameters and applies them as one phasor.
        """
        arr = move_axis(data, dyn_axis, -2)
        phi, eps = self._align_params(ops.to_numpy(arr), sw_hz, sf_mhz, ppmlim,
                                 method=self.registration_method)
        n = ops.shape(arr)[-1]
        t = np.linspace(1.0 / sw_hz, n / sw_hz, n)
        phasor = np.exp(-1j * phi[..., None] - 2j * np.pi * t * eps[..., None])
        return move_axis(arr * ops.match_backend(phasor, arr), -2, dyn_axis)

    def outlier_mask(self, met, tags):
        """
        Keep-mask over dynamics, the tensor form of FSL-MRS remove_unlike.

        A batched tensor cannot go ragged, so outliers are masked rather than
        dropped and the mask is consumed by averaging — which must therefore
        be enabled. Results equal the NIfTI path's after that average.

        Args:
            met: Metabolite tensor, spectral axis last.
            tags: Higher-dimension tags.

        Returns:
            Boolean keep mask over the dynamic axis, or None when no
            multi-dynamic dimension exists.
        """
        if self.remove_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor outlier removal method: {self.remove_method}")
        if 'DIM_DYN' not in tags:
            return None
        if ops.shape(met)[4 + tags.index('DIM_DYN')] <= 1:
            return None
        if ops.shape(met)[1:4] != (1, 1, 1) or len(tags) != 1:
            raise ValueError('Outlier removal is only specified for SVS data with a single '
                             'dynamic dimension (as in FSL-MRS remove_unlike).')
        if not self.average:
            raise NotImplementedError('The tensor path expresses outlier removal as a zero-weight '
                                      'mask consumed by averaging; enable average=True or use '
                                      'NIfTI_RawProcessor.')
        return self._unlike_mask(ops.to_numpy(met))

    def combine_averages(self, met, wat, tags, mask=None):
        """
        Averages dynamics; a keep-mask turns this into a weighted mean.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor in the same layout, or None.
            tags: Higher-dimension tags, mutated in place.
            mask: Optional boolean keep mask over the dynamic axis.

        Returns:
            Tuple of (met, wat, tags) with the dynamic dimension collapsed.
        """
        if self.average_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor averaging method: {self.average_method}")
        if 'DIM_DYN' not in tags:
            return met, wat, tags
        dyn_axis = 4 + tags.index('DIM_DYN')
        if ops.shape(met)[dyn_axis] <= 1:
            return met, wat, tags

        if mask is None:
            combined = ops.mean(met, axis=dyn_axis)
        else:
            weights = mask.astype(np.float64) / mask.sum(axis=-1, keepdims=True)
            combined = ops.sum(met * ops.match_backend(weights[..., None], met), axis=dyn_axis)
        if wat is not None and ops.shape(wat)[dyn_axis] > 1:
            wat = ops.mean(wat, axis=dyn_axis)
        elif wat is not None:
            wat = wat[(slice(None),) * dyn_axis + (0,)]

        tags.remove('DIM_DYN')
        self._dropped_tags.add('DIM_DYN')
        return combined, wat, tags

    def _squeeze_singletons(self, met, wat, tags):
        """
        Drops singleton higher dimensions, the tensor form of safe_squeeze.
        """
        for i in reversed(range(len(tags))):
            axis = 4 + i
            if ops.shape(met)[axis] == 1:
                met = met[(slice(None),) * axis + (0,)]
                if wat is not None and ops.shape(wat)[axis] == 1:
                    wat = wat[(slice(None),) * axis + (0,)]
                self._dropped_tags.add(tags[i])
                tags.pop(i)
        return met, wat, tags

    def eddy_current_correction(self, met, wat):
        """
        Eddy current correction against the water reference (or the data itself).

        'own' mirrors own_nifti_ecc: the Gaussian-smoothed unwrapped reference
        phase is removed. 'fsl-mrs' mirrors preproc.eddy_correct: the raw
        reference phase is removed. The water corrects against itself.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor in the same layout, or None.

        Returns:
            Tuple of corrected (met, wat).
        """
        ref = ops.to_numpy(wat if wat is not None else met)
        if self.ecc_method == 'own':
            phasor = np.exp(-1j * self._ecc_phase(ref))
        elif self.ecc_method == 'fsl-mrs':
            phasor = np.exp(-1j * np.angle(ref))
        else:
            raise ValueError(f"Unknown ECC method: {self.ecc_method}")
        met = met * ops.match_backend(phasor, met)
        wat = wat * ops.match_backend(phasor, wat) if wat is not None else None
        return met, wat

    def water_removal(self, met, sw_hz, sf_mhz):
        """
        Removes the residual water peak, the tensor form of HLSVD.

        The top Hankel components come from a truncated Lanczos SVD (the
        algorithm of hlsvdpropy's sparse path — a dense decomposition would
        dominate the whole pipeline's runtime); the water model is invariant
        to the basis of that subspace, so this matches the dense reference.
        The modeled signal inside (-0.15, 0.15) ppm is subtracted
        differentiably.

        Args:
            met: Metabolite tensor, spectral axis last.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.

        Returns:
            Metabolite tensor with the water model subtracted.
        """
        if self.water_removal_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor water removal method: {self.water_removal_method}")
        from scipy.sparse.linalg import svds

        arr = ops.to_numpy(met).astype(np.complex128)
        n = arr.shape[-1]
        m = n // 2
        k = min(20, n - m - 1, m)
        flat = arr.reshape(-1, n)
        uk = np.empty((flat.shape[0], n - m, k), dtype=np.complex128)
        for i, fid in enumerate(flat):
            hankel = np.lib.stride_tricks.sliding_window_view(fid, m + 1)   # (n-m, m+1)
            uk[i] = svds(hankel, k=k)[0]
        uk = uk.reshape(arr.shape[:-1] + (n - m, k))
        model = self._hlsvd_water_model(uk, arr, sw_hz, sf_mhz, (-0.15, 0.15), k=k)
        return met - ops.match_backend(model, met)

    def shift_to_reference(self, met, sw_hz, sf_mhz):
        """
        Shifts the peak found in (2.9, 3.1) ppm to the tCr reference 3.027 ppm.

        The peak is located on a four-fold zero-padded spectrum, exactly as
        FSL-MRS shiftToRef does, and the shift is applied as a phase ramp.

        Args:
            met: Metabolite tensor, spectral axis last.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.

        Returns:
            Frequency-shifted metabolite tensor.
        """
        if self.shift_ref_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor frequency shifting method: {self.shift_ref_method}")
        arr = ops.to_numpy(met)
        n = arr.shape[-1]
        spec = fid_to_spec(np.concatenate(
            [arr, np.zeros(arr.shape[:-1] + (3 * n,), dtype=arr.dtype)], axis=-1))
        first, last = ppm_window(4 * n, sw_hz, sf_mhz, (2.9, 3.1))
        peak = np.argmax(np.abs(spec[..., first:last]), axis=-1)
        shift_hz = (ppm_shift_axis(4 * n, sw_hz, sf_mhz)[first:last][peak] - 3.027) * sf_mhz
        t = np.linspace(0, n / sw_hz, n)                            # FSL freqshift time axis
        return met * ops.match_backend(np.exp(-2j * np.pi * t * shift_hz[..., None]), met)

    def phase_correction(self, met, wat, sw_hz, sf_mhz):
        """
        Zero-order phase correction on the maximum of a search window.

        The metabolite phase comes from (2.9, 3.1) ppm, the water phase from
        (4.55, 4.7) ppm, each on a four-fold zero-padded spectrum as in
        FSL-MRS phaseCorrect (without the optional HLSVD flattening).

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor in the same layout, or None.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.

        Returns:
            Tuple of phased (met, wat).
        """
        if self.phase_correct_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor phase correction method: {self.phase_correct_method}")

        def phase(data, window):
            arr = ops.to_numpy(data)
            n = arr.shape[-1]
            spec = fid_to_spec(np.concatenate(
                [arr, np.zeros(arr.shape[:-1] + (3 * n,), dtype=arr.dtype)], axis=-1))
            first, last = ppm_window(4 * n, sw_hz, sf_mhz, window)
            spec = spec[..., first:last]
            peak = np.argmax(np.abs(spec), axis=-1)
            angle = -np.angle(np.take_along_axis(spec, peak[..., None], axis=-1))
            return data * ops.match_backend(np.exp(1j * angle), data)

        met = phase(met, (2.9, 3.1))
        wat = phase(wat, (4.55, 4.7)) if wat is not None else None
        return met, wat

    #************************#
    #   detached estimates   #
    #************************#
    # The numerics of each step's parameter estimation, mirroring FSL-MRS /
    # suspect / hlsvdpropy verbatim — parity-tested, so resist beautifying.

    #******************#
    #   wsvd weights   #
    #******************#
    @staticmethod
    def _wsvd_weights(vh0, w, w_inv, cov, with_reference):
        """
        Per-coil combination weights of the wSVD method (Rodgers & Robson 2010).

        Derived so that "sum(X * weights, coil)" on the raw (unwhitened) data
        reproduces FSL-MRS combine_FIDs: 'svd_weights' when weights come from a
        reference, 'svd' when they come from the data itself.

        Args:
            vh0: First right singular vector of the whitened matrix, (..., C).
            w: Pre-whitening matrix, (..., C, C).
            w_inv: Its inverse, (..., C, C).
            cov: Coil covariance (identity when prewhitening is off), (..., C, C).
            with_reference: True for the reference-weight branch.

        Returns:
            Complex weights, (..., C).
        """
        amp = np.einsum('...j,...ji->...i', vh0, w_inv)
        rescale = np.linalg.norm(amp, axis=-1, keepdims=True) * amp[..., :1] / np.abs(amp[..., :1])
        if with_reference:
            scaled = np.conj(amp / rescale)
            alpha = np.einsum('...ij,...j->...i', np.linalg.inv(cov), scaled)
            return alpha * np.conj(rescale) * rescale
        return np.einsum('...ij,...j->...i', w, np.conj(vh0)) * rescale


    #***************#
    #   alignment   #
    #***************#
    @staticmethod
    def _align_params(fids, sw_hz, sf_mhz, ppmlim, niter=2, method='fsl-mrs'):
        """
        Phase and frequency shifts aligning transients, on the FSL-MRS objective.

        Minimizes || extract(e^{-i phi} shift(FID, eps)) - extract(target) || per
        transient, with the target fixed to the transient nearest the mean of the
        unaligned data (as phase_freq_align does across its iterations).

        Two estimators solve it. 'fsl-mrs' runs the same per-transient Powell
        search FSL-MRS runs, for full parity. 'own' folds the closed-form optimal
        phase into the objective and descends the frequency shift with a
        vectorized pattern search — much faster for many transients, but on noisy
        data the objective is locally rugged and the two solvers settle in
        micro-minima a fraction of a Hz apart. Both descend from zero: the
        objective has spurious far-away minima (a large shift pushes signal out
        of the ppm window), which any global search would happily fall into.

        Args:
            fids: Transients, (..., D, T) complex.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.
            ppmlim: ppm window of the comparison.
            niter: Refinement iterations against the fixed target.
            method: 'fsl-mrs' (Powell, parity) or 'own' (pattern search, speed).

        Returns:
            Accumulated (phi, eps) per transient, each (..., D).
        """
        fids = np.asarray(fids, dtype=np.complex128)
        n = fids.shape[-1]
        t = np.linspace(1.0 / sw_hz, n / sw_hz, n)                  # FSL timeAxis (starts at dwell)
        first, last = ppm_window(n, sw_hz, sf_mhz, ppmlim)

        avg = fids.mean(axis=-2, keepdims=True)
        pick = np.argmin(np.linalg.norm(fids - avg, axis=-1), axis=-1)
        target = np.take_along_axis(fids, pick[..., None, None], axis=-2)[..., 0, :]
        t_win = fid_to_spec(target)[..., first:last]
        normalization = np.linalg.norm(target, axis=-1)

        current = fids.copy()
        phi_total = np.zeros(fids.shape[:-1])
        eps_total = np.zeros(fids.shape[:-1])

        for _ in range(niter):
            if method == 'fsl-mrs':
                phi, eps = RawProcessor._powell_step(current, t, t_win, normalization, first, last)
            else:
                eps = RawProcessor._pattern_step(current, t, t_win, first, last)
                s_win = fid_to_spec(
                    current * np.exp(-2j * np.pi * t * eps[..., None]))[..., first:last]
                phi = np.angle(np.sum(s_win * np.conj(t_win[..., None, :]), axis=-1))
            current = np.exp(-1j * phi[..., None]) * current \
                * np.exp(-2j * np.pi * t * eps[..., None])
            phi_total += phi
            eps_total += eps

        return phi_total, eps_total


    @staticmethod
    def _powell_step(current, t, t_win, normalization, first, last):
        """One alignment pass with FSL-MRS's per-transient Powell search."""
        from scipy.optimize import minimize

        shape = current.shape[:-1]
        width = t_win.shape[-1]
        flat = current.reshape(-1, current.shape[-1])
        flat_win = np.broadcast_to(t_win[..., None, :], shape + (width,)).reshape(-1, width)
        flat_norm = np.broadcast_to(normalization[..., None], shape).reshape(-1)

        phi = np.zeros(flat.shape[0])
        eps = np.zeros(flat.shape[0])
        for i, (fid, win, norm) in enumerate(zip(flat, flat_win, flat_norm)):
            def cf(p):
                shifted = np.exp(-1j * p[0]) * fid * np.exp(-2j * np.pi * t * p[1])
                return np.linalg.norm((fid_to_spec(shifted)[first:last] - win) / norm)
            res = minimize(cf, np.array([0, 0]), method='Powell')
            phi[i], eps[i] = res.x
        return phi.reshape(shape), eps.reshape(shape)


    @staticmethod
    def _pattern_step(current, t, t_win, first, last, step_hz=1.0, tol_hz=0.01, max_iter=60):
        """One alignment pass of the vectorized frequency pattern search.

        Descends in steps of *step_hz*, halving on failure, until every transient
        converged below *tol_hz* — which sits far below any in-vivo linewidth.
        """
        def profile_cost(eps):
            s_win = fid_to_spec(
                current * np.exp(-2j * np.pi * t * eps[..., None]))[..., first:last]
            cross = np.abs(np.sum(s_win * np.conj(t_win[..., None, :]), axis=-1))
            return np.sum(np.abs(s_win) ** 2, axis=-1) - 2.0 * cross

        eps = np.zeros(current.shape[:-1])
        step = np.full(eps.shape, step_hz)
        cost = profile_cost(eps)
        for _ in range(max_iter):
            if np.all(step < tol_hz):
                break
            c_minus = profile_cost(eps - step)
            c_plus = profile_cost(eps + step)
            go_minus = (c_minus < cost) & (c_minus <= c_plus)
            go_plus = (c_plus < cost) & (c_plus < c_minus)
            eps = np.where(go_minus, eps - step, np.where(go_plus, eps + step, eps))
            cost = np.where(go_minus, c_minus, np.where(go_plus, c_plus, cost))
            step = np.where(go_minus | go_plus, step, step / 2)
        return eps


    #***********************#
    #   outlier detection   #
    #***********************#
    @staticmethod
    def _unlike_mask(fids, sdlimit=1.96, niter=2):
        """
        Keep-mask over transients, mirroring FSL-MRS identifyUnlikeFIDs (ppmlim=None).

        Args:
            fids: Transients, (..., D, T) complex.
            sdlimit: Exclusion limit in standard deviations.
            niter: Number of target-refinement iterations.

        Returns:
            Boolean keep mask, (..., D).
        """
        fids = np.asarray(fids, dtype=np.complex128)
        specs = fid_to_spec(fids)
        target = np.median(fids.real, axis=-2) + 1j * np.median(fids.imag, axis=-2)
        keep = np.ones(fids.shape[:-1], dtype=bool)
        for _ in range(niter):
            metric = np.linalg.norm(specs - fid_to_spec(target)[..., None, :], axis=-1)
            avg = metric.mean(axis=-1, keepdims=True)
            std = metric.std(axis=-1, keepdims=True)
            keep = np.abs(metric - avg) <= sdlimit * std
            masked_r = np.where(keep[..., None], fids.real, np.nan)
            masked_i = np.where(keep[..., None], fids.imag, np.nan)
            target = np.nanmedian(masked_r, axis=-2) + 1j * np.nanmedian(masked_i, axis=-2)
        return keep


    #***************#
    #   ecc phase   #
    #***************#
    @staticmethod
    def _ecc_phase(refs, width=32):
        """
        Smoothed unwrapped phase of the reference FIDs, (..., T).

        Mirrors suspect's sliding_gaussian as used by own_nifti_ecc: edge-padded
        with 10-point edge means, correlated with a normalized Gaussian window.
        """
        phase = np.unwrap(np.angle(np.asarray(refs)), axis=-1)
        window = np.exp(-np.linspace(-3, 3, width) ** 2)
        window /= window.sum()
        offset = (width - 1) // 2
        left = np.broadcast_to(phase[..., :10].mean(axis=-1, keepdims=True),
                               phase.shape[:-1] + (offset,))
        right = np.broadcast_to(phase[..., -10:].mean(axis=-1, keepdims=True),
                                phase.shape[:-1] + (width - 1 - offset,))
        padded = np.concatenate([left, phase, right], axis=-1)
        return np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1) @ window


    #*****************#
    #   hlsvd model   #
    #*****************#
    @staticmethod
    def _hlsvd_water_model(uk, fids, sw_hz, sf_mhz, limits, k=20):
        """
        Sum-of-Lorentzians model of the in-band components, per FID.

        Continues hlsvdpropy.hlsvdpro from the truncated left singular vectors of
        the Hankel matrix (shift-invariance least squares, poles, Vandermonde
        least squares) and reconstructs the components inside *limits* the way
        fsl_mrs.utils.preproc.remove._hlsvd does (limits in ppm, no shift).

        Args:
            uk: Top-k left singular vectors, (..., L, k).
            fids: The FIDs modeled, (..., T) complex.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.
            limits: ppm limits of components to keep.
            k: Number of singular components.

        Returns:
            The modeled water FID, (..., T) complex.
        """
        dwell = 1.0 / sw_hz
        fids = np.asarray(fids, dtype=np.complex128)
        n = fids.shape[-1]

        # complex matmul raises spurious fp-flag warnings on some BLAS builds
        # (Apple Accelerate); the results are finite and verified by parity tests
        with np.errstate(all='ignore'):
            zp = np.linalg.pinv(uk[..., :-1, :]) @ uk[..., 1:, :]   # (..., k, k)
        roots = np.linalg.eigvals(zp)                               # (..., k)

        flat_fids = fids.reshape(-1, n)
        flat_roots = roots.reshape(-1, k)
        flat_amps = np.zeros_like(flat_roots)
        for i, (fid, rts) in enumerate(zip(flat_fids, flat_roots)):
            zeta = np.vander(rts, N=n, increasing=True).T           # (T, k)
            flat_amps[i] = np.linalg.lstsq(zeta, fid, rcond=None)[0]
        amps = flat_amps.reshape(roots.shape)

        with np.errstate(divide='ignore', invalid='ignore'):
            freq_hz = np.arctan2(roots.imag, roots.real) / (2 * np.pi) / dwell
            damp_s = dwell / np.log(np.abs(roots))
            in_band = (freq_hz > limits[0] * sf_mhz) & (freq_hz < limits[1] * sf_mhz)

            t = np.linspace(0, dwell * (n - 1), n)
            phase = np.arctan2(amps.imag, amps.real)
            lines = np.abs(amps)[..., None] * np.exp(
                t / damp_s[..., None] + 2j * np.pi * (freq_hz[..., None] * t
                                                      + np.degrees(phase)[..., None] / 360.0))
        return np.sum(np.where(in_band[..., None], lines, 0), axis=-2)

    #****************#
    #   write-back   #
    #****************#
    def _spectral_axis_back(self, data_array):
        """
        Undo the spectral-axis move for the rank actually returned.

        The processor collapses higher dimensions, so the inverse permutation
        cannot assume the input's rank; the output always carries the batch
        and spatial axes first and the spectral axis last.
        """
        self._axis_rank = len(ops.shape(data_array))
        return super()._spectral_axis_back(data_array)

    def _output_dim_tags(self, source):
        """
        The source's tags minus the dimensions this run collapsed.
        """
        tags = [t for t in (source.dim_tags or []) if t and t not in self._dropped_tags]
        return (tags + [None, None, None])[:3]