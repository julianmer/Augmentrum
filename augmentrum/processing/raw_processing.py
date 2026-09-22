####################################################################################################
#                                       raw_processing.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Processing of raw (uncombined, unaveraged) MRS data. One RawProcessor, three engines:   #
#          the FSL-MRS reference per NIfTI object on the list backend, the same pipeline batched   #
#          and differentiable on every tensor backend, and a fully batched torch engine that       #
#          keeps every estimate on the data's device and consumes per-sample coil/transient masks. #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import warnings


from nifti_mrs_plus import ops

# own
from augmentrum.processing.utils import (safe_squeeze, fid_to_spec, ppm_reference,
                                         ppm_shift_axis, ppm_window, move_axis)
from augmentrum.processing.domain import Domain
from augmentrum.core.base_module import BaseModule
from augmentrum.core.pool import WATER_SHARED
from augmentrum.core import Backend


#*****************#
#   precision     #
#*****************#
def _complex_like(x):
    """*x* as a NumPy array of the complex dtype of its own precision: complex64 for single-
    precision data, complex128 for double (augmentrum.core.precision)."""
    x = np.asarray(x)
    return x.astype(np.result_type(x.dtype, np.complex64), copy=False)


#**************************************************************************************************#
#                                        Class RawProcessor                                        #
#**************************************************************************************************#
#                                                                                                  #
# Raw-data processing on any backend: the FSL-MRS reference per NIfTI object, or its batched       #
# differentiable twin on tensors — one module, the backend picks the engine.                       #
#                                                                                                  #
#**************************************************************************************************#

class RawProcessor(BaseModule):
    """
    Raw-data processing on any backend — one module, the backend picks the engine.

    On the NIFTI_LIST backend each subject runs through the FSL-MRS reference
    functions (bit-exact, shapes may differ per subject). On tensor backends
    the same pipeline runs batched: every step splits into a detached estimate
    (mirroring the FSL-MRS formulas) and a differentiable application
    (backend-native multiply and sum), so results track the reference to
    floating-point tolerance while gradients flow through the signal path on
    torch / jax / tensorflow. Parameters and defaults are one set; the few
    per-engine methods are named below.

    Deliberate differences of the tensor engine:
      * outlier removal keeps the dynamic axis and masks outliers instead of
        dropping them: consumed by the weighted average when averaging (then
        results equal the reference exactly), zeroed in place otherwise, with
        the mask exposed as ``last_keep_mask_``;
      * alignment with registration_method='pattern' swaps the per-transient
        Powell search for a closed-form phase and a vectorized pattern descent
        over the frequency shift — equal objective values, but on noisy data
        the solvers settle in micro-minima a fraction of a Hz apart (the
        default 'fsl-mrs' method runs the same Powell search as FSL-MRS and
        matches it);
      * water removal takes the top Hankel components from a truncated
        Lanczos SVD (hlsvdpropy's sparse path) instead of a dense
        decomposition, so it matches to floating-point tolerance rather than
        bit-exactly.

    registration_method='torch' selects the batched device engine for the
    whole tensor path (augmentrum.processing.torch_engine): every estimate
    runs as whole-batch torch operations on the data's own device, with no
    per-transient loop, no SciPy and no host round trip - NumPy input is
    processed on CPU torch and handed back. Its differences from the
    reference, all measured on the COWS raw data:
      * registration emulates Powell's decisive first line searches (the
        bracket FSL-MRS's search opens from zero shift) and solves the rest
        exactly; on noisy transients whose cost ripples at the bin scale a
        few in a thousand settle in a neighbouring minimum of equal cost;
      * wSVD whitens with a Cholesky factor instead of an eigendecomposition
        (the weights are invariant to that choice);
      * the outlier metric is taken in the time domain (Parseval);
      * prewhitening that per-sample draws switch off is not warned about;
      * adaptive coil combination and water removal keep their NumPy
        estimates.

    Per-sample coil and transient masks (a sampler drawing with
    per_sample=True) are consumed on every tensor engine: coils combine and
    transients align, reject and average over the kept entries only, which
    equals processing the gathered subset. The torch engine does so batched;
    the others process sample by sample.

    On tensors, the water reference keeps its own dimension layout (read off
    the injected 'water_dim_tags': one transient next to thirty-two
    metabolite averages is the common case), and ppm referencing assumes 1H
    data.
    """

    SUPPORTED_BACKENDS = tuple(Backend)
    DOMAIN = Domain(spectral='time')

    # Coil combination and averaging aggregate over exactly the drawn entries.
    MASKS = 'consume'

    #: Registration methods that run only on tensors.
    TENSOR_ONLY_REGISTRATION = ('pattern', 'torch')

    _dropped_tags = frozenset()
    _dropped_water_tags = frozenset()

    def __init__(self, conj=False, coil=True, align=True, remove_outliers=True, average=True,
                 ecc=True, truncate=False, remove_water=False, shift_ref=True, phase_correct=True,
                 coil_method='fsl-mrs', registration_method='fsl-mrs', remove_method='fsl-mrs',
                 average_method='fsl-mrs', ecc_method='smoothed', water_removal_method='fsl-mrs',
                 shift_ref_method='fsl-mrs', phase_correct_method='fsl-mrs', **kwargs):
        """
        Initializes the processor; one set of flags and defaults for both engines.

        Args:
            conj (bool): Whether to conjugate the data. Off by default: the
                loaders deliver standard NIfTI-MRS, which FSL-MRS reads in
                the right orientation; True mirrors legacy data stored with
                the opposite time-domain convention.
            coil (bool): Whether to perform coil combination.
            align (bool): Whether to align dynamics.
            remove_outliers (bool): Whether to remove/mask outlier averages.
            average (bool): Whether to average dynamics.
            ecc (bool): Whether to perform eddy current correction.
            truncate (bool): Whether to truncate the FID.
            remove_water (bool): Whether to remove residual water peak.
            shift_ref (bool): Whether to shift spectrum to reference peak.
            phase_correct (bool): Whether to perform phase correction.

            coil_method (str): Coil combination method ('fsl-mrs' or 'adaptive').
            registration_method (str): Registration method ('fsl-mrs' for the
                FSL-MRS Powell search; 'pattern' for the fast vectorized
                search, tensor engine only; 'torch' for the batched device
                engine of every step, see above).
            remove_method (str): Outlier removal method ('fsl-mrs').
            average_method (str): Averaging method ('fsl-mrs').
            ecc_method (str): Eddy current correction method ('fsl-mrs' for the raw
                reference phase, 'smoothed' for its Gaussian-smoothed form).
            water_removal_method (str): Water removal method ('fsl-mrs').
            shift_ref_method (str): Frequency shifting method ('fsl-mrs').
            phase_correct_method (str): Phase correction method ('fsl-mrs').
        """
        super().__init__()

        # The pattern and torch registrations exist only batched, so those
        # configurations narrow support and let a NIfTI-list batch route to
        # the tensor engine instead of failing on the list engine.
        if registration_method in self.TENSOR_ONLY_REGISTRATION:
            self.SUPPORTED_BACKENDS = tuple(b for b in Backend
                                            if b is not Backend.NIFTI_LIST)

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

    #**********************#
    #   nifti-list path    #
    #**********************#
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
        self._warned_no_prewhiten = False

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
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

        # A water reference tagged with a single transient loses that axis
        # here rather than at the squeeze below: nibabel drops a trailing
        # singleton from the array while the header keeps it, and FSL-MRS's
        # average and align index the array by the header. The values are
        # untouched, and every later stage sees what it would have anyway.
        if (data_wat is not None and 'DIM_DYN' in getattr(data_wat, 'dim_tags', [])
                and data_wat.shape[data_wat.dim_position('DIM_DYN')] == 1):
            data_wat = safe_squeeze(data_wat, dims=['DIM_DYN'])

        if self.conj: # conjugate if needed
            data_met = proc.conjugate(data_met)
            data_wat = proc.conjugate(data_wat) if data_wat is not None else None

        if self.coil: # coil combination
            data_met, data_wat = self._coil_combine_nifti(data_met, data_wat,
                                                   method=self.coil_method, report=report)

        if self.align:  # registration
            data_met, data_wat = self._registration_nifti(data_met, data_wat,
                                                   method=self.registration_method, report=report)

        if self.remove_outliers:  # remove outlier averages
            data_met, data_wat = self._remove_unlike_nifti(data_met, data_wat,
                                                      method=self.remove_method, report=report)

        if self.average:  # averaging
            data_met, data_wat = self._combine_averages_nifti(data_met, data_wat,
                                                       method=self.average_method, report=report)

        if 'DIM_DYN' in data_met.dim_tags or 'DIM_COIL' in data_met.dim_tags:
            data_met = safe_squeeze(data_met)
        if data_wat is not None and ('DIM_DYN' in data_wat.dim_tags or 'DIM_COIL' in data_wat.dim_tags):
            data_wat = safe_squeeze(data_wat)

        if self.ecc:  # eddy current correction
            data_met, data_wat = self._ecc_nifti(data_met, data_wat,
                                                              method=self.ecc_method, report=report)

        if self.truncate:  # truncation or zero-filling
            data_met = proc.truncate_or_pad(data_met, -1, 'first', report=report)   # truncation
            data_wat = proc.truncate_or_pad(data_wat, -1, 'first') if data_wat is not None else None

        if self.remove_water:   # unsuppressed water removal
            data_met, data_wat = self._water_removal_nifti(data_met, data_wat,
                                                    method=self.water_removal_method, report=report)

        if self.shift_ref:  # frequency shift to reference
            data_met, data_wat = self._shift_to_reference_nifti(data_met, data_wat,
                                                         method=self.shift_ref_method, report=report)

        if self.phase_correct:   # phase correction
            data_met, data_wat = self._phase_correction_nifti(data_met, data_wat,
                                                       method=self.phase_correct_method, report=report)

        return data_met, data_wat

    def _coil_combine_nifti(self, data_met, data_wat=None, method='fsl-mrs', report=None):
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
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc
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
                from augmentrum.processing.utils import nifti_coil_combination_adaptive
                data_met, data_wat = nifti_coil_combination_adaptive(data_met, data_wat, report=report)
            else:
                raise ValueError(f"Unknown coil combination method: {method}")
        return data_met, data_wat

    def _coil_combine_adaptive(self, met, wat, tags, coil_axis):
        """
        The adaptive combination, batched: dephase with the reference,
        combine with its phase-only eigenvector.

        Mirrors nifti_coil_combination_adaptive: the reference (water, or the
        data itself) is averaged over dynamics, its phase removed point by
        point, and the FID-A eigenvector estimate combines the dephased
        channels. Estimate detached, combination differentiable.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor in the same layout, or None.
            tags: Higher-dimension tags (still carrying DIM_COIL).
            coil_axis: Where the coil dimension sits.

        Returns:
            Combined (met, wat), the coil axis summed away.
        """
        from augmentrum.processing.utils import estimate_csm

        met_tc = move_axis(met, coil_axis, -1)                      # (..., T, C)
        wat_tc = move_axis(wat, coil_axis, -1) if wat is not None else None
        others = [t for t in tags if t != 'DIM_COIL']

        ref_tc = wat_tc if wat_tc is not None else met_tc
        if 'DIM_DYN' in others:
            ref_tc = ops.mean(ref_tc, axis=4 + others.index('DIM_DYN'), keepdims=True)

        ref = _complex_like(ops.to_numpy(ref_tc))
        phase = np.exp(-1j * np.angle(ref))
        lead = ref.shape[:-2]
        flat = (ref * phase).reshape((-1,) + ref.shape[-2:])
        csm = np.stack([estimate_csm(voxel)[:, 0] for voxel in flat])
        csm = csm.reshape(lead + (ref.shape[-1],))
        csmsq = np.real((csm * np.conj(csm)).sum(-1, keepdims=True))

        weights = (np.conj(csm)[..., None, :] * phase
                   / (csmsq[..., None] + np.finfo(float).eps))
        met_c = ops.sum(met_tc * ops.match_backend(weights, met_tc), axis=-1)
        wat_c = (ops.sum(wat_tc * ops.match_backend(weights, wat_tc), axis=-1)
                 if wat_tc is not None else None)
        return met_c, wat_c

    def _registration_nifti(self, data_met, data_wat=None, method='fsl-mrs', report=None):
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
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc
        if method == 'fsl-mrs':
            if 'DIM_DYN' in getattr(data_met, 'dim_tags', []) and data_met.shape[data_met.dim_position('DIM_DYN')] > 1:
                # squeeze coil dim if still present
                if 'DIM_COIL' in data_met.dim_tags:
                    data_met = data_met.copy(remove_dim='DIM_COIL')
                data_met = proc.align(data_met, 'DIM_DYN', ppmlim=(0.2, 4.2), report=report)
            # The water aligns along its own transients, and only when there
            # is more than one: FSL-MRS align indexes a list of MRS objects
            # that is not a list for a single FID, and a lone transient has
            # nothing to align to anyway — the squeeze takes the axis later.
            if (data_wat is not None and 'DIM_DYN' in getattr(data_wat, 'dim_tags', [])
                    and data_wat.shape[data_wat.dim_position('DIM_DYN')] > 1):
                if 'DIM_COIL' in data_wat.dim_tags:
                    data_wat = data_wat.copy(remove_dim='DIM_COIL')
                data_wat = proc.align(data_wat, 'DIM_DYN', ppmlim=(0, 8))
        else:
            raise ValueError(f"Unknown registration method: {method}")
        return data_met, data_wat

    def _remove_unlike_nifti(self, data_met, data_wat=None, method='fsl-mrs', report=None):
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
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc
        if 'DIM_DYN' in getattr(data_met, 'dim_tags', []) and data_met.shape[data_met.dim_position('DIM_DYN')] > 1:
            if method == 'fsl-mrs':
                data_met, _ = proc.remove_unlike(data_met, report=report)  # remove outlier averages
            else:
                raise ValueError(f"Unknown outlier removal method: {method}")
        return data_met, data_wat

    def _combine_averages_nifti(self, data_met, data_wat=None, method='fsl-mrs', report=None):
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
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc
        if 'DIM_DYN' in getattr(data_met, 'dim_tags', []):
            if data_met.shape[data_met.dim_position('DIM_DYN')] > 1:
                data_met = proc.average(data_met, 'DIM_DYN', report=report)  # combine averages
        if data_wat is not None and 'DIM_DYN' in getattr(data_wat, 'dim_tags', []):
            if data_wat is not None and data_wat.shape[data_wat.dim_position('DIM_DYN')] > 1:
                data_wat = proc.average(data_wat, 'DIM_DYN')
        return data_met, data_wat

    def _ecc_nifti(self, data_met, data_wat=None, method='smoothed', report=None):
        """
        Performs eddy current correction on the MRS data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            method (str): Eddy current correction method ('fsl-mrs' or 'smoothed').
            report: Optional report object for logging processing steps.

        Returns:
            None. The metabolite data is modified in place.
        """
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc
        if self.ecc_method == 'fsl-mrs':
            data_met = proc.ecc(data_met, data_wat if data_wat is not None else data_met,
                                report=report)  # eddy current correction
            data_wat = proc.ecc(data_wat, data_wat) if data_wat is not None else None
        elif self.ecc_method == 'smoothed':
            from augmentrum.processing.utils import nifti_ecc_smoothed
            data_met = nifti_ecc_smoothed(data_met, data_wat if data_wat is not None else data_met, report=report)
            data_wat = nifti_ecc_smoothed(data_wat, data_wat) if data_wat is not None else None
        else:
            raise ValueError(f"Unknown ECC method: {self.ecc_method}")
        return data_met, data_wat

    def _water_removal_nifti(self, data_met, data_wat=None, method='fsl-mrs', report=None):
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
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc
        if method == 'fsl-mrs':
            data_met = proc.remove_peaks(data_met, [-0.15, 0.15], limit_units='ppm',
                                         report=report)  # remove residual water
        else:
            raise ValueError(f"Unknown water removal method: {method}")
        return data_met, data_wat

    def _shift_to_reference_nifti(self, data_met, data_wat=None, method='fsl-mrs', report=None):
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
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc
        if method == 'fsl-mrs':
            data_met = proc.shift_to_reference(data_met, 3.027, (2.9, 3.1), report=report)  # shift to ref
        else:
            raise ValueError(f"Unknown frequency shifting method: {method}")
        return data_met, data_wat

    def _phase_correction_nifti(self, data_met, data_wat=None, method='fsl-mrs', report=None):
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
        from fsl_mrs.utils.preproc import nifti_mrs_proc as proc
        if method == 'fsl-mrs':
            data_met = proc.phase_correct(data_met, (2.9, 3.1), report=report)  # phase corretion
            data_wat = proc.phase_correct(data_wat, (4.55, 4.7), hlsvd=False) if data_wat is not None else None
        else:
            raise ValueError(f"Unknown phase correction method: {method}")
        return data_met, data_wat

    def _estimate_noise_cov(self, data, noise=None, no_prewhiten=False):
        """
        Estimates the noise covariance for coil combination from the FID tails.

        Below FSL-MRS's threshold of ten noise samples per coil (a single
        transient of 2048 points leaves 205 for 32 coils) the covariance
        cannot be estimated, so prewhitening is switched off instead, as the
        tensor engine does per subject. The identity is handed on explicitly:
        given no covariance, proc.coilcombine repeats the estimate itself and
        raises the very error caught here.

        Args:
            data: Metabolite NIfTI-MRS object carrying DIM_COIL.
            noise: Passed through to proc.coilcombine.
            no_prewhiten: Whether prewhitening is already off.

        Returns:
            Tuple of (noise, covariance, no_prewhiten) for proc.coilcombine.
        """
        from fsl_mrs.utils.preproc.combine import estimate_noise_cov, CovarianceEstimationError

        stacked_data = [dd for dd, _ in
                        data.iterate_over_dims(dim='DIM_COIL', iterate_over_space=True, reduce_dim_index=True)]
        try:
            covariance = estimate_noise_cov(np.asarray(stacked_data))
        except CovarianceEstimationError:
            self._warn_no_prewhiten()
            no_prewhiten = True
            covariance = np.eye(data.shape[data.dim_position('DIM_COIL')])
        return noise, covariance, no_prewhiten

    def _warn_no_prewhiten(self):
        """
        Warns once per run that coil combination goes without prewhitening.

        Once, because a batch of single-transient scans would otherwise say
        the same thing for every subject.
        """
        if not getattr(self, '_warned_no_prewhiten', False):
            self._warned_no_prewhiten = True
            warnings.warn('Too few noise samples to estimate the coil covariance (FSL-MRS '
                          'needs ten per coil); combining coils without prewhitening.',
                          RuntimeWarning)

    @staticmethod
    def _per_sample(values, ndim):
        """Per-sample values shaped to broadcast against a tensor's leading axis."""
        return np.asarray(values, dtype=float).reshape((-1,) + (1,) * (ndim - 1))

    @staticmethod
    def _sf_samples(values, batch, n_points, sw_hz):
        """
        The per-sample spectrometer frequencies, or None when one value serves the batch.

        A batch is sliced with one pair of ppm bounds, so the frequencies may differ
        only by less than a bin - which is what a scanner's own referencing does (parts
        per million), not what a different field or nucleus does. The latter is refused
        rather than silently processed at one scan's frequency.

        Args:
            values: The batch's frequencies in MHz, or None.
            batch: How many samples the batch has.
            n_points: Points per FID; the windows are also checked zero-filled and
                truncated by a point, as the engines slice them.
            sw_hz: Spectral width in Hz.

        Returns:
            (batch,) array of frequencies, or None when they agree.
        """
        if values is None:
            return None
        sf = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        if sf.size == 1 or bool(np.all(sf == sf[0])):
            return None
        if sf.size != batch:
            raise ValueError(f"'sf_mhz_samples' has {sf.size} values for a batch of {batch}.")
        for n in (n_points, 4 * n_points, n_points - 1, 4 * (n_points - 1)):
            for lim in ((2.9, 3.1), (4.55, 4.7), (0.2, 4.2), (0.0, 8.0), (-0.15, 0.15)):
                # the bounds move monotonically with the frequency, so the extremes decide
                if ppm_window(n, sw_hz, float(sf.min()), lim) != \
                        ppm_window(n, sw_hz, float(sf.max()), lim):
                    raise ValueError(
                        'RawProcessor got scans whose spectrometer frequencies resolve '
                        f'different ppm windows ({sf.min():.6f}-{sf.max():.6f} MHz moves '
                        f'{lim} ppm across a bin): process them in separate batches.')
        return sf

    def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
        """
        Runs the processing pipeline on a batched tensor, spectral axis last.

        Args:
            data_array: Metabolite data, (batch, X, Y, Z, higher dims..., T).
            water_array: Water reference in the untransposed NIfTI layout
                (batch, X, Y, Z, T, higher dims...), optional. It is a
                separate acquisition with its own layout — one transient
                next to thirty-two metabolite averages, say — so its axes are
                read off 'water_dim_tags', never off the data's.
            backend: The array backend the tensors live on.
            **kwargs: Injected metadata; sw_hz, sf_mhz, dim_tags and
                water_dim_tags are used. Without water_dim_tags the water is
                taken to share the data's layout for the axes it has.

        Returns:
            Tuple of processed (data_array, water_array), collapsed dimensions
            removed (the water is returned in its untransposed layout).
        """
        sw_hz = kwargs.get('sw_hz')
        sf_mhz = kwargs.get('sf_mhz')
        if sw_hz is None or sf_mhz is None:
            raise ValueError("RawProcessor needs 'sw_hz' and 'sf_mhz' — provide them or process "
                             "data with header metadata attached.")

        sf_samples = self._sf_samples(kwargs.get('sf_mhz_samples'), ops.shape(data_array)[0],
                                      ops.shape(data_array)[-1], sw_hz)
        if sf_samples is not None:
            sf_mhz = float(sf_samples[0])       # every sample resolves the same ppm windows

        masks = {tag: mask for tag, mask in (kwargs.get('dim_masks') or {}).items()
                 if mask is not None}
        self.dim_masks_ = {}
        if masks and self.registration_method != 'torch':
            return self._process_per_sample(data_array, water_array, backend, masks, sf_samples,
                                            **kwargs)

        met, wat = data_array, water_array
        self._warned_no_prewhiten = False

        # Tags past the array's rank name trailing singleton axes that
        # nibabel already squeezed away (a lone transient, say): they are
        # gone from the tensor, so they are gone from the output's tags too.
        given = [t for t in (kwargs.get('dim_tags') or []) if t]
        tags = given[:max(0, len(ops.shape(met)) - 5)]
        self._dropped_tags = set(given[len(tags):])
        self._dropped_water_tags = set()

        wtags = []
        if wat is not None:
            given_w = kwargs.get('water_dim_tags')
            given_w = [t for t in (given if given_w is None else given_w) if t]
            wtags = self._water_tags(wat, given_w)
            self._dropped_water_tags = set(given_w[len(wtags):])
            if len(ops.shape(wat)) > 5:
                wat = move_axis(wat, self.SPECTRAL_AXIS, -1)

        if self.registration_method == 'torch':
            met, wat = self._process_torch(met, wat, tags, wtags, sw_hz, sf_mhz, masks,
                                           kwargs.get('pool_origin'),
                                           kwargs.get('water_pool_origin'), sf_samples)
            if wat is not None and len(ops.shape(wat)) > 5:
                wat = move_axis(wat, -1, self.SPECTRAL_AXIS)
            return met, wat

        if self.conj:
            met = ops.complex_from(ops.real(met), -ops.imag(met))
            wat = ops.complex_from(ops.real(wat), -ops.imag(wat)) if wat is not None else None

        if self.coil:
            met, wat, tags, wtags = self.coil_combine(met, wat, tags, wtags)

        if self.align:
            met, wat, tags, wtags = self.registration(met, wat, tags, wtags, sw_hz, sf_mhz)

        mask = self.outlier_mask(met, tags) if self.remove_outliers else None

        if mask is not None and not self.average:
            # No average to consume the mask, so removal is expressed as
            # zeros: the tensor stays rectangular where the NIfTI path would
            # drop transients, and last_keep_mask_ says which ones survived.
            met = met * ops.match_backend(mask[..., None].astype(np.float64), met)

        if self.average:
            met, wat, tags, wtags = self.combine_averages(met, wat, tags, wtags, mask)

        met, wat, tags, wtags = self._squeeze_singletons(met, wat, tags, wtags)

        if self.ecc:
            met, wat = self.eddy_current_correction(met, wat)

        if self.truncate:
            met = met[..., 1:]
            wat = wat[..., 1:] if wat is not None else None

        if self.remove_water:
            met = self.water_removal(met, sw_hz, sf_mhz, sf_samples)

        if self.shift_ref:
            met = self.shift_to_reference(met, sw_hz, sf_mhz, sf_samples)

        if self.phase_correct:
            met, wat = self.phase_correction(met, wat, sw_hz, sf_mhz)

        if wat is not None and len(ops.shape(wat)) > 5:
            wat = move_axis(wat, -1, self.SPECTRAL_AXIS)
        return met, wat

    def coil_combine(self, met, wat, tags, wtags=None):
        """
        Coil combination on the tensor engine; weights from the water when present.

        'fsl-mrs' estimates per-subject noise covariance from the last tenth
        of every FID (prewhitening is disabled per subject when there are too
        few samples, as in FSL-MRS) and derives wSVD weights; 'adaptive'
        mirrors nifti_coil_combination_adaptive, dephasing with the reference
        and combining with its phase-only eigenvector. Either way the
        combination is a differentiable weighted sum over the coil axis.

        The reference is the water averaged over its own transients, so one
        set of weights per voxel serves every metabolite average: the water
        keeps its own layout throughout and the weights are broadcast onto
        the data's.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor, spectral axis last, or None.
            tags: Higher-dimension tags of the data, mutated in place.
            wtags: Higher-dimension tags of the water, mutated in place
                (None: the data's, for the axes the water has).

        Returns:
            Tuple of (met, wat, tags, wtags) with the coil dimensions collapsed.
        """
        wtags = self._water_tags(wat, tags if wtags is None else wtags)
        if 'DIM_COIL' not in tags:
            return met, wat, tags, wtags
        coil_axis = 4 + tags.index('DIM_COIL')
        n_coil = ops.shape(met)[coil_axis]
        if n_coil <= 1:
            return met, wat, tags, wtags
        if self.coil_method not in ('fsl-mrs', 'adaptive'):
            raise ValueError(f"Unknown tensor coil combination method: {self.coil_method}")

        wcoil_axis = None
        if wat is not None:
            # Same receive array or nothing: the weights are per element.
            if 'DIM_COIL' not in wtags:
                raise ValueError('The water reference has no coil dimension to combine '
                                 'alongside the data.')
            wcoil_axis = 4 + wtags.index('DIM_COIL')
            if ops.shape(wat)[wcoil_axis] != n_coil:
                raise ValueError('Reference and data coil dimension does not match.')

        combine = (self._coil_combine_adaptive if self.coil_method == 'adaptive'
                   else self._coil_combine_wsvd)
        met, wat = combine(met, wat, tags, wtags, coil_axis, wcoil_axis)

        tags.remove('DIM_COIL')
        self._dropped_tags.add('DIM_COIL')
        if wat is not None:
            wtags.remove('DIM_COIL')
            self._dropped_water_tags.add('DIM_COIL')
        return met, wat, tags, wtags

    def _coil_combine_wsvd(self, met, wat, tags, wtags, coil_axis, wcoil_axis):
        """
        The wSVD combination, batched: FSL-MRS coilcombine on tensors.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor, spectral axis last, or None.
            tags: Higher-dimension tags of the data (still carrying DIM_COIL).
            wtags: Higher-dimension tags of the water.
            coil_axis: Where the data's coil dimension sits.
            wcoil_axis: Where the water's coil dimension sits, or None.

        Returns:
            Combined (met, wat), the coil axes summed away.
        """
        n_time = ops.shape(met)[-1]
        n_batch = ops.shape(met)[0]
        n_coil = ops.shape(met)[coil_axis]

        # per-subject noise covariance and whitening, from the FID tails
        noise = ops.to_numpy(met[..., int(0.9 * n_time):])
        noise = np.moveaxis(noise, coil_axis, -1).reshape(n_batch, -1, n_coil)
        eye = np.eye(n_coil, dtype=np.result_type(noise.dtype, np.complex64))
        cov = np.empty((n_batch, n_coil, n_coil), dtype=eye.dtype)
        white = np.empty_like(cov)
        white_inv = np.empty_like(cov)
        for b, samples in enumerate(noise):
            if samples.shape[0] < 10 * n_coil:
                self._warn_no_prewhiten()
                cov[b], white[b], white_inv[b] = eye, eye, eye      # prewhitening disabled
            else:
                cov[b] = np.cov(samples, rowvar=False)
                d, v = np.linalg.eigh(cov[b], UPLO='U')
                white[b] = v @ np.diag(1 / np.sqrt(d))
                white_inv[b] = np.linalg.inv(white[b])

        met_tc = move_axis(met, coil_axis, -1)                     # (..., T, C)
        wat_tc = move_axis(wat, wcoil_axis, -1) if wat is not None else None

        if wat_tc is not None:
            source_tc, with_reference = self._transient_mean(wat_tc, wtags), True
        else:
            source_tc, with_reference = met_tc, False

        lead = len(ops.shape(source_tc)) - 2
        shape = (n_batch,) + (1,) * (lead - 1) + (n_coil, n_coil)
        source = _complex_like(ops.to_numpy(source_tc))
        _, _, vh = np.linalg.svd(source @ white.reshape(shape), full_matrices=False)
        weights = self._wsvd_weights(vh[..., 0, :], white.reshape(shape),
                                     white_inv.reshape(shape), cov.reshape(shape),
                                     with_reference)[..., None, :]          # (..., 1, C)

        if wat_tc is not None:
            wat = ops.sum(wat_tc * ops.match_backend(weights, wat_tc), axis=-1)
            weights = self._onto_data(weights, len(ops.shape(met_tc)))
        met = ops.sum(met_tc * ops.match_backend(weights, met_tc), axis=-1)
        return met, wat

    def _coil_combine_adaptive(self, met, wat, tags, wtags, coil_axis, wcoil_axis):
        """
        The adaptive combination, batched: dephase with the reference,
        combine with its phase-only eigenvector.

        Mirrors nifti_coil_combination_adaptive: the reference (water, or the
        data itself) is averaged over its transients, its phase removed point
        by point, and the FID-A eigenvector estimate combines the dephased
        channels. Estimate detached, combination differentiable.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor, spectral axis last, or None.
            tags: Higher-dimension tags of the data (still carrying DIM_COIL).
            wtags: Higher-dimension tags of the water.
            coil_axis: Where the data's coil dimension sits.
            wcoil_axis: Where the water's coil dimension sits, or None.

        Returns:
            Combined (met, wat), the coil axes summed away.
        """
        from augmentrum.processing.utils import estimate_csm

        met_tc = move_axis(met, coil_axis, -1)                      # (..., T, C)
        wat_tc = move_axis(wat, wcoil_axis, -1) if wat is not None else None

        ref_tc = (self._transient_mean(wat_tc, wtags) if wat_tc is not None
                  else self._transient_mean(met_tc, tags))

        ref = _complex_like(ops.to_numpy(ref_tc))
        phase = np.exp(-1j * np.angle(ref))
        lead = ref.shape[:-2]
        flat = (ref * phase).reshape((-1,) + ref.shape[-2:])
        csm = np.stack([estimate_csm(voxel)[:, 0] for voxel in flat])
        csm = csm.reshape(lead + (ref.shape[-1],))
        csmsq = np.real((csm * np.conj(csm)).sum(-1, keepdims=True))

        weights = (np.conj(csm)[..., None, :] * phase
                   / (csmsq[..., None] + np.finfo(float).eps))     # (..., T, C)
        if wat_tc is not None:
            wat = ops.sum(wat_tc * ops.match_backend(weights, wat_tc), axis=-1)
            weights = self._onto_data(weights, len(ops.shape(met_tc)))
        met = ops.sum(met_tc * ops.match_backend(weights, met_tc), axis=-1)
        return met, wat

    @staticmethod
    def _water_tags(wat, tags):
        """
        The water's higher-dimension tags, as a fresh list cut to its rank.

        Callers without tags for the water pass the data's: the water is then
        taken to share the data's layout for as many higher axes as it has —
        the bare-tensor convention from before water tags were injected.
        """
        if wat is None:
            return []
        return [t for t in tags if t][:max(0, len(ops.shape(wat)) - 5)]

    @staticmethod
    def _transient_mean(data_tc, tags):
        """
        Averages over the DIM_DYN of *tags* (kept as a singleton), coil axis last.

        The coil axis was moved behind the spectral one, so a dynamic axis
        that followed it has stepped down by one — hence the index among the
        tags without DIM_COIL.
        """
        others = [t for t in tags if t != 'DIM_COIL']
        if 'DIM_DYN' not in others:
            return data_tc
        return ops.mean(data_tc, axis=4 + others.index('DIM_DYN'), keepdims=True)

    @staticmethod
    def _onto_data(weights, rank):
        """
        Reshapes water-derived weights to broadcast over the data's layout.

        Weights come as "(B, X, Y, Z, <water dims>, T', C)" and the data is
        "(B, X, Y, Z, <data dims>, T, C)" with *rank* axes; the water's own
        higher dims are singletons by now (its transients went into the
        reference), so they fold away and the data's take their place. Left
        to broadcasting, a batch axis would be paired with a data dimension
        instead — a B-by-B product for a batch of more than one, and a
        silently wrong combination for a batch of one.

        Args:
            weights: Per-coil weights in the water's layout.
            rank: Number of axes of the data (spectral, then coil, last).

        Returns:
            The weights reshaped for the data.
        """
        lead, tail = weights.shape[:4], weights.shape[-2:]
        if int(np.prod(weights.shape[4:-2])) != 1:
            raise ValueError('The water reference must carry no higher dimension besides '
                             'coils and transients to serve as a per-voxel reference.')
        return weights.reshape(lead + (1,) * (rank - 6) + tail)

    def registration(self, met, wat, tags, wtags, sw_hz, sf_mhz):
        """
        Aligns dynamics in phase and frequency (spectral registration).

        A leftover coil dimension is reduced to its first element the way the
        FSL-MRS path's copy(remove_dim='DIM_COIL') does. The metabolite data
        aligns within (0.2, 4.2) ppm, the water within (0, 8) ppm — each along
        its own dynamic dimension, and only where there is more than one
        transient to align. Method 'fsl-mrs' reproduces the FSL-MRS Powell
        search per transient; 'pattern' is the vectorized pattern search
        (much faster, results equal in objective value but not identical on
        noisy data).

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor, spectral axis last, or None.
            tags: Higher-dimension tags of the data, mutated in place.
            wtags: Higher-dimension tags of the water, mutated in place.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.

        Returns:
            Tuple of (met, wat, tags, wtags).
        """
        if self.registration_method not in ('fsl-mrs', 'pattern', 'torch'):
            raise ValueError(f"Unknown tensor registration method: {self.registration_method}")
        met, tags = self._align_transients(met, tags, self._dropped_tags,
                                           sw_hz, sf_mhz, (0.2, 4.2))
        if wat is not None:
            wat, wtags = self._align_transients(wat, wtags, self._dropped_water_tags,
                                                sw_hz, sf_mhz, (0, 8))
        return met, wat, tags, wtags

    def _align_transients(self, data, tags, dropped, sw_hz, sf_mhz, ppmlim):
        """
        Aligns *data* along its DIM_DYN when there is more than one transient.

        A coil dimension still present is cut to its first element first, as
        the list path does — and recorded in *dropped*, with the tags kept in
        step.
        """
        if 'DIM_DYN' not in tags or ops.shape(data)[4 + tags.index('DIM_DYN')] <= 1:
            return data, tags
        if 'DIM_COIL' in tags:
            coil_axis = 4 + tags.index('DIM_COIL')
            data = data[(slice(None),) * coil_axis + (0,)]
            tags.remove('DIM_COIL')
            dropped.add('DIM_COIL')
        dyn_axis = 4 + tags.index('DIM_DYN')
        return self._apply_alignment(data, dyn_axis, sw_hz, sf_mhz, ppmlim), tags

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
        dropped. When averaging follows, the weighted mean consumes the mask
        and the result equals the NIfTI path's exactly. Without averaging, the
        outlier transients are zeroed in place and the mask is exposed as
        "last_keep_mask_" — any later hand-averaging must respect it, or the
        zeros dilute the mean.

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
        mask = self._unlike_mask(ops.to_numpy(met))
        self.last_keep_mask_ = mask
        return mask

    def combine_averages(self, met, wat, tags, wtags, mask=None):
        """
        Averages dynamics; a keep-mask turns this into a weighted mean.

        The water averages over its own transients when it has more than
        one; a lone transient is left to the squeeze, its values untouched.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor, spectral axis last, or None.
            tags: Higher-dimension tags of the data, mutated in place.
            wtags: Higher-dimension tags of the water, mutated in place.
            mask: Optional boolean keep mask over the data's dynamic axis.

        Returns:
            Tuple of (met, wat, tags, wtags) with the dynamic dimensions collapsed.
        """
        if self.average_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor averaging method: {self.average_method}")

        if 'DIM_DYN' in tags and ops.shape(met)[4 + tags.index('DIM_DYN')] > 1:
            dyn_axis = 4 + tags.index('DIM_DYN')
            if mask is None:
                met = ops.mean(met, axis=dyn_axis)
            else:
                weights = mask.astype(np.float64) / mask.sum(axis=-1, keepdims=True)
                met = ops.sum(met * ops.match_backend(weights[..., None], met), axis=dyn_axis)
            tags.remove('DIM_DYN')
            self._dropped_tags.add('DIM_DYN')

        if wat is not None and 'DIM_DYN' in wtags \
                and ops.shape(wat)[4 + wtags.index('DIM_DYN')] > 1:
            wat = ops.mean(wat, axis=4 + wtags.index('DIM_DYN'))
            wtags.remove('DIM_DYN')
            self._dropped_water_tags.add('DIM_DYN')
        return met, wat, tags, wtags

    def _squeeze_singletons(self, met, wat, tags, wtags):
        """
        Drops singleton higher dimensions, the tensor form of safe_squeeze.

        Data and water are squeezed each on their own layout, and only while
        an acquisition dimension (coils or transients) is still tagged, as the
        list path does.
        """
        if 'DIM_DYN' in tags or 'DIM_COIL' in tags:
            met, tags = self._squeeze(met, tags, self._dropped_tags)
        if wat is not None and ('DIM_DYN' in wtags or 'DIM_COIL' in wtags):
            wat, wtags = self._squeeze(wat, wtags, self._dropped_water_tags)
        return met, wat, tags, wtags

    @staticmethod
    def _squeeze(data, tags, dropped):
        """Drops the singleton axes among *tags*, recording them in *dropped*."""
        for i in reversed(range(len(tags))):
            axis = 4 + i
            if ops.shape(data)[axis] == 1:
                data = data[(slice(None),) * axis + (0,)]
                dropped.add(tags[i])
                tags.pop(i)
        return data, tags

    def eddy_current_correction(self, met, wat):
        """
        Eddy current correction against the water reference (or the data itself).

        'smoothed' mirrors nifti_ecc_smoothed: the Gaussian-smoothed unwrapped reference
        phase is removed. 'fsl-mrs' mirrors preproc.eddy_correct: the raw
        reference phase is removed. The water corrects against itself.

        Args:
            met: Metabolite tensor, spectral axis last.
            wat: Water tensor, one FID per voxel by now (or the data's own
                layout), or None.

        Returns:
            Tuple of corrected (met, wat).
        """
        if wat is not None and len(ops.shape(wat)) > 5 \
                and tuple(ops.shape(wat)) != tuple(ops.shape(met)):
            raise ValueError('Reference and data shape must match or the reference must be '
                             'a single FID per voxel (as in eddy current correction).')
        ref = ops.to_numpy(wat if wat is not None else met)
        if self.ecc_method == 'smoothed':
            phasor = np.exp(-1j * self._ecc_phase(ref))
        elif self.ecc_method == 'fsl-mrs':
            phasor = np.exp(-1j * np.angle(ref))
        else:
            raise ValueError(f"Unknown ECC method: {self.ecc_method}")
        met = met * ops.match_backend(phasor, met)
        wat = wat * ops.match_backend(phasor, wat) if wat is not None else None
        return met, wat

    def water_removal(self, met, sw_hz, sf_mhz, sf_samples=None):
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
            sf_samples: Per-sample frequencies, when the batch's scans differ.

        Returns:
            Metabolite tensor with the water model subtracted.
        """
        if self.water_removal_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor water removal method: {self.water_removal_method}")
        from scipy.sparse.linalg import svds

        arr = _complex_like(ops.to_numpy(met))
        n = arr.shape[-1]
        m = n // 2
        k = min(20, n - m - 1, m)
        flat = arr.reshape(-1, n)
        uk = np.empty((flat.shape[0], n - m, k), dtype=arr.dtype)
        for i, fid in enumerate(flat):
            hankel = np.lib.stride_tricks.sliding_window_view(fid, m + 1)   # (n-m, m+1)
            uk[i] = svds(hankel, k=k)[0]
        uk = uk.reshape(arr.shape[:-1] + (n - m, k))
        sf = sf_mhz if sf_samples is None else self._per_sample(sf_samples, arr.ndim)
        model = self._hlsvd_water_model(uk, arr, sw_hz, sf, (-0.15, 0.15), k=k)
        return met - ops.match_backend(model, met)

    def shift_to_reference(self, met, sw_hz, sf_mhz, sf_samples=None):
        """
        Shifts the peak found in (2.9, 3.1) ppm to the tCr reference 3.027 ppm.

        The peak is located on a four-fold zero-padded spectrum, exactly as
        FSL-MRS shiftToRef does, and the shift is applied as a phase ramp.

        Args:
            met: Metabolite tensor, spectral axis last.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.
            sf_samples: Per-sample frequencies, when the batch's scans differ.

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
        if sf_samples is None:
            shift_hz = (ppm_shift_axis(4 * n, sw_hz, sf_mhz)[first:last][peak] - 3.027) * sf_mhz
        else:                        # "ppm_shift_axis" per sample, written out to stay vectorized
            sf = self._per_sample(sf_samples, peak.ndim)
            hz = np.linspace(-sw_hz / 2, sw_hz / 2, 4 * n)[first:last][peak]
            shift_hz = (hz / sf + ppm_reference('1H') - 3.027) * sf
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

    #*************************#
    #   per-sample fallback   #
    #*************************#
    def _process_per_sample(self, data_array, water_array, backend, masks, sf_samples=None,
                            **kwargs):
        """
        Per-sample masks on the NumPy-estimate engines: every sample's drawn
        subset is gathered and processed on its own, which is exactly what the
        masks stand for. Slow, as those engines are; the torch engine does the
        same batched.

        Args:
            data_array: Data, spectral axis last.
            water_array: Water in its untransposed layout, or None.
            backend: The array backend.
            masks: {tag: (batch, n) bool} over the data's dimensions.
            sf_samples: Every sample's frequency where the batch's scans
                differ, else None; each sample is processed at its own.
            **kwargs: As for process_tensor.

        Returns:
            "(data, water)" as process_tensor returns them.
        """
        given = [t for t in (kwargs.get('dim_tags') or []) if t]
        given_w = kwargs.get('water_dim_tags')
        given_w = [t for t in (given if given_w is None else given_w) if t]
        if ('DIM_COIL' in masks and not self.coil) or ('DIM_DYN' in masks and not self.average):
            raise NotImplementedError(
                "Per-sample masks over a dimension this processor keeps need "
                "registration_method='torch', which carries them through.")

        inner = {key: value for key, value in kwargs.items()
                 if key not in ('dim_masks', 'pool_origin', 'water_pool_origin',
                                'sf_mhz_samples')}
        outs, water_outs, layout = [], [], None
        for i in range(ops.shape(data_array)[0]):
            sample = data_array[i:i + 1]
            water = water_array[i:i + 1] if water_array is not None else None
            if sf_samples is not None:
                inner['sf_mhz'] = float(sf_samples[i])
            for tag, mask in masks.items():
                keep = np.flatnonzero(np.asarray(ops.to_numpy(mask))[i])
                sample = ops.take(sample, keep, axis=4 + given.index(tag))
                if water is not None and tag in WATER_SHARED and tag in given_w:
                    water = ops.take(water, keep, axis=5 + given_w.index(tag))
            out, water_out = self.process_tensor(sample, water, backend, **inner)
            state = (tuple(ops.shape(out)[1:]), sorted(self._dropped_tags),
                     sorted(self._dropped_water_tags))
            if layout is not None and state != layout:
                raise ValueError("The drawn subsets left the samples in different layouts; "
                                 "use registration_method='torch'.")
            layout = state
            outs.append(out)
            water_outs.append(water_out)

        water_out = ops.concatenate(water_outs, axis=0) if water_array is not None else None
        return ops.concatenate(outs, axis=0), water_out

    #******************#
    #   torch engine   #
    #******************#
    #: Higher dimensions the torch engine knows how to process.
    TORCH_DIMS = ('DIM_COIL', 'DIM_DYN')

    #: Replay the torch engine as CUDA graphs on a GPU (see "_process_torch"). The numbers are
    #: the same either way; False launches every kernel from Python, as a debugger may want.
    CUDA_GRAPHS = True

    def _process_torch(self, met, wat, tags, wtags, sw_hz, sf_mhz, masks, origin=None,
                       water_origin=None, sf_samples=None):
        """
        The raw pipeline as batched torch operations on the data's device.

        Data and water are brought to one layout, (B, V, C, D, T) - voxels,
        coils, transients, points, a singleton for a dimension that is absent
        - and every aggregate is restricted to the entries each sample drew.
        Where the values are still the pool's, per-subject results the pool
        cached (noise moments, the water's Gram matrix) replace recomputation;
        they are exact, since a subset's statistics are sums and sub-matrices
        of the whole's.

        On a GPU the pipeline is replayed as CUDA graphs
        ("torch_engine.GraphedSteps"): a batch is a few thousand small kernels,
        which the host takes far longer to launch than the device to run. A
        replay is the recorded kernels on the recorded tensors, so its numbers
        are the eager ones. Everything that decides which kernels run is the
        graphs' signature - shapes, tags, flags, the ppm windows, the pool
        whose caches are read - and what a graph cannot hold (the batch's
        spectrometer frequency, a Python value that changes from batch to
        batch, and MAGMA's solve for the coil weights) runs eagerly between
        the graphs. Where the scans of a batch differ in frequency, each one's
        own enters as a tensor on the device, an input like the masks. A batch
        that needs gradients runs eagerly.

        Args:
            met: Data, (B, X, Y, Z, <tags>, T).
            wat: Water, (B, X, Y, Z, <wtags>, T), or None.
            tags: The data's higher-dimension tags, mutated as dimensions go.
            wtags: The water's, likewise.
            sw_hz: Spectral width in Hz.
            sf_mhz: Spectrometer frequency in MHz.
            masks: {tag: (B, n) bool} per-sample masks over the data's dimensions.
            origin: PoolOrigin of the data, or None.
            water_origin: PoolOrigin of the water, or None.
            sf_samples: Every sample's frequency in MHz where the batch's
                scans differ ("_sf_samples"), else None.

        Returns:
            "(met, wat)" with collapsed dimensions removed, spectral axis last,
            on the input's backend; "dim_masks_" holds the masks of the
            dimensions that remain.
        """
        import torch
        from augmentrum.processing import torch_engine as engine

        for group, name in ((tags, 'data'), (wtags, 'water')):
            unknown = [t for t in group if t not in self.TORCH_DIMS]
            if unknown:
                raise ValueError(f"registration_method='torch' processes {self.TORCH_DIMS}, "
                                 f"but the {name} carries {unknown}; use 'fsl-mrs' or "
                                 f"'pattern'.")
        as_torch = lambda a: (a if ops.is_torch(a)
                              else torch.from_numpy(np.asarray(ops.to_numpy(a))))
        x, spatial = self._to_torch_layout(as_torch(met), tags)
        w = self._to_torch_layout(as_torch(wat), wtags)[0] if wat is not None else None
        masks = {tag: torch.as_tensor(np.asarray(ops.to_numpy(mask))) if not ops.is_torch(mask)
                 else mask for tag, mask in masks.items()}
        masks = {tag: mask.to(device=x.device, dtype=torch.bool) for tag, mask in masks.items()}

        # the pools whose per-subject caches stand in for this batch's statistics
        b, v, c, d, n = x.shape
        pools = tuple(held if self._pool_holds(held, role, group, n) else None
                      for held, role, group in ((origin, 'data', tags),
                                                (water_origin, 'water', wtags)))

        # A raw batch is a quarter of a gigabyte of coils and transients, and the coil
        # combination is all that reads it: where the pool's cached moments cover the rest, it
        # meets its weights in a step and stays out of the graphs, which then neither copy it
        # nor keep a second one.
        deferred = (self.coil and not self.conj and 'DIM_COIL' in tags and c > 1
                    and self.coil_method == 'fsl-mrs' and w is not None and pools[0] is not None)
        inputs = dict(met=None if deferred else x, wat=w, coil_mask=masks.get('DIM_COIL'),
                      dyn_mask=masks.get('DIM_DYN'))
        for key, held in zip(('index', 'water_index'), pools):
            inputs[key] = (torch.as_tensor(held.indices, device=x.device)
                           if held is not None else None)
        pools = tuple(held.pool if held is not None else None for held in pools)

        # every scan's own frequency where the batch's differ: gathered from the pool's cache on
        # the device where it applies, else uploaded with the batch - never between the graphs
        if sf_samples is None:
            inputs['sf'] = None
        elif pools[0] is not None:
            inputs['sf'] = pools[0].cached('RawProcessor.sf_mhz',
                                           self._pool_sf_mhz)[inputs['index']]
        else:
            inputs['sf'] = torch.as_tensor(np.asarray(sf_samples, dtype=float), device=x.device)

        # the ppm windows, the only thing the frequency decides that a graph must know
        n_w = int(w.shape[-1]) if w is not None else n
        cut = 1 if self.truncate else 0
        spans = dict(align=engine.window_spans(n, sw_hz, sf_mhz, (0.2, 4.2)),
                     water_align=engine.window_spans(n_w, sw_hz, sf_mhz, (0, 8)),
                     shift=engine.window_spans(4 * (n - cut), sw_hz, sf_mhz, (2.9, 3.1)),
                     water_phase=engine.window_spans(4 * (n_w - cut), sw_hz, sf_mhz,
                                                     (4.55, 4.7)))
        steps = lambda given: self._torch_steps(given, list(tags), list(wtags), sw_hz, spans,
                                                pools, x.shape, spatial)
        values = {'sf_mhz': sf_mhz, 'sf_samples': sf_samples, 'met': x}

        tensors = [t for t in list(inputs.values()) + [x] if t is not None]
        if (self.CUDA_GRAPHS and x.device.type == 'cuda'
                and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))):
            layout = lambda t: None if t is None else (tuple(t.shape), t.stride(), t.dtype)
            signature = (str(x.device), tuple(tags), tuple(wtags), sw_hz,
                         tuple(sorted(spans.items())), tuple(id(p) for p in pools),
                         self._engine_settings(), layout(x), spatial,
                         tuple((key, layout(t)) for key, t in inputs.items()))
            graphs = self.__dict__.get('_graphs')
            if graphs is None:
                graphs = self._graphs = engine.GraphedSteps()
            with torch.cuda.device(x.device):
                report = graphs(signature, steps, inputs, values,
                                keep=tuple(p for p in pools if p is not None))
        else:
            report = engine.run_steps(steps(inputs), values)

        tags[:], wtags[:] = report['tags'], report['wtags']
        self._dropped_tags |= report['dropped']
        self._dropped_water_tags |= report['water_dropped']
        if report['no_prewhiten']:
            self._warn_no_prewhiten()
        if report['alignment'] is not None:
            self.last_alignment_ = report['alignment']
        if report['keep_mask'] is not None:
            self.last_keep_mask_ = report['keep_mask']
        self.dim_masks_ = report['dim_masks']
        met_out, wat_out = report['met'], report['wat']
        if not ops.is_torch(met):
            met_out = ops.match_backend(met_out.numpy(), met)
            wat_out = ops.match_backend(wat_out.numpy(), wat) if wat_out is not None else None
            self.dim_masks_ = {tag: mask.numpy() for tag, mask in self.dim_masks_.items()}
        return met_out, wat_out

    def _engine_settings(self):
        """Every setting that decides which kernels the torch engine runs."""
        return (self.conj, self.coil, self.align, self.remove_outliers, self.average, self.ecc,
                self.truncate, self.remove_water, self.shift_ref, self.phase_correct,
                self.coil_method, self.remove_method, self.average_method, self.ecc_method,
                self.water_removal_method, self.shift_ref_method, self.phase_correct_method)

    def _torch_steps(self, inputs, tags, wtags, sw_hz, spans, pools, shape, spatial):
        """
        The torch engine as a stepwise computation ("torch_engine.Step").

        It reads nothing that varies from batch to batch but *inputs*, uses the
        spectrometer frequency only inside its steps, and changes nothing of
        the processor's: what a call leaves behind is reported instead.

        Args:
            inputs: 'met' and 'wat' (spectral axis last, water optional), the
                'coil_mask' and 'dyn_mask' (or None), and the pool indices of
                data and water, 'index' and 'water_index', where their pool's
                caches apply (else None).
            tags: The data's higher-dimension tags, a copy changed as
                dimensions go.
            wtags: The water's, likewise.
            sw_hz: Spectral width in Hz.
            spans: The bins of every ppm window used, by purpose.
            pools: The TensorPools of data and water whose caches apply, or None.
            shape: The data's "(B, V, C, D, T)" layout shape; 'met' is None where the
                coil combination reads the batch in a step instead (see "_process_torch").
            spatial: Its spatial shape, for the way back.

        Returns:
            dict with 'met' and 'wat' in the output layout, 'dim_masks',
            'alignment' and 'keep_mask' (or None), the remaining 'tags' and
            'wtags', the 'dropped' and 'water_dropped' tags, and whether
            prewhitening had to be dropped ('no_prewhiten').
        """
        import torch
        from augmentrum.processing import torch_engine as engine

        report = dict(dropped=set(), water_dropped=set(), no_prewhiten=False, alignment=None,
                      keep_mask=None)
        entry_tags, entry_wtags = list(tags), list(wtags)
        x, w = inputs['met'], inputs['wat']
        b, v, c, d, n = shape
        coil_mask, dyn_mask = inputs['coil_mask'], inputs['dyn_mask']

        if self.conj:
            x = torch.conj_physical(x)
            w = torch.conj_physical(w) if w is not None else None

        if self.coil and 'DIM_COIL' in tags and c > 1:
            if w is not None:
                if 'DIM_COIL' not in wtags:
                    raise ValueError('The water reference has no coil dimension to combine '
                                     'alongside the data.')
                if w.shape[2] != c:
                    raise ValueError('Reference and data coil dimension does not match.')
            if self.coil_method == 'fsl-mrs':
                x, w = yield from self._torch_wsvd(x, w, coil_mask, dyn_mask, inputs, pools,
                                                   entry_tags, entry_wtags, report, shape)
            elif self.coil_method == 'adaptive':
                if coil_mask is not None:
                    raise NotImplementedError("Adaptive coil combination takes no coil masks; "
                                              "use coil_method='fsl-mrs'.")
                # its eigenvector estimate is NumPy's: a step outside any graph
                x, w = yield engine.Step(self._torch_adaptive, x, w, dyn_mask)
            else:
                raise ValueError(f"Unknown tensor coil combination method: {self.coil_method}")
            tags.remove('DIM_COIL')
            report['dropped'].add('DIM_COIL')
            if w is not None:
                wtags.remove('DIM_COIL')
                report['water_dropped'].add('DIM_COIL')
            coil_mask = None

        if self.align:
            x, coil_mask, estimates = self._torch_align(
                x, tags, report['dropped'], coil_mask, dyn_mask, sw_hz, spans['align'])
            if estimates is not None:
                report['alignment'] = estimates
            if w is not None:
                w = self._torch_align(w, wtags, report['water_dropped'], None, None,
                                      sw_hz, spans['water_align'])[0]

        valid = dyn_mask
        if self.remove_outliers:
            if self.remove_method != 'fsl-mrs':
                raise ValueError(f"Unknown tensor outlier removal method: {self.remove_method}")
            if 'DIM_DYN' in tags and x.shape[3] > 1:
                if spatial != (1, 1, 1) or len(tags) != 1:
                    raise ValueError('Outlier removal is only specified for SVS data with a '
                                     'single dynamic dimension (as in FSL-MRS remove_unlike).')
                every = torch.ones(b, x.shape[3], dtype=torch.bool, device=x.device)
                valid = engine.unlike_mask(x[:, 0, 0], every if dyn_mask is None else dyn_mask)
                report['keep_mask'] = valid.reshape(b, 1, 1, 1, -1)

        remaining = {}
        if self.average and self.average_method != 'fsl-mrs':
            raise ValueError(f"Unknown tensor averaging method: {self.average_method}")
        if 'DIM_DYN' in tags and x.shape[3] > 1:
            if not self.average:
                if valid is not None:
                    # no average to consume the mask: removal is expressed as zeros
                    x = x * valid.to(x.real.dtype)[:, None, None, :, None]
                    remaining['DIM_DYN'] = valid
            else:
                if valid is None:
                    x = x.mean(dim=3, keepdim=True)
                else:
                    weights = valid.to(x.real.dtype)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                    x = (x * weights[:, None, None, :, None]).sum(dim=3, keepdim=True)
                tags.remove('DIM_DYN')
                report['dropped'].add('DIM_DYN')
        if self.average and w is not None and 'DIM_DYN' in wtags and w.shape[3] > 1:
            w = w.mean(dim=3, keepdim=True)
            wtags.remove('DIM_DYN')
            report['water_dropped'].add('DIM_DYN')

        if coil_mask is not None and 'DIM_COIL' in tags:
            x = x * coil_mask.to(x.real.dtype)[:, None, :, None, None]
            remaining['DIM_COIL'] = coil_mask

        for group, dropped, data in ((tags, report['dropped'], x),
                                     (wtags, report['water_dropped'], w)):
            if data is not None and ('DIM_DYN' in group or 'DIM_COIL' in group):
                for tag in list(group):
                    if data.shape[2 if tag == 'DIM_COIL' else 3] == 1:
                        group.remove(tag)
                        dropped.add(tag)

        x, w = yield from self._torch_corrections(x, w, tags, wtags, sw_hz, spans,
                                                  inputs['sf'])

        report.update(
            met=self._from_torch_layout(x, spatial, tags),
            wat=self._from_torch_layout(w, spatial, wtags) if w is not None else None,
            dim_masks={tag: mask for tag, mask in remaining.items() if tag in tags},
            tags=tags, wtags=wtags)
        return report

    def _torch_wsvd(self, x, w, coil_mask, dyn_mask, inputs, pools, tags, wtags, report, shape):
        """
        wSVD coil combination of the torch engine, over each sample's drawn
        coils; a stepwise computation, like "_torch_steps".

        The noise covariance pools the last tenth of every drawn transient (as
        FSL-MRS estimate_noise_cov does over the gathered array), from moments
        the pool caches per subject and transient where it can; the weights
        come from the water averaged over its transients, or from each
        transient itself without a water.
        """
        import torch
        from augmentrum.processing import torch_engine as engine

        b, v, c, d, n = shape
        tail = n - int((1 - engine.NOISE_FRACTION) * n)
        if pools[0] is not None:
            second, first = pools[0].cached(
                ('RawProcessor.noise_moments', tuple(tags), n),
                lambda pool: self._pool_noise_moments(pool.data, tags, tail))
            index = inputs['index']
            second, first = second[index], first[index]
            if self.conj:
                second, first = second.conj(), first.conj()
        else:
            second, first = engine.noise_moments(x[..., n - tail:])
        cov, samples = engine.noise_covariance(second, first, v * tail, dyn_mask)

        active = (coil_mask if coil_mask is not None
                  else torch.ones(b, c, dtype=torch.bool, device=cov.device))
        whiten = samples >= engine.MIN_SAMPLES_PER_COIL * active.sum(dim=1)
        if coil_mask is None and dyn_mask is None and v * d * tail < \
                engine.MIN_SAMPLES_PER_COIL * c:
            report['no_prewhiten'] = True

        if w is None:
            gram = engine.reference_gram(x.permute(0, 1, 3, 4, 2))          # (B, V, D, C, C)
            weights = yield from engine.wsvd_weight_steps(gram, cov, active, whiten, False)
            return engine.combine_coils(x, weights.to(x.dtype)).unsqueeze(2), None

        if pools[1] is not None:
            gram = pools[1].cached(
                ('RawProcessor.water_gram', tuple(wtags), n),
                lambda pool: self._pool_water_gram(pool.water, wtags))
            gram = gram[inputs['water_index']]
            gram = gram.conj() if self.conj else gram
        else:
            gram = engine.reference_gram(w.mean(dim=3).transpose(-1, -2))   # (B, V, C, C)
        weights = yield from engine.wsvd_weight_steps(gram, cov, active, whiten, True)
        if x is None:
            return (yield engine.Step(self._combine_raw, weights, w, late=('met',)))
        x = engine.combine_coils(x, weights.to(x.dtype))
        w = engine.combine_coils(w, weights.to(w.dtype))
        return x.unsqueeze(2), w.unsqueeze(2)

    @staticmethod
    def _combine_raw(weights, water, met):
        """The coil combination of a batch that stayed outside the graphs (see "_torch_steps")."""
        from augmentrum.processing import torch_engine as engine

        return (engine.combine_coils(met, weights.to(met.dtype)).unsqueeze(2),
                engine.combine_coils(water, weights.to(water.dtype)).unsqueeze(2))

    @staticmethod
    def _pool_holds(origin, role, tags, n):
        """Whether *origin* is a pool entry of *role* laid out like the tensor at hand."""
        if origin is None or origin.role != role:
            return False
        pool_tags = origin.pool.data_tags if role == 'data' else origin.pool.water_tags
        tensor = origin.pool.data if role == 'data' else origin.pool.water
        return ([t for t in (pool_tags or []) if t] == list(tags)
                and ops.shape(tensor)[4] == n)

    def _pool_noise_moments(self, pool_data, tags, tail, subjects=8):
        """Per-subject, per-transient noise moments of a whole pool, a few subjects at a time."""
        import torch
        from augmentrum.processing import torch_engine as engine

        layout = self._to_torch_layout(move_axis(pool_data, 4, -1), tags)[0]
        moments = [engine.noise_moments(chunk[..., chunk.shape[-1] - tail:])
                   for chunk in layout.split(subjects)]
        return torch.cat([m[0] for m in moments]), torch.cat([m[1] for m in moments])

    @staticmethod
    def _pool_sf_mhz(pool):
        """The pooled subjects' spectrometer frequencies in MHz, a tensor on the pool's device."""
        import torch

        values = [float(v[0] if hasattr(v, '__getitem__') else v)
                  for v in (n.spectrometer_frequency for n in pool.source[0].nifti_list)]
        return torch.tensor(values, dtype=torch.float64, device=pool.device)

    def _pool_water_gram(self, pool_water, wtags):
        """The Gram matrix of every pooled water, averaged over its transients."""
        from augmentrum.processing import torch_engine as engine

        layout = self._to_torch_layout(move_axis(pool_water, 4, -1), wtags)[0]
        return engine.reference_gram(layout.mean(dim=3).transpose(-1, -2))

    def _torch_adaptive(self, x, w, dyn_mask):
        """
        The adaptive combination in the torch engine's layout: its eigenvector
        estimate stays in NumPy (as in the reference), the combination on the device.
        """
        import torch
        from augmentrum.processing.utils import estimate_csm

        if w is not None:
            ref = w.mean(dim=3)
        elif dyn_mask is None:
            ref = x.mean(dim=3)
        else:
            weights = dyn_mask.to(x.real.dtype)[:, None, None, :, None]
            ref = (x * weights).sum(dim=3) / weights.sum(dim=3)
        ref = _complex_like(ops.to_numpy(ref)).transpose(0, 1, 3, 2)   # (B, V, T, C)
        phase = np.exp(-1j * np.angle(ref))
        flat = (ref * phase).reshape((-1,) + ref.shape[-2:])
        csm = np.stack([estimate_csm(voxel)[:, 0] for voxel in flat])
        csm = csm.reshape(ref.shape[:2] + (ref.shape[-1],))
        csmsq = np.real((csm * np.conj(csm)).sum(-1, keepdims=True))
        weights = torch.as_tensor(np.conj(csm)[..., None, :] * phase
                                  / (csmsq[..., None] + np.finfo(float).eps), device=x.device)
        x = torch.einsum('bvcdt,bvtc->bvdt', x, weights.to(x.dtype)).unsqueeze(2)
        if w is not None:
            w = torch.einsum('bvcdt,bvtc->bvdt', w, weights.to(w.dtype)).unsqueeze(2)
        return x, w

    def _torch_align(self, x, tags, dropped, coil_mask, dyn_mask, sw_hz, spans):
        """
        Registration along DIM_DYN, where there is more than one transient.

        A coil dimension still present is cut to its first drawn element first,
        as the list path's copy(remove_dim='DIM_COIL') does with the gathered
        array. The comparison window is given by its bins, *spans*.

        Returns:
            "(x, coil_mask, estimates)": the aligned tensor, the coil mask left
            (None once cut), and the (phi, eps) per voxel and transient, or None.
        """
        import torch
        from augmentrum.processing import torch_engine as engine

        b, v, c, d, n = x.shape
        if 'DIM_DYN' not in tags or d <= 1:
            return x, coil_mask, None
        if 'DIM_COIL' in tags:
            first = (engine.first_true(coil_mask) if coil_mask is not None
                     else torch.zeros(b, dtype=torch.long, device=x.device))
            x = x.gather(2, first.reshape(b, 1, 1, 1, 1).expand(b, v, 1, d, n))
            tags.remove('DIM_COIL')
            dropped.add('DIM_COIL')
            coil_mask = None

        flat = x[:, :, 0].reshape(b * v, d, n)
        valid = (torch.ones(b * v, d, dtype=torch.bool, device=x.device) if dyn_mask is None
                 else dyn_mask.repeat_interleave(v, dim=0))
        phi, eps = engine.align(flat, valid, sw_hz, None, None, spans=spans)
        x = flat * engine.alignment_phasor(phi, eps, n, sw_hz, x.dtype)
        return x.reshape(b, v, 1, d, n), coil_mask, (phi.reshape(b, v, d), eps.reshape(b, v, d))

    def _torch_corrections(self, x, w, tags, wtags, sw_hz, spans, sf=None):
        """
        Eddy current correction, truncation, water removal, referencing and
        phasing; a stepwise computation, like "_torch_steps", on the windows'
        bins *spans*. *sf* is every sample's own spectrometer frequency (MHz,
        a (B,) tensor on the device) where the batch's scans differ, else None.
        """
        import torch
        from augmentrum.processing import torch_engine as engine

        if self.ecc:
            if w is not None and wtags and (wtags != tags or w.shape != x.shape):
                raise ValueError('Reference and data shape must match or the reference must be '
                                 'a single FID per voxel (as in eddy current correction).')
            ref = w if w is not None else x
            if self.ecc_method == 'smoothed':
                phase = engine.ecc_phase(ref)
            elif self.ecc_method == 'fsl-mrs':
                phase = torch.angle(ref.to(engine.complex_of(ref)))
            else:
                raise ValueError(f"Unknown ECC method: {self.ecc_method}")
            phasor = torch.polar(torch.ones_like(phase), -phase)
            x = x * phasor.to(x.dtype)
            w = w * phasor.to(w.dtype) if w is not None else None

        if self.truncate:
            x = x[..., 1:]
            w = w[..., 1:] if w is not None else None

        if self.remove_water:
            # HLSVD runs in NumPy, on each scan's frequency
            x = yield engine.Step(self.water_removal, x, sw_hz, late=('sf_mhz', 'sf_samples'))

        n = x.shape[-1]
        if self.shift_ref:
            if self.shift_ref_method != 'fsl-mrs':
                raise ValueError(f"Unknown tensor frequency shifting method: "
                                 f"{self.shift_ref_method}")
            flat = x.reshape(-1, n)
            if sf is None:
                shift = yield from engine.peak_shift_steps(flat, sw_hz, spans['shift'], 3.027)
            else:
                # every FID against its own scan's frequency, inside the graph
                own = sf[:, None].expand(x.shape[0], flat.shape[0] // x.shape[0]).reshape(-1)
                shift = engine.peak_shift_each(flat, sw_hz, spans['shift'], 3.027, own)
            x = x * engine.shift_phasor(shift, n, sw_hz, x.dtype).reshape(x.shape[:-1] + (n,))

        if self.phase_correct:
            if self.phase_correct_method != 'fsl-mrs':
                raise ValueError(f"Unknown tensor phase correction method: "
                                 f"{self.phase_correct_method}")

            def phased(data, window):
                angle = engine.peak_phase(data.reshape(-1, data.shape[-1]), sw_hz, None, None,
                                          window)
                factor = torch.polar(torch.ones_like(angle), angle)
                return data * factor.reshape(data.shape[:-1] + (1,)).to(data.dtype)

            x = phased(x, spans['shift'])
            w = phased(w, spans['water_phase']) if w is not None else None
        return x, w

    @staticmethod
    def _to_torch_layout(array, tags):
        """
        (B, X, Y, Z, <tags>, T) as a torch (B, V, C, D, T), with the spatial shape.

        Absent dimensions become singletons; NumPy input is wrapped, not copied.
        """
        import torch

        x = array if ops.is_torch(array) else torch.from_numpy(np.asarray(ops.to_numpy(array)))
        spatial = tuple(x.shape[1:4])
        present = list(tags)
        for tag in RawProcessor.TORCH_DIMS:
            if tag not in present:
                x = x.unsqueeze(-2)
                present.append(tag)
        order = [0, 1, 2, 3] + [4 + present.index(t) for t in RawProcessor.TORCH_DIMS] \
            + [x.dim() - 1]
        x = x.permute(order)
        return x.reshape((x.shape[0], -1) + tuple(x.shape[4:])), spatial

    @staticmethod
    def _from_torch_layout(x, spatial, tags):
        """(B, V, C, D, T) back to (B, X, Y, Z, <tags>, T); other dimensions must be singletons."""
        b, _, c, d, n = x.shape
        y = x.reshape((b,) + tuple(spatial) + (c, d, n))
        present = list(RawProcessor.TORCH_DIMS)
        for i in reversed(range(len(present))):
            if present[i] not in tags:
                if y.shape[4 + i] != 1:
                    raise RuntimeError(f"{present[i]} was dropped while still {y.shape[4 + i]} "
                                       f"long.")
                y = y.squeeze(4 + i)
                present.pop(i)
        order = [0, 1, 2, 3] + [4 + present.index(t) for t in tags] + [y.dim() - 1]
        return y.permute(order)

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
        search FSL-MRS runs, for full parity. 'pattern' folds the closed-form optimal
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
            method: 'fsl-mrs' (Powell, parity) or 'pattern' (vectorized search, speed).

        Returns:
            Accumulated (phi, eps) per transient, each (..., D).
        """
        fids = _complex_like(fids)
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
        fids = _complex_like(fids)
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

        Mirrors suspect's sliding_gaussian as used by nifti_ecc_smoothed: edge-padded
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
        fids = _complex_like(fids)
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
        The data's tags minus the dimensions this run collapsed on the data.
        """
        return self._remaining_tags(source, self._dropped_tags)

    def _output_water_dim_tags(self, source):
        """
        The water's tags minus the dimensions this run collapsed on the water.

        Tracked apart from the data's: the water keeps a transient the data
        averages away, or has its lone one squeezed while the data's stay.
        """
        return self._remaining_tags(source, self._dropped_water_tags)

    @staticmethod
    def _remaining_tags(source, dropped):
        """The source's tags without *dropped*, padded to the three NIfTI-MRS slots."""
        tags = [t for t in (source.dim_tags or []) if t and t not in dropped]
        return (tags + [None, None, None])[:3]