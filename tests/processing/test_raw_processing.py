"""
Tests for RawProcessor: the FSL-MRS list engine, the tensor engine, and their parity.

Tests cover:
- Processor initialization and configuration
- Processing individual steps (conjugate, coil combine, etc.)
- Full pipeline integration
- Backend compatibility
- Edge cases and error handling
"""

import pytest
import numpy as np
from copy import deepcopy

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from augmentrum.processing.raw_processing import RawProcessor
from augmentrum.core import NIfTI_MRS_Plus, Backend


#**************************************************************************************************#
#                               Class TestRawProcessorInitialization                               #
#**************************************************************************************************#
#                                                                                                  #
# Test RawProcessor initialization.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class TestRawProcessorInitialization:
    """Test RawProcessor initialization."""

    def test_default_initialization(self):
        """Test processor initializes with default parameters."""
        processor = RawProcessor()

        assert processor.conj == True
        assert processor.coil == True
        assert processor.align == True
        assert processor.remove_outliers == True
        assert processor.average == True
        assert processor.ecc == True
        assert processor.truncate == False
        assert processor.remove_water == False
        assert processor.shift_ref == True
        assert processor.phase_correct == True

    def test_custom_initialization(self):
        """Test processor initializes with custom parameters."""
        processor = RawProcessor(
            conj=False,
            coil=False,
            align=False,
            remove_outliers=False,
            average=False,
            ecc=False,
            truncate=True,
            remove_water=True,
            shift_ref=False,
            phase_correct=False
        )

        assert processor.conj == False
        assert processor.coil == False
        assert processor.truncate == True
        assert processor.remove_water == True

    def test_method_initialization(self):
        """Test processor initializes with custom methods."""
        processor = RawProcessor(
            coil_method='adaptive',
            ecc_method='smoothed',
            volatile=True
        )

        assert processor.coil_method == 'adaptive'
        assert processor.ecc_method == 'smoothed'

    def test_supported_backends(self):
        """One module, every backend: the list engine and the tensor engine."""
        processor = RawProcessor()

        for backend in Backend:
            assert backend in processor.SUPPORTED_BACKENDS, backend

    def test_pattern_registration_narrows_to_the_tensor_engine(self):
        """'pattern' exists only batched, so that config routes list batches."""
        processor = RawProcessor(registration_method='pattern')

        assert Backend.NIFTI_LIST not in processor.SUPPORTED_BACKENDS
        assert Backend.NUMPY in processor.SUPPORTED_BACKENDS


#**************************************************************************************************#
#                                    Class TestProcessNIfTIList                                    #
#**************************************************************************************************#
#                                                                                                  #
# Test process_nifti_list method.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class TestProcessNIfTIList:
    """Test process_nifti_list method."""

    def test_process_empty_list(self):
        """Test processing empty list returns empty lists."""
        processor = RawProcessor(conj=True, coil=False, align=False,
                                      remove_outliers=False, average=False,
                                      ecc=False, volatile=True)
        result_data, result_water = processor.process_nifti_list([], None)

        assert result_data == []
        assert result_water is None

    def test_process_single_subject(self, dummy_nifti_single_coil):
        """Test processing single subject."""
        processor = RawProcessor(conj=True, coil=False, align=False,
                                      remove_outliers=False, average=False,
                                      ecc=False, volatile=True)
        data_list = [dummy_nifti_single_coil]
        result_data, result_water = processor.process_nifti_list(data_list, None)

        assert len(result_data) == 1
        assert result_water is None
        assert result_data[0] is not None

    def test_process_multiple_subjects(self, dummy_nifti_single_coil):
        """Test processing multiple subjects."""
        processor = RawProcessor(conj=True, coil=False, align=False,
                                      remove_outliers=False, average=False,
                                      ecc=False, volatile=True)
        data_list = [deepcopy(dummy_nifti_single_coil) for _ in range(3)]
        result_data, result_water = processor.process_nifti_list(data_list, None)

        assert len(result_data) == 3
        assert result_water is None

    def test_process_with_water_reference(self, dummy_nifti_single_coil, dummy_nifti_water):
        """Test processing with water reference."""
        processor = RawProcessor(conj=True, coil=False, align=False,
                                      remove_outliers=False, average=False,
                                      ecc=False, volatile=True)
        data_list = [dummy_nifti_single_coil]
        water_list = [dummy_nifti_water]

        result_data, result_water = processor.process_nifti_list(data_list, water_list)

        assert len(result_data) == 1
        assert len(result_water) == 1
        assert result_water[0] is not None


#**************************************************************************************************#
#                                Class TestProcessingSingleSubject                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test _process_single method.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestProcessingSingleSubject:
    """Test _process_single method."""

    def test_conjugate_step(self, dummy_nifti_single_coil):
        """Test conjugation step."""
        test_nifti = dummy_nifti_single_coil.copy()
        original_data = test_nifti[:].copy()

        processor = RawProcessor(
            conj=True, coil=False, align=False, remove_outliers=False,
            average=False, ecc=False, shift_ref=False, phase_correct=False,
            volatile=True
        )

        result, _ = processor._process_single(test_nifti)

        np.testing.assert_array_almost_equal(result[:], np.conj(original_data))

    def test_no_conjugate_step(self, dummy_nifti_single_coil):
        """Test that data is unchanged when conjugate is disabled."""
        test_nifti = dummy_nifti_single_coil.copy()
        original_data = test_nifti[:].copy()

        processor = RawProcessor(
            conj=False, coil=False, align=False, remove_outliers=False,
            average=False, ecc=False, shift_ref=False, phase_correct=False,
            volatile=True
        )

        result, _ = processor._process_single(test_nifti)

        np.testing.assert_array_almost_equal(result[:], original_data)

    def test_coil_combination_fsl_method(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test coil combination with FSL-MRS method."""
        processor = RawProcessor(
            conj=False, coil=True, align=False, remove_outliers=False,
            average=False, ecc=False, coil_method='fsl-mrs', volatile=True
        )

        result_met, result_wat = processor._process_single(dummy_nifti_mrs, dummy_nifti_water)

        assert 'DIM_COIL' not in result_met.dim_tags

    def test_coil_combination_adaptive_method(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test coil combination with adaptive method."""
        processor = RawProcessor(
            conj=False, coil=True, align=False, remove_outliers=False,
            average=False, ecc=False, coil_method='adaptive', volatile=True
        )

        result_met, result_wat = processor._process_single(dummy_nifti_mrs, dummy_nifti_water)

        assert 'DIM_COIL' not in result_met.dim_tags

    def test_averaging_step(self, dummy_nifti_mrs):
        """Test averaging of dynamics."""
        processor = RawProcessor(
            conj=False, coil=True, align=False, remove_outliers=False,
            average=True, ecc=False, volatile=True
        )

        result, _ = processor._process_single(dummy_nifti_mrs)

        if 'DIM_DYN' in result.dim_tags:
            assert result.shape[result.dim_position('DIM_DYN')] == 1


#**************************************************************************************************#
#                                    Class TestCoilCombination                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test coil_combine method.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestCoilCombination:
    """Test coil_combine method."""

    def test_coil_combine_single_coil(self, dummy_nifti_single_coil):
        """Test coil combination on single coil data."""
        processor = RawProcessor(volatile=True)
        result_met, result_wat = processor._coil_combine_nifti(dummy_nifti_single_coil, None)

        assert result_met is not None
        assert result_wat is None

    def test_coil_combine_fsl_method(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test FSL-MRS coil combination method."""
        processor = RawProcessor(coil_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor._coil_combine_nifti(dummy_nifti_mrs, dummy_nifti_water)

        assert 'DIM_COIL' not in result_met.dim_tags
        if result_wat is not None:
            assert 'DIM_COIL' not in result_wat.dim_tags

    def test_coil_combine_adaptive_method(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test adaptive coil combination method."""
        processor = RawProcessor(coil_method='adaptive', volatile=True)
        result_met, result_wat = processor._coil_combine_nifti(dummy_nifti_mrs, dummy_nifti_water)

        assert 'DIM_COIL' not in result_met.dim_tags

    def test_coil_combine_methods_exist(self):
        """Test that both coil combination methods are configurable."""
        processor_fsl = RawProcessor(coil_method='fsl-mrs', volatile=True)
        processor_adaptive = RawProcessor(coil_method='adaptive', volatile=True)

        assert processor_fsl.coil_method == 'fsl-mrs'
        assert processor_adaptive.coil_method == 'adaptive'

    def test_coil_combine_single_coil_no_error(self, dummy_nifti_single_coil):
        """Test that single coil data passes through without error."""
        processor = RawProcessor(coil_method='invalid', volatile=True)
        result_met, result_wat = processor._coil_combine_nifti(dummy_nifti_single_coil, None)

        assert result_met is not None


#**************************************************************************************************#
#                                      Class TestRegistration                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test registration method.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestRegistration:
    """Test registration method."""

    def test_registration_with_dynamics(self, dummy_nifti_mrs):
        """Test registration on data with dynamics."""
        data = dummy_nifti_mrs.copy(remove_dim='DIM_COIL')

        processor = RawProcessor(volatile=True)
        result_met, result_wat = processor._registration_nifti(data, None)

        assert result_met is not None
        assert 'DIM_DYN' in result_met.dim_tags

    def test_registration_without_dynamics(self, dummy_nifti_single_coil):
        """Test registration on data without dynamics."""
        processor = RawProcessor(volatile=True)
        result_met, result_wat = processor._registration_nifti(dummy_nifti_single_coil, None)

        assert result_met is not None

    def test_registration_methods_exist(self):
        """Test that registration method parameter is stored."""
        processor_fsl = RawProcessor(registration_method='fsl-mrs', volatile=True)

        assert processor_fsl.registration_method == 'fsl-mrs'


#**************************************************************************************************#
#                                       Class TestAveraging                                        #
#**************************************************************************************************#
#                                                                                                  #
# Test combine_averages method.                                                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestAveraging:
    """Test combine_averages method."""

    def test_averaging_with_dynamics(self, dummy_nifti_mrs):
        """Test averaging on data with dynamics."""
        data = dummy_nifti_mrs.copy(remove_dim='DIM_COIL')

        processor = RawProcessor(volatile=True)
        result_met, result_wat = processor._combine_averages_nifti(data, None)

        if 'DIM_DYN' in result_met.dim_tags:
            assert result_met.shape[result_met.dim_position('DIM_DYN')] == 1

    def test_averaging_without_dynamics(self, dummy_nifti_single_coil):
        """Test averaging on data without dynamics."""
        processor = RawProcessor(volatile=True)
        result_met, result_wat = processor._combine_averages_nifti(dummy_nifti_single_coil, None)

        assert result_met is not None


#**************************************************************************************************#
#                                 Class TestEddyCurrentCorrection                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test eddy_current_correction method.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class TestEddyCurrentCorrection:
    """Test eddy_current_correction method."""

    def test_ecc_fsl_method(self, dummy_nifti_single_coil):
        """Test FSL-MRS ECC method."""
        processor = RawProcessor(ecc_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor._ecc_nifti(
            dummy_nifti_single_coil, None
        )

        assert result_met is not None

    def test_ecc_own_method(self, dummy_nifti_single_coil):
        """Test custom ECC method."""
        processor = RawProcessor(ecc_method='smoothed', volatile=True)
        result_met, result_wat = processor._ecc_nifti(
            dummy_nifti_single_coil, None
        )

        assert result_met is not None

    def test_ecc_invalid_method(self, dummy_nifti_single_coil):
        """Test that invalid method raises error."""
        processor = RawProcessor(ecc_method='invalid', volatile=True)

        with pytest.raises(ValueError, match="Unknown ECC method"):
            processor._ecc_nifti(dummy_nifti_single_coil, None)


#**************************************************************************************************#
#                                      Class TestWaterRemoval                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test water_removal method.                                                                       #
#                                                                                                  #
#**************************************************************************************************#
class TestWaterRemoval:
    """Test water_removal method."""

    def test_water_removal_fsl_method(self, dummy_nifti_single_coil):
        """Test FSL-MRS water removal method."""
        processor = RawProcessor(water_removal_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor._water_removal_nifti(dummy_nifti_single_coil, None)

        assert result_met is not None


#**************************************************************************************************#
#                                   Class TestFrequencyShifting                                    #
#**************************************************************************************************#
#                                                                                                  #
# Test shift_to_reference method.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class TestFrequencyShifting:
    """Test shift_to_reference method."""

    def test_shift_ref_fsl_method(self, dummy_nifti_single_coil):
        """Test FSL-MRS frequency shifting method."""
        processor = RawProcessor(shift_ref_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor._shift_to_reference_nifti(dummy_nifti_single_coil, None)

        assert result_met is not None


#**************************************************************************************************#
#                                    Class TestPhaseCorrection                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test phase_correction method.                                                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestPhaseCorrection:
    """Test phase_correction method."""

    def test_phase_correct_fsl_method(self, dummy_nifti_single_coil):
        """Test FSL-MRS phase correction method."""
        processor = RawProcessor(phase_correct_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor._phase_correction_nifti(dummy_nifti_single_coil, None)

        assert result_met is not None


#**************************************************************************************************#
#                                      Class TestIntegration                                       #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests for complete processing pipeline.                                              #
#                                                                                                  #
#**************************************************************************************************#
class TestIntegration:
    """Integration tests for complete processing pipeline."""

    def test_full_pipeline_with_all_steps(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test full processing pipeline with all steps enabled."""
        processor = RawProcessor(
            conj=True, coil=True, align=True, remove_outliers=True,
            average=True, ecc=True, truncate=False, remove_water=False,
            shift_ref=True, phase_correct=True, volatile=True
        )

        result_met, result_wat = processor._process_single(dummy_nifti_mrs, dummy_nifti_water)

        assert result_met is not None
        assert 'DIM_COIL' not in result_met.dim_tags

    def test_multiple_subjects_through_pipeline(self, dummy_nifti_mrs):
        """Test processing multiple subjects through pipeline."""
        processor = RawProcessor(volatile=True)
        data_list = [deepcopy(dummy_nifti_mrs) for _ in range(3)]

        result_data, result_water = processor.process_nifti_list(data_list, None)

        assert len(result_data) == 3
        assert all(r is not None for r in result_data)

    def test_pipeline_preserves_list_length(self, dummy_nifti_single_coil):
        """Test that pipeline preserves number of subjects."""
        processor = RawProcessor(volatile=True)
        n_subjects = 5
        data_list = [deepcopy(dummy_nifti_single_coil) for _ in range(n_subjects)]

        result_data, _ = processor.process_nifti_list(data_list, None)

        assert len(result_data) == n_subjects


#**************************************************************************************************#
#                                       Class TestEdgeCases                                        #
#**************************************************************************************************#
#                                                                                                  #
# Test edge cases and error handling.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_process_with_mismatched_water_list_length(self, dummy_nifti_single_coil,
                                                        dummy_nifti_water):
        """Test processing with mismatched data and water list lengths."""
        processor = RawProcessor(volatile=True)
        data_list = [dummy_nifti_single_coil, dummy_nifti_single_coil]
        water_list = [dummy_nifti_water]

        with pytest.raises(IndexError):
            processor.process_nifti_list(data_list, water_list)


#**************************************************************************************************#
#                                  Class TestBackendCompatibility                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test backend compatibility.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
class TestBackendCompatibility:
    """Test backend compatibility."""

    def test_every_backend_is_native(self):
        """The list engine serves NIFTI_LIST; the tensor engine all the rest."""
        processor = RawProcessor()

        for backend in Backend:
            assert processor.supports_backend(backend), backend


#*****************************#
#   fsl provenance recording   #
#*****************************#
def test_fsl_provenance_is_recorded(dummy_nifti_single_coil):
    """
    RawProcessor records provenance per executed FSL operation.

    This is the coverage behind the registry test's own_provenance exemption:
    constructor arguments are not echoed into self.params provenance because
    every step that runs writes its own FSL-MRS entry — an absent entry means
    the step did not run, which records what happened rather than the intent.
    """
    processor = RawProcessor(conj=True, coil=False, align=False,
                                   remove_outliers=False, average=False, ecc=True,
                                   shift_ref=True, phase_correct=True, volatile=True)
    out, _ = processor.process_nifti_list([dummy_nifti_single_coil.copy()], None)

    methods = [entry['Method'] for entry in out[0].hdr_ext['ProcessingApplied']]
    assert methods == ['Conjugation', 'Eddy current correction',
                       'Frequency and phase correction', 'Phasing']


#**************************************#
#   synthetic parity data for tensor   #
#**************************************#
# Structured spectra rather than pure noise: peak-based steps (align, shift,
# phase) need real peaks in their search windows for parity to be meaningful.
SW, SF, N_T, N_C, N_D, N_B = 2000.0, 123.0, 512, 3, 6, 2
TAGS = ['DIM_COIL', 'DIM_DYN', None]


def _synth_subject(rng):
    """One subject (met, wat) shaped (1, 1, 1, T, C, D), peaks upright after conj."""
    t = np.arange(N_T) / SW

    def lorentz(ppm, amp, damp):
        return amp * np.exp(2j * np.pi * (ppm - 4.65) * SF * t) * np.exp(-t / damp)

    met0 = (lorentz(2.01, 1.0, 0.08) + lorentz(3.027, 0.8, 0.08)
            + lorentz(3.21, 0.5, 0.08) + lorentz(4.65, 3.0, 0.05))
    wat0 = lorentz(4.65, 50.0, 0.1)
    eddy = np.exp(-1j * 0.3 * np.exp(-t / 0.05))
    sens = rng.normal(1, 0.3, N_C) + 1j * rng.normal(0, 0.3, N_C)

    met = np.zeros((1, 1, 1, N_T, N_C, N_D), dtype=complex)
    wat = np.zeros((1, 1, 1, N_T, N_C, N_D), dtype=complex)
    for d in range(N_D):
        jitter = np.exp(-1j * rng.uniform(-0.2, 0.2)) \
            * np.exp(2j * np.pi * rng.uniform(-3, 3) * t)
        amp = 3.0 if d == 2 else rng.uniform(0.95, 1.05)            # d == 2 is an outlier
        for c in range(N_C):
            met[0, 0, 0, :, c, d] = sens[c] * jitter * amp * met0 * eddy \
                + 0.01 * (rng.normal(size=N_T) + 1j * rng.normal(size=N_T))
            wat[0, 0, 0, :, c, d] = sens[c] * wat0 * eddy \
                + 0.01 * (rng.normal(size=N_T) + 1j * rng.normal(size=N_T))
    return met, wat


def _synth_batch(seed=7):
    """NIfTI objects plus their tensor-layout batches (met T-last, wat untransposed)."""
    rng = np.random.default_rng(seed)
    mets, wats = [], []
    for _ in range(N_B):
        m, w = _synth_subject(rng)
        for arr, out in ((m, mets), (w, wats)):
            nifti = gen_nifti_mrs(arr, 1 / SW, SF)
            nifti.set_dim_tag(4, 'DIM_COIL')
            nifti.set_dim_tag(5, 'DIM_DYN')
            out.append(nifti)
    met_t = np.moveaxis(np.stack([n[:] for n in mets]), 4, -1)      # (B, 1, 1, 1, C, D, T)
    wat_t = np.stack([n[:] for n in wats])                          # (B, 1, 1, 1, T, C, D)
    return mets, wats, met_t, wat_t


ALL_OFF = dict(conj=False, coil=False, align=False, remove_outliers=False, average=False,
               ecc=False, truncate=False, remove_water=False, shift_ref=False,
               phase_correct=False)


def _parity_error(flags, with_water=True, **methods):
    """Max-abs relative error between the NIfTI and tensor paths for *flags*."""
    mets, wats, met_t, wat_t = _synth_batch()
    ref_list, _ = RawProcessor(volatile=True, **flags, **methods).process_nifti_list(
        mets, wats if with_water else None)
    got, _ = RawProcessor(volatile=True, **flags, **methods).process_tensor(
        met_t, wat_t if with_water else None, sw_hz=SW, sf_mhz=SF, dim_tags=TAGS)

    ref = np.stack([np.squeeze(n[:]) for n in ref_list])
    got = np.squeeze(np.asarray(got))
    if ref.shape != got.shape:                  # nifti keeps T first when dims remain
        ref = np.moveaxis(ref, 1, -1)
    return np.abs(ref - got).max() / np.abs(ref).max()


#**************************************************************************************************#
#                                  Class TestTensorParity                                          #
#**************************************************************************************************#
#                                                                                                  #
# Every RawProcessor step against its RawProcessor counterpart.                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestTensorParity:
    """Every RawProcessor step against its RawProcessor counterpart."""

    def test_conjugate(self):
        assert _parity_error({**ALL_OFF, 'conj': True}) < 1e-12

    def test_coil_combine_with_reference(self):
        assert _parity_error({**ALL_OFF, 'coil': True}) < 1e-10

    def test_coil_combine_without_reference(self):
        assert _parity_error({**ALL_OFF, 'coil': True}, with_water=False) < 1e-10

    def test_outliers_and_average(self):
        flags = {**ALL_OFF, 'coil': True, 'remove_outliers': True, 'average': True}
        assert _parity_error(flags) < 1e-10

    @pytest.mark.parametrize('method', ['smoothed', 'fsl-mrs'])
    def test_eddy_current_correction(self, method):
        flags = {**ALL_OFF, 'coil': True, 'average': True, 'ecc': True}
        assert _parity_error(flags, ecc_method=method) < 1e-10

    def test_truncate(self):
        assert _parity_error({**ALL_OFF, 'coil': True, 'average': True, 'truncate': True}) < 1e-10

    def test_shift_to_reference(self):
        assert _parity_error({**ALL_OFF, 'coil': True, 'average': True, 'shift_ref': True}) < 1e-10

    def test_phase_correct(self):
        flags = {**ALL_OFF, 'coil': True, 'average': True, 'phase_correct': True}
        assert _parity_error(flags) < 1e-10

    def test_coil_combine_adaptive_with_reference(self):
        flags = {**ALL_OFF, 'coil': True}
        assert _parity_error(flags, coil_method='adaptive') < 1e-10

    def test_coil_combine_adaptive_without_reference(self):
        flags = {**ALL_OFF, 'coil': True}
        assert _parity_error(flags, with_water=False, coil_method='adaptive') < 1e-10

    def test_water_removal_hlsvd(self):
        flags = {**ALL_OFF, 'coil': True, 'average': True, 'remove_water': True}
        assert _parity_error(flags) < 1e-4

    def test_full_default_pipeline(self):
        """The full pipeline (Powell alignment) tracks the NIfTI path."""
        flags = dict(conj=True, coil=True, align=True, remove_outliers=True, average=True,
                     ecc=True, truncate=False, remove_water=True, shift_ref=True,
                     phase_correct=True)
        assert _parity_error(flags) < 1e-5

    def test_own_alignment_solves_same_objective(self):
        """The fast search reaches Powell-level values of the FSL objective."""
        from augmentrum.processing.utils import fid_to_spec, ppm_window
        align_params = RawProcessor._align_params
        mets, wats, met_t, wat_t = _synth_batch()
        pre = dict(ALL_OFF, conj=True, coil=True)
        ref_list, _ = RawProcessor(volatile=True, **pre).process_nifti_list(mets, wats)
        fids = np.stack([np.squeeze(n[:]).T for n in ref_list])     # (B, D, T)

        costs = {}
        for method in ('fsl-mrs', 'pattern'):
            phi, eps = align_params(fids, SW, SF, (0.2, 4.2), method=method)
            t = np.linspace(1 / SW, N_T / SW, N_T)
            first, last = ppm_window(N_T, SW, SF, (0.2, 4.2))
            avg = fids.mean(axis=1, keepdims=True)
            pick = np.argmin(np.linalg.norm(fids - avg, axis=-1), axis=-1)
            target = np.take_along_axis(fids, pick[:, None, None], axis=1)[:, 0]
            t_win = fid_to_spec(target)[..., first:last]
            aligned = np.exp(-1j * phi[..., None]) * fids \
                * np.exp(-2j * np.pi * t * eps[..., None])
            diff = fid_to_spec(aligned)[..., first:last] - t_win[:, None]
            costs[method] = np.linalg.norm(diff, axis=-1)

        assert np.all(costs['pattern'] <= costs['fsl-mrs'] * 1.1 + 1e-9)


#**************************************************************************************************#
#                                  Class TestTensorWrapper                                         #
#**************************************************************************************************#
#                                                                                                  #
# RawProcessor through the NIfTI_MRS_Plus dispatch: layout, tags, and write-back.                  #
#                                                                                                  #
#**************************************************************************************************#
class TestTensorWrapper:
    """RawProcessor through the NIfTI_MRS_Plus dispatch: layout, tags, write-back."""

    def test_wrapper_collapses_dims_and_matches(self):
        mets, wats, _, _ = _synth_batch()
        data = NIfTI_MRS_Plus(nifti_list=[m.copy() for m in mets], backend=Backend.NUMPY)
        water = NIfTI_MRS_Plus(nifti_list=[w.copy() for w in wats], backend=Backend.NUMPY)

        out, _ = RawProcessor()(data, water)
        got = np.squeeze(np.asarray(out.get_data(Backend.NUMPY)))

        assert out.get_data(Backend.NUMPY).shape == (N_B, 1, 1, 1, N_T)
        assert out.dim_tags == [None, None, None]

        ref_list, _ = RawProcessor(volatile=True).process_nifti_list(mets, wats)
        ref = np.stack([np.squeeze(n[:]) for n in ref_list])
        assert np.abs(ref - got).max() / np.abs(ref).max() < 1e-5

    def test_missing_metadata_raises(self):
        with pytest.raises(ValueError, match='sw_hz'):
            RawProcessor().process_tensor(np.zeros((1, 1, 1, 1, 8), complex))

    def test_outlier_mask_zeroes_without_average(self):
        """Without an average to consume the mask, outliers are zeroed in place."""
        _, _, met_t, _ = _synth_batch()
        processor = RawProcessor(**{**ALL_OFF, 'coil': True, 'remove_outliers': True},
                                 volatile=True)
        out, _ = processor.process_tensor(met_t, sw_hz=SW, sf_mhz=SF, dim_tags=TAGS)

        mask = np.squeeze(processor.last_keep_mask_)
        assert not mask.all() and mask.any(), "the synthetic outlier must be caught"
        out = np.squeeze(np.asarray(out))                       # (B, D, T)
        assert np.abs(out[~mask]).max() == 0, "outliers must be zeroed"
        assert np.abs(out[mask]).max() > 0, "survivors must pass through"


#**************************************************************************************************#
#                                  Class TestTensorGradients                                       #
#**************************************************************************************************#
#                                                                                                  #
# Gradients flow through the signal path on torch, and torch matches numpy.                        #
#                                                                                                  #
#**************************************************************************************************#
class TestTensorGradients:
    """Gradients flow through the signal path on torch, and torch matches numpy."""

    def test_torch_gradients_and_numpy_parity(self):
        torch = pytest.importorskip('torch')
        _, _, met_t, wat_t = _synth_batch()

        met = torch.tensor(met_t, dtype=torch.complex128, requires_grad=True)
        wat = torch.tensor(wat_t, dtype=torch.complex128, requires_grad=True)
        processor = RawProcessor(remove_water=True, volatile=True)
        out_met, out_wat = processor.process_tensor(
            met, wat, backend=Backend.PYTORCH, sw_hz=SW, sf_mhz=SF, dim_tags=TAGS)

        (out_met.abs().sum() + out_wat.abs().sum()).backward()
        for grad in (met.grad, wat.grad):
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert grad.abs().max() > 0

        ref_met, _ = RawProcessor(remove_water=True, volatile=True).process_tensor(
            met_t.copy(), wat_t.copy(), sw_hz=SW, sf_mhz=SF, dim_tags=TAGS)
        err = np.abs(out_met.detach().numpy() - ref_met).max() / np.abs(ref_met).max()
        assert err < 1e-6
