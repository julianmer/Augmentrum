"""
Tests for NIfTI_RawProcessor processing functionality.

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
from augmentrum.processing.nifti_raw_processor import NIfTI_RawProcessor
from augmentrum.core import Backend


#**************************************************************************************************#
#                            Class TestNIfTIRawProcessorInitialization                             #
#**************************************************************************************************#
#                                                                                                  #
# Test NIfTI_RawProcessor initialization.                                                          #
#                                                                                                  #
#**************************************************************************************************#
class TestNIfTIRawProcessorInitialization:
    """Test NIfTI_RawProcessor initialization."""

    def test_default_initialization(self):
        """Test processor initializes with default parameters."""
        processor = NIfTI_RawProcessor()

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
        processor = NIfTI_RawProcessor(
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
        processor = NIfTI_RawProcessor(
            coil_method='adaptive',
            ecc_method='own',
            volatile=True
        )

        assert processor.coil_method == 'adaptive'
        assert processor.ecc_method == 'own'

    def test_supported_backends(self):
        """Test processor declares correct supported backends."""
        processor = NIfTI_RawProcessor()

        assert Backend.NIFTI_LIST in processor.SUPPORTED_BACKENDS


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
        processor = NIfTI_RawProcessor(conj=True, coil=False, align=False,
                                      remove_outliers=False, average=False,
                                      ecc=False, volatile=True)
        result_data, result_water = processor.process_nifti_list([], None)

        assert result_data == []
        assert result_water is None

    def test_process_single_subject(self, dummy_nifti_single_coil):
        """Test processing single subject."""
        processor = NIfTI_RawProcessor(conj=True, coil=False, align=False,
                                      remove_outliers=False, average=False,
                                      ecc=False, volatile=True)
        data_list = [dummy_nifti_single_coil]
        result_data, result_water = processor.process_nifti_list(data_list, None)

        assert len(result_data) == 1
        assert result_water is None
        assert result_data[0] is not None

    def test_process_multiple_subjects(self, dummy_nifti_single_coil):
        """Test processing multiple subjects."""
        processor = NIfTI_RawProcessor(conj=True, coil=False, align=False,
                                      remove_outliers=False, average=False,
                                      ecc=False, volatile=True)
        data_list = [deepcopy(dummy_nifti_single_coil) for _ in range(3)]
        result_data, result_water = processor.process_nifti_list(data_list, None)

        assert len(result_data) == 3
        assert result_water is None

    def test_process_with_water_reference(self, dummy_nifti_single_coil, dummy_nifti_water):
        """Test processing with water reference."""
        processor = NIfTI_RawProcessor(conj=True, coil=False, align=False,
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

        processor = NIfTI_RawProcessor(
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

        processor = NIfTI_RawProcessor(
            conj=False, coil=False, align=False, remove_outliers=False,
            average=False, ecc=False, shift_ref=False, phase_correct=False,
            volatile=True
        )

        result, _ = processor._process_single(test_nifti)

        np.testing.assert_array_almost_equal(result[:], original_data)

    def test_coil_combination_fsl_method(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test coil combination with FSL-MRS method."""
        processor = NIfTI_RawProcessor(
            conj=False, coil=True, align=False, remove_outliers=False,
            average=False, ecc=False, coil_method='fsl-mrs', volatile=True
        )

        result_met, result_wat = processor._process_single(dummy_nifti_mrs, dummy_nifti_water)

        assert 'DIM_COIL' not in result_met.dim_tags

    def test_coil_combination_adaptive_method(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test coil combination with adaptive method."""
        processor = NIfTI_RawProcessor(
            conj=False, coil=True, align=False, remove_outliers=False,
            average=False, ecc=False, coil_method='adaptive', volatile=True
        )

        result_met, result_wat = processor._process_single(dummy_nifti_mrs, dummy_nifti_water)

        assert 'DIM_COIL' not in result_met.dim_tags

    def test_averaging_step(self, dummy_nifti_mrs):
        """Test averaging of dynamics."""
        processor = NIfTI_RawProcessor(
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
        processor = NIfTI_RawProcessor(volatile=True)
        result_met, result_wat = processor.coil_combine(dummy_nifti_single_coil, None)

        assert result_met is not None
        assert result_wat is None

    def test_coil_combine_fsl_method(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test FSL-MRS coil combination method."""
        processor = NIfTI_RawProcessor(coil_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor.coil_combine(dummy_nifti_mrs, dummy_nifti_water)

        assert 'DIM_COIL' not in result_met.dim_tags
        if result_wat is not None:
            assert 'DIM_COIL' not in result_wat.dim_tags

    def test_coil_combine_adaptive_method(self, dummy_nifti_mrs, dummy_nifti_water):
        """Test adaptive coil combination method."""
        processor = NIfTI_RawProcessor(coil_method='adaptive', volatile=True)
        result_met, result_wat = processor.coil_combine(dummy_nifti_mrs, dummy_nifti_water)

        assert 'DIM_COIL' not in result_met.dim_tags

    def test_coil_combine_methods_exist(self):
        """Test that both coil combination methods are configurable."""
        processor_fsl = NIfTI_RawProcessor(coil_method='fsl-mrs', volatile=True)
        processor_adaptive = NIfTI_RawProcessor(coil_method='adaptive', volatile=True)

        assert processor_fsl.coil_method == 'fsl-mrs'
        assert processor_adaptive.coil_method == 'adaptive'

    def test_coil_combine_single_coil_no_error(self, dummy_nifti_single_coil):
        """Test that single coil data passes through without error."""
        processor = NIfTI_RawProcessor(coil_method='invalid', volatile=True)
        result_met, result_wat = processor.coil_combine(dummy_nifti_single_coil, None)

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

        processor = NIfTI_RawProcessor(volatile=True)
        result_met, result_wat = processor.registration(data, None)

        assert result_met is not None
        assert 'DIM_DYN' in result_met.dim_tags

    def test_registration_without_dynamics(self, dummy_nifti_single_coil):
        """Test registration on data without dynamics."""
        processor = NIfTI_RawProcessor(volatile=True)
        result_met, result_wat = processor.registration(dummy_nifti_single_coil, None)

        assert result_met is not None

    def test_registration_methods_exist(self):
        """Test that registration method parameter is stored."""
        processor_fsl = NIfTI_RawProcessor(registration_method='fsl-mrs', volatile=True)

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

        processor = NIfTI_RawProcessor(volatile=True)
        result_met, result_wat = processor.combine_averages(data, None)

        if 'DIM_DYN' in result_met.dim_tags:
            assert result_met.shape[result_met.dim_position('DIM_DYN')] == 1

    def test_averaging_without_dynamics(self, dummy_nifti_single_coil):
        """Test averaging on data without dynamics."""
        processor = NIfTI_RawProcessor(volatile=True)
        result_met, result_wat = processor.combine_averages(dummy_nifti_single_coil, None)

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
        processor = NIfTI_RawProcessor(ecc_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor.eddy_current_correction(
            dummy_nifti_single_coil, None
        )

        assert result_met is not None

    def test_ecc_own_method(self, dummy_nifti_single_coil):
        """Test custom ECC method."""
        processor = NIfTI_RawProcessor(ecc_method='own', volatile=True)
        result_met, result_wat = processor.eddy_current_correction(
            dummy_nifti_single_coil, None
        )

        assert result_met is not None

    def test_ecc_invalid_method(self, dummy_nifti_single_coil):
        """Test that invalid method raises error."""
        processor = NIfTI_RawProcessor(ecc_method='invalid', volatile=True)

        with pytest.raises(ValueError, match="Unknown ECC method"):
            processor.eddy_current_correction(dummy_nifti_single_coil, None)


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
        processor = NIfTI_RawProcessor(water_removal_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor.water_removal(dummy_nifti_single_coil, None)

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
        processor = NIfTI_RawProcessor(shift_ref_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor.shift_to_reference(dummy_nifti_single_coil, None)

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
        processor = NIfTI_RawProcessor(phase_correct_method='fsl-mrs', volatile=True)
        result_met, result_wat = processor.phase_correction(dummy_nifti_single_coil, None)

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
        processor = NIfTI_RawProcessor(
            conj=True, coil=True, align=True, remove_outliers=True,
            average=True, ecc=True, truncate=False, remove_water=False,
            shift_ref=True, phase_correct=True, volatile=True
        )

        result_met, result_wat = processor._process_single(dummy_nifti_mrs, dummy_nifti_water)

        assert result_met is not None
        assert 'DIM_COIL' not in result_met.dim_tags

    def test_multiple_subjects_through_pipeline(self, dummy_nifti_mrs):
        """Test processing multiple subjects through pipeline."""
        processor = NIfTI_RawProcessor(volatile=True)
        data_list = [deepcopy(dummy_nifti_mrs) for _ in range(3)]

        result_data, result_water = processor.process_nifti_list(data_list, None)

        assert len(result_data) == 3
        assert all(r is not None for r in result_data)

    def test_pipeline_preserves_list_length(self, dummy_nifti_single_coil):
        """Test that pipeline preserves number of subjects."""
        processor = NIfTI_RawProcessor(volatile=True)
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
        processor = NIfTI_RawProcessor(volatile=True)
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

    def test_nifti_list_backend_supported(self):
        """Test that NIFTI_LIST backend is supported."""
        processor = NIfTI_RawProcessor()

        assert Backend.NIFTI_LIST in processor.SUPPORTED_BACKENDS

    def test_other_backends_not_supported(self):
        """Test that other backends are not supported."""
        processor = NIfTI_RawProcessor()

        assert Backend.NUMPY not in processor.SUPPORTED_BACKENDS
        assert Backend.PYTORCH not in processor.SUPPORTED_BACKENDS
        assert Backend.TENSORFLOW not in processor.SUPPORTED_BACKENDS
