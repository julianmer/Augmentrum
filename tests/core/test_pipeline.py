"""
Tests for AugmentationPipeline functionality.

Tests cover:
- Pipeline creation and initialization
- Adding/removing modules
- Backend compatibility checking
- Automatic backend conversion
- Module execution order
- Integration with BaseModule
- Error handling
"""

import pytest
import numpy as np
from augmentrum.core import BaseModule, NIfTI_MRS_Plus, Backend
from augmentrum.core.pipeline import AugmentationPipeline


#**************************************************************************************************#
#                                    Class TestPipelineCreation                                    #
#**************************************************************************************************#
#                                                                                                  #
# Test pipeline creation and initialization.                                                       #
#                                                                                                  #
#**************************************************************************************************#
class TestPipelineCreation:
    """Test pipeline creation and initialization."""

    def test_create_empty_pipeline(self):
        """Test creating an empty pipeline."""
        pipeline = AugmentationPipeline([])

        assert pipeline is not None
        assert len(pipeline.steps) == 0

    def test_create_pipeline_with_modules(self):
        """Test creating pipeline with modules."""
        class Module1(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        class Module2(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        pipeline = AugmentationPipeline([Module1(), Module2()])

        assert len(pipeline.steps) == 2

    def test_create_pipeline_from_module_instances(self):
        """Test creating pipeline from module instances."""
        class TestModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = TestModule()
        pipeline = AugmentationPipeline([module])

        assert pipeline.steps[0] is module


#**************************************************************************************************#
#                                   Class TestPipelineExecution                                    #
#**************************************************************************************************#
#                                                                                                  #
# Test pipeline execution.                                                                         #
#                                                                                                  #
#**************************************************************************************************#
class TestPipelineExecution:
    """Test pipeline execution."""

    def test_execute_single_module(self, dummy_nifti_list):
        """Test executing pipeline with single module."""
        class CounterModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            call_count = 0

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                CounterModule.call_count += 1
                return data_list, water_list

        pipeline = AugmentationPipeline([CounterModule()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, result_water = pipeline(data=nifti_plus, water=None)

        assert CounterModule.call_count == 1
        assert isinstance(result_data, NIfTI_MRS_Plus)

    def test_execute_multiple_modules_in_order(self, dummy_nifti_list):
        """Test that modules execute in correct order."""
        execution_order = []

        class Module1(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                execution_order.append(1)
                return data_list, water_list

        class Module2(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                execution_order.append(2)
                return data_list, water_list

        class Module3(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                execution_order.append(3)
                return data_list, water_list

        pipeline = AugmentationPipeline([Module1(), Module2(), Module3()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        pipeline(data=nifti_plus, water=None)

        assert execution_order == [1, 2, 3]

    def test_execute_with_water_reference(self, dummy_nifti_list):
        """Test executing pipeline with water reference."""
        class WaterCheckModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            received_water = None

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                WaterCheckModule.received_water = water_list
                return data_list, water_list

        pipeline = AugmentationPipeline([WaterCheckModule()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        water_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list[:2], backend=Backend.NIFTI_LIST)

        pipeline(data=nifti_plus, water=water_plus)

        assert WaterCheckModule.received_water is not None
        assert len(WaterCheckModule.received_water) == 2

    def test_data_flows_through_pipeline(self, dummy_nifti_list):
        """Test that data modifications flow through pipeline."""
        class FilterModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                # Return only first 3 subjects
                return data_list[:3], water_list

        class CountModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            received_count = 0

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                CountModule.received_count = len(data_list)
                return data_list, water_list

        pipeline = AugmentationPipeline([FilterModule(), CountModule()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert CountModule.received_count == 3
        assert len(result_data) == 3


#**************************************************************************************************#
#                              Class TestPipelineBackendCompatibility                              #
#**************************************************************************************************#
#                                                                                                  #
# Test pipeline backend compatibility checking.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestPipelineBackendCompatibility:
    """Test pipeline backend compatibility checking."""

    def test_compatible_backends(self, dummy_nifti_list):
        """Test pipeline with all compatible backends."""
        class Module1(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        class Module2(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        pipeline = AugmentationPipeline([Module1(), Module2()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Should execute without warnings
        result_data, _ = pipeline(data=nifti_plus, water=None)
        assert isinstance(result_data, NIfTI_MRS_Plus)

    def test_backend_conversion_warning(self, dummy_nifti_list):
        """Test that pipeline warns when backend conversion is needed."""
        class NIfTIModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        class NumpyModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NUMPY]
            def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
                return data_array, water_array

        pipeline = AugmentationPipeline([NIfTIModule(), NumpyModule()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Should warn about backend conversion
        with pytest.warns(UserWarning, match="Converting to"):
            result_data, _ = pipeline(data=nifti_plus, water=None)

    def test_modules_with_all_backends_supported(self, dummy_nifti_list):
        """Test modules that support all backends."""
        class AllBackendsModule(BaseModule):
            SUPPORTED_BACKENDS = []  # Empty = all backends

            def forward(self, data, water=None, **kwargs):
                return data, water

        pipeline = AugmentationPipeline([AllBackendsModule(), AllBackendsModule()])

        # Test with NIFTI_LIST
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)
        assert isinstance(result_data, NIfTI_MRS_Plus)

        # Test with NUMPY
        nifti_plus_np = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)
        result_data, _ = pipeline(data=nifti_plus_np, water=None)
        assert isinstance(result_data, NIfTI_MRS_Plus)


#**************************************************************************************************#
#                               Class TestPipelineBackendConversion                                #
#**************************************************************************************************#
#                                                                                                  #
# Test automatic backend conversion.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestPipelineBackendConversion:
    """Test automatic backend conversion."""

    def test_automatic_conversion_nifti_to_numpy(self, dummy_nifti_list):
        """Test automatic conversion from NIFTI_LIST to NUMPY."""
        class NumpyModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NUMPY]
            received_backend = None

            def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
                NumpyModule.received_backend = backend
                return data_array, water_array

        pipeline = AugmentationPipeline([NumpyModule()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        with pytest.warns(UserWarning):
            result_data, _ = pipeline(data=nifti_plus, water=None)

        # Backend should have been converted to NUMPY
        assert NumpyModule.received_backend == Backend.NUMPY

    def test_conversion_preserves_data(self, dummy_nifti_list):
        """Test that backend conversion preserves data."""
        class CheckDataModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NUMPY]
            received_shape = None

            def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
                CheckDataModule.received_shape = data_array.shape
                return data_array, water_array

        pipeline = AugmentationPipeline([CheckDataModule()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        with pytest.warns(UserWarning):
            pipeline(data=nifti_plus, water=None)

        # Shape should be preserved
        assert CheckDataModule.received_shape[0] == len(dummy_nifti_list)


#**************************************************************************************************#
#                                     Class TestPipelineKwargs                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test passing kwargs through pipeline.                                                            #
#                                                                                                  #
#**************************************************************************************************#
class TestPipelineKwargs:
    """Test passing kwargs through pipeline."""

    def test_kwargs_passed_to_modules(self, dummy_nifti_list):
        """Test that kwargs are passed to all modules."""
        class KwargsModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            received_kwargs = {}

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                KwargsModule.received_kwargs = kwargs
                return data_list, water_list

        pipeline = AugmentationPipeline([KwargsModule()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        pipeline(data=nifti_plus, water=None, custom_param='value', number=42)

        assert KwargsModule.received_kwargs['custom_param'] == 'value'
        assert KwargsModule.received_kwargs['number'] == 42

    def test_kwargs_passed_through_all_modules(self, dummy_nifti_list):
        """Test that kwargs propagate through all modules."""
        received_params = []

        class Module1(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                received_params.append(kwargs.get('test_param'))
                return data_list, water_list

        class Module2(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                received_params.append(kwargs.get('test_param'))
                return data_list, water_list

        pipeline = AugmentationPipeline([Module1(), Module2()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        pipeline(data=nifti_plus, water=None, test_param='propagated')

        assert received_params == ['propagated', 'propagated']


#**************************************************************************************************#
#                                 Class TestPipelineErrorHandling                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test pipeline error handling.                                                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    def test_error_on_module_without_basemodule(self, dummy_nifti_list):
        """Test warning when pipeline contains non-BaseModule objects."""
        class NotAModule:
            def __call__(self, data, water=None, **kwargs):
                return data, water

        # Pipeline should warn during construction
        with pytest.warns(UserWarning, match="does not inherit from BaseModule"):
            pipeline = AugmentationPipeline([NotAModule()])

        # Should still execute
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)
        assert isinstance(result_data, NIfTI_MRS_Plus)

    def test_empty_pipeline_returns_data_unchanged(self, dummy_nifti_list):
        """Test that empty pipeline returns data unchanged."""
        pipeline = AugmentationPipeline([])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, result_water = pipeline(data=nifti_plus, water=None)

        assert result_data is nifti_plus
        assert result_water is None


#**************************************************************************************************#
#                                      Class TestPipelineRepr                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test pipeline string representation.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class TestPipelineRepr:
    """Test pipeline string representation."""

    def test_repr_empty_pipeline(self):
        """Test __repr__ for empty pipeline."""
        pipeline = AugmentationPipeline([])
        repr_str = repr(pipeline)

        assert 'AugmentationPipeline' in repr_str
        assert 'steps=[]' in repr_str

    def test_repr_with_modules(self):
        """Test __repr__ with modules."""
        class Module1(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        class Module2(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NUMPY]
            def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
                return data_array, water_array

        pipeline = AugmentationPipeline([Module1(), Module2()])
        repr_str = repr(pipeline)

        assert 'AugmentationPipeline' in repr_str
        assert 'Module1' in repr_str
        assert 'Module2' in repr_str


#**************************************************************************************************#
#                                  Class TestPipelineIntegration                                   #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests with real-world scenarios.                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestPipelineIntegration:
    """Integration tests with real-world scenarios."""

    def test_realistic_pipeline(self, dummy_nifti_list):
        """Test realistic pipeline with multiple processing steps."""
        class SamplingModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                # Simulate sampling - take first 3 subjects
                return data_list[:3], water_list

        class ProcessingModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                # Simulate processing
                return data_list, water_list

        class AugmentationModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                # Simulate augmentation
                return data_list, water_list

        pipeline = AugmentationPipeline([
            SamplingModule(),
            ProcessingModule(),
            AugmentationModule()
        ])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == 3

    def test_pipeline_with_volatile_mode(self, dummy_nifti_list):
        """Test pipeline respects volatile mode (no logging)."""
        class LogCheckModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        pipeline = AugmentationPipeline([LogCheckModule()])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=True)

        result_data, _ = pipeline(data=nifti_plus, water=None)

        # Metadata should still be empty in volatile mode
        assert result_data.metadata_common == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
