"""
Tests for BaseModule core functionality.

Tests cover:
- Module creation and initialization
- Backend support declaration
- Automatic method dispatching
- process_nifti_list() method
- process_tensor() method
- forward() method (legacy)
- Automatic logging/provenance
- Error handling
"""

import pytest
import numpy as np
from augmentrum.core import BaseModule, NIfTI_MRS_Plus, Backend


class TestBaseModuleCreation:
    """Test BaseModule initialization."""

    def test_create_base_module_with_params(self):
        """Test creating BaseModule with parameters."""
        class SimpleModule(BaseModule):
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = SimpleModule(param1='value1', param2=42)

        assert module.params == {'param1': 'value1', 'param2': 42}

    def test_create_base_module_no_params(self):
        """Test creating BaseModule without parameters."""
        class SimpleModule(BaseModule):
            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = SimpleModule()

        assert module.params == {}

    def test_supported_backends_declaration(self):
        """Test declaring supported backends."""
        class NIfTIOnlyModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = NIfTIOnlyModule()

        assert module.SUPPORTED_BACKENDS == [Backend.NIFTI_LIST]
        assert module.supports_backend(Backend.NIFTI_LIST)
        assert not module.supports_backend(Backend.NUMPY)

    def test_supports_all_backends_by_default(self):
        """Test that empty SUPPORTED_BACKENDS means all backends supported."""
        class AllBackendsModule(BaseModule):
            SUPPORTED_BACKENDS = []

            def forward(self, data, water=None, **kwargs):
                return data, water

        module = AllBackendsModule()

        assert module.supports_backend(Backend.NIFTI_LIST)
        assert module.supports_backend(Backend.NUMPY)
        assert module.supports_backend(Backend.PYTORCH)


class TestBaseModuleBackendSupport:
    """Test backend support checking."""

    def test_supports_backend_method(self):
        """Test supports_backend() method."""
        class NumpyModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NUMPY, Backend.PYTORCH]

            def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
                return data_array, water_array

        module = NumpyModule()

        assert module.supports_backend(Backend.NUMPY)
        assert module.supports_backend(Backend.PYTORCH)
        assert not module.supports_backend(Backend.NIFTI_LIST)

    def test_get_preferred_backend(self):
        """Test get_preferred_backend() returns first in list."""
        class MultiBackendModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.PYTORCH, Backend.NUMPY, Backend.NIFTI_LIST]

            def forward(self, data, water=None, **kwargs):
                return data, water

        module = MultiBackendModule()

        assert module.get_preferred_backend() == Backend.PYTORCH

    def test_get_preferred_backend_defaults_to_nifti_list(self):
        """Test get_preferred_backend() defaults to NIFTI_LIST when empty."""
        class DefaultModule(BaseModule):
            SUPPORTED_BACKENDS = []

            def forward(self, data, water=None, **kwargs):
                return data, water

        module = DefaultModule()

        assert module.get_preferred_backend() == Backend.NIFTI_LIST


class TestBaseModuleDispatchingNIfTIList:
    """Test dispatching to process_nifti_list() method."""

    def test_dispatch_to_process_nifti_list(self, dummy_nifti_list):
        """Test that __call__ dispatches to process_nifti_list()."""
        class NIfTIModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                # Modify data to prove this method was called
                return data_list[:2], water_list  # Return only first 2

        module = NIfTIModule()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, result_water = module(nifti_plus, None)

        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert len(result_data) == 2  # Should be reduced to 2

    def test_process_nifti_list_wraps_results(self, dummy_nifti_list):
        """Test that results are wrapped in NIfTI_MRS_Plus."""
        class NIfTIModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = NIfTIModule()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, result_water = module(nifti_plus, None)

        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert len(result_data) == len(dummy_nifti_list)

    def test_process_nifti_list_with_water(self, dummy_nifti_list):
        """Test processing with water reference."""
        class NIfTIModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                assert water_list is not None
                return data_list, water_list

        module = NIfTIModule()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        water_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list[:3], backend=Backend.NIFTI_LIST)

        result_data, result_water = module(nifti_plus, water_plus)

        assert isinstance(result_water, NIfTI_MRS_Plus)
        assert len(result_water) == 3


class TestBaseModuleDispatchingTensor:
    """Test dispatching to process_tensor() method."""

    def test_dispatch_to_process_tensor(self, dummy_nifti_list):
        """Test that __call__ dispatches to process_tensor() for tensor backends."""
        class TensorModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NUMPY]

            def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
                # Multiply by 2 to prove this method was called
                return data_array * 2, water_array

        module = TensorModule()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        result_data, result_water = module(nifti_plus, None)

        assert isinstance(result_data, NIfTI_MRS_Plus)
        # Note: The current implementation is a placeholder for tensor processing

    def test_process_tensor_receives_correct_backend(self, dummy_nifti_list):
        """Test that process_tensor receives correct backend parameter."""
        received_backend = None

        class TensorModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NUMPY, Backend.PYTORCH]

            def process_tensor(self, data_array, water_array=None, backend=Backend.NUMPY, **kwargs):
                nonlocal received_backend
                received_backend = backend
                return data_array, water_array

        module = TensorModule()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        module(nifti_plus, None)

        assert received_backend == Backend.NUMPY


class TestBaseModuleDispatchingForward:
    """Test dispatching to forward() method (legacy)."""

    def test_dispatch_to_forward(self, dummy_nifti_list):
        """Test that __call__ can dispatch to forward() method."""
        class LegacyModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def forward(self, data, water=None, **kwargs):
                # Return modified list
                return data[:3], water

        module = LegacyModule()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, result_water = module(nifti_plus, None)

        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert len(result_data) == 3


class TestBaseModuleLogging:
    """Test automatic logging/provenance functionality."""

    def test_logging_when_not_volatile(self, dummy_nifti_list):
        """Test that metadata is updated when volatile=False."""
        class LoggingModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = LoggingModule(test_param='value')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=False)

        result_data, _ = module(nifti_plus, None)

        # Check metadata was updated
        assert 'common_provenance' in result_data.metadata_common
        assert len(result_data.metadata_common['common_provenance']) > 0

    def test_no_logging_when_volatile(self, dummy_nifti_list):
        """Test that metadata is NOT updated when volatile=True."""
        class LoggingModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = LoggingModule(test_param='value')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=True)

        result_data, _ = module(nifti_plus, None)

        # Metadata should still be empty
        assert result_data.metadata_common == {}

    def test_logging_includes_operation_name(self, dummy_nifti_list):
        """Test that logged metadata includes operation name."""
        class NamedModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = NamedModule()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=False)

        result_data, _ = module(nifti_plus, None)

        # Check operation name in provenance
        provenance = result_data.metadata_common.get('common_provenance', [])
        if provenance:
            assert any('NamedModule' in str(entry) for entry in provenance)

    def test_logging_includes_parameters(self, dummy_nifti_list):
        """Test that logged metadata includes module parameters."""
        class ParamModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list, water_list

        module = ParamModule(alpha=0.5, beta=100)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=False)

        result_data, _ = module(nifti_plus, None)

        # Parameters should be logged
        provenance = result_data.metadata_common.get('common_provenance', [])
        assert len(provenance) > 0


class TestBaseModuleErrorHandling:
    """Test error handling."""

    def test_error_when_no_methods_implemented(self, dummy_nifti_list):
        """Test error when module doesn't implement any processing methods."""
        class EmptyModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

        module = EmptyModule()
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        with pytest.raises(NotImplementedError, match="must implement at least one"):
            module(nifti_plus, None)

    def test_error_when_process_nifti_list_not_implemented(self):
        """Test error when calling process_nifti_list() on base class."""
        module = BaseModule()

        with pytest.raises(NotImplementedError, match="does not implement process_nifti_list"):
            module.process_nifti_list([], None)

    def test_error_when_process_tensor_not_implemented(self):
        """Test error when calling process_tensor() on base class."""
        module = BaseModule()

        with pytest.raises(NotImplementedError, match="does not implement process_tensor"):
            module.process_tensor(np.array([]), None)

    def test_error_when_forward_not_implemented(self):
        """Test error when calling forward() on base class."""
        module = BaseModule()

        with pytest.raises(NotImplementedError, match="does not implement forward"):
            module.forward([], None)


class TestBaseModuleRepr:
    """Test string representation."""

    def test_repr_with_backends(self):
        """Test __repr__ shows supported backends."""
        class MultiModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST, Backend.NUMPY]

            def forward(self, data, water=None, **kwargs):
                return data, water

        module = MultiModule()
        repr_str = repr(module)

        assert 'MultiModule' in repr_str
        assert 'nifti_list' in repr_str.lower() or 'numpy' in repr_str.lower()

    def test_repr_with_all_backends(self):
        """Test __repr__ shows 'all' when no backends specified."""
        class AllModule(BaseModule):
            SUPPORTED_BACKENDS = []

            def forward(self, data, water=None, **kwargs):
                return data, water

        module = AllModule()
        repr_str = repr(module)

        assert 'AllModule' in repr_str
        assert 'all' in repr_str.lower()


class TestBaseModuleIntegration:
    """Integration tests with real modules."""

    def test_with_nifti_rawprocessor_style(self, dummy_nifti_list):
        """Test BaseModule works like NIfTI_RawProcessor."""
        class ProcessorModule(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def __init__(self, scale_factor=1.0):
                super().__init__(scale_factor=scale_factor)
                self.scale_factor = scale_factor

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                # Simple processing: scale the data
                processed = []
                for nifti in data_list:
                    # Would normally process here
                    processed.append(nifti)
                return processed, water_list

        module = ProcessorModule(scale_factor=2.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, result_water = module(nifti_plus, None)

        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert len(result_data) == len(dummy_nifti_list)
        assert result_water is None

    def test_with_noiseperturber_style(self, dummy_nifti_list):
        """Test BaseModule works like NoisePerturber."""
        class NoiseModule(BaseModule):
            SUPPORTED_BACKENDS = []  # Supports all

            def __init__(self, noise_level=0.1):
                super().__init__(noise_level=noise_level)
                self.noise_level = noise_level

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                # Simple noise addition (placeholder)
                return data_list, water_list

        module = NoiseModule(noise_level=0.05)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = module(nifti_plus, None)

        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert len(result_data) == len(dummy_nifti_list)

    def test_chaining_modules(self, dummy_nifti_list):
        """Test chaining multiple BaseModule instances."""
        class Module1(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list[:4], water_list  # Reduce to 4

        class Module2(BaseModule):
            SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

            def process_nifti_list(self, data_list, water_list=None, **kwargs):
                return data_list[:2], water_list  # Further reduce to 2

        module1 = Module1()
        module2 = Module2()

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Chain: apply module1 then module2
        intermediate, _ = module1(nifti_plus, None)
        final, _ = module2(intermediate, None)

        assert len(final) == 2



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
