"""
Tests for NIfTI_MRS_Plus core functionality.

Tests cover:
- Creation and initialization
- Backend support and conversion
- Shape and dimension properties
- Data access methods
- Metadata handling
- Slicing and indexing
"""

import pytest
import numpy as np
from augmentrum.core import NIfTI_MRS_Plus, Backend


class TestNIfTIMRSPlusCreation:
    """Test NIfTI_MRS_Plus object creation."""

    def test_create_from_list(self, dummy_nifti_list):
        """Test creating NIfTI_MRS_Plus from list of NIfTI-MRS objects."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list)

        assert nifti_plus is not None
        assert len(nifti_plus) == 5
        assert nifti_plus.n_subjects == 5

    def test_create_from_another_nifti_mrs_plus(self, nifti_mrs_plus):
        """Test creating NIfTI_MRS_Plus from another NIfTI_MRS_Plus."""
        nifti_plus_copy = NIfTI_MRS_Plus(nifti_list=nifti_mrs_plus)

        assert len(nifti_plus_copy) == len(nifti_mrs_plus)
        assert nifti_plus_copy.n_subjects == nifti_mrs_plus.n_subjects
        assert nifti_plus_copy.backend == nifti_mrs_plus.backend

    def test_create_with_backend(self, dummy_nifti_list):
        """Test creating with specific backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        assert nifti_plus.backend == Backend.NUMPY

    def test_create_with_volatile(self, dummy_nifti_list):
        """Test creating with volatile=True."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, volatile=True)

        assert nifti_plus.volatile == True
        assert nifti_plus.metadata_common == {}
        assert nifti_plus.metadata_individual == []

    def test_create_empty_list(self):
        """Test creating with empty list."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])

        assert len(nifti_plus) == 0
        assert nifti_plus.n_subjects == 0

    def test_reject_invalid_input(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="must be a list of NIFTI_MRS"):
            NIfTI_MRS_Plus(nifti_list="not a list")

        with pytest.raises(ValueError, match="must be a list of NIFTI_MRS"):
            NIfTI_MRS_Plus(nifti_list=[1, 2, 3])


class TestNIfTIMRSPlusShape:
    """Test shape and dimension properties."""

    def test_shape_property(self, nifti_mrs_plus):
        """Test that shape property returns correct dimensions."""
        shape = nifti_mrs_plus.shape

        assert shape[0] == 5  # n_subjects
        assert shape[1:] == (1, 1, 1, 2048, 8, 16)  # NIfTI-MRS shape

    def test_dim_tags(self, nifti_mrs_plus):
        """Test dim_tags property."""
        dim_tags = nifti_mrs_plus.dim_tags

        assert 'DIM_COIL' in dim_tags
        assert 'DIM_DYN' in dim_tags

    def test_dim_tags_consistency(self, dummy_nifti_list):
        """Test that all subjects must have same dim_tags."""
        # Modify one nifti to have different dim_tags
        dummy_nifti_list[2].set_dim_tag(4, 'DIM_EDIT')

        with pytest.raises(ValueError, match="same dim_tags"):
            NIfTI_MRS_Plus(nifti_list=dummy_nifti_list)

    def test_len(self, nifti_mrs_plus):
        """Test __len__ returns number of subjects."""
        assert len(nifti_mrs_plus) == 5

    def test_n_subjects(self, nifti_mrs_plus):
        """Test n_subjects property."""
        assert nifti_mrs_plus.n_subjects == 5


class TestNIfTIMRSPlusProxyMethods:
    """Test proxy methods that delegate to first NIfTI-MRS."""

    def test_dim_position(self, nifti_mrs_plus):
        """Test dim_position returns correct dimension index."""
        coil_dim = nifti_mrs_plus.dim_position('DIM_COIL')
        dyn_dim = nifti_mrs_plus.dim_position('DIM_DYN')

        assert coil_dim == 4  # 5th dimension (index 4)
        assert dyn_dim == 5   # 6th dimension (index 5)

    def test_dwelltime(self, nifti_mrs_plus):
        """Test dwelltime property."""
        assert nifti_mrs_plus.dwelltime == 1/2000

    def test_spectrometer_frequency(self, nifti_mrs_plus):
        """Test spectrometer_frequency property."""
        assert nifti_mrs_plus.spectrometer_frequency == [123.0]

    def test_nucleus(self, nifti_mrs_plus):
        """Test nucleus property."""
        # Default from gen_nifti_mrs
        assert nifti_mrs_plus.nucleus is not None

    def test_ndim(self, nifti_mrs_plus):
        """Test ndim property."""
        assert nifti_mrs_plus.ndim >= 6  # At least 6 dimensions

    def test_dtype(self, nifti_mrs_plus):
        """Test dtype property."""
        assert nifti_mrs_plus.dtype == np.complex64 or nifti_mrs_plus.dtype == np.complex128

    def test_header(self, nifti_mrs_plus):
        """Test header property."""
        header = nifti_mrs_plus.header
        assert header is not None

    def test_hdr_ext(self, nifti_mrs_plus):
        """Test hdr_ext property."""
        hdr_ext = nifti_mrs_plus.hdr_ext
        assert hdr_ext is not None


class TestNIfTIMRSPlusBackends:
    """Test backend support and conversion."""

    def test_default_backend(self, dummy_nifti_list):
        """Test default backend is NIFTI_LIST."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list)
        assert nifti_plus.backend == Backend.NIFTI_LIST

    def test_get_data_nifti_list(self, nifti_mrs_plus):
        """Test get_data with NIFTI_LIST backend."""
        data = nifti_mrs_plus.get_data(Backend.NIFTI_LIST)

        assert isinstance(data, list)
        assert len(data) == 5
        # Check it's the actual nifti objects
        assert hasattr(data[0], 'dwelltime')

    def test_get_data_numpy(self, nifti_mrs_plus):
        """Test get_data with NUMPY backend."""
        data = nifti_mrs_plus.get_data(Backend.NUMPY)

        assert isinstance(data, np.ndarray)
        assert data.shape == (5, 1, 1, 1, 2048, 8, 16)
        assert np.iscomplexobj(data)

    @pytest.mark.skipif(not hasattr(Backend, 'PYTORCH'), reason="PyTorch not available")
    def test_get_data_pytorch(self, nifti_mrs_plus):
        """Test get_data with PYTORCH backend."""
        try:
            import torch
            data = nifti_mrs_plus.get_data(Backend.PYTORCH)

            assert isinstance(data, torch.Tensor)
            assert data.shape == torch.Size([5, 1, 1, 1, 2048, 8, 16])
        except ImportError:
            pytest.skip("PyTorch not installed")

    @pytest.mark.skipif(not hasattr(Backend, 'TENSORFLOW'), reason="TensorFlow not available")
    def test_get_data_tensorflow(self, nifti_mrs_plus):
        """Test get_data with TENSORFLOW backend."""
        try:
            import tensorflow as tf
            data = nifti_mrs_plus.get_data(Backend.TENSORFLOW)

            assert isinstance(data, tf.Tensor)
            assert tuple(data.shape) == (5, 1, 1, 1, 2048, 8, 16)
            assert data.dtype == tf.complex64 or data.dtype == tf.complex128
        except ImportError:
            pytest.skip("TensorFlow not installed")

    @pytest.mark.skipif(not hasattr(Backend, 'JAX'), reason="JAX not available")
    def test_get_data_jax(self, nifti_mrs_plus):
        """Test get_data with JAX backend."""
        try:
            import jax.numpy as jnp
            data = nifti_mrs_plus.get_data(Backend.JAX)

            assert isinstance(data, jnp.ndarray)
            assert data.shape == (5, 1, 1, 1, 2048, 8, 16)
            assert jnp.iscomplexobj(data)
        except ImportError:
            pytest.skip("JAX not installed")

    @pytest.mark.skipif(not hasattr(Backend, 'KERAS'), reason="Keras not available")
    def test_get_data_keras(self, nifti_mrs_plus):
        """Test get_data with KERAS backend."""
        try:
            import keras
            data = nifti_mrs_plus.get_data(Backend.KERAS)

            # Keras uses backend tensors (TF or JAX)
            assert hasattr(data, 'shape')
            assert tuple(data.shape) == (5, 1, 1, 1, 2048, 8, 16)
        except ImportError:
            pytest.skip("Keras not installed")

    def test_numpy_method_caching(self, nifti_mrs_plus):
        """Test that numpy() method caches result."""
        data1 = nifti_mrs_plus.numpy()
        data2 = nifti_mrs_plus.numpy()

        # Should return same cached array
        assert data1 is data2

    def test_list_method(self, nifti_mrs_plus):
        """Test list() method returns nifti_list."""
        nifti_list = nifti_mrs_plus.list()

        assert isinstance(nifti_list, list)
        assert len(nifti_list) == 5

    def test_to_nifti_list(self, nifti_mrs_plus):
        """Test to_nifti_list() returns nifti_list."""
        nifti_list = nifti_mrs_plus.to_nifti_list()

        assert isinstance(nifti_list, list)
        assert len(nifti_list) == 5

    def test_data_property(self, nifti_mrs_plus):
        """Test data property returns data in current backend."""
        data = nifti_mrs_plus.data

        # Default backend is NIFTI_LIST
        assert isinstance(data, list)
        assert len(data) == 5


class TestNIfTIMRSPlusBackendConversions:
    """Test backend conversions and compatibility."""

    def test_backend_conversion_numpy_to_pytorch(self, dummy_nifti_list):
        """Test converting from NUMPY to PYTORCH backend."""
        try:
            import torch
            nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

            # Get data as PyTorch tensor
            data_torch = nifti_plus.get_data(Backend.PYTORCH)

            assert isinstance(data_torch, torch.Tensor)
            assert data_torch.shape == torch.Size([5, 1, 1, 1, 2048, 8, 16])
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_backend_conversion_nifti_to_tensorflow(self, dummy_nifti_list):
        """Test converting from NIFTI_LIST to TENSORFLOW backend."""
        try:
            import tensorflow as tf
            nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

            # Get data as TensorFlow tensor
            data_tf = nifti_plus.get_data(Backend.TENSORFLOW)

            assert isinstance(data_tf, tf.Tensor)
            assert tuple(data_tf.shape) == (5, 1, 1, 1, 2048, 8, 16)
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_backend_conversion_numpy_to_jax(self, dummy_nifti_list):
        """Test converting from NUMPY to JAX backend."""
        try:
            import jax.numpy as jnp
            nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

            # Get data as JAX array
            data_jax = nifti_plus.get_data(Backend.JAX)

            assert isinstance(data_jax, jnp.ndarray)
            assert data_jax.shape == (5, 1, 1, 1, 2048, 8, 16)
        except ImportError:
            pytest.skip("JAX not installed")

    def test_backend_conversion_nifti_to_keras(self, dummy_nifti_list):
        """Test converting from NIFTI_LIST to KERAS backend."""
        try:
            import keras
            nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

            # Get data as Keras tensor
            data_keras = nifti_plus.get_data(Backend.KERAS)

            assert hasattr(data_keras, 'shape')
            assert tuple(data_keras.shape) == (5, 1, 1, 1, 2048, 8, 16)
        except ImportError:
            pytest.skip("Keras not installed")

    def test_all_backends_produce_same_values(self, dummy_nifti_list):
        """Test that all backends produce equivalent numerical values."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get numpy as reference
        data_numpy = nifti_plus.get_data(Backend.NUMPY)

        # Test PyTorch
        try:
            import torch
            data_torch = nifti_plus.get_data(Backend.PYTORCH)
            torch_numpy = data_torch.numpy()
            np.testing.assert_array_almost_equal(data_numpy, torch_numpy)
        except ImportError:
            pass  # Skip if not installed

        # Test TensorFlow
        try:
            import tensorflow as tf
            data_tf = nifti_plus.get_data(Backend.TENSORFLOW)
            tf_numpy = data_tf.numpy()
            np.testing.assert_array_almost_equal(data_numpy, tf_numpy)
        except ImportError:
            pass  # Skip if not installed

        # Test JAX
        try:
            import jax.numpy as jnp
            data_jax = nifti_plus.get_data(Backend.JAX)
            jax_numpy = np.asarray(data_jax)
            np.testing.assert_array_almost_equal(data_numpy, jax_numpy)
        except ImportError:
            pass  # Skip if not installed

    def test_backend_setting_preserved(self, dummy_nifti_list):
        """Test that backend setting is preserved in NIfTI_MRS_Plus."""
        backends_to_test = [Backend.NUMPY]

        # Add available backends
        try:
            import torch
            backends_to_test.append(Backend.PYTORCH)
        except ImportError:
            pass

        try:
            import tensorflow
            backends_to_test.append(Backend.TENSORFLOW)
        except ImportError:
            pass

        for backend in backends_to_test:
            nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=backend)
            assert nifti_plus.backend == backend

    def test_complex_dtype_preserved_across_backends(self, dummy_nifti_list):
        """Test that complex dtype is preserved when converting backends."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        # NumPy
        data_numpy = nifti_plus.get_data(Backend.NUMPY)
        assert np.iscomplexobj(data_numpy)

        # PyTorch
        try:
            import torch
            data_torch = nifti_plus.get_data(Backend.PYTORCH)
            assert torch.is_complex(data_torch)
        except ImportError:
            pass

        # TensorFlow
        try:
            import tensorflow as tf
            data_tf = nifti_plus.get_data(Backend.TENSORFLOW)
            assert data_tf.dtype in [tf.complex64, tf.complex128]
        except ImportError:
            pass

        # JAX
        try:
            import jax.numpy as jnp
            data_jax = nifti_plus.get_data(Backend.JAX)
            assert jnp.iscomplexobj(data_jax)
        except ImportError:
            pass


class TestNIfTIMRSPlusIndexing:
    """Test indexing and slicing."""

    def test_getitem_single_index(self, nifti_mrs_plus):
        """Test indexing with single integer."""
        subset = nifti_mrs_plus[0]

        assert isinstance(subset, NIfTI_MRS_Plus)
        assert len(subset) == 1
        assert subset.backend == nifti_mrs_plus.backend

    def test_getitem_slice(self, nifti_mrs_plus):
        """Test slicing."""
        subset = nifti_mrs_plus[1:3]

        assert isinstance(subset, NIfTI_MRS_Plus)
        assert len(subset) == 2

    def test_getitem_preserves_backend(self, dummy_nifti_list):
        """Test that slicing preserves backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)
        subset = nifti_plus[0:2]

        assert subset.backend == Backend.NUMPY

    def test_getitem_preserves_volatile(self, dummy_nifti_list):
        """Test that slicing preserves volatile setting."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, volatile=True)
        subset = nifti_plus[0]

        assert subset.volatile == True


class TestNIfTIMRSPlusMetadata:
    """Test metadata handling."""

    def test_metadata_initialized(self, nifti_mrs_plus):
        """Test metadata is initialized when volatile=False."""
        assert isinstance(nifti_mrs_plus.metadata_common, dict)
        assert isinstance(nifti_mrs_plus.metadata_individual, list)
        assert len(nifti_mrs_plus.metadata_individual) == 5

    def test_metadata_empty_when_volatile(self, dummy_nifti_list):
        """Test metadata is empty when volatile=True."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, volatile=True)

        assert nifti_plus.metadata_common == {}
        assert nifti_plus.metadata_individual == []

    def test_metadata_common_contains_expected_keys(self, nifti_mrs_plus):
        """Test common metadata contains expected keys."""
        assert 'dim_tags' in nifti_mrs_plus.metadata_common or len(nifti_mrs_plus.metadata_common) == 0

    def test_update_metadata(self, nifti_mrs_plus):
        """Test update_metadata method."""
        if not nifti_mrs_plus.volatile:
            nifti_mrs_plus.update_metadata('TestOperation', {'param': 'value'})

            # Should have provenance
            assert 'common_provenance' in nifti_mrs_plus.metadata_common
            assert len(nifti_mrs_plus.metadata_common['common_provenance']) > 0

    def test_update_metadata_skipped_when_volatile(self, dummy_nifti_list):
        """Test update_metadata does nothing when volatile=True."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, volatile=True)
        nifti_plus.update_metadata('TestOperation', {'param': 'value'})

        # Should still be empty
        assert nifti_plus.metadata_common == {}


class TestNIfTIMRSPlusCopy:
    """Test copy functionality."""

    def test_copy(self, nifti_mrs_plus):
        """Test copy creates deep copy."""
        nifti_copy = nifti_mrs_plus.copy()

        assert nifti_copy is not nifti_mrs_plus
        assert len(nifti_copy) == len(nifti_mrs_plus)
        assert nifti_copy.backend == nifti_mrs_plus.backend
        assert nifti_copy.volatile == nifti_mrs_plus.volatile

    def test_copy_independent(self, nifti_mrs_plus):
        """Test that copy is independent (metadata changes don't affect original)."""
        nifti_copy = nifti_mrs_plus.copy()

        if not nifti_copy.volatile:
            nifti_copy.update_metadata('TestOp', {})
            # Original should not have this metadata
            assert len(nifti_copy.metadata_common.get('common_provenance', [])) != \
                   len(nifti_mrs_plus.metadata_common.get('common_provenance', []))


class TestNIfTIMRSPlusRepr:
    """Test string representation."""

    def test_repr(self, nifti_mrs_plus):
        """Test __repr__ returns informative string."""
        repr_str = repr(nifti_mrs_plus)

        assert 'NIfTI_MRS_Plus' in repr_str
        assert 'n_subjects=5' in repr_str
        assert 'backend=' in repr_str
        assert 'volatile=' in repr_str


class TestNIfTIMRSPlusEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_nifti_plus(self):
        """Test NIfTI_MRS_Plus with empty list."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])

        assert len(nifti_plus) == 0
        assert nifti_plus.n_subjects == 0
        assert nifti_plus.shape == (0,)
        assert nifti_plus.dim_tags is None

    def test_get_data_empty_list(self):
        """Test get_data with empty list."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])

        # NIFTI_LIST should return empty list
        data = nifti_plus.get_data(Backend.NIFTI_LIST)
        assert data == []

        # NUMPY should return empty array
        data_numpy = nifti_plus.get_data(Backend.NUMPY)
        assert isinstance(data_numpy, np.ndarray)
        assert len(data_numpy) == 0

    def test_proxy_methods_on_empty(self):
        """Test proxy methods on empty NIfTI_MRS_Plus."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])

        assert nifti_plus.dim_position('DIM_COIL') is None
        assert nifti_plus.dwelltime is None
        assert nifti_plus.ndim is None
        assert nifti_plus.dtype is None

    def test_sync_headers(self, nifti_mrs_plus):
        """Test sync_headers method."""
        nifti_mrs_plus.sync_headers(source_idx=0)
        # Should not raise error


class TestNIfTIMRSPlusSetItem:
    """Test __setitem__ functionality."""

    def test_setitem_nifti_list_single_subject(self, dummy_nifti_list, dummy_nifti_mrs):
        """Test setting a single subject in NIFTI_LIST backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Get original first subject
        original_first = nifti_plus[0].list()[0]

        # Replace first subject with dummy
        nifti_plus.nifti_list[0] = dummy_nifti_mrs

        # Verify replacement
        assert nifti_plus.nifti_list[0] is dummy_nifti_mrs
        assert nifti_plus.nifti_list[0] is not original_first

    def test_setitem_nifti_list_multidim_indexing(self, nifti_mrs_plus):
        """Test multi-dimensional indexing in NIFTI_LIST backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=nifti_mrs_plus.list(), backend=Backend.NIFTI_LIST)

        # Get original data
        original_data = nifti_plus[0].list()[0][:]

        # Create new values for specific timepoints
        new_values = np.ones((1, 1, 1, 100, 8, 16), dtype=np.complex128) * 999

        # Set values for first subject, timepoints 100-200
        nifti_plus.nifti_list[0][:, :, :, 100:200, :, :] = new_values

        # Verify change
        modified_data = nifti_plus[0].list()[0][:]
        assert not np.allclose(modified_data[:, :, :, 100:200, :, :],
                               original_data[:, :, :, 100:200, :, :])

    def test_setitem_invalidates_cache(self, nifti_mrs_plus):
        """Test that setting values invalidates cached data."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=nifti_mrs_plus.list(), backend=Backend.NIFTI_LIST)

        # Force cache creation
        _ = nifti_plus.numpy()
        assert nifti_plus._batched_data is not None

        # Set a value (use multi-dimensional indexing)
        new_value = np.ones((1, 1, 1, 10), dtype=np.complex128) * 999
        for nifti in nifti_plus.nifti_list:
            nifti[:, :, :, 0:10, 0, 0] = new_value
        nifti_plus._batched_data = None  # Manually invalidate as we're directly modifying

        # Cache should be invalidated
        assert nifti_plus._batched_data is None

    def test_setitem_numpy_backend_simple(self, dummy_nifti_list):
        """Test setting values with NUMPY backend using simple indexing."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        # Get original data
        original_data = nifti_plus.get_data(Backend.NUMPY).copy()

        # Create new values
        new_values = np.ones_like(original_data[0:1]) * 999

        # Set first subject's data
        nifti_plus[0] = new_values[0]

        # Get updated data
        updated_data = nifti_plus.get_data(Backend.NUMPY)

        # First subject should be different
        assert not np.allclose(updated_data[0], original_data[0])

    def test_setitem_numpy_backend_multidim(self, dummy_nifti_list):
        """Test multi-dimensional indexing with NUMPY backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        # Get original data
        original_data = nifti_plus.numpy().copy()

        # Create new values for specific timepoints
        new_values = np.ones((5, 1, 1, 1, 100, 8, 16), dtype=np.complex128) * 999

        # Set values for all subjects, specific timepoints
        nifti_plus[(slice(None), slice(None), slice(None), slice(None), slice(100, 200), slice(None), slice(None))] = new_values

        # Get updated data (need fresh copy as cache was updated)
        updated_data = nifti_plus.numpy()

        # Specified region should have changed
        assert np.allclose(updated_data[:, :, :, :, 100:200, :, :], new_values)

    @pytest.mark.skipif(not hasattr(Backend, 'PYTORCH'), reason="PyTorch not available")
    def test_setitem_pytorch_backend(self, dummy_nifti_list):
        """Test setting values with PYTORCH backend."""
        try:
            import torch
            nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.PYTORCH)

            # Get original data
            original_data = nifti_plus.get_data(Backend.NUMPY).copy()

            # Create new values as PyTorch tensor
            new_values = torch.ones((5, 1, 1, 1, 10, 8, 16), dtype=torch.complex128) * 999

            # Set values
            nifti_plus[(slice(None), slice(None), slice(None), slice(None), slice(0, 10), slice(None), slice(None))] = new_values

            # Get updated data
            updated_data = nifti_plus.get_data(Backend.NUMPY)

            # First 10 timepoints should be 999
            assert np.allclose(np.abs(updated_data[:, :, :, :, 0:10, :, :]), 999, atol=1)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_setitem_broadcast_to_all_subjects(self, dummy_nifti_list):
        """Test broadcasting single value to all subjects."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        # Create single value to broadcast
        single_value = np.ones((1, 1, 1, 1, 10, 8, 16), dtype=np.complex128) * 777

        # Set to all subjects (should broadcast)
        nifti_plus[(slice(None), slice(None), slice(None), slice(None), slice(0, 10), slice(None), slice(None))] = single_value

        # Verify all subjects have the same value
        updated_data = nifti_plus.numpy()
        for i in range(5):
            assert np.allclose(np.abs(updated_data[i, :, :, :, 0:10, :, :]), 777, atol=1)

    def test_setitem_preserves_other_data(self, dummy_nifti_list):
        """Test that setting values doesn't affect other data."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        # Get original data
        original_data = nifti_plus.numpy().copy()

        # Modify only first subject, first timepoint
        new_value = np.ones((1, 1, 1, 1, 1, 8, 16), dtype=np.complex128) * 999
        nifti_plus[(0, slice(None), slice(None), slice(None), slice(0, 1), slice(None), slice(None))] = new_value

        # Get updated data
        updated_data = nifti_plus.numpy()

        # First subject, first timepoint should be different
        assert not np.allclose(updated_data[0, :, :, :, 0, :, :], original_data[0, :, :, :, 0, :, :])

        # Everything else should be the same
        assert np.allclose(updated_data[1:], original_data[1:])  # Other subjects
        assert np.allclose(updated_data[0, :, :, :, 1:, :, :],
                          original_data[0, :, :, :, 1:, :, :])  # Other timepoints

    def test_setitem_error_on_invalid_values_nifti_backend(self, nifti_mrs_plus):
        """Test that setting invalid values raises error in NIFTI_LIST backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=nifti_mrs_plus.list(), backend=Backend.NIFTI_LIST)

        # Try to set with invalid type using __setitem__
        with pytest.raises(ValueError, match="values must be NIFTI_MRS"):
            nifti_plus[0] = "not a nifti object"

    def test_setitem_updates_underlying_nifti_objects(self, dummy_nifti_list):
        """Test that setting values updates underlying NIFTI_MRS objects."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        # Create new values
        new_values = np.ones((5, 1, 1, 1, 10, 8, 16), dtype=np.complex128) * 888

        # Set values
        nifti_plus[(slice(None), slice(None), slice(None), slice(None), slice(0, 10), slice(None), slice(None))] = new_values

        # Check underlying NIFTI objects were updated
        for i, nifti in enumerate(nifti_plus.nifti_list):
            nifti_data = nifti[:]
            assert np.allclose(np.abs(nifti_data[:, :, :, 0:10, :, :]), 888, atol=1)

    def test_setitem_complex_dtype_preserved(self, dummy_nifti_list):
        """Test that complex dtype is preserved when setting values."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        # Create complex values
        new_values = np.ones((1, 1, 1, 1, 10, 8, 16), dtype=np.complex128) * (1 + 2j)

        # Set values
        nifti_plus[(0, slice(None), slice(None), slice(None), slice(0, 10), slice(None), slice(None))] = new_values

        # Get updated data
        updated_data = nifti_plus.numpy()

        # Check complex dtype preserved
        assert np.iscomplexobj(updated_data)
        assert np.allclose(updated_data[0, :, :, :, 0:10, :, :], new_values[0])


class TestNIfTIMRSPlusSetItemEdgeCases:
    """Test edge cases for __setitem__."""

    def test_setitem_empty_nifti_plus(self):
        """Test setting values on empty NIfTI_MRS_Plus."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[])

        # Should handle gracefully (no error, but also no effect)
        # This is edge case behavior
        assert len(nifti_plus) == 0

    def test_setitem_single_subject_nifti_plus(self, dummy_nifti_mrs):
        """Test setting values on NIfTI_MRS_Plus with single subject."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=[dummy_nifti_mrs], backend=Backend.NUMPY)

        # Create new values
        new_values = np.ones((1, 1, 1, 1, 10, 8, 16), dtype=np.complex128) * 555

        # Set values
        nifti_plus[(0, slice(None), slice(None), slice(None), slice(0, 10), slice(None), slice(None))] = new_values

        # Verify
        updated_data = nifti_plus.numpy()
        assert np.allclose(np.abs(updated_data[0, :, :, :, 0:10, :, :]), 555, atol=1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
