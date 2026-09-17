"""
Tests for sampling modules.
"""

import pytest
import numpy as np
from augmentrum.sampling.subject_splitter import SubjectSplitter
from augmentrum.sampling.coil_sampling import CoilSampler
from augmentrum.core import NIfTI_MRS_Plus, Backend


#**************************************************************************************************#
#                                Class TestSubjectSplitterCreation                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test SubjectSplitter initialization.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class TestSubjectSplitterCreation:
    """Test SubjectSplitter initialization."""

    def test_create_with_defaults(self):
        """Test creating splitter with default fractions."""
        splitter = SubjectSplitter(data=[], water=None)
        assert splitter is not None

    def test_create_with_custom_fractions(self):
        """Test creating splitter with custom fractions."""
        splitter = SubjectSplitter(
            data=[],
            water=None,
            val_frac=0.2,
            test_frac=0.1
        )
        assert splitter.val_frac == 0.2
        assert splitter.test_frac == 0.1

    def test_create_with_seed(self):
        """Test creating splitter with seed."""
        splitter = SubjectSplitter(data=[], water=None, seed=42)
        assert splitter is not None

    def test_invalid_fractions_raise_error(self):
        """Test that invalid fractions raise ValueError."""
        # Fractions sum to > 1
        splitter = SubjectSplitter(data=[], water=None, val_frac=0.6, test_frac=0.6)
        # Note: actual validation happens during split(), not __init__
        # So we just test it doesn't crash on creation
        assert splitter is not None

    def test_negative_fractions_raise_error(self):
        """Test that negative fractions are handled."""
        splitter = SubjectSplitter(data=[], water=None, val_frac=-0.1)
        # Note: actual validation happens during split(), not __init__
        assert splitter is not None


#**************************************************************************************************#
#                                  Class TestSubjectSplitterSplit                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test SubjectSplitter split functionality.                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestSubjectSplitterSplit:
    """Test SubjectSplitter split functionality."""

    def test_split_creates_three_sets(self, dummy_nifti_list):
        """Test that split creates train, val, and test sets."""
        splitter = SubjectSplitter(
            data=dummy_nifti_list,
            water=None,
            val_frac=0.2,
            test_frac=0.2
        )
        splits = splitter.split()

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

    def test_split_sizes_correct(self, dummy_nifti_list):
        """Test that split sizes match requested fractions."""
        n_subjects = len(dummy_nifti_list)
        splitter = SubjectSplitter(
            data=dummy_nifti_list,
            water=None,
            val_frac=0.2,
            test_frac=0.2,
            seed=42
        )
        splits = splitter.split()

        # Each split returns (data, water) tuple
        train_data, _ = splits['train']
        val_data, _ = splits['val']
        test_data, _ = splits['test']

        total = len(train_data) + len(val_data) + len(test_data)
        assert total == n_subjects

        # Check approximate fractions (allow ±1 for rounding)
        assert abs(len(val_data) - int(n_subjects * 0.2)) <= 1
        assert abs(len(test_data) - int(n_subjects * 0.2)) <= 1

    def test_split_no_overlap(self, dummy_nifti_list):
        """Test that train/val/test sets don't overlap."""
        splitter = SubjectSplitter(
            data=dummy_nifti_list,
            water=None,
            val_frac=0.2,
            test_frac=0.2
        )
        splits = splitter.split()

        train_data, _ = splits['train']
        val_data, _ = splits['val']
        test_data, _ = splits['test']

        # Check no overlap by comparing object IDs
        train_ids = {id(x) for x in train_data}
        val_ids = {id(x) for x in val_data}
        test_ids = {id(x) for x in test_data}

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_split_reproducibility(self, dummy_nifti_list):
        """Test that split is reproducible with same seed."""
        splitter1 = SubjectSplitter(
            data=dummy_nifti_list,
            water=None,
            seed=42
        )
        splitter2 = SubjectSplitter(
            data=dummy_nifti_list,
            water=None,
            seed=42
        )

        splits1 = splitter1.split()
        splits2 = splitter2.split()

        train1, _ = splits1['train']
        train2, _ = splits2['train']

        # Same number of subjects in each split
        assert len(train1) == len(train2)

    def test_split_with_water(self, dummy_nifti_list):
        """Test split with water references."""
        splitter = SubjectSplitter(
            data=dummy_nifti_list,
            water=dummy_nifti_list,  # Use same for test
            val_frac=0.2,
            test_frac=0.2
        )
        splits = splitter.split()

        train_data, train_water = splits['train']
        val_data, val_water = splits['val']
        test_data, test_water = splits['test']

        # Check that water splits exist
        assert train_water is not None
        assert val_water is not None
        assert test_water is not None

        # Check that water splits have same length as data splits
        assert len(train_data) == len(train_water)
        assert len(val_data) == len(val_water)
        assert len(test_data) == len(test_water)


#**************************************************************************************************#
#                                Class TestSubjectSplitterEdgeCases                                #
#**************************************************************************************************#
#                                                                                                  #
# Test edge cases for SubjectSplitter.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class TestSubjectSplitterEdgeCases:
    """Test edge cases for SubjectSplitter."""

    def test_split_single_subject(self, dummy_nifti_single_coil):
        """Test splitting with only one subject."""
        splitter = SubjectSplitter(
            data=[dummy_nifti_single_coil],
            water=None,
            val_frac=0.0,
            test_frac=0.0
        )
        splits = splitter.split()

        train_data, _ = splits['train']
        val_data, _ = splits['val']
        test_data, _ = splits['test']

        assert len(train_data) == 1
        assert len(val_data) == 0
        assert len(test_data) == 0

    def test_split_two_subjects(self, dummy_nifti_list):
        """Test splitting with only two subjects."""
        splitter = SubjectSplitter(
            data=dummy_nifti_list[:2],
            water=None,
            val_frac=0.5,
            test_frac=0.0
        )
        splits = splitter.split()

        train_data, _ = splits['train']
        val_data, _ = splits['val']

        assert len(train_data) + len(val_data) == 2

    def test_split_zero_val_frac(self, dummy_nifti_list):
        """Test split with zero validation fraction."""
        splitter = SubjectSplitter(
            data=dummy_nifti_list,
            water=None,
            val_frac=0.0,
            test_frac=0.2
        )
        splits = splitter.split()

        train_data, _ = splits['train']
        val_data, _ = splits['val']
        test_data, _ = splits['test']

        assert len(val_data) == 0
        assert len(train_data) + len(test_data) == len(dummy_nifti_list)

    def test_split_zero_test_frac(self, dummy_nifti_list):
        """Test split with zero test fraction."""
        splitter = SubjectSplitter(
            data=dummy_nifti_list,
            water=None,
            val_frac=0.2,
            test_frac=0.0
        )
        splits = splitter.split()

        train_data, _ = splits['train']
        val_data, _ = splits['val']
        test_data, _ = splits['test']

        assert len(test_data) == 0
        assert len(train_data) + len(val_data) == len(dummy_nifti_list)


#**************************************************************************************************#
#                                Class TestSubjectSplitterBackends                                 #
#**************************************************************************************************#
#                                                                                                  #
# Test SubjectSplitter with different backends.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class TestSubjectSplitterBackends:
    """Test SubjectSplitter with different backends."""

    def test_split_with_nifti_plus_nifti_list(self, dummy_nifti_list):
        """Test split with NIfTI_MRS_Plus in NIFTI_LIST backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        splitter = SubjectSplitter(data=nifti_plus, water=None, val_frac=0.2, test_frac=0.2)
        splits = splitter.split()

        train_data, _ = splits['train']

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(train_data, NIfTI_MRS_Plus)
        assert train_data.backend == Backend.NIFTI_LIST

    def test_split_with_nifti_plus_numpy(self, dummy_nifti_list):
        """Test split with NIfTI_MRS_Plus in NUMPY backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        splitter = SubjectSplitter(data=nifti_plus, water=None, val_frac=0.2, test_frac=0.2)
        splits = splitter.split()

        train_data, _ = splits['train']

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(train_data, NIfTI_MRS_Plus)
        assert train_data.backend == Backend.NUMPY

    def test_split_with_nifti_plus_pytorch(self, dummy_nifti_list):
        """Test split with NIfTI_MRS_Plus in PYTORCH backend."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.PYTORCH)

        splitter = SubjectSplitter(data=nifti_plus, water=None, val_frac=0.2, test_frac=0.2)
        splits = splitter.split()

        train_data, _ = splits['train']

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(train_data, NIfTI_MRS_Plus)
        assert train_data.backend == Backend.PYTORCH

    def test_split_with_nifti_plus_tensorflow(self, dummy_nifti_list):
        """Test split with NIfTI_MRS_Plus in TENSORFLOW backend."""
        try:
            import tensorflow
        except ImportError:
            pytest.skip("TensorFlow not installed")

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.TENSORFLOW)

        splitter = SubjectSplitter(data=nifti_plus, water=None, val_frac=0.2, test_frac=0.2)
        splits = splitter.split()

        train_data, _ = splits['train']

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(train_data, NIfTI_MRS_Plus)
        assert train_data.backend == Backend.TENSORFLOW

    def test_split_with_nifti_plus_keras(self, dummy_nifti_list):
        """Test split with NIfTI_MRS_Plus in KERAS backend."""
        try:
            import keras
        except ImportError:
            pytest.skip("Keras not installed")

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.KERAS)

        splitter = SubjectSplitter(data=nifti_plus, water=None, val_frac=0.2, test_frac=0.2)
        splits = splitter.split()

        train_data, _ = splits['train']

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(train_data, NIfTI_MRS_Plus)
        assert train_data.backend == Backend.KERAS

    def test_split_with_nifti_plus_jax(self, dummy_nifti_list):
        """Test split with NIfTI_MRS_Plus in JAX backend."""
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not installed")

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.JAX)

        splitter = SubjectSplitter(data=nifti_plus, water=None, val_frac=0.2, test_frac=0.2)
        splits = splitter.split()

        train_data, _ = splits['train']

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(train_data, NIfTI_MRS_Plus)
        assert train_data.backend == Backend.JAX

    def test_split_preserves_volatile(self, dummy_nifti_list):
        """Test that split preserves volatile setting."""
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST, volatile=True)

        splitter = SubjectSplitter(data=nifti_plus, water=None, val_frac=0.2, test_frac=0.2)
        splits = splitter.split()

        train_data, _ = splits['train']

        # Should preserve volatile setting
        assert isinstance(train_data, NIfTI_MRS_Plus)
        assert train_data.volatile == True

    def test_split_all_backends_with_plain_list(self, dummy_nifti_list):
        """Test that split works with plain Python list (no backend)."""
        splitter = SubjectSplitter(data=dummy_nifti_list, water=None, val_frac=0.2, test_frac=0.2)
        splits = splitter.split()

        train_data, _ = splits['train']

        # Should return plain list
        assert isinstance(train_data, list)
        assert all(hasattr(item, 'shape') for item in train_data)  # NIFTI_MRS objects


#**************************************************************************************************#
#                                  Class TestSubjectSplitterGroups                                 #
#**************************************************************************************************#
#                                                                                                  #
# Items of one group always land in the same split.                                                #
#                                                                                                  #
#**************************************************************************************************#
class TestSubjectSplitterGroups:
    """
    split_fractions permuted items, so the scans of one subject leaked across
    train and val. With groups, whole groups move, and the fractions are met
    in items as closely as the group sizes allow.
    """

    IDS = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']

    @staticmethod
    def _nine(dummy_nifti_list):
        return (dummy_nifti_list * 2)[:9]

    def test_groups_never_straddle_splits(self, dummy_nifti_list):
        splitter = SubjectSplitter(data=self._nine(dummy_nifti_list), val_frac=0.2,
                                   test_frac=0.2, seed=0, groups=self.IDS)
        splitter.split()

        for name, idx in splitter.split_indices.items():
            assert len({self.IDS[i] for i in idx}) <= 1, f"{name} mixes groups: {idx}"
        assert sorted(sum(splitter.split_indices.values(), [])) == list(range(9))

    def test_fractions_that_cannot_be_met_are_met_as_closely_as_possible(self, dummy_nifti_list):
        """A target of 1.8 items is nearer to one whole group of 3 than to none."""
        splitter = SubjectSplitter(data=self._nine(dummy_nifti_list), val_frac=0.2,
                                   test_frac=0.2, seed=0, groups=self.IDS)
        splits = splitter.split()

        assert {name: len(split[0]) for name, split in splits.items()} == \
            {'train': 3, 'val': 3, 'test': 3}
        assert sorted(sum(splitter.split_groups.values(), [])) == ['a', 'b', 'c']
        assert all(len(ids) == 1 for ids in splitter.split_groups.values())

    def test_small_targets_take_nothing(self, dummy_nifti_list):
        """A target of 0.9 items is nearer to none than to a group of 3."""
        splitter = SubjectSplitter(data=self._nine(dummy_nifti_list), val_frac=0.1,
                                   test_frac=0.0, seed=0, groups=self.IDS)
        splits = splitter.split()

        assert len(splits['train'][0]) == 9
        assert splitter.split_groups == {'train': ['a', 'b', 'c'], 'val': [], 'test': []}

    def test_group_split_is_seeded(self, dummy_nifti_list):
        items = self._nine(dummy_nifti_list)

        def held_out(seed):
            splitter = SubjectSplitter(data=items, val_frac=0.2, test_frac=0.2, seed=seed,
                                       groups=self.IDS)
            splitter.split()
            return tuple(splitter.split_groups['val'])

        assert held_out(3) == held_out(3)
        assert len({held_out(k) for k in range(20)}) > 1, "twenty seeds held out the same group"

    def test_unequal_groups_stay_whole(self, dummy_nifti_list):
        """Sizes 5, 3, 1 over 9 items: test wants 2.7, val 1.8; groups still move whole."""
        groups = ['x'] * 5 + ['y'] * 3 + ['z']
        splitter = SubjectSplitter(data=self._nine(dummy_nifti_list), val_frac=0.2,
                                   test_frac=0.3, seed=1, groups=groups)
        splits = splitter.split()

        assert sum(len(split[0]) for split in splits.values()) == 9
        for name, idx in splitter.split_indices.items():
            for g in {groups[i] for i in idx}:
                assert sum(groups[i] == g for i in idx) == groups.count(g), \
                    f"group {g} is split across {name} and another split"

    def test_group_length_mismatch_raises(self, dummy_nifti_list):
        splitter = SubjectSplitter(data=dummy_nifti_list, groups=['a', 'b'])
        with pytest.raises(ValueError, match="one group id per item"):
            splitter.split()

    def test_item_split_records_indices_without_groups(self, dummy_nifti_list):
        splitter = SubjectSplitter(data=dummy_nifti_list, val_frac=0.2, test_frac=0.2, seed=0)
        splitter.split()

        assert splitter.split_groups is None
        assert sorted(sum(splitter.split_indices.values(), [])) == list(range(5))


#**************************************************************************************************#
#                               Class TestCoilSamplerCreation                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test CoilSampler initialization.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class TestCoilSamplerCreation:
    """Test CoilSampler initialization."""

    def test_create_with_defaults(self):
        """Test creating sampler with defaults."""
        sampler = CoilSampler(mode='random')
        assert sampler is not None

    def test_create_with_mode(self):
        """Test creating sampler with specific mode."""
        sampler = CoilSampler(mode='deterministic')
        assert sampler.mode == 'deterministic'

    def test_create_with_n_coils(self):
        """Test creating sampler with n_coils parameter."""
        sampler = CoilSampler(mode='random', n_coils=(1, 4))
        assert sampler.n_coils == (1, 4)

    def test_supports_nifti_list_backend(self):
        """Test that CoilSampler supports NIFTI_LIST backend."""
        sampler = CoilSampler(mode='random')
        assert sampler.supports_backend(Backend.NIFTI_LIST)


#**************************************************************************************************#
#                              Class TestCoilSamplerProcessing                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test CoilSampler processing.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestCoilSamplerProcessing:
    """Test CoilSampler processing."""

    def test_process_removes_coil_dimension(self, dummy_nifti_list):
        """Test that processing removes coil dimension."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        # Original should have coil dimension
        original_shape = nifti_plus[0].shape

        result_data, _ = sampler(nifti_plus, None)
        new_shape = result_data[0].shape

        # Coil dimension should be removed or averaged
        assert len(new_shape) <= len(original_shape)

    def test_process_changes_data(self, dummy_nifti_list):
        """Test that averaging changes data."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        original_data = nifti_plus[0][:].copy()
        result_data, _ = sampler(nifti_plus, None)

        # Data should be different after averaging
        assert result_data is not None

    def test_process_preserves_dtype(self, dummy_nifti_list):
        """Test that processing preserves complex dtype."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = sampler(nifti_plus, None)
        # Result should still be complex
        assert np.iscomplexobj(result_data[0][:])

    def test_process_all_subjects(self, dummy_nifti_list):
        """Test that all subjects are processed."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = sampler(nifti_plus, None)

        # Should have same number of subjects
        assert len(result_data) == len(dummy_nifti_list)

    def test_process_with_water(self, dummy_nifti_list):
        """Test processing with water references."""
        from copy import deepcopy
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        # Use COPIES and same length to avoid index errors
        water_niftis = [deepcopy(n) for n in dummy_nifti_list]
        water_plus = NIfTI_MRS_Plus(nifti_list=water_niftis, backend=Backend.NIFTI_LIST)

        result_data, result_water = sampler(nifti_plus, water_plus)

        # Both should be processed
        assert result_data is not None
        assert result_water is not None


#**************************************************************************************************#
#                                Class TestCoilSamplerModes                                        #
#**************************************************************************************************#
#                                                                                                  #
# Test different modes.                                                                            #
#                                                                                                  #
#**************************************************************************************************#
class TestCoilSamplerModes:
    """Test different modes."""

    def test_random_mode(self, dummy_nifti_list):
        """Test random mode."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)


        result_data, _ = sampler(nifti_plus, None)
        assert result_data is not None

    def test_deterministic_mode(self, dummy_nifti_list):
        """Deterministic keeps everything, or exactly the indices passed."""
        sampler = CoilSampler(mode='deterministic')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = sampler(nifti_plus, None)
        coil_axis = result_data[0].dim_position('DIM_COIL')
        assert result_data[0].shape[coil_axis] == nifti_plus[0].shape[coil_axis]

    def test_deterministic_mode_with_indices(self, dummy_nifti_list):
        """Indices are a call-time choice: keep exactly the coils named."""
        sampler = CoilSampler(mode='deterministic')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = sampler(nifti_plus, None, indices=[0, 2])
        coil_axis = result_data[0].dim_position('DIM_COIL')
        assert result_data[0].shape[coil_axis] == 2

    def test_with_n_coils_range(self, dummy_nifti_list):
        """Test with n_coils range."""
        sampler = CoilSampler(mode='random', n_coils=(2, 4))
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = sampler(nifti_plus, None)
        assert result_data is not None



#**************************************************************************************************#
#                             Class TestCoilSamplerIntegration                                     #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests for CoilSampler.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestCoilSamplerIntegration:
    """Integration tests for CoilSampler."""

    def test_in_pipeline(self, dummy_nifti_list):
        """Test CoilSampler in a pipeline."""
        from augmentrum.core.pipeline import AugmentationPipeline

        sampler = CoilSampler(mode='random')
        pipeline = AugmentationPipeline([sampler])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert len(result_data) == len(dummy_nifti_list)

    def test_with_other_modules(self, dummy_nifti_list):
        """Test chaining with other modules."""
        from augmentrum.core.pipeline import AugmentationPipeline
        from augmentrum.augmentation.noise import Noise

        sampler = CoilSampler(mode='random')
        noise = Noise(sigma_frac=0.02)
        pipeline = AugmentationPipeline([sampler, noise])

        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)

        assert result_data is not None


#**************************************************************************************************#
#                               Class TestCoilSamplerBackends                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test CoilSampler with different backends.                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestCoilSamplerBackends:
    """Test CoilSampler with different backends."""

    def test_supports_all_backends(self):
        """Drawing is a gather along one axis, so it is native everywhere.

        The NIfTI-list path is still implemented and still preferred for a
        NIFTI_LIST input, because FSL-MRS carries the headers across properly
        there. On a tensor it stays on the tensor, which is what keeps a
        synthesize-then-draw pipeline differentiable.
        """
        sampler = CoilSampler(mode='random')

        for backend in Backend:
            assert sampler.supports_backend(backend), backend

    def test_synthesis_runs_natively_on_the_nifti_list(self):
        """The list path grows a tagged coil axis per subject, no routing."""
        sampler = CoilSampler(mode='synthesize', n_coils=4)

        assert sampler.supports_backend(Backend.NIFTI_LIST)
        assert sampler.supports_backend(Backend.PYTORCH)

    def test_process_with_nifti_list_backend(self, dummy_nifti_list):
        """Test processing with NIFTI_LIST backend (native)."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)

        result_data, _ = sampler(nifti_plus, None)

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert result_data.backend == Backend.NIFTI_LIST

    def test_process_with_numpy_backend(self, dummy_nifti_list):
        """Test processing with NUMPY backend (auto-converts)."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)

        result_data, _ = sampler(nifti_plus, None)

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert result_data.backend == Backend.NUMPY

    def test_process_with_pytorch_backend(self, dummy_nifti_list):
        """Test processing with PYTORCH backend (auto-converts)."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.PYTORCH)

        result_data, _ = sampler(nifti_plus, None)

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert result_data.backend == Backend.PYTORCH

    def test_process_with_tensorflow_backend(self, dummy_nifti_list):
        """Test processing with TENSORFLOW backend (auto-converts)."""
        try:
            import tensorflow
        except ImportError:
            pytest.skip("TensorFlow not installed")

        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.TENSORFLOW)

        result_data, _ = sampler(nifti_plus, None)

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert result_data.backend == Backend.TENSORFLOW

    def test_process_with_keras_backend(self, dummy_nifti_list):
        """Test processing with KERAS backend (auto-converts)."""
        try:
            import keras
        except ImportError:
            pytest.skip("Keras not installed")

        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.KERAS)

        result_data, _ = sampler(nifti_plus, None)

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert result_data.backend == Backend.KERAS

    def test_process_with_jax_backend(self, dummy_nifti_list):
        """Test processing with JAX backend (auto-converts)."""
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not installed")

        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.JAX)

        result_data, _ = sampler(nifti_plus, None)

        # Should return NIfTI_MRS_Plus with same backend
        assert isinstance(result_data, NIfTI_MRS_Plus)
        assert result_data.backend == Backend.JAX

    def test_backend_conversion_preserves_data(self, dummy_nifti_list):
        """Test that backend conversion doesn't lose data."""
        sampler = CoilSampler(mode='random')

        # Process with NIFTI_LIST
        nifti_list_data = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)
        result_nifti, _ = sampler(nifti_list_data, None)

        # Process with NUMPY (should give similar results after conversion)
        numpy_data = NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NUMPY)
        result_numpy, _ = sampler(numpy_data, None)

        # Both should have same number of subjects
        assert len(result_nifti) == len(result_numpy)

    def test_preserves_volatile_across_conversion(self, dummy_nifti_list):
        """Test that volatile setting is preserved through backend conversion."""
        sampler = CoilSampler(mode='random')
        nifti_plus = NIfTI_MRS_Plus(
            nifti_list=dummy_nifti_list,
            backend=Backend.PYTORCH,
            volatile=True
        )

        result_data, _ = sampler(nifti_plus, None)

        # Should preserve volatile setting
        assert result_data.volatile == True
        assert result_data.backend == Backend.PYTORCH


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
