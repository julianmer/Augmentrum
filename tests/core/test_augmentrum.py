"""
Tests for the main Augmentrum class API.

Tests cover:
- Augmentrum initialization with various configurations
- Pipeline creation and execution
- Dataloader generation for different splits
- Backend configuration
- Integration with the augmentation pipeline
"""

import pytest
import numpy as np
from augmentrum import Augmentrum
from augmentrum.core import Backend, NIfTI_MRS_Plus


class TestAugmentrumCreation:
    """Test Augmentrum initialization."""

    def test_create_with_data_only(self, dummy_nifti_list):
        """Test creating Augmentrum with data only."""
        augmenter = Augmentrum(data=dummy_nifti_list)

        assert augmenter is not None
        assert augmenter.data is not None
        assert len(augmenter.data) == len(dummy_nifti_list)

    def test_create_with_data_and_water(self, dummy_nifti_list):
        """Test creating Augmentrum with water reference."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            water=dummy_nifti_list[:2]
        )

        assert augmenter.water is not None
        assert len(augmenter.water) == 2

    def test_create_with_default_pipeline(self, dummy_nifti_list):
        """Test default pipeline creation."""
        augmenter = Augmentrum(data=dummy_nifti_list)

        assert augmenter.pipeline is not None
        assert len(augmenter.pipeline.steps) > 0

    def test_create_with_custom_pipeline_list(self, dummy_nifti_list):
        """Test custom pipeline with list of strings."""
        pipeline = ['coil_sampling', 'processing', 'noise']
        augmenter = Augmentrum(data=dummy_nifti_list, pipeline=pipeline)

        assert len(augmenter.pipeline.steps) == len(pipeline)

    def test_create_with_backend(self, dummy_nifti_list):
        """Test creating with specific backend."""
        augmenter = Augmentrum(data=dummy_nifti_list, backend='numpy')

        assert augmenter.backend == Backend.NUMPY

    def test_create_with_split_ratios(self, dummy_nifti_list):
        """Test creating with custom split ratios."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            val_frac=0.2,
            test_frac=0.1
        )

        assert augmenter.val_frac == 0.2
        assert augmenter.test_frac == 0.1


class TestAugmentrumPipeline:
    """Test pipeline functionality."""

    def test_pipeline_string_names_resolution(self, dummy_nifti_list):
        """Test that pipeline string names are resolved correctly."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise']
        )

        assert len(augmenter.pipeline.steps) == 1

    def test_empty_pipeline(self, dummy_nifti_list):
        """Test empty pipeline."""
        augmenter = Augmentrum(data=dummy_nifti_list, pipeline=[])

        assert len(augmenter.pipeline.steps) == 0


class TestAugmentrumDataloaders:
    """Test dataloader generation."""

    def test_get_dataloader_default(self, dummy_nifti_list):
        """Test getting dataloader with defaults."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.get_dataloader()

        assert loader is not None

    def test_get_dataloader_with_batch_size(self, dummy_nifti_list):
        """Test getting dataloader with specific batch size."""
        augmenter = Augmentrum(data=dummy_nifti_list)
        loader = augmenter.get_dataloader(batch_size=2)

        assert loader is not None

    def test_get_dataloader_with_shuffle(self, dummy_nifti_list):
        """Test getting dataloader with shuffle."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.get_dataloader(shuffle=True)

        assert loader is not None

    def test_get_dataloader_with_backend(self, dummy_nifti_list):
        """Test getting dataloader with specific backend."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.get_dataloader(backend='numpy')

        assert loader is not None

    def test_train_dataloader(self, dummy_nifti_list):
        """Test train_dataloader() convenience method."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.train_dataloader()

        assert loader is not None

    def test_val_dataloader(self, dummy_nifti_list):
        """Test val_dataloader() convenience method."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.val_dataloader()

        assert loader is not None

    def test_test_dataloader(self, dummy_nifti_list):
        """Test test_dataloader() convenience method."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.test_dataloader()

        assert loader is not None

    def test_all_dataloaders(self, dummy_nifti_list):
        """Test getting all dataloaders."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)

        train_loader = augmenter.train_dataloader()
        val_loader = augmenter.val_dataloader()
        test_loader = augmenter.test_dataloader()

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None


class TestAugmentrumIntegration:
    """Integration tests for complete workflows."""

    def test_dataloader_iteration(self, dummy_nifti_list):
        """Test iterating through dataloader."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2, pipeline=[])
        loader = augmenter.train_dataloader()

        # Try to get one batch
        batch = next(iter(loader))
        assert batch is not None

    def test_with_single_subject(self, dummy_nifti_single_coil):
        """Test with single subject."""
        augmenter = Augmentrum(data=[dummy_nifti_single_coil], batch_size=1)
        loader = augmenter.get_dataloader()

        assert loader is not None


class TestAugmentrumEdgeCases:
    """Test edge cases and error handling."""

    def test_batch_size_larger_than_dataset(self, dummy_nifti_list):
        """Test batch size larger than dataset."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=100)
        loader = augmenter.get_dataloader()

        assert loader is not None

    def test_water_different_length_than_data(self, dummy_nifti_list):
        """Test water reference with different length than data."""
        # This should work - water can be different length
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            water=dummy_nifti_list[:2]
        )

        assert len(augmenter.data) == len(dummy_nifti_list)
        assert len(augmenter.water) == 2


class TestAugmentrumRepr:
    """Test string representation."""

    def test_repr(self, dummy_nifti_list):
        """Test __repr__ method."""
        augmenter = Augmentrum(data=dummy_nifti_list)
        repr_str = repr(augmenter)

        assert 'Augmentrum' in repr_str
        assert 'subjects' in repr_str.lower()

    def test_str(self, dummy_nifti_list):
        """Test __str__ method."""
        augmenter = Augmentrum(data=dummy_nifti_list)
        str_str = str(augmenter)

        assert 'Augmentrum' in str_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
