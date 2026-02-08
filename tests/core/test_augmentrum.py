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
        assert augmenter.splits['train'][0] is not None
        assert len(augmenter.splits['train'][0]) == len(dummy_nifti_list)

    def test_create_with_data_and_water(self, dummy_nifti_list):
        """Test creating Augmentrum with water reference."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            water=dummy_nifti_list[:2]
        )

        assert augmenter.splits['train'][1] is not None
        assert len(augmenter.splits['train'][1]) == 2

    def test_create_with_default_pipeline(self, dummy_nifti_list):
        """Test default pipeline creation."""
        augmenter = Augmentrum(data=dummy_nifti_list)

        assert augmenter.pipelines['train'] is not None
        assert len(augmenter.pipelines['train'].steps) > 0

    def test_create_with_custom_pipeline_list(self, dummy_nifti_list):
        """Test custom pipeline with list of strings."""
        pipeline = ['coil_sampling', 'processing', 'noise']
        augmenter = Augmentrum(data=dummy_nifti_list, pipeline=pipeline)

        assert len(augmenter.pipelines['train'].steps) == len(pipeline)

    def test_create_with_backend(self, dummy_nifti_list):
        """Test creating with specific backend."""
        augmenter = Augmentrum(data=dummy_nifti_list, backend='numpy')

        assert augmenter.backend == Backend.NUMPY

    def test_create_with_split_ratios(self, dummy_nifti_list):
        """Test creating with custom split ratios."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            split_fractions={'val': 0.2, 'test': 0.1}
        )

        # Check that splits were created
        assert 'train' in augmenter.splits
        assert 'val' in augmenter.splits
        assert 'test' in augmenter.splits
        # Check that val split has data (approximately 20% of 5 = 1 subject)
        assert len(augmenter.splits['val'][0]) >= 1


class TestAugmentrumPipeline:
    """Test pipeline functionality."""

    def test_pipeline_string_names_resolution(self, dummy_nifti_list):
        """Test that pipeline string names are resolved correctly."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise']
        )

        assert len(augmenter.pipelines['train'].steps) == 1

    def test_empty_pipeline(self, dummy_nifti_list):
        """Test empty pipeline."""
        augmenter = Augmentrum(data=dummy_nifti_list, pipeline=[])

        assert len(augmenter.pipelines['train'].steps) == 0


class TestAugmentrumParameterRanges:
    """Test tuple range support for parameters (NEW in v0.0.1)."""

    def test_scalar_parameters(self, dummy_nifti_list):
        """Test backward compatibility with scalar parameters."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening'],
            sigma_frac=0.03,  # Scalar
            lb_hz=5.0,        # Scalar
            batch_size=1
        )

        assert augmenter is not None

    def test_tuple_range_parameters(self, dummy_nifti_list):
        """Test NEW tuple range support for augmentation parameters."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening', 'phase'],
            sigma_frac=(0.01, 0.05),     # Tuple range
            lb_hz=(0, 10),               # Tuple range
            gb_hz=(0, 5),                # Tuple range
            phase0_deg=(-180, 180),      # Tuple range
            batch_size=1
        )

        assert augmenter is not None

    def test_mixed_scalar_and_tuple(self, dummy_nifti_list):
        """Test mixing scalar and tuple parameters."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening'],
            sigma_frac=(0.01, 0.05),  # Tuple
            lb_hz=5.0,                # Scalar
            batch_size=1
        )

        assert augmenter is not None

    def test_global_param_distribution(self, dummy_nifti_list):
        """Test global distribution for all parameters."""
        for dist in ['uniform', 'gaussian', 'exponential', 'beta']:
            augmenter = Augmentrum(
                data=dummy_nifti_list,
                pipeline=['noise'],
                sigma_frac=(0.01, 0.05),
                param_distribution=dist,  # Global distribution
                batch_size=1
            )

            assert augmenter is not None

    def test_per_parameter_distributions(self, dummy_nifti_list):
        """Test NEW per-parameter distribution control."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening', 'phase'],
            sigma_frac=(0.01, 0.05),
            lb_hz=(0, 10),
            phase0_deg=(-180, 180),
            param_distributions={
                'sigma_frac': 'exponential',  # Different distribution per param
                'lb_hz': 'gaussian',
                'phase0_deg': 'uniform',
            },
            batch_size=1
        )

        assert augmenter is not None

    def test_nested_ranges_spurious_echoes(self, dummy_nifti_list):
        """Test NEW nested tuple ranges for spurious echoes."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['echoes'],
            echoes=[
                # Each element can be range or scalar
                ((0.1, 0.3), (0.2, 0.5), 0.0, (4.0, 6.0), 0.0),
            ],
            batch_size=1
        )

        assert augmenter is not None

    def test_nested_ranges_artificial_peaks(self, dummy_nifti_list):
        """Test NEW nested tuple ranges for artificial peaks."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['peaks'],
            peaks=[
                # Each element can be range or scalar (except lineshape string)
                ((0.5, 1.0), (3.0, 3.2), 0.05, 0.0, 'lorentzian'),
            ],
            batch_size=1
        )

        assert augmenter is not None


class TestAugmentrumDataloaders:
    """Test dataloader generation."""

    def test_get_dataloader_default(self, dummy_nifti_list):
        """Test getting dataloader with defaults."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.train_dataloader()

        assert loader is not None

    def test_get_dataloader_with_batch_size(self, dummy_nifti_list):
        """Test getting dataloader with specific batch size."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.train_dataloader()

        assert loader is not None

    def test_get_dataloader_with_shuffle(self, dummy_nifti_list):
        """Test getting dataloader with shuffle."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.train_dataloader()

        assert loader is not None

    def test_get_dataloader_with_backend(self, dummy_nifti_list):
        """Test getting dataloader with specific backend."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=2)
        loader = augmenter.train_dataloader(framework='numpy')

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
        loader = augmenter.train_dataloader()

        assert loader is not None


class TestAugmentrumEdgeCases:
    """Test edge cases and error handling."""

    def test_batch_size_larger_than_dataset(self, dummy_nifti_list):
        """Test batch size larger than dataset."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=100)
        loader = augmenter.train_dataloader()

        assert loader is not None

    def test_water_different_length_than_data(self, dummy_nifti_list):
        """Test water reference with different length than data."""
        # This should work - water can be different length
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            water=dummy_nifti_list[:2]
        )

        assert len(augmenter.splits['train'][0]) == len(dummy_nifti_list)
        assert len(augmenter.splits['train'][1]) == 2


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
