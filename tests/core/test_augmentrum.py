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


#**************************************************************************************************#
#                                   Class TestAugmentrumCreation                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test Augmentrum initialization.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                   Class TestAugmentrumPipeline                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test pipeline functionality.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                               Class TestAugmentrumParameterRanges                                #
#**************************************************************************************************#
#                                                                                                  #
# Test tuple range support for parameters (NEW in v0.0.1).                                         #
#                                                                                                  #
#**************************************************************************************************#
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
            zero_order_deg=(-180, 180),  # Tuple range
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
            zero_order_deg=(-180, 180),
            param_distributions={
                'sigma_frac': 'exponential',  # Different distribution per param
                'lb_hz': 'gaussian',
                'zero_order_deg': 'uniform',
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


#**************************************************************************************************#
#                                 Class TestAugmentrumDataloaders                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test dataloader generation.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                 Class TestAugmentrumIntegration                                  #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests for complete workflows.                                                        #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                  Class TestAugmentrumEdgeCases                                   #
#**************************************************************************************************#
#                                                                                                  #
# Test edge cases and error handling.                                                              #
#                                                                                                  #
#**************************************************************************************************#
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


#**************************************************************************************************#
#                                     Class TestAugmentrumRepr                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test string representation.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
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


#*************#
#   helpers   #
#*************#
def _small_niftis(n=6, n_pts=128, subject_ids=None, seed=0):
    """
    *n* single-voxel FIDs, short enough that a pipeline runs in milliseconds.

    Args:
        subject_ids: One id per item written to the "SubjectID" header field,
            the way a loader that knows the subject would.
    """
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fid = (rng.standard_normal((1, 1, 1, n_pts))
               + 1j * rng.standard_normal((1, 1, 1, n_pts))).astype(np.complex64)
        nifti = gen_nifti_mrs(fid, 1 / 2000, 123.0)
        if subject_ids is not None:
            nifti.add_hdr_field('SubjectID', subject_ids[i], doc='subject of this scan')
        out.append(nifti)
    return out


def _first_batches(aug, n=3, split='train'):
    """The first *n* data batches of a fresh loader, as NumPy arrays."""
    loader = aug.dataloader(framework='numpy', split=split)
    return [np.asarray(next(loader)[0]) for _ in range(n)]


#**************************************************************************************************#
#                                 Class TestAugmentrumReproducibility                              #
#**************************************************************************************************#
#                                                                                                  #
# One seed fixes the whole run: subject draws, ranged parameters and module perturbations.         #
#                                                                                                  #
#**************************************************************************************************#
class TestAugmentrumReproducibility:
    """
    "Augmentrum(seed=...)" must make a run repeatable end to end.

    Before, the seed fixed only the split: subject indices came from the
    "random" module, ranged parameters from the global np.random state and
    every module built by name drew from OS entropy.
    """

    RANGED = dict(pipeline=['noise', 'line_broadening', 'baseline'],
                  sigma_frac=(0.0, 0.1), lb_hz=(0.0, 5.0), baseline_frac=(0.01, 0.1),
                  batch_size=3)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_same_seed_gives_identical_batches(self, backend):
        data = _small_niftis()
        first = _first_batches(Augmentrum(data=data, backend=backend, seed=1, **self.RANGED))
        second = _first_batches(Augmentrum(data=data, backend=backend, seed=1, **self.RANGED))

        for a, b in zip(first, second):
            assert np.array_equal(a, b), f"{backend}: seed=1 did not replay the run"

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_different_seeds_differ(self, backend):
        data = _small_niftis()
        one = _first_batches(Augmentrum(data=data, backend=backend, seed=1, **self.RANGED), 1)
        two = _first_batches(Augmentrum(data=data, backend=backend, seed=2, **self.RANGED), 1)

        assert not np.array_equal(one[0], two[0])

    def test_reseed_changes_and_replays_the_draws(self):
        aug = Augmentrum(data=_small_niftis(), backend='numpy', seed=1, **self.RANGED)

        aug.reseed(5)
        first = _first_batches(aug, 1)[0]
        aug.reseed(6)
        other = _first_batches(aug, 1)[0]
        aug.reseed(5)
        again = _first_batches(aug, 1)[0]

        assert not np.array_equal(first, other), "reseed(6) after reseed(5) drew the same"
        assert np.array_equal(first, again), "reseed(5) did not restart its stream"

    def test_hand_built_unseeded_pipeline_becomes_reproducible(self):
        """A pipeline assembled from unseeded modules is seeded by the Augmentrum it joins."""
        from augmentrum.core.pipeline import AugmentationPipeline
        from augmentrum.augmentation import BaselineAugmentation, Noise

        data = _small_niftis()

        def build():
            return AugmentationPipeline([Noise(sigma_frac=0.05), BaselineAugmentation()])

        one = _first_batches(Augmentrum(data=data, pipeline=build(), backend='numpy',
                                        seed=3, batch_size=2))
        two = _first_batches(Augmentrum(data=data, pipeline=build(), backend='numpy',
                                        seed=3, batch_size=2))

        for a, b in zip(one, two):
            assert np.array_equal(a, b)

    def test_explicit_module_seed_is_respected(self):
        """A module the user seeded keeps its seed under a seeded Augmentrum."""
        from augmentrum.augmentation import Noise

        aug = Augmentrum(data=_small_niftis(), pipeline=[Noise(sigma_frac=0.05, seed=7), 'noise'],
                         sigma_frac=0.05, backend='numpy', seed=1)
        pinned, derived = aug.pipelines['train'].steps

        assert pinned.rng.seed == 7
        assert derived.rng.seed != 7

    def test_unseeded_run_keeps_the_seed_it_drew(self):
        """seed=None draws a root from entropy but records it, so the run can be repeated."""
        data = _small_niftis()
        aug = Augmentrum(data=data, backend='numpy', seed=None, **self.RANGED)
        replay = Augmentrum(data=data, backend='numpy', seed=aug.seed, **self.RANGED)

        assert isinstance(aug.seed, int)
        assert np.array_equal(_first_batches(aug, 1)[0], _first_batches(replay, 1)[0])

    def test_torch_workers_draw_distinct_batches(self):
        """Two workers used to fork identical streams and yield every batch twice."""
        import torch

        aug = Augmentrum(data=_small_niftis(1), pipeline=['noise'], sigma_frac=0.1,
                         batch_size=1, backend='pytorch', seed=0)
        loader = iter(aug.as_torch_dataloader(num_workers=2))
        batches = [next(loader) for _ in range(4)]

        assert not torch.equal(batches[0], batches[1])
        assert not torch.equal(batches[2], batches[3])


#**************************************************************************************************#
#                                    Class TestAugmentrumFixedMode                                 #
#**************************************************************************************************#
#                                                                                                  #
# 'fixed' draws a ranged parameter once per split and keeps it across dataloader() calls.          #
#                                                                                                  #
#**************************************************************************************************#
class TestAugmentrumFixedMode:
    """Fixed mode re-sampled its ranges on every dataloader() call: fixed within one pass only."""

    def test_fixed_parameters_survive_dataloader_calls(self):
        aug = Augmentrum(data=_small_niftis(), pipeline=['line_broadening'], lb_hz=(0.0, 20.0),
                         mode='fixed', backend='numpy', seed=1, batch_size=6)

        first = np.asarray(next(aug.dataloader(framework='numpy', shuffle=False))[0])
        second = np.asarray(next(aug.dataloader(framework='numpy', shuffle=False))[0])

        assert np.array_equal(first, second), "a second dataloader() redrew the fixed range"

    def test_fixed_parameters_match_across_seeded_instances(self):
        kwargs = dict(pipeline=['noise'], sigma_frac=(0.0, 0.5), mode='fixed', backend='numpy',
                      seed=1, batch_size=6)
        a = Augmentrum(data=_small_niftis(), **kwargs)
        b = Augmentrum(data=_small_niftis(), **kwargs)

        assert np.array_equal(a._fixed_batch_params('train')[0]['sigma_frac'],
                              b._fixed_batch_params('train')[0]['sigma_frac'])

    def test_reseed_redraws_fixed_parameters(self):
        aug = Augmentrum(data=_small_niftis(), pipeline=['noise'], sigma_frac=(0.0, 0.5),
                         mode='fixed', backend='numpy', seed=1, batch_size=6)
        before = np.array(aug._fixed_batch_params('train')[0]['sigma_frac'])

        aug.reseed(2)

        assert not np.array_equal(before, aug._fixed_batch_params('train')[0]['sigma_frac'])


#**************************************************************************************************#
#                                  Class TestAugmentrumKwargsValidation                            #
#**************************************************************************************************#
#                                                                                                  #
# A kwarg no module accepts is a typo, and a typo must not silently switch a step off.             #
#                                                                                                  #
#**************************************************************************************************#
class TestAugmentrumKwargsValidation:
    """Unknown kwargs used to be swallowed, so 'sigma_frc' ran the noise at its default."""

    def test_unknown_kwarg_raises_with_suggestion(self):
        with pytest.raises(ValueError, match="sigma_frc.*Did you mean 'sigma_frac'"):
            Augmentrum(data=_small_niftis(), pipeline=['noise', 'line_broadening'],
                       sigma_frc=0.05, lb_hz=1.0)

    def test_unknown_kwarg_names_every_offender(self):
        with pytest.raises(ValueError) as err:
            Augmentrum(data=_small_niftis(), pipeline=['noise'], bogus=1, sigma_frc=0.05)

        assert "'bogus'" in str(err.value) and "'sigma_frc'" in str(err.value)

    def test_default_pipeline_rejects_augmentation_kwargs(self):
        """The default RawProcessor pipeline accepts its own flags, not noise levels."""
        with pytest.raises(ValueError, match="sigma_frac"):
            Augmentrum(data=_small_niftis(), sigma_frac=0.05)

    def test_global_sampling_keys_are_accepted(self):
        aug = Augmentrum(data=_small_niftis(), pipeline=['noise'], sigma_frac=(0.01, 0.05),
                         param_distribution='gaussian',
                         param_distributions={'sigma_frac': 'beta'})

        assert aug is not None

    def test_kwargs_for_a_user_pipeline_are_checked_against_its_steps(self):
        from augmentrum.core.pipeline import AugmentationPipeline
        from augmentrum.augmentation import Noise

        pipe = AugmentationPipeline([Noise(sigma_frac=0.05)])
        Augmentrum(data=_small_niftis(), pipeline=pipe, sigma_frac=0.05)
        with pytest.raises(ValueError, match="lb_hz"):
            Augmentrum(data=_small_niftis(), pipeline=pipe, lb_hz=2.0)

    def test_unknown_module_name_suggests_a_registry_name(self):
        with pytest.raises(ValueError, match="Unknown module 'nois'.*Did you mean 'noise'"):
            Augmentrum(data=_small_niftis(), pipeline=['nois'])


#**************************************************************************************************#
#                                   Class TestAugmentrumStepKwargs                                 #
#**************************************************************************************************#
#                                                                                                  #
# A step's own kwargs reach that step alone and override the globals there.                        #
#                                                                                                  #
#**************************************************************************************************#
class TestAugmentrumStepKwargs:
    """
    One global kwarg was routed to every module naming it, so an 'all'
    pipeline could not set LineBroadening and Apodization apart.
    """

    def test_dict_and_tuple_entries_build_the_named_module(self):
        from augmentrum.augmentation import Apodization, LineBroadening, Noise

        aug = Augmentrum(data=_small_niftis(), pipeline=[
            {'noise': {'sigma_frac': 0.01}},
            ('line_broadening', {'lb_hz': 2.0, 'mode': 'lorentzian'}),
            ('apodization', {'mode': 'exponential', 'lb_hz': 1.0}),
        ], backend='numpy')
        noise, broadening, apod = aug.pipelines['train'].steps

        assert isinstance(noise, Noise) and noise.sigma_frac == 0.01
        assert isinstance(broadening, LineBroadening) and broadening.lb_hz == 2.0
        assert broadening.mode == 'lorentzian'
        assert isinstance(apod, Apodization) and apod.lb_hz == 1.0

    def test_step_range_is_sampled_for_that_step_only(self):
        aug = Augmentrum(data=_small_niftis(), pipeline=[
            'line_broadening',
            {'apodization': {'mode': 'exponential', 'lb_hz': (1.0, 2.0)}},
        ], backend='numpy', seed=0)
        pipeline = aug.pipelines['train']

        assert pipeline.step_kwargs == [{}, {'mode': 'exponential', 'lb_hz': (1.0, 2.0)}]
        assert 0 not in pipeline.module_params, "the step range leaked to LineBroadening"
        params = pipeline.sample_batch_parameters(4)
        assert 1.0 <= params[1]['lb_hz'] <= 2.0

    def test_step_kwargs_override_the_global(self):
        aug = Augmentrum(data=_small_niftis(), pipeline=[
            'line_broadening',
            {'apodization': {'mode': 'exponential', 'lb_hz': 1.0}},
        ], lb_hz=(0.0, 5.0), backend='numpy', seed=0)
        params = aug.pipelines['train'].sample_batch_parameters(2)

        # lb_hz is a per-sample parameter, so the global range gives a vector
        assert np.all((0.0 <= params[0]['lb_hz']) & (params[0]['lb_hz'] <= 5.0)), \
            "the global range should still reach step 0"
        assert params[1]['lb_hz'] == 1.0, "the step scalar should win over the global range"

    def test_global_reaching_two_steps_warns_once_naming_them(self):
        with pytest.warns(UserWarning, match="'lb_hz' -> LineBroadening \\(step 0\\) and "
                                             "Apodization \\(step 1\\)") as record:
            Augmentrum(data=_small_niftis(), pipeline=['line_broadening', 'apodization'],
                       lb_hz=(0.0, 5.0), backend='numpy')

        assert sum('reaches more than one step' in str(w.message) for w in record) == 1

    def test_unknown_step_kwarg_raises(self):
        with pytest.raises(ValueError,
                           match="'noise' \\(Noise\\) does not accept \\['sigmafrac'\\]"):
            Augmentrum(data=_small_niftis(), pipeline=[{'noise': {'sigmafrac': 0.05}}])

    def test_module_instance_with_step_kwargs(self):
        from augmentrum.augmentation import Noise

        entry = (Noise(sigma_frac=0.1), {'sigma_frac': (0.0, 0.2)})
        aug = Augmentrum(data=_small_niftis(), pipeline=[entry], backend='numpy', seed=0)
        params = aug.pipelines['train'].sample_batch_parameters(2)

        assert np.all((0.0 <= params[0]['sigma_frac']) & (params[0]['sigma_frac'] <= 0.2))

    def test_malformed_entries_raise(self):
        with pytest.raises(ValueError, match="names one module"):
            Augmentrum(data=_small_niftis(), pipeline=[{'noise': {}, 'phase': {}}])
        with pytest.raises(ValueError, match="Pipeline entries are"):
            Augmentrum(data=_small_niftis(), pipeline=[42])

    def test_apodization_without_its_width_fails_at_build_time(self):
        with pytest.raises(ValueError, match="'apodization'.*needs 'lb_hz'"):
            Augmentrum(data=_small_niftis(), pipeline=['apodization'])

        # ... and is satisfied by a global range, a step value, or auto_lb
        Augmentrum(data=_small_niftis(), pipeline=['apodization'], lb_hz=(0.0, 5.0))
        Augmentrum(data=_small_niftis(), pipeline=[{'apodization': {'lb_hz': 2.0}}])
        Augmentrum(data=_small_niftis(), pipeline=[{'apodization': {'auto_lb': True,
                                                                     'target_pts': 64}}])

    def test_ranged_noise_level_drops_the_default_level(self):
        """A ranged snr_db used to sit beside the sigma_frac default, which won at run time."""
        aug = Augmentrum(data=_small_niftis(), pipeline=['noise'], snr_db=(10, 30), backend='numpy')
        noise = aug.pipelines['train'].steps[0]

        assert noise.sigma_frac is None
        assert noise.snr_db == 10


#**************************************************************************************************#
#                                  Class TestAugmentrumSourceUntouched                             #
#**************************************************************************************************#
#                                                                                                  #
# The pool of subjects a dataloader draws from must never be written back into.                    #
#                                                                                                  #
#**************************************************************************************************#
class TestAugmentrumSourceUntouched:
    """
    On-the-fly batches aliased the pool's NIFTI_MRS objects and the tensor
    path materialized its result into them, so noise accumulated batch after
    batch: x2 after the second draw of a subject, x4 after the fourth.
    """

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("volatile", [True, False])
    def test_source_unchanged_after_three_batches(self, backend, volatile):
        data = _small_niftis()
        before = [nifti[:].copy() for nifti in data]

        aug = Augmentrum(data=data, pipeline=['noise'], sigma_frac=0.1, batch_size=4,
                         backend=backend, volatile=volatile, seed=0)
        loader = aug.dataloader(framework='numpy')
        for _ in range(3):
            next(loader)

        pool = aug.splits['train'][0]
        for i, original in enumerate(before):
            assert np.array_equal(pool[i][:], original), f"subject {i} was modified in place"
            assert np.array_equal(data[i][:], original), f"input object {i} was modified"

    def test_batches_do_not_accumulate(self):
        """Every batch of one subject differs from the clean FID by the same noise level."""
        data = _small_niftis(1)
        clean = data[0][:]
        aug = Augmentrum(data=data, pipeline=['noise'], sigma=0.1, batch_size=1,
                         backend='numpy', seed=0)
        loader = aug.dataloader(framework='numpy')
        levels = [np.std(np.asarray(next(loader)[0])[0] - clean) for _ in range(4)]

        assert max(levels) < 2 * min(levels), f"noise grew batch by batch: {levels}"


#**************************************************************************************************#
#                                   Class TestAugmentrumGroupSplits                                #
#**************************************************************************************************#
#                                                                                                  #
# Items of one group always land in one split, whether the ids are given or read from headers.     #
#**************************************************************************************************#
class TestAugmentrumGroupSplits:
    """split_fractions permuted items, so the nine scans of a subject leaked across splits."""

    IDS = ['s1', 's1', 's1', 's2', 's2', 's2', 's3', 's3', 's3']

    def _assert_grouped(self, aug):
        sizes = {name: len(split[0]) for name, split in aug.splits.items()}
        assert sizes == {'train': 3, 'val': 3, 'test': 3}, sizes
        assert sorted(sum(aug.split_groups.values(), [])) == ['s1', 's2', 's3']
        for name, ids in aug.split_groups.items():
            assert len(ids) == 1, f"{name} holds {ids}"
            for nifti in aug.splits[name][0].list():
                assert Augmentrum._header_group(nifti) in (None, *ids)

    def test_explicit_groups_keep_a_subject_together(self):
        aug = Augmentrum(data=_small_niftis(9), groups=self.IDS, backend='numpy', pipeline=[],
                         split_fractions={'val': 0.2, 'test': 0.2}, seed=0)
        self._assert_grouped(aug)

    def test_subject_id_header_is_read_when_every_item_has_one(self):
        aug = Augmentrum(data=_small_niftis(9, subject_ids=self.IDS), backend='numpy',
                         pipeline=[], split_fractions={'val': 0.2, 'test': 0.2}, seed=0)
        assert aug.groups == self.IDS
        self._assert_grouped(aug)

    def test_partial_headers_fall_back_to_item_splitting(self):
        data = _small_niftis(9, subject_ids=self.IDS)
        data[4].remove_hdr_field('SubjectID')
        aug = Augmentrum(data=data, backend='numpy', pipeline=[],
                         split_fractions={'val': 0.2, 'test': 0.2}, seed=0)

        assert aug.groups is None and aug.split_groups is None
        sizes = {name: len(split[0]) for name, split in aug.splits.items()}
        assert sizes == {'train': 7, 'val': 1, 'test': 1}

    def test_group_split_is_deterministic_per_seed(self):
        one = Augmentrum(data=_small_niftis(9), groups=self.IDS, backend='numpy', pipeline=[],
                         split_fractions={'val': 0.2, 'test': 0.2}, seed=4).split_groups
        two = Augmentrum(data=_small_niftis(9), groups=self.IDS, backend='numpy', pipeline=[],
                         split_fractions={'val': 0.2, 'test': 0.2}, seed=4).split_groups
        assert one == two

    def test_groups_length_is_checked(self):
        with pytest.raises(ValueError, match="one group id per item"):
            Augmentrum(data=_small_niftis(9), groups=['a', 'b'], backend='numpy', pipeline=[])

    def test_split_groups_reported_without_splitting(self):
        aug = Augmentrum(data=_small_niftis(9), groups=self.IDS, backend='numpy', pipeline=[])
        assert aug.split_groups == {'train': ['s1', 's2', 's3'], 'val': [], 'test': []}

    def test_explicit_indices_warn_when_a_group_straddles_splits(self):
        with pytest.warns(RuntimeWarning, match="one group in several splits"):
            aug = Augmentrum(data=_small_niftis(9), groups=self.IDS, backend='numpy', pipeline=[],
                             split_indices={'train': range(0, 5), 'val': range(5, 9)})
        assert aug.split_groups == {'train': ['s1', 's2'], 'val': ['s2', 's3']}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
