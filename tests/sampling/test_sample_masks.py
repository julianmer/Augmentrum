####################################################################################################
#                                     test_sample_masks.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: Holds per-sample drawing to what it promises: every sample its own count within the     #
#          inclusive range and its own subset, kept as a mask the tensor never loses shape for,    #
#          reproducible under a seed, and carried through a pipeline to whatever consumes it.      #
#                                                                                                  #
####################################################################################################

"""
Tests for samplers drawing with per_sample=True.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from nifti_mrs_plus import Backend

from augmentrum import Augmentrum
from augmentrum.core.dataset_utils import create_random_generator
from augmentrum.core.pipeline import AugmentationPipeline
from augmentrum.core.pool import masks_of
from augmentrum.sampling import AverageSampler, CoilSampler

torch = pytest.importorskip('torch')

TAGS = ['DIM_COIL', 'DIM_DYN', None]


#**************#
#   fixtures   #
#**************#
@pytest.fixture
def batch():
    """A (16, 1, 1, 1, 64, 8, 10) raw batch, coils and transients behind the points."""
    generator = torch.Generator().manual_seed(0)
    shape = (16, 1, 1, 1, 64, 8, 10)
    return torch.complex(torch.randn(shape, generator=generator),
                         torch.randn(shape, generator=generator))


def _raw_niftis(n=3, coils=4, transients=6, points=64):
    """NIfTI-MRS objects with coils and transients, and one-transient waters."""
    rng = np.random.default_rng(0)
    mets, wats = [], []
    for _ in range(n):
        met = gen_nifti_mrs(rng.standard_normal((1, 1, 1, points, coils, transients)) + 0j,
                            1 / 2000, 123.0)
        met.set_dim_tag(4, 'DIM_COIL')
        met.set_dim_tag(5, 'DIM_DYN')
        wat = gen_nifti_mrs(rng.standard_normal((1, 1, 1, points, coils)) + 0j, 1 / 2000, 123.0)
        wat.set_dim_tag(4, 'DIM_COIL')
        mets.append(met)
        wats.append(wat)
    return mets, wats


#************#
#   counts   #
#************#
def test_every_sample_draws_its_own_count(batch):
    sampler = AverageSampler(n_averages=(2, 9), per_sample=True, seed=0)
    counts = set()
    for _ in range(20):
        out, _ = sampler.process_tensor(batch, dim_tags=TAGS)
        mask = sampler.dim_masks_['DIM_DYN']
        assert out is batch, "a mask leaves the tensor as it was"
        assert mask.shape == (16, 10) and mask.dtype == torch.bool
        counts.update(mask.sum(dim=1).tolist())
    assert min(counts) == 2 and max(counts) == 9, "the range is inclusive at both ends"
    assert len(counts) == 8


def test_a_count_array_is_taken_sample_by_sample(batch):
    wanted = np.arange(1, 17) % 8 + 1
    sampler = CoilSampler(n_coils=wanted, per_sample=True, seed=0)
    sampler.process_tensor(batch, dim_tags=TAGS)
    assert sampler.dim_masks_['DIM_COIL'].sum(dim=1).tolist() == wanted.tolist()


def test_the_pipeline_draws_one_count_per_sample(batch):
    pipeline = AugmentationPipeline([CoilSampler(per_sample=True)], user_kwargs={'n_coils': (1, 8)},
                                    seed=0)
    counts = pipeline.sample_batch_parameters(16)[0]['n_coils']
    assert counts.shape == (16,) and counts.min() >= 1 and counts.max() <= 8
    assert CoilSampler().PER_SAMPLE_PARAMS == (), "the batch-wide gather keeps one count"


def test_subsets_are_uniform_and_differ_between_samples(batch):
    sampler = CoilSampler(n_coils=3, per_sample=True, seed=0)
    hits = torch.zeros(8)
    subsets = set()
    for _ in range(50):
        sampler.process_tensor(batch, dim_tags=TAGS)
        mask = sampler.dim_masks_['DIM_COIL']
        hits += mask.sum(dim=0)
        subsets.update(tuple(row.tolist()) for row in mask)
    assert len(subsets) > 20
    assert hits.min() > 0.7 * hits.mean(), "every coil is drawn about as often"


def test_a_seed_replays_and_draws_still_vary(batch):
    first = AverageSampler(n_averages=(1, None), per_sample=True, seed=7)
    again = AverageSampler(n_averages=(1, None), per_sample=True, seed=7)
    masks = []
    for sampler in (first, again):
        sampler.process_tensor(batch, dim_tags=TAGS)
        masks.append(sampler.dim_masks_['DIM_DYN'])
    assert torch.equal(masks[0], masks[1])
    first.process_tensor(batch, dim_tags=TAGS)
    assert not torch.equal(first.dim_masks_['DIM_DYN'], masks[0])


def test_everything_kept_needs_no_mask(batch):
    sampler = AverageSampler(n_averages=10, per_sample=True, seed=0)
    sampler.process_tensor(batch, dim_tags=TAGS)
    assert 'DIM_DYN' not in sampler.dim_masks_
    sampler = AverageSampler(mode='deterministic', per_sample=True)
    sampler.process_tensor(batch, dim_tags=TAGS)
    assert sampler.dim_masks_ == {}


#*************#
#   windows   #
#*************#
@pytest.mark.parametrize('scheme, stride', [('consecutive', 1), ('strided', 2)])
def test_windows_per_sample(batch, scheme, stride):
    sampler = AverageSampler(n_averages=(2, 4), scheme=scheme, stride=stride, per_sample=True,
                             seed=0)
    sampler.process_tensor(batch, dim_tags=TAGS)
    for row in sampler.dim_masks_['DIM_DYN']:
        kept = torch.nonzero(row)[:, 0]
        assert 2 <= len(kept) <= 4
        assert torch.all(torch.diff(kept) == stride)


#****************#
#   refinement   #
#****************#
def test_a_second_draw_refines_the_first(batch):
    wide = CoilSampler(n_coils=6, per_sample=True, seed=0)
    wide.process_tensor(batch, dim_tags=TAGS)
    narrow = CoilSampler(n_coils=(1, 8), per_sample=True, seed=1)
    narrow.process_tensor(batch, dim_tags=TAGS, dim_masks=wide.dim_masks_)
    first, second = wide.dim_masks_['DIM_COIL'], narrow.dim_masks_['DIM_COIL']
    assert not torch.any(second & ~first)
    assert torch.all(second.sum(dim=1) <= 6)


def test_a_gather_carries_the_mask_along(batch):
    masker = AverageSampler(n_averages=5, per_sample=True, seed=0)
    masker.process_tensor(batch, dim_tags=TAGS)
    gather = AverageSampler(n_averages=4, seed=0)
    out, _ = gather.process_tensor(batch, dim_tags=TAGS, dim_masks=masker.dim_masks_)
    assert out.shape[-1] == 4
    assert gather.dim_masks_['DIM_DYN'].shape == (16, 4)


def test_numpy_data_gets_numpy_masks(batch):
    sampler = CoilSampler(n_coils=(1, 4), per_sample=True, seed=0)
    out, _ = sampler.process_tensor(batch.numpy(), dim_tags=TAGS)
    mask = sampler.dim_masks_['DIM_COIL']
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert mask.shape == (16, 8) and 1 <= mask.sum(axis=1).min() <= mask.sum(axis=1).max() <= 4


#******************#
#   list backend   #
#******************#
def test_list_path_cuts_each_subject_to_its_own_count():
    mets, wats = _raw_niftis()
    sampler = CoilSampler(n_coils=np.array([1, 3, 2]), per_sample=True, seed=0)
    out, water = sampler.process_nifti_list(mets, wats)
    assert [n.shape[4] if len(n.shape) > 4 else 1 for n in out] == [1, 3, 2]
    assert [w.shape[4] if len(w.shape) > 4 else 1 for w in water] == [1, 3, 2]


#**************#
#   pipeline   #
#**************#
def test_masks_travel_to_the_consumer_and_leftovers_become_zeros():
    mets, wats = _raw_niftis()
    aug = Augmentrum(mets, wats, pipeline=[{'coil_sampling': {'per_sample': True}},
                                           {'average_sampling': {'per_sample': True}}],
                     n_coils=(1, 2), n_averages=(1, 3), backend='pytorch', batch_size=3,
                     volatile=True, seed=0)
    pool, water = aug.splits['train']
    batches = create_random_generator(pool, water, aug.pipelines['train'], 3,
                                      pool=aug._pool('train', pool, water))
    data, _ = next(batches)
    tensor = data.get_data(Backend.PYTORCH)
    masks = masks_of(data)
    assert set(masks) == {'DIM_COIL', 'DIM_DYN'}
    kept = tensor[:, 0, 0, 0, 0].abs() > 0                                  # (B, C, D)
    assert torch.equal(kept.any(dim=2), masks['DIM_COIL'])
    assert torch.equal(kept.any(dim=1), masks['DIM_DYN'])


def test_a_module_that_cannot_honour_masks_refuses_them():
    mets, wats = _raw_niftis()
    aug = Augmentrum(mets, wats, pipeline=[{'coil_sampling': {'per_sample': True}}, 'noise'],
                     sigma_frac=0.1, backend='pytorch', batch_size=2, volatile=True, seed=0)
    with pytest.raises(ValueError, match='per-sample masks'):
        next(aug.dataloader())


def test_elementwise_modules_let_masks_pass():
    mets, wats = _raw_niftis()
    aug = Augmentrum(mets, wats, pipeline=[{'average_sampling': {'per_sample': True}},
                                           'line_broadening', 'processing'],
                     lb_hz=2.0, registration_method='torch', backend='pytorch', batch_size=2,
                     volatile=True, seed=0)
    data, water = next(aug.dataloader())
    assert data.shape == (2, 1, 1, 1, 64)


def test_per_sample_draws_replay_under_a_seed():
    mets, wats = _raw_niftis()

    def first_batches(seed):
        aug = Augmentrum(mets, wats, pipeline=[{'coil_sampling': {'per_sample': True}},
                                               {'average_sampling': {'per_sample': True}},
                                               'processing'],
                         n_coils=(1, 4), n_averages=(1, 6), registration_method='torch',
                         backend='pytorch', batch_size=3, volatile=True, seed=seed)
        loader = aug.dataloader()
        return [next(loader)[0] for _ in range(2)]

    one, two, other = first_batches(3), first_batches(3), first_batches(4)
    assert all(torch.equal(a, b) for a, b in zip(one, two))
    assert not torch.equal(one[0], other[0])
    assert not torch.equal(one[0], one[1])
