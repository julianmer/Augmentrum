####################################################################################################
#                                   test_dimension_sampling.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds the dimension sampler to what a shorter scan looks like, through its average       #
#          variant: fewer repeats along DIM_DYN, nothing else touched, native on any backend.       #
#                                                                                                  #
####################################################################################################

"""
Tests for drawing a subset of the averages.

Averages are their own acquisition axis, so the thing worth guarding is that
this sampler acts on DIM_DYN and on nothing else - including when a coil axis
is sitting next to it.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs_plus import ops

from augmentrum.core import Backend
from augmentrum.sampling import AverageSampler, CoilSampler

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


#**************#
#   fixtures   #
#**************#

@pytest.fixture
def batch():
    """A batch carrying both a coil and an average axis."""
    rng = np.random.default_rng(0)
    shape = (2, 4, 4, 2, 8, 6, 10)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


TAGS = ['DIM_COIL', 'DIM_DYN', None]


#**************#
#   drawing    #
#**************#

def test_averages_are_drawn(batch):
    """Fewer repeats is the whole point."""
    drawn, _ = AverageSampler(n_averages=4, seed=0).process_tensor(batch, dim_tags=TAGS)

    assert drawn.shape[-1] == 4


def test_only_the_average_axis_is_touched(batch):
    """Coils belong to their own sampler; this one must leave them alone."""
    drawn, _ = AverageSampler(n_averages=4, seed=0).process_tensor(batch, dim_tags=TAGS)

    assert drawn.shape[:-1] == batch.shape[:-1], "something other than DIM_DYN moved"


def test_an_exact_count_is_honored(batch):
    """A bare number means exactly that many, not up to that many."""
    for wanted in (1, 3, 10):
        drawn, _ = AverageSampler(n_averages=wanted, seed=0).process_tensor(
            batch, dim_tags=TAGS)
        assert drawn.shape[-1] == wanted


def test_a_range_stays_inside_it(batch):
    """A range draws within its bounds, every time."""
    counts = {AverageSampler(n_averages=(2, 5), seed=s).process_tensor(
        batch, dim_tags=TAGS)[0].shape[-1] for s in range(12)}

    assert counts, "nothing was drawn at all"
    assert min(counts) >= 2 and max(counts) <= 5, counts


def test_asking_for_more_than_exists_is_clamped(batch):
    """A protocol asking for 50 averages of 10 gets the 10 that are there."""
    drawn, _ = AverageSampler(n_averages=(50, None), seed=0).process_tensor(
        batch, dim_tags=TAGS)

    assert drawn.shape[-1] == batch.shape[-1]


def test_deterministic_keeps_everything(batch):
    """Nothing is discarded unless a draw asks for it."""
    kept, _ = AverageSampler(mode='deterministic').process_tensor(batch, dim_tags=TAGS)

    assert np.array_equal(kept, batch)


def test_data_without_averages_is_left_alone(batch):
    """Nothing named DIM_DYN means nothing to draw from."""
    untagged, _ = AverageSampler(n_averages=2, seed=0).process_tensor(
        batch, dim_tags=['DIM_COIL', None, None])

    assert np.array_equal(untagged, batch)


def test_an_unknown_mode_is_refused():
    """A typo should not silently become a mode that does nothing."""
    with pytest.raises(ValueError, match="mode must be"):
        AverageSampler(mode='randomised')


def test_a_seed_replays_exactly(batch):
    """Reproducible on demand, varying without a seed."""
    first, _ = AverageSampler(n_averages=(1, 9), seed=7).process_tensor(batch, dim_tags=TAGS)
    again, _ = AverageSampler(n_averages=(1, 9), seed=7).process_tensor(batch, dim_tags=TAGS)

    assert np.array_equal(first, again)


#**************#
#   backends   #
#**************#

def test_every_backend_stays_native(batch):
    """A draw is a gather, so it never needs to leave the backend it met."""
    reference = None

    for backend in Backend:
        if backend is Backend.NIFTI_LIST:
            continue
        try:
            native = ops.match_backend(batch, _probe(backend))
        except (ImportError, RuntimeError):
            pytest.skip(f"{backend.value} not installed")

        drawn, _ = AverageSampler(n_averages=4, seed=0).process_tensor(
            native, dim_tags=TAGS)

        assert type(drawn).__module__.split('.')[0] == \
               type(native).__module__.split('.')[0], f"{backend.value} left its backend"

        result = ops.to_numpy(drawn)
        if reference is None:
            reference = result
        assert np.abs(result - reference).max() < 1e-5


def _probe(backend: Backend):
    """A one-element tensor on *backend*, to promote against."""
    if backend is Backend.NUMPY:
        return np.zeros(1, np.complex64)
    if backend is Backend.PYTORCH:
        import torch
        return torch.zeros(1, dtype=torch.complex64)
    if backend is Backend.JAX:
        import jax.numpy as jnp
        return jnp.zeros(1, jnp.complex64)
    if backend is Backend.TENSORFLOW:
        import tensorflow as tf
        return tf.zeros(1, tf.complex64)
    if backend is Backend.KERAS:
        import keras
        return keras.ops.zeros(1, 'complex64')
    raise RuntimeError(f"no probe for {backend}")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_gradients_survive_both_samplers(batch):
    """Coils then averages, and the graph reaches back through both."""
    tensor = torch.from_numpy(batch).requires_grad_(True)

    drawn, _ = CoilSampler(mode='random', n_coils=3, seed=0).process_tensor(
        tensor, dim_tags=TAGS)
    drawn, _ = AverageSampler(n_averages=4, seed=0).process_tensor(drawn, dim_tags=TAGS)

    assert tuple(drawn.shape[-2:]) == (3, 4)
    drawn.abs().sum().backward()
    assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0
