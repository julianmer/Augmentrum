####################################################################################################
#                                         test_pool.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: Holds the stacked subject pool to its promises: batches drawn by indexing equal the     #
#          subjects, nothing ever writes into the pool or its objects, a reused buffer never       #
#          overwrites a batch someone kept, and seeded runs replay.                                #
#                                                                                                  #
####################################################################################################

"""
Tests for TensorPool, PooledBatch and the dataloaders drawing from them.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from nifti_mrs_plus import Backend, NIfTI_MRS_Plus

from augmentrum import Augmentrum
from augmentrum.core.dataset_utils import convert_batch_to_backend
from augmentrum.core.pool import PooledBatch, TensorPool, origin_of

torch = pytest.importorskip('torch')

CUDA = torch.cuda.is_available()


#*************#
#   helpers   #
#*************#
def _raw_niftis(n=4, coils=3, transients=5, points=64, seed=0):
    """Raw NIfTI-MRS subjects (coils, transients) and one-transient waters."""
    rng = np.random.default_rng(seed)
    mets, wats = [], []
    for _ in range(n):
        values = rng.standard_normal((1, 1, 1, points, coils, transients)) \
            + 1j * rng.standard_normal((1, 1, 1, points, coils, transients))
        met = gen_nifti_mrs(values.astype(np.complex64), 1 / 2000, 123.0)
        met.set_dim_tag(4, 'DIM_COIL')
        met.set_dim_tag(5, 'DIM_DYN')
        wat = gen_nifti_mrs(values[..., 0].astype(np.complex64), 1 / 2000, 123.0)
        wat.set_dim_tag(4, 'DIM_COIL')
        mets.append(met)
        wats.append(wat)
    return mets, wats


def _pool(backend=Backend.PYTORCH, volatile=True, device=None):
    mets, wats = _raw_niftis()
    data = NIfTI_MRS_Plus(mets, backend=backend, volatile=volatile)
    water = NIfTI_MRS_Plus(wats, backend=backend, volatile=volatile)
    return mets, wats, TensorPool.build(data, water, backend, device)


#***************#
#   admission   #
#***************#
def test_tensor_backends_are_pooled_and_the_list_backend_is_not():
    mets, wats = _raw_niftis()
    for backend in (Backend.PYTORCH, Backend.NUMPY):
        data = NIfTI_MRS_Plus(mets, backend=backend)
        assert TensorPool.build(data, None, backend) is not None
    data = NIfTI_MRS_Plus(mets, backend=Backend.NIFTI_LIST)
    assert TensorPool.build(data, None, Backend.NIFTI_LIST) is None


def test_subjects_of_differing_shapes_are_not_pooled():
    mets, _ = _raw_niftis(2)
    other, _ = _raw_niftis(1, transients=4)
    data = NIfTI_MRS_Plus(mets + other, backend=Backend.PYTORCH)
    assert TensorPool.build(data, None, Backend.PYTORCH) is None


#*************#
#   drawing   #
#*************#
def test_a_batch_is_the_subjects_it_names():
    mets, wats, pool = _pool()
    data, water = pool.batch([2, 0, 2])
    assert isinstance(data, PooledBatch) and isinstance(water, PooledBatch)
    for row, subject in enumerate([2, 0, 2]):
        assert np.array_equal(data.get_data(Backend.PYTORCH)[row].numpy(), mets[subject][:])
        assert np.array_equal(water.get_data(Backend.PYTORCH)[row].numpy(), wats[subject][:])
    assert data.dim_tags == mets[0].dim_tags
    assert origin_of(data).indices.tolist() == [2, 0, 2]


def test_materializing_never_writes_into_the_pool_objects():
    mets, _, pool = _pool()
    before = [m[:].copy() for m in mets]
    data, _ = pool.batch([1, 1])
    data.set_data(data.get_data(Backend.PYTORCH) * 3, Backend.PYTORCH)
    objects = data.list()
    assert all(o is not m for o in objects for m in mets)
    assert objects[0] is not objects[1], "a subject drawn twice gets two objects"
    assert np.allclose(objects[0][:], 3 * before[1])
    assert all(np.array_equal(m[:], b) for m, b in zip(mets, before))


def test_headers_are_taken_over_before_they_change():
    mets, _, pool = _pool(volatile=False)
    tags = list(mets[0].dim_tags)
    data, _ = pool.batch([0])
    data.update_metadata('Scaled', {'factor': 3})
    data.set_dim_tag(5, 'DIM_EDIT')
    assert mets[0].dim_tags == tags
    assert 'ProcessingApplied' not in mets[0].hdr_ext \
        or all(e['Method'] != 'Scaled' for e in mets[0].hdr_ext['ProcessingApplied'])
    assert data.list()[0].hdr_ext['ProcessingApplied'][-1]['Method'] == 'Scaled'


def test_the_batch_tensor_is_handed_over_as_it_is():
    _, _, pool = _pool()
    data, water = pool.batch([0, 1])
    tensor, water_tensor = convert_batch_to_backend(data, water, Backend.PYTORCH)
    assert tensor.data_ptr() == data.get_data(Backend.PYTORCH).data_ptr()
    array, _ = convert_batch_to_backend(data, water, Backend.NUMPY)
    assert isinstance(array, np.ndarray) and array.shape == tuple(tensor.shape)


def test_the_cache_is_the_pools_own():
    _, _, pool = _pool()
    calls = []
    first = pool.cached('key', lambda p: calls.append(p) or len(calls))
    assert pool.cached('key', lambda p: calls.append(p) or len(calls)) == first == 1
    assert calls == [pool]


#****************#
#   the buffer   #
#****************#
@pytest.mark.parametrize('backend', ['pytorch', 'numpy'])
def test_a_kept_batch_is_never_overwritten(backend):
    mets, wats = _raw_niftis()
    aug = Augmentrum(mets, wats, pipeline=['tap'], backend=backend, batch_size=2,
                     volatile=True, seed=0)
    loader = aug.dataloader(framework='numpy')
    kept = [next(loader)[0] for _ in range(4)]
    frozen = [np.array(k, copy=True) for k in kept]
    for _ in range(4):
        next(loader)
    assert all(np.array_equal(np.asarray(k), f) for k, f in zip(kept, frozen))
    pool = np.asarray(aug._pool('train', *aug.splits['train']).data)
    assert all(any(np.array_equal(row, subject) for subject in pool)
               for k in kept for row in np.asarray(k))


def test_a_released_buffer_is_reused():
    mets, wats = _raw_niftis()
    aug = Augmentrum(mets, wats, pipeline=['line_broadening'], lb_hz=1.0, backend='pytorch',
                     batch_size=2, volatile=True, seed=0)
    loader = aug.dataloader()
    buffers = set()
    for _ in range(4):
        next(loader)
        buffers.add(id(aug._pool('train', *aug.splits['train'])._buffers['data']))
    assert len(buffers) == 1


#*****************#
#   dataloaders   #
#*****************#
@pytest.mark.parametrize('backend', ['pytorch', 'numpy'])
@pytest.mark.parametrize('volatile', [True, False])
def test_the_source_stays_untouched(backend, volatile):
    mets, wats = _raw_niftis()
    before = [m[:].copy() for m in mets]
    aug = Augmentrum(mets, wats, pipeline=['noise'], sigma_frac=0.1, batch_size=3,
                     backend=backend, volatile=volatile, seed=0)
    loader = aug.dataloader(framework='python')
    for _ in range(3):
        next(loader)
    assert all(np.array_equal(m[:], b) for m, b in zip(mets, before))


def test_pooled_batches_equal_copied_batches():
    """The pool changes where a batch comes from, not what it is."""
    mets, wats = _raw_niftis()

    def batches(pooled):
        aug = Augmentrum(mets, wats, pipeline=['noise', 'line_broadening'], sigma_frac=0.1,
                         lb_hz=(0.0, 3.0), batch_size=3, backend='pytorch', volatile=True,
                         seed=5)
        if not pooled:
            aug._pool = lambda *args: None
        loader = aug.dataloader()
        return [next(loader)[0] for _ in range(3)]

    for a, b in zip(batches(True), batches(False)):
        assert torch.equal(a, b)


def test_a_replaced_split_gets_a_new_pool():
    mets, wats = _raw_niftis()
    aug = Augmentrum(mets, wats, pipeline=[], backend='pytorch', batch_size=2, volatile=True)
    first = aug._pool('train', *aug.splits['train'])
    assert aug._pool('train', *aug.splits['train']) is first
    others, other_waters = _raw_niftis(seed=1)
    aug.splits['train'] = (NIfTI_MRS_Plus(others, backend=Backend.PYTORCH, volatile=True),
                           NIfTI_MRS_Plus(other_waters, backend=Backend.PYTORCH, volatile=True))
    second = aug._pool('train', *aug.splits['train'])
    assert second is not first
    assert np.array_equal(second.data[0].numpy(), others[0][:])
    assert aug.refresh_pools()._pools == {}


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_batches_live_on_the_device():
    mets, wats = _raw_niftis()
    aug = Augmentrum(mets, wats, pipeline=[{'coil_sampling': {'per_sample': True}},
                                           'processing'],
                     registration_method='torch', backend='pytorch', device='cuda',
                     batch_size=2, volatile=True, seed=0)
    data, water = next(aug.dataloader())
    assert data.device.type == 'cuda' and water.device.type == 'cuda'
    assert next(iter(aug.as_torch_dataloader())).device.type == 'cuda'
