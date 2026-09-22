####################################################################################################
#                                            pool.py                                               #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: The subject pool a dataloader draws from, stacked once on the device; the batches       #
#          drawn from it, which borrow the pool's NIfTI headers and never write into them; and     #
#          the per-sample dimension masks that stand in for ragged coil and transient subsets.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nifti_mrs_plus import NIfTI_MRS_Plus, Backend, ops
from nifti_mrs_plus.core import get_provenance


__all__ = ['TensorPool', 'PooledBatch', 'PoolOrigin', 'masks_of', 'set_masks', 'origin_of',
           'set_origin', 'rewrap', 'apply_masks', 'water_masks', 'finalize_masks']


#********************************#
#   per-sample dimension masks   #
#********************************#
# A sampler that draws a different number of coils or transients for every
# sample cannot shrink a rectangular tensor, so it leaves the tensor whole and
# says which entries each sample kept: a boolean (batch, n) mask per dimension
# tag, carried on the NIfTI_MRS_Plus batch next to its tensor. Modules that
# aggregate over the dimension consume the masks; modules acting on every
# element alike let them pass; at the end of a pipeline the masked entries are
# zeroed, so an unconsumed mask never reads as data that was drawn.

#: Attribute of a NIfTI_MRS_Plus batch holding its masks, {tag: (batch, n) bool}.
MASKS_ATTR = 'dim_masks'

#: Attribute of a NIfTI_MRS_Plus batch naming the pool entries its values still equal.
ORIGIN_ATTR = 'pool_origin'

#: Dimensions the water reference shares with the data: the receive elements.
WATER_SHARED = frozenset({'DIM_COIL'})


def masks_of(batch) -> Dict[str, Any]:
    """The per-sample masks a batch carries, as a fresh dict (empty for none)."""
    return dict(getattr(batch, MASKS_ATTR, None) or {})


def set_masks(batch, masks):
    """Attach *masks* to *batch* (no-op for None); returns the batch."""
    if batch is not None:
        setattr(batch, MASKS_ATTR, dict(masks or {}))
    return batch


def water_masks(masks) -> Dict[str, Any]:
    """The masks that apply to the water reference too: those of shared dimensions."""
    return {tag: mask for tag, mask in (masks or {}).items() if tag in WATER_SHARED}


def origin_of(batch) -> Optional['PoolOrigin']:
    """The pool entries a batch's values still equal, or None."""
    return getattr(batch, ORIGIN_ATTR, None) if batch is not None else None


def set_origin(batch, origin):
    """Record (or, with None, forget) where a batch's values came from; returns the batch."""
    if batch is not None:
        setattr(batch, ORIGIN_ATTR, origin)
    return batch


def _mask_factor(mask, like, shape):
    """A 0/1 factor of *like*'s backend and dtype, *mask* reshaped to *shape*."""
    if ops.is_torch(like):
        import torch
        mask = mask if ops.is_torch(mask) else torch.as_tensor(np.asarray(mask),
                                                               device=like.device)
        return mask.reshape(shape).to(device=like.device, dtype=like.dtype)
    return ops.match_backend(np.asarray(ops.to_numpy(mask), dtype=np.float64).reshape(shape),
                             like)


def apply_masks(array, masks, tags, first_axis=5):
    """
    Zero the entries of *array* that its masks exclude.

    Args:
        array: Batch tensor, dimension tag i at axis "first_axis + i".
        masks: {tag: (batch, n) bool}.
        tags: The array's higher-dimension tags.
        first_axis: Axis of the first tagged dimension (5 in the NIfTI layout,
            4 with the spectral axis moved last).

    Returns:
        The masked array; *array* itself when nothing applies.
    """
    tags = [t for t in (tags or []) if t]
    rank = len(ops.shape(array))
    for tag, mask in (masks or {}).items():
        if tag not in tags or first_axis + tags.index(tag) >= rank:
            continue
        shape = [1] * rank
        shape[0] = ops.shape(array)[0]
        shape[first_axis + tags.index(tag)] = ops.shape(array)[first_axis + tags.index(tag)]
        array = array * _mask_factor(mask, array, shape)
    return array


def finalize_masks(batch):
    """
    A batch whose unconsumed masks are applied, the masks kept for reference.

    The end of a pipeline hands the tensor to someone who cannot see the
    masks, so the entries they exclude become zeros there - a coil or
    transient that was not drawn never reads as one that was.
    """
    masks = masks_of(batch)
    if batch is None or not masks or batch.backend == Backend.NIFTI_LIST:
        return batch
    tensor = batch.get_data(batch.backend)
    masked = apply_masks(tensor, masks, batch.dim_tags)
    if masked is tensor:
        return batch
    out = rewrap(batch, batch.nifti_list, batch.backend, batch.volatile, batch.state)
    out.set_data(masked, batch.backend, dim_tags=batch.dim_tags)
    return set_masks(out, masks)


#**************************************************************************************************#
#                                        Class PooledBatch                                         #
#**************************************************************************************************#
#                                                                                                  #
# A batch drawn from a pool: its tensor is authoritative, the pool's NIfTI objects are borrowed.   #
#                                                                                                  #
#**************************************************************************************************#
class PooledBatch(NIfTI_MRS_Plus):
    """
    A batch drawn from a TensorPool: its tensor is authoritative, the pool's
    NIfTI objects are borrowed for their headers only.

    Those objects are the pool's own, so nothing may write into them - the
    reason the dataloader used to copy every subject of every batch. Instead
    the copy is deferred to the one moment it is needed: materializing
    rebuilds fresh objects from the tensor rather than writing in place, and
    provenance is recorded on the batch and written into the rebuilt objects.
    Every processing step wraps its output in the class of its input, so the
    guarantee holds along the whole pipeline.
    """

    def __init__(self, nifti_list, backend=None, volatile=False, metadata=None, state=None,
                 headers=None):
        # A pool is uniform by construction and its objects were checked when it
        # was built, so the base class's check - which parses a header, for every
        # wrapper of every step - is skipped: it starts as the empty batch. What
        # the headers say (the objects' shape, their frequencies, ...) is read
        # once per batch into *headers* and shared by every wrapper around the
        # same objects ("rewrap"), since borrowed headers never change.
        objects = list(nifti_list.nifti_list if isinstance(nifti_list, NIfTI_MRS_Plus)
                       else nifti_list)
        super().__init__([], backend=backend, volatile=True, state=state)
        self.volatile = volatile
        self.nifti_list = objects
        self.n_subjects = len(objects)
        self._headers = headers if headers is not None else {}
        if objects and 'object_shape' not in self._headers:
            self._headers['object_shape'] = tuple(objects[0].shape)
        self._shape = ((self.n_subjects,) + self._headers['object_shape'] if objects else (0,))
        if not volatile:
            self._init_metadata(objects, metadata)
        self._borrowed = True
        self._pending_provenance = []

    #***************#
    #   ownership   #
    #***************#
    def materialize(self):
        """
        Hand out values as NIfTI objects of this batch's own.

        A pending tensor becomes freshly built objects (the borrowed ones are
        never written into); without one the borrowed objects are copied.
        Recorded provenance lands in the new objects' headers.
        """
        if not self._borrowed:
            return super().materialize()
        self._headers = {}                  # the objects about to be built have headers of their own
        if self._tensor_dirty and self._cached_tensor is not None:
            arr = ops.to_numpy(self._cached_tensor)
            self.nifti_list = [self._rebuild(nifti, arr[i])
                               for i, nifti in enumerate(self.nifti_list)]
            self._tensor_dirty = False
        else:
            self.nifti_list = [nifti.copy() for nifti in self.nifti_list]
        self._borrowed = False
        for entry in self._pending_provenance:
            self._write_provenance(*entry)
        self._pending_provenance = []
        return self

    def _own(self):
        """Take ownership of the objects before touching their headers or values."""
        if self._borrowed:
            self.materialize()

    def update_metadata(self, operation, details, individual_idx=None):
        """Record provenance; the headers get it once the objects are this batch's own."""
        if self.volatile:
            return
        entry = (datetime.now().isoformat(), operation, str(details), individual_idx)
        if self._borrowed:
            self._pending_provenance.append(entry)
            self.metadata_common.setdefault('common_provenance', []).append(
                {'Timestamp': entry[0], 'Operation': operation, 'Details': entry[2]})
            return
        self._write_provenance(*entry)

    def _write_provenance(self, timestamp, operation, details, individual_idx):
        """The NIfTI-MRS "ProcessingApplied" entry of one recorded operation."""
        program = get_provenance()
        indices = range(len(self.nifti_list)) if individual_idx is None else individual_idx
        for idx in indices:
            nifti = self.nifti_list[idx]
            if getattr(nifti, 'hdr_ext', None) is None:
                continue
            applied = (nifti.hdr_ext['ProcessingApplied']
                       if 'ProcessingApplied' in nifti.hdr_ext else [])
            applied.append({'Time': timestamp, 'Program': program['program'],
                            'Version': program['version'], 'Method': operation,
                            'Details': details})
            nifti.add_hdr_field('ProcessingApplied', applied)

    def set_dim_tag(self, index, tag):
        self._own()
        return super().set_dim_tag(index, tag)

    def sync_headers(self, source_idx=0):
        self._own()
        return super().sync_headers(source_idx)

    def __setitem__(self, idx, values):
        if not (self._backend != Backend.NIFTI_LIST and isinstance(idx, slice)
                and idx == slice(None) and not isinstance(values, list)):
            self._own()
        return super().__setitem__(idx, values)

    def to(self, device):
        """The batch on *device*, still borrowing (and still tensor-authoritative)."""
        tensor = self.get_data(Backend.PYTORCH).to(device)
        out = rewrap(self, self.nifti_list, Backend.PYTORCH, self.volatile, self._state)
        out.set_data(tensor, Backend.PYTORCH, dim_tags=self._pending_tags)
        set_masks(out, masks_of(self))
        return out


def rewrap(source, nifti_list, backend, volatile, state):
    """
    A new batch around *nifti_list*, of *source*'s kind.

    A pooled batch stays pooled, carrying its unwritten provenance, so its
    objects stay protected however many steps wrap it. The new batch counts
    its objects as borrowed even where the source owned them: they are still
    the source's, and an earlier stage must not change under a later one.
    """
    if isinstance(source, PooledBatch):
        # the same objects carry the same headers, so what was read from them carries over
        same = nifti_list is source.nifti_list and source._borrowed
        out = PooledBatch(nifti_list=nifti_list, backend=backend, volatile=volatile,
                          state=state, headers=source._headers if same else None)
        out._pending_provenance = list(source._pending_provenance)
        return out
    return NIfTI_MRS_Plus(nifti_list=nifti_list, backend=backend, volatile=volatile,
                          state=state)


#**************************************************************************************************#
#                                        Class PoolOrigin                                          #
#**************************************************************************************************#
#                                                                                                  #
# Which pool entries a batch's values still equal.                                                 #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class PoolOrigin:
    """
    Which pool entries a batch's values still equal, exactly.

    Set when a batch is drawn and kept only through steps that leave values
    untouched (samplers that mask, taps), so a consumer may take per-subject
    results the pool cached rather than recompute them from the batch.

    Attributes:
        pool: The TensorPool the batch was drawn from.
        indices: Pool index of every sample, a (batch,) tensor on the pool's backend.
        role: 'data' or 'water'.
    """
    pool: 'TensorPool'
    indices: Any
    role: str


#**************************************************************************************************#
#                                         Class TensorPool                                         #
#**************************************************************************************************#
#                                                                                                  #
# The subject pool stacked once on the device, drawn from by indexing.                             #
#                                                                                                  #
#**************************************************************************************************#
class TensorPool:
    """
    The subject pool stacked once on the device, drawn from by indexing.

    Drawing a batch is one gather on the device: no NIfTI object is copied or
    read, and the gather writes into memory of its own, so the pool can never
    be written into through a batch. The NIfTI objects stay the source of
    metadata. On CPU, where every fresh allocation of a raw batch pays its page
    faults, the gather reuses the previous batch's memory - but only when
    nothing refers to it any more: no name, view, array or yielded tensor.

    The pool is a snapshot: it is keyed by the identity of the subject
    objects, so a split that is replaced gets a new pool, but values written
    into the same objects are not seen - "Augmentrum.refresh_pools" rebuilds.
    Results that depend on a subject alone can be cached on the pool
    ("cached"); they go with it.

    Args:
        data: NIfTI_MRS_Plus pool of subjects, uniform in shape and tags.
        water: Their water references, or None.
        backend: Tensor backend the pool lives on.
        device: Torch device (None: CPU); ignored on other backends.
    """

    #: Threads reading the subjects when the pool is stacked.
    READERS = 4

    def __init__(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus],
                 backend: Backend, device=None):
        self.source = (data, water)
        self.backend = backend
        self.device = device
        self.volatile = data.volatile
        self.key = self.fingerprint(data, water)
        self.data_tags = list(data.dim_tags)
        self.water_tags = list(water.dim_tags) if water is not None else None
        self.data = self._stack(data)
        self.water = self._stack(water) if water is not None else None
        self._cache: Dict[Any, Any] = {}
        self._buffers: Dict[str, Any] = {}

    #***************#
    #   admission   #
    #***************#
    @staticmethod
    def poolable(data, water, backend) -> bool:
        """
        Whether a split can be pooled: a tensor backend and subjects that stack.

        The NIfTI-list backend processes objects one by one, and subjects of
        differing shapes or tags cannot share a tensor; both keep the
        per-batch copies.
        """
        if backend == Backend.NIFTI_LIST or data is None or len(data) == 0:
            return False
        for group in (data, water):
            if group is None:
                continue
            if len(group) != len(data):
                return False
            first = group.nifti_list[0]
            if any(n.shape != first.shape or n.dim_tags != first.dim_tags
                   for n in group.nifti_list[1:]):
                return False
        return True

    @classmethod
    def build(cls, data, water, backend, device=None) -> Optional['TensorPool']:
        """A pool of the split, or None where it cannot be pooled."""
        if not cls.poolable(data, water, backend):
            return None
        return cls(data, water, backend, device)

    @staticmethod
    def fingerprint(data, water) -> Tuple:
        """Identity of the pooled containers and objects."""
        return (id(data), id(water), tuple(map(id, data.nifti_list)),
                tuple(map(id, water.nifti_list)) if water is not None else None)

    def matches(self, data, water) -> bool:
        """Whether this pool still stands for *data* and *water*."""
        return self.key == self.fingerprint(data, water) and self.source[0] is data

    def _stack(self, group):
        """
        The group's values as one tensor on the pool's backend and device.

        Stacked into C order: NIfTI arrays come in Fortran order, and a pool
        that kept it would make every gather stride across the whole array.
        Subjects are read on a few threads, since a file-backed object
        decompresses on every read and zlib lets go of the interpreter.
        """
        from concurrent.futures import ThreadPoolExecutor

        first = group.nifti_list[0][:]
        arr = np.empty((len(group),) + first.shape, dtype=first.dtype)
        arr[0] = first

        def read(i):
            arr[i] = group.nifti_list[i][:]

        with ThreadPoolExecutor(max_workers=self.READERS) as readers:
            list(readers.map(read, range(1, len(group))))
        if self.backend == Backend.PYTORCH:
            import torch
            tensor = torch.from_numpy(arr)
            return tensor if self.device is None else tensor.to(self.device)
        if self.backend == Backend.NUMPY:
            return arr
        return NIfTI_MRS_Plus._convert(arr, self.backend)

    #*************#
    #   drawing   #
    #*************#
    def _indices(self, indices):
        """Subject indices as an index array of the pool's backend."""
        if self.backend == Backend.PYTORCH:
            from augmentrum.processing.torch_engine import upload
            return upload(np.asarray(indices, dtype=np.int64), self.data.device)
        return np.asarray(indices, dtype=np.int64)

    def batch(self, indices) -> Tuple[PooledBatch, Optional[PooledBatch]]:
        """
        The batch of subjects *indices*: one gather per tensor, no copies of objects.

        Returns:
            "(data, water)" PooledBatch objects (water None without a reference).
        """
        idx = self._indices(indices)
        data_src, water_src = self.source
        out = []
        for role, group, tensor, tags in (('data', data_src, self.data, self.data_tags),
                                          ('water', water_src, self.water, self.water_tags)):
            if group is None:
                out.append(None)
                continue
            batch = PooledBatch([group.nifti_list[i] for i in indices], backend=self.backend,
                                volatile=self.volatile)
            batch.set_data(self._gather(role, tensor, idx), self.backend, dim_tags=tags)
            set_origin(batch, PoolOrigin(self, idx, role))
            out.append(batch)
        return out[0], out[1]

    def _gather(self, role, tensor, idx):
        """
        The subjects *idx* of *tensor*, gathered into a buffer on CPU.

        The buffer is the previous batch's, if nothing but the pool refers to
        it; otherwise a new one is made and the old left to whoever holds it.
        """
        cpu_torch = self.backend == Backend.PYTORCH and tensor.device.type == 'cpu'
        if not (cpu_torch or self.backend == Backend.NUMPY):
            return ops.take(tensor, idx, axis=0)
        shape = (len(idx),) + tuple(tensor.shape[1:])
        if role not in self._buffers or tuple(self._buffers[role].shape) != shape \
                or not _unreferenced(self._buffers, role):
            if cpu_torch:
                import torch
                self._buffers[role] = torch.empty(shape, dtype=tensor.dtype)
            else:
                self._buffers[role] = np.empty(shape, dtype=tensor.dtype)
        buffer = self._buffers[role]
        if cpu_torch:
            import torch
            return torch.index_select(tensor, 0, idx, out=buffer)
        return np.take(tensor, idx, axis=0, out=buffer)

    #*************#
    #   caching   #
    #*************#
    def cached(self, key, build):
        """
        A per-subject result for the whole pool, computed once by *build(pool)*.

        Only for results that depend on a subject's pooled values alone; the
        cache lives and dies with the pool.
        """
        if key not in self._cache:
            self._cache[key] = build(self)
        return self._cache[key]

    def __repr__(self):
        shape = tuple(ops.shape(self.data))
        return (f"TensorPool(subjects={shape[0]}, shape={shape[1:]}, "
                f"backend={self.backend.value}, device={self.device})")


def _unreferenced(holder, key) -> bool:
    """
    Whether only *holder* refers to its entry *key*: no other name (the
    holder's reference and getrefcount's own argument make two) and, for a
    tensor, no view or array sharing its storage. Unknown means referenced.
    """
    if sys.getrefcount(holder[key]) > 2:
        return False
    if isinstance(holder[key], np.ndarray):
        return True
    try:
        import torch
        return torch._C._storage_Use_Count(holder[key].untyped_storage()._cdata) == 1
    except (AttributeError, RuntimeError, TypeError):
        return False
