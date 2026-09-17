####################################################################################################
#                                     dimension_sampling.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Drawing a subset along one of the higher NIfTI-MRS dimensions, and the average sampler  #
#          that is exactly that pointed at DIM_DYN. Coils need synthesis as well as drawing, so    #
#          they get their own module and inherit the drawing from here.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from nifti_mrs_plus import ops

# own
from augmentrum.core import Backend
from augmentrum.core.base_module import BaseModule
from augmentrum.processing.utils import update_processing_prov


__all__ = ['DimensionSampler', 'AverageSampler']


#**************************************************************************************************#
#                                     Class DimensionSampler                                       #
#**************************************************************************************************#
#                                                                                                  #
# Draws a subset along one higher dimension, simulating a shorter acquisition.                     #
#                                                                                                  #
#**************************************************************************************************#
class DimensionSampler(BaseModule):
    """
    Draws a subset along one higher dimension, simulating a shorter acquisition.

    A scan that keeps fewer coils, or fewer averages, is a scan that took less
    time and carries less signal. Simulating that is the same operation either
    way - keep some of one axis - so a subclass supplies only which axis it
    means, and the drawing itself is written once.

    Drawing is a gather, so it runs natively wherever the data lives and keeps
    its gradients. A NIFTI_LIST input goes through FSL-MRS instead, which
    carries the headers and provenance across properly.

    A gather keeps one count for the whole batch, since a tensor cannot go
    ragged. With "per_sample" every sample draws its own count and subset
    instead: on the list path each subject is cut to its own, and on tensors
    the axis stays whole while a boolean (batch, n) mask per dimension, carried
    on the batch, says which entries each sample kept. A consumer (the raw
    processor) then combines and averages over the kept entries only - exactly
    what it would do with the subset gathered - and whatever is left unconsumed
    at the end of a pipeline is zeroed. A mask forgets the order a subset was
    drawn in; the only consequence is the combined signal's global phase,
    pinned to the first coil kept, which eddy current and phase correction
    remove.

    Args:
        mode: "random" to draw a subset, "deterministic" to keep everything or
            exactly the indices passed to the call.
        count: How many to keep, as "(min, max)" inclusive. A bare number means
            exactly that many; "None" means anywhere from one to all of them.
            With *per_sample*, a pipeline draws one count per sample, and a
            (batch,) array of counts is taken entry by entry.
        scheme: How the kept indices relate to each other. "random" (default)
            draws them anywhere; "consecutive" keeps one contiguous window at
            a random start; "strided" keeps every *stride*-th index from a
            random start. Windows only mean something along an axis with a
            temporal order, i.e. the averages.
        stride: Spacing between kept indices for scheme="strided".
        seed: Fixes the sequence. Draws still vary from call to call.
        per_sample: Draw a count and subset per sample, as masks on tensors
            (see above). Off by default, which keeps the batch-wide gather.
    """

    #: Higher-dimension tag this sampler draws along, e.g. "DIM_COIL".
    DIM_TAG: str = ''

    #: How the provenance record names what was done.
    OPERATION: str = 'Dimension Sampling'

    #: Whether the water reference shares this axis, so a draw applies to it
    #: too. Receive elements are the same hardware for both acquisitions, so
    #: a coil subset must be kept on the water as well; transients are not -
    #: the water has its own one or two - so an average draw leaves it alone.
    WATER_SHARES_DIM: bool = False

    SUPPORTED_BACKENDS = tuple(Backend)

    SCHEMES = ('random', 'consecutive', 'strided')

    # A count range is inclusive and integer: (8, 32) may give 8 and may give 32.
    INTEGER_PARAMS = ('count', 'n_averages', 'n_coils')

    # Drawing masks refines them; gathering carries them along the kept entries.
    # A mask leaves the values alone, so a masked batch keeps its pool origin.
    MASKS = 'draw'
    PRESERVES_VALUES = True

    def __init__(self, mode: str = 'random', count=None, scheme: str = 'random',
                 stride: int = 1, seed=None, per_sample: bool = False):
        super().__init__()

        if mode not in ('random', 'deterministic'):
            raise ValueError(
                f"mode must be 'random' or 'deterministic', got {mode!r}.")
        if scheme not in self.SCHEMES:
            raise ValueError(
                f"scheme must be one of {self.SCHEMES}, got {scheme!r}.")
        if int(stride) < 1:
            raise ValueError(f"stride must be at least 1, got {stride!r}.")

        self.mode = mode
        self.count = self.as_range(count)
        self.scheme = scheme
        self.stride = int(stride)
        self.per_sample = bool(per_sample)
        if self.per_sample:
            # the pipeline then samples a count range once per sample
            self.PER_SAMPLE_PARAMS = self.INTEGER_PARAMS

    #************#
    #   counts   #
    #************#
    @staticmethod
    def as_range(count):
        """
        Read a count as an inclusive "(min, max)" range.

        Examples:
            4 -> (4, 4)         exactly four
            (1, 8) -> (1, 8)    anywhere from one to eight
            None -> (1, None)   at least one, as many as there are
            array([3, 7]) -> array([3, 7])   exactly that many, sample by sample
        """
        if count is None:
            return (1, None)
        if isinstance(count, (int, float, np.integer, np.floating)):
            return (int(count), int(count))
        if isinstance(count, tuple):
            return count
        if isinstance(count, (np.ndarray, list)):
            counts = np.asarray(count)
            return (int(counts), int(counts)) if counts.ndim == 0 else counts.astype(np.int64)
        raise TypeError(f"Count must be int, float, tuple or array, got {type(count)}")

    def _limits(self, n_total):
        """
        Turn the inclusive range into the exclusive upper bound a draw wants.

        Examples::

            (1, None) over 4  ->  (1, 5)   # draws 1, 2, 3 or 4
            (2, 3)    over 4  ->  (2, 4)   # draws 2 or 3
            (6, None) over 4  ->  (4, 5)   # clamped to what exists
        """
        low, high = self.count
        return (1 if low is None else max(1, min(int(low), n_total)),
                (n_total if high is None else min(int(high), n_total)) + 1)

    def draw(self, n_total, index=None):
        """
        Indices to keep out of *n_total*, or None to keep all of them.

        Args:
            n_total: Size of the axis drawn from.
            index: Which sample a per-sample count array is read at (the list
                path draws subject by subject).
        """
        if self.mode == 'deterministic':
            return None

        rng = self.rng.numpy_rng()
        if isinstance(self.count, np.ndarray):
            if index is None:
                raise ValueError("A count per sample needs per-sample drawing: pass "
                                 "per_sample=True, or give one count for the batch.")
            kept = int(np.clip(self.count[index % len(self.count)], 1, n_total))
        else:
            low, high = self._limits(n_total)
            if low >= high:
                return None
            kept = int(rng.integers(low, high))

        if self.scheme == 'random':
            return rng.permutation(n_total)[:kept].tolist()

        # A window of *kept* picks, *step* apart, from a random start. The
        # window must fit: its last pick is start + step * (kept - 1), so the
        # count is clamped before the start is drawn rather than after.
        step = self.stride if self.scheme == 'strided' else 1
        kept = min(kept, (n_total - 1) // step + 1)
        start = int(rng.integers(0, n_total - step * (kept - 1)))
        return list(range(start, start + step * kept, step))

    #*************#
    #   drawing   #
    #*************#
    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.NUMPY, **kwargs):
        """
        Draw along this sampler's dimension, wherever the data lives.

        Args:
            data_array: Batch carrying this sampler's dimension. Left untouched
                when it does not - there is nothing to draw from.
            water_array: Passed through unchanged, unless the water shares
                this dimension ("WATER_SHARES_DIM") - then the same indices
                are kept along the water's own axis of it.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: Absorbs what BaseModule injects, and reads "dim_tags"
                from it to find the axis ("water_dim_tags" for the water's).
                A bare tensor names no axes, so without them there is
                nothing to act on.

        Returns:
            "(data, water)".
        """
        tags = list(kwargs.get('dim_tags') or ())
        if self.DIM_TAG not in tags:
            return data_array, water_array

        axis = 5 + tags.index(self.DIM_TAG)
        shape = ops.shape(data_array)
        if axis >= len(shape):
            return data_array, water_array

        masks = dict(kwargs.get('dim_masks') or {})
        self.dim_masks_ = masks
        if self.per_sample:
            mask = self.draw_mask(int(shape[axis]), int(shape[0]), data_array,
                                  masks.get(self.DIM_TAG))
            if mask is not None:
                masks[self.DIM_TAG] = mask
            return data_array, water_array

        keep = self.draw(int(shape[axis]))
        if keep is None:
            return data_array, water_array

        data_array = ops.take(data_array, np.asarray(keep), axis=axis)
        if water_array is not None and self.WATER_SHARES_DIM:
            water_array = self._take_from_water(water_array, keep, kwargs)
        if self.DIM_TAG in masks:
            masks[self.DIM_TAG] = ops.take(masks[self.DIM_TAG], np.asarray(keep), axis=1)
        return data_array, water_array

    #******************#
    #   sample masks   #
    #******************#
    def draw_mask(self, n_total, batch, like, existing=None):
        """
        Which entries each sample keeps: a (batch, n_total) boolean mask.

        Counts come from a per-sample array or are drawn per sample from the
        range; subsets are drawn on *like*'s device from this module's
        generator (torch data gets a torch mask, anything else a NumPy one).
        A mask already on the axis is refined: the new subset is drawn among
        the entries it kept.

        Args:
            n_total: Size of the axis.
            batch: Number of samples.
            like: The data, for backend and device.
            existing: A mask over the axis from an earlier draw, or None.

        Returns:
            The mask, or *existing* (None included) where every sample keeps
            everything it has.
        """
        if self.mode == 'deterministic':
            return existing
        if isinstance(self.count, np.ndarray):
            counts = np.clip(np.resize(self.count, batch), 1, n_total)
            if existing is None and (counts == n_total).all():
                return None
            low = high = None
        else:
            low, high = self._limits(n_total)             # high is exclusive
            if existing is None and low >= n_total:
                return None
            counts = None

        if ops.is_torch(like):
            return self._mask_torch(n_total, batch, like.device, counts, low, high, existing)
        return self._mask_numpy(n_total, batch, counts, low, high, existing)

    def _mask_torch(self, n_total, batch, device, counts, low, high, existing):
        """"draw_mask" on the data's torch device, without leaving it."""
        import torch

        probe = torch.empty(0, device=device)
        if counts is None:
            u = self.rng.uniform((batch,), like=probe, dtype='float64')
            counts = torch.clamp(low + torch.floor(u * (high - low)).long(), max=high - 1)
        else:
            from augmentrum.processing.torch_engine import upload
            counts = upload(counts, device)
        if existing is not None:
            counts = torch.minimum(counts, existing.sum(dim=1))

        u = self.rng.uniform((batch, n_total), like=probe)
        position = torch.arange(n_total, device=device).expand(batch, n_total)
        if self.scheme == 'random':
            if existing is not None:
                u = torch.where(existing, u, 2.0)                # drawn from the kept only
            ranks = torch.empty_like(position).scatter_(1, torch.argsort(u, dim=1), position)
            return ranks < counts[:, None]

        step = self.stride if self.scheme == 'strided' else 1
        kept = torch.clamp(counts, max=(n_total - 1) // step + 1)
        start = torch.floor(u[:, 0] * (n_total - step * (kept - 1))).long()
        return self._window(position - start[:, None], kept, step, existing)

    def _mask_numpy(self, n_total, batch, counts, low, high, existing):
        """"draw_mask" in NumPy, for every backend that is not torch."""
        if counts is None:
            u = self.rng.uniform((batch,), dtype='float64')
            counts = np.minimum(low + np.floor(u * (high - low)).astype(np.int64), high - 1)
        if existing is not None:
            existing = np.asarray(ops.to_numpy(existing), dtype=bool)
            counts = np.minimum(counts, existing.sum(axis=1))

        u = self.rng.uniform((batch, n_total))
        position = np.broadcast_to(np.arange(n_total), (batch, n_total))
        if self.scheme == 'random':
            if existing is not None:
                u = np.where(existing, u, 2.0)                   # drawn from the kept only
            ranks = np.argsort(np.argsort(u, axis=1, kind='stable'), axis=1, kind='stable')
            return ranks < counts[:, None]

        step = self.stride if self.scheme == 'strided' else 1
        kept = np.minimum(counts, (n_total - 1) // step + 1)
        start = np.floor(u[:, 0] * (n_total - step * (kept - 1))).astype(np.int64)
        return self._window(position - start[:, None], kept, step, existing)

    @staticmethod
    def _window(offset, kept, step, existing):
        """
        A window of *kept* picks, *step* apart, at the offsets given; the start
        was drawn so that it fits, as the gather's window does.
        """
        mask = (offset >= 0) & (offset < step * kept[:, None]) & (offset % step == 0)
        return mask & existing if existing is not None else mask

    def _take_from_water(self, water_array, keep, kwargs):
        """
        Keep *keep* along the water's own axis of this dimension.

        The water's layout is its own ("water_dim_tags"); without it, the
        data's tags stand in for the axes the water has. A water that does
        not carry the dimension is returned as it is.
        """
        wtags = kwargs.get('water_dim_tags')
        if wtags is None:
            wtags = kwargs.get('dim_tags')
        wtags = list(wtags or ())
        if self.DIM_TAG not in wtags:
            return water_array

        axis = 5 + wtags.index(self.DIM_TAG)
        if axis >= len(ops.shape(water_array)):
            return water_array
        return ops.take(water_array, np.asarray(keep), axis=axis)

    def process_nifti_list(self, data_list, water_list=None, indices=None, **kwargs):
        """
        Draw from each NIFTI_MRS in the list, keeping its headers correct.

        Args:
            data_list: Metabolite MRS data.
            water_list: Water reference data, optional; drawn from along a
                shared dimension only (see "WATER_SHARES_DIM").
            indices: Exactly what to keep, instead of drawing.

        Returns:
            "(processed_data_list, processed_water_list)".
        """
        processed_data, processed_water = [], []

        for i, data_met in enumerate(data_list):
            data_wat = water_list[i] if water_list is not None else None
            data_met, data_wat = self._split(data_met, data_wat, indices, index=i)

            processed_data.append(data_met)
            if water_list is not None:
                processed_water.append(data_wat if data_wat is not None else water_list[i])

        return processed_data, (processed_water if water_list is not None else None)

    def _split(self, data_met, data_wat, indices, index=None):
        """Keep *indices* along this dimension; without them, draw for sample *index*."""
        from fsl_mrs.core.nifti_mrs import split

        tag = self.DIM_TAG
        if tag not in getattr(data_met, 'dim_tags', []):
            return data_met, data_wat

        if indices is None:
            n_total = data_met.shape[data_met.dim_position(tag)]
            indices = (list(range(n_total)) if self.mode == 'deterministic'
                       else self.draw(n_total, index=index))
        if indices is None:
            return data_met, data_wat

        _, data_met = split(data_met, tag, indices)
        details = f'{__name__}.process_nifti_list, {tag} indices={indices}.'
        update_processing_prov(data_met, self.OPERATION, details)

        # The water is drawn from only along a dimension it shares with the
        # data; its own transients stay, and so does its provenance then.
        if (data_wat is not None and self.WATER_SHARES_DIM
                and tag in getattr(data_wat, 'dim_tags', [])):
            _, data_wat = split(data_wat, tag, indices)
            update_processing_prov(data_wat, self.OPERATION, details)

        return data_met, data_wat


#**************************************************************************************************#
#                                      Class AverageSampler                                        #
#**************************************************************************************************#
#                                                                                                  #
# Draws a subset of the averages, simulating a shorter scan.                                       #
#                                                                                                  #
#**************************************************************************************************#
class AverageSampler(DimensionSampler):
    """
    Draws a subset of the averages, simulating a shorter scan.

    Repeating an acquisition and averaging is how MRS buys signal-to-noise, at
    a cost in time, so the number of averages is the knob a protocol trades
    against. Keeping fewer of them reproduces what a shorter scan would have
    measured - the same spectrum, noisier - which is the variability worth
    training against.

    There is nothing to it beyond naming the dimension, because drawing along
    an axis is all this needs. The water reference is left alone: it has its
    own one or two transients, which are not the data's to draw from.

    Args:
        mode: "random" to draw a subset, "deterministic" to keep everything or
            exactly the indices passed to the call.
        n_averages: How many to keep, as "(min, max)" inclusive. A bare number
            means exactly that many; "None" means anywhere from one to all.
        scheme: "random" draws the kept averages anywhere; "consecutive" keeps
            a contiguous window at a random start, which is what an actually
            shorter scan would have recorded; "strided" keeps every *stride*-th
            transient of a window, thinning the scan instead of ending it.
        stride: Spacing between kept transients for scheme="strided".
        seed: Fixes the sequence. Draws still vary from call to call.
        per_sample: Every sample draws its own count and transients; on tensors
            they are kept as a mask the raw processor averages over (see
            DimensionSampler).

    Examples:
        >>> import numpy as np
        >>> batch = np.ones((2, 1, 1, 1, 512, 32), np.complex64)
        >>> drawn, _ = AverageSampler(n_averages=8, seed=0).process_tensor(
        ...     batch, dim_tags=['DIM_DYN', None, None])
        >>> drawn.shape
        (2, 1, 1, 1, 512, 8)
    """

    DIM_TAG = 'DIM_DYN'
    OPERATION = 'Average Sampling'

    def __init__(self, mode: str = 'random', n_averages=None,
                 scheme: str = 'random', stride: int = 1, seed=None,
                 per_sample: bool = False):
        super().__init__(mode=mode, count=n_averages, scheme=scheme,
                         stride=stride, seed=seed, per_sample=per_sample)

    #*******************#
    #   count aliases   #
    #*******************#
    # The pipeline injects per-batch values under the constructor's own
    # argument name, so that name has to reach the count the draw reads.
    @property
    def n_averages(self):
        return self.count

    @n_averages.setter
    def n_averages(self, value):
        self.count = self.as_range(value)
