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

    Args:
        mode: "random" to draw a subset, "deterministic" to keep everything or
            exactly the indices passed to the call.
        count: How many to keep, as "(min, max)" inclusive. A bare number means
            exactly that many; "None" means anywhere from one to all of them.
        seed: Fixes the sequence. Draws still vary from call to call.
    """

    #: Higher-dimension tag this sampler draws along, e.g. "DIM_COIL".
    DIM_TAG: str = ''

    #: How the provenance record names what was done.
    OPERATION: str = 'Dimension Sampling'

    SUPPORTED_BACKENDS = tuple(Backend)

    def __init__(self, mode: str = 'random', count=None, seed=None):
        super().__init__()

        if mode not in ('random', 'deterministic'):
            raise ValueError(
                f"mode must be 'random' or 'deterministic', got {mode!r}.")

        self.mode = mode
        self.count = self.as_range(count)

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
        """
        if count is None:
            return (1, None)
        if isinstance(count, (int, float)):
            return (int(count), int(count))
        if isinstance(count, tuple):
            return count
        raise TypeError(f"Count must be int, float, or tuple, got {type(count)}")

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

    def draw(self, n_total):
        """Indices to keep out of *n_total*, or None to keep all of them."""
        if self.mode == 'deterministic':
            return None

        low, high = self._limits(n_total)
        if low >= high:
            return None

        rng = self.rng.numpy_rng()
        return rng.permutation(n_total)[:int(rng.integers(low, high))].tolist()

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
            water_array: Passed through unchanged.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: Absorbs what BaseModule injects, and reads "dim_tags"
                from it to find the axis. A bare tensor names no axes, so
                without them there is nothing to act on.

        Returns:
            "(data, water_unchanged)".
        """
        tags = list(kwargs.get('dim_tags') or ())
        if self.DIM_TAG not in tags:
            return data_array, water_array

        axis = 5 + tags.index(self.DIM_TAG)
        shape = ops.shape(data_array)
        if axis >= len(shape):
            return data_array, water_array

        keep = self.draw(int(shape[axis]))
        if keep is None:
            return data_array, water_array

        return ops.take(data_array, np.asarray(keep), axis=axis), water_array

    def process_nifti_list(self, data_list, water_list=None, indices=None, **kwargs):
        """
        Draw from each NIFTI_MRS in the list, keeping its headers correct.

        Args:
            data_list: Metabolite MRS data.
            water_list: Water reference data, optional.
            indices: Exactly what to keep, instead of drawing.

        Returns:
            "(processed_data_list, processed_water_list)".
        """
        processed_data, processed_water = [], []

        for i, data_met in enumerate(data_list):
            data_wat = water_list[i] if water_list is not None else None
            data_met, data_wat = self._split(data_met, data_wat, indices)

            processed_data.append(data_met)
            if water_list is not None:
                processed_water.append(data_wat if data_wat is not None else water_list[i])

        return processed_data, (processed_water if water_list is not None else None)

    def _split(self, data_met, data_wat, indices):
        """Keep *indices* along this dimension, drawing them when none were given."""
        from fsl_mrs.core.nifti_mrs import split

        tag = self.DIM_TAG
        if tag not in getattr(data_met, 'dim_tags', []):
            return data_met, data_wat

        if indices is None:
            n_total = data_met.shape[data_met.dim_position(tag)]
            indices = (list(range(n_total)) if self.mode == 'deterministic'
                       else self.draw(n_total))
        if indices is None:
            return data_met, data_wat

        _, data_met = split(data_met, tag, indices)
        if data_wat is not None and tag in getattr(data_wat, 'dim_tags', []):
            _, data_wat = split(data_wat, tag, indices)

        details = f'{__name__}.process_nifti_list, {tag} indices={indices}.'
        update_processing_prov(data_met, self.OPERATION, details)
        if data_wat is not None:
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
    an axis is all this needs.

    Args:
        mode: "random" to draw a subset, "deterministic" to keep everything or
            exactly the indices passed to the call.
        n_averages: How many to keep, as "(min, max)" inclusive. A bare number
            means exactly that many; "None" means anywhere from one to all.
        seed: Fixes the sequence. Draws still vary from call to call.

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

    def __init__(self, mode: str = 'random', n_averages=None, seed=None):
        super().__init__(mode=mode, count=n_averages, seed=seed)

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
