####################################################################################################
#                                     edit_combination.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-14                                                                              #
#                                                                                                  #
# Purpose: Combines the two conditions of edited MRS (MEGA-PRESS ON/OFF) along DIM_EDIT, so the    #
#          result follows the same augmentation pipeline as any unedited spectrum.                 #
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


__all__ = ['EditCombiner']


#**************************************************************************************************#
#                                       Class EditCombiner                                         #
#**************************************************************************************************#
#                                                                                                  #
# Combines the two edit conditions along DIM_EDIT into one spectrum.                               #
#                                                                                                  #
#**************************************************************************************************#
class EditCombiner(BaseModule):
    """
    Combines the two edit conditions along DIM_EDIT into one spectrum.

    Edited MRS interleaves two acquisitions and the metabolite of interest
    lives in their combination: the difference reveals what the editing
    pulse touched (GABA+ at 3 ppm for MEGA-PRESS), the sum is the ordinary
    spectrum. Following FID-A's op_combinesubspecs, the combination is
    "(a ± b) / 2", so the result keeps the scale of a single condition.

    Which sign means "difference" depends on how the vendor stored the
    conditions: Big GABA Philips SDAT keeps the second condition
    phase-inverted, so there mode='sum' *is* the edited difference. That is
    the acquisition-scheme dependence the mode expresses — this module never
    guesses it.

    Data without a DIM_EDIT axis passes through untouched, so the module can
    sit in a pipeline that sees edited and unedited subjects alike. A water
    reference passes through unchanged either way: water is not edited, and
    collapsing its conditions is the raw processor's business.

    Args:
        mode: 'diff' for (a − b)·scale, 'sum' for (a + b)·scale, with a and b
            the conditions in stored order.
        scale: Factor on the combination. The default 1/2 keeps the result on
            the scale of one condition (FID-A convention).

    Examples:
        >>> import numpy as np
        >>> batch = np.ones((1, 1, 1, 1, 64, 8, 2), np.complex64)
        >>> combined, _ = EditCombiner(mode='diff').process_tensor(
        ...     batch, dim_tags=['DIM_DYN', 'DIM_EDIT', None])
        >>> combined.shape
        (1, 1, 1, 1, 64, 8)
    """

    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    # The conditions are combined away, and the tag goes with the axis.
    REMOVES_DIM_TAGS = ('DIM_EDIT',)

    MODES = ('diff', 'sum')

    def __init__(self, mode: str = 'diff', scale: float = 0.5):
        super().__init__()

        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")

        self.mode = mode
        self.scale = float(scale)

    #*****************#
    #   combination   #
    #*****************#
    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.NUMPY, **kwargs):
        """
        Combine the edit conditions, wherever the data lives.

        Args:
            data_array: Batch carrying a length-2 DIM_EDIT axis. Left
                untouched when it does not — there is nothing to combine.
            water_array: Passed through unchanged.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: Reads "dim_tags" to find the edit axis.

        Returns:
            "(combined, water_unchanged)" — the edit axis is gone.
        """
        tags = list(kwargs.get('dim_tags') or ())
        if 'DIM_EDIT' not in tags:
            return data_array, water_array

        axis = 5 + tags.index('DIM_EDIT')
        shape = tuple(int(n) for n in ops.shape(data_array))
        if axis >= len(shape):
            return data_array, water_array
        if shape[axis] != 2:
            raise ValueError(
                f"DIM_EDIT carries {shape[axis]} conditions; combining is "
                f"defined for exactly two.")

        first = ops.take(data_array, np.array([0]), axis=axis)
        second = ops.take(data_array, np.array([1]), axis=axis)
        combined = (first - second) if self.mode == 'diff' else (first + second)
        combined = combined * ops.cast_like(
            ops.match_backend(np.asarray(self.scale), combined), combined)

        return ops.reshape(combined, shape[:axis] + shape[axis + 1:]), water_array
