####################################################################################################
#                                      subject_splitter.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Implements SubjectSplitter for reproducible subject-wise train/val/test splits.         #
#          Supports stratified or random splitting modes.                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Dict, List, Optional, Sequence
import numpy as np


#**************************************************************************************************#
#                                      Class SubjectSplitter                                       #
#**************************************************************************************************#
#                                                                                                  #
# Splits subjects into train/val/test sets.                                                        #
#                                                                                                  #
#**************************************************************************************************#
class SubjectSplitter:
    """
    Splits subjects into train/val/test sets.

    Works with any backend (NIfTI_MRS_Plus, lists, or raw data).
    Returns splits in the same format as input.

    Items can be grouped: all scans of one subject, say, carry the same group
    id and then always land in the same split, so that a model validated on a
    subject it trained on cannot pass as generalizing. Without groups each item
    is placed on its own.
    """

    def __init__(self, data, water=None, seed=0, val_frac=0.1, test_frac=0.1,
                 groups: Optional[Sequence] = None):
        """
        Initialization.

        Args:
            data: Input data (NIfTI_MRS_Plus, list, or other).
            water: Water reference data (same format as data), optional.
            seed (int): Random seed for reproducibility.
            val_frac (float): Fraction of data for validation set.
            test_frac (float): Fraction of data for test set.
            groups: One hashable id per item. Items sharing an id are kept in
                one split, and the fractions are honoured in number of items
                as closely as the group sizes allow. None splits item-wise.
        """
        self.data = data
        self.water = water
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.groups = None if groups is None else list(groups)

        # Filled by split(): which item indices, and which group ids, each
        # split holds. Kept so a caller can report or check the assignment
        # without having to identify the objects it got back.
        self.split_indices: Optional[Dict[str, List[int]]] = None
        self.split_groups: Optional[Dict[str, list]] = None

    def __call__(self, **kwargs):
        """
        Perform the split and return the dictionary of splits.
        """
        return self.split()

    def split(self):
        """
        Splits subjects into train/val/test sets.

        Handles NIfTI_MRS_Plus with any backend by converting to list,
        splitting, and optionally converting back.

        Returns:
            dict: Dictionary with 'train', 'val', and 'test' keys.
                  Each value is a tuple of (data, water).
        """
        # Import here to avoid circular imports
        from augmentrum.core import NIfTI_MRS_Plus

        # Store original type and backend for later reconstruction
        is_nifti_plus = isinstance(self.data, NIfTI_MRS_Plus)
        original_backend = self.data.backend if is_nifti_plus else None
        original_volatile = self.data.volatile if is_nifti_plus else False

        # Convert to list for splitting
        if is_nifti_plus:
            data_list = self.data.to_nifti_list()
        else:
            data_list = self.data if isinstance(self.data, list) else list(self.data)

        if self.water is not None:
            if isinstance(self.water, NIfTI_MRS_Plus):
                water_list = self.water.to_nifti_list()
            else:
                water_list = self.water if isinstance(self.water, list) else list(self.water)

            # Validate that water_list and data_list have the same length
            if len(water_list) != len(data_list):
                raise ValueError(
                    f"Mismatch between data and water lists: "
                    f"data has {len(data_list)} subjects but water has {len(water_list)} subjects. "
                    f"They must have the same length."
                )
        else:
            water_list = None

        n_total = len(data_list)
        if self.groups is not None:
            if len(self.groups) != n_total:
                raise ValueError(
                    f"groups has {len(self.groups)} entries for {n_total} items; "
                    f"give one group id per item."
                )
            selection = self._split_by_group(n_total)
        else:
            selection = self._split_by_item(n_total)

        self.split_indices = selection
        self.split_groups = None
        if self.groups is not None:
            self.split_groups = {name: self._sorted_ids({self.groups[i] for i in idx})
                                 for name, idx in selection.items()}

        def get_lists(sel):
            data_sel = [data_list[i] for i in sel]
            water_sel = [water_list[i] for i in sel] if water_list is not None else None

            # Optionally convert back to NIfTI_MRS_Plus with original backend
            if is_nifti_plus and len(data_sel) > 0:
                data_sel = NIfTI_MRS_Plus(
                    nifti_list=data_sel,
                    backend=original_backend,
                    volatile=original_volatile
                )
                if water_sel is not None:
                    water_sel = NIfTI_MRS_Plus(
                        nifti_list=water_sel,
                        backend=original_backend,
                        volatile=original_volatile
                    )

            return data_sel, water_sel

        return {name: get_lists(selection[name]) for name in ('train', 'val', 'test')}

    #*************************#
    #   assignment strategy   #
    #*************************#
    def _split_by_item(self, n_total: int) -> Dict[str, List[int]]:
        """
        Place every item on its own: a random permutation cut at the fractions.

        Args:
            n_total: Number of items.

        Returns:
            Item indices per split.
        """
        idxs = self.rng.permutation(n_total).tolist()
        n_test = int(n_total * self.test_frac)
        n_val = int(n_total * self.val_frac)
        return {
            'train': idxs[n_test + n_val:],
            'val': idxs[n_test:n_test + n_val],
            'test': idxs[:n_test],
        }

    def _split_by_group(self, n_total: int) -> Dict[str, List[int]]:
        """
        Place whole groups, filling test then val as close to their targets as the sizes allow.

        Groups are visited in a seeded random order. A group joins the split
        being filled when doing so brings the item count nearer to the
        split's target than leaving it out would; otherwise it stays for the
        next split, and whatever is left is training. Greedy rather than an
        exact subset sum, which is more than a split deserves: with three
        scans per subject and a target of 1.8 items, taking one subject (3,
        off by 1.2) beats taking none (off by 1.8), and that is the answer a
        person would give too.

        Args:
            n_total: Number of items.

        Returns:
            Item indices per split, each in original order.
        """
        members: Dict[object, List[int]] = {}
        for index, group in enumerate(self.groups):
            members.setdefault(group, []).append(index)

        ids = list(members)
        remaining = [ids[k] for k in self.rng.permutation(len(ids))]
        targets = {'test': n_total * self.test_frac, 'val': n_total * self.val_frac}

        chosen: Dict[str, list] = {}
        for name in ('test', 'val'):
            count, taken, kept = 0, [], []
            for group in remaining:
                size = len(members[group])
                if abs(count + size - targets[name]) < abs(count - targets[name]):
                    taken.append(group)
                    count += size
                else:
                    kept.append(group)
            chosen[name] = taken
            remaining = kept
        chosen['train'] = remaining

        return {name: sorted(i for group in chosen[name] for i in members[group])
                for name in ('train', 'val', 'test')}

    @staticmethod
    def _sorted_ids(ids) -> list:
        """Group ids in a stable order; mixed types fall back to their string form."""
        try:
            return sorted(ids)
        except TypeError:
            return sorted(ids, key=str)
