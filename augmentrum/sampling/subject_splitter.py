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
    """

    def __init__(self, data, water=None, seed=0, val_frac=0.1, test_frac=0.1):
        """
        Initialization.

        Args:
            data: Input data (NIfTI_MRS_Plus, list, or other).
            water: Water reference data (same format as data), optional.
            seed (int): Random seed for reproducibility.
            val_frac (float): Fraction of data for validation set.
            test_frac (float): Fraction of data for test set.
        """
        self.data = data
        self.water = water
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.rng = np.random.default_rng(seed)

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

        # Perform random split
        idxs = self.rng.permutation(len(data_list)).tolist()
        n_total = len(idxs)
        n_test = int(n_total * self.test_frac)
        n_val = int(n_total * self.val_frac)

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

        return {
            'train': get_lists(idxs[n_test + n_val:]),
            'val': get_lists(idxs[n_test:n_test + n_val]),
            'test': get_lists(idxs[:n_test]),
        }