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
import torch


#**************************************************************************************************#
#                                          Class SubjectSampler                                    #
#**************************************************************************************************#
#                                                                                                  #
# Simplistic subject splitter for train/val/test splits. Supports random splitting.                #
#                                                                                                  #
#**************************************************************************************************#
class SubjectSplitter:
    """
    Splits subjects into train/val/test sets.
    """

    def __init__(self, data, water=None, seed=0, val_frac=0.1, test_frac=0.1):
        """
        Initialization.

        Args:
            data (List): List of subjects (e.g., file paths or IDs).
            water (List, optional): List of water reference subjects. Not used in this splitter.
            seed (int): Random seed for reproducibility.
            val_frac (float): Fraction of data for validation set.
            test_frac (float): Fraction of data for test set.
        """
        self.data = data
        self.water = water
        self.val_frac = val_frac
        self.test_frac = test_frac
        torch.manual_seed(seed)

    def __call__(self, **kwargs):
        """
        Perform the split and return the dictionary of splits.
        """
        return self.split()

    def split(self):
        """
        Splits subjects into train/val/test sets.

        Returns:
            dict: Dictionary with 'train', 'val', and 'test' keys mapping to lists of subjects.
        """
        idxs = torch.randperm(len(self.data)).tolist()
        n_total = len(idxs)
        n_test = int(n_total * self.test_frac)
        n_val = int(n_total * self.val_frac)

        def get_lists(sel):
            data_list = [self.data[i] for i in sel]
            water_list = [self.water[i] for i in sel] if self.water is not None else None
            return data_list, water_list

        return {
            'train': get_lists(idxs[n_test + n_val:]),
            'val': get_lists(idxs[n_test:n_test + n_val]),
            'test': get_lists(idxs[:n_test]),
        }