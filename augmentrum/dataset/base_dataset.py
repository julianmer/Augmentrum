####################################################################################################
#                                        base_dataset.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Defines the abstract BaseMRSDataset class. This serves as the common interface for      #
#          MRS datasets, integrating subject sampling, raw preprocessing, and augmentation.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# internal
from augmentrum.augmentation.pipeline import AugmentationPipeline
from augmentrum.augmentation.signal_peturber import NoisePerturber
from augmentrum.processing.raw_processor import RawProcessor
from augmentrum.sampling.coil_average_sampler import CoilAverageSampler
from augmentrum.sampling.subject_splitter import SubjectSplitter



#**************************************************************************************************#
#                                        Class BaseMRSDataset                                      #
#**************************************************************************************************#
#                                                                                                  #
# Abstract base class for MRS datasets. Integrates subject sampling, raw preprocessing, and        #
# augmentation. Supports both random and deterministic sampling modes.                             #
#                                                                                                  #
#**************************************************************************************************#
class BaseMRSDataset(Dataset):
    """
    Takes care of building the pipeline and managing random/deterministic sampling modes.
    """

    def __init__(self, data, water=None, mode='random', pipeline=None,
                 n_coils=(1, None), n_averages=(1, None), to_tensor=True):
        """
        Initialization.

        Args:
            data (List): List of subjects (NiftiMRS objects).
            water (List, optional): List of water reference subjects (NiftiMRS objects
            mode (str): 'random' or 'deterministic' sampling mode.
            pipeline (AugmentationPipeline, optional): Augmentation pipeline to apply.
            n_coils (tuple): Min/max number of coils to sample.
            n_averages (tuple): Min/max number of averages to sample.
            to_tensor (bool): Whether to convert outputs to PyTorch tensors.
        """
        self.data = data
        self.water = water
        self.mode = mode
        self.n_coils = n_coils
        self.n_averages = n_averages
        self.to_tensor = to_tensor

        if pipeline is None:
            self.pipeline = AugmentationPipeline([CoilAverageSampler(mode=mode),
                                                  RawProcessor()])
        else:
            self.pipeline = pipeline

        if self.mode == 'deterministic':
            self.deterministic_idxs = []
            self.build_deterministic_indices()
            print(self.deterministic_idxs)

    def __len__(self):
        """
        Returns the length of the dataset. In deterministic mode, this is the number of
        precomputed indices. In random mode, the length is undefined (returns 0).
        """
        if self.mode == 'deterministic':
            return len(self.deterministic_idxs)
        else:
            return 0  # random mode has no fixed length

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset. Only works in deterministic mode.
        In random mode, raises an error and suggests using iter_batches instead.

        Args:
            idx (int): Index of the item to retrieve.
        """
        if self.mode == 'deterministic':
            subj_idx, coil_idx, average_idx = self.deterministic_idxs[idx]
            data = self.data[subj_idx].copy()
            water = self.water[subj_idx].copy() if self.water is not None else None
            data, water = self.pipeline(data=data, water=water, coil_indices=coil_idx,
                                        average_indices=average_idx)
            if self.to_tensor:
                data, water = self.to_tensor_batch([data], [water])
            return data, water

        else:
            raise RuntimeError("Use iter_batches for random mode")

    def to_tensor_batch(self, data, water=None):
        """
        Converts a batch of data and water references to PyTorch tensors.

        Args:
            data (List): List of MRS data (NiftiMRS objects).
            water (List, optional): List of water reference MRS data (NiftiM
        """
        data = [torch.from_numpy(np.asarray(d[:])) for d in data]
        data = torch.squeeze(torch.stack(data, dim=0), dim=(1, 2, 3, 4))

        if water is not None:
            water = [torch.from_numpy(np.asarray(d[:])) for d in water]
            water = torch.squeeze(torch.stack(water, dim=0), dim=(1, 2, 3, 4))
        return data, water

    def iter_batches(self, batch_size=16):
        """
        Infinite generator yielding batches of data. Only works in random mode.
        In deterministic mode, use DataLoader instead.

        Args:
            batch_size (int): Number of samples per batch.
        """
        while True:
            batch_data, batch_water = [], []
            for _ in range(batch_size):
                s_idx = torch.randint(0, len(self.data), (1,)).item()
                data = self.data[s_idx].copy()
                water = self.water[s_idx].copy() if self.water is not None else None
                data, water = self.pipeline(data=data, water=water)
                batch_data.append(data)
                batch_water.append(water)
            if self.to_tensor:
                batch_data, batch_water = self.to_tensor_batch(batch_data, batch_water)
            yield batch_data, batch_water

    def build_deterministic_indices(self):
        """
        Builds the list of deterministic indices for sampling.

        Currently samples all combinations of subjects, coils, and averages
        within the specified min/max ranges, but sorted.

        Note: This could be extended to sample all combinations of coil/average indices.
        """
        for subj_idx, d in enumerate(self.data):
            if 'DIM_COIL' in getattr(d, 'dim_tags', []):
                min_c, max_c = self._get_limits(self.n_coils, d.shape[d.dim_position('DIM_COIL')] - 1)
                c_max = d.shape[d.dim_position('DIM_COIL')] - 1
            else:
                min_c, max_c, c_max = 0, 1, 0
            if 'DIM_DYN' in getattr(d, 'dim_tags', []):
                min_a, max_a = self._get_limits(self.n_averages, d.shape[d.dim_position('DIM_DYN')] - 1)
                a_max = d.shape[d.dim_position('DIM_DYN')] - 1
            else:
                min_a, max_a, a_max = 0, 1, 0
            coil_combinations = []
            for n_c in range(min_c, max_c + 1):
                for n_a in range(min_a, max_a + 1):
                    self.deterministic_idxs.append((subj_idx, c_max - n_c, a_max - n_a))

    def _get_limits(self, n, n_max):
        """
        Converts (min, max) tuple with possible None values to concrete integer limits.

        Args:
            n (tuple): (min, max) tuple where either can be None.
            n_max (int): Maximum possible value.
        """
        if n[0] is None:
            min_n = n_max
        else:
            min_n = max(0, min(n[0], n_max))
        if n[1] is None:
            max_n = n_max
        else:
            max_n = min(n[1], n_max)
        return min_n, max_n


#**************************************************************************************************#
#                                     Class BaseMRSDatasetLoader                                   #
#**************************************************************************************************#
#                                                                                                  #
# Dataset loader that manages train/val/test splits, batching, and caching for MRS datasets        #
# based on BaseMRSDataset. Supports both random and deterministic sampling modes.                  #
#                                                                                                  #
#**************************************************************************************************#
class BaseMRSDatasetLoader:
    """
    Manages train/val/test splits, batching, and caching for MRS datasets based on BaseMRSDataset.
    Supports both random and deterministic sampling modes.
    """

    def __init__(self, data, water=None, batch_size=16, seed=0, val_frac=0.1, test_frac=0.1,
                 n_coils=(1, None), n_averages=(1, None), pipelines=None, cache_det=False,
                 sampling_mode=None, to_tensor=True, **kwargs):
        """
        Initialization.

        Args:
            data (List): List of subjects (NiftiMRS objects).
            water (List, optional): List of water reference subjects (NiftiMRS objects
            batch_size (int): Number of samples per batch.
            seed (int): Random seed for reproducibility.
            val_frac (float): Fraction of data for validation set.
            test_frac (float): Fraction of data for test set.
            n_coils (tuple): Min/max number of coils to sample.
            n_averages (tuple): Min/max number of averages to sample.
            pipelines (dict, optional): Dictionary with 'train', 'val', 'test' keys mapping to
                                        AugmentationPipeline objects. If None, default pipelines are created.
            cache_det (bool): Whether to cache deterministic data for faster loading.
            sampling_mode (dict, optional): Dictionary with 'train', 'val', 'test' keys mapping to
                                            'random' or 'deterministic'. If None, defaults to
                                            {'train': 'random', 'val': 'deterministic', 'test': 'deterministic'}.
            to_tensor (bool): Whether to convert outputs to PyTorch tensors.
            **kwargs: Additional arguments, e.g., for SignalPerturber.
        """
        self.batch_size = batch_size
        self.n_coils = n_coils
        self.n_averages = n_averages
        self.sampling_mode = {'train': 'random',
                              'val': 'deterministic',
                              'test': 'deterministic'} \
            if sampling_mode is None else sampling_mode
        self.to_tensor = to_tensor

        # if peturber is in kwargs, pass it to pipeline creation
        if 'perturber_args' in kwargs:
            perturber_args = kwargs.pop('perturber_args', {})
        else:
            perturber_args = {
                'amp_mean': 0.0,
                'amp_var_low': 0.0, 'amp_var_high': 0.0,
                'phase_low': 0.0, 'phase_high': 0.0,
                'freq_low': 0.0, 'freq_high': 0.0,
                'misalign': False
            }

        # split into train/val/test
        self.splits = SubjectSplitter(data, water, seed=seed, val_frac=val_frac,
                                      test_frac=test_frac).split()

        # build pipelines for each mode if needed
        if pipelines is None:
            self.pipelines = {}
            for mode in ['train', 'val', 'test']:
                steps = [CoilAverageSampler(mode=self.sampling_mode[mode],
                                            n_coils=n_coils, n_averages=n_averages),
                         RawProcessor(**kwargs)]
                if mode == 'train':
                    steps.append(NoisePerturber(**perturber_args))
                self.pipelines[mode] = AugmentationPipeline(steps)
        else:
            self.pipelines = pipelines

        # cache deterministic data for all modes
        if cache_det:
            self.cached_data = {}
            for mode in ['train', 'val', 'test']:
                if self.sampling_mode[mode] == 'deterministic':
                    data, water = self.splits[mode]
                    dataset = BaseMRSDataset(data=data, water=water, mode='deterministic',
                                             pipeline=self.pipelines[mode], n_coils=self.n_coils,
                                             n_averages=self.n_averages, to_tensor=self.to_tensor)
                    self.cached_data[mode] = dataset
                else:
                    self.cached_data[mode] = None
        else:
            self.cached_data = {mode: None for mode in ['train', 'val', 'test']}

    def select_with_mode(self, mode, shuffle=False):
        """
        Selects the appropriate DataLoader, generator, or raw list based on the sampling mode.

        Args:
            mode (str): 'train', 'val', or 'test'.
            shuffle (bool): Whether to shuffle the data (only for deterministic mode).
            as_dataloader (bool): If True, returns a PyTorch DataLoader. If False, returns raw data.
        """
        if self.cached_data[mode] is not None:
            dataset = self.cached_data[mode]
        else:
            data, water = self.splits[mode]
            dataset = BaseMRSDataset(
                data=data, water=water, mode=self.sampling_mode[mode],
                pipeline=self.pipelines[mode], n_coils=self.n_coils,
                n_averages=self.n_averages, to_tensor=self.to_tensor
            )

        if self.sampling_mode[mode] == 'random':
            def infinite_gen():
                while True:
                    yield next(dataset.iter_batches(batch_size=self.batch_size))
            return infinite_gen()

        elif self.sampling_mode[mode] == 'deterministic':
            if self.to_tensor:
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
            else:
                data, water = zip(*[dataset[i] for i in range(len(dataset))])
                return list(data), list(water)
        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling_mode[mode]}")

    def train_dataloader(self):
        """ Returns the training DataLoader or generator/raw list. """
        return self.select_with_mode('train', shuffle=True)

    def val_dataloader(self):
        """ Returns the validation DataLoader or generator/raw list. """
        return self.select_with_mode('val', shuffle=False)

    def test_dataloader(self):
        """ Returns the test DataLoader or generator/raw list. """
        return self.select_with_mode('test', shuffle=False)