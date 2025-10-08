####################################################################################################
#                                   coil_average_sampler.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Implements sampling of coils and averages from raw MRS data.                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from fsl_mrs.core.nifti_mrs import split


#**************************************************************************************************#
#                                        Class CoilAverageSampler                                  #
#**************************************************************************************************#
#                                                                                                  #
# Samples coils and averages from raw MRS data. Supports random/deterministic selection            #
# and reweighting.                                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class CoilAverageSampler:
    """
    Samples coils and averages from raw MRS data.
    """

    def __init__(self, mode='random', n_coils=(1, None), n_averages=(1, None), reweight=False):
        """
        Initializes the sampler.

        Args:
            mode (str): 'random' or 'deterministic' sampling mode.
            n_coils (tuple): Min/max number of coils to sample.
            n_averages (tuple): Min/max number of averages to sample.
            reweight (bool): Whether to reweight signals after sampling.
        """
        self.mode = mode
        self.n_coils = n_coils
        self.n_averages = n_averages
        self.reweight = reweight

    def __call__(self, data_met, data_wat=None, coil_indices=None, average_indices=None, **kwargs):
        """
        Samples coils and averages from the data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional.
            coil_indices: List of coil indices to select (for deterministic mode).
            average_indices: List of average indices to select (for deterministic mode).
            **kwargs: Additional arguments.
        """
        dim_tags = getattr(data_met, 'dim_tags', [])
        has_coil = 'DIM_COIL' in dim_tags
        has_dyn = 'DIM_DYN' in dim_tags

        # sampling logic
        if self.mode == 'random':
            if coil_indices is not None or average_indices is not None:
                raise ValueError("coil_indices and average_indices should be None in random mode")

            if has_coil:
                coil_dim = data_met.dim_position('DIM_COIL')
                min_c, max_c = self._get_limits(self.n_coils, data_met.shape[coil_dim] - 1)
                if min_c < max_c:
                    num_coils = torch.randint(min_c, max_c, (1,)).item()
                    coil_indices =  torch.randperm(data_met.shape[coil_dim])[:num_coils].tolist()

            if has_dyn:
                dyn_dim = data_met.dim_position('DIM_DYN')
                min_a, max_a = self._get_limits(self.n_averages, data_met.shape[dyn_dim] - 1)
                if min_a < max_a:
                    num_averages = torch.randint(min_a, max_a, (1,)).item()
                    average_indices = torch.randperm(data_met.shape[dyn_dim])[:num_averages].tolist()

        elif self.mode == 'deterministic':
            assert coil_indices is not None or average_indices is not None

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # apply sampling
        if has_coil and coil_indices is not None:
            if isinstance(coil_indices, list) or (isinstance(coil_indices, int) and coil_indices > 0):
                _, data_met = split(data_met, 'DIM_COIL', coil_indices)
                if data_wat is not None:
                    _, data_wat = split(data_wat, 'DIM_COIL', coil_indices)
        if has_dyn and average_indices is not None:
            if isinstance(average_indices, list) or (isinstance(average_indices, int) and average_indices > 0):
                _, data_met = split(data_met, 'DIM_DYN', average_indices)

        if self.reweight:
            data_met, data_wat = self._reweight_signals(data_met, data_wat)
        return data_met, data_wat

    def _reweight_signals(self, data, water):
        """
        Reweights coils and averages based on random weights.
        TODO: implement reweighting.

        Args:
            data: MRS data (NiftiMRS object).
            water: Water reference MRS data (NiftiMRS object), optional.
        """
        raise NotImplementedError("Reweighting not implemented yet.")

    def _get_limits(self, n, n_max):
        """
        Computes valid min/max limits for sampling.

        Args:
            n (tuple): (min, max) tuple where each can be None.
            n_max (int): Maximum allowable value.
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