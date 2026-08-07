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

from fsl_mrs.core.nifti_mrs import split

# own
from augmentrum.processing.utils import update_processing_prov
from augmentrum.core.base_module import BaseModule
from augmentrum.core import Backend


#**************************************************************************************************#
#                                     Class CoilAverageSampler                                     #
#**************************************************************************************************#
#                                                                                                  #
# Samples coils and averages from raw MRS data.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class CoilAverageSampler(BaseModule):
    """
    Samples coils and averages from raw MRS data.

    Uses FSL-MRS `split` on NIfTI-MRS objects, so this module operates
    natively on the NIFTI_LIST backend.  For any other backend the base
    class automatically routes the call through process_nifti_list and
    returns the result in the original backend format.
    """

    SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

    def __init__(self, mode='random', n_coils=(1, None), n_averages=(1, None), reweight=False):
        """
        Initializes the sampler.

        Args:
            mode (str): 'random' or 'deterministic' sampling mode.
            n_coils (tuple or int): Min/max number of coils to sample, or exact value.
                                   Examples: (1, 8) = range, 4 = exactly 4
            n_averages (tuple or int): Min/max number of averages to sample, or exact value.
            reweight (bool): Whether to reweight signals after sampling.
        """
        super().__init__()

        self.mode = mode

        # Normalize to tuples: int → (value, value), tuple → tuple
        self.n_coils = self._normalize_param(n_coils)
        self.n_averages = self._normalize_param(n_averages)
        self.reweight = reweight

    def _normalize_param(self, param):
        """
        Normalize parameter to tuple format.

        Args:
            param: Either tuple (min, max) or int/float (exact value)

        Returns:
            Tuple (min, max)

        Examples:
            4 → (4, 4)  # Exactly 4
            (1, 8) → (1, 8)  # Range 1-8
            None → (1, None)  # Default
        """
        if param is None:
            return (1, None)
        elif isinstance(param, (int, float)):
            # Single value → exact value
            return (int(param), int(param))
        elif isinstance(param, tuple):
            return param
        else:
            raise TypeError(f"Parameter must be int, float, or tuple, got {type(param)}")


    def process_nifti_list(self, data_list, water_list=None, coil_indices=None, average_indices=None, **kwargs):
        """
        Samples coils and averages from each NIFTI_MRS in the list.

        Args:
            data_list: List of metabolite MRS data (NIFTI_MRS objects).
            water_list: List of water reference MRS data (NIFTI_MRS objects), optional.
            coil_indices: List of coil indices to select (for deterministic mode).
            average_indices: List of average indices to select (for deterministic mode).
            **kwargs: Additional arguments.

        Returns:
            Tuple of (processed_data_list, processed_water_list)
        """
        processed_data = []
        processed_water = []

        for i, data_met in enumerate(data_list):
            data_wat = water_list[i] if water_list is not None else None

            # Sample coils and averages for this subject
            data_met, data_wat = self._process_single(data_met, data_wat, coil_indices, average_indices)

            processed_data.append(data_met)
            if water_list is not None:
                processed_water.append(data_wat if data_wat is not None else water_list[i])

        return processed_data, (processed_water if water_list is not None else None)

    def _process_single(self, data_met, data_wat=None, coil_indices=None, average_indices=None):
        """
        Samples coils and averages from a single subject's data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional.
            coil_indices: List of coil indices to select (for deterministic mode).
            average_indices: List of average indices to select (for deterministic mode).
        """
        data_met, data_wat = self.sample_coils(data_met, data_wat, coil_indices)
        data_met, data_wat = self.sample_averages(data_met, data_wat, average_indices)

        if self.reweight:
            data_met, data_wat = self._reweight_signals(data_met, data_wat)
        return data_met, data_wat

    def sample_coils(self, data_met, data_wat=None, coil_indices=None):
        """
        Samples specified coils from the data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional
            coil_indices: List of coil indices to select (if provided, use deterministic mode).
        """
        dim_tags = getattr(data_met, 'dim_tags', [])
        has_coil = 'DIM_COIL' in dim_tags

        # Determine mode at runtime: if coil_indices provided, use them (deterministic)
        # Otherwise, sample randomly (if mode='random') or use all (if mode='deterministic' but no indices)
        if coil_indices is not None:
            # Deterministic mode: use provided indices
            pass  # Will apply indices below
        elif self.mode == 'random' and has_coil:
            # Random mode: sample random coils
            coil_dim = data_met.dim_position('DIM_COIL')
            n_total  = data_met.shape[coil_dim]          # actual coil count
            min_c, max_c = self._get_limits(self.n_coils, n_total)
            if min_c < max_c:
                rng = self.rng.numpy_rng()
                num_coils   = int(rng.integers(min_c, max_c))
                coil_indices = rng.permutation(n_total)[:num_coils].tolist()
        elif self.mode == 'deterministic' and has_coil:
            # Deterministic mode but no indices provided: use all coils
            coil_dim = data_met.dim_position('DIM_COIL')
            coil_indices = list(range(data_met.shape[coil_dim]))

        # apply sampling
        if has_coil and coil_indices is not None:
            if isinstance(coil_indices, list) or (isinstance(coil_indices, int) and coil_indices > 0):
                _, data_met = split(data_met, 'DIM_COIL', coil_indices)
                if data_wat is not None:
                    _, data_wat = split(data_wat, 'DIM_COIL', coil_indices)

            # update processing provenance
            processing_info = f'{__name__}.sample_coils, '
            processing_info += f'coil_indices={coil_indices}.'
            update_processing_prov(data_met, 'Random Coil Sampling', processing_info)
            if data_wat is not None:
                update_processing_prov(data_wat, 'Random Coil Sampling', processing_info)

        return data_met, data_wat

    def sample_averages(self, data_met, data_wat=None, average_indices=None):
        """
        Samples specified averages from the data.

        Args:
            data_met: Metabolite MRS data (NiftiMRS object).
            data_wat: Water reference MRS data (NiftiMRS object), optional.
            average_indices: List of average indices to select.
        """
        dim_tags = getattr(data_met, 'dim_tags', [])
        has_dyn = 'DIM_DYN' in dim_tags

        # Determine mode at runtime: if average_indices provided, use them (deterministic)
        # Otherwise, sample randomly (if mode='random') or use all (if mode='deterministic' but no indices)
        if average_indices is not None:
            # Deterministic mode: use provided indices
            pass  # Will apply indices below
        elif self.mode == 'random' and has_dyn:
            # Random mode: sample random averages
            dyn_dim = data_met.dim_position('DIM_DYN')
            n_total = data_met.shape[dyn_dim]            # actual average count
            min_a, max_a = self._get_limits(self.n_averages, n_total)
            if min_a < max_a:
                rng = self.rng.numpy_rng()
                num_averages    = int(rng.integers(min_a, max_a))
                average_indices = rng.permutation(n_total)[:num_averages].tolist()
        elif self.mode == 'deterministic' and has_dyn:
            # Deterministic mode but no indices provided: use all averages
            dyn_dim = data_met.dim_position('DIM_DYN')
            average_indices = list(range(data_met.shape[dyn_dim]))

        # apply sampling
        if has_dyn and average_indices is not None:
            if isinstance(average_indices, list) or (isinstance(average_indices, int) and average_indices > 0):
                _, data_met = split(data_met, 'DIM_DYN', average_indices)

            # update processing provenance
            processing_info = f'{__name__}.sample_averages, '
            processing_info += f'coil_indices={average_indices}.'
            update_processing_prov(data_met, 'Random Average Sampling', processing_info)

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

    def _sample_snr_improvement(self, data, water):
        """
        Samples coils and averages to improve SNR.
        TODO: implement SNR improvement sampling.

        Args:
            data: MRS data (NiftiMRS object).
            water: Water reference MRS data (NiftiMRS object), optional.
        """
        raise NotImplementedError("SNR improvement sampling not implemented yet.")

    def _get_limits(self, n, n_total):
        """
        Return "(min_n, max_exclusive)" for an exclusive-upper-bound draw.

        Args:
            n (tuple): "(min, max)" — both *inclusive*, either can be "None".
                       "None" on min → 1.  "None" on max → "n_total" (use all).
            n_total (int): Total number of items available (coils / averages).

        Examples::

            _get_limits((1, None), 4)  →  (1, 5)   # randint → 1, 2, 3 or 4  ✓
            _get_limits((2, 3),    4)  →  (2, 4)   # randint → 2 or 3         ✓
            _get_limits((4, 4),    4)  →  (4, 5)   # randint → always 4       ✓
            _get_limits((6, None), 4)  →  (4, 5)   # clamps to available      ✓
        """
        if n is None:
            return 1, n_total + 1

        min_val, max_val = n

        min_n = 1 if min_val is None else max(1, min(int(min_val), n_total))
        max_n = n_total if max_val is None else min(int(max_val), n_total)

        # +1 because torch.randint upper bound is exclusive
        return min_n, max_n + 1
