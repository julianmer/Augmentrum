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

# own
from augmentrum.processing.utils import update_processing_prov
from augmentrum.core.base_module import BaseModule
from augmentrum.core import Backend


#**************************************************************************************************#
#                                        Class CoilAverageSampler                                  #
#**************************************************************************************************#
#                                                                                                  #
# Samples coils and averages from raw MRS data. Supports random/deterministic selection            #
# and reweighting.                                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class CoilAverageSampler(BaseModule):
    """
    Samples coils and averages from raw MRS data.

    Supports all backends by using NIfTI-MRS metadata to identify dimensions,
    then efficiently sampling from the underlying data (whether NIFTI objects or tensors).

    For non-NIFTI_LIST backends, automatically converts to NIFTI_LIST for dimension
    inspection, performs sampling, then converts back to original backend.

    This ensures compatibility with all backends while maintaining efficiency.
    """

    SUPPORTED_BACKENDS = [Backend.NIFTI_LIST, Backend.NUMPY, Backend.PYTORCH,
                         Backend.TENSORFLOW, Backend.KERAS, Backend.JAX]

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
        super().__init__(mode=mode, n_coils=n_coils, n_averages=n_averages, reweight=reweight)

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

    def __call__(self, data, water=None, **kwargs):
        """
        Process data with automatic backend conversion.

        If backend is not NIFTI_LIST, automatically converts to NIFTI_LIST,
        processes (using NIfTI-MRS metadata for dimension info), then converts back.

        This allows efficient processing while maintaining backend compatibility.
        """
        from augmentrum.core import NIfTI_MRS_Plus

        if not isinstance(data, NIfTI_MRS_Plus):
            # Plain list - wrap it
            data = NIfTI_MRS_Plus(nifti_list=data, backend=Backend.NIFTI_LIST)

        # Store original backend
        original_backend = data.backend
        original_volatile = data.volatile

        # Convert to NIFTI_LIST if needed (for dimension inspection)
        if original_backend != Backend.NIFTI_LIST:
            # Convert to NIFTI_LIST for processing
            data = NIfTI_MRS_Plus(
                nifti_list=data.to_nifti_list(),
                backend=Backend.NIFTI_LIST,
                volatile=original_volatile
            )
            if water is not None:
                water = NIfTI_MRS_Plus(
                    nifti_list=water.to_nifti_list(),
                    backend=Backend.NIFTI_LIST,
                    volatile=original_volatile
                )

        # Process using parent class method (calls process_nifti_list)
        result_data, result_water = super().__call__(data, water, **kwargs)

        # Convert back to original backend if needed
        if original_backend != Backend.NIFTI_LIST:
            result_data = NIfTI_MRS_Plus(
                nifti_list=result_data.to_nifti_list(),
                backend=original_backend,
                volatile=original_volatile
            )
            if result_water is not None:
                result_water = NIfTI_MRS_Plus(
                    nifti_list=result_water.to_nifti_list(),
                    backend=original_backend,
                    volatile=original_volatile
                )

        return result_data, result_water

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
            min_c, max_c = self._get_limits(self.n_coils, data_met.shape[coil_dim] - 1)
            if min_c < max_c:
                num_coils = torch.randint(min_c, max_c, (1,)).item()
                coil_indices = torch.randperm(data_met.shape[coil_dim])[:num_coils].tolist()
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
            min_a, max_a = self._get_limits(self.n_averages, data_met.shape[dyn_dim] - 1)
            if min_a < max_a:
                num_averages = torch.randint(min_a, max_a, (1,)).item()
                average_indices = torch.randperm(data_met.shape[dyn_dim])[:num_averages].tolist()
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