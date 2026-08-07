####################################################################################################
#                                      pipeline.py                                                 #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-07                                                                              #
#                                                                                                  #
# Purpose: Defines AugmentationPipeline, a modular class to chain multiple augmentations           #
#          in sequence with automatic backend compatibility handling.                              #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from typing import List, Optional, Tuple
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.core.base_module import BaseModule
import warnings


#**************************************************************************************************#
#                                    Class AugmentationPipeline                                    #
#**************************************************************************************************#
#                                                                                                  #
# Chains multiple augmentation steps in sequence with backend compatibility.                       #
#                                                                                                  #
#**************************************************************************************************#
class AugmentationPipeline:
    """
    Chains multiple augmentation steps in sequence with backend compatibility.

    The pipeline:
    1. Stores user parameters (which may be tuples for ranges)
    2. Samples parameters per-batch (for on-the-fly mode)
    3. Injects sampled values into modules temporarily
    4. Checks each module's supported backends
    5. Automatically converts data to compatible backend if needed
    6. Applies each step in sequence
    7. Provides warnings if conversions are needed
    
    Uses introspection to automatically discover module parameters!
    """

    def __init__(self, steps: List[BaseModule], module_names=None, user_kwargs=None):
        """
        Initializes the pipeline with a list of augmentation steps.

        Args:
            steps: List of augmentation step instances (must inherit from BaseModule).
            module_names: List of module name strings (e.g., ['phase', 'noise'])
            user_kwargs: Dictionary of all user-provided parameters
        """
        self.steps = steps
        self.module_names = module_names or []
        self.user_kwargs = user_kwargs or {}
        self._validate_steps()
        
        # Store parameters for each module (extracted from user_kwargs using introspection)
        self.module_params = self._extract_module_params()

    def _extract_module_params(self):
        """
        Extract relevant parameters for each module from user kwargs.
        
        Uses introspection to automatically discover what parameters each module accepts,
        then extracts matching values from user_kwargs.
        
        This is flexible - no need to maintain a hardcoded PARAM_MAPPING!
        """
        import inspect
        
        module_params = {}
        
        for idx, step in enumerate(self.steps):
            # Get the module's __init__ signature
            try:
                sig = inspect.signature(step.__class__.__init__)
                # Get parameter names (excluding 'self' and 'kwargs')
                param_names = [
                    p.name for p in sig.parameters.values()
                    if p.name not in ['self', 'kwargs', 'args']
                ]
            except Exception:
                # Fallback: try to get from step.params if it exists
                param_names = []
                if hasattr(step, 'params'):
                    param_names = list(step.params.keys())
            
            # Extract matching parameters from user_kwargs
            params = {}
            for param_name in param_names:
                if param_name in self.user_kwargs:
                    params[param_name] = self.user_kwargs[param_name]
            
            if params:
                module_params[idx] = params
        
        return module_params

    def _validate_steps(self):
        """Validate that all steps are BaseModule instances."""
        for i, step in enumerate(self.steps):
            if not isinstance(step, BaseModule):
                warnings.warn(
                    f"Step {i} ({type(step).__name__}) does not inherit from BaseModule. "
                    f"Backend compatibility checking will be skipped for this step."
                )

    def sample_batch_parameters(self, batch_size: int):
        """
        Sample parameters for an entire batch (one value per sample).

        For on-the-fly mode with batch_size=16, this samples 16 different
        values for each parameter that has a range.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            Dictionary mapping step index to parameter dictionaries
            Format: {step_idx: {param_name: sampled_value}}
        """
        # Get distribution settings from user_kwargs
        global_distribution = self.user_kwargs.get('param_distribution', 'uniform')
        per_param_distributions = self.user_kwargs.get('param_distributions', {})

        batch_params = {}

        # Sample from stored module_params (extracted from user_kwargs)
        for step_idx, params in self.module_params.items():
            step_params = {}
            
            for param_name, param_value in params.items():
                # Get distribution for this parameter
                distribution = per_param_distributions.get(param_name, global_distribution)
                
                # Sample value
                sampled_val = self._sample_from_range(param_value, distribution)
                step_params[param_name] = sampled_val

            if step_params:
                batch_params[step_idx] = step_params

        return batch_params
    
    def _sample_from_range(self, param, distribution: str = 'uniform'):
        """
        Sample a scalar value from a parameter (range or scalar).

        Args:
            param: Either scalar (float/int) or tuple (min, max) for range
            distribution: 'uniform', 'gaussian', 'normal', 'exponential', 'beta'

        Returns:
            Scalar value
        """
        import numpy as np

        # If already scalar, return as-is
        if isinstance(param, (int, float)):
            return float(param)

        # If None, return None
        if param is None:
            return None

        # If tuple, sample based on distribution
        if isinstance(param, tuple) and len(param) == 2:
            min_val, max_val = param

            if min_val is None and max_val is None:
                return None
            if min_val is None:
                min_val = 0.0
            if max_val is None:
                max_val = min_val * 2.0

            # Sample based on distribution
            if distribution == 'uniform':
                return np.random.uniform(min_val, max_val)

            elif distribution in ['gaussian', 'normal']:
                # Gaussian centered at midpoint, std = range/6 (99.7% within range)
                mean = (min_val + max_val) / 2.0
                std = (max_val - min_val) / 6.0
                value = np.random.normal(mean, std)
                return np.clip(value, min_val, max_val)

            elif distribution == 'exponential':
                # Exponential biased toward min_val
                scale = (max_val - min_val) / 3.0
                value = min_val + np.random.exponential(scale)
                return np.clip(value, min_val, max_val)

            elif distribution == 'beta':
                # Beta distribution (slightly biased to center)
                alpha, beta = 2.0, 2.0
                value = np.random.beta(alpha, beta)
                return min_val + value * (max_val - min_val)

            else:
                # Default to uniform
                return np.random.uniform(min_val, max_val)

        # If single-element tuple
        if isinstance(param, tuple) and len(param) == 1:
            return float(param[0])

        # Otherwise return as-is
        return param

    def __call__(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None,
                 batch_params=None, **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Applies the augmentation steps in sequence to the data.

        Args:
            data: Input MRS data as NIfTI_MRS_Plus object (can be a batch)
            water: Optional water reference as NIfTI_MRS_Plus object
            batch_params: Optional dict of sampled parameters per step (for on-the-fly mode)
            **kwargs: Additional arguments passed to each step.

        Returns:
            Tuple of (processed_data, processed_water)
        """
        from fsl_mrs.core.nifti_mrs import NIFTI_MRS

        # Auto-wrap single NIFTI_MRS objects to NIfTI_MRS_Plus
        if isinstance(data, NIFTI_MRS):
            data = NIfTI_MRS_Plus([data], volatile=True)
        if isinstance(water, NIFTI_MRS):
            water = NIfTI_MRS_Plus([water], volatile=True)

        current_data = data
        current_water = water

        for i, step in enumerate(self.steps):
            # Inject batch parameters if provided (for on-the-fly mode)
            if batch_params and i in batch_params:
                # Temporarily set sampled parameters on the step
                original_attrs = {}
                for param_name, param_value in batch_params[i].items():
                    if hasattr(step, param_name):
                        original_attrs[param_name] = getattr(step, param_name)
                    setattr(step, param_name, param_value)

                try:
                    # Check backend compatibility (only for BaseModule instances)
                    if isinstance(step, BaseModule):
                        current_backend = current_data.backend

                        # Ensure backend is a Backend enum
                        if isinstance(current_backend, str):
                            from nifti_mrs_plus import Backend
                            current_backend = Backend[current_backend.upper()]

                        if not step.supports_backend(current_backend):
                            # Need to convert to compatible backend
                            preferred_backend = step.get_preferred_backend()

                            warnings.warn(
                                f"Step {i} ({step.__class__.__name__}) does not support "
                                f"backend {current_backend.value}. Converting to {preferred_backend.value}. "
                                f"Supported backends: {[b.value for b in step.SUPPORTED_BACKENDS]}"
                            )

                            # Convert data
                            current_data = self._convert_backend(current_data, preferred_backend)
                            if current_water is not None:
                                current_water = self._convert_backend(current_water, preferred_backend)

                    # Apply the step with batch parameters
                    current_data, current_water = step(current_data, current_water, **kwargs)
                finally:
                    # Restore original attributes
                    for param_name, orig_value in original_attrs.items():
                        setattr(step, param_name, orig_value)
            else:
                # No batch params, use module's initialized values (fixed mode)
                # Check backend compatibility
                if isinstance(step, BaseModule):
                    current_backend = current_data.backend

                    # Ensure backend is a Backend enum
                    if isinstance(current_backend, str):
                        from nifti_mrs_plus import Backend
                        current_backend = Backend[current_backend.upper()]

                    if not step.supports_backend(current_backend):
                        # Need to convert to compatible backend
                        preferred_backend = step.get_preferred_backend()

                        warnings.warn(
                            f"Step {i} ({step.__class__.__name__}) does not support "
                            f"backend {current_backend.value}. Converting to {preferred_backend.value}. "
                            f"Supported backends: {[b.value for b in step.SUPPORTED_BACKENDS]}"
                        )

                        # Convert data
                        current_data = self._convert_backend(current_data, preferred_backend)
                        if current_water is not None:
                            current_water = self._convert_backend(current_water, preferred_backend)

                # Apply the step
                current_data, current_water = step(current_data, current_water, **kwargs)

        return current_data, current_water

    def _convert_backend(self, nifti_plus: NIfTI_MRS_Plus, target_backend: Backend) -> NIfTI_MRS_Plus:
        """
        Convert NIfTI_MRS_Plus to a different backend.

        Args:
            nifti_plus: Input NIfTI_MRS_Plus
            target_backend: Target backend

        Returns:
            New NIfTI_MRS_Plus with target backend
        """
        if nifti_plus.backend == target_backend:
            return nifti_plus

        # Get data in target format
        converted_data = nifti_plus.get_data(target_backend)

        if target_backend == Backend.NIFTI_LIST:
            # Data is already a list of NIFTI_MRS
            return NIfTI_MRS_Plus(
                nifti_list=converted_data,
                backend=target_backend,
                volatile=nifti_plus.volatile
            )
        else:
            # For tensor backends, create new NIfTI_MRS_Plus with same list but different backend
            return NIfTI_MRS_Plus(
                nifti_list=nifti_plus.list(),
                backend=target_backend,
                volatile=nifti_plus.volatile
            )

    def get_backend_sequence(self) -> List[str]:
        """
        Get the sequence of preferred backends for each step.
        Useful for debugging and optimization.

        Returns:
            List of backend names
        """
        backends = []
        for step in self.steps:
            if isinstance(step, BaseModule):
                backends.append(step.get_preferred_backend().value)
            else:
                backends.append("unknown")
        return backends

    def __repr__(self) -> str:
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"AugmentationPipeline(steps={step_names})"
