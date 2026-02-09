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

from typing import List, Optional, Tuple
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.core.base_module import BaseModule
import warnings


#**************************************************************************************************#
#                                    Class AugmentationPipeline                                    #
#**************************************************************************************************#
#                                                                                                  #
# Chains multiple augmentation/processing steps with backend compatibility checking.               #
#                                                                                                  #
#**************************************************************************************************#
class AugmentationPipeline:
    """
    Chains multiple augmentation steps in sequence with backend compatibility.

    The pipeline:
    1. Checks each module's supported backends
    2. Automatically converts data to compatible backend if needed
    3. Applies each step in sequence
    4. Provides warnings if conversions are needed
    """

    def __init__(self, steps: List[BaseModule]):
        """
        Initializes the pipeline with a list of augmentation steps.

        Args:
            steps: List of augmentation step instances (must inherit from BaseModule).
        """
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        """Validate that all steps are BaseModule instances."""
        for i, step in enumerate(self.steps):
            if not isinstance(step, BaseModule):
                warnings.warn(
                    f"Step {i} ({type(step).__name__}) does not inherit from BaseModule. "
                    f"Backend compatibility checking will be skipped for this step."
                )

    def __call__(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None,
                 **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Applies the augmentation steps in sequence to the data.

        Args:
            data: Input MRS data as NIfTI_MRS_Plus object or NIFTI_MRS object.
            water: Optional water reference as NIfTI_MRS_Plus object or NIFTI_MRS object.
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
            # Check backend compatibility
            if isinstance(step, BaseModule):
                current_backend = current_data.backend

                # Ensure backend is a Backend enum
                if isinstance(current_backend, str):
                    from augmentrum.core.nifti_mrs_plus import Backend
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
