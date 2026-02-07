####################################################################################################
#                                        base_module.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-06                                                                              #
#                                                                                                  #
# Purpose: Unified base class for all pipeline modules - handles backends, logging, everything     #
#                                                                                                  #
####################################################################################################

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from augmentrum.core import NIfTI_MRS_Plus, Backend


class BaseModule(ABC):
    """
    Unified base class for ALL augmentation/processing pipeline modules.

    Handles:
    - Multiple backend support (NIfTI list, NumPy, PyTorch, TensorFlow, JAX, Keras)
    - Automatic backend detection and conversion
    - Provenance logging (only if not volatile)
    - Smart dispatching to appropriate processing method

    Subclasses just implement the methods they support:
    - process_nifti_list() for FSL-MRS style processing
    - process_tensor() for array/tensor operations
    - forward() as alias (for compatibility)
    """

    # Subclasses can override - if empty, module supports all backends
    SUPPORTED_BACKENDS: List[Backend] = []

    def __init__(self, **kwargs):
        """
        Initialize module with optional parameters.

        Args:
            **kwargs: Module-specific parameters, stored in self.params
        """
        self.params = kwargs

    def __call__(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None,
                 **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Main entry point - handles everything automatically.

        - Checks backend compatibility
        - Routes to appropriate processing method
        - Handles logging/provenance (if not volatile)
        - Returns NIfTI_MRS_Plus objects
        """
        # Determine which method to use based on backend
        backend = data.backend

        # Check if we support this backend (if SUPPORTED_BACKENDS specified)
        if self.SUPPORTED_BACKENDS and backend not in self.SUPPORTED_BACKENDS:
            # Use first supported backend
            backend = self.SUPPORTED_BACKENDS[0] if self.SUPPORTED_BACKENDS else Backend.NIFTI_LIST

        # Check which methods are actually implemented (overridden from BaseModule)
        has_process_nifti_list = (
            self.__class__.process_nifti_list is not BaseModule.process_nifti_list
        )
        has_process_tensor = (
            self.__class__.process_tensor is not BaseModule.process_tensor
        )
        has_forward = (
            self.__class__.forward is not BaseModule.forward
        )

        # Try methods in order of preference
        if backend == Backend.NIFTI_LIST:
            # Use NIfTI list processing
            if has_process_nifti_list:
                return self._process_via_nifti_list(data, water, **kwargs)
            elif has_forward:
                return self._process_via_forward(data, water, **kwargs)
        else:
            # Use tensor processing
            if has_process_tensor:
                return self._process_via_tensor(data, water, backend, **kwargs)
            elif has_forward:
                return self._process_via_forward(data, water, **kwargs)

        # Fallback to forward if nothing else works
        if has_forward:
            return self._process_via_forward(data, water, **kwargs)

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement at least one of: "
            f"process_nifti_list(), process_tensor(), or forward()"
        )

    def _process_via_nifti_list(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus],
                                **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """Process using NIfTI list method."""
        # Get lists
        data_list = data.list()
        water_list = water.list() if water is not None else None

        # Process
        processed_data, processed_water = self.process_nifti_list(
            data_list, water_list, **kwargs
        )

        # Log provenance (only if not volatile)
        operation_name = self.__class__.__name__
        operation_details = {'method': 'process_nifti_list', 'params': self.params, **kwargs}

        # Wrap back into NIfTI_MRS_Plus
        data_out = NIfTI_MRS_Plus(
            nifti_list=processed_data,
            backend=data.backend,
            volatile=data.volatile
        )

        # Update metadata if not volatile
        if not data.volatile:
            data_out.update_metadata(operation_name, operation_details)

        water_out = None
        if processed_water is not None:
            water_out = NIfTI_MRS_Plus(
                nifti_list=processed_water,
                backend=water.backend if water else Backend.NIFTI_LIST,
                volatile=water.volatile if water else data.volatile
            )
            if water and not water.volatile:
                water_out.update_metadata(operation_name, operation_details)

        return data_out, water_out

    def _process_via_tensor(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus],
                           backend: Backend, **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """Process using tensor method."""
        # Get data in backend format
        data_array = data.get_data(backend)
        water_array = water.get_data(backend) if water is not None else None

        # Process
        processed_data, processed_water = self.process_tensor(
            data_array, water_array, backend=backend, **kwargs
        )

        # Log provenance
        operation_name = self.__class__.__name__
        operation_details = {'method': 'process_tensor', 'backend': backend.value,
                           'params': self.params, **kwargs}

        # For tensor processing, we need to reconstruct NIfTI_MRS_Plus
        # For now, keep same nifti_list but note processing was done
        data_out = NIfTI_MRS_Plus(
            nifti_list=data.list(),  # Keep original list structure
            backend=backend,
            volatile=data.volatile
        )

        if not data.volatile:
            data_out.update_metadata(operation_name, operation_details)

        water_out = None
        if water is not None:
            water_out = NIfTI_MRS_Plus(
                nifti_list=water.list(),
                backend=backend,
                volatile=water.volatile
            )
            if not water.volatile:
                water_out.update_metadata(operation_name, operation_details)

        return data_out, water_out

    def _process_via_forward(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus],
                            **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """Process using legacy forward method (for compatibility)."""
        # Get appropriate data format based on backend
        if data.backend == Backend.NIFTI_LIST:
            data_in = data.list()
            water_in = water.list() if water is not None else None
        else:
            data_in = data.get_data()
            water_in = water.get_data() if water is not None else None

        # Process using forward
        processed_data, processed_water = self.forward(data_in, water_in, **kwargs)

        # Log provenance
        operation_name = self.__class__.__name__
        operation_details = {'method': 'forward', 'params': self.params, **kwargs}

        # Wrap results
        if isinstance(processed_data, list):
            data_out = NIfTI_MRS_Plus(
                nifti_list=processed_data,
                backend=data.backend,
                volatile=data.volatile
            )
        else:
            # If forward returns something else, keep original structure
            data_out = data.copy()

        if not data.volatile:
            data_out.update_metadata(operation_name, operation_details)

        water_out = None
        if processed_water is not None:
            if isinstance(processed_water, list):
                water_out = NIfTI_MRS_Plus(
                    nifti_list=processed_water,
                    backend=water.backend if water else Backend.NIFTI_LIST,
                    volatile=water.volatile if water else data.volatile
                )
            else:
                water_out = water.copy() if water else None

            if water_out and not water_out.volatile:
                water_out.update_metadata(operation_name, operation_details)

        return data_out, water_out

    def supports_backend(self, backend: Backend) -> bool:
        """Check if this module supports the given backend."""
        if not self.SUPPORTED_BACKENDS:
            return True  # Supports all if not specified
        return backend in self.SUPPORTED_BACKENDS

    def get_preferred_backend(self) -> Backend:
        """Get the preferred backend for this module."""
        if self.SUPPORTED_BACKENDS:
            return self.SUPPORTED_BACKENDS[0]
        return Backend.NIFTI_LIST

    # Subclasses can implement these methods as needed
    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None,
                          **kwargs) -> Tuple[List, Optional[List]]:
        """
        Process lists of NIfTI-MRS objects (for FSL-MRS style processing).

        Override this if your module uses FSL-MRS functions.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement process_nifti_list()"
        )

    def process_tensor(self, data_array, water_array=None, backend: Backend = Backend.NUMPY,
                      **kwargs) -> Tuple:
        """
        Process tensor/array data (for vectorized operations).

        Override this if your module uses NumPy/PyTorch/etc. operations.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement process_tensor()"
        )

    def forward(self, data, water=None, **kwargs):
        """
        Legacy processing method (for compatibility with old modules).

        Override this for simple modules that don't need backend-specific logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement forward()"
        )

    def __repr__(self) -> str:
        backends = [b.value for b in self.SUPPORTED_BACKENDS] if self.SUPPORTED_BACKENDS else ['all']
        return f"{self.__class__.__name__}(backends={backends})"




