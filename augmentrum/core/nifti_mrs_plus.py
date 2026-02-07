####################################################################################################
#                                        nifti_mrs_plus.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-06                                                                              #
#                                                                                                  #
# Purpose: NIfTI-MRS+ wrapper for efficient multi-subject processing with multiple backends        #
#                                                                                                  #
####################################################################################################

import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from copy import deepcopy
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

from fsl_mrs.core.nifti_mrs import NIFTI_MRS


class Backend(Enum):
    """Supported data backends."""
    NIFTI_LIST = "nifti_list"  # List of NIFTI_MRS objects
    NUMPY = "numpy"             # NumPy arrays
    PYTORCH = "pytorch"         # PyTorch tensors
    TENSORFLOW = "tensorflow"   # TensorFlow tensors
    JAX = "jax"                 # JAX arrays
    KERAS = "keras"             # Keras tensors (backed by TF/JAX)


class NIfTI_MRS_Plus:
    """
    Handles batch-aware NIfTI_MRS data.

    This class wraps multiple 'NIFTI_MRS' objects into a batched representation.
    It provides access to the full batched tensor, header synchronization, and intuitive indexing.
    Supports multiple backends for different processing pipelines.

    Attributes:
        nifti_list (List[NIFTI_MRS]): List of individual subject MRS objects.
        backend (Backend): Backend for data representation (NIFTI_LIST, NumPy, PyTorch, etc.)
        volatile (bool): If True, skip metadata updates for speed

    Raises:
        TypeError: If any element in the list is not a NIFTI_MRS instance.
        ValueError: If all subjects don't have the same dim_tags.
    """

    def __init__(
        self,
        nifti_list: Union[List[NIFTI_MRS], 'NIfTI_MRS_Plus'],
        backend: Optional[Backend] = None,
        volatile: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the NIfTI_MRS_Plus object with data and header.

        Args:
            nifti_list: List of NIFTI_MRS objects or another NIfTI_MRS_Plus
            backend: Desired backend for pipeline processing (default: NIFTI_LIST)
            volatile: If True, skip metadata updates for speed
            metadata: Optional metadata dictionary
        """
        self.volatile = volatile
        self._backend = backend or Backend.NIFTI_LIST

        # Always store as list of NIfTI-MRS internally
        if isinstance(nifti_list, NIfTI_MRS_Plus):
            self.nifti_list = nifti_list.nifti_list
            if not volatile:
                self.metadata_common = deepcopy(nifti_list.metadata_common)
                self.metadata_individual = deepcopy(nifti_list.metadata_individual)
            else:
                self.metadata_common = {}
                self.metadata_individual = []
        elif isinstance(nifti_list, list) and len(nifti_list) > 0:
            # Check if list contains NIFTI-MRS-like objects (check for key attributes)
            first = nifti_list[0]
            if not (hasattr(first, 'shape') and hasattr(first, 'dwelltime')):
                error_msg = f"Data must be a list of NIFTI_MRS objects or NIfTI_MRS_Plus, got list of {type(first).__name__}"
                raise ValueError(error_msg)

            # Validate that all subjects have the same dim_tags
            first_dim_tags = first.dim_tags if hasattr(first, 'dim_tags') else None
            for i, nifti_obj in enumerate(nifti_list[1:], 1):
                obj_dim_tags = nifti_obj.dim_tags if hasattr(nifti_obj, 'dim_tags') else None
                if obj_dim_tags != first_dim_tags:
                    raise ValueError(f"All NIfTI-MRS objects must have the same dim_tags. "
                                     f"Subject 0 has {first_dim_tags}, but subject {i} has {obj_dim_tags}")

            self.nifti_list = nifti_list
            if not volatile:
                self._init_metadata(nifti_list, metadata)
            else:
                self.metadata_common = {}
                self.metadata_individual = []
        elif isinstance(nifti_list, list) and len(nifti_list) == 0:
            # Empty list - create empty NIfTI_MRS_Plus
            self.nifti_list = []
            self.metadata_common = {}
            self.metadata_individual = []
        else:
            error_msg = f"Data must be a list of NIFTI_MRS objects or NIfTI_MRS_Plus, got {type(nifti_list)}"
            if isinstance(nifti_list, list):
                if len(nifti_list) == 0:
                    error_msg += " (empty list)"
                else:
                    error_msg += f" (list of {type(nifti_list[0]).__name__})"
            raise ValueError(error_msg)

        # Track shape
        self.n_subjects = len(self.nifti_list)
        self._shape = (self.n_subjects,) + self.nifti_list[0].shape if self.n_subjects > 0 else (0,)

        # Cached batched data
        self._batched_data: Optional[np.ndarray] = None

    def _init_metadata(self, original_data: List[NIFTI_MRS], metadata: Optional[Dict] = None):
        """Initialize metadata from original NIfTI-MRS objects or provided dict."""
        if metadata is not None:
            self.metadata_common = metadata.get('common', {})
            self.metadata_individual = metadata.get('individual', [])
            return

        # Extract common and individual metadata from NIFTI_MRS objects
        self.metadata_common = {}
        self.metadata_individual = []

        # Common metadata (shared across all subjects)
        if len(original_data) > 0:
            first = original_data[0]
            self.metadata_common['dim_tags'] = first.dim_tags if hasattr(first, 'dim_tags') else []
            self.metadata_common['dwelltime'] = first.dwelltime if hasattr(first, 'dwelltime') else None
            self.metadata_common['spectrometer_frequency'] = first.spectrometer_frequency if hasattr(first, 'spectrometer_frequency') else None
            self.metadata_common['nucleus'] = first.nucleus if hasattr(first, 'nucleus') else None

            # Check if hdr_ext exists and extract common fields
            if hasattr(first, 'hdr_ext') and first.hdr_ext is not None:
                if 'EchoTime' in first.hdr_ext:
                    self.metadata_common['EchoTime'] = first.hdr_ext['EchoTime']
                if 'RepetitionTime' in first.hdr_ext:
                    self.metadata_common['RepetitionTime'] = first.hdr_ext['RepetitionTime']

        # Individual metadata (specific to each subject)
        for nifti_obj in original_data:
            individual_meta = {}
            if hasattr(nifti_obj, 'hdr_ext') and nifti_obj.hdr_ext is not None:
                if 'ProcessingApplied' in nifti_obj.hdr_ext:
                    individual_meta['ProcessingApplied'] = deepcopy(nifti_obj.hdr_ext['ProcessingApplied'])
                if 'ProcessingProvenance' in nifti_obj.hdr_ext:
                    individual_meta['ProcessingProvenance'] = deepcopy(nifti_obj.hdr_ext['ProcessingProvenance'])
            self.metadata_individual.append(individual_meta)

    @property
    def backend(self) -> Backend:
        """Current backend setting."""
        return self._backend

    @property
    def shape(self) -> tuple:
        """Return shape of the batched data (n_subjects, ...)."""
        return self._shape

    @property
    def dim_tags(self):
        """
        Return dimension tags from the first subject.
        All subjects are validated to have the same dim_tags during initialization.
        """
        if self.n_subjects > 0:
            return self.nifti_list[0].dim_tags
        return None

    @property
    def data(self):
        """
        Return data in the current backend format.
        For backward compatibility and pipeline use.
        """
        return self.get_data()

    def update_metadata(self, operation: str, details: Dict[str, Any],
                       individual_idx: Optional[List[int]] = None):
        """
        Update metadata with processing information.

        Args:
            operation: Name of operation applied
            details: Details of the operation
            individual_idx: If provided, update only these indices (for sample-specific ops)
        """
        if self.volatile:
            return  # Skip metadata updates in volatile mode

        from datetime import datetime

        provenance = {
            'Timestamp': datetime.now().isoformat(),
            'Operation': operation,
            'Details': str(details)
        }

        if individual_idx is None:
            # Common operation applied to all subjects
            if 'common_provenance' not in self.metadata_common:
                self.metadata_common['common_provenance'] = []
            self.metadata_common['common_provenance'].append(provenance)
        else:
            # Individual operations
            for idx in individual_idx:
                if idx < len(self.metadata_individual):
                    if 'ProcessingProvenance' not in self.metadata_individual[idx]:
                        self.metadata_individual[idx]['ProcessingProvenance'] = []
                    self.metadata_individual[idx]['ProcessingProvenance'].append(provenance)

    def copy(self) -> 'NIfTI_MRS_Plus':
        """Create a deep copy."""
        new_nifti_list = [nifti.copy() for nifti in self.nifti_list]

        return NIfTI_MRS_Plus(
            nifti_list=new_nifti_list,
            backend=self._backend,
            volatile=self.volatile,
            metadata={'common': deepcopy(self.metadata_common),
                     'individual': deepcopy(self.metadata_individual)}
        )

    def numpy(self) -> np.ndarray:
        """
        Return batched tensor of shape [B, ...] where B = number of subjects.
        Caches the result for efficiency.
        """
        if self._batched_data is None:
            if len(self.nifti_list) == 0:
                self._batched_data = np.array([])
            else:
                self._batched_data = np.stack([n[:] for n in self.nifti_list], axis=0)
        return self._batched_data

    def list(self) -> List[NIFTI_MRS]:
        """
        Return list of NIFTI_MRS objects.
        """
        return self.nifti_list

    def get_data(self, backend: Optional[Backend] = None):
        """
        Get data in the specified backend format.

        Args:
            backend: Desired backend (uses instance backend if None)

        Returns:
            Data in requested format (list of NIFTI_MRS, numpy array, or tensor)
        """
        target = backend or self._backend

        if target == Backend.NIFTI_LIST:
            return self.nifti_list

        # Use cached numpy array if available, otherwise compute it
        array = self.numpy()

        if target == Backend.NUMPY:
            return array
        elif target == Backend.PYTORCH:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            return torch.from_numpy(array)
        elif target == Backend.TENSORFLOW:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow not available")
            return tf.convert_to_tensor(array)
        elif target == Backend.JAX:
            if not JAX_AVAILABLE:
                raise ImportError("JAX not available")
            return jnp.array(array)
        elif target == Backend.KERAS:
            if not KERAS_AVAILABLE:
                raise ImportError("Keras not available")
            import keras.ops as ops
            return ops.convert_to_tensor(array)
        else:
            raise ValueError(f"Unknown backend: {target}")

    def to_nifti_list(self) -> List[NIFTI_MRS]:
        """Return the internal list of NIfTI-MRS objects."""
        return self.nifti_list

    # Proxy methods that delegate to first NIFTI_MRS (all have same structure)
    def dim_position(self, dim_tag: str) -> Optional[int]:
        """
        Get the position of a dimension tag.
        Delegates to first NIFTI_MRS (all subjects have same dim_tags).
        """
        if self.n_subjects > 0:
            return self.nifti_list[0].dim_position(dim_tag)
        return None

    def set_dim_tag(self, index: int, tag: str):
        """
        Set dimension tag for all subjects.
        Since all subjects must have same dim_tags, this updates all.
        """
        for nifti in self.nifti_list:
            nifti.set_dim_tag(index, tag)
        # Update cached common metadata
        if not self.volatile and self.n_subjects > 0:
            self.metadata_common['dim_tags'] = self.nifti_list[0].dim_tags

    def sync_headers(self, source_idx: int = 0) -> None:
        """
        Sync headers from one subject to all others.

        Updates all headers to match the header of the subject at 'source_idx'.

        Args:
            source_idx (int): Index of the reference subject.
        """
        if self.n_subjects == 0:
            return

        ref_hdr = deepcopy(self.nifti_list[source_idx].hdr_ext)
        for nifti in self.nifti_list:
            nifti.hdr_ext = deepcopy(ref_hdr)

        # Update cached metadata
        if not self.volatile:
            self.metadata_common['hdr_ext'] = ref_hdr

    @property
    def dwelltime(self):
        """Dwelltime from metadata (all subjects have same value)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].dwelltime
        return self.metadata_common.get('dwelltime')

    @property
    def spectrometer_frequency(self):
        """Spectrometer frequency from metadata (all subjects have same value)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].spectrometer_frequency
        return self.metadata_common.get('spectrometer_frequency')

    @property
    def nucleus(self):
        """Nucleus from metadata (all subjects have same value)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].nucleus
        return self.metadata_common.get('nucleus')

    @property
    def ndim(self):
        """Number of dimensions (from header extension)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].ndim
        return None

    @property
    def dtype(self):
        """Data type of the underlying data."""
        if self.n_subjects > 0:
            return self.nifti_list[0].dtype
        return None

    @property
    def header(self):
        """Header from first NIFTI_MRS (all have same structure)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].header
        return None

    @property
    def hdr_ext(self):
        """Header extension from first NIFTI_MRS (all have same structure)."""
        if self.n_subjects > 0:
            return self.nifti_list[0].hdr_ext
        return None

    def __getitem__(self, idx):
        """Get subset of subjects."""
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        new_nifti_list = self.nifti_list[idx]
        new_individual = self.metadata_individual[idx] if not self.volatile else []

        return NIfTI_MRS_Plus(
            nifti_list=new_nifti_list,
            backend=self._backend,
            volatile=self.volatile,
            metadata={'common': self.metadata_common, 'individual': new_individual}
        )

    def __setitem__(self, idx, values):
        """
        Set data values.

        Supports setting values for all backends. When setting values, the cached
        data is invalidated to ensure consistency.

        Args:
            idx: Index to set (can be int, slice, tuple for multi-dimensional indexing)
            values: Values to set (can be array, tensor, or list depending on backend)

        Examples:
            # For NIFTI_LIST backend
            nifti_plus[0] = new_nifti_mrs  # Set entire subject

            # For array/tensor backends
            nifti_plus[:, :, :, :, 100:200] = new_values  # Set specific timepoints
        """
        if self._backend == Backend.NIFTI_LIST:
            # Setting in NIFTI_LIST backend
            if isinstance(idx, (int, slice)):
                # Subject-level assignment
                if isinstance(values, list):
                    # Assigning list of NIFTI_MRS to multiple subjects
                    if isinstance(idx, int):
                        self.nifti_list[idx] = values[0] if len(values) == 1 else values
                    else:
                        self.nifti_list[idx] = values
                elif hasattr(values, 'dwelltime'):
                    # Assigning single NIFTI_MRS
                    self.nifti_list[idx] = values
                else:
                    raise ValueError(f"For NIFTI_LIST backend, values must be NIFTI_MRS object(s), got {type(values)}")
            else:
                # Multi-dimensional indexing - set data within each NIFTI_MRS
                if isinstance(values, list):
                    for i, nifti in enumerate(self.nifti_list):
                        nifti[idx] = values[i]
                else:
                    # Broadcast same values to all subjects
                    for nifti in self.nifti_list:
                        nifti[idx] = values

            # Invalidate cache
            self._batched_data = None

        else:
            # Setting in array/tensor backends
            # Need to update both the cached array and the underlying NIFTI objects

            # Convert values to numpy if needed
            if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
                values_np = values.numpy()
            elif TF_AVAILABLE and isinstance(values, tf.Tensor):
                values_np = values.numpy()
            elif JAX_AVAILABLE and isinstance(values, jnp.ndarray):
                values_np = np.asarray(values)
            elif isinstance(values, np.ndarray):
                values_np = values
            else:
                values_np = np.asarray(values)

            # Update cached array if it exists
            if self._batched_data is not None:
                self._batched_data[idx] = values_np

            # Update underlying NIFTI objects
            # For batched indexing like nifti_plus[0, :, :, :, 100:200]
            if isinstance(idx, tuple) and len(idx) > 0:
                # First element is subject index
                if isinstance(idx[0], int):
                    # Single subject
                    subject_idx = idx[0]
                    data_idx = idx[1:] if len(idx) > 1 else (slice(None),)
                    self.nifti_list[subject_idx][data_idx] = values_np[0] if values_np.ndim > 0 and values_np.shape[0] == 1 else values_np
                elif isinstance(idx[0], slice):
                    # Multiple subjects
                    subject_slice = idx[0]
                    data_idx = idx[1:] if len(idx) > 1 else (slice(None),)
                    subjects = self.nifti_list[subject_slice]
                    for i, nifti in enumerate(subjects):
                        # Handle broadcasting: if values_np has only 1 subject, broadcast to all
                        if values_np.shape[0] == 1:
                            nifti[data_idx] = values_np[0]
                        elif i < len(values_np):
                            nifti[data_idx] = values_np[i]
                        else:
                            raise ValueError(f"Not enough values to assign: need {len(subjects)}, got {len(values_np)}")
                else:
                    # Array indexing
                    raise NotImplementedError("Advanced indexing with arrays not yet supported for __setitem__")
            else:
                # Simple indexing - update all underlying NIFTI objects
                for i, nifti in enumerate(self.nifti_list):
                    nifti[idx] = values_np[i] if values_np.ndim > 0 and len(values_np) > i else values_np

    def __len__(self) -> int:
        """Number of subjects."""
        return self.n_subjects

    def __repr__(self) -> str:
        return (f"NIfTI_MRS_Plus(n_subjects={self.n_subjects}, shape={self.shape}, "
                f"backend={self._backend.value}, volatile={self.volatile})")
