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
            elif has_process_nifti_list:
                # Fallback: use NIfTI list processing even for tensor backends
                return self._process_via_nifti_list(data, water, **kwargs)

        # Fallback to forward if nothing else works
        if has_forward:
            return self._process_via_forward(data, water, **kwargs)

        # Last resort: try nifti list
        if has_process_nifti_list:
            return self._process_via_nifti_list(data, water, **kwargs)

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
        """Process using tensor method.

        Following the zea pattern, data is passed to ``process_tensor()`` in its
        **native backend format** (NumPy / PyTorch / JAX / TensorFlow) so that
        gradients and device placement are preserved.  The write-back into
        NIFTI_MRS storage always goes through NumPy (unavoidable since NIfTI is
        a NumPy-backed format).

        Performance tiers
        -----------------
        1. **Fast** — same N_PTS in and out (all augmentations except truncate/zerofill):
           one vectorised tensor op, then ``nifti[:] = sample`` in-place.  Zero extra
           allocations.

        2. **Medium** — N_PTS changes (e.g. truncation, zero-fill): tensor slice/pad is
           still vectorised, but write-back must rebuild each NIfTI_MRS object
           (O(B) constructor calls).  A ``RuntimeWarning`` is emitted.

        3. **Slow (~ routing)** — module only implements ``process_nifti_list``:
           falls back to the NIfTI-list path, processes each subject individually.
           This is what sampling modules (CoilAverageSampler, NIfTI_RawProcessor)
           always use because they change DIM_COIL / DIM_DYN, not just values.

        Rule of thumb: keep **N_PTS and all extra dimensions uniform** across the
        whole batch for tier-1 speed everywhere.
        """
        from augmentrum.utils.tensor_ops import to_numpy

        # Inject spectral metadata so modules can use sw_hz / sf_mhz without
        # requiring FSL-MRS objects.
        # Primary source: metadata_common (populated when volatile=False).
        # Fallback: read directly from nifti_list[0] (works even in volatile mode).
        if 'sw_hz' not in kwargs:
            dwell = data.metadata_common.get('dwelltime')
            if dwell is None and data.n_subjects > 0:
                try:
                    dwell = data.nifti_list[0].dwelltime
                except Exception:
                    pass
            if dwell is not None:
                kwargs['sw_hz'] = 1.0 / dwell

        if 'sf_mhz' not in kwargs:
            sf = data.metadata_common.get('spectrometer_frequency')
            if sf is None and data.n_subjects > 0:
                try:
                    sf = data.nifti_list[0].spectrometer_frequency
                except Exception:
                    pass
            if sf is not None:
                kwargs['sf_mhz'] = sf[0] if hasattr(sf, '__getitem__') else sf

        # ── Get data in native backend format (not forced to numpy!) ──
        data_array = data.get_data(backend)
        water_array = water.get_data(backend) if water is not None else None

        # ── Process (receives native tensors — preserves gradients) ──
        processed_data, processed_water = self.process_tensor(
            data_array, water_array, backend=backend, **kwargs
        )

        # ── Write back into NIfTI_MRS list (must go through numpy) ──
        nifti_list_out = data.list()
        if processed_data is not None:
            processed_np = to_numpy(processed_data)
            # Detect N_PTS change (e.g. truncation, zero-fill)
            in_npts = data_array.shape[-1] if hasattr(data_array, 'shape') else None
            out_npts = processed_np.shape[-1] if processed_np is not None else None
            if in_npts is not None and out_npts is not None and out_npts != in_npts:
                import warnings
                warnings.warn(
                    f"{self.__class__.__name__}: N_PTS changed {in_npts} → {out_npts} "
                    f"(same for all {processed_np.shape[0]} batch members — module params "
                    f"are uniform by design). "
                    f"NIfTI objects will be rebuilt with the new length. "
                    f"This is correct but ~slower than the normal in-place write-back path. "
                    f"For maximum throughput, keep N_PTS the same throughout the pipeline "
                    f"(e.g. prefer Apodization[exponential] over Apodization[truncate] "
                    f"when batching on a tensor backend).",
                    stacklevel=3,
                )
            for i, nifti in enumerate(nifti_list_out):
                if i < processed_np.shape[0]:
                    sample_np = processed_np[i]
                    if sample_np.shape == nifti[:].shape:
                        nifti[:] = sample_np
                    else:
                        # N_PTS changed — rebuild this NIfTI_MRS object from scratch
                        from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
                        try:
                            affine = nifti.getAffine('voxel', 'world')
                        except Exception:
                            affine = None
                        nucleus = (nifti.nucleus[0]
                                   if hasattr(nifti, 'nucleus') and nifti.nucleus else '1H')
                        dim_tags = (list(nifti.dim_tags)
                                    if hasattr(nifti, 'dim_tags') else [None, None, None])
                        new_nifti = gen_nifti_mrs(
                            data=sample_np,
                            dwelltime=nifti.dwelltime,
                            spec_freq=nifti.spectrometer_frequency[0],
                            nucleus=nucleus,
                            dim_tags=dim_tags,
                            affine=affine,
                        )
                        # Copy extra header fields
                        if hasattr(nifti, 'hdr_ext'):
                            for key in nifti.hdr_ext:
                                if key not in ('SpectrometerFrequency', 'ResonantNucleus',
                                               'dim_5', 'dim_6', 'dim_7'):
                                    try:
                                        new_nifti.add_hdr_field(key, nifti.hdr_ext[key])
                                    except Exception:
                                        pass
                        nifti_list_out[i] = new_nifti

        # Log provenance
        operation_name = self.__class__.__name__
        operation_details = {'method': 'process_tensor', 'backend': backend.value,
                           'params': self.params}

        data_out = NIfTI_MRS_Plus(
            nifti_list=nifti_list_out,
            backend=backend,
            volatile=data.volatile
        )

        if not data.volatile:
            data_out.update_metadata(operation_name, operation_details)

        water_out = None
        if water is not None:
            water_list_out = water.list()
            if processed_water is not None:
                processed_water_np = to_numpy(processed_water)
                for i, nifti in enumerate(water_list_out):
                    if i < processed_water_np.shape[0]:
                        sample_np = processed_water_np[i]
                        if sample_np.shape == nifti[:].shape:
                            nifti[:] = sample_np
                        else:
                            from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
                            try:
                                affine = nifti.getAffine('voxel', 'world')
                            except Exception:
                                affine = None
                            nucleus = (nifti.nucleus[0]
                                       if hasattr(nifti, 'nucleus') and nifti.nucleus else '1H')
                            dim_tags = (list(nifti.dim_tags)
                                        if hasattr(nifti, 'dim_tags') else [None, None, None])
                            new_nifti = gen_nifti_mrs(
                                data=sample_np,
                                dwelltime=nifti.dwelltime,
                                spec_freq=nifti.spectrometer_frequency[0],
                                nucleus=nucleus,
                                dim_tags=dim_tags,
                                affine=affine,
                            )
                            if hasattr(nifti, 'hdr_ext'):
                                for key in nifti.hdr_ext:
                                    if key not in ('SpectrometerFrequency', 'ResonantNucleus',
                                                   'dim_5', 'dim_6', 'dim_7'):
                                        try:
                                            new_nifti.add_hdr_field(key, nifti.hdr_ext[key])
                                        except Exception:
                                            pass
                            water_list_out[i] = new_nifti

            water_out = NIfTI_MRS_Plus(
                nifti_list=water_list_out,
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




