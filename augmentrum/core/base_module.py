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

#*************#
#   imports   #
#*************#
import functools
import inspect
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from augmentrum.core import NIfTI_MRS_Plus, Backend
from nifti_mrs_plus.core import DataState
from nifti_mrs_plus import ops
from nifti_mrs_plus.random import SeedGenerator
from augmentrum.core import precision as prec
from augmentrum.core.pool import (masks_of, set_masks, origin_of, set_origin, rewrap,
                                  water_masks)


#**************************************************************************************************#
#                                        Class BaseModule                                          #
#**************************************************************************************************#
#                                                                                                  #
# Unified base class for ALL augmentation/processing pipeline modules.                             #
#                                                                                                  #
#**************************************************************************************************#
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

    # Empty means none. Every subclass declares the backends it handles, so a
    # module that forgets fails immediately instead of silently claiming to
    # support everything. Use tuple(Backend) for a module that works anywhere.
    SUPPORTED_BACKENDS: Tuple[Backend, ...] = ()

    # Dimension tags a module adds to the data, outermost first. A module that
    # grows the array's rank must name the axes it added: a rebuilt NIfTI object
    # takes its tags from the source, which by definition does not have them.
    ADDS_DIM_TAGS: Tuple[str, ...] = ()

    # Dimension tags a module consumes. A module that collapses an axis must
    # name it, for the same reason: the source still carries the tag the
    # output no longer has an axis for.
    REMOVES_DIM_TAGS: Tuple[str, ...] = ()

    # Which domain this module works in. None means it does not care and runs
    # wherever it finds the data, which is the common case. A module that names
    # one is moved there and back; one that also sets STRICT is not moved at
    # all, because it defines the domain it operates in and being handed another
    # is a pipeline mistake rather than something to silently correct.
    DOMAIN = None
    STRICT: bool = False

    # Parameters this module can broadcast per sample. The pipeline samples a
    # range once per batch by default; for names listed here it draws one value
    # per sample instead, so a batch carries a spread rather than one point of
    # it. Only modules whose processing paths handle a vector may declare this.
    PER_SAMPLE_PARAMS: Tuple[str, ...] = ()

    # Parameters that count things. A range on one of these is drawn as an
    # integer over the inclusive bounds, so "n_averages=(8, 32)" can actually
    # give 32. The pipeline also reads an "int" annotation or default off the
    # constructor; this is for names typed neither way, such as the samplers'
    # counts, which default to None.
    INTEGER_PARAMS: Tuple[str, ...] = ()

    # How a module meets the per-sample dimension masks a sampler drawing with
    # per_sample=True leaves on a batch (see augmentrum.core.pool). 'pass': it
    # acts on every element alike, so the masks flow through untouched.
    # 'consume' / 'draw': it takes them as "dim_masks" and reports the masks
    # left afterwards as "dim_masks_". None: it would aggregate over entries a
    # sample never drew, so a batch carrying masks is refused rather than
    # processed as if every coil and transient were there.
    MASKS: Optional[str] = None

    # Whether a module that returns its input tensor itself has left the values
    # exactly as they were (samplers that only mask, taps). The batch then
    # keeps its pool origin, and consumers may use results the pool cached.
    PRESERVES_VALUES: bool = False

    # The working precision, set by every module's "precision" argument: 'single', 'double',
    # or None to follow the data. The data are cast to it on the way in (augmentrum.core.
    # precision), so a module that follows the dtype of what it is given computes, and
    # returns, in that precision.
    precision: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """
        Wrap a subclass's "__init__" so its arguments are recorded automatically.

        "self.params" is what reaches the NIfTI provenance header, so it has to
        match the constructor exactly. Binding the call against the signature
        gets every argument, including defaults the caller never mentioned,
        without the subclass doing anything.
        """
        super().__init_subclass__(**kwargs)

        init = cls.__dict__.get('__init__')
        if init is None or getattr(init, '_records_params', False):
            return

        signature = inspect.signature(init)
        kinds = {p.name: p.kind for p in signature.parameters.values()}

        @functools.wraps(init)
        def __init__(self, *args, precision=None, **kwargs):
            init(self, *args, **kwargs)

            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()

            params = {}
            for name, value in bound.arguments.items():
                if name == 'self' or kinds[name] is inspect.Parameter.VAR_POSITIONAL:
                    continue
                if kinds[name] is inspect.Parameter.VAR_KEYWORD:
                    params.update(value or {})
                else:
                    params[name] = value

            # every module takes "precision", whatever its own signature (pipeline.
            # constructor_params adds it); set last, so an outer class's value wins
            self.precision = prec.check(precision)
            if precision is not None:
                params['precision'] = precision
            self.params = params
            self.rng = SeedGenerator(params.get('seed'))

        __init__._records_params = True
        cls.__init__ = __init__

    def __init__(self, **kwargs):
        """
        Initialize module with optional parameters.

        Subclasses need not pass anything: their own arguments are recorded from
        their signature. Anything given here is merged in, for modules that add
        parameters not named in their constructor.

        Args:
            **kwargs: Extra parameters to record alongside the constructor's.
        """
        self.params = {**getattr(self, 'params', {}), **kwargs}

        # Every module draws from one generator, held for its lifetime: a seeded
        # run reproduces while each batch still receives a fresh perturbation.
        self.rng = SeedGenerator(self.params.get('seed'))

    #**********************#
    #   per-sample values   #
    #**********************#
    @staticmethod
    def per_sample(value, ndim):
        """
        A per-sample vector shaped to broadcast over a "(batch, ...)" tensor.

        Scalars pass through untouched, so a call site stays one line whether
        the pipeline injected one value for the batch or one per sample.

        Args:
            value: A scalar, or a "(batch,)" vector of per-sample values.
            ndim: Rank of the tensor the value will multiply.

        Returns:
            The scalar itself, or the vector reshaped to "(batch, 1, ..., 1)".
        """
        arr = np.asarray(value)
        if arr.ndim == 0:
            return value
        return arr.reshape((-1,) + (1,) * (ndim - 1))

    @staticmethod
    def sample_of(value, index):
        """
        One subject's value out of a per-sample vector; scalars pass through.

        The NIfTI-list paths process one subject at a time, so a vector the
        pipeline sampled for the batch has to be read entry by entry there.
        """
        arr = np.asarray(value)
        if arr.ndim == 0:
            return value
        return arr[index % len(arr)]

    def __call__(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None,
                 **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Main entry point - handles everything automatically.

        - Moves the data to a domain this module works in, and back after
        - Checks backend compatibility
        - Routes to appropriate processing method
        - Handles logging/provenance (if not volatile)
        - Returns NIfTI_MRS_Plus objects
        """
        if self.DOMAIN is not None and not self.DOMAIN.satisfied_by(data.state):
            return self._via_domain(data, water, **kwargs)
        return self._dispatch(data, water, **kwargs)

    def _via_domain(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus],
                    **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Run this module in the domain it needs, and put the data back.

        Calling a module directly is not a pipeline, so there is nothing to plan
        against and the transform has to be undone straight away. A pipeline
        does better: it looks at the whole chain and moves the data once for a
        run of modules that want the same thing.

        A module marked "STRICT" is not moved. Those define the domain they
        operate in, so being handed another is a mistake in the pipeline rather
        than something to paper over.
        """
        from augmentrum.processing.domain import DomainError, DomainTransform

        if self.STRICT:
            raise DomainError(
                f"{self.__class__.__name__} works in {self.DOMAIN}, but the data is "
                f"spectral={data.state.spectral}, spatial={data.state.spatial}. "
                f"Insert a DomainTransform before it, or let a pipeline plan the "
                f"transforms for you."
            )

        was = data.state
        moved, water = DomainTransform(spectral=self.DOMAIN.spectral,
                                       spatial=self.DOMAIN.spatial)(data, water)
        out, water = self._dispatch(moved, water, **kwargs)

        back = DomainTransform(
            spectral=was.spectral if self.DOMAIN.spectral else None,
            spatial=was.spatial if self.DOMAIN.spatial else None)
        out, water = back(out, water)
        return out.set_state(out.state.having(last=self.__class__.__name__)), water

    def _dispatch(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None,
                  **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """Pick the processing method that suits this module and the backend."""
        # Determine which method to use based on backend
        backend = data.backend

        # Fall back to a backend this module actually handles
        if backend not in self.SUPPORTED_BACKENDS:
            backend = self.get_preferred_backend()

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
        # NIfTI objects keep the dtype they are stored in: a list-path module works on
        # that, and its output is put into the working precision
        cast_out = lambda out: (out if out is None or self.precision is None else out.set_data(
            prec.cast(out.get_data(out.backend if out.backend != Backend.NIFTI_LIST
                                   else Backend.NUMPY), self.precision)))

        # Wrap back into NIfTI_MRS_Plus
        data_out = NIfTI_MRS_Plus(
            nifti_list=processed_data,
            backend=data.backend,
            volatile=data.volatile,
            state=self.output_state(data.state)
        )

        data_out = cast_out(data_out)
        # Update metadata if not volatile
        if not data.volatile:
            data_out.update_metadata(operation_name, operation_details)

        water_out = None
        if processed_water is not None:
            water_out = NIfTI_MRS_Plus(
                nifti_list=processed_water,
                backend=water.backend if water else Backend.NIFTI_LIST,
                volatile=water.volatile if water else data.volatile,
                state=self.output_state(water.state if water else data.state)
            )
            water_out = cast_out(water_out)
            if water and not water.volatile:
                water_out.update_metadata(operation_name, operation_details)

        return data_out, water_out

    def _process_via_tensor(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus],
                           backend: Backend, **kwargs) -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """Process using tensor method.

        Data is passed to "process_tensor()" in its **native backend format**
        (NumPy / PyTorch / JAX / TensorFlow) so that gradients and device
        placement are preserved.

        Performance tiers
        -----------------
        1. **Fast** — shape unchanged (all augmentations except truncate/zerofill):
           one vectorized tensor op, then the result is handed to
           "NIfTI_MRS_Plus.set_data", which keeps it on its own backend. Nothing
           is converted, so an autograd graph survives the whole pipeline and is
           only resolved when something asks for NIfTI objects.

        2. **Medium** — shape changes (e.g. truncation, zero-fill): the tensor op is
           still vectorized, but write-back must rebuild each NIfTI_MRS object
           through NumPy (O(B) constructor calls), which ends the graph. A
           "RuntimeWarning" is emitted.

        3. **Slow (~ routing)** — module only implements "process_nifti_list":
           falls back to the NIfTI-list path, processes each subject individually.
           This is what sampling modules (CoilSampler in its list-only modes)
           always use because they change DIM_COIL / DIM_DYN, not just values.

        Rule of thumb: keep **N_PTS and all extra dimensions uniform** across the
        whole batch for tier-1 speed everywhere.
        """
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

        # Inject spatial geometry (matrix, voxel size, FOV) the same way, so
        # modules that need to reason about k-space read it off the NIfTI-MRS
        # data rather than having it passed in by hand and drifting from it.
        # Absent or unreadable geometry is not an error — most modules never
        # look at it, and a bare-array workflow legitimately has none.
        if 'geometry' not in kwargs and data.n_subjects > 0:
            try:
                from augmentrum.sampling.kspace_sampling import KspaceGeometry
                kwargs['geometry'] = KspaceGeometry.read_header_geometry(data)
            except Exception:
                pass

        # Inject the dimension tags too. A bare tensor has no way to say which
        # of its trailing axes is coils and which is averages, so a module that
        # addresses one by name needs to be told.
        if 'dim_tags' not in kwargs and data.n_subjects > 0:
            kwargs['dim_tags'] = data.dim_tags

        # The water reference is a separate acquisition with its own layout: a
        # single transient where the metabolite scan has thirty-two, say. A
        # module handling both must not read the water's axes off the data's.
        if 'water_dim_tags' not in kwargs and water is not None and water.n_subjects > 0:
            kwargs['water_dim_tags'] = water.dim_tags

        # The nucleus fixes the ppm reference (4.65 ppm for 1H, the FSL-MRS
        # convention), so every module places features on the same axis.
        if 'nucleus' not in kwargs and data.n_subjects > 0:
            nucleus = data.nucleus
            if isinstance(nucleus, (list, tuple)):
                nucleus = nucleus[0] if nucleus else None
            if nucleus is not None:
                kwargs['nucleus'] = str(nucleus)

        # And where the data is, so a module can act on the domain it is in
        # rather than assume one.
        if 'state' not in kwargs:
            kwargs['state'] = data.state

        # Per-sample masks go to the modules that honour them, and nowhere else.
        masks = masks_of(data)
        takes_masks = self.MASKS in ('consume', 'draw')
        if masks and not takes_masks and self.MASKS != 'pass':
            raise ValueError(
                f"{self.__class__.__name__} cannot take the per-sample masks over "
                f"{sorted(masks)} that a sampler drawing with per_sample=True left on the "
                f"batch: it would treat every coil and transient as drawn. Place it after "
                f"'processing', which consumes the masks, or draw with per_sample=False."
            )
        if takes_masks:
            kwargs['dim_masks'] = masks
            self.dim_masks_ = dict(masks)

        # Values that still equal pool entries let a consumer use what the pool cached.
        origins = (origin_of(data), origin_of(water))
        if origins[0] is not None:
            kwargs.setdefault('pool_origin', origins[0])
        if origins[1] is not None:
            kwargs.setdefault('water_pool_origin', origins[1])

        # ── Get data in native backend format (not forced to numpy!), in the working precision ──
        original = data.get_data(backend)
        water_original = water.get_data(backend) if water is not None else None
        data_array = prec.cast(original, self.precision)
        water_array = prec.cast(water_original, self.precision)

        # ── Process (receives native tensors — preserves gradients) ──
        moved = self._spectral_axis_last(data_array)
        processed_data, processed_water = self.process_tensor(
            moved if moved is not None else data_array,
            water_array, backend=backend, **kwargs
        )
        if moved is not None:
            processed_data = self._spectral_axis_back(processed_data)

        # ── Write back ──
        operation_name = self.__class__.__name__
        operation_details = {'method': 'process_tensor', 'backend': backend.value,
                           'params': self.params}

        data_out = self._wrap_processed(data, processed_data, backend,
                                        operation_name, operation_details)

        water_out = None
        if water is not None:
            water_out = self._wrap_processed(water, processed_water, backend,
                                             operation_name, operation_details,
                                             dim_tags=self._output_water_dim_tags(water))

        masks = self.dim_masks_ if takes_masks else masks
        set_masks(data_out, masks)
        set_masks(water_out, water_masks(masks))
        if self.PRESERVES_VALUES:
            # values still equal the pool's only if nothing, not even a cast, touched them
            set_origin(data_out, origins[0] if processed_data is original else None)
            set_origin(water_out, origins[1] if processed_water is water_original else None)

        return data_out, water_out

    #****************#
    #   write-back   #
    #****************#
    def _wrap_processed(self, source: NIfTI_MRS_Plus, processed,
                        backend: Backend, operation_name: str,
                        operation_details: Dict, dim_tags: Optional[List] = None
                        ) -> NIfTI_MRS_Plus:
        """
        Wrap a processed tensor back into a "NIfTI_MRS_Plus".

        The tensor is installed with "set_data" and never converted, so an
        autograd graph and device placement survive to the end of the pipeline.
        A module that resized or added a dimension is no exception: fitting the
        NIfTI objects to it is deferred to materialization, which is the single
        point at which data becomes NumPy again.

        Args:
            source: The batch the tensor came from.
            processed: The tensor to install, or None to leave the source's.
            backend: The backend the tensor lives on.
            operation_name: What to record in the provenance.
            operation_details: The provenance record's details.
            dim_tags: Higher-dimension tags of the processed tensor; the
                source's, minus what this module collapsed, when not given.
        """
        # A batch drawn from a pool stays one, so its borrowed objects stay safe.
        out = rewrap(source, source.nifti_list, backend, source.volatile,
                     self.output_state(source.state))
        if processed is not None:
            if dim_tags is None:
                dim_tags = self._output_dim_tags(source)
            out.set_data(processed, backend, dim_tags=dim_tags)

        if not source.volatile:
            out.update_metadata(operation_name, operation_details)

        return out

    #***********************#
    #   the spectral axis   #
    #***********************#
    #: Where NIfTI-MRS keeps the spectral points once a batch axis is in front.
    SPECTRAL_AXIS = 4

    def _spectral_axis_last(self, data_array):
        """
        Bring the spectral axis to the end, if this module needs it there.

        Every spectral module is written against the last axis, because that is
        where the spectral points sit in the usual "(batch, X, Y, Z, T)". Add a
        coil or average axis behind it and the last axis is no longer T - so a
        line broadening would decay across coils instead of along the FID, and
        return a perfectly plausible volume while doing it.

        Only modules that named a spectral domain are moved: they are exactly
        the ones that read along it. Modules working element by element do not
        care what order the axes come in.

        Args:
            data_array: The batch, in the NIfTI layout.

        Returns:
            The batch with T last, or None when nothing needed moving.
        """
        if self.DOMAIN is None or self.DOMAIN.spectral is None:
            return None

        rank = len(ops.shape(data_array))
        if rank <= self.SPECTRAL_AXIS + 1:
            return None

        self._axis_rank = rank
        order = ([d for d in range(rank) if d != self.SPECTRAL_AXIS]
                 + [self.SPECTRAL_AXIS])
        return ops.transpose(data_array, order)

    def _spectral_axis_back(self, data_array):
        """Undo :meth:"_spectral_axis_last", leaving the caller's layout."""
        rank = self._axis_rank
        order = (list(range(self.SPECTRAL_AXIS)) + [rank - 1]
                 + list(range(self.SPECTRAL_AXIS, rank - 1)))
        return ops.transpose(data_array, order)

    def output_state(self, state: 'DataState') -> 'DataState':
        """
        Where the data is once this module has run.

        Recorded whether or not the batch is volatile, because it is a fixed
        handful of facts rather than a growing record. Most modules leave the
        data where they found it and only sign their name; one that moves it
        between domains, or undersamples it, overrides this.

        Args:
            state: The state on the way in.

        Returns:
            The state on the way out.
        """
        return state.having(last=self.__class__.__name__)

    def _output_dim_tags(self, source: NIfTI_MRS_Plus) -> List:
        """
        Higher-dimension tags the processed data carries.

        Stated in full rather than as what this module added, because the
        objects being wrapped are the source's own and may already be behind by
        several steps. A full list composes; a delta against a stale list does
        not.

        Added axes go last, because that is where the modules that grow the
        array put them; removed tags drop out with the axis they named.
        """
        tags = [t for t in (source.dim_tags or [])
                if t is not None and t not in self.REMOVES_DIM_TAGS]
        tags += list(self.ADDS_DIM_TAGS)
        tags += [None] * (3 - len(tags))
        return tags[:3]

    def _output_water_dim_tags(self, source: NIfTI_MRS_Plus) -> List:
        """
        Higher-dimension tags the processed water carries.

        The water is a separate acquisition with its own layout, so a module
        that collapses its dimensions on their own terms (the raw processor
        averages the water's transients, not the data's) overrides this; by
        default the water follows the data's rule.
        """
        return self._output_dim_tags(source)

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
        return backend in self.SUPPORTED_BACKENDS

    def get_preferred_backend(self) -> Backend:
        """The backend this module would rather be given."""
        if not self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"{self.__class__.__name__}.SUPPORTED_BACKENDS is empty, so the "
                f"module declares that it supports no backend at all. List the "
                f"backends it handles, or tuple(Backend) if it works anywhere."
            )
        return self.SUPPORTED_BACKENDS[0]

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
        backends = (['all'] if tuple(self.SUPPORTED_BACKENDS) == tuple(Backend)
                    else [b.value for b in self.SUPPORTED_BACKENDS])
        return f"{self.__class__.__name__}(backends={backends})"


#**************************************************************************************************#
#                                            Class Tap                                             #
#**************************************************************************************************#
#                                                                                                  #
# Identity marker naming a pipeline stage so dataloaders can yield it.                             #
#                                                                                                  #
#**************************************************************************************************#
class Tap(BaseModule):
    """
    Identity marker naming a pipeline stage so dataloaders can yield it.

    A tap changes nothing about the data. Its presence tells the pipeline to
    snapshot the batch as it passes, so the stage can be referenced by name in
    an "outputs" spec — e.g. "pipeline=['noise', 'tap:clean', 'undersampling']"
    with "outputs=(('data', 'water'), ('clean', 'clean.water'))" yields
    supervised (input, target) pairs, the input fully augmented and the target
    frozen at the tap. In pipeline name lists the sugar "'tap:<name>'" sets the
    name; a bare "'tap'" is simply named 'tap'.
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # A tap copies nothing and changes nothing: masks and pool origins survive it.
    MASKS = 'pass'
    PRESERVES_VALUES = True

    def __init__(self, name: str = 'tap'):
        super().__init__()
        self.name = name

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        return data_array, water_array

    def process_nifti_list(self, data_list, water_list=None, **kwargs):
        return data_list, water_list




