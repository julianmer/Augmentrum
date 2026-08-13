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
from augmentrum.core.base_module import BaseModule, Tap
from augmentrum.processing.domain import Domain
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

    def __init__(self, steps: List[BaseModule], module_names=None, user_kwargs=None,
                 end_domain=None, domain_planning='auto'):
        """
        Initializes the pipeline with a list of augmentation steps.

        Args:
            steps: List of augmentation step instances (must inherit from BaseModule).
            module_names: List of module name strings (e.g., ['phase', 'noise'])
            user_kwargs: Dictionary of all user-provided parameters
            end_domain: Where the data should be left. Defaults to the NIfTI-MRS
                canonical form, time domain and image space, so the result can
                be written out. Pass a "Domain" to stay somewhere else - staying
                in k-space, for instance, so spatial augmentation acts there.
            domain_planning: 'auto' (default) inserts DomainTransforms wherever
                a module's declared domain is not already satisfied; transforms
                the user placed themselves count, so a correctly hand-placed
                chain gets nothing added. 'strict' inserts nothing and raises
                DomainError instead, for users who place every transform
                themselves and want mistakes surfaced rather than fixed.
        """
        if domain_planning not in ('auto', 'strict'):
            raise ValueError(f"domain_planning must be 'auto' or 'strict', "
                             f"got {domain_planning!r}")
        self.steps = steps
        self.end_domain = end_domain
        self.domain_planning = domain_planning
        self.module_names = module_names or []
        self.user_kwargs = user_kwargs or {}
        self._validate_steps()

        # Store parameters for each module (extracted from user_kwargs using introspection)
        self.module_params = self._extract_module_params()

        # Named taps: stages this pipeline snapshots as the data passes, so an
        # outputs spec can yield any stage, not just the end.
        self._tap_indices = {}
        for i, step in enumerate(steps):
            if isinstance(step, Tap):
                if step.name in self._tap_indices:
                    raise ValueError(
                        f"Duplicate tap name '{step.name}' in pipeline. Each tap "
                        f"must have a unique name to be referenced in outputs."
                    )
                self._tap_indices[step.name] = i

    @property
    def tap_names(self) -> Tuple[str, ...]:
        """Names of the taps in this pipeline, in pipeline order."""
        return tuple(self._tap_indices)

    @property
    def has_taps(self) -> bool:
        """Whether this pipeline snapshots any stage."""
        return bool(self._tap_indices)

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
        Sample the ranged parameters for the coming batch.

        The default is one value per batch, shared by every sample. Parameters
        a module names in its "PER_SAMPLE_PARAMS" are drawn as a "(batch_size,)"
        vector instead, so the batch carries a spread of the range rather than
        one point of it.

        Args:
            batch_size: Number of samples in the batch, and so the length of
                any per-sample draw.

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
            step = self.steps[step_idx]
            per_sample = getattr(step, 'PER_SAMPLE_PARAMS', ())
            step_params = {}

            for param_name, param_value in params.items():
                # Get distribution for this parameter
                distribution = per_param_distributions.get(param_name, global_distribution)

                # Sample value — a vector where the module can broadcast one
                size = batch_size if param_name in per_sample else None
                sampled_val = self._sample_from_range(param_value, distribution, size)
                step_params[param_name] = sampled_val

            if step_params:
                batch_params[step_idx] = step_params

        return batch_params

    def _sample_from_range(self, param, distribution: str = 'uniform', size=None):
        """
        Sample from a parameter (range or scalar).

        Args:
            param: Either scalar (float/int) or tuple (min, max) for range
            distribution: 'uniform', 'gaussian', 'normal', 'exponential', 'beta'
            size: None for a single scalar draw, or a count for a vector of
                independent draws. Scalars pass through either way — a fixed
                value is fixed for every sample.

        Returns:
            Scalar value, or an array of *size* draws for a ranged parameter.
        """
        import numpy as np

        # If already scalar, return as-is. The type is preserved: casting to
        # float here used to break integer parameters injected through kwargs
        # (np.random.default_rng(0.0) raises where default_rng(0) works).
        if isinstance(param, (int, float)):
            return param

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
            if distribution in ['gaussian', 'normal']:
                # Gaussian centered at midpoint, std = range/6 (99.7% within range)
                mean = (min_val + max_val) / 2.0
                std = (max_val - min_val) / 6.0
                value = np.random.normal(mean, std, size)
                return np.clip(value, min_val, max_val)

            elif distribution == 'exponential':
                # Exponential biased toward min_val
                scale = (max_val - min_val) / 3.0
                value = min_val + np.random.exponential(scale, size)
                return np.clip(value, min_val, max_val)

            elif distribution == 'beta':
                # Beta distribution (slightly biased to center)
                alpha, beta = 2.0, 2.0
                value = np.random.beta(alpha, beta, size)
                return min_val + value * (max_val - min_val)

            else:
                # 'uniform', and the default for anything unrecognized
                return np.random.uniform(min_val, max_val, size)

        # If single-element tuple
        if isinstance(param, tuple) and len(param) == 1:
            return float(param[0])

        # Otherwise return as-is
        return param

    def domain_plan(self, state=None):
        """
        The steps as they will actually run, with domain transforms inserted.

        Planned once for the whole chain rather than left to each module, which
        is what keeps the transforms down: a module that names a domain forces
        one only if the data is not already there, so a run of modules wanting
        the same thing shares a single move. Modules that name no domain never
        force anything.

        Args:
            state: Where the data starts. Defaults to the canonical form.

        Returns:
            "[(index, step)]", where *index* is the step's position in this
            pipeline and "-1" marks a transform the plan inserted - so a caller
            can still match sampled parameters to the step they belong to.

        Raises:
            DomainError: If a strict module is reached in the wrong domain.
                Those define the domain they operate in, so the plan says what
                to insert rather than guessing.
        """
        from nifti_mrs_plus.core import DataState
        from augmentrum.processing.domain import DomainError, DomainTransform

        state = state if state is not None else DataState()
        planned = []

        for index, step in enumerate(self.steps):
            wanted = getattr(step, 'DOMAIN', None)
            if wanted is not None and not wanted.satisfied_by(state):
                if getattr(step, 'STRICT', False) or self.domain_planning == 'strict':
                    raise DomainError(
                        f"{type(step).__name__} at step {index} works in {wanted}, but "
                        f"the data reaches it as spectral={state.spectral}, "
                        f"spatial={state.spatial}. Add a DomainTransform before it."
                    )
                move = DomainTransform(spectral=wanted.spectral, spatial=wanted.spatial)
                planned.append((-1, move))
                state = move.output_state(state)

            planned.append((index, step))
            if hasattr(step, 'output_state'):
                state = step.output_state(state)

        # Leave the data somewhere it can be written out, unless told otherwise.
        # A strict plan inserts nothing: the data ends wherever the user's own
        # transforms left it, unless an end_domain states a requirement.
        if self.domain_planning == 'strict':
            if self.end_domain is not None and not self.end_domain.satisfied_by(state):
                raise DomainError(
                    f"The pipeline ends with spectral={state.spectral}, "
                    f"spatial={state.spatial}, but end_domain asks for "
                    f"{self.end_domain}. Add a closing DomainTransform."
                )
            return planned
        end = self.end_domain or Domain(spectral='time', spatial='image')
        if not end.satisfied_by(state):
            planned.append((-1, DomainTransform(spectral=end.spectral,
                                                spatial=end.spatial)))
        return planned

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
            Tuple of (processed_data, processed_water), or — when the pipeline
            contains taps — (processed_data, processed_water, taps) with
            "taps: {name: (data, water)}", each snapshot end-domain aligned.
            The arity is structural, fixed at construction: a pipeline with a
            tap in it was asked for its stages, so a stale two-tuple unpack
            fails loudly instead of silently dropping them.
        """
        from fsl_mrs.core.nifti_mrs import NIFTI_MRS

        # Auto-wrap single NIFTI_MRS objects to NIfTI_MRS_Plus
        if isinstance(data, NIFTI_MRS):
            data = NIfTI_MRS_Plus([data], volatile=True)
        if isinstance(water, NIFTI_MRS):
            water = NIfTI_MRS_Plus([water], volatile=True)

        current_data = data
        current_water = water
        taps = {}

        # Inject sampled parameters BEFORE planning the domains: a module's
        # declared domain may depend on the value that will actually run
        # (PhaseShift's first-order ramp needs a spectrum only when nonzero),
        # so the plan has to see this batch's values, not constructor defaults.
        original_attrs = {}
        for index, params in (batch_params or {}).items():
            step = self.steps[index]
            original_attrs[index] = {}
            for param_name, param_value in params.items():
                if hasattr(step, param_name):
                    original_attrs[index][param_name] = getattr(step, param_name)
                setattr(step, param_name, param_value)

        try:
            for i, step in self.domain_plan(data.state):
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

                # Apply the step
                current_data, current_water = step(current_data, current_water, **kwargs)

                # Snapshot the stage as it passes a tap. The snapshot must own
                # its values: later steps write into shared NIfTI objects, so a
                # bare reference would silently read back the fully augmented
                # data.
                if isinstance(step, Tap):
                    taps[step.name] = (
                        self._snapshot(current_data),
                        self._snapshot(current_water) if current_water is not None else None,
                    )
        finally:
            # Restore original attributes
            for index, attrs in original_attrs.items():
                for param_name, orig_value in attrs.items():
                    setattr(self.steps[index], param_name, orig_value)

        if not self._tap_indices:
            return current_data, current_water

        # Bring every snapshot to the same end domain as the output, mirroring
        # the end clause of domain_plan, so all yielded stages are aligned. A
        # tap that fired mid-chain may sit in k-space; one before any domain
        # move is already there and is left untouched.
        from augmentrum.processing.domain import DomainTransform

        end = self.end_domain or Domain(spectral='time', spatial='image')
        for name, (tap_data, tap_water) in taps.items():
            if not end.satisfied_by(tap_data.state):
                tap_data, tap_water = DomainTransform(
                    spectral=end.spectral, spatial=end.spatial)(tap_data, tap_water)
                taps[name] = (tap_data, tap_water)

        return current_data, current_water, taps

    def _snapshot(self, data: NIfTI_MRS_Plus) -> NIfTI_MRS_Plus:
        """
        The batch as it stands, owned so later steps cannot touch it.

        Two sharing mechanisms make a bare reference wrong: list-path modules
        write "nifti[:] = ..." into the very objects they were handed, and
        every step's output wraps the *same* nifti list — so a snapshot without
        its own pending tensor re-materializes as whatever ran last. Owning the
        data breaks both. On the volatile tensor path (the training hot path)
        this is free when a pending tensor exists: "set_data" installs it and
        marks the snapshot dirty, so it never restacks from the shared list.
        Non-volatile data is copied outright, which also freezes the tap's
        provenance at exactly the tap point.
        """
        if data.backend == Backend.NIFTI_LIST or not data.volatile:
            return data.copy()

        snap = NIfTI_MRS_Plus(nifti_list=data.nifti_list, backend=data.backend,
                              volatile=data.volatile, state=data.state)
        snap.set_data(data.get_data(data.backend), data.backend, dim_tags=data.dim_tags)
        return snap

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
