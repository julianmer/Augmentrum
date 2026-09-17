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
import difflib
import inspect
import typing
import warnings
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from nifti_mrs_plus.random import SeedGenerator
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.core.base_module import BaseModule, Tap
from augmentrum.core.pool import finalize_masks, masks_of, rewrap, set_masks
from augmentrum.processing.domain import Domain


#***************#
#   seeding     #
#***************#
def child_seed(seed, key: Sequence[int]) -> int:
    """
    A seed derived from *seed* for one named consumer.

    Every random stream in a run - subject draws, ranged parameters, each
    module's own generator - is a child of one root seed, so that a single
    integer fixes the whole run. The children are spawned through NumPy's
    "SeedSequence" with a spawn key rather than by arithmetic on the seed, so
    that neighbouring keys give unrelated streams and adding a consumer does
    not shift the seeds of the others.

    Args:
        seed: The root seed (any non-negative int).
        key: The consumer's position, e.g. "(split_index, role)".

    Returns:
        A 63-bit int usable by "np.random.default_rng" and "SeedGenerator".
    """
    sequence = np.random.SeedSequence(int(seed), spawn_key=tuple(int(k) for k in key))
    return int(sequence.generate_state(1, dtype=np.uint64)[0]) & 0x7FFFFFFFFFFFFFFF


#*******************#
#   introspection   #
#*******************#
def constructor_params(module) -> List[str]:
    """
    The names a module's constructor accepts, as a pipeline may set them.

    Read from the signature rather than kept in a table, so a new module or
    argument is picked up without anyone editing a list. "self" and the
    variadic catch-alls are left out: a "**kwargs" that a constructor merely
    swallows is not a name a user can meaningfully set.

    Args:
        module: A module class or instance.

    Returns:
        The parameter names, in signature order; empty if uninspectable.
    """
    cls = module if isinstance(module, type) else module.__class__
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        params = getattr(module, 'params', None)
        return list(params.keys()) if isinstance(params, dict) else []
    return [p.name for p in sig.parameters.values()
            if p.name != 'self'
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                               inspect.Parameter.VAR_KEYWORD)]


def integer_params(module) -> set:
    """
    The constructor parameters of *module* that count things.

    A range on one of these is drawn as an integer over the inclusive bounds.
    Read from three places, because none alone covers every module: an "int"
    annotation ("n_pts: Optional[int]"), an int default ("order: int = 3"),
    and the module's own "INTEGER_PARAMS" for names typed neither way.

    Args:
        module: A module class or instance.
    """
    cls = module if isinstance(module, type) else module.__class__
    names = set(getattr(cls, 'INTEGER_PARAMS', ()) or ())
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return names

    for p in sig.parameters.values():
        if _is_integer(p.default):
            names.add(p.name)
            continue
        annotation = p.annotation
        # Optional[int] / Union[int, None] name int among their arguments
        candidates = (typing.get_args(annotation) or (annotation,)) \
            if typing.get_origin(annotation) is typing.Union else (annotation,)
        if any(a is int for a in candidates):
            names.add(p.name)
    return names


def suggest_names(name: str, valid, n: int = 3) -> str:
    """
    A short "did you mean" clause for an unknown name, empty if nothing is close.

    Args:
        name: The name that was not recognized.
        valid: The names that would have been.
        n: How many suggestions at most.
    """
    close = difflib.get_close_matches(name, sorted(set(valid)), n=n, cutoff=0.5)
    return f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""


def _is_number(value) -> bool:
    """A real scalar - int or float, NumPy or plain - and not a bool."""
    return isinstance(value, (int, float, np.integer, np.floating)) \
        and not isinstance(value, (bool, np.bool_))


def _is_integer(value) -> bool:
    """A plain or NumPy integer, and not a bool."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def is_range(value) -> bool:
    """
    Whether *value* is a "(min, max)" range the pipeline samples from.

    Only a 2-tuple of numbers, or of numbers and None, counts. A tuple of
    tuples is structure, not a range - ResidualWater's "peaks" is one - and a
    list is left to the module too, so nested ranges inside it stay intact.

    Args:
        value: Any parameter value.
    """
    return (isinstance(value, tuple) and len(value) == 2
            and all(v is None or _is_number(v) for v in value))


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

    Randomness is drawn from two places: ranged parameters from this
    pipeline's own NumPy generator, and each module's perturbations from the
    module's "SeedGenerator". "reseed" derives both from one seed, so a
    pipeline built from unseeded modules becomes reproducible the moment it is
    handed to a seeded "Augmentrum". A module given an explicit "seed" keeps
    it.
    """

    #: Keys of "user_kwargs" that steer sampling rather than any module.
    GLOBAL_KEYS = ('param_distribution', 'param_distributions')

    def __init__(self, steps: List[BaseModule], module_names=None, user_kwargs=None,
                 end_domain=None, domain_planning='auto', step_kwargs=None, seed=None):
        """
        Initializes the pipeline with a list of augmentation steps.

        Args:
            steps: List of augmentation step instances (must inherit from BaseModule).
            module_names: List of module name strings (e.g., ['phase', 'noise'])
            user_kwargs: Dictionary of all user-provided parameters. A key is
                routed to every step whose constructor names it.
            step_kwargs: Per-step parameters, one dict per step (or None),
                aligned with *steps*. These reach that step only and override
                *user_kwargs* there; a range is sampled per batch like any
                other, a scalar is injected as given.
            seed: Fixes every random draw this pipeline makes - see "reseed".
                None leaves ranged parameters on a fresh generator and the
                modules on whatever they were built with.
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

        if step_kwargs is None:
            step_kwargs = [{} for _ in steps]
        if len(step_kwargs) != len(steps):
            raise ValueError(
                f"step_kwargs has {len(step_kwargs)} entries for {len(steps)} steps; "
                f"it must be aligned with the steps, with None or {{}} for a step "
                f"that takes nothing."
            )
        self.step_kwargs = [dict(kw or {}) for kw in step_kwargs]

        # Ranged parameters are drawn from here, never from the global
        # np.random state, so a seed fixes them and nothing else can shift them.
        self.seed = None
        self.rng = np.random.default_rng()
        if seed is not None:
            self.reseed(seed)

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

    def reseed(self, seed) -> 'AugmentationPipeline':
        """
        Derive every random stream in this pipeline from one seed.

        Two things draw: this pipeline, for ranged parameters, and each step,
        for its own perturbations. Both get a child of *seed*, spawned by
        position, so two pipelines built the same way and seeded alike replay
        the same run - including one assembled by hand from unseeded modules.

        A step constructed with an explicit "seed" is left alone: the user
        pinned it on purpose, and a pipeline seed should not silently undo that.

        Args:
            seed: The root seed for this pipeline.

        Returns:
            "self", so calls can be chained.
        """
        self.seed = int(seed)
        self.rng = np.random.default_rng(child_seed(self.seed, (0,)))

        for index, step in enumerate(self.steps):
            if not isinstance(step, BaseModule):
                continue
            params = getattr(step, 'params', None) or {}
            if params.get('seed') is not None:
                continue
            step.rng = SeedGenerator(child_seed(self.seed, (1, index)))
        return self

    def _extract_module_params(self):
        """
        Extract relevant parameters for each module from user kwargs.

        Uses introspection to automatically discover what parameters each module accepts,
        then extracts matching values from user_kwargs. Per-step kwargs are laid
        over the top, so a global value reaches every step that names it while
        a step value reaches its step alone.

        A global key that lands on two or more steps is reported once: "lb_hz"
        names a line width in both LineBroadening and Apodization, and a user
        who meant one of them would otherwise never learn that both moved.
        """
        module_params = {}
        landed = {}   # global key -> [step labels it reached without an override]

        for idx, step in enumerate(self.steps):
            param_names = constructor_params(step)
            overrides = self.step_kwargs[idx] if idx < len(self.step_kwargs) else {}

            params = {}
            for param_name in param_names:
                if param_name in self.user_kwargs and param_name not in overrides:
                    params[param_name] = self.user_kwargs[param_name]
                    landed.setdefault(param_name, []).append(
                        f"{step.__class__.__name__} (step {idx})")
            params.update(overrides)

            if params:
                module_params[idx] = params

        shared = {key: steps for key, steps in landed.items() if len(steps) > 1}
        if shared:
            lines = [f"  {key!r} -> {' and '.join(steps)}" for key, steps in shared.items()]
            warnings.warn(
                "A global parameter reaches more than one step, so the same value "
                "is applied in each of them:\n" + "\n".join(lines) + "\n"
                "To set one step alone, give the value per step, e.g. "
                "pipeline=[{'line_broadening': {'lb_hz': (0, 5)}}, ...].",
                UserWarning, stacklevel=3,
            )

        return module_params

    def _validate_steps(self):
        """Validate that all steps are BaseModule instances."""
        for i, step in enumerate(self.steps):
            if not isinstance(step, BaseModule):
                warnings.warn(
                    f"Step {i} ({type(step).__name__}) does not inherit from BaseModule. "
                    f"Backend compatibility checking will be skipped for this step."
                )

    def sample_batch_parameters(self, batch_size: int, rng=None):
        """
        Sample the ranged parameters for the coming batch.

        The default is one value per batch, shared by every sample. Parameters
        a module names in its "PER_SAMPLE_PARAMS" are drawn as a "(batch_size,)"
        vector instead, so the batch carries a spread of the range rather than
        one point of it.

        Args:
            batch_size: Number of samples in the batch, and so the length of
                any per-sample draw.
            rng: NumPy generator to draw from. Defaults to the pipeline's own,
                so a seeded pipeline replays; a caller that wants a draw
                independent of the batch stream - fixed-mode parameters, say -
                passes its own.

        Returns:
            Dictionary mapping step index to parameter dictionaries
            Format: {step_idx: {param_name: sampled_value}}
        """
        rng = self.rng if rng is None else rng

        # Get distribution settings from user_kwargs
        global_distribution = self.user_kwargs.get('param_distribution', 'uniform')
        per_param_distributions = self.user_kwargs.get('param_distributions', {})

        batch_params = {}

        # Sample from stored module_params (extracted from user_kwargs)
        for step_idx, params in self.module_params.items():
            step = self.steps[step_idx]
            per_sample = getattr(step, 'PER_SAMPLE_PARAMS', ())
            counts = integer_params(step)
            step_params = {}

            for param_name, param_value in params.items():
                # Get distribution for this parameter
                distribution = per_param_distributions.get(param_name, global_distribution)

                # Sample value — a vector where the module can broadcast one
                size = batch_size if param_name in per_sample else None
                sampled_val = self._sample_from_range(
                    param_value, distribution, size, rng, integral=param_name in counts)
                step_params[param_name] = sampled_val

            if step_params:
                batch_params[step_idx] = step_params

        return batch_params

    def _sample_from_range(self, param, distribution: str = 'uniform', size=None, rng=None,
                           integral: bool = False):
        """
        Sample from a parameter (range or scalar).

        Args:
            param: Either scalar (float/int) or tuple (min, max) for range
            distribution: 'uniform', 'gaussian', 'normal', 'exponential', 'beta'
            size: None for a single scalar draw, or a count for a vector of
                independent draws. Scalars pass through either way — a fixed
                value is fixed for every sample.
            rng: NumPy generator to draw from; the pipeline's own by default.
            integral: The parameter counts things. With integer bounds the
                draw is then an integer over the inclusive range, as the
                samplers document "(8, 32)" - a float in [8, 32) cast by the
                module would never reach 32. Off by default, since "(0, 10)"
                on a line width means a continuous range written with int
                literals; "sample_batch_parameters" turns it on for the
                parameters a module types as integers.

        Returns:
            Scalar value, or an array of *size* draws for a ranged parameter.
            A range with a None bound is handed over unchanged, because only
            the module knows what "up to all" is.
        """
        rng = self.rng if rng is None else rng

        # If already scalar, return as-is. The type is preserved: casting to
        # float here used to break integer parameters injected through kwargs
        # (np.random.default_rng(0.0) raises where default_rng(0) works).
        if _is_number(param) or isinstance(param, (bool, np.bool_)):
            return param

        # If None, return None
        if param is None:
            return None

        # If tuple, sample based on distribution
        if is_range(param):
            min_val, max_val = param

            # "(1, None)" means "one up to however many there are": the count
            # is data-dependent and the sampler interprets it, so it must
            # arrive intact rather than be turned into a number here.
            if min_val is None or max_val is None:
                return param

            integral = integral and _is_integer(min_val) and _is_integer(max_val)
            lo, hi = float(min_val), float(max_val)

            # Sample based on distribution
            if distribution in ['gaussian', 'normal']:
                # Gaussian centered at midpoint, std = range/6 (99.7% within range)
                mean = (lo + hi) / 2.0
                std = (hi - lo) / 6.0
                value = np.clip(rng.normal(mean, std, size), lo, hi)

            elif distribution == 'exponential':
                # Exponential biased toward min_val
                scale = (hi - lo) / 3.0
                value = np.clip(lo + rng.exponential(scale, size), lo, hi)

            elif distribution == 'beta':
                # Beta distribution (slightly biased to center)
                alpha, beta = 2.0, 2.0
                value = lo + rng.beta(alpha, beta, size) * (hi - lo)

            elif integral:
                # 'uniform' on integers: every value in [lo, hi] equally likely
                value = rng.integers(min_val, max_val + 1, size)

            else:
                # 'uniform', and the default for anything unrecognized
                value = rng.uniform(lo, hi, size)

            if integral:
                # A shaped draw on integer bounds is rounded onto the grid the
                # bounds define, and clipped so rounding cannot leave the range.
                value = np.clip(np.rint(value), min_val, max_val).astype(int)
                return int(value) if size is None else value
            return float(value) if size is None else value

        # If single-element tuple
        if isinstance(param, tuple) and len(param) == 1:
            return float(param[0])

        # Otherwise return as-is: strings, lists, nested tuples and objects are
        # the module's business.
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

        # Masks nothing consumed become zeros: whoever takes the tensor cannot
        # see them, and a coil or transient that was not drawn must not read
        # as one that was.
        current_data = finalize_masks(current_data)
        current_water = finalize_masks(current_water)

        if not self._tap_indices:
            return current_data, current_water

        # Bring every snapshot to the same end domain as the output, mirroring
        # the end clause of domain_plan, so all yielded stages are aligned. A
        # tap that fired mid-chain may sit in k-space; one before any domain
        # move is already there and is left untouched.
        from augmentrum.processing.domain import DomainTransform

        end = self.end_domain or Domain(spectral='time', spatial='image')
        for name, (tap_data, tap_water) in taps.items():
            tap_data, tap_water = finalize_masks(tap_data), finalize_masks(tap_water)
            taps[name] = (tap_data, tap_water)
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
            return set_masks(data.copy(), masks_of(data))

        snap = rewrap(data, data.nifti_list, data.backend, data.volatile, data.state)
        snap.set_data(data.get_data(data.backend), data.backend, dim_tags=data.dim_tags)
        return set_masks(snap, masks_of(data))

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
        seed = f", seed={self.seed}" if self.seed is not None else ""
        return f"AugmentationPipeline(steps={step_names}{seed})"
