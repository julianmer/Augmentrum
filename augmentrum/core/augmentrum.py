####################################################################################################
#                                         augmentrum.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#          J. T. LaMaster (jlamaste@gmail.com)                                                     #
#          K. C. Igwe (kci2104@columbia.edu)                                                       #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Main Augmentrum class - backend-agnostic MRS data augmentation                          #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import os
import warnings
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any
import numpy as np
from augmentrum import __version__
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.core.base_module import BaseModule, Tap
from augmentrum.core.pipeline import (AugmentationPipeline, child_seed, constructor_params,
                                      is_range, suggest_names)

# Import helper functions from dataset_utils
from augmentrum.core.dataset_utils import (
    create_random_generator,
    create_fixed_generator,
    wrap_generator_for_framework,
    output_tokens,
    flatten_structure,
    map_structure,
)
from augmentrum.sampling.subject_splitter import SubjectSplitter
from augmentrum.core.pool import TensorPool

# Processing modules
from augmentrum.processing.raw_processing import RawProcessor
from augmentrum.sampling.coil_sampling import CoilSampler
from augmentrum.sampling.dimension_sampling import AverageSampler
from augmentrum.sampling.kspace_sampling import KspaceUndersampling

# Augmentation modules
from augmentrum.augmentation.amplitude_scaling import AmplitudeScaling
from augmentrum.augmentation.noise import Noise
from augmentrum.augmentation.line_broadening import LineBroadening
from augmentrum.augmentation.baseline_augmentation import BaselineAugmentation
from augmentrum.augmentation.residual_water import ResidualWater
from augmentrum.augmentation.macromolecules import Macromolecules
from augmentrum.augmentation.spurious_echoes import SpuriousEchoes
from augmentrum.augmentation.artificial_peaks import ArtificialPeaks
from augmentrum.augmentation.eddy_current import EddyCurrent
from augmentrum.augmentation.apodization import Apodization
from augmentrum.augmentation.phase_frequency import PhaseShift, FrequencyShift
from augmentrum.augmentation.spatial_augmentations import SpatialAugmentations
from augmentrum.augmentation.transient_synthesis import TransientSynthesizer
from augmentrum.augmentation.zero_fill import ZeroFill
from augmentrum.processing.edit_combination import EditCombiner


__all__ = ['Augmentrum']


#**************************************************************************************************#
#                                         Class Augmentrum                                         #
#**************************************************************************************************#
#                                                                                                  #
# Main Augmentrum class - backend-agnostic MRS data augmentation.                                  #
#                                                                                                  #
#**************************************************************************************************#
class Augmentrum:
    """
    Main Augmentrum class - backend-agnostic MRS data augmentation.

    Provides easy-to-use interface for:
    - Loading NIFTI_MRS data (metabolite + water)
    - Optional train/val/test splitting, subject-wise when group ids are known
    - Flexible augmentation pipelines (different per split), configurable per step
    - On-the-fly (random) or fixed (exact) augmentation parameters
    - Multi-backend support (PyTorch, NumPy, TensorFlow, JAX)
    - One seed for the whole run: split, subject draws, ranges and modules

    Example 1: On-the-fly augmentation with RANGE sampling::

        augmenter = Augmentrum(
            data=nifti_list,
            water=water_list,
            pipeline=['coil_sampling', 'processing', 'noise', 'line_broadening'],
            mode='on-the-fly',
            n_coils=(1, 8),               # Random 1-8 coils (inclusive)
            n_averages=(4, 16),           # Random 4-16 averages (inclusive)
            sigma_frac=(0.01, 0.05),      # Random noise 1-5%
            lb_hz=(0, 10),                # Random broadening 0-10 Hz
            param_distribution='uniform', # How to sample from ranges (default)
            batch_size=32,
            backend='numpy'
        )

        # Each batch will have DIFFERENT random augmentations!
        for batch_data, batch_water in augmenter.dataloader(framework='numpy'):
            train_model(batch_data)

    Example 2: Fixed augmentation (exact values)::

        augmenter = Augmentrum(
            data=nifti_list,
            water=water_list,
            pipeline=['coil_sampling', 'processing', 'noise'],
            mode='fixed',
            n_coils=4,          # Always exactly 4 coils
            n_averages=8,       # Always exactly 8 averages
            sigma_frac=0.03,    # Always 3% noise (scalar = exact value)
            batch_size=32,
            backend='pytorch'
        )

        # Each batch will have IDENTICAL augmentations, and so will every
        # dataloader() call: a ranged parameter in 'fixed' mode is drawn once
        # per split and reused until reseed().
        for batch_data, batch_water in augmenter.dataloader():
            validate_model(batch_data)

    Example 3: Per-step parameters::

        # A global kwarg reaches every step whose constructor names it, so
        # lb_hz would move LineBroadening AND Apodization. A step's own
        # kwargs reach that step alone and override the globals there.
        augmenter = Augmentrum(
            data=nifti_list,
            pipeline=[
                'coil_sampling',
                {'noise': {'sigma_frac': (0.0, 0.02)}},
                {'line_broadening': {'lb_hz': (0.0, 5.0)}},
                ('apodization', {'mode': 'exponential', 'lb_hz': 2.0}),
                'baseline_bspline',            # alias fixing mode='bspline'
            ],
            n_coils=(1, 8),
            backend='pytorch'
        )

    Example 4: Gaussian distribution sampling::

        augmenter = Augmentrum(
            data=nifti_list,
            pipeline=['noise', 'line_broadening'],
            sigma_frac=(0.01, 0.05),    # Range
            lb_hz=(0, 10),              # Range
            param_distribution='gaussian',  # Sample with Gaussian (centered at midpoint)
            backend='pytorch'
        )
        # Samples will be more concentrated around 0.03 (midpoint) for sigma_frac
        # and around 5.0 (midpoint) for lb_hz, with tails at the extremes

    Example 5: With train/val/test splitting::

        augmenter = Augmentrum(
            data=nifti_list,
            water=water_list,
            split_fractions={'val': 0.1, 'test': 0.1},
            groups=subject_ids,          # one id per item; a subject never straddles splits
            pipelines={
                'train': ['processing', 'noise', 'line_broadening', 'baseline'],
                'val': ['processing'],
                'test': None
            },
            modes={
                'train': 'on-the-fly',  # Random augmentations for training
                'val': 'fixed',         # Fixed params for validation
                'test': 'fixed'         # Fixed params for testing
            },
            n_coils=(1, 8),
            sigma_frac=(0.01, 0.05),
            batch_size=32,
            backend='pytorch',
            seed=1,                      # the whole run replays from this
        )

        train_dl = augmenter.train_dataloader()
        val_dl = augmenter.val_dataloader()
        augmenter.split_groups['val']    # -> the subject ids held out for validation
    """

    # Available augmentation modules. A value is a class, or a "(class,
    # fixed_kwargs)" pair for a name that means one configuration of a class;
    # see "resolve_module".
    AVAILABLE_MODULES = {
        # Processing
        'coil_sampling': CoilSampler,
        'average_sampling': AverageSampler,
        'transient_synthesis': TransientSynthesizer,
        'edit_combination': EditCombiner,
        'processing': RawProcessor,

        # Noise
        'noise': Noise,

        # Amplitude
        'amplitude': AmplitudeScaling,
        'amplitude_scaling': AmplitudeScaling,

        # Line broadening
        'line_broadening': LineBroadening,
        'broadening': LineBroadening,

        # Baseline
        'baseline': BaselineAugmentation,
        'baseline_random_walk': (BaselineAugmentation, {'mode': 'random_walk'}),
        'baseline_bspline': (BaselineAugmentation, {'mode': 'bspline'}),
        'baseline_polynomial': (BaselineAugmentation, {'mode': 'polynomial'}),

        # Phase & Frequency
        'phase': PhaseShift,
        'phase_shift': PhaseShift,
        'frequency_shift': FrequencyShift,

        # Artifacts
        'residual_water': ResidualWater,
        'macromolecules': Macromolecules,
        'water': ResidualWater,
        'spurious_echoes': SpuriousEchoes,
        'echoes': SpuriousEchoes,
        'artificial_peaks': ArtificialPeaks,
        'peaks': ArtificialPeaks,
        'eddy_current': EddyCurrent,
        'eddy': EddyCurrent,

        # Modification
        'apodization': Apodization,
        'apod': Apodization,
        'zero_filling': ZeroFill,
        'zero_fill': ZeroFill,

        # Spatial
        'spatial': SpatialAugmentations,
        'spatial_augmentations': SpatialAugmentations,

        # k-space
        'undersampling': KspaceUndersampling,
        'kspace_undersampling': KspaceUndersampling,

        # Pipeline control
        'tap': Tap,
    }

    def __init__(
        self,
        data: Union[List, NIfTI_MRS_Plus],
        water: Optional[Union[List, NIfTI_MRS_Plus]] = None,

        # Splitting
        split_fractions: Optional[Dict[str, float]] = None,  # e.g., {'val': 0.1, 'test': 0.1}
        split_indices: Optional[Dict[str, Sequence[int]]] = None,  # explicit, overrides fractions
        groups: Optional[Sequence] = None,  # one hashable id per item, e.g. the subject
        seed: Optional[int] = 42,

        # Pre-processing (applied once and cached)
        pre_pipeline: Optional[Union[List, AugmentationPipeline, Dict[str, Union[List, AugmentationPipeline]]]] = None,  # Fixed preprocessing

        # Augmentation
        pipeline: Optional[Union[List, AugmentationPipeline]] = None,  # Single pipeline
        pipelines: Optional[Dict[str, Union[List, AugmentationPipeline]]] = None,  # Per-split pipelines

        # Augmentation mode
        mode: str = 'on-the-fly',  # 'on-the-fly' or 'fixed'
        modes: Optional[Dict[str, str]] = None,  # Per-split modes

        # What the dataloaders yield
        outputs: Optional[Union[Tuple, Dict[str, Tuple]]] = None,

        # General
        batch_size: int = 16,
        backend: Union[str, Backend] = 'pytorch',
        device: Optional[str] = None,  # 'cuda', 'cpu', or None (auto-detect)
        volatile: bool = False,
        domain_planning: str = 'auto',  # 'auto' inserts DomainTransforms; 'strict' raises instead

        **kwargs  # Module-specific parameters
    ):
        """
        Initialize Augmentrum.

        Args:
            data: List of NIFTI_MRS objects or NIfTI_MRS_Plus
            water: Optional water references
            split_fractions: Dict like {'val': 0.1, 'test': 0.1}, train gets rest
            split_indices: Explicit item indices per split; overrides the fractions.
            groups: One hashable id per item (a subject id, say). Items of one
                    group always land in the same split, and the fractions are
                    met in number of items as closely as the group sizes allow.
                    None reads a user-defined "SubjectID" field from every
                    NIfTI header extension when all items carry one, and
                    otherwise splits item-wise. "split_groups" reports the
                    assignment.
            seed: Root seed for the whole run: the split, which subjects each
                  batch draws, every ranged parameter and every module's own
                  perturbations derive from it, so two instances built alike
                  with the same seed yield the same batches. None draws a root
                  from OS entropy; the value drawn is kept in ".seed" so the
                  run can still be repeated.
            pre_pipeline: Fixed preprocessing applied ONCE and cached. Can be:
                         - List of module names: ['coil_sampling', 'processing'] (same for all splits)
                         - AugmentationPipeline object (same for all splits)
                         - Dict mapping split names to pipelines (different per split):
                           {'train': ['coil_sampling', 'processing'], 'val': ['processing'], 'test': None}
                         These steps run before the main pipeline and results are stored.
                         Useful for expensive operations that don't need randomization.
            pipeline: Single pipeline for all splits: an AugmentationPipeline, or a
                      list whose entries are each a registry name ('noise'), a module
                      instance (Noise(sigma=0.1)), a one-key dict with that step's own
                      kwargs ({'noise': {'sigma_frac': (0.0, 0.02)}}) or a
                      (name, kwargs) tuple. Step kwargs reach that step only and
                      override the global **kwargs there.
            pipelines: Dict mapping split names to pipelines (overrides 'pipeline')
            mode: Single mode for all splits:
                  'on-the-fly' = Random sampling from ranges (e.g., n_coils=(1,8) picks randomly)
                  'fixed' = Use exact values provided (e.g., n_coils=4 always uses 4 coils);
                            a range is drawn once per split and then kept.
            modes: Dict mapping split names to modes (overrides 'mode')
            batch_size: Batch size
            backend: 'pytorch', 'numpy', 'tensorflow', 'keras', 'jax', or Backend enum
            device: Where the PyTorch backend keeps the subject pool and builds
                    batches ('cuda', 'cuda:1', 'cpu'); None keeps them on the CPU.
                    On tensor backends every split whose subjects share one
                    shape is stacked once into a pool on this device, batches
                    are drawn from it by indexing - no NIfTI object is copied
                    per batch - and the dataloaders hand out tensors on it.
                    The pool is a snapshot of the subjects' values; call
                    "refresh_pools" after changing them in place. A CUDA pool
                    does not survive forked DataLoader workers: use
                    num_workers=0 there, the device is the parallelism.
            volatile: Skip metadata updates for speed
            **kwargs: Module parameters, routed by name to every module in the
                pipelines whose constructor accepts them. A key no module accepts
                raises ValueError - a typo must not turn a step into a no-op.

                ALL PARAMETERS support both tuple ranges and exact values:
                  - Tuple (min, max) = randomly sample from range, once per batch
                    (per sample for parameters a module lists in PER_SAMPLE_PARAMS).
                    Integer-typed parameters draw integers over the inclusive
                    range; a None bound ("(1, None)") is passed to the module.
                  - Scalar (float/int/str/bool) = use exact value

                SAMPLING (coil_sampling / average_sampling / transient_synthesis):
                  - n_coils: (1, 8) or 4          - n_averages: (4, 16) or 8
                  - per_sample: draw coils / averages per sample, kept as masks
                    on tensors (give it per step to target one sampler)
                  - n_transients: 32, tr_s, drift_hz_per_min, ...

                PROCESSING ('processing' = RawProcessor):
                  - conj, coil, align, remove_outliers, average, ecc, truncate,
                    remove_water, shift_ref, phase_correct (bools)
                  - coil_method: 'fsl-mrs', 'adaptive'
                  - registration_method: 'fsl-mrs', 'pattern', 'torch' (the
                    batched device engine of every step)
                  - ecc_method, remove_method, average_method, water_removal_method,
                    shift_ref_method, phase_correct_method

                AUGMENTATION:
                  - noise: exactly one of sigma_frac (0.01, 0.05), snr, snr_db, sigma;
                    global_scale
                  - amplitude: scale_factor (0.7, 1.3)
                  - line_broadening: lb_hz (0, 10), gb_hz (0, 5)
                  - phase: zero_order_deg (-180, 180), first_order_deg (-90, 90)
                  - frequency_shift: shift_hz (-5, 5)
                  - baseline: baseline_frac (0.01, 0.1), step_sd, bounds_amp, smooth_pts,
                    knots_per_ppm, ed_per_ppm, phase_deg, order, ppm_windows
                  - residual_water: center_ppm, amplitude_scale (0.05, 0.2), phase_deg,
                    peaks, model
                  - macromolecules: mm_source, mm_scale (0.1, 0.2), source_params
                  - eddy_current: std_rad (0.3, 1.0), lp_cut_hz, strength (0.5, 1.5),
                    remove_linear
                  - spurious_echoes: echoes, global_phase_deg, alpha_reference
                  - artificial_peaks: peaks, ref_ppm, amp_mode
                  - apodization: lb_hz, gb_hz, n_pts, frac_pts, auto_lb, target_damp,
                    target_pts
                  - zero_fill: target_pts
                  - undersampling: ksp_mode, acceleration_factor, us_seed, ...

                Names shared by several modules (lb_hz, gb_hz, phase_deg, peaks,
                target_pts) reach all of them when given globally, with a warning;
                give them per step to target one (see "pipeline").

                DISTRIBUTION (for sampling from ranges):
                  - param_distribution: 'uniform' (default), 'gaussian', 'exponential', 'beta'
                      Global default for ALL parameters

                  - param_distributions: Dict mapping parameter names to specific distributions
                      Per-parameter control (overrides param_distribution)
                      Examples:
                        param_distributions={
                            'sigma_frac': 'gaussian',      # Gaussian for noise
                            'lb_hz': 'exponential',        # Exponential for broadening
                            'zero_order_deg': 'uniform'    # Uniform for phase
                        }

                  Note: If neither specified, default is 'uniform'
        """
        # Convert backend
        if isinstance(backend, str):
            self.backend = Backend[backend.upper()]
        else:
            self.backend = backend

        self.batch_size = batch_size
        self.device = device  # where tensor pools and batches live (None: CPU)
        self._pools: Dict[str, Tuple[tuple, Optional[TensorPool]]] = {}
        self.volatile = volatile
        self.domain_planning = domain_planning
        self.kwargs = kwargs

        # One root seed fixes the whole run. An unseeded run draws its root
        # from OS entropy and keeps it, so it can still be repeated afterwards.
        self.seed = int(seed) if seed is not None else int.from_bytes(os.urandom(8), 'little')

        # Convert data to NIfTI_MRS_Plus
        if not isinstance(data, NIfTI_MRS_Plus):
            self.data_all = NIfTI_MRS_Plus(nifti_list=data, backend=self.backend, volatile=volatile)
        else:
            self.data_all = data

        if water is not None and not isinstance(water, NIfTI_MRS_Plus):
            self.water_all = NIfTI_MRS_Plus(nifti_list=water, backend=self.backend, volatile=volatile)
        elif water is not None:
            self.water_all = water
        else:
            self.water_all = None

        # Group ids keep a subject's scans together across splits
        self.groups = self._resolve_groups(groups)

        # Handle splitting
        if split_indices is not None:
            self._create_splits_from_indices(split_indices)
        elif split_fractions is not None:
            self._create_splits(split_fractions, self.seed)
        else:
            # No splitting - all data goes to train, but create empty val/test
            empty_data = NIfTI_MRS_Plus(nifti_list=[], backend=self.backend, volatile=self.volatile)
            self.splits = {
                'train': (self.data_all, self.water_all),
                'val': (empty_data, None),
                'test': (empty_data, None)
            }
            self.split_groups = self._groups_of({'train': list(range(len(self.data_all))),
                                                 'val': [], 'test': []})

        # Build every pipeline first and validate the kwargs against all of
        # them, so a typo is caught before any preprocessing has been paid for.
        self._create_pre_pipeline(pre_pipeline)
        self._create_pipelines(pipeline, pipelines)
        self._create_modes(mode, modes)
        self._create_outputs(outputs)
        self._validate_kwargs()

        # Derive every random stream from the root seed, then run the cached
        # preprocessing on seeded pipelines.
        self._seed_streams()
        self._apply_pre_pipelines()

        # Additional features for flexibility
        self.callbacks = []  # Custom augmentation callbacks
        self.stats = {  # Statistics tracking
            'batches_generated': 0,
            'samples_generated': 0,
            'split_stats': {split: {'batches': 0, 'samples': 0} for split in self.splits.keys()}
        }

    #**************#
    #   grouping   #
    #**************#
    #: Header extension field a loader may set to name the subject of a scan.
    GROUP_FIELD = 'SubjectID'

    def _resolve_groups(self, groups) -> Optional[list]:
        """
        The group id of every item, from the argument or the NIfTI headers.

        A loader that knows which subject a scan belongs to writes it as a
        user-defined "SubjectID" header field (nifti_mrs wraps those as
        "{'Value': ..., 'Description': ...}"). When every item carries one,
        that is the grouping; a partial set is ignored rather than mixed with
        made-up ids, because a half-grouped split is not a subject-wise split.

        Args:
            groups: The caller's ids, or None to look in the headers.

        Returns:
            One id per item, or None to split item-wise.
        """
        n_total = len(self.data_all)
        if groups is not None:
            groups = list(groups)
            if len(groups) != n_total:
                raise ValueError(
                    f"groups has {len(groups)} entries for {n_total} items; "
                    f"give one group id per item."
                )
            return groups

        ids = [self._header_group(nifti) for nifti in self.data_all.list()]
        if ids and all(value is not None for value in ids):
            return ids
        return None

    @classmethod
    def _header_group(cls, nifti):
        """The item's "SubjectID" header value, or None when it has none."""
        try:
            value = nifti.hdr_ext[cls.GROUP_FIELD]
        except (KeyError, TypeError, AttributeError):
            return None
        if isinstance(value, dict):
            value = value.get('Value')
        return value

    def _groups_of(self, split_indices: Dict[str, Sequence[int]]) -> Optional[Dict[str, list]]:
        """
        Sorted group ids per split, or None when the items are ungrouped.

        Args:
            split_indices: Item indices per split.
        """
        if self.groups is None:
            return None
        return {name: SubjectSplitter._sorted_ids({self.groups[i] for i in idx})
                for name, idx in split_indices.items()}

    #***********#
    #   splits  #
    #***********#
    def _create_splits(self, split_fractions: Dict[str, float], seed: int):
        """Create train/val/test splits."""
        val_frac = split_fractions.get('val', 0.0)
        test_frac = split_fractions.get('test', 0.0)

        splitter = SubjectSplitter(
            self.data_all.list(),
            self.water_all.list() if self.water_all is not None else None,
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac,
            groups=self.groups,
        )

        splits_raw = splitter.split()
        self.split_groups = splitter.split_groups

        # Convert back to NIfTI_MRS_Plus
        self.splits = {}
        for split_name, (data_list, water_list) in splits_raw.items():
            data_plus = NIfTI_MRS_Plus(nifti_list=data_list, backend=self.backend, volatile=self.volatile)
            water_plus = NIfTI_MRS_Plus(nifti_list=water_list, backend=self.backend, volatile=self.volatile) if water_list else None
            self.splits[split_name] = (data_plus, water_plus)

    def _create_splits_from_indices(self, split_indices: Dict[str, Sequence[int]]):
        """
        Build splits from explicit subject indices instead of random fractions.

        Use this whenever membership is a property of the data rather than
        something to draw — held-out test subjects defined by a challenge or a
        study protocol, a site-wise split, a pinned reproduction of an earlier
        run. Split names are free-form, so a dataset with two separate test sets
        can keep them apart:

            Augmentrum(data=all_subjects,
                       split_indices={'train': range(0, 19), 'val': range(19, 24),
                                      'test_track1': [24, 25, 26, 27, 28],
                                      'test_track2': [29, 30, 31]})

        and each is reachable via "dataloader(split='test_track1')".
        """
        n_total = len(self.data_all)
        data_list = self.data_all.list()
        water_list = self.water_all.list() if self.water_all is not None else None

        seen: Dict[int, str] = {}
        self.splits = {}
        chosen: Dict[str, List[int]] = {}
        for split_name, indices in split_indices.items():
            idx = [int(i) for i in indices]

            for i in idx:
                if not (0 <= i < n_total):
                    raise IndexError(
                        f"split_indices['{split_name}'] contains index {i}, but there "
                        f"are only {n_total} subjects."
                    )
                if i in seen:
                    raise ValueError(
                        f"Subject {i} appears in both '{seen[i]}' and '{split_name}'. "
                        "Overlapping splits would leak data between them."
                    )
                seen[i] = split_name
            chosen[split_name] = idx

            data_plus = NIfTI_MRS_Plus(
                nifti_list=[data_list[i] for i in idx],
                backend=self.backend, volatile=self.volatile,
            )
            water_plus = None
            if water_list is not None:
                water_plus = NIfTI_MRS_Plus(
                    nifti_list=[water_list[i] for i in idx],
                    backend=self.backend, volatile=self.volatile,
                )
            self.splits[split_name] = (data_plus, water_plus)

        unassigned = n_total - len(seen)
        if unassigned:
            warnings.warn(
                f"{unassigned} of {n_total} subjects are not in any split and will "
                f"never be sampled.",
                RuntimeWarning, stacklevel=3,
            )

        # Explicit indices are the caller's call, but a subject straddling two
        # of them is almost never what was meant, so say so.
        self.split_groups = self._groups_of(chosen)
        if self.split_groups is not None:
            straddling = {}
            for name, ids in self.split_groups.items():
                for group in ids:
                    straddling.setdefault(group, []).append(name)
            leaked = {g: names for g, names in straddling.items() if len(names) > 1}
            if leaked:
                warnings.warn(
                    f"split_indices place items of one group in several splits: "
                    f"{leaked}. Scans of one subject in both train and val leak.",
                    RuntimeWarning, stacklevel=3,
                )

    def _create_pre_pipeline(self, pre_pipeline):
        """
        Create and apply pre-pipeline (fixed preprocessing that runs once and is cached).

        The pre-pipeline is applied to all data ONCE during initialization, and the results
        are cached. This is useful for expensive operations that don't need randomization,
        like coil combination, eddy current correction, etc.

        Supports per-split pre-pipelines for different preprocessing strategies per split.

        Args:
            pre_pipeline: Can be:
                         - None: No pre-pipeline
                         - List of module names: same pre-pipeline for all splits
                         - AugmentationPipeline object: same for all splits
                         - Dict mapping split names to pipelines: different per split
        """
        self.pre_pipeline = None
        self.pre_pipelines = None
        self.preprocessed_splits = None
        if pre_pipeline is None:
            return

        # Check if it's a dict (per-split pre-pipelines)
        if isinstance(pre_pipeline, dict):
            # Per-split pre-pipelines
            self.pre_pipelines = {}

            for split_name in self.splits.keys():
                split_pre_pipeline = pre_pipeline.get(split_name, None)

                if split_pre_pipeline is None:
                    self.pre_pipelines[split_name] = None
                elif isinstance(split_pre_pipeline, AugmentationPipeline):
                    self.pre_pipelines[split_name] = split_pre_pipeline
                else:
                    # Build from list of module names
                    self.pre_pipelines[split_name] = self._build_pipeline_from_list(split_pre_pipeline)

            self.pre_pipeline = None  # Not used when per-split

        else:
            # Single pre-pipeline for all splits
            if isinstance(pre_pipeline, AugmentationPipeline):
                self.pre_pipeline = pre_pipeline
            else:
                self.pre_pipeline = self._build_pipeline_from_list(pre_pipeline)

            # Create dict for all splits
            self.pre_pipelines = {split_name: self.pre_pipeline for split_name in self.splits.keys()}

    def _apply_pre_pipelines(self):
        """
        Run the pre-pipelines over every split and cache the results.

        Separate from building them so that the kwargs can be validated and
        the seeds derived first: a typo should surface before minutes of coil
        combination, and a stochastic preprocessing step should replay.
        """
        if self.pre_pipelines is None:
            return

        print(f"Applying pre-pipeline to all data (this happens ONCE)...")

        # Apply pre-pipeline to each split and cache results
        self.preprocessed_splits = {}

        for split_name, (data, water) in self.splits.items():
            split_pre_pipeline = self.pre_pipelines[split_name]

            if split_pre_pipeline is None:
                # No pre-pipeline for this split
                print(f"  {split_name} split: No pre-pipeline")
                self.preprocessed_splits[split_name] = (data, water)
                continue

            if len(data) == 0:  # Empty split
                self.preprocessed_splits[split_name] = (data, water)
                continue

            print(f"  Processing {split_name} split ({len(data)} subjects)...")
            print(f"    Pre-pipeline: {[m.__class__.__name__ for m in split_pre_pipeline.steps]}")

            # Process each subject through pre-pipeline
            processed_data_list = []
            processed_water_list = [] if water is not None else None

            for i in range(len(data)):
                subj_data = data[i]
                subj_water = water[i] if water is not None else None

                # Apply pre-pipeline
                proc_data, proc_water = split_pre_pipeline(subj_data, subj_water)

                processed_data_list.append(proc_data.list()[0])
                if processed_water_list is not None and proc_water is not None:
                    processed_water_list.append(proc_water.list()[0])

            # Wrap processed data in target backend
            # NIfTI_MRS_Plus will handle conversion automatically
            processed_data = NIfTI_MRS_Plus(
                nifti_list=processed_data_list,
                backend=self.backend,  # Use target backend directly
                volatile=self.volatile
            )
            processed_water = None
            if processed_water_list:
                processed_water = NIfTI_MRS_Plus(
                    nifti_list=processed_water_list,
                    backend=self.backend,
                    volatile=self.volatile
                )

            # Store preprocessed data
            self.preprocessed_splits[split_name] = (processed_data, processed_water)

        print(f"✓ Pre-pipeline applied and cached!")

        # Replace splits with preprocessed versions
        self.splits = self.preprocessed_splits

    def _create_pipelines(self, pipeline, pipelines):
        """Create augmentation pipelines for each split."""
        self.pipelines = {}

        if pipelines is not None:
            # Per-split pipelines provided
            for split_name in self.splits.keys():
                split_pipeline = pipelines.get(split_name, None)
                if split_pipeline is None:
                    self.pipelines[split_name] = AugmentationPipeline([])
                elif isinstance(split_pipeline, AugmentationPipeline):
                    self.pipelines[split_name] = split_pipeline
                else:
                    # Build from list of module names
                    self.pipelines[split_name] = self._build_pipeline_from_list(split_pipeline)

        elif pipeline is not None:
            # Single pipeline for all splits
            if isinstance(pipeline, AugmentationPipeline):
                built_pipeline = pipeline
            else:
                built_pipeline = self._build_pipeline_from_list(pipeline)

            for split_name in self.splits.keys():
                self.pipelines[split_name] = built_pipeline

        else:
            # Default: processing for all splits, built like any named step so
            # that only the kwargs RawProcessor accepts reach it.
            default_pipeline = self._build_pipeline_from_list(['processing'])
            for split_name in self.splits.keys():
                self.pipelines[split_name] = default_pipeline

    #*******************#
    #   the registry    #
    #*******************#
    @classmethod
    def resolve_module(cls, name: str):
        """
        The class behind a registry name, and the arguments the name fixes.

        A registry value is a class, or a "(class, fixed_kwargs)" pair for a
        name that means one configuration of a class: 'baseline_bspline' is
        BaselineAugmentation with mode='bspline', and used to run the random
        walk because the alias set nothing.

        Args:
            name: A key of "AVAILABLE_MODULES".

        Returns:
            "(module_class, fixed_kwargs)", the kwargs a fresh dict.

        Raises:
            ValueError: For a name not in the registry, with close matches.
        """
        if name not in cls.AVAILABLE_MODULES:
            raise ValueError(
                f"Unknown module '{name}'. Available: {list(cls.AVAILABLE_MODULES.keys())}."
                f"{suggest_names(name, cls.AVAILABLE_MODULES)}"
            )
        entry = cls.AVAILABLE_MODULES[name]
        if isinstance(entry, tuple):
            module_class, fixed_kwargs = entry
            return module_class, dict(fixed_kwargs)
        return entry, {}

    @classmethod
    def accepted_parameters(cls, pipelines) -> set:
        """
        The kwargs the modules of a pipeline spec accept, before it is built.

        A dataset factory carries defaults that only mean something with a
        certain module present - a voxel size for the spatial augmentation,
        an absolute sigma for the noise - and a user who hands the factory an
        empty or different pipeline must not be refused for defaults they
        never asked for. This answers "would this kwarg reach anything" from
        the spec alone, so a factory can drop the defaults that would not.

        Args:
            pipelines: A pipeline spec as "Augmentrum" takes it: None, a list
                of entries, an "AugmentationPipeline", or a dict of those per
                split.

        Returns:
            The union of the constructor parameter names of every module named
            or instantiated in the spec, plus the sampling controls.
        """
        accepted = set(AugmentationPipeline.GLOBAL_KEYS)
        specs = list(pipelines.values()) if isinstance(pipelines, dict) else [pipelines]
        for spec in specs:
            if spec is None:
                accepted.update(constructor_params(RawProcessor))
            elif isinstance(spec, AugmentationPipeline):
                for step in spec.steps:
                    accepted.update(constructor_params(step))
            else:
                for entry in spec:
                    name, module, _ = cls._parse_pipeline_entry(entry)
                    if name == 'tap' or (name and name.startswith('tap:')):
                        continue
                    if module is None:
                        module, _ = cls.resolve_module(name)
                    accepted.update(constructor_params(module))
        return accepted

    @staticmethod
    def _parse_pipeline_entry(entry):
        """
        One pipeline entry as "(name, module, step_kwargs)".

        Accepted forms: a registry name, a module instance, a one-key dict
        "{name_or_module: {kwargs}}" and a "(name_or_module, {kwargs})" pair.
        Exactly one of *name* and *module* is set.

        Args:
            entry: The entry as the user wrote it.

        Returns:
            "(name, module, step_kwargs)".
        """
        step_kwargs = {}
        if isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(
                    f"A pipeline dict entry names one module, e.g. {{'noise': {{...}}}}; "
                    f"got {len(entry)} keys {list(entry)}."
                )
            (entry, step_kwargs), = entry.items()
        elif isinstance(entry, tuple):
            if len(entry) != 2 or not isinstance(entry[1], (dict, type(None))):
                raise ValueError(
                    f"A pipeline tuple entry is (name, {{kwargs}}), got {entry!r}."
                )
            entry, step_kwargs = entry

        step_kwargs = dict(step_kwargs or {})
        if isinstance(entry, str):
            return entry, None, step_kwargs
        if isinstance(entry, BaseModule):
            return None, entry, step_kwargs
        raise ValueError(
            f"Pipeline entries are module names, module instances, {{name: kwargs}} "
            f"dicts or (name, kwargs) tuples; got {entry!r}."
        )

    @staticmethod
    def _is_constructor_value(value) -> bool:
        """
        Whether a per-step value is handed to the constructor as given.

        A range is sampled per batch instead, and a list or tuple may carry
        nested ranges - echoes, peaks - that are injected per batch as well.
        Everything else, objects included, is what the constructor wants.
        """
        return not is_range(value) and not isinstance(value, (list, tuple))

    @staticmethod
    def _placeholder(value):
        """A representative constructor value for a range: its lower bound."""
        if is_range(value):
            low, high = value
            return low if low is not None else high
        return value

    def _check_required(self, name: str, module, provided) -> None:
        """
        Raise if a step is missing a parameter its configuration cannot do without.

        A module that declares "required_parameters()" names what its mode
        needs; a value counts as present when it is set on the instance or
        will arrive per batch from a range.

        Args:
            name: The registry name, for the message.
            module: The constructed step.
            provided: Names a global or per-step kwarg will inject.
        """
        required = getattr(module, 'required_parameters', None)
        if not callable(required):
            return
        names = tuple(required())
        if not names:
            return
        if any(getattr(module, n, None) is not None or n in provided for n in names):
            return
        options = ' or '.join(repr(n) for n in names)
        raise ValueError(
            f"'{name}' ({module.__class__.__name__}, mode={getattr(module, 'mode', None)!r}) "
            f"needs {options}. Pass a value or a range, e.g. "
            f"pipeline=[{{'{name}': {{'{names[0]}': ...}}}}] or {names[0]}=... globally."
        )

    def _build_pipeline_from_list(self, entries, seed=None) -> AugmentationPipeline:
        """
        Build a pipeline from a list of entries.

        Each entry is a registry name, a module instance, a one-key dict
        "{name: {kwargs}}" or a "(name, {kwargs})" pair. A named module is
        constructed from, in rising precedence, the builder's defaults, the
        global kwargs it accepts (scalars only - ranges are sampled per
        batch), the kwargs its registry alias fixes, and its own step kwargs.

        Args:
            entries: The pipeline as the user wrote it.
            seed: Optional seed for the pipeline; "_seed_streams" sets one later
                either way.

        Returns:
            The pipeline, carrying the per-step kwargs for batch sampling.
        """
        modules, names, step_kwargs = [], [], []

        # Default parameters for modules that require them at init
        # TODO: Maybe solve at module level instead
        DEFAULT_PARAMS = {
            'noise': {'sigma_frac': 0.02},
        }

        # Parameters that are alternative ways to say the same thing. A module
        # given two members of a group cannot tell which the caller meant and
        # rightly refuses, so a user-supplied member must suppress the default
        # for every other member rather than arriving alongside it. A ranged
        # member is constructed at its lower bound: the module resolves the
        # members in a fixed order at run time, so a default left in place would
        # win over the range injected per batch.
        EXCLUSIVE_GROUPS = [
            {'snr', 'snr_db', 'sigma', 'sigma_frac'},   # Noise
        ]

        for entry in entries:
            name, module, kwargs = self._parse_pipeline_entry(entry)

            if module is not None:
                # A ready instance; only its step kwargs can still be checked.
                accepted = constructor_params(module)
                unknown = [k for k in kwargs if k not in accepted]
                if unknown:
                    raise ValueError(
                        f"{module.__class__.__name__} does not accept {unknown}; it accepts "
                        f"{accepted}.{suggest_names(unknown[0], accepted)}"
                    )
                modules.append(module)
                names.append(module.__class__.__name__)
                step_kwargs.append(kwargs)
                continue

            # 'tap:<name>' names the tap; a bare 'tap' keeps the default name.
            if name == 'tap' or name.startswith('tap:'):
                modules.append(Tap(name=name.partition(':')[2] or 'tap'))
                names.append(name)
                step_kwargs.append({})
                continue

            module_class, fixed_kwargs = self.resolve_module(name)
            init_param_names = constructor_params(module_class)

            unknown = [k for k in kwargs if k not in init_param_names]
            if unknown:
                raise ValueError(
                    f"'{name}' ({module_class.__name__}) does not accept {unknown}; it "
                    f"accepts {init_param_names}.{suggest_names(unknown[0], init_param_names)}"
                )

            # Only inject scalar values (bool, int, float, str) from the global kwargs.
            # Range tuples (e.g. lb_hz=(0, 10)) are meant for runtime sampling only.
            global_kwargs = {k: v for k, v in self.kwargs.items() if k in init_param_names}
            scalar_kwargs = {k: v for k, v in global_kwargs.items()
                             if isinstance(v, (bool, int, float, str))}
            step_values = {k: v for k, v in kwargs.items() if self._is_constructor_value(v)}

            defaults = dict(DEFAULT_PARAMS.get(name, {}))
            given = {**global_kwargs, **kwargs}
            for group in EXCLUSIVE_GROUPS:
                members = group & set(given)
                if not members:
                    continue
                for key in group:
                    defaults.pop(key, None)
                for key in members:
                    if key not in scalar_kwargs and key not in step_values:
                        defaults[key] = self._placeholder(given[key])

            # Specific beats general: a step's own value over the alias's
            # fixed configuration, and that over a global scalar.
            construct_params = {**defaults, **scalar_kwargs, **fixed_kwargs, **step_values}
            module = module_class(**construct_params)
            self._check_required(name, module, set(given))

            modules.append(module)
            names.append(name)
            step_kwargs.append(kwargs)

        # Pass ALL user kwargs to Pipeline — it handles parameter extraction/sampling
        return AugmentationPipeline(
            modules,
            module_names=names,
            user_kwargs=self.kwargs,
            step_kwargs=step_kwargs,
            domain_planning=self.domain_planning,
            seed=seed,
        )

    def _create_modes(self, mode, modes):
        """Create augmentation modes for each split."""
        self.modes = {}

        if modes is not None:
            # Per-split modes
            for split_name in self.splits.keys():
                self.modes[split_name] = modes.get(split_name, 'on-the-fly')
        else:
            # Single mode for all
            for split_name in self.splits.keys():
                self.modes[split_name] = mode

    def _create_outputs(self, outputs):
        """
        Validate and store per-split outputs specs.

        An outputs spec is an arbitrarily nested tuple of stage tokens deciding
        what the dataloaders yield and in which structure: "'data'" / "'water'"
        are the pipeline end, "'<tap>'" / "'<tap>.water'" a tapped stage, e.g.
        "outputs=(('data', 'water'), ('clean', 'clean.water'))" for supervised
        (input, target) pairs. None keeps the classic "(data, water)" pair.
        Tokens are checked here, at construction, not at iteration time.
        """
        per_split = outputs if isinstance(outputs, dict) else \
            {split: outputs for split in self.splits}

        self.outputs = {}
        for split_name in self.splits:
            spec = per_split.get(split_name)
            if spec is not None:
                valid = {'data', 'water'}
                for tap in self.pipelines[split_name].tap_names:
                    valid.update({tap, f"{tap}.water"})
                unknown = [t for t in output_tokens(spec) if t not in valid]
                if unknown:
                    raise ValueError(
                        f"outputs for split '{split_name}' reference unknown stage(s) "
                        f"{unknown}. Available: {sorted(valid)}. Add a 'tap:<name>' "
                        f"to the pipeline to expose a stage."
                    )
            self.outputs[split_name] = spec

    #****************#
    #   validation   #
    #****************#
    def _validate_kwargs(self) -> None:
        """
        Reject any kwarg that no module in any pipeline accepts.

        Every unknown key used to be swallowed, so "sigma_frc=0.05" silently
        left the noise at its default and an arm of an ablation ran without
        the augmentation it was named after. The valid names are the union of
        the constructor parameters of every step in every pipeline and
        pre-pipeline, plus the sampling controls. RawProcessor's "**kwargs" is
        not a source of names: its constructor swallows them unread.
        """
        accepted = set(AugmentationPipeline.GLOBAL_KEYS)
        pipelines = list(self.pipelines.values())
        if self.pre_pipelines is not None:
            pipelines += [p for p in self.pre_pipelines.values() if p is not None]
        for pipe in pipelines:
            for step in pipe.steps:
                accepted.update(constructor_params(step))

        unknown = [key for key in self.kwargs if key not in accepted]
        if not unknown:
            return

        hints = []
        for key in unknown:
            hints.append(f"{key!r}{suggest_names(key, accepted)}")
        steps = sorted({step.__class__.__name__ for pipe in pipelines for step in pipe.steps})
        raise ValueError(
            f"No module in the pipelines accepts: {'; '.join(hints)}. "
            f"Pipelines contain {steps}, which together accept {sorted(accepted)}."
        )

    #*************#
    #   seeding   #
    #*************#
    def _seed_streams(self) -> None:
        """
        Derive every random stream from the root seed.

        Per split, by position: a generator for the subject draws (and the
        fixed-mode shuffle), one for the fixed-mode parameter draw, and a seed
        for the split's pipeline. A pipeline shared by several splits is seeded
        once, by the first split that holds it; a stochastic module keeps an
        explicit seed of its own (see "AugmentationPipeline.reseed").
        """
        self._rngs: Dict[str, np.random.Generator] = {}
        self._fixed_rngs: Dict[str, np.random.Generator] = {}
        self._fixed_params: Dict[str, dict] = {}

        seeded = set()
        for index, split in enumerate(self.splits):
            self._rngs[split] = np.random.default_rng(child_seed(self.seed, (index, 1)))
            self._fixed_rngs[split] = np.random.default_rng(child_seed(self.seed, (index, 2)))

            for role, table in ((0, self.pipelines), (3, self.pre_pipelines)):
                pipe = (table or {}).get(split)
                if pipe is None or id(pipe) in seeded:
                    continue
                seeded.add(id(pipe))
                pipe.reseed(child_seed(self.seed, (index, role)))

    def reseed(self, seed: int) -> 'Augmentrum':
        """
        Restart every random stream from a new root seed.

        Subject draws, ranged parameters, fixed-mode values and the modules'
        own perturbations all re-derive; the split itself stays, since
        membership is data, not a draw. This is also what each DataLoader
        worker calls on its copy, so workers stop producing the same batches.

        Args:
            seed: The new root seed.

        Returns:
            "self", so calls can be chained.
        """
        self.seed = int(seed)
        self._seed_streams()
        return self

    def _fixed_batch_params(self, split: str) -> dict:
        """
        The parameters a 'fixed' split uses, drawn once and kept.

        Drawn from the split's own generator rather than the pipeline's, so
        the value does not depend on which loaders were created before, and
        cached so every "dataloader()" call replays it. "reseed" clears it.

        Args:
            split: The split name.
        """
        if split not in self._fixed_params:
            self._fixed_params[split] = self.pipelines[split].sample_batch_parameters(
                self.batch_size, rng=self._fixed_rngs[split])
        return self._fixed_params[split]

    #***********#
    #   pools   #
    #***********#
    def _pool(self, split: str, data, water) -> Optional[TensorPool]:
        """
        The split's subjects stacked on the device, built on first use.

        None where a split cannot be pooled (the NIfTI-list backend, subjects
        of differing shapes), which keeps the per-batch copies. A pool that no
        longer stands for the split's objects - a replaced split - is rebuilt.
        """
        key = TensorPool.fingerprint(data, water)
        held = self._pools.get(split)
        if held is None or held[0] != key:
            device = self.device if self.backend == Backend.PYTORCH else None
            held = (key, TensorPool.build(data, water, self.backend, device))
            self._pools[split] = held
        return held[1]

    def refresh_pools(self) -> 'Augmentrum':
        """
        Drop the stacked pools, so the next batches restack the subjects.

        Needed only after writing new values into the subjects' NIfTI objects
        in place: a pool is a snapshot of them.
        """
        self._pools = {}
        return self

    def _get_dataloader(self, split: str = 'train', framework: str = None, shuffle: bool = None):
        """
        Get dataloader for a specific split.

        Args:
            split: 'train', 'val', or 'test'
            framework: 'pytorch', 'numpy', 'tensorflow', 'keras', 'jax'
            shuffle: Whether to shuffle (only for fixed mode)

        Returns:
            Framework-specific dataloader/generator
        """
        if split not in self.splits:
            raise ValueError(f"Split '{split}' not available. Available: {list(self.splits.keys())}")

        # Get data for this split
        data, water = self.splits[split]
        pipeline = self.pipelines[split]
        mode = self.modes[split]
        outputs = self.outputs[split]
        pool = self._pool(split, data, water)

        # Create generator based on mode
        if mode == 'on-the-fly' or mode == 'random':  # Support legacy 'random'
            # On-the-fly: pipeline samples randomly from ranges
            generator = create_random_generator(
                data=data,
                water=water,
                pipeline=pipeline,
                batch_size=self.batch_size,
                outputs=outputs,
                rng=self._rngs[split],
                pool=pool,
            )
        elif mode == 'fixed' or mode == 'deterministic':  # Support legacy 'deterministic'
            # Fixed: use exact values (modules will use fixed params)
            shuffle_val = shuffle if shuffle is not None else (split == 'train')

            # For fixed mode, we just iterate over subjects (no combinatorial explosion)
            generator = create_fixed_generator(
                data=data,
                water=water,
                pipeline=pipeline,
                batch_size=self.batch_size,
                shuffle=shuffle_val,
                outputs=outputs,
                rng=self._rngs[split],
                fixed_params=self._fixed_batch_params(split),
                pool=pool,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'on-the-fly' or 'fixed'")

        # Wrap for framework
        return wrap_generator_for_framework(generator, self.backend, framework,
                                            structured=outputs is not None)

    def dataloader(self, framework: str = None, split: str = 'train', shuffle: bool = None):
        """
        Get dataloader for any split (defaults to 'train').

        Args:
            framework: 'pytorch', 'numpy', 'tensorflow', 'keras', 'jax'
            split: Split name. Any key of "self.splits" — including custom
                   names created via "split_indices" (e.g. 'test_track1').
            shuffle: Whether to shuffle (only honoured in 'fixed' mode)

        Returns:
            Framework-specific dataloader/generator
        """
        return self._get_dataloader(split, framework, shuffle)

    def train_dataloader(self, framework: str = None, **kwargs):
        """Get training dataloader."""
        return self._get_dataloader('train', framework, kwargs.get('shuffle', True))

    def val_dataloader(self, framework: str = None, **kwargs):
        """Get validation dataloader."""
        return self._get_dataloader('val', framework, kwargs.get('shuffle', False))

    def test_dataloader(self, framework: str = None, **kwargs):
        """Get test dataloader."""
        return self._get_dataloader('test', framework, kwargs.get('shuffle', False))

    def as_torch_dataloader(self, split: str = 'train', **dataloader_kwargs):
        """
        Return a PyTorch DataLoader for the given split, ready for use with
        PyTorch Lightning's Trainer — no external wrapper class required.

        Each item yielded is a complex tensor (B, ..., N_time) directly
        consumable by FrameworkNN.  Batch size is controlled by the
        Augmentrum batch_size parameter set at construction.

        With "num_workers > 0" every worker receives a copy of this object and,
        left alone, would replay the same seeded streams - two workers, two
        identical batches at a time. A worker_init_fn therefore reseeds each
        copy from the root seed, the worker id and torch's per-epoch base seed
        (so that epochs differ too, and replay when torch is seeded). A
        worker_init_fn of your own is called after it.

        Args:
            split: 'train', 'val', or 'test'
            **dataloader_kwargs: Extra args forwarded to DataLoader
                                 (e.g. num_workers, pin_memory).

        Returns:
            torch.utils.data.DataLoader
        """
        import torch
        from torch.utils.data import DataLoader, IterableDataset, get_worker_info

        def _to_cfloat(tensor):
            if tensor is None:
                return None
            if not tensor.is_complex():
                tensor = tensor.to(torch.cfloat)
            return tensor

        class _AugIterableDataset(IterableDataset):
            def __init__(self_inner, aug, split_name):
                self_inner.aug = aug
                self_inner.split = split_name

            def __iter__(self_inner):
                aug, split_name = self_inner.aug, self_inner.split
                gen = aug._get_dataloader(split_name, framework='pytorch')
                if aug.outputs[split_name] is not None:
                    # Custom outputs: yield the spec's structure with complex
                    # tensors at the leaves.
                    for batch in gen:
                        yield map_structure(_to_cfloat, batch)
                    return
                for batch_data, _ in gen:
                    if batch_data is None:
                        continue
                    yield _to_cfloat(batch_data)

        user_init = dataloader_kwargs.pop('worker_init_fn', None)

        def _worker_init(worker_id):
            info = get_worker_info()
            aug = info.dataset.aug
            # info.seed is torch's base_seed + worker_id, and base_seed is
            # drawn afresh from torch's generator every epoch: mixing it in
            # keeps epochs distinct without a persistent worker.
            aug.reseed(child_seed(aug.seed, (1 + worker_id, int(info.seed) & 0xFFFFFFFF)))
            if user_init is not None:
                user_init(worker_id)

        return DataLoader(_AugIterableDataset(self, split), batch_size=None,
                          worker_init_fn=_worker_init, **dataloader_kwargs)

    def visualize_pipeline(self, split: str = 'train', detailed: bool = True) -> str:
        """
        Create a beautiful visualization of the augmentation pipeline.

        Args:
            split: Which split's pipeline to visualize
            detailed: If True, show module parameters

        Returns:
            Formatted string visualization
        """
        if split not in self.pipelines:
            return f"No pipeline for split '{split}'"

        pipeline = self.pipelines[split]

        # Build visualization
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + f"{'🔬 AUGMENTRUM PIPELINE 🔬':^78}" + "║")
        lines.append("╠" + "═" * 78 + "╣")

        # Info
        data, water = self.splits[split]
        n_subjects = len(data)
        n_water = len(water) if water is not None else 0

        # Build info lines
        info_line_1 = f"║  📊 Split: {split:20} Subjects: {n_subjects:4}  Mode: {self.modes[split]:15}║"
        lines.append(info_line_1)

        # Water info line
        if water is not None:
            water_info = f"Water: {n_water:4}"
        else:
            water_info = "Water: None"
        backend_info = f"Backend: {self.backend.name:15}"
        batch_info = f"Batch Size: {self.batch_size:4}"
        volatile_info = f"Volatile: {self.volatile!s:5}"

        info_line_2 = f"║  🎯 {backend_info} {batch_info}  {volatile_info}  ║"
        lines.append(info_line_2)

        if water is not None:
            water_line = f"║  💧 {water_info}" + " " * (73 - len(water_info)) + "║"
            lines.append(water_line)

        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║  🔄 Pipeline Steps:" + " " * 57 + "║")
        lines.append("║" + " " * 78 + "║")

        # Steps
        if len(pipeline.steps) == 0:
            lines.append("║   No augmentation modules configured" + " " * 40 + "║")
        else:
            for i, module in enumerate(pipeline.steps, 1):
                module_name = module.__class__.__name__
                emoji = self._get_module_emoji(module_name)
                lines.append(f"║   {i}. {emoji} {module_name:50}║")

                if detailed:
                    # Show key parameters
                    params = self._get_module_params(module)
                    if params:
                        lines.append(f"║      {params:70}║")

                if i < len(pipeline.steps):
                    lines.append("║      ↓" + " " * 71 + "║")

        lines.append("╚" + "═" * 78 + "╝")

        return "\n".join(lines)

    def _get_module_emoji(self, name: str) -> str:
        """Get emoji for module type."""
        emoji_map = {
            'CoilSampler': '📡',
            'AverageSampler': '🔁',
            'RawProcessor': '⚙️',
            'Noise': '🔊',
            'LineBroadening': '〰️',
            'BaselineAugmentation': '📈',
            'PhaseShift': '🔄',
            'FrequencyShift': '↔️',
            'ResidualWater': '💧',
            'SpuriousEchoes': '👻',
            'ArtificialPeaks': '⛰️',
            'EddyCurrent': '🌀',
            'Apodization': '✂️',
            'SpatialAugmentations': '🗺️',
        }
        return emoji_map.get(name, '🔧')

    def _get_module_params(self, module) -> str:
        """Get key parameters of a module by introspecting its attributes."""
        # List of common parameter names to display (in order of priority)
        param_names = [
            'n_coils', 'n_averages', 'mode',
            'sigma_frac', 'snr', 'sigma',
            'lb_hz', 'gb_hz',
            'zero_order_deg', 'first_order_deg', 'shift_hz',
            'baseline_frac', 'baseline_mode',
            'water_amp', 'water_width',
            'n_echoes', 'echo_time',
            'n_peaks', 'peak_shift',
            'eddy_std', 'apod_lb',
            'dim', 'prob', 'min_coils', 'max_coils',
            'max_z_angle_deg', 'zoom_min', 'zoom_max', 'shear_max',
        ]

        params = []
        for param_name in param_names:
            if hasattr(module, param_name):
                value = getattr(module, param_name)
                if value is not None and value is not False:
                    # Skip default/empty values
                    if isinstance(value, (int, float)) and value == 0:
                        continue
                    if isinstance(value, str) and value == '':
                        continue

                    # Use original parameter name
                    params.append(f"{param_name}={value}")

                    # Limit to 3 params for readability
                    if len(params) >= 3:
                        break

        return ", ".join(params) if params else ""

    def show_pipeline(self, split: str = 'train', detailed: bool = True):
        """
        Print a beautiful visualization of the pipeline.

        Args:
            split: Which split to visualize ('train', 'val', 'test')
            detailed: If True, show module parameters; if False, show only names

        Example:
            >>> augmenter = Augmentrum(...)
            >>> augmenter.show_pipeline('train', detailed=True)
        """
        print(self.visualize_pipeline(split=split, detailed=detailed))

    def add_callback(self, callback_fn):
        """
        Add a custom callback function that will be called after each augmentation.

        Args:
            callback_fn: Function with signature: fn(data, water, split_name, batch_idx) -> (data, water)

        Example:
            >>> def my_callback(data, water, split, batch_idx):
            ...     print(f"Processing batch {batch_idx} from {split}")
            ...     return data, water
            >>> augmenter.add_callback(my_callback)
        """
        self.callbacks.append(callback_fn)

    def clear_callbacks(self):
        """Remove all callbacks."""
        self.callbacks = []

    def update_module_params(self, split: str, module_name: str, **params):
        """
        Update parameters of a specific module in the pipeline on-the-fly.

        Args:
            split: Which split's pipeline to update ('train', 'val', 'test')
            module_name: Name of the module class (e.g., 'Noise')
            **params: Parameters to update

        Example:
            >>> # Change noise level dynamically
            >>> augmenter.update_module_params('train', 'Noise', sigma_frac=0.05)
            >>> # Change broadening parameters
            >>> augmenter.update_module_params('train', 'LineBroadening', lb_hz=(0, 10))
        """
        if split not in self.pipelines:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.pipelines.keys())}")

        pipeline = self.pipelines[split]
        updated = False

        for module in pipeline.steps:
            if module.__class__.__name__ == module_name:
                for key, value in params.items():
                    if hasattr(module, key):
                        setattr(module, key, value)
                        updated = True

        if not updated:
            raise ValueError(f"Module '{module_name}' not found in {split} pipeline or parameters don't exist")

    def get_stats(self, reset: bool = False) -> Dict[str, Any]:
        """
        Get augmentation statistics.

        Args:
            reset: If True, reset statistics after returning

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = augmenter.get_stats()
            >>> print(f"Generated {stats['batches_generated']} batches")
        """
        stats = self.stats.copy()
        if reset:
            self.reset_stats()
        return stats

    def reset_stats(self):
        """Reset all statistics counters."""
        self.stats = {
            'batches_generated': 0,
            'samples_generated': 0,
            'split_stats': {split: {'batches': 0, 'samples': 0} for split in self.splits.keys()}
        }

    def profile_pipeline(self, split: str = 'train', n_batches: int = 10):
        """
        Profile the pipeline performance.

        Args:
            split: Which split to profile
            n_batches: Number of batches to process for profiling

        Returns:
            Dictionary with timing statistics

        Example:
            >>> profile = augmenter.profile_pipeline('train', n_batches=100)
            >>> print(f"Average time per batch: {profile['avg_time_per_batch']:.3f}s")
        """
        import time

        dataloader = self.dataloader(split=split, framework='numpy')

        times = []
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            start = time.time()
            # Process through pipeline (already done in dataloader)
            elapsed = time.time() - start
            times.append(elapsed)

        import numpy as np
        return {
            'split': split,
            'n_batches': len(times),
            'total_time': sum(times),
            'avg_time_per_batch': np.mean(times),
            'std_time_per_batch': np.std(times),
            'min_time': min(times),
            'max_time': max(times),
        }

    def export_batch(self, output_dir, format='nifti-mrs', metadata=None, split='train',
                     n_batches=1, prefix='augmented', save_water=True):
        """
        Export augmented batches to disk in various formats.

        This method enables efficient batch export of augmented datasets in formats
        compatible with existing MRS software (FSL-MRS, Osprey, LCModel, etc.).

        Currently supports:
        - NIFTI-MRS: Individual NIFTI files (one per augmented spectrum) ✓
        - HDF5: Single file with all metadata (data only, limited reconstruction) ✓

        Planned future formats:
        - LCModel RAW format
        - Osprey-compatible structure

        Args:
            output_dir: Directory to save exported data
            format: Output format ('nifti-mrs', 'hdf5')
            metadata: Optional metadata dictionary to embed
            split: Which data split to export ('train', 'val', 'test')
            n_batches: Number of batches to export
            prefix: Filename prefix (default: 'augmented')
            save_water: Whether to save water reference if available

        Returns:
            Dictionary with saved file paths and export info

        Example:
            >>> info = augmenter.export_batch(
            ...     'output/augmented_dataset',
            ...     format='nifti-mrs',
            ...     n_batches=10,
            ...     prefix='aug'
            ... )
            >>> print(f"Saved {info['n_files']} NIFTI-MRS files")
        """
        from pathlib import Path
        import numpy as np

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        format = format.lower()
        if format not in ['nifti-mrs', 'hdf5']:
            raise ValueError(f"Unsupported format: {format}. Use 'nifti-mrs' or 'hdf5'")

        # Get dataloader for the split
        if split == 'train':
            dataloader = self.train_dataloader(framework='numpy')
        elif split == 'val':
            dataloader = self.val_dataloader(framework='numpy')
        elif split == 'test':
            dataloader = self.test_dataloader(framework='numpy')
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")

        saved_files = []
        saved_water_files = []
        batch_count = 0
        total_spectra = 0

        print(f"Exporting {n_batches} batch(es) to {format.upper()} format...")

        outputs_spec = self.outputs[split]

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            if outputs_spec is None:
                data_batch, water_batch = batch
                extra_stages = []
            else:
                # Custom outputs: match the yielded structure against the spec's
                # tokens. 'data'/'water' keep their classic roles; every other
                # stage is exported under its own token.
                named = dict(zip(output_tokens(outputs_spec), flatten_structure(batch)))
                data_batch = named.pop('data', None)
                water_batch = named.pop('water', None)
                if data_batch is None:
                    raise ValueError(
                        f"export_batch needs the 'data' token in the outputs spec for "
                        f"split '{split}'; got {output_tokens(outputs_spec)}."
                    )
                extra_stages = [(token, arr) for token, arr in named.items()
                                if arr is not None]

            batch_count += 1

            # Remove spatial dimensions if present
            while data_batch.ndim > 2:
                data_batch = np.squeeze(data_batch)

            # Ensure 2D [batch, time]
            if data_batch.ndim == 1:
                data_batch = data_batch[np.newaxis, :]

            n_spectra_in_batch = data_batch.shape[0]

            if format == 'nifti-mrs':
                # Save each spectrum as individual NIFTI-MRS file
                # We need to reconstruct NIFTI_MRS objects from the augmented data

                # Get reference NIFTI for metadata
                ref_data, _ = self.splits[split]
                ref_nifti = ref_data[0] if len(ref_data) > 0 else None

                if ref_nifti is None:
                    raise ValueError("No reference NIFTI-MRS data available for export")

                # Import here to avoid circular dependency
                from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

                for spec_idx in range(n_spectra_in_batch):
                    spectrum = data_batch[spec_idx]

                    # Ensure spectrum has proper dimensions for NIFTI-MRS (at least 4D: x, y, z, time)
                    if spectrum.ndim == 1:
                        # Add spatial dimensions: (1, 1, 1, time)
                        spectrum = spectrum.reshape(1, 1, 1, -1)
                    elif spectrum.ndim == 2:
                        # Assume (coils, time) - add spatial: (1, 1, 1, time, coils)
                        spectrum = spectrum.T.reshape(1, 1, 1, spectrum.shape[1], spectrum.shape[0])

                    # Create new NIFTI_MRS object with augmented data using gen_nifti_mrs
                    # Get parameters from reference
                    dwelltime = ref_nifti.dwelltime if hasattr(ref_nifti, 'dwelltime') else 1/2000
                    spec_freq = ref_nifti.spectrometer_frequency[0] if hasattr(ref_nifti, 'spectrometer_frequency') else 123.0

                    # Create NIFTI_MRS with proper structure
                    new_nifti = gen_nifti_mrs(spectrum, dwelltime, spec_freq)

                    # Add provenance metadata using add_hdr_field
                    from datetime import datetime
                    processing_entry = {
                        'Time': datetime.now().isoformat(),
                        'Program': 'augmentrum',
                        'Version': __version__,
                        'Method': 'export_batch',
                        'Details': {
                            'split': split,
                            'batch': batch_idx,
                            'spectrum_in_batch': spec_idx
                        }
                    }

                    # Add user metadata if provided
                    if metadata:
                        processing_entry['UserMetadata'] = metadata

                    # Add to ProcessingApplied list
                    new_nifti.add_hdr_field('ProcessingApplied', [processing_entry])

                    # Save file
                    global_idx = total_spectra + spec_idx
                    filename = f"{prefix}_{global_idx:06d}.nii.gz"
                    filepath = output_path / filename
                    new_nifti.save(str(filepath))
                    saved_files.append(str(filepath))

                # Save water reference if available
                if save_water and water_batch is not None:
                    while water_batch.ndim > 2:
                        water_batch = np.squeeze(water_batch)
                    if water_batch.ndim == 1:
                        water_batch = water_batch[np.newaxis, :]

                    for spec_idx in range(water_batch.shape[0]):
                        water_spectrum = water_batch[spec_idx]

                        # Ensure proper dimensions
                        if water_spectrum.ndim == 1:
                            water_spectrum = water_spectrum.reshape(1, 1, 1, -1)
                        elif water_spectrum.ndim == 2:
                            water_spectrum = water_spectrum.T.reshape(1, 1, 1, water_spectrum.shape[1], water_spectrum.shape[0])

                        # Create water NIFTI_MRS using gen_nifti_mrs
                        dwelltime = ref_nifti.dwelltime if hasattr(ref_nifti, 'dwelltime') else 1/2000
                        spec_freq = ref_nifti.spectrometer_frequency[0] if hasattr(ref_nifti, 'spectrometer_frequency') else 123.0

                        water_nifti = gen_nifti_mrs(water_spectrum, dwelltime, spec_freq)

                        # Add water-specific provenance
                        from datetime import datetime
                        water_processing = {
                            'Time': datetime.now().isoformat(),
                            'Program': 'augmentrum',
                            'Version': __version__,
                            'Method': 'export_batch',
                            'DataType': 'water_reference',
                            'Details': {
                                'split': split,
                                'batch': batch_idx,
                                'spectrum_in_batch': spec_idx
                            }
                        }
                        water_nifti.add_hdr_field('ProcessingApplied', [water_processing])

                        global_idx = total_spectra + spec_idx
                        water_filename = f"{prefix}_water_{global_idx:06d}.nii.gz"
                        water_filepath = output_path / water_filename
                        water_nifti.save(str(water_filepath))
                        saved_water_files.append(str(water_filepath))

                # Save any tapped stages under their token names
                for token, stage_batch in extra_stages:
                    while stage_batch.ndim > 2:
                        stage_batch = np.squeeze(stage_batch)
                    if stage_batch.ndim == 1:
                        stage_batch = stage_batch[np.newaxis, :]

                    for spec_idx in range(stage_batch.shape[0]):
                        stage_spectrum = stage_batch[spec_idx]
                        if stage_spectrum.ndim == 1:
                            stage_spectrum = stage_spectrum.reshape(1, 1, 1, -1)
                        elif stage_spectrum.ndim == 2:
                            stage_spectrum = stage_spectrum.T.reshape(
                                1, 1, 1, stage_spectrum.shape[1], stage_spectrum.shape[0])

                        dwelltime = ref_nifti.dwelltime if hasattr(ref_nifti, 'dwelltime') else 1/2000
                        spec_freq = ref_nifti.spectrometer_frequency[0] if hasattr(ref_nifti, 'spectrometer_frequency') else 123.0

                        stage_nifti = gen_nifti_mrs(stage_spectrum, dwelltime, spec_freq)

                        global_idx = total_spectra + spec_idx
                        stage_filename = (f"{prefix}_{token.replace('.', '_')}_"
                                          f"{global_idx:06d}.nii.gz")
                        stage_filepath = output_path / stage_filename
                        stage_nifti.save(str(stage_filepath))
                        saved_files.append(str(stage_filepath))

            elif format == 'hdf5':
                # Save batch to HDF5
                import h5py

                hdf5_path = output_path / f"{prefix}_batch_{batch_idx:04d}.h5"

                with h5py.File(hdf5_path, 'w') as f:
                    # Store data
                    f.create_dataset('data', data=data_batch, compression='gzip')

                    if water_batch is not None and save_water:
                        f.create_dataset('water', data=water_batch, compression='gzip')

                    for token, stage_batch in extra_stages:
                        f.create_dataset(token, data=stage_batch, compression='gzip')

                    # Store metadata
                    f.attrs['augmentrum_version'] = __version__
                    f.attrs['split'] = split
                    f.attrs['batch_idx'] = batch_idx
                    f.attrs['n_spectra'] = n_spectra_in_batch

                    if metadata:
                        import json
                        f.attrs['user_metadata'] = json.dumps(metadata)

                saved_files.append(str(hdf5_path))

            total_spectra += n_spectra_in_batch
            print(f"  Batch {batch_idx + 1}/{n_batches}: Saved {n_spectra_in_batch} spectra")

        result = {
            'format': format,
            'output_dir': str(output_path),
            'n_batches': batch_count,
            'n_spectra': total_spectra,
            'n_files': len(saved_files),
            'files': saved_files
        }

        if saved_water_files:
            result['n_water_files'] = len(saved_water_files)
            result['water_files'] = saved_water_files

        print(f"✓ Export complete: {total_spectra} spectra in {len(saved_files)} file(s)")

        return result

    def __repr__(self):
        n_total = sum(len(v[0]) for v in self.splits.values())
        splits_info = ", ".join([f"{k}={len(v[0])}" for k, v in self.splits.items()])
        return (f"Augmentrum({n_total} subjects, splits=[{splits_info}], "
                f"backend={self.backend.name}, batch_size={self.batch_size})")

    def __str__(self):
        """User-friendly string representation."""
        n_total = sum(len(v[0]) for v in self.splits.values())
        splits_detail = ", ".join([f"{k}:{len(v[0])}" for k, v in self.splits.items()])
        return f"Augmentrum({n_total} subjects | {splits_detail} | {self.backend.name})"
