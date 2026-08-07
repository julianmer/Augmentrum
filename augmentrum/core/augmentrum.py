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
import warnings
from typing import List, Optional, Sequence, Union, Dict, Any
from augmentrum import __version__
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.core.pipeline import AugmentationPipeline

# Import helper functions from dataset_utils
from augmentrum.core.dataset_utils import (
    create_random_generator,
    create_fixed_generator,
    create_deterministic_generator,  # Legacy support
    wrap_generator_for_framework,
)
from augmentrum.sampling.subject_splitter import SubjectSplitter

# Processing modules
from augmentrum.processing.nifti_raw_processor import NIfTI_RawProcessor
from augmentrum.sampling.coil_average_sampler import CoilAverageSampler
from augmentrum.sampling.kspace_sampling import KspaceUndersampling

# Augmentation modules
from augmentrum.augmentation.amplitude_scaling import AmplitudeScaling
from augmentrum.augmentation.gaussian_noise import GaussianNoise
from augmentrum.augmentation.line_broadening import LineBroadening
from augmentrum.augmentation.baseline_augmentation import BaselineAugmentation
from augmentrum.augmentation.residual_water import ResidualWater
from augmentrum.augmentation.spurious_echoes import SpuriousEchoes
from augmentrum.augmentation.artificial_peaks import ArtificialPeaks
from augmentrum.augmentation.eddy_current import EddyCurrent
from augmentrum.augmentation.apodization import Apodization
from augmentrum.augmentation.phase_frequency import PhaseShift, FrequencyShift
from augmentrum.augmentation.spatial_augmentations import SpatialAugmentations
from augmentrum.augmentation.zero_fill import ZeroFill


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
    - Optional train/val/test splitting
    - Flexible augmentation pipelines (different per split)
    - On-the-fly (random) or fixed (exact) augmentation parameters
    - Multi-backend support (PyTorch, NumPy, TensorFlow, JAX)

    Example 1: On-the-fly augmentation with RANGE sampling
        "`python
        augmenter = Augmentrum(
            data=nifti_list,
            water=water_list,
            pipeline=['coil_sampling', 'processing', 'noise', 'line_broadening'],
            mode='on-the-fly',
            n_coils=(1, 8),         # Random 1-8 coils
            n_averages=(4, 16),     # Random 4-16 averages
            sigma_frac=(0.01, 0.05),  # Random noise 1-5%
            lb_hz=(0, 10),          # Random broadening 0-10 Hz
            phase0_deg=(-180, 180), # Random phase -180 to 180°
            param_distribution='uniform',  # How to sample from ranges (default)
            batch_size=32,
            backend='numpy'
        )

        # Each batch will have DIFFERENT random augmentations!
        for batch_data, batch_water in augmenter.dataloader(framework='numpy'):
            train_model(batch_data)
        "`

    Example 2: Fixed augmentation (exact values)
        "`python
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

        # Each batch will have IDENTICAL augmentations
        for batch_data, batch_water in augmenter.dataloader():
            validate_model(batch_data)
        "`

    Example 3: Gaussian distribution sampling
        "`python
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
        "`

    Example 3: With train/val/test splitting
        "`python
        augmenter = Augmentrum(
            data=nifti_list,
            water=water_list,
            split_fractions={'val': 0.1, 'test': 0.1},
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
            # Training uses ranges
            n_coils=(1, 8),
            sigma_frac=(0.01, 0.05),
            # But validation/test use fixed values (extracted from ranges)
            batch_size=32,
            backend='pytorch'
        )

        train_dl = augmenter.train_dataloader()
        val_dl = augmenter.val_dataloader()
        "`
    """

    # Available augmentation modules
    AVAILABLE_MODULES = {
        # Processing
        'coil_sampling': CoilAverageSampler,
        'average_sampling': CoilAverageSampler,
        'processing': NIfTI_RawProcessor,

        # Noise
        'noise': GaussianNoise,
        'gaussian_noise': GaussianNoise,

        # Amplitude
        'amplitude': AmplitudeScaling,
        'amplitude_scaling': AmplitudeScaling,

        # Line broadening
        'line_broadening': LineBroadening,
        'broadening': LineBroadening,

        # Baseline
        'baseline': BaselineAugmentation,
        'baseline_random_walk': BaselineAugmentation,
        'baseline_bspline': BaselineAugmentation,
        'baseline_polynomial': BaselineAugmentation,

        # Phase & Frequency
        'phase': PhaseShift,
        'phase_shift': PhaseShift,
        'frequency_shift': FrequencyShift,

        # Artifacts
        'residual_water': ResidualWater,
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
    }

    def __init__(
        self,
        data: Union[List, NIfTI_MRS_Plus],
        water: Optional[Union[List, NIfTI_MRS_Plus]] = None,

        # Splitting
        split_fractions: Optional[Dict[str, float]] = None,  # e.g., {'val': 0.1, 'test': 0.1}
        split_indices: Optional[Dict[str, Sequence[int]]] = None,  # explicit, overrides fractions
        seed: int = 42,

        # Pre-processing (applied once and cached)
        pre_pipeline: Optional[Union[List, AugmentationPipeline, Dict[str, Union[List, AugmentationPipeline]]]] = None,  # Fixed preprocessing

        # Augmentation
        pipeline: Optional[Union[List, AugmentationPipeline]] = None,  # Single pipeline
        pipelines: Optional[Dict[str, Union[List, AugmentationPipeline]]] = None,  # Per-split pipelines

        # Augmentation mode
        mode: str = 'on-the-fly',  # 'on-the-fly' or 'fixed'
        modes: Optional[Dict[str, str]] = None,  # Per-split modes

        # General
        batch_size: int = 16,
        backend: Union[str, Backend] = 'pytorch',
        device: Optional[str] = None,  # 'cuda', 'cpu', or None (auto-detect)
        volatile: bool = False,

        **kwargs  # Module-specific parameters
    ):
        """
        Initialize Augmentrum.

        Args:
            data: List of NIFTI_MRS objects or NIfTI_MRS_Plus
            water: Optional water references
            split_fractions: Dict like {'val': 0.1, 'test': 0.1}, train gets rest
            seed: Random seed for splitting
            pre_pipeline: Fixed preprocessing applied ONCE and cached. Can be:
                         - List of module names: ['coil_sampling', 'processing'] (same for all splits)
                         - AugmentationPipeline object (same for all splits)
                         - Dict mapping split names to pipelines (different per split):
                           {'train': ['coil_sampling', 'processing'], 'val': ['processing'], 'test': None}
                         These steps run before the main pipeline and results are stored.
                         Useful for expensive operations that don't need randomization.
            pipeline: Single pipeline for all splits (list of module names or AugmentationPipeline)
            pipelines: Dict mapping split names to pipelines (overrides 'pipeline')
            mode: Single mode for all splits:
                  'on-the-fly' = Random sampling from ranges (e.g., n_coils=(1,8) picks randomly)
                  'fixed' = Use exact values provided (e.g., n_coils=4 always uses 4 coils)
            modes: Dict mapping split names to modes (overrides 'mode')
            batch_size: Batch size
            backend: 'pytorch', 'numpy', 'tensorflow', 'keras', 'jax', or Backend enum
            device: Device for PyTorch backend ('cuda', 'cpu', etc.). Note: Use .to() method
                    on batches for GPU support instead of this parameter.
            volatile: Skip metadata updates for speed
            **kwargs: Module-specific parameters, including:
                ALL PARAMETERS support both tuple ranges and exact values:
                  - Tuple (min, max) = randomly sample from range
                  - Scalar (float/int) = use exact value

                SAMPLING PARAMETERS:
                  - n_coils: (1, 8) or 4
                  - n_averages: (4, 16) or 8

                AUGMENTATION PARAMETERS
                  - sigma_frac: (0.01, 0.05) or 0.03
                  - lb_hz: (0, 10) or 5.0
                  - gb_hz: (0, 5) or 2.0
                  - phase0_deg: (-180, 180) or 0.0
                  - phase1_deg: (-90, 90) or 0.0
                  - shift_hz: (-5, 5) or 0.0
                  - baseline_frac: (0.01, 0.1) or 0.05
                  - water_amp: (0.05, 0.2) or 0.1
                  - eddy_std: (0.3, 1.0) or 0.6
                  - ... etc.

                DISTRIBUTION (for sampling from ranges):
                  - param_distribution: 'uniform' (default), 'gaussian', 'exponential', 'beta'
                      Global default for ALL parameters

                  - param_distributions: Dict mapping parameter names to specific distributions
                      Per-parameter control (overrides param_distribution)
                      Examples:
                        param_distributions={
                            'sigma_frac': 'gaussian',  # Gaussian for noise
                            'lb_hz': 'exponential',    # Exponential for broadening
                            'phase0_deg': 'uniform'    # Uniform for phase
                        }

                  Note: If neither specified, default is 'uniform'

                PROCESSING OPTIONS:
                  - coil_method: 'fsl-mrs', 'adaptive'
                  - baseline_mode: 'random_walk', 'bspline', 'polynomial'
                  - ...
        """
        # Convert backend
        if isinstance(backend, str):
            self.backend = Backend[backend.upper()]
        else:
            self.backend = backend

        self.batch_size = batch_size
        self.device = device  # Store device ('cuda', 'cpu', or None)
        self.volatile = volatile
        self.kwargs = kwargs

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

        # Handle splitting
        if split_indices is not None:
            self._create_splits_from_indices(split_indices)
        elif split_fractions is not None:
            self._create_splits(split_fractions, seed)
        else:
            # No splitting - all data goes to train, but create empty val/test
            empty_data = NIfTI_MRS_Plus(nifti_list=[], backend=self.backend, volatile=self.volatile)
            self.splits = {
                'train': (self.data_all, self.water_all),
                'val': (empty_data, None),
                'test': (empty_data, None)
            }

        # Handle pre-pipeline (applied once and cached)
        self._create_pre_pipeline(pre_pipeline)

        # Handle pipelines
        self._create_pipelines(pipeline, pipelines)

        # Handle modes
        self._create_modes(mode, modes)

        # Additional features for flexibility
        self.callbacks = []  # Custom augmentation callbacks
        self.stats = {  # Statistics tracking
            'batches_generated': 0,
            'samples_generated': 0,
            'split_stats': {split: {'batches': 0, 'samples': 0} for split in self.splits.keys()}
        }

    def _create_splits(self, split_fractions: Dict[str, float], seed: int):
        """Create train/val/test splits."""
        val_frac = split_fractions.get('val', 0.0)
        test_frac = split_fractions.get('test', 0.0)

        splitter = SubjectSplitter(
            self.data_all.list(),
            self.water_all.list() if self.water_all is not None else None,
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac
        )

        splits_raw = splitter.split()

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
        if pre_pipeline is None:
            self.pre_pipeline = None
            self.pre_pipelines = None
            self.preprocessed_splits = None
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
            # Default: processing for all splits
            default_pipeline = AugmentationPipeline([
                NIfTI_RawProcessor(**self.kwargs)
            ])
            for split_name in self.splits.keys():
                self.pipelines[split_name] = default_pipeline

    def _build_pipeline_from_list(self, module_names: List[str]) -> AugmentationPipeline:
        """Build pipeline from list of module name strings."""
        import inspect

        modules = []

        # Default parameters for modules that require them at init
        # TODO: Maybe solve at module level instead
        DEFAULT_PARAMS = {
            'noise':          {'sigma_frac': 0.02},
            'gaussian_noise': {'sigma_frac': 0.02},
        }

        # Parameters that are alternative ways to say the same thing. A module
        # given two members of a group cannot tell which the caller meant and
        # rightly refuses, so a user-supplied member must suppress the default
        # for every other member rather than arriving alongside it.
        EXCLUSIVE_GROUPS = [
            {'snr', 'snr_db', 'sigma', 'sigma_frac'},   # GaussianNoise
        ]

        for name in module_names:
            if name not in self.AVAILABLE_MODULES:
                raise ValueError(f"Unknown module '{name}'. Available: {list(self.AVAILABLE_MODULES.keys())}")

            module_class = self.AVAILABLE_MODULES[name]

            # --- Introspect constructor to find which user kwargs apply ---
            try:
                sig = inspect.signature(module_class.__init__)
                init_param_names = [
                    p.name for p in sig.parameters.values()
                    if p.name not in ('self', 'args', 'kwargs')
                ]
            except Exception:
                init_param_names = []

            # Only inject scalar values (bool, int, float, str) from user_kwargs.
            # Range tuples (e.g. lb_hz=(0, 10)) are meant for runtime sampling only.
            scalar_kwargs = {
                k: v for k, v in self.kwargs.items()
                if k in init_param_names and isinstance(v, (bool, int, float, str))
            }

            defaults = dict(DEFAULT_PARAMS.get(name, {}))
            for group in EXCLUSIVE_GROUPS:
                if group & set(scalar_kwargs):
                    for key in group:
                        defaults.pop(key, None)

            construct_params = {**defaults, **scalar_kwargs}
            modules.append(module_class(**construct_params))

        # Pass ALL user kwargs to Pipeline — it handles parameter extraction/sampling
        return AugmentationPipeline(
            modules,
            module_names=module_names,
            user_kwargs=self.kwargs
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

        # Create generator based on mode
        if mode == 'on-the-fly' or mode == 'random':  # Support legacy 'random'
            # On-the-fly: pipeline samples randomly from ranges
            generator = create_random_generator(
                data=data,
                water=water,
                pipeline=pipeline,
                batch_size=self.batch_size
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
                shuffle=shuffle_val
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'on-the-fly' or 'fixed'")

        # Wrap for framework
        return wrap_generator_for_framework(generator, self.backend, framework)

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

        Args:
            split: 'train', 'val', or 'test'
            **dataloader_kwargs: Extra args forwarded to DataLoader
                                 (e.g. num_workers, pin_memory).

        Returns:
            torch.utils.data.DataLoader
        """
        import torch
        from torch.utils.data import DataLoader, IterableDataset

        _aug   = self
        _split = split

        class _AugIterableDataset(IterableDataset):
            def __iter__(self_inner):
                gen = _aug._get_dataloader(_split, framework='pytorch')
                for batch_data, _ in gen:
                    if batch_data is None:
                        continue
                    if not batch_data.is_complex():
                        batch_data = batch_data.to(torch.cfloat)
                    yield batch_data

        return DataLoader(_AugIterableDataset(), batch_size=None, **dataloader_kwargs)

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
            'CoilAverageSampler': '📡',
            'NIfTI_RawProcessor': '⚙️',
            'GaussianNoise': '🔊',
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
            module_name: Name of the module class (e.g., 'GaussianNoise')
            **params: Parameters to update

        Example:
            >>> # Change noise level dynamically
            >>> augmenter.update_module_params('train', 'GaussianNoise', sigma_frac=0.05)
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
        for i, (data, water) in enumerate(dataloader):
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

        for batch_idx, (data_batch, water_batch) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

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

            elif format == 'hdf5':
                # Save batch to HDF5
                import h5py

                hdf5_path = output_path / f"{prefix}_batch_{batch_idx:04d}.h5"

                with h5py.File(hdf5_path, 'w') as f:
                    # Store data
                    f.create_dataset('data', data=data_batch, compression='gzip')

                    if water_batch is not None and save_water:
                        f.create_dataset('water', data=water_batch, compression='gzip')

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
