####################################################################################################
#                                         augmentrum.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#          J. T. LaMaster (jlamaste@gmail.com)                                                     #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Main Augmentrum class - backend-agnostic MRS data augmentation                          #
#                                                                                                  #
####################################################################################################

from typing import List, Optional, Union, Dict, Any, Tuple
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.core.pipeline import AugmentationPipeline

# Import helper functions from dataset_utils
from augmentrum.core.dataset_utils import (
    create_random_generator,
    create_deterministic_generator,
    wrap_generator_for_framework,
)
from augmentrum.sampling.subject_splitter import SubjectSplitter

# Processing modules
from augmentrum.processing.nifti_raw_processor import NIfTI_RawProcessor
from augmentrum.sampling.coil_average_sampler import CoilAverageSampler

# Augmentation modules
from augmentrum.augmentation.gaussian_noise import GaussianNoise
from augmentrum.augmentation.line_broadening import LineBroadening
from augmentrum.augmentation.baseline_augmentation import BaselineAugmentation
from augmentrum.augmentation.residual_water import ResidualWater
from augmentrum.augmentation.spurious_echoes import SpuriousEchoes
from augmentrum.augmentation.artificial_peaks import ArtificialPeaks
from augmentrum.augmentation.eddy_current import EddyCurrent
from augmentrum.augmentation.apodization import Apodization
from augmentrum.augmentation.phase_frequency import PhaseShift, FrequencyShift


class Augmentrum:
    """
    Main Augmentrum class - backend-agnostic MRS data augmentation.

    Provides easy-to-use interface for:
    - Loading NIFTI_MRS data (metabolite + water)
    - Optional train/val/test splitting
    - Flexible augmentation pipelines (different per split)
    - Random or deterministic sampling modes
    - Multi-backend support (PyTorch, NumPy, TensorFlow, JAX)

    Example 1: Simple augmentation (no splitting)
        ```python
        augmenter = Augmentrum(
            data=nifti_list,
            water=water_list,
            pipeline=['processing', 'noise', 'line_broadening'],
            mode='random',
            batch_size=32,
            backend='numpy'
        )

        # Get dataloader
        for batch_data, batch_water in augmenter.dataloader(framework='numpy'):
            train_model(batch_data)
        ```

    Example 2: With train/val/test splitting
        ```python
        augmenter = Augmentrum(
            data=nifti_list,
            water=water_list,
            split_fractions={'val': 0.1, 'test': 0.1},  # train gets rest (0.8)
            pipelines={
                'train': ['processing', 'noise', 'line_broadening', 'baseline'],
                'val': ['processing'],
                'test': None
            },
            modes={
                'train': 'random',
                'val': 'deterministic',
                'test': 'deterministic'
            },
            batch_size=32,
            backend='pytorch'
        )

        train_dl = augmenter.train_dataloader()
        val_dl = augmenter.val_dataloader()
        ```
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
        'apodization': Apodization,
        'apod': Apodization,
    }

    def __init__(
        self,
        data: Union[List, NIfTI_MRS_Plus],
        water: Optional[Union[List, NIfTI_MRS_Plus]] = None,

        # Splitting
        split_fractions: Optional[Dict[str, float]] = None,  # e.g., {'val': 0.1, 'test': 0.1}
        seed: int = 42,

        # Augmentation
        pipeline: Optional[Union[List, AugmentationPipeline]] = None,  # Single pipeline
        pipelines: Optional[Dict[str, Union[List, AugmentationPipeline]]] = None,  # Per-split pipelines

        # Sampling mode
        mode: str = 'random',  # 'random' or 'deterministic'
        modes: Optional[Dict[str, str]] = None,  # Per-split modes

        # Sampling parameters
        n_coils: Tuple[Optional[int], Optional[int]] = (1, None),
        n_averages: Tuple[Optional[int], Optional[int]] = (1, None),

        # General
        batch_size: int = 16,
        backend: Union[str, Backend] = 'pytorch',
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
            pipeline: Single pipeline for all splits (list of module names or AugmentationPipeline)
            pipelines: Dict mapping split names to pipelines (overrides 'pipeline')
            mode: Single mode for all splits ('random' or 'deterministic')
            modes: Dict mapping split names to modes (overrides 'mode')
            n_coils: (min, max) coils to sample. None means use all.
            n_averages: (min, max) averages to sample. None means use all.
            batch_size: Batch size
            backend: 'pytorch', 'numpy', 'tensorflow', 'keras', 'jax', or Backend enum
            volatile: Skip metadata updates for speed
            **kwargs: Additional module parameters (e.g., coil_method='mean')
        """
        # Convert backend
        if isinstance(backend, str):
            self.backend = Backend[backend.upper()]
        else:
            self.backend = backend

        self.batch_size = batch_size
        self.n_coils = n_coils
        self.n_averages = n_averages
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
        if split_fractions is not None:
            self._create_splits(split_fractions, seed)
        else:
            # No splitting - all data is "train"
            self.splits = {
                'train': (self.data_all, self.water_all)
            }

        # Handle pipelines
        self._create_pipelines(pipeline, pipelines)

        # Handle modes
        self._create_modes(mode, modes)

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
            # Default: coil_sampling + processing for all splits
            default_pipeline = AugmentationPipeline([
                CoilAverageSampler(n_coils=self.n_coils, n_averages=self.n_averages),
                NIfTI_RawProcessor(**self.kwargs)
            ])
            for split_name in self.splits.keys():
                self.pipelines[split_name] = default_pipeline

    def _build_pipeline_from_list(self, module_names: List[str]) -> AugmentationPipeline:
        """Build pipeline from list of module name strings."""
        modules = []
        for name in module_names:
            if name not in self.AVAILABLE_MODULES:
                raise ValueError(f"Unknown module '{name}'. Available: {list(self.AVAILABLE_MODULES.keys())}")

            module_class = self.AVAILABLE_MODULES[name]

            # Pass relevant kwargs to module
            if module_class == CoilAverageSampler:
                modules.append(module_class(n_coils=self.n_coils, n_averages=self.n_averages))
            elif module_class == NIfTI_RawProcessor:
                modules.append(module_class(**self.kwargs))
            else:
                # Try to pass kwargs, filter by module's __init__ signature
                import inspect
                sig = inspect.signature(module_class.__init__)
                valid_kwargs = {k: v for k, v in self.kwargs.items() if k in sig.parameters}
                modules.append(module_class(**valid_kwargs))

        return AugmentationPipeline(modules)

    def _create_modes(self, mode, modes):
        """Create sampling modes for each split."""
        self.modes = {}

        if modes is not None:
            # Per-split modes
            for split_name in self.splits.keys():
                self.modes[split_name] = modes.get(split_name, 'random')
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
            shuffle: Whether to shuffle (only for deterministic mode)

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
        if mode == 'random':
            generator = create_random_generator(
                data=data,
                water=water,
                pipeline=pipeline,
                batch_size=self.batch_size
            )
        elif mode == 'deterministic':
            shuffle_val = shuffle if shuffle is not None else (split == 'train')
            generator = create_deterministic_generator(
                data=data,
                water=water,
                pipeline=pipeline,
                batch_size=self.batch_size,
                n_coils=self.n_coils,
                n_averages=self.n_averages,
                shuffle=shuffle_val
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Wrap for framework
        return wrap_generator_for_framework(generator, self.backend, framework)

    def dataloader(self, framework: str = None, **kwargs):
        """
        Get dataloader (for single dataset without splitting).

        Args:
            framework: 'pytorch', 'numpy', 'tensorflow', 'keras', 'jax'
            **kwargs: Additional arguments (e.g., shuffle=True)

        Returns:
            Framework-specific dataloader/generator
        """
        return self._get_dataloader('train', framework, **kwargs)

    def train_dataloader(self, framework: str = None, **kwargs):
        """Get training dataloader."""
        return self._get_dataloader('train', framework, kwargs.get('shuffle', True))

    def val_dataloader(self, framework: str = None, **kwargs):
        """Get validation dataloader."""
        return self._get_dataloader('val', framework, kwargs.get('shuffle', False))

    def test_dataloader(self, framework: str = None, **kwargs):
        """Get test dataloader."""
        return self._get_dataloader('test', framework, kwargs.get('shuffle', False))

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
        lines.append(f"║  📊 Split: {split:20} Subjects: {len(data):4}  Mode: {self.modes[split]:15}║")
        lines.append(f"║  🎯 Backend: {self.backend.name:15} Batch Size: {self.batch_size:4}  Volatile: {self.volatile!s:5}  ║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║  🔄 Pipeline Steps:" + " " * 57 + "║")
        lines.append("║" + " " * 78 + "║")

        # Steps
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
        }
        return emoji_map.get(name, '🔧')

    def _get_module_params(self, module) -> str:
        """Get key parameters of a module."""
        params = []
        if hasattr(module, 'n_coils') and module.n_coils:
            params.append(f"coils={module.n_coils}")
        if hasattr(module, 'n_averages') and module.n_averages:
            params.append(f"avg={module.n_averages}")
        if hasattr(module, 'mode') and module.mode:
            params.append(f"mode={module.mode}")
        if hasattr(module, 'sigma_frac') and module.sigma_frac:
            params.append(f"σ_frac={module.sigma_frac}")

        return ", ".join(params) if params else ""

    def __repr__(self):
        splits_info = ", ".join([f"{k}={len(v[0])}" for k, v in self.splits.items()])
        return (f"Augmentrum(splits=[{splits_info}], "
                f"backend={self.backend.name}, batch_size={self.batch_size})")


__all__ = ['Augmentrum']
