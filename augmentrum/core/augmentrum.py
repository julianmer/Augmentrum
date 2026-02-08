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

from typing import List, Optional, Union, Dict, Any
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
    - On-the-fly (random) or fixed (exact) augmentation parameters
    - Multi-backend support (PyTorch, NumPy, TensorFlow, JAX)

    Example 1: On-the-fly augmentation with RANGE sampling
        ```python
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
        ```

    Example 2: Fixed augmentation (exact values)
        ```python
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
        ```

    Example 3: Gaussian distribution sampling
        ```python
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
        ```

    Example 3: With train/val/test splitting
        ```python
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

        # Augmentation mode
        mode: str = 'on-the-fly',  # 'on-the-fly' or 'fixed'
        modes: Optional[Dict[str, str]] = None,  # Per-split modes

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
            mode: Single mode for all splits:
                  'on-the-fly' = Random sampling from ranges (e.g., n_coils=(1,8) picks randomly)
                  'fixed' = Use exact values provided (e.g., n_coils=4 always uses 4 coils)
            modes: Dict mapping split names to modes (overrides 'mode')
            batch_size: Batch size
            backend: 'pytorch', 'numpy', 'tensorflow', 'keras', 'jax', or Backend enum
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
                  - ... (see _get_module_kwargs for full list)
        """
        # Convert backend
        if isinstance(backend, str):
            self.backend = Backend[backend.upper()]
        else:
            self.backend = backend

        self.batch_size = batch_size
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
            # No splitting - all data goes to train, but create empty val/test
            empty_data = NIfTI_MRS_Plus(nifti_list=[], backend=self.backend, volatile=self.volatile)
            self.splits = {
                'train': (self.data_all, self.water_all),
                'val': (empty_data, None),
                'test': (empty_data, None)
            }

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
        modules = []
        for name in module_names:
            if name not in self.AVAILABLE_MODULES:
                raise ValueError(f"Unknown module '{name}'. Available: {list(self.AVAILABLE_MODULES.keys())}")

            module_class = self.AVAILABLE_MODULES[name]

            # Get smart module-specific kwargs
            module_kwargs = self._get_module_kwargs(name, self.kwargs)

            # Create module instance
            modules.append(module_class(**module_kwargs))

        return AugmentationPipeline(modules)

    def _sample_from_range(self, param, distribution: str = 'uniform'):
        """
        Sample a scalar value from a parameter (range or scalar).

        Args:
            param: Either scalar (float/int) or tuple (min, max) for range
            distribution: 'uniform', 'gaussian', 'normal', 'exponential', 'beta'
                         (only used if param is a tuple)

        Returns:
            Scalar value

        Examples:
            _sample_from_range(0.03) → 0.03
            _sample_from_range((0.01, 0.05), 'uniform') → random 0.01-0.05 (uniform)
            _sample_from_range((0.01, 0.05), 'gaussian') → random ~mean=0.03, std based on range
        """
        import numpy as np

        # If already scalar, return as-is
        if isinstance(param, (int, float)):
            return float(param)

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
                max_val = min_val * 2.0  # Arbitrary default

            # Sample based on distribution
            if distribution == 'uniform':
                return np.random.uniform(min_val, max_val)

            elif distribution in ['gaussian', 'normal']:
                # Gaussian centered at midpoint, std = range/6 (99.7% within range)
                mean = (min_val + max_val) / 2.0
                std = (max_val - min_val) / 6.0
                value = np.random.normal(mean, std)
                # Clip to range
                return np.clip(value, min_val, max_val)

            elif distribution == 'exponential':
                # Exponential biased toward min_val
                scale = (max_val - min_val) / 3.0
                value = min_val + np.random.exponential(scale)
                return np.clip(value, min_val, max_val)

            elif distribution == 'beta':
                # Beta distribution (slightly biased to center)
                alpha, beta = 2.0, 2.0
                value = np.random.beta(alpha, beta)
                return min_val + value * (max_val - min_val)

            else:
                # Default to uniform
                return np.random.uniform(min_val, max_val)

        # If single-element tuple
        if isinstance(param, tuple) and len(param) == 1:
            return float(param[0])

        # Otherwise return as-is
        return param

    def _get_module_kwargs(self, name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get module-specific kwargs with smart defaults.
        Changes range parameters (tuples) to scalars by sampling.

        Args:
            name: Module name
            kwargs: User-provided kwargs

        Returns:
            Dictionary of module-specific parameters (all scalars)
        """
        module_kwargs = {}

        # Get sampling distribution
        global_distribution = kwargs.get('param_distribution', 'uniform')
        per_param_distributions = kwargs.get('param_distributions', {})

        def sample(param_name, param_value):
            """Helper to sample with correct distribution for this parameter."""
            dist = per_param_distributions.get(param_name, global_distribution)
            return self._sample_from_range(param_value, dist)

        # Processing (NIfTI_RawProcessor)
        if name == 'processing':
            # Extract all processing-related parameters
            module_kwargs['conj'] = kwargs.get('conj', True)
            module_kwargs['coil'] = kwargs.get('coil', True)
            module_kwargs['align'] = kwargs.get('align', True)
            module_kwargs['remove_outliers'] = kwargs.get('remove_outliers', True)
            module_kwargs['average'] = kwargs.get('average', True)
            module_kwargs['ecc'] = kwargs.get('ecc', True)
            module_kwargs['truncate'] = kwargs.get('truncate', False)
            module_kwargs['remove_water'] = kwargs.get('remove_water', False)
            module_kwargs['shift_ref'] = kwargs.get('shift_ref', True)
            module_kwargs['phase_correct'] = kwargs.get('phase_correct', True)
            module_kwargs['coil_method'] = kwargs.get('coil_method', 'fsl-mrs')
            module_kwargs['registration_method'] = kwargs.get('registration_method', 'fsl-mrs')
            module_kwargs['remove_method'] = kwargs.get('remove_method', 'fsl-mrs')
            module_kwargs['average_method'] = kwargs.get('average_method', 'fsl-mrs')
            module_kwargs['ecc_method'] = kwargs.get('ecc_method', 'own')
            module_kwargs['water_removal_method'] = kwargs.get('water_removal_method', 'fsl-mrs')
            module_kwargs['shift_ref_method'] = kwargs.get('shift_ref_method', 'fsl-mrs')
            module_kwargs['phase_correct_method'] = kwargs.get('phase_correct_method', 'fsl-mrs')

        # Coil/Average Sampling
        elif name in ['coil_sampling', 'average_sampling']:
            module_kwargs['mode'] = kwargs.get('mode', 'random')
            module_kwargs['n_coils'] = kwargs.get('n_coils', (1, None))
            module_kwargs['n_averages'] = kwargs.get('n_averages', (1, None))

        # Gaussian Noise
        elif name in ['noise', 'gaussian_noise']:
            if 'snr_db' in kwargs:
                module_kwargs['snr_db'] = sample('snr_db', kwargs['snr_db'])
            elif 'sigma' in kwargs:
                module_kwargs['sigma'] = sample('sigma', kwargs['sigma'])
            else:
                sigma_frac = kwargs.get('sigma_frac', 0.02)
                module_kwargs['sigma_frac'] = sample('sigma_frac', sigma_frac)


        # Line Broadening
        elif name in ['line_broadening', 'broadening']:
            lb_hz = kwargs.get('lb_hz', 3.0)
            gb_hz = kwargs.get('gb_hz', 2.0)
            module_kwargs['lb_hz'] = sample('lb_hz', lb_hz)
            module_kwargs['gb_hz'] = sample('gb_hz', gb_hz)
            module_kwargs['mode'] = kwargs.get('broadening_mode', 'voigt')

        # Baseline Augmentation
        elif name == 'baseline':
            module_kwargs['mode'] = kwargs.get('baseline_mode', 'random_walk')
            baseline_frac = kwargs.get('baseline_frac', 0.05)
            module_kwargs['baseline_frac'] = sample('baseline_frac', baseline_frac)
            module_kwargs['smooth_pts'] = kwargs.get('smooth_pts', 151)
        elif name == 'baseline_random_walk':
            module_kwargs['mode'] = 'random_walk'
            baseline_frac = kwargs.get('baseline_frac', 0.05)
            module_kwargs['baseline_frac'] = sample('baseline_frac', baseline_frac)
        elif name == 'baseline_bspline':
            module_kwargs['mode'] = 'bspline'
            module_kwargs['knots_per_ppm'] = kwargs.get('knots_per_ppm', 12)
        elif name == 'baseline_polynomial':
            module_kwargs['mode'] = 'polynomial'
            module_kwargs['order'] = kwargs.get('poly_order', 3)

        # Phase Shift
        elif name in ['phase', 'phase_shift']:
            zero_order = kwargs.get('zero_order_deg', 0.0)
            first_order = kwargs.get('first_order_deg', 0.0)
            module_kwargs['zero_order_deg'] = sample('zero_order_deg', zero_order)
            module_kwargs['first_order_deg'] = sample('first_order_deg', first_order)

        # Frequency Shift
        elif name == 'frequency_shift':
            shift_hz = kwargs.get('shift_hz', 0.0)
            module_kwargs['shift_hz'] = sample('shift_hz', shift_hz)

        # Residual Water
        elif name in ['residual_water', 'water']:
            center_ppm = kwargs.get('water_ppm', 4.7)
            phase_deg = kwargs.get('water_phase', 0.0)
            amplitude = kwargs.get('water_amp', 0.1)
            module_kwargs['center_ppm'] = sample('water_ppm', center_ppm)
            module_kwargs['phase_deg'] = sample('water_phase', phase_deg)
            module_kwargs['amplitude_scale'] = sample('water_amp', amplitude)

        # Spurious Echoes
        elif name in ['spurious_echoes', 'echoes']:
            if 'echoes' in kwargs:
                # Allow each element to be a range or scalar
                # Format: [(amp, width, phase, freq, time), ...]
                # Each element can be: scalar or (min, max) tuple
                echoes_spec = kwargs['echoes']
                normalized_echoes = []

                for echo_params in echoes_spec:
                    # Each echo_params is a tuple of (amp, width, phase, freq, time)
                    # Each element can be scalar or (min, max)
                    normalized_params = []
                    param_names = ['echo_amp', 'echo_width', 'echo_phase', 'echo_freq', 'echo_time']

                    for i, param in enumerate(echo_params):
                        param_name = f"{param_names[i]}_{len(normalized_echoes)}"
                        normalized_params.append(sample(param_name, param))

                    normalized_echoes.append(tuple(normalized_params))

                module_kwargs['echoes'] = normalized_echoes
            else:
                # Default: single echo with ranges
                module_kwargs['echoes'] = [(0.1, 0.2, 0.0, 5.0, 0.0)]

        # Artificial Peaks
        elif name in ['artificial_peaks', 'peaks']:
            if 'peaks' in kwargs:
                # Allow each element to be a range or scalar
                # Format: [(amp, freq, width, phase, lineshape), ...]
                # Each element can be: scalar or (min, max) tuple
                peaks_spec = kwargs['peaks']
                normalized_peaks = []

                for peak_params in peaks_spec:
                    # Each peak_params is a tuple of (amp, freq, width, phase, lineshape)
                    # lineshape is typically a string, others can be ranges
                    normalized_params = []
                    param_names = ['peak_amp', 'peak_freq', 'peak_width', 'peak_phase', 'peak_lineshape']

                    for i, param in enumerate(peak_params):
                        param_name = f"{param_names[i]}_{len(normalized_peaks)}"
                        # Don't sample if it's a string (lineshape)
                        if isinstance(param, str):
                            normalized_params.append(param)
                        else:
                            normalized_params.append(sample(param_name, param))

                    normalized_peaks.append(tuple(normalized_params))

                module_kwargs['peaks'] = normalized_peaks

        # Eddy Current
        elif name in ['eddy_current', 'eddy']:
            module_kwargs['mode'] = kwargs.get('eddy_mode', 'synthetic')
            std_rad = kwargs.get('eddy_std', 0.6)
            strength = kwargs.get('eddy_strength', 1.0)
            module_kwargs['std_rad'] = sample('eddy_std', std_rad)
            module_kwargs['strength'] = sample('eddy_strength', strength)

        # Apodization
        elif name in ['apodization', 'apod']:
            module_kwargs['mode'] = kwargs.get('apod_mode', 'exponential')
            if module_kwargs['mode'] == 'exponential':
                lb_hz = kwargs.get('apod_lb', 3.0)
                module_kwargs['lb_hz'] = sample('apod_lb', lb_hz)
            else:
                module_kwargs['n_pts'] = kwargs.get('apod_npts', 1024)

        return module_kwargs

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
            'eddy_std', 'apod_lb'
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


__all__ = ['Augmentrum']
