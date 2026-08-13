"""
Tests for parameter sampling (tuple ranges and distributions).

Tests cover:
- Tuple range support for all augmentation parameters
- Global distribution sampling
- Per-parameter distribution control
- Nested ranges for complex augmentations (echoes, peaks)
- Distribution behavior verification
"""

import pytest
import numpy as np
from augmentrum import Augmentrum
from augmentrum.core import Backend
from augmentrum.core.pipeline import AugmentationPipeline
from augmentrum.augmentation import (AmplitudeScaling, FrequencyShift,
                                     LineBroadening, Noise, PhaseShift)


#**************************************************************************************************#
#                                 Class TestParameterRangeSupport                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test that all parameters support tuple ranges.                                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestParameterRangeSupport:
    """Test that all parameters support tuple ranges."""

    def test_noise_parameters_ranges(self, dummy_nifti_list):
        """Test noise parameters with ranges."""
        # sigma_frac
        aug1 = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            sigma_frac=(0.01, 0.05),
            batch_size=1
        )
        assert aug1 is not None

        # snr_db
        aug2 = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            snr_db=(10, 30),
            batch_size=1
        )
        assert aug2 is not None

        # sigma
        aug3 = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            sigma=(0.001, 0.01),
            batch_size=1
        )
        assert aug3 is not None

    def test_broadening_parameters_ranges(self, dummy_nifti_list):
        """Test line broadening parameters with ranges."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['line_broadening'],
            lb_hz=(0, 10),
            gb_hz=(0, 5),
            batch_size=1
        )

        assert augmenter is not None

    def test_phase_parameters_ranges(self, dummy_nifti_list):
        """Test phase parameters with ranges."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['phase'],
            zero_order_deg=(-180, 180),
            first_order_deg=(-90, 90),
            batch_size=1
        )

        assert augmenter is not None

    def test_frequency_shift_range(self, dummy_nifti_list):
        """Test frequency shift with range."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['frequency_shift'],
            shift_hz=(-5, 5),
            batch_size=1
        )

        assert augmenter is not None

    def test_baseline_parameters_ranges(self, dummy_nifti_list):
        """Test baseline parameters with ranges."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['baseline'],
            baseline_frac=(0.01, 0.1),
            batch_size=1
        )

        assert augmenter is not None

    def test_water_parameters_ranges(self, dummy_nifti_list):
        """Test residual water parameters with ranges."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['water'],
            water_ppm=(4.65, 4.75),
            water_phase=(-45, 45),
            water_amp=(0.05, 0.2),
            batch_size=1
        )

        assert augmenter is not None

    def test_eddy_current_parameters_ranges(self, dummy_nifti_list):
        """Test eddy current parameters with ranges."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['eddy'],
            eddy_std=(0.3, 1.0),
            eddy_strength=(0.5, 1.5),
            batch_size=1
        )

        assert augmenter is not None

    def test_apodization_parameters_ranges(self, dummy_nifti_list):
        """Test apodization parameters with ranges."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['apod'],
            apod_lb=(0, 10),
            batch_size=1
        )

        assert augmenter is not None


#**************************************************************************************************#
#                                  Class TestDistributionSampling                                  #
#**************************************************************************************************#
#                                                                                                  #
# Test different distribution types.                                                               #
#                                                                                                  #
#**************************************************************************************************#
class TestDistributionSampling:
    """Test different distribution types."""

    def test_uniform_distribution(self, dummy_nifti_list):
        """Test uniform distribution (default)."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            sigma_frac=(0.01, 0.05),
            param_distribution='uniform',
            batch_size=1
        )

        assert augmenter is not None

    def test_gaussian_distribution(self, dummy_nifti_list):
        """Test Gaussian/normal distribution."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            sigma_frac=(0.01, 0.05),
            param_distribution='gaussian',
            batch_size=1
        )

        assert augmenter is not None

    def test_exponential_distribution(self, dummy_nifti_list):
        """Test exponential distribution."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            sigma_frac=(0.01, 0.05),
            param_distribution='exponential',
            batch_size=1
        )

        assert augmenter is not None

    def test_beta_distribution(self, dummy_nifti_list):
        """Test beta distribution."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            sigma_frac=(0.01, 0.05),
            param_distribution='beta',
            batch_size=1
        )

        assert augmenter is not None

    def test_all_distributions_work(self, dummy_nifti_list):
        """Test that all distribution types are accepted."""
        distributions = ['uniform', 'gaussian', 'normal', 'exponential', 'beta']

        for dist in distributions:
            augmenter = Augmentrum(
                data=dummy_nifti_list,
                pipeline=['noise'],
                sigma_frac=(0.01, 0.05),
                param_distribution=dist,
                batch_size=1
            )
            assert augmenter is not None


#**************************************************************************************************#
#                               Class TestPerParameterDistributions                                #
#**************************************************************************************************#
#                                                                                                  #
# Test per-parameter distribution control.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class TestPerParameterDistributions:
    """Test per-parameter distribution control."""

    def test_per_parameter_distributions_basic(self, dummy_nifti_list):
        """Test basic per-parameter distributions."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening'],
            sigma_frac=(0.01, 0.05),
            lb_hz=(0, 10),
            param_distributions={
                'sigma_frac': 'exponential',
                'lb_hz': 'gaussian',
            },
            batch_size=1
        )

        assert augmenter is not None

    def test_per_parameter_overrides_global(self, dummy_nifti_list):
        """Test that per-parameter distributions override global."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening'],
            sigma_frac=(0.01, 0.05),
            lb_hz=(0, 10),
            param_distribution='uniform',  # Global
            param_distributions={
                'sigma_frac': 'exponential',  # Override for sigma_frac
            },
            batch_size=1
        )

        assert augmenter is not None

    def test_multiple_parameter_distributions(self, dummy_nifti_list):
        """Test multiple parameters with different distributions."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening', 'phase', 'baseline'],
            sigma_frac=(0.01, 0.05),
            lb_hz=(0, 10),
            phase0_deg=(-180, 180),
            baseline_frac=(0.01, 0.1),
            param_distributions={
                'sigma_frac': 'exponential',
                'lb_hz': 'gaussian',
                'phase0_deg': 'uniform',
                'baseline_frac': 'beta',
            },
            batch_size=1
        )

        assert augmenter is not None


#**************************************************************************************************#
#                                      Class TestNestedRanges                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test nested tuple ranges for complex augmentations.                                              #
#                                                                                                  #
#**************************************************************************************************#
class TestNestedRanges:
    """Test nested tuple ranges for complex augmentations."""

    def test_spurious_echoes_nested_ranges(self, dummy_nifti_list):
        """Test spurious echoes with nested ranges."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['echoes'],
            echoes=[
                # (amp, width, phase, freq, time)
                ((0.1, 0.3), (0.2, 0.5), 0.0, (4.0, 6.0), 0.0),
            ],
            batch_size=1
        )

        assert augmenter is not None

    def test_spurious_echoes_mixed_ranges_scalars(self, dummy_nifti_list):
        """Test spurious echoes with mixed ranges and scalars."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['echoes'],
            echoes=[
                # First echo: mix of ranges and scalars
                ((0.1, 0.3), 0.3, 0.0, (4.0, 6.0), 0.0),
                # Second echo: different mix
                (0.2, (0.2, 0.5), (-90, 90), 5.0, 0.05),
            ],
            batch_size=1
        )

        assert augmenter is not None

    def test_artificial_peaks_nested_ranges(self, dummy_nifti_list):
        """Test artificial peaks with nested ranges."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['peaks'],
            peaks=[
                # (amp, freq, width, phase, lineshape)
                ((0.5, 1.0), (3.0, 3.2), 0.05, 0.0, 'lorentzian'),
            ],
            batch_size=1
        )

        assert augmenter is not None

    def test_artificial_peaks_mixed_ranges_scalars(self, dummy_nifti_list):
        """Test artificial peaks with mixed ranges and scalars."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['peaks'],
            peaks=[
                # First peak
                ((0.5, 1.0), (3.0, 3.2), 0.05, 0.0, 'lorentzian'),
                # Second peak with different mix
                (0.8, 2.0, (0.03, 0.07), (-45, 45), 'gaussian'),
            ],
            batch_size=1
        )

        assert augmenter is not None

    def test_nested_ranges_with_distributions(self, dummy_nifti_list):
        """Test nested ranges with per-parameter distributions."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['echoes'],
            echoes=[
                ((0.1, 0.3), (0.2, 0.5), 0.0, (4.0, 6.0), 0.0),
            ],
            param_distributions={
                'echo_amp_0': 'uniform',
                'echo_width_0': 'gaussian',
                'echo_freq_0': 'exponential',
            },
            batch_size=1
        )

        assert augmenter is not None


#**************************************************************************************************#
#                              Class TestParameterSamplingIntegration                              #
#**************************************************************************************************#
#                                                                                                  #
# Integration tests for parameter sampling.                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestParameterSamplingIntegration:
    """Integration tests for parameter sampling."""

    def test_complex_pipeline_with_all_ranges(self, dummy_nifti_list):
        """Test complex pipeline with all parameter types."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening', 'phase', 'baseline', 'echoes'],
            # All with ranges
            sigma_frac=(0.01, 0.05),
            lb_hz=(0, 10),
            gb_hz=(0, 5),
            phase0_deg=(-180, 180),
            phase1_deg=(-90, 90),
            baseline_frac=(0.01, 0.1),
            echoes=[
                ((0.1, 0.3), (0.2, 0.5), 0.0, (4.0, 6.0), 0.0),
            ],
            # Per-parameter distributions
            param_distributions={
                'sigma_frac': 'exponential',
                'lb_hz': 'gaussian',
                'phase0_deg': 'uniform',
                'baseline_frac': 'exponential',
                'echo_amp_0': 'uniform',
            },
            batch_size=2,
            backend='numpy'
        )

        assert augmenter is not None
        assert augmenter.pipelines is not None
        assert len(augmenter.pipelines) > 0

    def test_backward_compatibility_scalars_still_work(self, dummy_nifti_list):
        """Test that old scalar-only code still works (backward compatibility)."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening', 'phase'],
            sigma_frac=0.03,        # Scalar (old style)
            lb_hz=5.0,              # Scalar (old style)
            phase0_deg=0.0,         # Scalar (old style)
            batch_size=1,
            backend='numpy'
        )

        assert augmenter is not None

    def test_mixed_old_and_new_style(self, dummy_nifti_list):
        """Test mixing old (scalar) and new (range) parameter styles."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening', 'phase'],
            sigma_frac=(0.01, 0.05),  # New style (range)
            lb_hz=5.0,                # Old style (scalar)
            phase0_deg=(-180, 180),   # New style (range)
            batch_size=1,
            backend='numpy'
        )

        assert augmenter is not None


#**************************************************************************************************#
#                                   Class TestPerSampleSampling                                    #
#**************************************************************************************************#
#                                                                                                  #
# A batch carries a spread of a ranged parameter, not one point of it.                             #
#                                                                                                  #
#**************************************************************************************************#
class TestPerSampleSampling:
    """
    Ranged parameters a module declares in PER_SAMPLE_PARAMS are drawn once
    per sample; everything else keeps the one-value-per-batch behavior. The
    module tests below drive the same attributes the pipeline injects.
    """

    @staticmethod
    def _fid_batch(n=3, n_pts=256, sw_hz=2000.0):
        """Identical decaying FIDs, so any spread in the output is the parameter's."""
        t = np.arange(n_pts) / sw_hz
        fid = np.exp(-20.0 * t) * np.exp(1j * 2.0 * np.pi * 30.0 * t)
        return np.tile(fid.astype(np.complex64), (n, 1))

    def test_declared_params_are_drawn_per_sample(self):
        """shift_hz is declared and comes back as a vector; first_order_deg is not."""
        pipeline = AugmentationPipeline(
            [FrequencyShift(), PhaseShift()],
            user_kwargs={'shift_hz': (5.0, 15.0), 'first_order_deg': (5.0, 10.0)})

        params = pipeline.sample_batch_parameters(6)

        shifts = params[0]['shift_hz']
        assert isinstance(shifts, np.ndarray) and shifts.shape == (6,)
        assert np.all((shifts >= 5.0) & (shifts <= 15.0))
        assert len(np.unique(shifts)) > 1, "six draws collapsed to one value"

        ramp = params[1]['first_order_deg']
        assert np.ndim(ramp) == 0, "an undeclared parameter must stay scalar"

    def test_frequency_shift_vector_matches_per_row_scalars(self):
        """One vector pass is exactly the per-row scalar passes stacked."""
        batch, shifts = self._fid_batch(), np.array([5.0, 10.0, 20.0])

        module = FrequencyShift()
        module.shift_hz = shifts
        out, _ = module.process_tensor(batch, sw_hz=2000.0)

        for i, shift in enumerate(shifts):
            ref, _ = FrequencyShift(shift_hz=float(shift)).process_tensor(
                batch[i:i + 1], sw_hz=2000.0)
            assert np.allclose(out[i], ref[0], atol=1e-6)

    def test_zero_order_phase_vector_matches_per_row_scalars(self):
        batch, phases = self._fid_batch(), np.array([-90.0, 0.0, 45.0])

        module = PhaseShift()
        module.zero_order_deg = phases
        out, _ = module.process_tensor(batch, sw_hz=2000.0)

        for i, phase in enumerate(phases):
            ref, _ = PhaseShift(zero_order_deg=float(phase)).process_tensor(
                batch[i:i + 1], sw_hz=2000.0)
            assert np.allclose(out[i], ref[0], atol=1e-6)

    def test_line_broadening_vector_matches_per_row_scalars(self):
        batch, widths = self._fid_batch(), np.array([0.0, 5.0, 15.0])

        module = LineBroadening(mode='lorentzian')
        module.lb_hz = widths
        out, _ = module.process_tensor(batch, sw_hz=2000.0)

        for i, width in enumerate(widths):
            ref, _ = LineBroadening(lb_hz=float(width), mode='lorentzian').process_tensor(
                batch[i:i + 1], sw_hz=2000.0)
            assert np.allclose(out[i], ref[0], atol=1e-6)

    def test_amplitude_vector_is_used_verbatim(self):
        batch, scales = self._fid_batch(), np.array([1.0, 2.0, 0.5])

        module = AmplitudeScaling(scale_factor=(0.9, 1.1))
        module.scale_factor = scales
        out, _ = module.process_tensor(batch)

        for i, scale in enumerate(scales):
            assert np.allclose(out[i], batch[i] * scale, atol=1e-6)

    def test_noise_sigma_vector_gives_each_sample_its_own_level(self):
        """Sigma zero leaves its row untouched while its neighbor gets noise."""
        batch = self._fid_batch(2)

        module = Noise(sigma=0.1, seed=0)
        module.sigma = np.array([0.0, 1.0])
        out, _ = module.process_tensor(batch)

        assert np.allclose(np.asarray(out)[0], batch[0])
        assert not np.allclose(np.asarray(out)[1], batch[1])

    def test_a_short_final_batch_trims_the_vectors(self):
        """A single pass ends on the remainder, and the vectors must follow."""
        from augmentrum.core.dataset_utils import _trim_batch_params

        params = {0: {'shift_hz': np.arange(8.0), 'first_order_deg': 3.0}}
        trimmed = _trim_batch_params(params, 3)

        assert trimmed[0]['shift_hz'].shape == (3,)
        assert trimmed[0]['first_order_deg'] == 3.0
