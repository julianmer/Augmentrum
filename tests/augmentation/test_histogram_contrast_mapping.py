####################################################################################################
#                            test_histogram_contrast_mapping.py                                    #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (jlamaste@gmail.com)                                                     #
#                                                                                                  #
# Created: 2026-09-12                                                                              #
#                                                                                                  #
# Purpose: HistogramContrastMapping's empirical quantile-function contrast remap - identity when   #
#          unconfigured, exact quantile-mapping arithmetic against a hand-computed reference,       #
#          masking/clipping robustness, template compatibility filtering, reproducible/differing    #
#          random template selection, Dirichlet interpolation, and complex-phase preservation.      #
#                                                                                                  #
####################################################################################################

"""
Tests for HistogramContrastMapping (empirical quantile-function contrast mapping).
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.augmentation.histogram_contrast_mapping import (
    ContrastTemplate, HistogramContrastMapping, _compatible, _empirical_quantile_table,
)

B, X, Y, Z, T = 2, 16, 16, 16, 1


def _blob_volume(batch=B, seed=0, lo=50.0, hi=150.0):
    """A foreground blob on a zero background - background must not dominate F_s."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((batch, X, Y, Z, T), dtype=np.float32)
    vol[:, 4:12, 4:12, 4:12, 0] = rng.uniform(lo, hi, size=(batch, 8, 8, 8))
    return vol


def _linear_template(scale=1.0, n=1001, **metadata):
    """Q_t(p) = scale * p - an easy-to-verify target quantile function."""
    p = np.linspace(0.0, 1.0, n)
    return ContrastTemplate(quantile_p=p, quantile_values=scale * p, **metadata)


#**********************#
#   identity/ablation  #
#**********************#
class TestIdentity:
    def test_no_template_no_bank_is_identity(self):
        vol = _blob_volume()
        mod = HistogramContrastMapping()
        out, water = mod.process_tensor(vol, water_array='untouched')
        assert np.array_equal(out, vol)
        assert water == 'untouched'
        assert mod.DOMAIN is None

    def test_ablated_alongside_a_configured_instance(self):
        """Disabling this module never depends on what else is in the pipeline."""
        vol = _blob_volume()
        off = HistogramContrastMapping()
        on = HistogramContrastMapping(template=_linear_template(2.0), seed=0)
        out_off, _ = off.process_tensor(vol)
        out_on, _ = on.process_tensor(vol)
        assert np.array_equal(out_off, vol)
        assert not np.array_equal(out_on, vol)


#***************************#
#   quantile-mapping math   #
#***************************#
class TestQuantileMapping:
    def test_matches_hand_computed_reference(self):
        """A uniform source mapped through Q_t(p) = p reproduces its own rank."""
        vol = np.zeros((1, X, Y, Z, T), dtype=np.float64)
        values = np.linspace(0.0, 1.0, X * Y * Z).reshape(X, Y, Z)
        vol[0, ..., 0] = values

        template = _linear_template(1.0)
        mod = HistogramContrastMapping(template=template, auto_mask_fraction=None,
                                       clip_percentiles=(0.0, 100.0), epsilon=1e-6, seed=0)
        out, _ = mod.process_tensor(vol)

        ranks = (np.argsort(np.argsort(values.reshape(-1))) + 0.5) / values.size
        expected = ranks.reshape(X, Y, Z)
        np.testing.assert_allclose(out[0, ..., 0], expected, atol=2e-3)

    def test_rank_preserving_and_monotonic(self):
        vol = _blob_volume(batch=1, seed=1)
        template = _linear_template(5.0)
        mod = HistogramContrastMapping(template=template, seed=0)
        out, _ = mod.process_tensor(vol)

        src_fg = vol[0, 4:12, 4:12, 4:12, 0].reshape(-1)
        out_fg = out[0, 4:12, 4:12, 4:12, 0].reshape(-1)
        order_src = np.argsort(src_fg)
        assert np.all(np.diff(out_fg[order_src]) >= -1e-9)

    def test_epsilon_clips_extreme_values(self):
        """A single extreme outlier lands at Q_t(1 - epsilon), not Q_t(1)."""
        vol = _blob_volume(batch=1, seed=2)
        vol[0, 5, 5, 5, 0] = 1e6   # a single wild outlier inside the foreground

        template = _linear_template(1.0)
        epsilon = 1e-2
        mod = HistogramContrastMapping(template=template, clip_percentiles=(0.0, 100.0),
                                       epsilon=epsilon, seed=0)
        out, _ = mod.process_tensor(vol)
        assert out[0, 5, 5, 5, 0] <= 1.0 - epsilon + 1e-9

    def test_percentile_clipping_bounds_influence_of_outliers(self):
        """With aggressive clipping, one outlier voxel barely perturbs the mapping of the rest."""
        vol = _blob_volume(batch=1, seed=3)
        clean = vol.copy()
        outlier = vol.copy()
        outlier[0, 5, 5, 5, 0] = 1e6

        template = _linear_template(1.0)
        common = dict(template=template, clip_percentiles=(1.0, 99.0), seed=0)
        out_clean, _ = HistogramContrastMapping(**common).process_tensor(clean)
        out_outlier, _ = HistogramContrastMapping(**common).process_tensor(outlier)

        # Everywhere except the outlier voxel itself, the mapping barely moves.
        diff = np.abs(out_clean - out_outlier)
        diff[0, 5, 5, 5, 0] = 0.0
        assert diff.max() < 0.05


#***************#
#   masking     #
#***************#
class TestMasking:
    def test_background_left_at_input_value_when_fill_background_none(self):
        vol = _blob_volume(batch=1, seed=4)
        mod = HistogramContrastMapping(template=_linear_template(3.0), fill_background=None, seed=0)
        out, _ = mod.process_tensor(vol)
        assert np.array_equal(out[0, 0, 0, 0, 0], vol[0, 0, 0, 0, 0])

    def test_background_set_to_fill_value(self):
        vol = _blob_volume(batch=1, seed=5)
        mod = HistogramContrastMapping(template=_linear_template(3.0), fill_background=-1.0, seed=0)
        out, _ = mod.process_tensor(vol)
        assert out[0, 0, 0, 0, 0] == -1.0
        assert out[0, 6, 6, 6, 0] != -1.0

    def test_explicit_mask_overrides_auto_threshold(self):
        vol = np.ones((1, X, Y, Z, T), dtype=np.float32) * 100.0
        mask = np.zeros((X, Y, Z, T), dtype=bool)
        mask[6:10, 6:10, 6:10, :] = True

        mod = HistogramContrastMapping(template=_linear_template(1.0), mask=mask,
                                       fill_background=0.0, seed=0)
        out, _ = mod.process_tensor(vol)
        assert np.all(out[0, 6:10, 6:10, 6:10, 0] != 0.0)
        assert np.all(out[0, 0:6, 0:6, 0:6, 0] == 0.0)

    def test_empty_mask_raises(self):
        vol = _blob_volume(batch=1, seed=6)
        mask = np.zeros((X, Y, Z, T), dtype=bool)
        mod = HistogramContrastMapping(template=_linear_template(1.0), mask=mask, seed=0)
        with pytest.raises(ValueError, match="no voxels"):
            mod.process_tensor(vol)


#***********************#
#   bias-field hook     #
#***********************#
class TestBiasFieldCorrection:
    def test_hook_is_applied_before_mapping(self):
        vol = _blob_volume(batch=1, seed=7)
        template = _linear_template(1.0)

        plain = HistogramContrastMapping(template=template, seed=0)
        out_plain, _ = plain.process_tensor(vol)

        # A spatial gradient (not just a monotonic rescale) changes voxels'
        # *relative* ranking, so it must actually shift the quantile mapping -
        # a uniform positive scale would not, since ranks are scale-invariant.
        gradient = np.linspace(0.0, 80.0, X, dtype=np.float32)[:, None, None, None]
        corrected = HistogramContrastMapping(
            template=template, bias_field_correction=lambda block: block + gradient, seed=0
        )
        out_corrected, _ = corrected.process_tensor(vol)

        assert not np.allclose(out_plain, out_corrected)

    def test_hook_output_shape_is_validated(self):
        vol = _blob_volume(batch=1, seed=8)
        mod = HistogramContrastMapping(
            template=_linear_template(1.0),
            bias_field_correction=lambda block: block[:-1], seed=0,
        )
        with pytest.raises(ValueError, match="same shape"):
            mod.process_tensor(vol)


#*******************************#
#   template compatibility      #
#*******************************#
class TestCompatibility:
    def test_categorical_and_numeric_fields(self):
        t = _linear_template(1.0, field_strength_T=0.075, sequence_family='T1w', TR_ms=12.0)
        assert _compatible(t, None, None)
        assert _compatible(t, {'sequence_family': 'T1w'}, None)
        assert not _compatible(t, {'sequence_family': 'T2w'}, None)
        assert _compatible(t, {'field_strength_T': 0.08}, None)      # within default 20% tol
        assert not _compatible(t, {'field_strength_T': 1.5}, None)

    def test_tolerance_override(self):
        t = _linear_template(1.0, field_strength_T=0.075)
        assert not _compatible(t, {'field_strength_T': 0.09}, {'field_strength_T': 0.05})
        assert _compatible(t, {'field_strength_T': 0.09}, {'field_strength_T': 0.5})

    def test_incompatible_bank_raises_on_process(self):
        vol = _blob_volume(batch=1, seed=9)
        bank = [_linear_template(1.0, field_strength_T=1.5, template_id='hf')]
        mod = HistogramContrastMapping(templates=bank, reference={'field_strength_T': 0.075}, seed=0)
        with pytest.raises(ValueError, match="No template"):
            mod.process_tensor(vol)


#***************************#
#   selection/interpolate   #
#***************************#
class TestTemplateSelection:
    def _bank(self):
        return [
            _linear_template(1.0, field_strength_T=0.075, sequence_family='T1w', template_id='a'),
            _linear_template(2.0, field_strength_T=0.075, sequence_family='T1w', template_id='b'),
            _linear_template(3.0, field_strength_T=1.5, sequence_family='T2w', template_id='c'),
        ]

    def test_select_mode_reproducible_with_fixed_seed(self):
        vol = _blob_volume()
        kwargs = dict(templates=self._bank(), mode='select',
                      reference={'field_strength_T': 0.075}, seed=123)
        out1, _ = HistogramContrastMapping(**kwargs).process_tensor(vol)
        out2, _ = HistogramContrastMapping(**kwargs).process_tensor(vol)
        assert np.array_equal(out1, out2)

    def test_select_mode_never_picks_incompatible_template(self):
        vol = _blob_volume()
        mod = HistogramContrastMapping(templates=self._bank(), mode='select',
                                       reference={'field_strength_T': 0.075}, seed=0)
        mod.process_tensor(vol)
        assert all(tid in ('a', 'b') for ids in mod.last_template_ids_ for tid in ids)

    def test_different_seeds_can_select_differently(self):
        vol = _blob_volume()
        bank = self._bank()
        seeds_seen = set()
        for seed in range(10):
            mod = HistogramContrastMapping(templates=bank, mode='select',
                                           reference={'field_strength_T': 0.075}, seed=seed)
            mod.process_tensor(vol)
            seeds_seen.add(tuple(tuple(ids) for ids in mod.last_template_ids_))
        assert len(seeds_seen) > 1

    def test_interpolate_weights_sum_to_one(self):
        vol = _blob_volume()
        mod = HistogramContrastMapping(templates=self._bank(), mode='interpolate', n_interpolate=2,
                                       reference={'field_strength_T': 0.075}, seed=0)
        mod.process_tensor(vol)
        for w in mod.last_weights_:
            assert abs(sum(w) - 1.0) < 1e-9

    def test_interpolate_is_quantile_mixture_not_cdf_average(self):
        """Q_mix(p) = sum_i w_i Q_i(p), evaluated directly against the two linear templates."""
        p = np.array([0.3])
        t_a = _linear_template(1.0, field_strength_T=0.075, sequence_family='T1w', template_id='a')
        t_b = _linear_template(2.0, field_strength_T=0.075, sequence_family='T1w', template_id='b')

        mod = HistogramContrastMapping(templates=[t_a, t_b], mode='interpolate', n_interpolate=2,
                                       reference={'field_strength_T': 0.075}, seed=5)
        rng = mod.rng.numpy_rng()
        chosen, weights = mod._select_templates(rng)

        expected = sum(w * t.evaluate(p) for t, w in zip(chosen, weights))
        mixture = np.zeros_like(p)
        for t, w in zip(chosen, weights):
            mixture += w * t.evaluate(p)
        np.testing.assert_allclose(mixture, expected)

    def test_single_template_overrides_bank(self):
        vol = _blob_volume(batch=1)
        fixed = _linear_template(4.0, template_id='fixed')
        mod = HistogramContrastMapping(template=fixed, templates=None, seed=0)
        mod.process_tensor(vol)
        assert mod.last_template_ids_ == [['fixed']]
        assert mod.last_weights_ == [[1.0]]

    def test_template_and_templates_both_given_raises(self):
        with pytest.raises(ValueError, match="not both"):
            HistogramContrastMapping(template=_linear_template(1.0), templates=[_linear_template(2.0)])


#***********************#
#   complex data         #
#***********************#
class TestComplexInput:
    def test_phase_is_preserved_magnitude_is_mapped(self):
        vol = _blob_volume(batch=1, seed=10)
        phase = 0.42
        cvol = (vol * np.exp(1j * phase)).astype(np.complex64)

        mod = HistogramContrastMapping(template=_linear_template(2.0), seed=0)
        out, _ = mod.process_tensor(cvol)

        assert np.iscomplexobj(out)
        np.testing.assert_allclose(np.angle(out[0, 5, 5, 5, 0]), phase, atol=1e-4)
        assert not np.isclose(np.abs(out[0, 5, 5, 5, 0]), np.abs(cvol[0, 5, 5, 5, 0]))

    def test_complex_background_fill_has_zero_imaginary_part(self):
        vol = _blob_volume(batch=1, seed=11)
        cvol = (vol * np.exp(1j * 0.7)).astype(np.complex64)
        mod = HistogramContrastMapping(template=_linear_template(1.0), fill_background=0.0, seed=0)
        out, _ = mod.process_tensor(cvol)
        assert out[0, 0, 0, 0, 0] == 0.0


#***************************#
#   ContrastTemplate         #
#***************************#
class TestContrastTemplate:
    def test_requires_strictly_increasing_quantile_p(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            ContrastTemplate(quantile_p=np.array([0.0, 0.5, 0.5]), quantile_values=np.array([0., 1., 2.]))

    def test_requires_matching_shapes(self):
        with pytest.raises(ValueError, match="1-D and the same length"):
            ContrastTemplate(quantile_p=np.array([0., 1.]), quantile_values=np.array([0., 1., 2.]))

    def test_from_image_matches_direct_construction(self):
        rng = np.random.default_rng(0)
        image = rng.uniform(0, 1, size=(1000,))
        knots, probs = _empirical_quantile_table(image, 101)
        template = ContrastTemplate.from_image(image, n_quantiles=101, template_id='t')
        np.testing.assert_allclose(template.quantile_values, knots)
        np.testing.assert_allclose(template.quantile_p, probs)

    def test_from_image_respects_mask(self):
        image = np.concatenate([np.zeros(500), np.full(500, 10.0)])
        mask = image > 0
        template = ContrastTemplate.from_image(image, mask=mask, n_quantiles=11)
        np.testing.assert_allclose(template.quantile_values, 10.0, atol=1e-4)


#***************************#
#   reproducibility          #
#***************************#
class TestReproducibility:
    def test_last_mapping_params_reflects_construction(self):
        vol = _blob_volume(batch=1)
        mod = HistogramContrastMapping(template=_linear_template(1.0), clip_percentiles=(2.0, 98.0),
                                       epsilon=1e-2, n_quantiles=257, seed=0)
        mod.process_tensor(vol)
        assert mod.last_mapping_params_['clip_percentiles'] == (2.0, 98.0)
        assert mod.last_mapping_params_['epsilon'] == 1e-2
        assert mod.last_mapping_params_['n_quantiles'] == 257

    def test_debug_outputs_gate_heavy_diagnostics(self):
        vol = _blob_volume(batch=1)
        quiet = HistogramContrastMapping(template=_linear_template(1.0), debug_outputs=False, seed=0)
        quiet.process_tensor(vol)
        assert quiet.last_debug_ is None

        verbose = HistogramContrastMapping(template=_linear_template(1.0), debug_outputs=True, seed=0)
        verbose.process_tensor(vol)
        assert verbose.last_debug_ is not None
        assert 'quantile_table' in verbose.last_debug_[0]
