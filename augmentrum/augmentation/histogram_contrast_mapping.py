####################################################################################################
#                                histogram_contrast_mapping.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (jlamaste@gmail.com)                                                     #
#                                                                                                  #
# Created: 2026-09-12                                                                              #
#                                                                                                  #
# Purpose: Empirical quantile-function contrast mapping between two image distributions - e.g. a   #
#          3T structural volume remapped onto the intensity distribution of a real low-field scan, #
#          or any other pairing of field strengths/protocols a template bank has been built for.   #
#          This is one contrast model among what will become several; it deliberately stops at     #
#          intensity remapping and never touches k-space, noise, or acquisition physics - those     #
#          stay the job of the modules this composes with.                                         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from nifti_mrs_plus import Backend
from nifti_mrs_plus import ops


__all__ = ['ContrastTemplate', 'HistogramContrastMapping']


#**************************************************************************************************#
#                                     empirical quantile tables                                    #
#**************************************************************************************************#
def _empirical_quantile_table(values: np.ndarray, n_quantiles: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    A "(knots, probs)" lookup table standing in for a continuous quantile function.

    "knots[i]" is the intensity at cumulative probability "probs[i]"; the same
    table answers both directions by swapping which array is the interpolation
    axis - "np.interp(x, knots, probs)" is the empirical CDF, "np.interp(p, probs,
    knots)" is the quantile function. Capped at "n_quantiles" points regardless of
    how many voxels went in, so a million-voxel volume costs the same lookup as a
    thousand-voxel one.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    probs = np.linspace(0.0, 1.0, int(n_quantiles))
    knots = np.quantile(values, probs)

    # Ties (a saturated or piecewise-constant source) leave "knots" non-strictly
    # increasing, which breaks np.interp's monotonicity assumption - keep the
    # first occurrence of each distinct value and its probability.
    knots, idx = np.unique(knots, return_index=True)
    probs = probs[idx]

    if knots.size < 2:
        # A perfectly flat block (e.g. an all-zero mask slice) has no
        # distribution to speak of; widen by an epsilon so interpolation stays
        # well-defined instead of raising.
        centre = float(knots[0]) if knots.size else 0.0
        knots = np.array([centre - 1e-6, centre + 1e-6])
        probs = np.array([0.0, 1.0])

    return knots, probs


#**************************************************************************************************#
#                                      Class ContrastTemplate                                      #
#**************************************************************************************************#
#                                                                                                  #
# One target intensity distribution, stored as a quantile function plus the acquisition metadata   #
# that makes it comparable (or not) to another template.                                          #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class ContrastTemplate:
    """
    A target quantile function "Q_t(p)" plus the metadata that identifies it.

    Storing "(quantile_p, quantile_values)" rather than a histogram or raw CDF is
    what makes interpolation between templates ("HistogramContrastMapping"'s
    "mode='interpolate'") a weighted sum of quantile functions rather than an
    arithmetic mean of CDFs - the two give different, non-interchangeable
    mixtures, and only the former corresponds to "a distribution partway
    between these scanners".

    Clinical labels ("T1w", "T2w", "FLAIR") alone do not make two templates
    comparable - protocol details do, so they are first-class fields here
    rather than folded into a free-text tag.

    Args:
        quantile_p: Strictly increasing probabilities in "[0, 1]", shape "(K,)".
        quantile_values: Target intensities at each "quantile_p", shape "(K,)".
        field_strength_T: Static field strength, in tesla, of the scanner this
            template characterizes.
        sequence_family: e.g. "'T1w'", "'T2w'", "'FLAIR'" - a starting point,
            not sufficient alone (see "TR_ms" / "TE_ms" / "TI_ms").
        TR_ms, TE_ms, TI_ms, flip_angle_deg, bandwidth_Hz: Protocol parameters
            distinguishing otherwise same-labelled acquisitions.
        resolution_mm: Voxel size, e.g. "(1.0, 1.0, 1.0)".
        scanner_id, coil_id, reconstruction_id: Free-text identifiers for the
            hardware/software that produced this template, kept separate from
            contrast so receive-sensitivity and reconstruction choices can be
            modeled independently where practical.
        template_id: Free-text identifier for this template itself, surfaced in
            "HistogramContrastMapping.last_template_ids_" for reproducibility.

    Note:
        This models empirical LF (or any target-field) *contrast*, not
        Bloch-equation relaxation physics - it is a fit to an observed
        intensity distribution, nothing more.
    """
    quantile_p: np.ndarray
    quantile_values: np.ndarray
    field_strength_T: Optional[float] = None
    sequence_family: Optional[str] = None
    TR_ms: Optional[float] = None
    TE_ms: Optional[float] = None
    TI_ms: Optional[float] = None
    flip_angle_deg: Optional[float] = None
    resolution_mm: Optional[Tuple[float, ...]] = None
    bandwidth_Hz: Optional[float] = None
    scanner_id: Optional[str] = None
    coil_id: Optional[str] = None
    reconstruction_id: Optional[str] = None
    template_id: Optional[str] = None

    def __post_init__(self):
        p = np.asarray(self.quantile_p, dtype=np.float64)
        v = np.asarray(self.quantile_values, dtype=np.float64)
        if p.ndim != 1 or v.ndim != 1 or p.shape != v.shape:
            raise ValueError(
                "quantile_p and quantile_values must be 1-D and the same length, "
                f"got shapes {p.shape} and {v.shape}."
            )
        if p.size < 2:
            raise ValueError("A quantile function needs at least 2 points.")
        if np.any(np.diff(p) <= 0):
            raise ValueError("quantile_p must be strictly increasing.")
        if p[0] < 0.0 or p[-1] > 1.0:
            raise ValueError("quantile_p must lie within [0, 1].")
        object.__setattr__(self, 'quantile_p', p)
        object.__setattr__(self, 'quantile_values', v)
        if self.resolution_mm is not None:
            object.__setattr__(self, 'resolution_mm', tuple(float(r) for r in self.resolution_mm))

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """"Q_t(p)", clamped to this template's own probability range."""
        p = np.clip(np.asarray(p, dtype=np.float64), self.quantile_p[0], self.quantile_p[-1])
        return np.interp(p, self.quantile_p, self.quantile_values)

    @classmethod
    def from_image(cls, image: np.ndarray, mask: Optional[np.ndarray] = None,
                   n_quantiles: int = 1001, **metadata: Any) -> 'ContrastTemplate':
        """
        Build a template directly from a reference image.

        A convenience constructor for turning a real (preferably high-SNR,
        fully sampled) reference scan into a template, per the "template bank"
        design - not required for hand-specified templates, which can pass
        "quantile_p"/"quantile_values" straight to the constructor.

        Args:
            image: The reference image, any shape.
            mask: Optional boolean foreground mask, same shape as "image".
                Voxels outside it do not contribute to the quantile function -
                background zeros/noise should not be allowed to dominate it.
            n_quantiles: Number of "(p, value)" knots to keep.
            **metadata: Forwarded to the constructor (field_strength_T, etc.).
        """
        values = np.asarray(image, dtype=np.float64).reshape(-1)
        if mask is not None:
            values = values[np.asarray(mask, dtype=bool).reshape(-1)]
        knots, probs = _empirical_quantile_table(values, n_quantiles)
        return cls(quantile_p=probs, quantile_values=knots, **metadata)


#**************************************************************************************************#
#                                       template compatibility                                     #
#**************************************************************************************************#
# What "compatible" means is not specified any more precisely than "field
# strength class / sequence / TR-TE-TI class / resolution class" upstream, so
# it is implemented here as configurable relative-tolerance matching rather
# than an invented set of discrete bucket boundaries - exact match for
# categorical fields, "|have - want| <= tol * |want|" for numeric ones.
_CATEGORICAL_COMPAT_FIELDS = ('sequence_family', 'scanner_id', 'coil_id', 'reconstruction_id')
_NUMERIC_COMPAT_FIELDS = ('field_strength_T', 'TR_ms', 'TE_ms', 'TI_ms', 'flip_angle_deg', 'bandwidth_Hz')
_VECTOR_COMPAT_FIELDS = ('resolution_mm',)
_DEFAULT_TOLERANCES: Dict[str, float] = {f: 0.2 for f in _NUMERIC_COMPAT_FIELDS + _VECTOR_COMPAT_FIELDS}


def _compatible(template: ContrastTemplate, reference: Optional[Dict[str, Any]],
                tolerances: Optional[Dict[str, float]]) -> bool:
    """Whether "template" matches every field named in "reference" (unset fields impose no constraint)."""
    if not reference:
        return True
    tol = tolerances or {}

    for name in _CATEGORICAL_COMPAT_FIELDS:
        want = reference.get(name)
        if want is not None and getattr(template, name) != want:
            return False

    for name in _NUMERIC_COMPAT_FIELDS:
        want = reference.get(name)
        if want is None:
            continue
        have = getattr(template, name)
        if have is None:
            return False
        rel_tol = tol.get(name, _DEFAULT_TOLERANCES[name])
        if abs(have - want) > rel_tol * max(abs(want), 1e-12):
            return False

    for name in _VECTOR_COMPAT_FIELDS:
        want = reference.get(name)
        if want is None:
            continue
        have = getattr(template, name)
        want_arr = np.asarray(want, dtype=np.float64)
        if have is None or len(have) != want_arr.size:
            return False
        have_arr = np.asarray(have, dtype=np.float64)
        rel_tol = tol.get(name, _DEFAULT_TOLERANCES[name])
        if np.any(np.abs(have_arr - want_arr) > rel_tol * np.maximum(np.abs(want_arr), 1e-12)):
            return False

    return True


#**************************************************************************************************#
#                                 Class HistogramContrastMapping                                   #
#**************************************************************************************************#
#                                                                                                  #
# Empirical quantile-function contrast mapping: "I_out(r) = Q_t(F_s(I_in(r)))". Image domain only  #
# - no acquisition noise, no k-space, no claim of relaxation physics.                              #
#                                                                                                  #
#**************************************************************************************************#
class HistogramContrastMapping(BaseModule):
    r"""
    Empirical quantile-function contrast mapping between two image distributions.

    "I_out(r) = Q_t(F_s(I_in(r)))"
    ---------------------------------
    "F_s" is the input image's own empirical CDF (estimated from foreground
    voxels only); "Q_t" is a target quantile function supplied by a
    :class:`ContrastTemplate`. Composing the two is a monotonic, rank-preserving
    remap: a voxel that sits at the 90th percentile of the input distribution
    is placed at the 90th percentile of the target one. Both directions of a
    field-strength conversion are the same operation - only which template is
    "target" differs - so this is not specific to any one field strength; it
    is the piece of a larger LF-synthesis design that does the empirical
    contrast step only. It is one contrast model, not the only one: a sibling
    quantitative (T1/T2/PD-based) contrast model can be added later without
    touching this class.

    What this is not
    -----------------
    Not a physical model of T1/T2 relaxation - it is a fit to an *observed*
    target intensity distribution. Not an acquisition simulator - it adds no
    noise, touches no k-space, and does not sample a trajectory; compose it
    with :class:`~augmentrum.augmentation.field_inhomogeneity.FieldInhomogeneity`,
    :class:`~augmentrum.augmentation.girf_artifacts.GIRFArtifacts`, coil and
    k-space modules for the acquisition side. Not a substitute for a real
    "LFScannerProfile" that also samples noise/coil/trajectory choices - this
    module supplies the contrast piece of such a profile; it does not decide
    the others.

    Preprocessing
    --------------
    "F_s" is estimated only from the foreground - either a supplied "mask", or
    (when "mask=None" and "auto_mask_fraction" is set) a simple threshold at
    "auto_mask_fraction" of the volume's 99th-percentile intensity. Foreground
    values are clipped to "clip_percentiles" before the quantile table is
    built, so a handful of outlier voxels cannot skew the whole mapping.
    Because the method is rank-based, any monotonic rescaling of the input
    (a separate "intensity normalization" pass) changes nothing about the
    result - clipping is the only preprocessing step that actually alters
    "F_s", so no separate normalization knob is exposed.

    "p = clip(F_s(x), epsilon, 1 - epsilon)" keeps a lone extreme voxel from
    being evaluated at "Q_t"'s own unstable endpoints.

    Template selection
    -------------------
    "template" (a single fixed :class:`ContrastTemplate`) always wins if given
    - deterministic, one target, no bank needed. Otherwise "templates" (a bank)
    is filtered against "reference" (see :func:`_compatible`) and:

    - "mode='select'": one compatible template is drawn at random per batch
      element (or used directly if only one is compatible).
    - "mode='interpolate'": "n_interpolate" compatible templates are drawn
      without replacement and mixed with Dirichlet weights,
      "Q_mix(p) = sum_i w_i * Q_i(p)" - a mixture of quantile functions, never
      an average of CDFs (the two are not interchangeable; only the former is
      "a distribution partway between these scanners").

    With neither "template" nor "templates" given, this module is the
    identity - matching the "supply None to disable" convention used
    elsewhere in Augmentrum (e.g. :class:`FieldInhomogeneity`).

    Args:
        templates: Bank of candidate templates for "mode='select'"/"'interpolate'".
        template: A single fixed target template, bypassing the bank/mode entirely.
        mode: "'select'" or "'interpolate'" (see above).
        reference: Metadata used to filter "templates" for compatibility (e.g.
            "{'field_strength_T': 0.075, 'sequence_family': 'T1w'}"). "None"
            (default) treats the whole bank as compatible.
        tolerances: Per-field relative-tolerance overrides for the numeric
            compatibility fields (default 0.2, i.e. +/-20%).
        n_interpolate: Number of templates drawn per batch element in
            "mode='interpolate'" (capped at however many are compatible).
        dirichlet_concentration: Concentration parameter for the Dirichlet
            mixture weights; 1.0 is uniform over the simplex.
        mask: Optional foreground mask, "(X, Y, Z[, T])" or one per subject,
            "(batch, X, Y, Z[, T])". Takes precedence over "auto_mask_fraction".
        auto_mask_fraction: Fallback foreground threshold, as a fraction of the
            volume's 99th-percentile intensity, used only when "mask=None".
            "None" disables masking (every voxel is foreground).
        clip_percentiles: "(low, high)" percentiles foreground voxels are
            clipped to before the source quantile function is estimated.
        epsilon: Endpoint margin the source CDF is clipped to before "Q_t" is
            evaluated, "p in [epsilon, 1 - epsilon]".
        n_quantiles: Number of "(p, value)" knots in the source quantile table.
        bias_field_correction: Optional "image -> image" callable applied to
            each batch element before masking/clipping (e.g. a wrapped N4).
            "None" (default) applies none - this module does not implement
            bias-field correction itself.
        fill_background: Value assigned outside the foreground mask in the
            output. "None" leaves background voxels at their input value.
        debug_outputs: When "True", also records the per-batch-element
            quantile table and clip range in "last_debug_".
        seed: RNG seed for template selection/Dirichlet weights.

    Examples:
        >>> lf = ContrastTemplate(quantile_p=np.linspace(0, 1, 5),
        ...                       quantile_values=np.array([0., 0.2, 0.4, 0.7, 1.0]),
        ...                       field_strength_T=0.075, sequence_family='T1w')
        >>> mapper = HistogramContrastMapping(template=lf)
        >>> out, water = mapper(volume_plus)   # (batch, X, Y, Z, T) NIfTI-MRS layout
    """

    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    MODES = ('select', 'interpolate')

    def __init__(self,
                 templates: Optional[Sequence[ContrastTemplate]] = None,
                 template: Optional[ContrastTemplate] = None,
                 mode: str = 'select',
                 reference: Optional[Dict[str, Any]] = None,
                 tolerances: Optional[Dict[str, float]] = None,
                 n_interpolate: int = 2,
                 dirichlet_concentration: float = 1.0,
                 mask: Optional[np.ndarray] = None,
                 auto_mask_fraction: Optional[float] = 0.1,
                 clip_percentiles: Tuple[float, float] = (0.5, 99.5),
                 epsilon: float = 1e-3,
                 n_quantiles: int = 1001,
                 bias_field_correction: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 fill_background: Optional[float] = 0.0,
                 debug_outputs: bool = False,
                 seed: Optional[int] = None):
        super().__init__()

        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")
        if template is not None and templates is not None:
            raise ValueError(
                "Pass either 'template' (a single fixed target) or 'templates' "
                "(a bank to select/interpolate from), not both."
            )
        if not 0.0 < epsilon < 0.5:
            raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}.")
        lo, hi = clip_percentiles
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(f"clip_percentiles must satisfy 0 <= low < high <= 100, got {clip_percentiles}.")
        if int(n_quantiles) < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}.")
        if int(n_interpolate) < 2:
            raise ValueError(f"n_interpolate must be >= 2, got {n_interpolate}.")
        if dirichlet_concentration <= 0.0:
            raise ValueError(f"dirichlet_concentration must be > 0, got {dirichlet_concentration}.")

        self.template = template
        self.templates = list(templates) if templates is not None else None
        self.mode = mode
        self.reference = dict(reference) if reference else None
        self.tolerances = dict(tolerances) if tolerances else None
        self.n_interpolate = int(n_interpolate)
        self.dirichlet_concentration = float(dirichlet_concentration)

        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.auto_mask_fraction = None if auto_mask_fraction is None else float(auto_mask_fraction)
        self.clip_percentiles = (float(lo), float(hi))
        self.epsilon = float(epsilon)
        self.n_quantiles = int(n_quantiles)
        self.bias_field_correction = bias_field_correction
        self.fill_background = None if fill_background is None else float(fill_background)
        self.debug_outputs = bool(debug_outputs)

        # Provenance - which template(s)/weights actually got used, per batch
        # element of the most recent call. Populated even without
        # "debug_outputs"; the heavier per-voxel arrays are gated on it.
        self.last_template_ids_: Optional[List[List[Optional[str]]]] = None
        self.last_weights_: Optional[List[List[float]]] = None
        self.last_mapping_params_: Optional[Dict[str, Any]] = None
        self.last_debug_: Optional[List[Dict[str, Any]]] = None

    @property
    def DOMAIN(self):
        """
        Intensity remapping is meaningless outside the image domain, but with
        no template configured this module is a true no-op - forcing a domain
        move for that would be a wasted round-trip, so none is requested then.
        """
        if self.template is not None or self.templates is not None:
            return Domain(spatial='image')
        return None

    #**************************#
    #   basemodule interface   #
    #**************************#
    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.PYTORCH, **kwargs):
        """
        Apply the quantile-function contrast mapping to a batch of images.

        Args:
            data_array: "(batch, X, Y, Z, T)" in the NIfTI layout - real or
                complex. Complex input is mapped by magnitude, with the
                original phase re-attached; this module never synthesizes
                phase itself (that is "complex object formation"'s job,
                upstream of this one).
            water_array: Passed through unchanged.
            backend: Unused; kept for the BaseModule signature.
            **kwargs: Absorbs whatever BaseModule injects (geometry, etc.).

        Returns:
            "(mapped_data, water_unchanged)", same shape and dtype in.
        """
        if self.template is None and self.templates is None:
            return data_array, water_array

        rank = len(ops.shape(data_array))
        if rank != 5:
            raise ValueError(
                "HistogramContrastMapping expects (batch, X, Y, Z, T) in the "
                f"NIfTI layout, got rank {rank}."
            )

        arr = ops.to_numpy(data_array)
        is_complex = np.iscomplexobj(arr)
        mag = np.abs(arr).astype(np.float64) if is_complex else arr.astype(np.float64)
        phase = np.angle(arr) if is_complex else None

        rng = self.rng.numpy_rng()
        n_batch = mag.shape[0]
        out_mag = np.empty_like(mag)

        template_ids: List[List[Optional[str]]] = []
        weights_out: List[List[float]] = []
        debug_entries: List[Dict[str, Any]] = [] if self.debug_outputs else []
        fg_masks: List[np.ndarray] = []

        for b in range(n_batch):
            block = mag[b]
            fg_mask = self._foreground_mask(block, b)
            fg_masks.append(fg_mask)
            if not fg_mask.any():
                raise ValueError(f"Foreground mask selected no voxels for batch element {b}.")

            corrected = (self.bias_field_correction(block)
                        if self.bias_field_correction is not None else block)
            if np.shape(corrected) != block.shape:
                raise ValueError(
                    "bias_field_correction must return an array with the same "
                    f"shape as its input, got {np.shape(corrected)} for {block.shape}."
                )

            fg_values = corrected[fg_mask]
            lo, hi = np.percentile(fg_values, self.clip_percentiles)
            clipped_fg = np.clip(fg_values, lo, hi)
            knots, probs = _empirical_quantile_table(clipped_fg, self.n_quantiles)

            p_full = np.interp(np.clip(corrected, lo, hi), knots, probs,
                               left=probs[0], right=probs[-1])
            p_full = np.clip(p_full, self.epsilon, 1.0 - self.epsilon)

            chosen, weights = self._select_templates(rng)
            q_full = np.zeros_like(p_full)
            for tmpl, w in zip(chosen, weights):
                q_full += w * tmpl.evaluate(p_full)

            out_mag[b] = np.where(fg_mask, q_full,
                                  block if self.fill_background is None else self.fill_background)

            template_ids.append([t.template_id for t in chosen])
            weights_out.append([float(w) for w in weights])
            if self.debug_outputs:
                debug_entries.append({
                    'batch_index': b, 'clip_range': (float(lo), float(hi)),
                    'quantile_table': (knots, probs),
                })

        self.last_template_ids_ = template_ids
        self.last_weights_ = weights_out
        self.last_mapping_params_ = {
            'clip_percentiles': self.clip_percentiles, 'epsilon': self.epsilon,
            'n_quantiles': self.n_quantiles, 'mode': self.mode,
        }
        self.last_debug_ = debug_entries if self.debug_outputs else None

        if is_complex:
            result = out_mag * np.exp(1j * phase)
            if self.fill_background is not None:
                fg_all = np.stack(fg_masks, axis=0)
                result = np.where(fg_all, result, self.fill_background)
        else:
            result = out_mag

        out = ops.cast_like(ops.match_backend(result, data_array), data_array)
        return out, water_array

    #***************#
    #   masking     #
    #***************#
    def _foreground_mask(self, block: np.ndarray, batch_idx: int) -> np.ndarray:
        """The boolean foreground mask for one batch element's spatial block."""
        if self.mask is not None:
            m = self.mask[batch_idx] if self.mask.ndim == block.ndim + 1 else self.mask
            if m.shape != block.shape:
                raise ValueError(f"mask shape {m.shape} does not match data shape {block.shape}.")
            return m

        if self.auto_mask_fraction is None:
            return np.ones(block.shape, dtype=bool)

        finite = block[np.isfinite(block)]
        if finite.size == 0:
            return np.zeros(block.shape, dtype=bool)
        robust_max = np.percentile(finite, 99.0)
        return block > (self.auto_mask_fraction * robust_max)

    #***************************#
    #   template selection      #
    #***************************#
    def _select_templates(self, rng: np.random.Generator
                          ) -> Tuple[List[ContrastTemplate], List[float]]:
        """One batch element's "(templates, weights)" for the mixture "sum_i w_i * Q_i(p)"."""
        if self.template is not None:
            return [self.template], [1.0]

        compatible = [t for t in self.templates if _compatible(t, self.reference, self.tolerances)]
        if not compatible:
            raise ValueError(
                "No template in the bank is compatible with 'reference' under the given tolerances."
            )

        if self.mode == 'select':
            idx = int(rng.integers(len(compatible))) if len(compatible) > 1 else 0
            return [compatible[idx]], [1.0]

        n = min(self.n_interpolate, len(compatible))
        idx = rng.choice(len(compatible), size=n, replace=False)
        chosen = [compatible[i] for i in idx]
        weights = rng.dirichlet(np.full(n, self.dirichlet_concentration))
        return chosen, weights.tolist()
