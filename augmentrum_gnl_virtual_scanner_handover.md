# Augmentrum GNL Virtual Scanner — Claude Code Handover

## Objective

Add a physically faithful gradient-nonlinearity (GNL) virtual-scanner model to Augmentrum. The model must generate **raw MR signal using the spatially nonlinear gradient field**, not by applying an image-space deformation as a shortcut. It should integrate with the existing GIRF-Sim outputs for eddy currents, concomitant fields, and mechanical modes.

The existing GIRF-Sim remains responsible for those effects. This implementation adds **GNL only**.

## Required GNL generation modes

Expose three mutually exclusive GNL source modes:

1. **Measured calibration (`measured`)**
   - Load scanner calibration data and use it directly.
   - Support the native calibration representation when possible.
   - Internally convert to a common scanner-field representation used by the forward model.
   - Preserve scanner-specific coefficients, units, coordinate convention, gradient-axis convention, and spatial scaling.

2. **Measured + stochastic variation (`measured_stochastic`)
   - Start from a measured calibration.
   - Add physically constrained stochastic perturbations to the calibration coefficients/field model.
   - Perturbations should be specified by harmonic order and optionally axis.
   - Support reproducible RNG seed.
   - The perturbed field must remain physically plausible and should not introduce arbitrary nonphysical spatial warps.

3. **Pure stochastic (`stochastic`)
   - Generate a scanner GNL model without calibration data.
   - Represent each physical gradient field with spherical-harmonic spatial terms.
   - Support harmonic orders through **9th order**.
   - Draw coefficients from configurable distributions, ideally parameterized by order and gradient axis.
   - Provide sensible defaults with decreasing magnitude at increasing harmonic order, while allowing all statistics to be overridden.
   - Support reproducible RNG seed.

## Four levels of realism

Expose `realism_level = 1..4` independently of the source mode:

### Level 1 — Analytical GNL
Idealized low-order spherical-harmonic model. Intended for unit tests and controlled experiments.

### Level 2 — Scanner-specific GNL
Measured/vendor scanner coefficients or equivalent scanner-specific calibration. No deliberate mismatch between truth and correction model.

### Level 3 — Calibrated scanner
Higher-fidelity measured calibration, potentially including higher-order terms and calibration-derived coefficient uncertainty.

### Level 4 — Imperfect virtual scanner
Generate acquisition data from a **truth GNL model** that is more complete/noisy than the correction model supplied to GradUnwarp. Deliberately model realistic mismatch such as:
- coefficient uncertainty
- omitted higher-order terms in the correction model
- calibration noise
- small coordinate/scale uncertainty

Do not make GradUnwarp's correction model identical to the truth model at Level 4. The purpose is to leave realistic residual artifacts after correction.

## Physical forward model

For each gradient axis `j`, represent the actual longitudinal gradient field as:

```text
Bz_j(r) = Bz_j_linear(r) + Bz_j_nonlinear(r)
```

with the nonlinear component represented by spatial basis functions, preferably spherical harmonics:

```text
Bz_j_nonlinear(r) = sum_n c[j,n] * Phi_n(r)
```

For gradient waveform `g_j(t)`, accumulate GNL phase as:

```text
phi_gnl(r,t) = gamma * sum_j [ Bz_j_nonlinear(r) * integral_0^t g_j(tau) d tau ]
```

More generally, retain the representation as:

```text
phi_gnl(r,t) = sum_n q[n,t] * Phi_n(r)
```

where `q[n,t]` is derived from the commanded gradient waveform and the calibration coefficients.

The MR signal model must include GNL in the **encoding phase**:

```text
s(t) = integral rho(r) * C(r) * exp(-i * phi_total(r,t)) dr
```

where `phi_total` already combines nominal Fourier encoding with the existing Augmentrum/GIRF-Sim effects and the new GNL term.

### Important
Do **not** implement GNL primarily as a post-hoc image-space displacement/warp.

Do **not** replace GNL with one globally modified k-space trajectory `k_gnl(t)`. The GNL encoding is spatially dependent. The effective local gradient differs across the object.

The existing Fourier-dual coordinate transformation machinery can still be used for transformations that are actually representable as coordinate operations, but GNL must enter the signal encoding as a spatially dependent phase/gradient-field term.

## B0 integration

Existing frequency-segmented B0 simulation remains valid and should be retained.

Treat B0 and GNL as separate physical phase terms:

```text
phi_total = phi_nominal + phi_B0 + phi_GNL + phi_GIRF
```

where `phi_GIRF` covers the already implemented eddy-current, concomitant-field, and mechanical-mode effects.

Do not fold B0 and GNL into one empirical warp model.

## Computational strategy

Optimize for large 3D volumes and long non-Cartesian acquisitions.

### Preferred representation

Use a low-dimensional harmonic representation:

```text
phi_gnl(r,t) = sum_n Phi_n(r) * q[n,t]
```

Precompute/cache:
- spatial harmonic basis maps `Phi_n(r)` for the requested FOV/grid
- coefficient matrices for each gradient axis
- temporal integrated-gradient terms `q[n,t]`

Do not repeatedly evaluate spherical harmonics for every voxel/sample if they can be cached.

### Avoid
- dense voxel-by-time GNL phase arrays when a basis representation can be used
- repeated construction of the same spatial harmonic fields for every repetition/shot
- unnecessary NUFFT calls
- converting between coordinate systems multiple times

### Preserve Augmentrum architecture

Use the existing augmented-coordinate / nominal-coordinate distinction wherever appropriate:
- augmented/physical model generates the true measurement
- nominal coordinates remain available to the reconstruction/regridding path
- GNL itself is not reduced to a nominal-coordinate warp

The implementation must compose efficiently with existing trajectory augmentations and GIRF-Sim.

## API/configuration requirements

All GNL settings must be exposed through the normal Augmentrum module configuration/API. Do not hide important simulation choices in module internals.

Suggested configuration structure:

```python
GNLConfig(
    enabled=True,
    source="measured",              # measured | measured_stochastic | stochastic
    realism_level=2,                 # 1..4

    calibration_path=None,
    calibration_format=None,

    max_harmonic_order=5,            # <= 9; 9 for stochastic/full model
    include_linear_terms=False,

    stochastic_seed=None,
    coefficient_distribution=None,
    coefficient_scale=None,
    coefficient_std_by_order=None,
    coefficient_std_by_axis=None,

    truth_model=None,                # for realism level 4
    correction_model=None,           # optional separate model for validation

    cache_basis=True,
    cache_phase=True,
    dtype=None,
)
```

Use the project's existing configuration style/naming conventions rather than forcing this exact class if Augmentrum has a preferred pattern.

### Calibration input

The loader should support:
- measured spherical-harmonic coefficient files
- equivalent scanner calibration representations already used by GradUnwarp, when practical
- explicit coordinate-system metadata where available

Validate units and conventions on load. Fail loudly on ambiguous units, axis ordering, or coordinate convention.

## Stochastic model requirements

The stochastic model must operate on **physical calibration parameters**, not arbitrary image deformations.

For each harmonic order `l` and gradient axis `j`, expose configurable:

```text
mean[l,j]
std[l,j]
correlation structure (optional)
seed
```

Recommended default behavior:
- preserve measured coefficients exactly when stochastic mode is disabled
- perturb measured coefficients with zero-mean errors when stochastic variation is enabled
- reduce default coefficient variance with increasing harmonic order
- optionally correlate coefficients within the same physical calibration family
- keep the random model separable from acquisition noise

The stochastic model must be reproducible.

## Reality of the scanner model

The truth model should describe the **actual magnetic field generated by the gradient coils** for a commanded waveform. It should therefore be possible to compute the local gradient field and accumulated phase from the scanner model.

The correction model should be separately selectable for validation, especially at realism level 4.

This separation is required so that GradUnwarp can be applied to simulated reconstructed data and leave residuals that arise from model/calibration mismatch rather than from an artificially perfect inverse.

## Validation / tests

Implement tests for at least:

1. Zero GNL -> exactly reproduce the pre-GNL forward model.
2. Pure linear gradient terms -> reproduce nominal Fourier encoding.
3. Single known harmonic -> compare numerical forward encoding against an independently computed analytical phase field.
4. Measured calibration -> verify coefficient loading and field reconstruction.
5. Stochastic mode -> same seed gives identical coefficients and data.
6. Different seeds -> statistically different but physically plausible fields.
7. Order 9 -> all supported harmonic terms evaluate correctly and remain numerically stable.
8. Realism level 4 -> correction model is not identical to truth model and produces nonzero residual after correction.
9. Combined B0 + GNL -> verify both phase terms are simultaneously present without double application.
10. Combined GNL + GIRF-Sim -> verify existing eddy-current, concomitant-field, and mechanical-mode effects are neither removed nor duplicated.

## Validation outputs

Provide optional debug outputs for:
- harmonic coefficients used for truth model
- correction-model coefficients
- spatial GNL field components
- accumulated GNL phase for selected time points
- local effective gradient vectors
- truth-vs-nominal displacement field derived from the encoding model
- truth-vs-correction residual field

These should be opt-in because the full-resolution arrays can be large.

## Interaction with GradUnwarp

The primary validation workflow should be:

```text
physical object
  -> nominal gradient waveforms
  -> true GNL field model
  -> GNL-aware MR signal generation
  -> nominal reconstruction
  -> GradUnwarp using correction calibration
  -> evaluate residual geometric/intensity/spectral artifacts
```

At realism level 4, the GradUnwarp input should normally represent the calibration available to a real user, not the hidden truth coefficients used to generate the data.

## Implementation priority

1. Build a common internal GNL field representation.
2. Implement measured calibration loading.
3. Implement harmonic-basis phase encoding.
4. Integrate with the existing Augmentrum forward operator.
5. Add measured+stochastic perturbations.
6. Add pure stochastic generation through 9th order.
7. Add realism levels and truth/correction model separation.
8. Add caching/vectorization and benchmark against the existing forward model.
9. Add unit/integration tests and optional diagnostic outputs.

## Non-goals

Do not redesign GIRF-Sim.

Do not replace the existing B0 frequency-segmentation implementation.

Do not turn GNL into an image-space-only augmentation.

Do not assume one globally warped k-space trajectory is an exact representation of GNL.
