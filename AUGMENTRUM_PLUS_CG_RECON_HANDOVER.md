# Claude Code Handover: Physics-Informed CG Reconstruction for Augmentrum+

## Goal

Implement a **model-based conjugate-gradient (CG) reconstruction pipeline** in Augmentrum+ that uses the same composable acquisition physics already available in the forward operator.

The scientific experiment is a **forward-model mismatch study**:

1. Start from one known digital ground-truth image.
2. Generate one k-space dataset with a high-accuracy **full Augmentrum+ forward model** containing B0/off-resonance, B1+ modulation, B1-/receive sensitivity where available, measured GIRF effects, concomitant fields, and gradient nonlinearity (GNL).
3. Reconstruct the **same k-space data** repeatedly using CG with forward models of increasing completeness.
4. Quantify whether reduced model mismatch improves image fidelity.
5. Run leave-one-effect-out reconstructions to identify which omitted physical terms create the largest residual errors.

The key claim is **not** that CG is novel. CG is intentionally conventional. The contribution is that Augmentrum+'s scanner/acquisition physics can be used inside a unified iterative encoding operator, allowing these effects to be corrected during reconstruction rather than serving only as forward augmentations or post-hoc image corrections.

---

## 1. Inspect the repositories first

Before writing reconstruction code, inspect the current:

- Augmentrum `spatial_maps` branch;
- GIRF-Sim repository;
- existing NUFFT implementation;
- composable forward operator;
- adjoint implementations, if any;
- B0 frequency-segmentation code;
- B1+ and B1- handling;
- GIRF integration;
- concomitant-field implementation;
- GNL implementation;
- fully sampled 2D spiral already used for the spatial-physics figure.

### Framework ambiguity

The user described the current operator as **"PyTorch compatible (written in Keras)"**. Determine what is actually present: native PyTorch, Keras 3 using the PyTorch backend, TensorFlow/Keras with an adapter, or simply a terminology mistake.

**Do not introduce a new Keras dependency unless the repository genuinely uses it.** Prefer the tensor framework already used by the current forward model. The rest of this document assumes the reconstruction-facing operator can consume and return complex PyTorch tensors.

---

## 2. Scientific encoding model

Use the repository implementation as the source of truth for signs, units, Fourier normalization, coordinates, spherical-harmonic conventions, and segmentation.

For fixed maps/calibration parameters the acquisition should be **linear in the unknown image** `x`.

Conceptually:

```text
x(r)
 |
 +-- B1+ spatial modulation
 +-- B1- / coil sensitivity
 +-- B0 time-dependent phase
 +-- concomitant-field phase
 +-- GNL spatial encoding phase
 +-- measured GIRF
 |      order 1 -> realized gradient / trajectory
 |      order 0 -> global phase
 |      higher orders only if actually present and not double-counted
 +-- NUFFT along realized trajectory
 v
y(k,t,c)
```

A generic conceptual form is:

```text
y_c(t_n) = integral x(r) M_tx(r) C_c(r)
           exp[-i 2pi k_actual(t_n).r]
           exp[i phi_B0(r,t_n)]
           exp[i phi_conc(r,t_n)]
           exp[i phi_GNL(r,t_n)]
           exp[i phi_GIRF,HO(r,t_n)] dr
```

Do not transplant this sign convention blindly. Match the code.

### No double counting

Inspect GIRF-Sim/Augmentrum carefully. If the measured Bacon GIRF supplies only order 0/1 while GNL supplies higher-order spatial encoding, that is clean. If higher-order GIRF coefficients already feed a `GIRFNonlinearPhase` term, ensure the separate GNL configuration is not adding the same contribution twice.

---

## 3. Reconstruction objective

Implement Tikhonov-regularized least squares:

```text
x_hat = argmin_x ||A x - y||_2^2 + lambda ||x||_2^2
```

Solve the normal equations:

```text
(A^H A + lambda I)x = A^H y
```

with conjugate gradients.

For the first implementation, **do not add TV, wavelets, learned priors, plug-and-play denoisers, or neural reconstruction**. Those would obscure whether improvement comes from a better encoding model.

Recommended defaults:

- `lambda = 0` for noiseless controlled experiments if stable;
- a small configurable `lambda > 0` only when needed for noise/numerical stability;
- identical lambda for every compared model.

---

## 4. Reconstruction-facing operator API

Create or adapt a minimal linear-operator interface, preferably by wrapping the existing composable operator:

```python
class LinearMRIOperator:
    def forward(self, x):
        """Image -> acquired k-space/data."""

    def adjoint(self, y):
        """Acquired data -> mathematical adjoint image."""

    def normal(self, x, lam=0.0):
        return self.adjoint(self.forward(x)) + lam * x
```

Physics must be toggleable using the repository's existing configuration pattern or something equivalent to:

```python
PhysicsConfig(
    b0=True,
    b1plus=True,
    b1minus=True,
    girf=True,
    concomitant=True,
    gnl=True,
)
```

The same operator machinery should support nominal, cumulative, leave-one-out, and full configurations.

---

## 5. Adjoint correctness is mandatory

CG on normal equations depends on a correct adjoint. Add automated tests before interpreting reconstruction results.

### 5.1 Adjoint / inner-product test

For random complex image `x` and random complex data `y`:

```text
lhs = <A x, y>
rhs = <x, A^H y>
relative_error = |lhs-rhs| / max(|lhs|, |rhs|)
```

Target approximately `<1e-5` in complex64 where practical; test in complex128 when supported.

Test separately:

1. nominal;
2. B0 only;
3. B1+ only;
4. GIRF only;
5. concomitant only;
6. GNL only;
7. full operator.

### 5.2 Linearity test

For complex scalars `a`, `b` and images `x1`, `x2`:

```text
A(a*x1+b*x2) ~= a*A(x1)+b*A(x2)
```

All fixed-map physics in this experiment should remain linear in `x`. If a component is nonlinear in the image, isolate/reformulate it instead of placing it inside ordinary linear CG.

### 5.3 Normal-operator test

Numerically verify:

```text
<A^H A x1, x2> ~= <x1, A^H A x2>
real(<x, A^H A x>) >= 0
```

within numerical precision.

---

## 6. Adjoint implementation notes

Reuse existing code wherever possible.

### B1+

The current project description says B1+ is **complex-valued multiplication**, not Bloch excitation.

If forward uses:

```text
x_mod = B1plus * x
```

then adjoint uses:

```text
conj(B1plus) * x
```

Do not implement reciprocal division in the adjoint.

### B1- / receive sensitivity

For multi-coil data:

```text
forward:  C_c(r) * x(r)
adjoint:  sum_c conj(C_c(r)) * x_c(r)
```

If the first experiment is single coil, keep the implementation multi-coil-ready.

### B0

The current Augmentrum implementation uses frequency segmentation. The adjoint must be the mathematical adjoint of **that same discretized segmented forward approximation**.

### GIRF trajectory

If order-1 GIRF changes the trajectory, both forward and adjoint must use the same realized trajectory.

### Phase terms

If forward multiplies by `exp(+i phi)`, adjoint uses `exp(-i phi)` according to the repository convention.

### Concomitant/GNL segmented implementation

If the current accelerated forward model bins samples into severity segments, the adjoint must use the same bins and representative coefficients.

---

## 7. CG solver

Implement a small reusable torch-native solver.

Suggested result/API:

```python
@dataclass
class CGResult:
    image: torch.Tensor
    residual_history: list[float]
    rel_residual_history: list[float]
    n_iter: int
    converged: bool


def cg_solve(
    normal_op,
    rhs,
    x0=None,
    max_iter=30,
    rtol=1e-6,
    atol=0.0,
    preconditioner=None,
    callback=None,
):
    ...
```

Use complex-safe inner products, e.g. `torch.vdot` over flattened tensors. Keep NumPy conversions outside the iterative loop.

For the ISMRM experiment gradients through CG are not required. Still, avoid unnecessary in-place operations and keep an optional differentiable path possible later.

First make **unpreconditioned CG** correct. Add simple preconditioning only if necessary, and use the same strategy for all model comparisons.

---

## 8. Density compensation warning

Do not confuse density compensation with the mathematical adjoint.

For iterative reconstruction:

- define a true forward/adjoint pair first;
- if DCF is useful for initialization or preconditioning, treat it explicitly as such;
- do not insert arbitrary DCF only into the adjoint and still call it `A^H`.

---

## 9. Avoid the most obvious inverse crime

Synthetic ground truth is desirable because image error is known exactly. However, if data generation and reconstruction use the identical discretized forward approximation, the full-model result becomes almost tautological.

Use a more accurate **reference data-generation operator `A_ref`** than the practical reconstruction operator.

Preferred reference settings:

- finer B0 frequency segmentation;
- finer concomitant/GNL severity segmentation;
- direct/per-sample evaluation for a small reference matrix if computationally feasible;
- measured GIRF trajectory;
- full spatial maps.

The practical `A_full` reconstruction can use normal production segmentation settings.

If direct reference simulation is too expensive, demonstrate convergence with sufficiently fine segmentation and document the reference settings.

---

## 10. Primary dataset/acquisition

Reuse the same setup as the Augmentrum+ spatial-physics ablation figure so the figures form a pair:

- one recognizable digital brain;
- the existing **fully sampled 2D spiral** from Augmentrum;
- same FOV/matrix where practical;
- same B0 map;
- same B1+ map;
- B1-/coil maps if used;
- measured Bacon GIRF;
- concomitant-field model;
- GNL model.

### Noise

Start with noiseless/high-SNR data so model mismatch is isolated. Add one fixed-SNR robustness experiment later if useful. Do not let noise become the main story.

---

## 11. Generate one reference dataset

Generate the data **once**:

```python
y_ref = A_ref.forward(x_ground_truth)
```

Every reconstruction below must use exactly the same `y_ref`.

This needs to be explicit in code, metadata, and Figure Panel A.

---

## 12. Main cumulative reconstruction series

Run the same CG solver/settings on:

### R0 — nominal

Nominal trajectory/NUFFT, with receive-coil handling included consistently if relevant.

### R1 — +B0

Nominal + B0/off-resonance.

### R2 — +B0 + B1+

R1 + supplied B1+ modulation.

### R3 — +B0 + B1+ + measured GIRF

R2 + measured GIRF trajectory/global response.

### R4 — +concomitant

R3 + concomitant fields.

### R5 — full

R4 + GNL.

If B1- is simply known coil sensitivity, include it in all reconstructions rather than treating it as an incremental new-physics step.

### Ordering caveat

The cumulative ordering is for visualization. It cannot uniquely assign importance because earlier/later terms interact. Use leave-one-out analysis for attribution.

---

## 13. Leave-one-effect-out reconstructions

Starting from the full reconstruction operator, remove one term at a time:

```text
FULL
FULL - B0
FULL - B1+
FULL - GIRF
FULL - concomitant
FULL - GNL
```

All use the same `y_ref`.

This answers:

> If every other acquisition effect is modeled correctly, how much reconstruction error remains when this one term is omitted?

Calculate:

- magnitude-image NRMSE;
- complex NRMSE if useful;
- SSIM if already available;
- one common full/reference-model data residual;
- optional phase RMSE inside the object mask.

---

## 14. Fair data-consistency metric

Do not compare only each solver's native residual under its own model. A wrong model can converge perfectly to the wrong inverse problem.

For every reconstructed image `x_hat_j`, evaluate with one common operator:

```text
true_model_residual_j = ||A_ref x_hat_j - y_ref|| / ||y_ref||
```

If `A_ref` is too expensive, use the same `A_full` evaluation operator for all reconstructions.

Save native CG residual histories for numerical convergence diagnostics, but do not use them alone as evidence of physical accuracy.

---

## 15. Initialization and fairness

Use identical:

- input data;
- initialization;
- max iterations;
- stopping rule;
- lambda;
- precision;
- device where possible.

Recommended initial implementation:

```text
x0 = 0
max_iter = 30
rtol = 1e-6
```

Run a convergence study before freezing final values. Every model gets the same reconstruction budget.

---

## 16. Caching / operator preparation

CG may call `A` and `A^H` dozens of times. Cache every static quantity possible:

- nominal and GIRF-adjusted trajectories;
- B0 segmentation coefficients/weights;
- concomitant severity bins and representative coefficients;
- GNL severity bins and representative coefficients;
- real-solid-harmonic basis evaluations;
- coil maps;
- B1+ maps;
- any static interpolation tables.

Prefer an API like:

```python
A = build_operator(config)
A.prepare()
result = cg_reconstruct(A, y)
```

Do not recompute physics segmentation inside every CG iteration.

---

## 17. Suggested code organization

Adapt to the repository's real structure rather than creating a parallel framework unnecessarily.

Conceptually:

```text
augmentrum/
    reconstruction/
        linear_operator.py
        cg.py
        metrics.py

scripts/
    ismrm/
        figure_physics_informed_cg.py

tests/
    reconstruction/
        test_adjointness.py
        test_linearity.py
        test_cg_core.py
        test_cg_nominal_nufft.py
        test_cg_full_operator.py
```

---

## 18. End-to-end tests

### CG core

Test:

- identity operator;
- known diagonal Hermitian positive-definite system;
- complex-valued system;
- convergence metadata.

### MRI operator

Test:

- nominal adjointness;
- each individual physics term;
- full operator;
- zero-effect parameters reduce exactly/approximately to nominal encoding.

### Small phantom regression

On ~32x32 data:

```text
x -> A_full -> y -> CG(A_full)
```

should recover `x` tightly in a noiseless, adequately sampled case.

Then:

```text
CG(A_nominal)
```

should show larger error when effects are intentionally nonzero.

---

# Figure plan

## Scientific message

> The same full-physics k-space dataset is reconstructed repeatedly. As the reconstruction operator accounts for more of the physics that generated the acquisition, spatial model-mismatch artifacts and quantitative reconstruction error decrease.

This should look like a **model-based reconstruction figure**, not a second artifact gallery.

---

## Panel A — Experimental schematic

Show:

```text
Ground truth
   |
   v
A_ref = full Augmentrum+ acquisition model
[B0 | B1+ | measured GIRF | concomitant | GNL]
   |
   v
ONE k-space dataset y_ref
   |
   +--> nominal CG
   +--> partial-physics CG
   +--> full-physics CG
```

Include a small fully sampled 2D spiral trajectory icon.

Visibly state:

> Same k-space data for every reconstruction

---

## Panel B — Main image comparison

Use four large images:

1. Ground truth
2. Nominal CG
3. Intermediate physics-aware CG
4. Full-physics CG

Recommended intermediate:

```text
+B0 +B1+ +GIRF
```

Use identical grayscale/windowing.

Report beneath each reconstruction:

- NRMSE;
- common/reference-model data residual.

---

## Panel C — Cumulative matched error maps

Show compact error maps for:

```text
Nominal
+B0
+B0+B1+
+B0+B1++GIRF
+Concomitant
Full (+GNL)
```

Use one common display scale.

Preferred error:

```text
|x_hat - x_GT|
```

with explicitly documented normalization.

This panel should visually show progressive reduction of model-mismatch structure.

---

## Panel D — Quantitative attribution

### D1. Error versus model completeness

X-axis:

```text
Nominal -> +B0 -> +B1+ -> +GIRF -> +Concomitant -> +GNL
```

Y-axis:

```text
NRMSE [%]
```

Use bars or a connected line.

### D2. Leave-one-effect-out sensitivity

Horizontal bars:

```text
Full
-B0
-B1+
-GIRF
-Concomitant
-GNL
```

Metric:

```text
Delta NRMSE relative to full
```

D2 is more scientifically important than a generic CG convergence plot because it does not depend on the arbitrary cumulative ordering.

---

## Optional convergence inset

If useful, plot **ground-truth NRMSE versus iteration** for nominal and full models.

A useful result would be:

- both CG solves numerically converge;
- nominal-model image error plateaus at a higher level;
- full-model reconstruction reaches lower image error.

That demonstrates that the nominal failure is model mismatch, not optimizer failure.

Do not use native residual alone for this conclusion.

---

## 19. Metrics

Primary:

```text
NRMSE = ||x_hat-x_GT|| / ||x_GT||
```

Report as percent.

Also calculate:

- common/reference-model data residual;
- SSIM on magnitude if already available;
- optional phase RMSE inside mask;
- PSNR only if useful.

Do not crowd the figure with every metric.

Recommended displayed metrics:

- NRMSE;
- common data residual.

---

## 20. Region-specific analysis

Whole-image NRMSE may hide localized improvements.

Optionally define fixed high-severity ROIs from the physics maps, e.g. the top 20% of absolute severity inside the brain mask for:

- |B0|;
- |B1+ - 1|;
- concomitant phase;
- GNL phase/encoding error.

Calculate reconstruction error inside these regions.

Fix the rule globally; do not tune thresholds independently to improve results.

---

## 21. Computational profiling

Record for each model:

- forward-call time;
- adjoint-call time;
- CG wall time;
- iteration count;
- peak GPU memory if straightforward.

Full physics will likely cost more. Report rather than hide that tradeoff.

---

## 22. Suggested experiment CLI/config

Create something like:

```bash
python scripts/ismrm/figure_physics_informed_cg.py \
  --config configs/ismrm_cg_full.yaml \
  --output outputs/ismrm_cg_recon
```

Config should capture at least:

```yaml
seed: 17
device: cuda
dtype: complex64

image:
  source: ...
  matrix: ...
  fov_mm: ...

sequence:
  type: existing_augmentrum_fully_sampled_2d_spiral
  path: ...

physics:
  b0_map: ...
  b1plus_map: ...
  b1minus_maps: ...
  girf: bacon_measured
  concomitant: true
  gnl: true

reference_forward:
  b0_segments: ...
  severity_segments: ...

reconstruction_forward:
  b0_segments: ...
  severity_segments: ...

cg:
  max_iter: 30
  rtol: 1e-6
  lambda: 0.0
  init: zeros

noise:
  enabled: false
  snr_db: 40
```

Adapt names to existing project conventions.

---

## 23. Required outputs

```text
outputs/ismrm_cg_recon/
    figure_physics_informed_cg.pdf
    figure_physics_informed_cg.svg
    figure_physics_informed_cg.png

    ground_truth.npz
    kspace_reference.npz

    recon_nominal.npz
    recon_b0.npz
    recon_b0_b1.npz
    recon_b0_b1_girf.npz
    recon_plus_concomitant.npz
    recon_full.npz

    leaveout_b0.npz
    leaveout_b1.npz
    leaveout_girf.npz
    leaveout_concomitant.npz
    leaveout_gnl.npz

    metrics.csv
    metrics.json
    convergence.json
    runtime.json
    config_resolved.yaml
    git_versions.json
    figure_caption.txt
```

Suggested metrics table:

```text
reconstruction,nrmse,ssim,true_model_residual,native_final_residual,iterations,runtime_s
```

---

## 24. B1+ wording constraint

Because B1+ is currently represented as a multiplicative complex spatial modulation rather than Bloch excitation, phrase this as:

- **B1+-aware reconstruction**;
- **reconstruction incorporating the supplied B1+ modulation map**.

Do not claim generic flip-angle correction or Bloch-based transmit correction.

---

## 25. GIRF wording constraint

For the primary experiment use the independently measured Bacon GIRF already integrated in GIRF-Sim.

Do not invent measured higher-order harmonics that are not present in the loaded measured data. Use the project's separate GNL term for higher-order spatial encoding if that is how the repositories are organized.

---

## 26. Success criteria

Before making scientific claims, demonstrate:

1. adjoint correctness for the complete operator;
2. CG convergence for every compared configuration;
3. full-model reconstruction is materially closer to ground truth than nominal reconstruction for full-physics data;
4. improvements occur in spatial regions predicted by the corresponding maps/encoding errors;
5. leave-one-out tests identify effect-specific residual errors;
6. improvement persists when reference data generation is more accurate than the practical reconstruction operator;
7. no model receives different iterations, initialization, regularization, or data;
8. every reconstruction uses the exact same k-space dataset.

If the full model does not improve reconstruction, debug before tuning figure appearance. Check:

- adjointness;
- coordinate/FOV consistency;
- phase sign;
- trajectory units;
- segmentation mismatch;
- duplicate physics terms;
- effect severity;
- CG conditioning/convergence.

---

## 27. Optional second-stage experiment

After the fully sampled spiral works, repeat at moderate undersampling, e.g. `R~2` or reduced spiral interleaves.

This tests whether physics-aware encoding remains useful in a harder inverse problem.

Do **not** make undersampling the first milestone. Fully sampled data are easier to debug and interpret.

---

## 28. What the figure should not claim

Do not claim:

- novel CG mathematics;
- in-vivo validation;
- universal scanner correction;
- correction of unknown maps/fields;
- Bloch-accurate B1+ excitation correction;
- vendor-specific GNL correction unless actual vendor coefficients are used;
- recovery of information not encoded in the acquisition.

The defensible claim is:

> **When scanner/acquisition physics are known or modeled, incorporating them into the iterative encoding operator reduces reconstruction error caused by forward-model mismatch.**

---

## 29. Draft figure caption (<500 characters)

> **Physics-aware iterative reconstruction with Augmentrum+. Fully sampled spiral data were generated from a common digital brain using the complete forward model, including B0/B1+ maps, measured GIRF response, concomitant fields, and gradient nonlinearity. CG reconstructions used increasingly complete encoding models. Matched error maps and leave-one-effect-out metrics show how incorporating acquisition physics reduces model-mismatch artifacts.**

Check the character count automatically before final export and keep it below 500 characters.

---

## 30. Results-text template

Do not fill numbers until the experiment has run.

> Reconstructing full-physics spiral data with the nominal encoding operator produced spatially structured residual error (NRMSE=[XX]%). Incorporating B0, B1+, and GIRF terms reduced NRMSE to [XX]%, while the complete operator including concomitant-field and GNL terms reached [XX]%. Leave-one-effect-out reconstructions showed the largest residual degradation when [effect] was omitted ([XX] percentage-point NRMSE increase). Both nominal and full CG solvers numerically converged, indicating that the remaining nominal-model error reflected encoding mismatch rather than optimizer failure.

---

## 31. Relationship to the previous Augmentrum+ figure

The existing spatial-physics ablation figure asks:

> **What does each acquisition effect do to the image?**

This new reconstruction figure asks:

> **If those effects are known and placed inside the inverse model, can reconstruction undo the resulting model-mismatch artifacts?**

Use the same anatomy, maps, measured GIRF, and 2D spiral wherever possible so the two figures read as one coherent story.

---

## 32. Relevant methodological precedent

The implementation should remain repository-consistent, but the scientific approach is consistent with established model-based MRI work:

- off-resonance can be included in non-Cartesian encoding/reconstruction using segmented approximations;
- GNL can be included directly in the non-Cartesian signal model rather than corrected only after reconstruction;
- Tikhonov-regularized non-Cartesian SENSE-type inverse problems can be solved with CG.

Useful literature to inspect if needed:

- *NonCartesian MR image reconstruction with integrated gradient nonlinearity correction*.
- *An efficient gridding reconstruction method for multishot non-Cartesian imaging with correction of off-resonance artifacts*.
- MaxGIRF / Cartesian MaxGIRF literature already referenced within GIRF-Sim.

Use external literature to validate the approach, not to overwrite the conventions already implemented in Augmentrum+/GIRF-Sim.
