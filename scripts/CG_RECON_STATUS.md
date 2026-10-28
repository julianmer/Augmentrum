# Status: physics-informed CG reconstruction (AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md) — pick up here

Started this session, after the spatial-physics ablation figure work (see
`SPATIAL_PHYSICS_FIGURE_STATUS.md` for that figure's own, separate history - this is a companion
figure, not a continuation of that investigation).

## What's done and verified

- `augmentrum/reconstruction/cg.py` — torch-native CG solver (`cg_solve`/`CGResult`). 4/4 tests
  pass (`tests/reconstruction/test_cg_core.py`): identity operator, diagonal Hermitian PD system,
  general complex Hermitian system, non-convergence metadata.
- `augmentrum/reconstruction/segmented_phase.py` — `SegmentedPhaseTerm`: generic severity-
  segmented approximation of ANY phase term with a `.phase(positions, t)` method (same interface
  every PhaseTerm in this repo already implements), so B0/concomitant/GNL all reuse ONE
  segmentation implementation rather than three bespoke ones. Bins voxels by a scalar severity,
  represents each bin by the EXACT phase curve at one representative voxel (not a fitted curve).
  4/4 tests pass. **Important fix during development**: the default severity metric must be
  SIGNED (phase at the last/a reference time sample), not RMS-over-t - RMS is sign-blind (+f and
  -f off-resonance give identical RMS severity despite opposite-sign phase curves), which caused
  near-pi representation errors regardless of segment count before this was caught and fixed.
- `augmentrum/reconstruction/linear_operator.py` — `LinearMRIOperator`/`PhysicsConfig`: a
  composable forward/adjoint MRI operator built on `GriddingNUFFT.forward`/`.adjoint` (confirmed
  to be a clean, DCF-free mathematical adjoint pair already - density compensation lives
  separately in `density_weights()`, never baked into `.adjoint()`, so no new adjoint derivation
  was needed for the NUFFT part itself) plus B1+/coil complex multiplication and the segmented
  phase terms. When multiple phase terms are active simultaneously, they are summed into ONE
  phase field and segmented TOGETHER (`_SumPhaseTerms`) - segmenting each independently and
  multiplying the results would NOT correctly reproduce `exp(i*(phiA+phiB))`. All 9 mandatory
  adjoint-correctness tests from the handover's own section 5 pass
  (`tests/reconstruction/test_linear_operator.py`): inner-product/adjoint test for nominal, B0,
  B1+, GIRF, concomitant, GNL, and the full operator; linearity; Hermitian + PSD normal operator.
- **Real units/API gotcha, fixed**: `GriddingNUFFT.forward`/`.adjoint` call `np.asarray(coords)`
  internally - `coords` MUST be a plain numpy array, never a CUDA (or any device) tensor, even
  though the image/position tensors passed elsewhere in the same operator can and should be on
  GPU. Caused a `TypeError: can't convert cuda:0 device type tensor to numpy` the first time the
  real-data experiment script ran; fixed in `figure_physics_informed_cg.py`'s context builder.
- `tests/reconstruction/test_cg_small_phantom.py` — handover section 18's "small phantom
  regression": CG(A_full) tightly recovers a smooth (deliberately non-sharp-edged) test image,
  and CG(A_nominal) shows materially larger error on the SAME full-physics data. 2/2 pass.
  **Phantom choice matters**: an EARLIER version of this test used a sharp-edged Shepp-Logan
  phantom and failed with ~18-20% recovery error regardless of CG iterations, regularization, or
  even swapping in an EXACT brute-force NUDFT (no gridding approximation at all) - this is real,
  unavoidable Gibbs-phenomenon residual from a discontinuous object's energy extending beyond any
  finite k-space bandwidth, not a bug. Confirmed by direct comparison against exact NUDFT giving
  the identical ~18% floor. Switched to a smooth multi-Gaussian test image, which has no such
  floor, so near-exact recovery is now the correct expected (and achieved) outcome.
- **`scripts/ismrm/figure_physics_informed_cg.py`** - the real experiment script (not a test),
  reusing `RepoAdapter`'s actual physics (real B0/B1+ maps, real measured Bacon GIRF, real
  concomitant/GNL phase terms, the same spiral GRE trajectory as the ablation figure) rather than
  synthetic stand-ins. Implements the full cumulative R0-R5 series and all 5 leave-one-out
  configs (handover sections 12-13), one shared `y_ref` (section 11), and the "common operator"
  fairness metric (section 14: every reconstruction is scored by `||A_full(x_hat) - y_ref|| /
  ||y_ref||`, not its own native CG residual). **Verified working end-to-end** on a quick
  matrix=32, 6-segment, 20-iteration validation run:
  - NRMSE decreases monotonically R0->R5: 0.2484 -> 0.2485 -> 0.2106 -> 0.1999 -> 0.1998 -> 0.1867
  - true_model_residual drops much more sharply: 0.0528 -> 0.0528 -> 0.0425 -> 0.0204 -> 0.0206 -> 0.0050
  - Leave-one-out attribution is sensible: removing B1+ hurts the MOST (nrmse 0.2390, worse than
    nominal!), GIRF and GNL matter moderately (~0.20), concomitant and B0 barely matter at 3T for
    this anatomy (removing either is within noise of the full result) - consistent with the
    spatial-physics ablation figure's own finding that B0/concomitant are small at 3T.
  - CG did not fully converge within 20 iterations at this quick setting (`converged=False` in
    the saved metrics) - the TREND is already clear and correct, but do not treat these specific
    numbers as final; rerun a real convergence study (handover section 15) before any numbers go
    in the actual abstract.

## Not done yet (in handover section order)

1. **Convergence study** (section 15) - current quick validation used max_iter=20, which does not
   fully converge. Need to determine real max_iter/rtol needed at the ACTUAL target matrix size
   (handover reuses the ablation figure's own matrix, likely 80, not this validation's 32) before
   freezing final settings; same budget for every compared model.
2. **Avoid-inverse-crime check** (section 9) - current validation uses the SAME discretization
   (`n_segments=6`) for both `y_ref` generation and every reconstruction operator. The handover
   wants a MORE ACCURATE reference operator (finer segmentation, or direct/exact evaluation for a
   small-enough matrix) than the practical reconstruction operator, so the full-model result
   isn't tautological. Needs a decision: either run `y_ref` at a much finer `n_segments` (cheap -
   y_ref is generated ONCE) while reconstructions use a coarser, faster setting, or justify why
   the shared setting is already fine-enough (e.g. show the segmentation error vs. exact phase is
   already below some threshold at the chosen segment count - `SegmentedPhaseTerm.
  max_abs_representation_error()` already exists for exactly this check, just hasn't been run at
   the real target resolution yet).
3. **Full-resolution run** at the ablation figure's own matrix/FOV (not yet attempted - matrix=32
   took ~4-35s per reconstruction depending on segment overhead; matrix=80 with real, finer
   segmentation will cost substantially more - budget/time this before committing to final
   settings, per the handover's own section 21 profiling requirement).
4. **Region-specific analysis** (section 20) - not implemented.
5. **Computational profiling** (section 21) - partial (runtime_s is already recorded per
   reconstruction in metrics.json) but not reported/summarized anywhere yet.
6. **The actual 4-panel ISMRM figure** (sections "Figure plan", Panel A-D) - NOT built yet. Only
   the underlying experiment/metrics pipeline exists. Panel A (schematic), B (4 example images),
   C (cumulative error maps), D1 (NRMSE vs. model completeness), D2 (leave-one-out delta-NRMSE
   bars) all still need plotting code, following the same general plotting conventions already
   established in `spatial_physics_ablation.py` (shared colorbars, `_imshow_clean`-style axes,
   etc. - reuse that script's helpers where sensible rather than reinventing them).
7. Required output files per handover section 23 (`figure_caption.txt`, `config_resolved.yaml`,
   `git_versions.json`, SVG/PDF exports, etc.) - only `metrics.json` and the raw `.npz` images
   exist right now.
8. Draft figure caption / results-text template (handover sections 29-30) - not started, can't be
   filled in with real numbers until the full-resolution, converged run exists.

## Key gotchas for next session (don't rediscover these)

- `GriddingNUFFT.forward`/`.adjoint` coords argument MUST be numpy, never a device tensor, even
  inside an otherwise-all-GPU operator.
- Density compensation (`density_weights()`) is a SEPARATE method from `.adjoint()` and must
  NEVER be folded into a CG-facing adjoint (handover section 8) - already respected throughout.
- Multiple simultaneously-active phase terms must be SUMMED then segmented together, not
  segmented independently and multiplied (`_SumPhaseTerms` already handles this correctly).
- Segmentation severity must be SIGNED, not an RMS/magnitude-only measure, or opposite-sign
  voxels with equal magnitude get wrongly binned together.
- A sharp-edged test phantom (Shepp-Logan) is the WRONG choice for validating operator/CG
  correctness specifically - its Gibbs-phenomenon floor (~18-20% NRMSE, confirmed present even
  with an exact brute-force NUDFT) will look like a bug but isn't one. Use a smooth test image for
  that purpose; save Shepp-Logan-like sharp phantoms for the real experiment, where that floor is
  an accurate, meaningful property of the real inverse problem (not an artifact to debug away).
- GNL's phase term must be built against k_actual when GIRF is enabled and k_nominal when it
  isn't (`gnl_term_nominal`/`gnl_term_actual`, selected by `config.girf` in
  `_operator_from_context`) - matches the existing ablation figure's own convention exactly.

None of this is committed as "final" scientific results - only the code/pipeline itself was
committed. Ask before committing any further changes (standing policy, unchanged).
