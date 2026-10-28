# Status: CG reconstruction stability/fitting investigation — pick up here

Scope: specifically the "why does CG diverge for some PhysicsConfigs" investigation within the
broader physics-informed CG reconstruction effort (see `CG_RECON_STATUS.md` for that project's
overall state - architecture, tests, the experiment script). This file is narrower: just the
numerical-stability/regularization/conditioning sub-problem, which is currently BLOCKING any
trustworthy full-resolution run.

## The symptom (confirmed, reproducible)

At matrix=128, running the real-physics CG experiment (`scripts/ismrm/figure_physics_informed_cg.py`)
across all 11 reconstructions (R0-R5 cumulative, 5 leave-one-out), configurations that include B0
but NOT GIRF diverge under CG with the handover's suggested light regularization (`lam=1e-6`):

- `R1_b0` (B0 only): NRMSE 3.00 (worse than a blank image)
- `R2_b0_b1` (B0+B1+, no GIRF): NRMSE 2.17
- `leaveout_girf` (full minus GIRF, i.e. B0+B1++concomitant+GNL, no GIRF): NRMSE 1.92

Every configuration that includes BOTH B0 and GIRF together is well-behaved (NRMSE 0.15-0.23).
`y_ref` (the one shared reference dataset) is generated with the FULL-physics operator, which
includes GIRF - so these divergent configs are reconstructing data that contains a real trajectory
discrepancy their own model doesn't know about.

The divergence signature itself: NRMSE decreases to a minimum within ~20-40 iterations, then
INCREASES again for the rest of the 100-iteration budget, with the image's max amplitude growing
far beyond the true object's own scale (confirmed via saved intermediate snapshots - see
`outputs/ismrm_cg_recon_matrix128/intermediate/R1_b0/iter_*.npz` from the run that first surfaced
this). The native CG residual is NON-monotonic over the same stretch. This is the textbook
signature of CG exploiting a near-null-space direction of `A^H A`: a direction that costs almost
nothing in the (weakly regularized) data-fit term but is essentially unconstrained by the data,
so CG happily grows it without bound even as the nominal residual keeps falling.

## Hypotheses tested and RULED OUT (each with a concrete negative result, not just reasoning)

1. **GIRF-as-second-trajectory architecture.** Originally, `config.girf=True` switched which
   trajectory (`k_nominal` vs `k_actual`) the NUFFT itself gridded onto - a much bigger structural
   change between GIRF-on/off than a bounded phase perturbation. Fixed: GIRF trajectory error is
   now represented as an exact phase term (`_GIRFTrajectoryErrorPhase`,
   `phi = -2*pi*dk(t).r` where `dk = k_actual - k_nominal`; `augmentrum/reconstruction/
   linear_operator.py`), derived from the exact identity
   `exp(-i*2*pi*k_actual.r) = exp(-i*2*pi*k_nominal.r) * exp(-i*2*pi*dk.r)`. The NUFFT now ALWAYS
   grids on `k_nominal`, matching the user's explicit instruction. All 19 reconstruction tests
   still pass after this change. **Result: did NOT fix the divergence.** `R1_b0` under the new
   architecture still diverges (NRMSE 2.32, same non-monotonic residual pattern) - ruling out
   "it's just an artifact of switching trajectories" as the explanation.
2. **Poor CG initialization (`x0=0`).** Tested starting CG from a density-compensated adjoint
   reconstruction instead (the same kind the ablation figure already computes) -
   `x0 = GriddingNUFFT.adjoint(y_ref * density_weights, k_nominal)`, which is itself a reasonable
   starting image (NRMSE 0.79, a believable blurred reconstruction). **Result: did NOT fix the
   divergence.** CG still drifted from this good starting point to NRMSE 2.27 over 100 iterations
   - ruling out "just needs a better starting guess." This is informative: it shows CG is actively
   driven away from a good solution by the iteration dynamics themselves, not failing to find a
   good point it starts near - consistent with a genuine null-space/conditioning problem that
   initialization alone cannot fix (a preconditioner, which reshapes every iteration's search
   direction rather than just the starting point, is a different, not-yet-tested hypothesis).

## Live hypothesis, not yet tested: single-coil conditioning

User's sharp pushback (verbatim reasoning, worth preserving): real clinical reconstruction
routinely omits GIRF/GNL/B1+ modeling and converges stably every day - it does NOT blow up to
2-3x the true image amplitude the way these configs do. That argues the instability is not an
inherent, expected property of "incomplete physics modeling" in general, but something more
specific to this experiment's setup.

The leading candidate: **every real clinical reconstruction that "just converges" is multi-coil.**
Coil sensitivity diversity provides substantial implicit conditioning that a single-coil
reconstruction fundamentally lacks - it is not just "missing physics," it is missing physics PLUS
none of the constraint multi-coil parallel imaging normally provides. This experiment is currently
single-coil throughout (`LinearMRIOperator.coil_map` defaults to an all-ones (N,) array; the
handover explicitly allows single-coil for a first pass, but may not have anticipated it being the
reason for a genuine numerical instability rather than just reduced accuracy).

**Augmentrum already has what's needed to test this** - confirmed by reading the code, not
assumed:
- `augmentrum/sampling/coil_sampling.py`'s `Birdcage` class: a synthetic coil array built from
  pure grid geometry (no data/download needed) - `Birdcage(n_coils=N).maps((nx,ny,nz))` returns
  RSS-normalized `(nx,ny,nz,N)` complex sensitivity maps directly. This is the standard synthetic
  multi-coil model used throughout the reconstruction literature (comparable to what sigpy/BART/
  ESPIRiT papers use for simulation).
- `T2starMove` in the same file: REAL measured maps from an actual 32-channel head array (open
  CC-BY-4.0 dataset, ~5.4GB one-time download, cached a few MB afterward).
- `Supplied` for maps already in hand.

**What does NOT exist yet**: multi-coil support in `LinearMRIOperator` itself. The class is
"multi-coil-SHAPED" per its own docstring (coil_map is a per-voxel array like b1plus_map, not
hardcoded away) but currently only accepts ONE coil map and has no per-coil loop in forward/
adjoint/normal. Per the handover's own section 6 guidance (already anticipated, not a new idea):
```
forward:  C_c(r) * x(r)   [per coil c, same NUFFT/phase terms, before the NUFFT]
adjoint:  sum_c conj(C_c(r)) * x_c(r)   [sum over coils' own adjoint reconstructions]
```
This needs: (a) `LinearMRIOperator` accepting a stack of coil maps instead of one, (b) forward
looping over coils (each producing its own k-space data vector, stacked into a
"(n_coils, T)"-shaped `d`), (c) adjoint summing `conj(C_c) * adjoint(d_c)` over coils, (d) the
segmented phase terms/NUFFT reused IDENTICALLY per coil (coil sensitivity is a plain multiplicative
factor independent of the other physics), (e) updated adjoint-correctness tests for the multi-coil
case specifically, since the current 9 adjoint tests are all single-coil.

## Suggested next steps, in order

1. Extend `LinearMRIOperator` to real multi-coil (per the sketch above), using `Birdcage(n_coils=8)`
   (or similar) as the synthetic array for this experiment - no new data/download needed.
2. Re-run the SAME `R1_b0`-style worst-case test (B0-only, no GIRF, lam=1e-6, 100 iterations) with
   multi-coil data and see whether the divergence disappears. This directly tests the live
   hypothesis before committing to any architecture change for the full run.
3. If multi-coil resolves it: re-run the small-phantom regression tests and the full 11-
   reconstruction matrix=128 experiment with multi-coil data, and treat single-coil as out of
   scope for the final result (document why).
4. If multi-coil does NOT resolve it: the remaining untested hypothesis is an actual preconditioner
   (not just `x0`) inside `cg_solve` - e.g. a DCF-informed operator applied every iteration, not
   just once at the start. Test that next, in the same isolated-worst-case-first manner.
5. Only after stability is resolved: revisit the earlier blanket-Tikhonov-lambda question (is the
   SAME lambda appropriate across configs of very different conditioning, once the underlying
   instability itself is fixed rather than papered over) - this was the original methodological
   objection that triggered this whole investigation, and remains unresolved independent of
   whatever fixes the raw instability.

## Do not re-litigate without new evidence

- GIRF-as-trajectory-switch vs phase-term: tested, ruled out as sole cause. The phase-term
  architecture is still correct/worth keeping (matches physical reasoning, requested explicitly),
  just not sufficient on its own.
- Better `x0` alone: tested, ruled out. A preconditioner is a DIFFERENT, not-yet-tested idea -
  don't conflate the two.
- Segment count (`n_segments=8`) was flagged as unjustified by the user but not yet investigated
  via the already-built `SegmentedPhaseTerm.max_abs_representation_error()` diagnostic - still
  open, likely NOT the cause of divergence (it's a representation-accuracy question, not a
  conditioning one) but still owed a real answer before any segment count is treated as final.

None of this is committed. Ask before committing (standing policy, unchanged).
