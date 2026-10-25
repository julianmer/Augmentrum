# Status: spatial-physics ablation figure — pick up here

Last worked: 2026-10-24 (evening), branch `spatial_maps`. Written mid-task because context was
about to compact — this captures a large batch of NEW, not-yet-implemented requests from the
user's most recent message, plus everything already done. Read this fully before continuing;
do not re-ask the user things already answered here.

## Where things stand right now (all done, working, in the script on disk)

`scripts/spatial_physics_ablation.py`'s `RepoAdapter` is fully implemented (not stubs) and
produces a real, physically-validated figure. Current outputs on disk:
- `outputs/spatial_physics_ablation/spatial_physics_ablation.png` (3T, i.e. "HF")
- `outputs/spatial_physics_ablation_55mT/spatial_physics_ablation.png` (0.055T, i.e. "ULF")

Both were just regenerated with `--matrix 80 --rotate90-ccw 1` and the fixes below. **These two
separate-figure outputs are about to be superseded** by a single combined ULF/HF/UHF figure per
the new requests below — don't treat them as the final target, they're the last-known-good
checkpoint if something breaks.

Implemented and validated this session (earlier checkpoint, still true):
- Real HCP B0 (subject 105923, GRE field map) and B1+ (subject 100408, actual-flip-angle map)
  data, with correct unit conversions (TE1=4.92ms/ΔTE=2.46ms from two independent official HCP
  sources; Siemens 12-bit phase convention; nominal flip 50° from the HCP Appendix I protocol
  card — all cited in the plan file, see below).
- `StaticOffResonancePhase`/`ConcomitantFieldPhase`/`GradientNonlinearityPhase` composed as real
  `PhaseTerm`s in ONE `ForwardOperator.forward()` call per condition (genuinely single-pass
  "combined", not five reconstructions composed afterward) — the core architecture bet, confirmed
  working (`combined` NRMSE stays just above the dominant single effect, never runaway).
  `_MultiShotConcomitantPhase` (in the script) handles per-shot phase-accumulation resets that
  the library's own single-shot `ConcomitantFieldPhase` dataclass doesn't.
- Position/flatten-order convention empirically verified (point-source round-trip test) and
  documented in `RepoAdapter`'s own class docstring: 2-D array `arr.T.reshape(-1)` matches
  `make_grid_2d`'s (Y-outer, X-inner) order; `GriddingNUFFT.adjoint` output needs no further
  transform; anything evaluated directly from `positions` (phase-map extras) needs
  `.reshape(ny,nx).T`.
- `--anatomy-source {shepp_logan,brainweb}` flag (default shepp_logan; brainweb's server is still
  unreachable, confirmed via both direct fetch and the `brainweb` pip package, which points at
  the same dead host).
- `--rotate90-ccw <n>` — empirically verified rotation direction (numpy's `rot90(k=1)` reads as
  *clockwise* under this script's `origin="lower"`, so CCW needs `k=-1` per step), applied
  uniformly to anatomy+mask+b0+b1 together.
- `--field-strength-t` properly rescales the real B0 map physically: `Hz_target = Hz_measured_3T *
  (field_strength_t / 3.0)` (ppm-level residual is field-independent; Hz scales linearly with B0)
  — NOT just relabeling the same Hz numbers.
- B1+ now plotted with the same diverging `coolwarm` convention as B0 (centered at 1.0, not 0).
- Bottom "difference" row: panels whose own signal is much smaller than the shared color scale
  now get multiplied by a round factor (10, 100, ...) and labeled (`×10` etc.) rather than just
  looking black — this is what revealed GIRF and GNL's real-but-small differences in the current
  figures.
- Shot count was hardcoded at 8 regardless of matrix (badly undersampled at matrix=80) — fixed to
  scale as `n_shots = max(16, round(matrix*0.4))`, empirically checked against the true phantom
  (NRMSE plateaus ~32 shots at matrix=80; more doesn't help, consistent with a single
  DCF-weighted adjoint's own irreducible error floor, not an undersampling artifact past that
  point). **This heuristic is about to be overridden by an explicit request below (100
  interleaves, 455 samples/shot) — use the explicit numbers, not this heuristic, once that's
  wired in.**
- Quantified (not guessed) why GNL shows no visible geometric distortion: the dummy
  `gradunwarp` test coil's displacement at the FOV edge is **0.46mm = 0.17 voxels** at this
  resolution — genuinely sub-pixel. Confirmed via
  `augmentrum.physics.gradient_nonlinearity.gradient_nonlinearity_displacement_m`.

Sanity-checked and passed: determinism across repeated runs; zero-coefficient GNL reproduces
`ideal` bit-for-bit (`max_abs_diff=0.0`, `array_equal=True`); concomitant is *analytically* zero
for the on-axis, no-concurrent-Gz 2D geometry (not a bug — see physics discussion below).

Full architecture/provenance detail (exact HCP file paths, TE/flip-angle citations, Bacon GIRF
data location, the full per-condition phase-term table) is at
`/home/johnlamaster/.claude/plans/parallel-cooking-beacon.md` — still accurate, not superseded.

## NEW requests from the user's latest message — not yet implemented, in the order given

1. **Build the registration pipeline** (user said yes, explicitly, to this — go ahead): register
   each real map's own source T1w onto the phantom via ANTs (now installed —
   `import ants` works in the python3.13 env), get the warp field, apply that warp to the
   corresponding B0/B1+ maps before resampling onto the figure's grid. Two separate subjects
   (105923 for B0, 100408 for B1+) means **two separate registrations** (each subject's own T1w
   → phantom), not one shared warp. Need to locate/load each subject's own T1w:
   - 100408's T1w: `/run/media/johnlamaster/data/datasets/hcp/100408_StructuralRecommended/100408/T1w/T1w_acpc_dc_restore_brain.nii.gz` (already identified earlier this session).
   - 105923's T1w: `.../MRSI_Challenge_Data/dev/connectome/Data/105923_3T/scans/103_T1w/NIFTI/105923_3T_T1w_MPR1.nii.gz` (raw, not bias-corrected/brain-extracted — identified earlier, not yet used).

2. **Treat both figures as sufficiently off-isocenter that the ULF (55mT) image shows concomitant
   effects.** This directly follows from the physics explained to the user this session: the
   concomitant Maxwell term's `z²`-type component is the one that activates with a nonzero slice
   offset even for a pure in-plane 2D readout (no concurrent Gz needed for THIS term — only the
   `x²+y²`/`xz`/`yz` terms need concurrent Gz, which still won't be present). Per that same
   analysis (documented in this session's conversation, see transcript if unclear): with z≠0 but
   Gz=0, the concomitant phase reduces to `(Gx²+Gy²)·z²/(2B0)` — **nonzero but spatially UNIFORM
   across the 2D slice** (same value at every in-plane position, since z is one constant offset
   for the whole slice) — it will show up as a uniform brightness/phase shift that scales
   strongly with 1/B0 (hence visible at 55mT, not at 3T/7T), not as within-slice geometric
   distortion. Need to: (a) give `GIRFArtifacts`/this script's own `positions` construction a
   nonzero `z` (currently hardcoded `z=0.0` in `gs.make_grid_2d(nx,ny,fov_x,fov_y,z=0.0)` inside
   `RepoAdapter.simulate_all` — change to a configurable offset, e.g. a new CLI flag), (b) decide
   a physically-reasonable offset (a few cm is typical for an off-center axial slice), (c) be
   ready to explain to the user when this ships that the result will be a uniform shift, not
   spatial distortion, per the physics above — do not silently let them think it'll look like a
   warp.

3. **DONE** — `build_shepp_logan_3d_ellipsoids(size, cache_path, force)` added to the script
   (right after `load_shepp_logan_anatomy`). Each 2-D ellipse's own `(a,b)` in-plane semi-axes
   from `_SHEPP_LOGAN_GEOMETRY` are reused directly; the third (through-slice) semi-axis, which a
   2-D phantom has no measurement for at all, is the geometric mean `c = sqrt(a*b)` - the "true
   sphere exactly when the in-plane ellipse is already a circle" reading of "deformed sphere"
   (documented in the function's own docstring, including why this was chosen over Kak & Slaney's
   own hand-tuned 3-D table, which has no 2-D-measurement basis to derive from). Cached to
   `scripts/data/shepp_logan_3d_ellipsoids.npz` (`volume` + `size` keys; `force=True` rebuilds).
   Verified visually (axial slices through the stack show the expected tapering skull/ventricles/
   tumours). **Not yet wired into the acquisition pipeline** — `get_spiral`/`simulate_all` are
   still hardcoded 2-D (`z=0`). A real 3-D acquisition path (needed for genuinely
   spatially-structured concomitant/GNL distortion, not just a uniform off-isocenter shift) is
   still a separate, bigger follow-up — this phantom is the prerequisite asset for it, not the
   whole task.

4. **Perturb gradunwarp's coefficients for a larger GNL displacement — TESTED, decided.**
   Quantified both candidate knobs directly (`gradient_nonlinearity_displacement_m`, FOV=0.22m,
   N=80, so 1 voxel = 2.75mm) before picking one:
   - **Shrinking `R0_m`** (the expansion's reference radius) is badly unstable: 0.5x → 3.85
     voxels (reasonable), but 0.25x → 84 voxels, 0.1x → 3534 voxels - it diverges explosively once
     positions move far outside the (now-shrunk) nominal radius in normalized units, i.e. exactly
     the "extrapolating a polynomial expansion outside its design region" failure mode. **Do not
     use R0_m shrinkage as the primary knob** - too sensitive to be controllable.
   - **Flat multiplicative coefficient scaling** (`coeffs._replace(**{f: getattr(coeffs,f)*k for
     f in coeffs._fields if f != 'R0_m'})`) is linear and well-behaved: k=10 → 1.67 voxels, k=50 →
     8.35 voxels, k=100 → 16.7 voxels. **Use this as the primary, controllable knob.** A factor in
     the ~20-50x range gives a clearly visible, non-absurd displacement (a few to ~8 voxels at
     this resolution).
   - Decision: use flat coefficient scaling (not R0_m) as the main lever, clearly labeled in the
     figure/metadata as an illustrative amplification (not a real vendor coil) - same disclosure
     norm as the existing runtime warning about the dummy coil already being fabricated. A mild
     R0_m reduction (e.g. 0.7-0.9x, nowhere near the instability threshold) could still be layered
     in as a secondary, more physically-motivated nudge ("more compact ULF bore") once the
     primary coefficient-scale knob is wired in and shown to look right - not required.
   - Still open: what multiplier to actually use per field strength for the ULF/HF/UHF stack (the
     real physical motivation - ULF/portable systems genuinely do have more compact, less-linear
     gradient coils - supports making the multiplier LARGER at ULF and smaller/1x at UHF, i.e.
     inversely related to field strength, mirroring the concomitant-field story rather than
     fighting it).

5. **Stack ULF, HF, and UHF examples in one figure** — currently two separate PNGs (3T and
   0.055T). Needs: (a) decide UHF's field strength (7T is the obvious standard choice, not yet
   confirmed with the user), (b) restructure `plot_figure`/`simulate_all` to run 3 field-strength
   conditions and lay them out as 3 stacked blocks (likely 3x the current 3-row layout, or a
   row-per-field-strength variant of the top row only with shared reconstruction/difference rows
   per field strength) rather than 3 entirely separate 3-row figures — the user said "stacking...
   in one figure," implying one combined output, not three side-by-side files. Needs a concrete
   layout decision before implementing (how much vertical space 3x the current 3-row block would
   need, whether to keep all 7 columns at each field strength or trim to the most illustrative
   subset).

6. **Spiral trajectory**: use **100 interleaves, 455 samples per shot** explicitly (overriding
   the `n_shots = max(16, round(matrix*0.4))` heuristic above) — the user gave explicit numbers,
   use them directly: `n_shots=100, samples_per_shot=455` in `RepoAdapter.get_spiral`'s
   `build_steam(...)` call. Given this is more than double the already-slower 32-shot run (which
   took ~82s per figure at matrix=80), expect a real runtime increase — time it before assuming
   it's fine for 3 stacked field strengths in one run.

7. **ULF real data**: user will try to find real ULF B0/B1 data themselves; use the current
   (HCP-derived + physically-rescaled) data for now. Not blocked on this.

8. **DONE** — GNL excitation-side modeling implemented as proposed and verified. `get_spiral`
   now returns `voi_m` (the STEAM sequence's own `voi_mm`, the box its 3 slice-selective pulses
   actually prescribe). In `simulate_all`: `GradientNonlinearityDisplacement.displace(positions)`
   (real, already-tested, same `r_eff = r - Δr(r)` sign convention as its own docstring) computes
   each voxel's effective position; `nominal_voi_mask = |r| <= voi_half` and
   `warped_voi_mask = |r_eff| <= voi_half` are both box indicators, applied as an extra
   multiplicative factor on the phantom. **Only `gnl`/`combined` use the warped mask** — every
   other condition (`ideal`/`b0`/`b1`/`concomitant`/`girf`) uses the nominal (undistorted) mask,
   so the ablation stays a fair comparison (same VOI baseline everywhere, only GNL's conditions
   see it warped). Verified: at `--gnl-coefficient-scale 30`, nominal VOI keeps 56.25% of the
   full grid (90.6% of the phantom's own actual tissue — the box clips only thin slivers at the
   very top/bottom of the skull outside the 0.75×FOV VOI, confirmed visually, not a bug); the
   warped VOI differs measurably (52.08% vs 56.25% of the grid) — a real, nonzero excitation-side
   effect. `nrmse_gnl` at that scale jumped to 58.6% (matrix=48, 3T) — now comparable in magnitude
   to B0, not negligible. New CLI flag `--gnl-coefficient-scale` (default 1.0 = unmodified dummy
   coil); new metadata keys `gnl_coefficient_scale`, `gnl_voi_m`,
   `gnl_excited_voxel_fraction_{nominal,warped}`, `gnl_excitation_model`. **Not yet regenerated
   into the two "real" output figures** (`outputs/spatial_physics_ablation{,_55mT}/`) — those
   still reflect the pre-excitation-modeling, scale=1.0 GNL. Do that (with whatever
   `--gnl-coefficient-scale` is eventually decided per field strength, per item 4 above) before
   showing the user a non-test run.

   (superseded — item 8 above is the implemented version of this question; kept only as history.)

## Session 3 additions (2026-10-24, later) — spiral count, GPU support

- **Spiral**: `get_spiral` now hardcodes `n_shots=100, samples_per_shot=455` (user's explicit
  numbers), replacing the old `max(16, round(matrix*0.4))` heuristic. At matrix=80 this gives a
  visibly much cleaner reconstruction (grain from the old 32-shot heuristic is essentially gone,
  replaced by the expected spiral-streak texture of a genuine well-sampled spiral) - confirmed
  visually, not just by NRMSE.
- **GPU support added and verified working**, per explicit user request ("don't waste time on
  CPU"). `--device {auto,cuda,cpu}` (default auto). Two real bugs found and fixed along the way,
  not just "made it run":
  1. GIRF-Sim's `ForwardOperator.forward`/`total_phase`/`SpectralPhantom.single_component`/
     `CoilSensitivity.get` hardcode several internal tensors (`t`, `phi`, `d`, `t2star_q_s`,
     `freq_q_hz`, the default coil map) with no `device=` - passing CUDA inputs into the
     unpatched classes crashes immediately on a device mismatch, it doesn't just run slow. Fixed
     via `_patch_girf_sim_for_device(gmx)` - an **in-process monkey-patch only, the GIRF-Sim
     checkout on disk is never touched** - applied once in `RepoAdapter.__init__` when
     `device != 'cpu'`.
  2. GIRF-Sim's own `waveform()`/`total_phase()` are NOT chunked over T at all (unlike its own
     NUDFT encode loop, which already had a `chunk_size` parameter) - they materialize the full
     `(N, T)` tensor up front. At this script's T~45500 samples that's multiple GB per complex
     intermediate tensor, enough to OOM a 12GB card before the (correctly chunked) NUDFT step
     even runs. Fixed by extending the same chunking to cover `waveform()`/the phase-term-driven
     `signal` construction too (phase terms themselves are evaluated once per condition, outside
     the chunk loop, since they're real-valued and much cheaper - only the complex
     waveform/signal intermediates are chunked). Also fixed `b0_hz_flat` being unnecessarily
     float64 (promotes `StaticOffResonancePhase`'s whole intermediate to float64, doubling that
     allocation) - now float32.
  3. Added explicit `del phantom, op, result` + `gc.collect()` + `torch.cuda.empty_cache()`
     between each of the 7 conditions in `encode_reconstruct` - without it, memory from one
     condition can still be attributed to the process when the next one allocates.
  - **Verified, not just "it didn't crash"**: matrix=80, 3T, 100 shots - GPU run (79s) produced
    numerically consistent NRMSE values and a visually identical figure to the CPU run (272s) at
    the same settings. ~3.4x speedup confirmed, not assumed.
  - `augmentrum/physics/gnl_field.py`'s `GradientNonlinearityPhase.phase()` also got a small,
    genuine device-correctness fix (`.to(positions.device)` on its return value) - this is a real
    library bug fix (it would have crashed given any CUDA `positions` even outside this script),
    not a monkey-patch, and is a tracked, uncommitted library change like the DCF fix earlier.
  - GPU memory is a real constraint at this problem size (6400 positions x 45500 samples): a
    12GB TITAN X was close to its limit even after all the above fixes. If a future field-strength
    stack (item 5, 3 conditions x 7 sub-conditions) OOMs even after chunking, reducing `chunk_size`
    in `forward()` (currently the default 4096) or the matrix size are the first things to try.

## Session 4 additions (2026-10-25) — real B0 multi-shot bug, GNL scale decided, off-isocenter done

- **Found and fixed a real bug**, independent of anything from session 3: `StaticOffResonancePhase`
  was being fed `t_full`, the GLOBAL flattened time axis across all 100 concatenated shots
  (reaching up to `100*455*4e-5 = 1.82s`), instead of each shot's own local time since its RF
  excitation. Since every shot physically resets transverse phase at its own `t=0`, B0 dephasing
  must reset with it too — the exact same shot-boundary problem `_MultiShotConcomitantPhase`
  already existed to fix for the Maxwell term, but `StaticOffResonancePhase` was being used
  directly, unwrapped. Symptom: `b0`/`combined` reconstructed as near-total noise (NRMSE ~78%)
  — confirmed this was NOT caused by the new slice-offset/GNL-scale knobs by reproducing it at
  matrix=48 with both at their defaults. Fixed with a new `_MultiShotStaticOffResonancePhase`
  class (right after `_MultiShotConcomitantPhase` in the script) that precomputes a per-shot-reset
  `t_local` vector once and reuses the same `2*pi*f*t` formula against it; wired into both the
  `b0` and `combined` conditions in `simulate_all` (one shared instance, `b0_term`). After the fix,
  `nrmse_b0` at matrix=80/3T dropped to ~49% — still substantial (this spiral's readout is ~18ms
  long per shot, and spirals are known to be very sensitive to B0 over long readouts), but now
  shows as localized blur concentrated where the real HCP B0 map is actually largest, not
  spatially-uniform noise. Confirmed visually, not just by NRMSE.
- **GNL-vs-field-strength question resolved with the user**: there is NO established physical law
  tying gradient nonlinearity to B0 (unlike concomitant fields, which genuinely go as 1/B0) — GNL
  is a property of the gradient coil's winding geometry, built by a physically separate system
  from the main field magnet. The user was asked directly and confirmed: keep the original
  "larger at ULF" choice, motivated narratively as portable/compact low-field systems using
  simpler, less-linear gradient coils (not a physical law — flagged as such in both the script's
  own comments and this doc).
- **Final `--gnl-coefficient-scale` values used**: 15 at 3T (~2.5 voxel displacement, modest but
  visible), 60 at 55mT (~10 voxel displacement, clearly visible geometric warping — confirmed
  visually, the 55mT GNL/combined panels now show real pinching/warping of the anatomy, not just a
  phase map).
- **Off-isocenter (`--slice-offset-mm 30`) applied to both figures** — confirms the physics
  predicted in session 3: at 3T, `nrmse_concomitant` stays ~0.0002% (negligible, as expected,
  concomitant ∝ 1/B0); at 55mT it becomes a real, nonzero 1.5% (visible at ×100 in the difference
  panel as a faint ring-shaped blur near the image edge). This is NOT geometric distortion — per
  the physics discussed with the user, a nonzero z with no concurrent through-slice gradient only
  activates a spatially UNIFORM (same at every in-plane position) but time-varying phase during
  the readout; for a non-Cartesian (spiral) reconstruction using the nominal trajectory, an
  unmodeled uniform time-varying phase behaves like a zero-order/global off-resonance term and
  shows up as mild blurring concentrated near high-spatial-frequency edges (textbook non-Cartesian
  behavior), not as a spatial warp. Keep stating this honestly if asked — it is working as
  physically expected, not "not distorting enough."
- **Both output figures regenerated and verified good**, matrix=80, rotate90-ccw=1, device=cuda,
  slice-offset-mm=30, ~78s each:
  - `outputs/spatial_physics_ablation/` (3T): gnl-coefficient-scale=15. NRMSEs: b0 48.6%, b1 6.3%,
    concomitant 0.0002%, girf 11.1%, gnl 51.3%, combined 59.6%.
  - `outputs/spatial_physics_ablation_55mT/` (55mT): gnl-coefficient-scale=60. NRMSEs: b0 1.2%, b1
    6.3%, concomitant 1.5%, girf 11.1%, gnl 80.7%, combined 79.6%.
  - Item 2 (off-isocenter so ULF shows concomitant) is now **DONE**, not just coded.
- **Follow-up fixes after the user inspected the regenerated figures, same session**:
  - User asked why the GNL/combined reconstructions looked "cropped" top/bottom. Real cause: the
    STEAM VOI (`voi_mm` in `get_spiral`) was `0.75*FOV` in-plane, SMALLER than this phantom's own
    skull outline (which reaches ~0.92 of the FOV half-width) - so even the non-GNL conditions
    were already clipping real anatomy at the top/bottom before any GNL displacement, and GNL's
    warped-VOI exclusion made it worse. STEAM was only ever used here as a vehicle to reach
    `mrs_seq`'s already-validated spiral trajectory generator, not to model anything
    STEAM-localization-specific - so the box was an unnecessary, unexamined side effect. Widened
    to `0.95*FOV` - confirmed this removes essentially all baseline (non-GNL) clipping and
    noticeably increased the B1+ NRMSE (6.3% -> 38.8%) by exposing more of the periphery, where
    B1+ inhomogeneity is usually most visible - a real improvement, not a regression.
  - User also asked why the concomitant panel was a single solid color with no scale. This part
    WAS correct physics (a uniform off-isocenter slice offset with no concurrent through-slice
    gradient gives a position-INDEPENDENT phase, hence a flat color), but the panel had no printed
    number, so a reader had no way to tell a flat 0.001 rad from a flat 10 rad. Fixed:
    `plot_top_row`'s concomitant panel now prints the actual value (`"RMS phase: 0.192 rad
    (uniform)"` at 55mT, `"...0.00352 rad (uniform)"` at 3T - correctly ~55x larger at low field,
    consistent with concomitant's 1/B0 scaling); the GNL panel now prints its actual 0-to-max
    range too (`"RMS phase: 0-4.93 rad"` at 3T, `"0-19.7 rad"` at 55mT).
  - **Both figures regenerated again with the VOI widening + value labels** (matrix=80,
    rotate90-ccw=1, device=cuda, slice-offset-mm=30, same gnl-coefficient-scale as before: 15 at
    3T, 60 at 55mT): 3T NRMSEs now b0 53.4%, b1 38.8%, concomitant 0.00022%, girf 12.0%, gnl 72.3%,
    combined 69.7%. 55mT NRMSEs now b0 1.5%, b1 38.8%, concomitant 1.4%, girf 12.0%, gnl 94.6%,
    combined 90.4%. GNL at 55mT still shows a small residual notch at the very top of the skull
    (GNL displacement at scale=60 still pushes a thin sliver past even the widened 0.95 VOI) -
    much smaller than before, and arguably fine to leave (STEAM VOI truncation under extreme GNL
    is itself a real phenomenon) - flag to the user only if it comes up again, don't preemptively
    "fix" further without being asked.

## Session 5 additions (2026-10-25, later) — registration pipeline, 200 shots, OOM fix #2

- **Registration pipeline (item 1) is now DONE.** Each subject's own T1w is rigidly registered
  (ANTs, `type_of_transform="Rigid"` - 6 DOF rotation+translation, explicitly NOT deformable/SyN
  per the user's own instruction: the phantom's hand-specified ellipse geometry has no real brain
  texture for a deformable algorithm to anatomically lock onto, so only a rigid alignment is
  well-posed) onto a NEW NIfTI export of the existing 3-D ellipsoid phantom
  (`_phantom_3d_nifti`, `scripts/data/shepp_logan_3d_phantom.nii.gz`, isotropic `fov_mm/size`
  voxels centered at the origin). The resulting transform is then applied (`ants.apply_transforms`)
  to that SAME subject's own field map (105923's B0 phase+magnitude; 100408's rAFI B1+ map),
  assuming the field map is already in the same native space as that subject's own T1w (true by
  construction for `rAFI.nii.gz` via HCP's own "r" = resampled-to-T1w convention; assumed, not
  independently re-verified, for 105923's field map). Key property exploited: the 3-D phantom's
  own z=0 axial slice is EXACTLY the 2-D Shepp-Logan phantom already used everywhere else (every
  ellipsoid's "c=sqrt(a*b)" choice only changes the shape away from z=0), and the existing
  downstream loader already picks the volume's middle slice - so NO changes were needed to
  `load_or_prepare_inputs`'s own cropping/resizing code, only to `_derive_hcp_b0_map`/
  `_derive_hcp_b1_map` (which now register-then-convert instead of just converting the raw
  native-space file). New functions: `_phantom_3d_nifti`, `_rigid_register_t1w_to_phantom` (caches
  the expensive part - a full 3-D T1w registration - under `scripts/data/registration_cache/`,
  keyed by subject label, NOT by `--outdir`, so repeated runs/field-strengths don't redo it),
  `_apply_rigid_transform`. New constants `_HCP_B0_T1W`/`_HCP_B1_T1W` (each subject's own T1w
  path, already identified in a prior session). Hit and fixed one real issue along the way: the
  magnitude field map file is 2 echoes (4-D), and `ants.apply_transforms` refuses a non-3-D volume
  without an explicit "imagetype" flag - fixed by extracting the first echo to its own 3-D NIfTI
  before registering, not by fighting that flag. Verified visually via `--prepare-only`: both maps
  now fill the correct extent of the phantom with plausible real structure (a B0 hotspot, a
  smooth bilateral B1+ coil pattern) rather than a crudely center-cropped guess.
- **Spiral shots raised from 100 to 200** per explicit request (`n_shots=200` in `get_spiral`,
  `samples_per_shot=455` unchanged, so `T` doubled to ~91000). Runtime roughly doubled as expected
  (~155s/figure at matrix=80 on GPU vs ~78s at 100 shots).
- **Removed the "Common anatomy" text label** from the Ideal panel per explicit request
  (`plot_top_row`, the `axs[0].text(...)` call deleted).
- **Found and fixed a second real OOM bug** triggered by the shot-count increase: the GPU-device
  patch's `total_phase()` materializes one full `(N, T)` REAL tensor via each phase term, and at
  least one term (the library's own `GradientNonlinearityPhase`) explicitly REFUSES a chunked `t`
  (it validates `t.shape[0] == k_traj.shape[1]`) - so that full-length tensor can never be chunked
  away, only moved. Fixed by having `total_phase` compute and return it on CPU (ample system RAM)
  instead of GPU VRAM, with only each chunk's own slice ever uploaded to the GPU inside `forward`'s
  existing per-chunk loop - this decouples phase-term memory from GPU VRAM entirely regardless of
  how large `T` gets in the future. Also moved the standalone RMS-phase-map extras computation
  (`conc_phase`/`gnl_phase`, used only for the top-row plots) onto CPU for the same reason, since
  they independently allocate another full `(N, T)` tensor and have no performance need for a GPU.
- **Both figures regenerated one more time** with all of the above combined (200 shots, no
  anatomy label, registered B0/B1+ maps, same gnl-coefficient-scale/slice-offset as before: 15/3T,
  60/55mT, 30mm offset both). This is the current authoritative state of
  `outputs/spatial_physics_ablation{,_55mT}/` - check `forward_metadata.json`/`metrics.json` in
  each outdir for this run's exact NRMSE values rather than trusting numbers quoted in earlier
  entries in this doc, which predate the registration pipeline and 200-shot change.

## Session 6 (in progress) — real streaking-artifact investigation, PAUSED mid-investigation

User noticed the "ideal" (zero-perturbation) reconstruction looks badly streaked despite heavy
oversampling. This turned into a real, multi-stage investigation - NOT resolved yet. Status:

**Confirmed, fixed, kept:**
1. `get_spiral()` switched from STEAM/`build_steam` (MRSI-tiling misuse, see Session 5) to a real
   spoiled spiral GRE (`mrs_seq.imaging.build_spiral_gre`, `acquisition_mode='mri2d'`) - a genuine
   GIRF-Sim feature the user added this session specifically for MRI (non-spectral) use. No more
   VOI box (`voi_mm=None` for GRE) - the GNL excitation-side masking mechanism from Session 4 was
   REMOVED (no physical analog without a VOI); GNL is now readout-encoding-phase-term only, same
   as b0/concomitant/girf. TR raised to 0.05s (15ms default was infeasible for an 18ms+ readout).
2. **Real trajectory bug found and fixed**: `GIRFModule.k_nominal` (from `seq_girf.py`'s
   `load_all_shots_batched`) unconditionally re-references every shot's k-space to k=0 at its own
   ADC start (`k = k - k[:,:,0:1]`, no disable flag exists). This manufactured a FAKE 200-way exact
   coincidence at k=(0,0) that has no physical counterpart - every shot actually starts at the SAME
   nonzero radius (0.40048 cycles/m, std 1e-17) at its own golden-angle azimuth. Fixed in
   `get_spiral()` by building `k_nominal` from `info['readouts'][i].k_achieved` (the true
   absolute/post-realizability trajectory from `build_spiral_gre`'s own return value) instead of
   `self._mod_nominal.k_nominal`. Verified safe: `dk_error` (measured-GIRF term) is a
   cumsum-vs-cumsum difference against the same internal zero-reference, so it's invariant to this
   swap (confirmed from `bacon_girf.py` source, not assumed); `gradients_per_shot` is also
   unaffected (gradients are rates, never touched by the k re-reference). **This fix is real and
   should stay, but it did NOT reduce the observed NRMSE at all** (0.5909 before -> 0.5916 after,
   matrix=80/3T/ideal condition) - so it was NOT the (or not the only) cause of the streaking.

**Ruled out (tested, not just reasoned about):**
- NUFFT gridding/interpolation kernel (exact brute-force adjoint shows the identical artifact).
- DCF kernel width / iteration count (`dcf_width` swept 2/4/6/8 Nyquist cells x 15/30/60
  iterations on the REAL trajectory - NRMSE got monotonically WORSE with wider kernel, no
  improvement from more iterations. Directly contradicts "kernel too narrow" as a fix).
- Exact-duplicate-point handling at k=0 (pre-fix trajectory) - collapsing the 200 duplicates and
  splitting weight by count changed NRMSE by <0.1% - the iteration was already self-consistent
  there.
- DCF global calibration - sum(w) vs expected covered-k-space-area ratio is only 1.09 (9% over),
  not dramatically broken.
- The k=0 re-referencing bug itself (just fixed, see above - no NRMSE change).

**Still unexplained / in progress**: an outside research agent (reachable by name/id
`a1ab17f1d789623f2` via SendMessage, full history in this session's transcript) independently
reproduced a CLEAN, isolated, from-scratch round-trip test using the pre-fix (buggy,
zero-referenced) trajectory and got NRMSE=1.47% - nowhere near our production pipeline's ~59%.
This means there is a SEPARATE, compounding bug specific to our actual production code path
(`RepoAdapter.simulate_all`'s `encode_reconstruct` closure: `positions`/`_flatten`/
`ForwardOperator.forward`/`GriddingNUFFT.adjoint`) that a simplified reimplementation does not
reproduce - most likely in the position/flatten-order convention the class docstring documents
(`arr.T.reshape(-1)` vs `make_grid_2d`'s Y-outer/X-inner order) - which was validated via a
point-source test EARLY in a much earlier session, before STEAM->GRE, before the trajectory
source change above, before several matrix/fov/shot-count changes - so that validation can no
longer be trusted without redoing it against the CURRENT code. The agent has been asked to redo
a fresh point-source round-trip test using TODAY's actual `RepoAdapter` methods/classes (not a
reimplementation) and report back. **This was the last action before the user had to disconnect
from the internet mid-session** - resume by checking for the agent's response (it may have
already landed as a task-notification by the time this is read back).

**Do not regenerate the two "official" output figures yet** - they will still show the
unexplained streaking until this is resolved. `outputs/spatial_physics_ablation{,_55mT}/` on disk
right now reflect an EARLIER, less-fixed state (pre-dates the GRE switch, VOI removal, and the
k=0 trajectory fix) and are stale in multiple ways - don't treat them as current.

Also newly present in the repo root, not yet read/incorporated: `AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md`
(user opened this in their IDE right as the pause happened - likely relevant to the "would an
iterative CG-SENSE reconstruction help" discussion earlier this session; read it on resume) and
`nce_handover.md` (a Noncentral-Chi-Error metric spec from Gemini, requested earlier by the user
as a NRMSE replacement - not yet implemented, blocked behind getting the reconstruction itself
correct first, since NCE's own `sigma` noise parameter needs a real, understood noise/error model
to be meaningful, which this investigation is precisely trying to establish).

## Suggested order of attack next session

All numbered items from the original request batch are now DONE (3D phantom, GNL excitation-side
modeling, explicit spiral shot count, GNL scale decision, off-isocenter concomitant, registration
pipeline - see Session 3/4/5 additions above). Remaining:
1. Stack into one ULF/HF/UHF figure (item 5) — the only item left. UHF field strength (7T assumed,
   not yet confirmed with the user) still needs confirming. A reasonable GNL scale for the UHF
   condition also needs picking (smaller than 3T's 15, per the "larger at ULF" story, e.g. 1x-5x —
   not yet decided). This is mostly a plotting/layout change now that every per-condition piece of
   physics has been individually validated.

None of this is committed. Ask before committing (standing policy, unchanged).
