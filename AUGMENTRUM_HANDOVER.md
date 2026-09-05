# GIRF-Sim → Augmentrum integration handover

Written for a fresh Claude Code session working in the Augmentrum repo, with
no prior context on GIRF-Sim. Covers (1) how to use GIRF-Sim as an installed
dependency, and (2) how to implement Augmentrum's own physics for the
placeholder artifact terms GIRF-Sim intentionally left unbuilt, without
forking or vendoring GIRF-Sim's source.

## 0. What GIRF-Sim is

A GIRF-based (Gradient Impulse Response Function) MRS/MRSI artifact
simulation and correction library: synthetic + real-measured (Bacon et al.)
GIRF models, arbitrary Pulseq `.seq` file loading, batched multi-shot
processing, and a composable forward-operator architecture designed
specifically so a consumer (Augmentrum) can add its own physics terms
(concomitant fields, static B0, gradient nonlinearity, coil sensitivities)
without touching GIRF-Sim's own source.

Five flat top-level modules (no package prefix — `import girf_synthetic`,
not `import girf_sim.girf_synthetic`):

| Module | What it's for |
|---|---|
| `girf_synthetic.py` | Core synthetic GIRF kernel bank, spatial solid-harmonic basis, parametric kernel sampling (`disabled`/`fixed`/`sampled` per-mechanism modes + returned ground truth), SVS voxel geometry (`paper_center`/`extended_voxel`) |
| `girf_mrsi_extensions.py` | MRSI extensions: batched multi-shot GIRF application, the joint k-t forward model, **the composable `ForwardOperator`/`PhaseTerm`/`DisplacementTerm` architecture (your main integration point)**, Nixon mechanical-sideband model |
| `bacon_girf.py` | Real measured-GIRF loader/forward model (Bacon, Jezzard & Clarke) — **now generalized to any user-supplied measured GIRF, not just Bacon's release** — batched variants, generic linear-interpolation helper |
| `seq_girf.py` | `.seq` file loading (any Pulseq file, not just ones this project generated), gradient/k-space extraction, batched multi-shot loading |
| `girf_module.py` | **`GIRFModule`** — the portable, two-tier entry point for external pipelines, plus real-data timing-alignment utilities |

## 1. Installation

GIRF-Sim ships a `pyproject.toml` (setuptools, flat `py-modules`, dependencies
`torch>=2.0`, `numpy`, `matplotlib`, `pypulseq>=1.4`). From Augmentrum's
environment:

```bash
pip install -e /path/to/GIRF-Sim
```

Verified end-to-end (imports + full test suite) from an unrelated working
directory in an isolated venv — it's a real, working dependency, not just a
sys.path hack.

## 2. Using GIRF-Sim as-is: `GIRFModule`

This is the primary API — don't reach into the other four modules directly
unless you need something `GIRFModule` doesn't expose.

```python
from girf_module import GIRFModule

# Synthetic (stochastic or fixed-seed) GIRF:
mod = GIRFModule.from_synthetic("some_sequence.seq", seed=0)

# Real measured GIRF — Bacon et al.'s released data:
mod = GIRFModule.from_measured("some_sequence.seq", bacon_data_dir="/path/to/zenodo/GIRFs")

# Real measured GIRF — YOUR OWN system's measurement (see §2a):
mod = GIRFModule.from_measured("some_sequence.seq", bacon=my_own_bacon_girf)
```

Key fields/methods on `GIRFModule`:

- `mod.k_nominal`, `mod.k_actual`, `mod.dk_error` — each `[num_shots, D, L]`
  cycles/m. **This matches Augmentrum's own trajectory convention** (`[B, S,
  D, L]`, confirmed directly from `kspace_reconstructor.py`'s own
  docstrings) with the batch axis implicit/dropped (`B=1`, one `.seq` file
  per `GIRFModule`) — stack multiple modules' tensors along a new leading
  axis to build a full `[B, S, D, L]` batch.
- `mod.get_shot(index)` — one shot's `k_nominal`/`k_actual`/`dk_error`, each
  `[D, L]`.
- `mod.phase_correction(positions, order=None)` — **synthetic tier only**:
  order-0 + order≥2 spatial phase, `[num_shots, N, L]` radians, for
  demodulating a reconstructed per-voxel signal. Raises `NotImplementedError`
  on the measured tier (Bacon-format released data covers order≤1 only,
  already captured in `k_actual`/`dk_error` — do not double-count it here).
- `mod.to_numpy()` / `mod.save_npz(path)` — for handing off to a non-torch
  or non-Python pipeline stage (mirrors Augmentrum's own
  `ShotIO.save_shots_npz` idiom).
- `mod.align_and_correct(measured_fid, predicted_phase_rad, search_range_s=(-1e-3, 1e-3))`
  and the standalone `estimate_fractional_delay(reference, measured, dt,
  search_range_s)` — **real-acquired-data use**: estimates the sub-sample
  timing offset between a GIRF-predicted phase and a real measured FID (via
  generalized cross-correlation), then returns the delay-aligned,
  demodulated FID. Read the caveat in §7 before using this.

**Non-obvious gotcha:** the polynomial spatial-harmonic basis this project
uses evaluates to exactly zero at position `(0,0,0)` for every order ≥1
term. A single-voxel query at isocenter therefore only ever sees the
order-0 (global) GIRF phase — not spatial (order≥1) effects, even though
they're real and present. Use an off-center position, or
`girf_synthetic.make_svs_positions(mode="extended_voxel", voxel_size_m=...)`
(spans the real voxel volume — reads a `.seq` file's `VOI_mm` if you have
one) to see them.

## 2a. Using your OWN measured GIRF (not Bacon's release)

A GIRF is a **fixed property of the gradient hardware** — how the whole
gradient chain responds to *any* input waveform — not something tied to one
particular sequence. (What *is* sequence-specific is the resulting k-space
error once you convolve a fixed GIRF with one sequence's particular
gradients — that part was already `.seq`-file-driven and is unchanged.) So
if you — or whoever built/calibrated Augmentrum's target scanner model —
have measured your own system's GIRF (by a Bacon-style method or any other),
you can plug it into the exact same downstream machinery without
reformatting it into Bacon et al.'s specific released `.npz` file layout:

```python
import torch
import bacon_girf as bg
from girf_module import GIRFModule

# H[b, a, f]: complex transfer function from a commanded gradient on
# physical input axis a (x=0,y=1,z=2) to output basis term b (0=B0, 1/2/3=x/y/z
# for order=1), at freq_hz[f]. Units: dimensionless for x/y/z, meters for B0.
H = ...           # [n_basis, 3, n_freq] complex, n_basis=4 for order=1
freq_hz = ...      # [n_freq] ascending, Hz
gammabar_hz_per_mT = ...   # your system's gyromagnetic ratio convention
adc_dwell_s = ...          # dwell time your GIRF was measured/estimated at

my_own_bacon = bg.BaconGIRF.from_arrays(H, freq_hz, gammabar_hz_per_mT, adc_dwell_s, order=1)
mod = GIRFModule.from_measured("some_sequence.seq", bacon=my_own_bacon)
```

`GIRFModule.from_measured` now takes `bacon_data_dir` **or** `bacon` (exactly
one, not both, not neither — it raises `ValueError` otherwise). Passing
`bacon_data_dir` is unchanged from before (Bacon et al.'s released layout,
loaded via `bacon_girf.load_bacon_girf`); passing a pre-built `bacon`
(`bacon_girf.BaconGIRF`, typically built via `BaconGIRF.from_arrays`) is the
new, generic path — this is the right seam for a per-scanner "measured GIRF"
config in an augmentation pipeline. Existing calls of the form
`GIRFModule.from_measured(seq_path, bacon_data_dir)` (positional or keyword)
are unaffected.

`BaconGIRF.from_arrays` only checks shapes/dtypes (complex `H`, matching
`freq_hz` length, `H.shape[0]` matching the expected basis size for `order`)
— it does **not** validate physical plausibility. Getting the axis-order and
unit conventions right (documented in the function's own docstring, and in
`bacon_girf.py`'s module docstring) is your responsibility; a wrong
convention silently produces wrong, not obviously-wrong, results downstream.

**Note the asymmetry this implies for augmentations:** a measured GIRF can
only ever be "fixed" (it's real hardware data, not a stochastic model — there
is no `sampled` mode for `bacon`, unlike the synthetic tier's
`GIRFKernelParams.*_mode`). What *can* vary per augmentation sample is the
trajectory (which `.seq` file / which shots) the fixed measured GIRF gets
applied to — that already works exactly as it did before this change, once
per `seq_path` you load.

## 3. The extensibility architecture: building your own physics terms

`girf_mrsi_extensions.py` defines two tiny abstract base classes and one
operator class that composes them — this is the actual plugin seam, and you
extend it entirely from Augmentrum's own code:

```python
from abc import ABC, abstractmethod

class PhaseTerm(ABC):
    """One additive contribution to the total encoding phase phi(r,t)."""
    name: str = "phase_term"
    @abstractmethod
    def phase(self, positions: Tensor, t: Tensor) -> Tensor:
        """Returns [N,T] radians, evaluated at `positions` [N,D] and `t` [T]."""

class DisplacementTerm(ABC):
    """One spatial-displacement contribution: r_eff = r + Delta r(r)."""
    name: str = "displacement_term"
    @abstractmethod
    def displace(self, positions: Tensor) -> Tensor:
        """Returns [N,D] effective positions."""
```

```python
class ForwardOperator:
    def __init__(
        self,
        positions: Tensor,          # [N,D]
        k_nominal: Tensor,          # [D,T] commanded trajectory
        dk_girf: Tensor,            # [D,T] GIRF-predicted trajectory ERROR
        phase_terms: Optional[Sequence[PhaseTerm]] = None,
        displacement_terms: Optional[Sequence[DisplacementTerm]] = None,
        coil: Optional[CoilSensitivity] = None,
    ): ...
    def forward(self, phantom, dt, noise_std=0.0, chunk_size=4096) -> JointKTMRSIResult: ...
```

`phase_terms` are summed additively; `displacement_terms` are applied in
order (each further displaces the position the next term, and the final
encoding exponential, see). **`ForwardOperator` itself never needs to
change** when you add a new term — that's the whole point of this design.
Subclass `PhaseTerm`/`DisplacementTerm` from Augmentrum's own repo (e.g.
`augmentrum/physics/concomitant.py`), `import` the ABCs from the installed
`girf_mrsi_extensions` package, and pass instances into `ForwardOperator`'s
constructor. No fork or vendored copy of GIRF-Sim needed.

Two real (non-placeholder) terms already exist and are good reference
implementations to model yours after: `GIRFGlobalPhase` (order-0) and
`GIRFNonlinearPhase` (order≥2), both in `girf_mrsi_extensions.py`.

### 3a. The three placeholders — what each one needs

All three currently raise `NotImplementedError` with a docstring explaining
exactly why. This section gives you what you need to actually implement
them in Augmentrum.

**`ConcomitantFieldPhase(PhaseTerm)`** — Maxwell/concomitant gradient
fields (the classic `∇·B=0`/`∇×B=0`-mandated second-order field that
accompanies any real linear gradient set).

- Needs: the actual **time-resolved gradient waveform** `G(t) =
  (Gx(t),Gy(t),Gz(t))` (NOT the harmonic-coefficient GIRF representation
  used elsewhere in this project — concomitant fields are a direct
  algebraic function of the real gradient, not of GIRF-filtered
  coefficients) and the scanner's **B0** (Tesla).
- Where to get them: `seq_girf.load_seq_gradients(path)` /
  `load_all_shots_batched(path)` already gives you `gradients_t_per_m`
  `[3,T]` or `[num_shots,3,T]` (T/m) for any loaded `.seq` file, on the
  file's own native raster. `B0` is already parsed into
  `sg.definitions["B0"]` (verify its units/sign convention against the
  specific file — not independently confirmed here).
- The physics: leading-order concomitant field along z is a standard
  second-order term, quadratic in the gradient amplitudes and quadratic in
  position — see Bernstein, Meyer, King & Zhou, "Concomitant gradient
  terms in phase contrast MR," MRM 39:300–308 (1998), and King, Ganin,
  Zhou & Bernstein, MRM 41:103–112 (1999) for the spiral/EPI-relevant
  form. **Do not copy a formula from memory (including from an LLM) into
  physics code without checking it against a primary source** — the exact
  numeric prefactors depend on the gradient-coil symmetry assumptions
  (ideal Maxwell-pair coils vs a specific vendor design), and getting a
  physics constant wrong here is exactly the kind of thing this project
  has been careful never to guess at. Derive/verify the coefficients
  directly from one of the cited references (or from Maxwell's equations
  for your specific coil model) before trusting the implementation.
- Once you have `B_conc(x,y,z,t)` in Hz (multiply by gammabar,
  `girf_mrsi_extensions.GAMMA_HZ_PER_T`, if you start from Tesla), the
  `phase()` method is just `2*pi*cumsum(B_conc, dim=-1)*dt` — see
  `GIRFGlobalPhase.phase()` for the exact pattern to follow.

**`StaticOffResonancePhase(PhaseTerm)`** — static/slowly-varying B0
inhomogeneity (residual shim, susceptibility).

- Needs: an externally-supplied per-voxel field map, `freq_map_hz` `[N]`
  Hz, matching your `positions` array.
- Where to get it: **Augmentrum's own field-map/susceptibility-simulation
  machinery** (a real field map, or a susceptibility simulation) —
  deliberately out of GIRF-Sim's scope; it will never be generated here.
  The class already accepts `freq_map_hz` as a constructor field and
  implements the pass-through formula
  (`2*pi*freq_map_hz.reshape(-1,1)*t.reshape(1,-1)`) once you supply it —
  you likely don't even need to subclass this one, just construct
  `StaticOffResonancePhase(freq_map_hz=your_map)` directly.

**`GradientNonlinearityDisplacement(DisplacementTerm)`** — vendor
gradient-coil deviation-from-linearity spatial displacement.

- Needs: **vendor-specific gradient-coil spherical-harmonic coefficients**
  (Siemens/GE/Philips coil models).
- Where to get them: if Augmentrum has access to real vendor calibration
  data, use it. If not, **leave this one raising** — GIRF-Sim's own
  discipline throughout this project has been to never invent a
  measured/calibration value and raise `NotImplementedError` instead; carry
  that discipline into Augmentrum rather than fabricating plausible-looking
  coefficients.

## 4. Critical gotcha for ablation studies: `dk_girf` is not a phase term

If you're running an ablation that tests each artifact mechanism one at a
time (e.g. "concomitant fields only, no GIRF"), it is **not enough** to
just omit `GIRFGlobalPhase`/`GIRFNonlinearPhase` from `phase_terms`.
`ForwardOperator.forward()` uses `dk_girf` **unconditionally** —
independent of `phase_terms` — to build the actual encoding trajectory:

```python
k_actual = apply_girf_trajectory_error(self.k_nominal, self.dk_girf)
```

So a real "GIRF off" condition needs **both**:
1. No GIRF `phase_terms`, **and**
2. `dk_girf=torch.zeros_like(k_nominal)` passed to the constructor.

Otherwise your "concomitant-only" condition silently still has a
GIRF-perturbed k-space trajectory baked in, contaminating the ablation.

### Suggested ablation pattern

```python
conditions = {
    "none":               dict(phase_terms=[],                         dk_girf=torch.zeros_like(k_nominal)),
    "girf_only":          dict(phase_terms=[girf_global, girf_nonlin],  dk_girf=real_dk_girf),
    "concomitant_only":   dict(phase_terms=[concomitant_term],          dk_girf=torch.zeros_like(k_nominal)),
    "concomitant_+_girf": dict(phase_terms=[girf_global, girf_nonlin, concomitant_term], dk_girf=real_dk_girf),
}
results = {}
for name, cfg in conditions.items():
    op = ForwardOperator(positions, k_nominal, cfg["dk_girf"], phase_terms=cfg["phase_terms"])
    results[name] = op.forward(phantom, dt)
```

## 5. General discipline to carry over

This project's one hard rule, worth keeping in Augmentrum: **never invent a
missing measured/calibration value.** Where real data isn't available
(vendor coil coefficients, a field map, an undisclosed third-party
algorithm), the code raises `NotImplementedError` with a docstring
explaining exactly what's missing and where it would come from, rather than
guessing a plausible-looking substitute. Keep that pattern for anything you
add.

## 6. Known limitations to be aware of

- **`estimate_fractional_delay`/`align_and_correct`** (real-data timing
  alignment) is a **generic, standard** generalized-cross-correlation
  method — explicitly **not** a reproduction of Bacon et al.'s own
  undisclosed real-data correction-pipeline alignment algorithm (that
  codebase is a different, unavailable repo from the GIRF-measurement one
  this project cloned). It also fundamentally needs real spectral
  content in the phase signal to localize a delay — a phase trace
  dominated by slow eddy/drift decay (near-DC) gives a poorly localized,
  potentially misleading estimate. Works well against a large,
  fast-varying phase burst (e.g. during an active gradient event); poorly
  against the near-flat inter-gradient baseline.
- **Measured tier** (`GIRFModule.from_measured`, whether via
  `bacon_data_dir` or your own `bacon`) only has order≤1 data — no order≥2
  spatial phase correction is available for it, and `phase_correction()`
  raises for this tier by design.
- A synthetic kernel's per-mechanism mode (`GIRFKernelParams.*_mode` —
  `"sampled"`/`"fixed"`/`"disabled"`) defaults to `"sampled"` everywhere;
  set it explicitly if you want a deterministic ablation-friendly kernel
  rather than a stochastic draw.

## 7. Where to look for more detail

The module docstrings in this repo are deliberately thorough — they record
what's verified vs. assumed, what's a deliberate design decision vs. a
flagged gap, and why. Read `girf_mrsi_extensions.py`'s `ForwardOperator`/
`PhaseTerm`/`DisplacementTerm` section, `bacon_girf.py`'s module docstring,
and `BaconGIRF.from_arrays`'s docstring before extending anything — they
answer most "why does it work this way" questions this handover doesn't
cover.
