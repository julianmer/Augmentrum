# Claude Code handover: Augmentrum+ spatial-physics ablation figure

## Goal

Complete `spatial_physics_ablation.py` so it generates the final scientific figure using the **current Augmentrum `spatial_maps` branch** and **GIRF-Sim**.

The plotting/data-preparation framework is already implemented. Your main job is to replace the small `RepoAdapter` stubs with calls to the repositories' canonical APIs, troubleshoot unit/coordinate issues, and replace development-only spatial maps with better open maps if appropriate.

Do not rewrite the physics if equivalent functionality already exists in Augmentrum or GIRF-Sim.

---

## Target figure

One common anatomy and one **fully sampled 2D spiral** are used for every condition.

Columns:

1. Ideal
2. +B0
3. +B1+
4. +Concomitant field
5. +Measured GIRF
6. +Gradient nonlinearity
7. Combined

Rows:

1. Physical input / map / trajectory perturbation
2. Reconstruction
3. Matched absolute difference from ideal

Scientific message:

> Different scanner/acquisition effects create distinct, spatially structured changes in the same underlying anatomy, and Augmentrum+ can compose those effects in a common forward model.

---

## Files delivered

- `spatial_physics_ablation.py`
- this handover
- `figure_caption.txt`

The Python script already handles:

- anatomy loading;
- optional NIfTI B0/B1+ maps;
- development-only synthetic B0/B1+ maps;
- matched cropping/resizing;
- common display ranges;
- difference maps;
- NRMSE;
- trajectory RMS/max error if nominal/realized trajectories are returned;
- publication layout;
- PDF/SVG/PNG export;
- JSON metric/metadata export;
- a <500-character draft caption.

Run first:

```bash
python spatial_physics_ablation.py --prepare-only --outdir outputs/spatial_ablation
```

That checks the generic pipeline without inventing acquisition artifacts.

---

# 1. Inspect Augmentrum before editing

Work from the `spatial_maps` branch.

Find the actual functions/classes for:

- fully sampled 2D spiral trajectory;
- PyPulseq/Pulseq sequence loading;
- forward encoding and reconstruction;
- frequency-segmented B0/off-resonance;
- B1+ modulation;
- concomitant-field phase;
- gradient nonlinearity;
- combined/composable forward model.

The user has stated:

- B0 frequency segmentation is already implemented.
- B1+ uses supplied maps via complex-valued multiplication; **do not describe it as Bloch excitation or flip-angle simulation**.
- concomitant-field and GNL terms are position-dependent phase terms accumulated along the actual per-shot Pulseq trajectory.
- the current implementation groups trajectory samples by severity and uses segmented forward-NUFFT/adjoint-reconstruction operations.
- GIRF-Sim is imported by Augmentrum for GIRF, GNL and related gradient-response physics.

Use the actual code as the source of truth.

---

# 2. Use Augmentrum's fully sampled 2D spiral

Do not create a toy Archimedean spiral in the figure script.

Identify Augmentrum's fully sampled 2D spiral and return through `RepoAdapter.get_spiral()`:

- nominal k-space trajectory;
- actual gradient waveform;
- gradient raster / sample timing;
- ADC timing/mask;
- sequence path/identifier.

Use exactly the same spiral for ideal, B0, B1+, concomitant, GIRF, GNL and combined panels.

Record trajectory units in metadata.

---

# 3. Measured GIRF

Use **GIRF-Sim's measured Bacon GIRF loader**.

Do not redownload/reparse it if GIRF-Sim already has a canonical loader.

The public Bacon project now also contains an open GIRF measurement pipeline and calibration sequence, but that is not needed for this figure. The relevant input here is the already measured GIRF that GIRF-Sim supports.

For the GIRF panel return:

```python
extras["k_nominal"]
extras["k_girf"]
```

and, if useful:

```python
extras["nominal_gradients"]
extras["realized_gradients"]
```

The top-row GIRF visualization should be the nominal vs realized k-space trajectory, not a fabricated "GIRF spatial map."

---

# 4. B0 panel

Use Augmentrum's actual frequency-segmented off-resonance model.

Do **not** substitute:
- image blurring,
- a geometric warp,
- Fourier phase multiplication unrelated to the existing implementation.

Use the same 2D spiral as all other panels.

The script currently creates a representative synthetic B0 map when no map is provided. That is acceptable for code development, but for the final figure prefer one of:

1. a quantitative/open measured B0 map with clearly documented provenance;
2. a validated field map already used in another open MR physics repository;
3. if neither integrates cleanly, keep the synthetic map but explicitly report it as a representative synthetic ΔB0 map.

Potential open source to inspect:
- `mriphysics/reVERSE-GIRF` release assets contain B0/B1 field maps and GIRF data for published examples.
- If using those maps, verify units, orientation, licensing, and whether resampling onto the selected anatomy is scientifically sensible.

Do not treat HCP reversed-PE spin-echo pairs as a ready-made quantitative ΔB0 map without processing/units verification.

Return the actual number of frequency segments in metadata.

---

# 5. B1+ panel

Use Augmentrum's existing B1+ implementation.

The user states that B1+ is applied through **complex-valued multiplication**, not Bloch simulation.

Figure/caption wording should therefore be:
- "B1+ modulation"
- "relative B1+ map"

and NOT:
- flip-angle simulation
- excitation simulation
- Bloch simulation

The current script can accept a measured B1+ NIfTI using `--b1-map`, or a representative smooth synthetic map for development.

If you find a suitable open measured B1+ map, wire its download/preprocessing into a helper rather than hardcoding a local path.

---

# 6. Concomitant-field panel

Use Augmentrum's implemented concomitant-field model.

From the user's implementation description:

- it is computed from the real gradient waveform `G(t)=(Gx,Gy,Gz)` and B0;
- it uses the standard symmetric-coil Maxwell formulation;
- spatial basis functions include `z²`, `x²+y²`, `xz`, `yz`;
- accumulated phase is trajectory/time dependent.

For the top row, return ONE clearly defined spatial summary:
- RMS concomitant phase across ADC/readout time, OR
- maximum absolute accumulated phase across ADC/readout time, OR
- phase at a deliberately chosen informative sample.

Prefer RMS or max-over-readout because it is less arbitrary.

Store the exact convention in metadata:
```python
metadata["concomitant_map_definition"] = "RMS phase across ADC samples"
```

Return:
```python
extras["concomitant_phase_map"]
```

---

# 7. Gradient-nonlinearity panel

Use the current GNL implementation used by Augmentrum/GIRF-Sim.

From the user's description, GNL uses higher-order harmonic coefficients and real solid harmonics to generate a position-dependent phase field.

For the top row return a spatial summary using the same convention as concomitant phase if possible:
- RMS phase across readout, or
- maximum absolute phase across readout.

Return:
```python
extras["gnl_phase_map"]
```

Do not label it a displacement map unless the code actually calculates displacement in physical units.

---

# 8. Combined panel

This is important.

Do NOT generate the combined image by taking reconstructed outputs from the five single-effect panels and applying them sequentially in image space.

Use the actual Augmentrum composable forward model so:
- B0,
- B1+,
- concomitant field,
- measured GIRF,
- GNL

are active together during forward encoding/reconstruction in the way the software intends.

Record the full configuration in metadata.

---

# 9. Ideal/reference reconstruction

Ideal should use:
- same anatomy,
- same nominal fully sampled spiral,
- same reconstruction,
- same matrix/FOV,
- same numerical settings,

but without B0/B1+/concomitant/GIRF/GNL perturbations.

This is the reference for all difference maps and NRMSE.

---

# 10. Anatomy

The script self-starts with `skimage.data.brain()` only as a development fallback.

For the final abstract figure, prefer a clearly citable digital/anatomical source such as BrainWeb if straightforward.

If downloading BrainWeb is awkward because its official site uses interactive/custom downloads, it is acceptable to:
- download manually once,
- save a local NIfTI,
- pass it via `--anatomy`.

Do not let data-download plumbing delay the physics figure.

The same single slice must be used for all panels.

---

# 11. Open field-map option

If you want measured B0/B1+ maps without relying on HCP, inspect:

`https://github.com/mriphysics/reVERSE-GIRF`

Its release package is documented as containing B0 and B1 maps.

Before using:
- verify license;
- inspect map units;
- inspect anatomy/FOV/orientation;
- check whether the maps are from 3T or 7T;
- determine whether resampling to our anatomy would create an obviously artificial spatial relationship.

If the maps do not make sense with the chosen anatomy, keep the figure explicitly synthetic rather than pretending unrelated maps are scanner-matched.

---

# 12. Expected `RepoAdapter.simulate_all()` output

Return:

```python
ForwardResult(
    images={
        "ideal": ...,
        "b0": ...,
        "b1": ...,
        "concomitant": ...,
        "girf": ...,
        "gnl": ...,
        "combined": ...,
    },
    extras={
        "k_nominal": ...,
        "k_girf": ...,
        "concomitant_phase_map": ...,
        "gnl_phase_map": ...,
        # optional:
        "nominal_gradients": ...,
        "realized_gradients": ...,
    },
    metadata={
        "sequence": ...,
        "trajectory_units": ...,
        "field_strength_T": ...,
        "girf_source": "Bacon measured GIRF",
        "b0_frequency_segments": ...,
        "severity_segments": ...,
        "concomitant_map_definition": ...,
        "gnl_map_definition": ...,
        "reconstruction": ...,
    }
)
```

Complex images are fine; the plotting layer uses magnitude.

---

# 13. Numerical checks before accepting the figure

For every panel:

- verify array orientation matches the anatomy;
- verify FOV and scanner-coordinate convention;
- verify B0 is in Hz where expected;
- verify B1+ is dimensionless/appropriately normalized;
- verify gradient units;
- verify k-space units;
- verify phase is radians;
- verify the same nominal acquisition is used in every ablation.

Run sanity checks:

### Ideal
- reference should not differ between repeated runs.

### B1+
- output modulation should spatially correspond to the B1+ map.

### B0
- increasing ΔB0 amplitude should monotonically increase the off-resonance error for the chosen spiral, all else equal.

### GIRF
- setting identity/no-error GIRF should reproduce nominal trajectory.

### Concomitant/GNL
- zero coefficients should reproduce ideal.
- effect should vanish/reduce appropriately at conditions where the underlying model predicts it.

### Combined
- should reduce to individual panels if all other effects are disabled one at a time.

Do not tune artifact severity solely to make panels dramatic. If a physically defensible effect is subtle, the matched difference map exists for that reason.

---

# 14. Display requirements

Keep:
- identical reconstruction grayscale across all columns;
- identical difference-map scale across all effect columns if feasible;
- B0 diverging map centered at zero;
- phase maps diverging and centered at zero if signed;
- B1+ sequential map around physically meaningful relative values.

The script already handles common reconstruction/difference scales.

If one effect dominates the difference scale so strongly that all others disappear:
1. first report that fact;
2. consider normalized relative-difference maps as an additional inset;
3. do not silently give every column its own scale without clear legends.

---

# 15. Output artifacts

Final run should create:

- `spatial_physics_ablation.pdf`
- `spatial_physics_ablation.svg`
- `spatial_physics_ablation.png`
- `metrics.json`
- `forward_metadata.json`
- `prepared_inputs.npz`
- `prepared_inputs_metadata.json`
- `figure_caption.txt`

Also save any expensive intermediate forward results to a cache if runtime is substantial.

---

# 16. Draft caption

Keep the caption under ISMRM's 500-character limit.

Current draft:

> Spatially resolved acquisition effects in Augmentrum+. A common digital brain and fully sampled 2D spiral acquisition were used to isolate B0 off-resonance, B1+ modulation, concomitant fields, measured GIRF response, and gradient nonlinearity. Each column shows the input map or trajectory perturbation, reconstructed image, and matched difference from the ideal acquisition. The final column combines all effects to illustrate scanner-dependent spatial variability.

If terminology changes after inspecting the implementation, update only what is necessary and keep it <500 characters.

---

# 17. Scientific wording guardrails

Use:
- "representative synthetic B0/B1+ map" if synthetic;
- "measured Bacon GIRF";
- "B1+ spatial modulation";
- "spatially varying phase";
- "fully sampled 2D spiral";
- "frequency-segmented off-resonance model";
- "combined forward model."

Avoid unless actually established:
- scanner validation;
- subject-specific calibration;
- Bloch simulation;
- true 7T/3T matched map;
- vendor-specific gradient nonlinearity;
- measured higher-order GIRF, unless the loaded measured dataset really contains it.

The point of this figure is mechanistic visual separation, not validation of a particular scanner.
