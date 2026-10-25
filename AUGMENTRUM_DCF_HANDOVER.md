# Handover: fix `GriddingNUFFT.density_weights` in Augmentrum

**Repo:** `/run/media/johnlamaster/data/Augmentrum`
**File:** `augmentrum/sampling/kspace_reconstructor.py`, `GriddingNUFFT.density_weights` (around line 202)
**Also affected (same function, called through):**
- `KspaceReconstructor.density_weights` (around line 420), used by `KspaceReconstructor.adjoint`-style reconstruction (around line 495);
- `augmentrum/sampling/kspace_sampling.py` around line 3795 (`dcf = gridder.density_weights(recon_coords)` in the dual-trajectory path).

## Summary

The Pipe density-compensation weights have two problems:

1. **They are wrong when the grid oversampling is large.**
   - At `osf=4` the weights are not proportional to the k-space area per sample. The summed weight per k-space ring varies by 12–33 % (coefficient of variation) across radius.
   - At `osf=2` the same trajectories give 0.5–1.3 %.
   - More iterations do not help: 20 and 60 iterations gave identical results.
2. **Their scale is arbitrary.** The function returns `w / w.max()`, so the reconstructed image intensity depends on the trajectory. Two reconstructions of the same object with different interleave counts cannot be compared directly.

Separately, the adjoint's absolute scale changes with `osf`: with the same weights, the image mean scales as 1/osf². This is worth fixing at the same time; see item 3 below.

## How this showed up

GIRF-Sim Figure 2 (`scripts/ismrm_figures/fig2_measured_girf_mri.py`) reconstructs a 128 × 128 Shepp-Logan phantom with `GriddingNUFFT(osf=4.0)`. It runs at `osf=4` because a comment in that script says the default `osf=2` gave speckle and ring artefacts with radial ramp weights. The trajectory is rotated copies of a spiral arm with 455 samples, at 88 or 400 interleaves.

With `density_weights`:
- the flat region of the reference image had a visible ripple (std/mean 0.08 at 88 interleaves, 0.13 at 400);
- the 400-interleave image was about twice as bright as the 88-interleave one.

An earlier radial version of the same figure had already been switched to hand-coded ramp weights because "the Pipe iteration did not converge cleanly". That is likely the same bug.

## Reproduction (Augmentrum only, about 1 min)

```python
"""Augmentrum GriddingNUFFT.density_weights vs exact area weights."""
import sys, numpy as np
sys.path.insert(0, "/run/media/johnlamaster/data/Augmentrum")
from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT

N = 128
yy, xx = np.mgrid[:N, :N] - N / 2
disc = ((xx ** 2 + yy ** 2) <= (0.35 * N) ** 2).astype(np.complex64)[None]  # uniform disc phantom
inner = (xx ** 2 + yy ** 2) <= (0.25 * N) ** 2                              # flat region

def spiral(S, L, turns):
    """Archimedean spiral interleaves, |k| <= 0.5 (normalized), [S*L, 2]."""
    t = np.linspace(0, 1, L)
    th = 2 * np.pi * np.arange(S)[:, None] / S
    z = 0.5 * t * np.exp(1j * (2 * np.pi * turns * t + th))
    return np.stack([z.real, z.imag], -1).reshape(-1, 2).clip(-0.5, 0.5 - 1e-6), z

def area_weights(z):
    """Exact area per sample for rotated interleaves: |k| |dk| |cos(angle(dk) - angle(k))|
    (Jacobian of (t, rotation angle) -> k)."""
    dk = np.gradient(z, axis=-1)
    return (np.abs(z) * np.abs(dk) * np.abs(np.cos(np.angle(dk) - np.angle(z)))).reshape(-1)

def ring_cv(c, w, n=10):
    """CV across radius of (summed weight in ring) / (ring area); 0 for correct DCF."""
    r = np.hypot(*c.T); e = np.linspace(0, 0.5, n + 1)
    d = np.array([w[(r >= a) & (r < b)].sum() / (b * b - a * a) for a, b in zip(e[:-1], e[1:])])
    return d.std() / d.mean()

for osf in (2.0, 4.0):
    nu = GriddingNUFFT((N, N), osf=osf)
    for S, L, turns in ((16, 2000, 4), (64, 455, 1), (400, 455, 1)):
        c, z = spiral(S, L, turns)
        c = c.astype(np.float32)
        y = nu.forward(disc, c)
        for name, w in (("pipe", nu.density_weights(c)), ("area", area_weights(z))):
            w = np.asarray(w, np.float64)
            img = np.abs(nu.adjoint(y * w[None].astype(np.float32), c)[0])
            print(f"osf={osf:g} S={S:3d} L={L:4d} turns={turns} {name}: ring CV {ring_cv(c, w):.3f}, "
                  f"flat ripple {img[inner].std() / img[inner].mean():.3f}, disc mean {img[inner].mean():.3g}")
```

Measured output, with the current code at commit `a069bf5`:

| osf | Interleaves × samples | Pipe ring CV | Area ring CV | Pipe disc mean |
|---|---|---|---|---|
| 2 | 16 × 2000 (4 turns) | 0.005 | 0.002 | 0.236 |
| 2 | 64 × 455 (1 turn) | 0.013 | 0.010 | 0.228 |
| 2 | 400 × 455 (1 turn) | 0.013 | 0.010 | **0.331** |
| 4 | 16 × 2000 (4 turns) | **0.153** | 0.002 | 0.214 |
| 4 | 64 × 455 (1 turn) | **0.333** | 0.010 | 0.209 |
| 4 | 400 × 455 (1 turn) | **0.122** | 0.010 | 0.253 |

- The analytic area weights stay correct at both `osf` values.
- Pipe fails only at `osf=4`.
- Pipe's "disc mean" changes with the trajectory (0.21–0.33 for the same object) because of the max-normalisation.

## Likely cause (verify before relying on it)

`density_weights` runs the Pipe–Menon iteration `w ← w / (C·Cᵀ w)` with the reconstruction's own gridding kernel: `KaiserBesselInterpolator(grid_size, osf, width=4)`. The kernel width is fixed at 4 cells of the oversampled grid. That is 2 Nyquist cells at `osf=2` but only 1 Nyquist cell at `osf=4`.

The Pipe fixed point approximates 1/density only if the kernel averages over the local sample spacing, including the gap between neighbouring interleaves or turns, which is up to about 1 Nyquist cell at the edge of a critically sampled trajectory. At 1 Nyquist cell the kernel is too narrow. The iteration then converges to weights that equalise the gridded sum on the fine grid, not the true area per sample.

This fits the data: the weights are fine at `osf=2`, bad at `osf=4`, and more iterations don't change them.

## Suggested fix

1. **Decouple the DCF kernel from the reconstruction `osf`.** Compute the Pipe weights on their own grid with a fixed width in Nyquist units, for example a KB kernel of width `2·osf` oversampled cells (2 Nyquist cells) on the same grid, or a dedicated `osf=2`, `width=4` grid. Better still, use the Zwart, Johnson & Pipe (MRM 2012) optimised DCF kernel, which is designed for exactly this.
   - Expose `dcf_osf` / `dcf_width` (or a `dcf_kernel`) arguments with a sensible default. Don't silently reuse the reconstruction interpolator.
   - Note: `density_weights` already builds its own `KaiserBesselInterpolator` when the interpolator is not a `GriddingKernel` (e.g. `LinearInterpolator` for `nufft_impl='interp'`). That branch is the natural place to apply the fixed DCF kernel in every case.
2. **Return weights with a physical scale instead of `w / w.max()`.** At the Pipe fixed point the gridded weighted sum is about 1 per cell, so the un-normalised `w` already has a meaningful scale. Normalise explicitly so that `sum(w)` equals the covered k-space area in Nyquist cells (π/4 · N² for a full disc on an N × N image), or so that a uniform-object reconstruction returns that object's value. Document the convention in the docstring.
   - Check callers that may rely on max = 1: `kspace_sampling.py` around line 3795 multiplies by `amplitude`, so check whether `amplitude` compensates for the max-normalisation.
3. **Optional, related:** normalise the adjoint (or gridding) so the image scale does not depend on `osf`. Currently the same weights give a disc mean that scales as 1/osf² (3.9e-5 at `osf=2` vs 9.7e-6 at `osf=4` in the "area" rows). If the convention after item 2 is "uniform object reconstructs to its value", this has to be handled anyway.
4. **Revisit the `osf=4` workaround.** GIRF-Sim moved to `osf=4` because of speckle at `osf=2` with radial ramp weights. Once the DCF is fixed, check whether `osf=2` (the Augmentrum default) is accurate enough again; if so, the GIRF-Sim figure can drop to `osf=2`.

## Acceptance tests (suggested for `tests/sampling/`)

Using the reproduction above as a parametrised test over `osf ∈ {1.5, 2, 4}` and trajectories {radial with 1× and 3× samples per spoke, Archimedean spiral 16 × 2000, 64 × 455, 400 × 455, and an out-and-back arm: spiral out, radial return}:

- ring-density CV of `density_weights` ≤ 0.03 (the analytic weights give ≤ 0.01);
- flat-region ripple of the disc reconstruction ≤ 1.5 × the analytic-weight value;
- uniform-disc reconstructed mean identical (±2 %) across interleave counts and across `osf`;
- no regression for the existing callers (`KspaceReconstructor`, the dual-trajectory path in `kspace_sampling.py`).

## GIRF-Sim side, once Augmentrum is fixed

`scripts/ismrm_figures/fig2_measured_girf_mri.py` currently works around this:
- `reconstruct_spiral_variant` uses hand-coded area (Jacobian) weights;
- `reconstruct_variant` (radial) uses hand-coded ramp weights.

After the fix:
- replace both with `nufft.density_weights(coords_nominal)`;
- delete the workaround comments;
- re-check `osf`;
- rerun Figure 2.

The spiral trajectory construction (`build_spiral_girf_trajectory`) is independent of this and stays.
