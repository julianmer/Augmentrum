# Handover: `make_grid_2d` pixel-centering bug

**File:** `girf_synthetic.py` (repo root), function `make_grid_2d`.

## The bug

```python
def make_grid_2d(nx, ny, fov_x, fov_y, z=0.0, device=None, dtype=torch.float32):
    xs = torch.linspace(-fov_x / 2.0, fov_x / 2.0, nx, device=device, dtype=dtype)
    ys = torch.linspace(-fov_y / 2.0, fov_y / 2.0, ny, device=device, dtype=dtype)
    ...
```

`torch.linspace(-fov/2, fov/2, n)` is **endpoint-inclusive**: it places `n` points spanning the
full `[-fov/2, fov/2]` interval with spacing `fov/(n-1)`. That is not the standard NUFFT/FFT
pixel-centered grid, which needs `n` points spanning `[-fov/2, fov/2)` with spacing `fov/n` (e.g.
`(arange(n) - n/2) * (fov/n)`).

For `nx=80` this is a `80/79 = 1.27%` per-step spacing error. It's small per pixel but
accumulates linearly with distance from center, reaching ~0.5 pixels of position error at the
FOV edge for `nx=80` - exactly at the rounding threshold, which is why it showed up as a clean
step function rather than gradual blur.

## How it was found and confirmed

Encoding a single point source through the real forward operator (`ForwardOperator`, using
`make_grid_2d`'s positions) and reconstructing it (`GriddingNUFFT.adjoint`) should put the
reconstructed peak back at the exact source pixel. Testing 8 asymmetric point locations in an
80x80 grid showed a consistent pattern: **0-pixel shift for indices below the grid midpoint
(40), +1-pixel shift at or above it** - independently in x and y. This is the textbook signature
of an `N/2` vs `N/2-1` (or equivalently, wrong grid spacing) convention mismatch between the
position grid used for encoding and the grid the reconstruction/k-space scaling assumes
(`kmax = nx/(2*fov)` elsewhere in this pipeline implicitly assumes `fov/nx` spacing).

Replacing the grid with the pixel-centered version eliminated the shift completely: **0/0 shift
at all 8 test locations**, vs. consistent +1 shifts on one or both axes before the fix.

## The fix

```python
def make_grid_2d(nx, ny, fov_x, fov_y, z=0.0, device=None, dtype=torch.float32):
    xs = (torch.arange(nx, device=device, dtype=dtype) - nx / 2.0) * (fov_x / nx)
    ys = (torch.arange(ny, device=device, dtype=dtype) - ny / 2.0) * (fov_y / ny)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    zz = torch.full_like(xx, z)
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
```

(Same change applies to any other `make_grid_*`/position-grid helper in this file built the same
way, if there are others - only `make_grid_2d` was checked here.)

## Important caveat: this does NOT fix the streaking/NRMSE issue

This bug affects **precise point-source localization** (important for registration, point-spread
accuracy, anything comparing absolute positions) but does **not** meaningfully change full-phantom
reconstruction NRMSE (measured 0.1893 after the fix vs. 0.1892-0.1894 before, across several other
tested configurations) - the remaining visible streaking in extended-object reconstructions is a
separate, apparently inherent property of single-pass density-compensated adjoint reconstruction
for this trajectory class (confirmed by matching an idealized bug-free synthetic spiral's NRMSE),
not something this grid fix resolves. Fix it for correctness, but don't expect it to change the
ablation figure's visual artifact level.
