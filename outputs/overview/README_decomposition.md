# Augmentrum+ overview diagram decomposition guide

## What is included

This bundle contains:

1. **Cropped PNG assets** from the generated overview figure.
2. **`augmentrum_overview_flat_embedded.svg`**  
   A flat SVG that embeds the full raster figure for reference.
3. **`augmentrum_overview_template.svg`**  
   An editable SVG scaffold with named placeholders for panel images.
4. **Crop manifests** in JSON and CSV.
5. **A Python starter script** (`overview_parts_starter.py`) for cropping/replacing assets.

## Important note about SVG

The original figure is a raster PNG, so it cannot be converted into a *fully editable true vector figure* automatically without redrawing it.
The provided SVG files are intended for **Inkscape assembly and replacement**:
- one is a flat embedded reference;
- the other is a clean **editable layout scaffold**.

## Suggested workflow

1. Open `augmentrum_overview_template.svg` in Inkscape.
2. Import the cropped PNGs as temporary guides.
3. Replace the image placeholders with real outputs from Python.
4. Redraw or refine arrows, labels, equations, and icons directly in Inkscape.
5. Keep the final typography and alignment in Inkscape.

## Approximate colormaps used / recommended replacements

These are **appearance-based approximations** rather than exact latent model settings.

### Scanner-characterization maps
- **B0 maps**: `coolwarm` or `RdBu_r`
- **small B1+ / B1- maps**: `coolwarm` with narrow dynamic range
- **small GIRF trajectory**: orange line work, no colormap
- **small concomitant map**: `turbo` or `twilight_shifted`
- **small GNL grid**: custom line colors (cyan / magenta / orange), not a scalar colormap

### Operator tiles
- **ΔB0(r)**: `coolwarm`
- **B1+(r), B1-(r)**: `turbo` (good visual match), or `viridis` if you want a more conservative scientific palette
- **Realized trajectory**: blue dashed nominal line + orange realized line
- **Spatially varying phase**: `turbo`
- **Encoding interface / k-space starburst**: grayscale or custom black background with white radial lines

### Application thumbnails
- **k-space starburst**: grayscale (`gray`)
- **MRI images**: grayscale (`gray`)

## Which parts are easy to generate in Python?

### Very suitable for Python
- B0 maps
- B1+ / B1- maps
- Spiral trajectories
- GIRF nominal-vs-realized trajectory overlays
- Concomitant and GNL scalar maps
- k-space starburst plots
- MRI thumbnails from NIfTI/NumPy arrays
- field-strength source/target images
- application example images
- error maps / artifact realization panels

### Better finished in Inkscape
- rounded panel containers
- section headers
- arrows between blocks
- equation typesetting / final formula placement
- icon strip at the bottom
- final text alignment and publication polish

## Recommended asset naming convention

Use the placeholder IDs in `augmentrum_overview_template.svg` so the final assembly stays organized.

Example:
- `ph_operator_b0_map`
- `ph_operator_realized_trajectory`
- `ph_app_improved_result`

## Starter colormap recommendations in Python

```python
import matplotlib.pyplot as plt

plt.imshow(b0_map, cmap="coolwarm")
plt.imshow(b1_map, cmap="turbo")
plt.imshow(phase_map, cmap="turbo")
plt.imshow(brain, cmap="gray")
```

If you want a more publication-conservative alternative to `turbo`, try:
- `viridis`
- `magma`
- `cividis`

## Practical note for Claude Code

Claude can regenerate all placeholder images from real data using the crop names and dimensions in the manifests, then place them into the SVG template or recreate the whole layout programmatically.

## B0 and B1+ maps:
- https://github.com/hMRI-group/hMRI-toolbox-demodata
- https://ora.ox.ac.uk/objects/uuid:55ca977f-62df-4cbf-b300-2dc893e36647
- https://doi.org/10.34973/m20r-jt17