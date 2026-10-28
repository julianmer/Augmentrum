#!/usr/bin/env python3
"""
Generates real-data replacements for the overview-figure placeholder crops, sized to match
crop_manifest.json's aspect ratio (at full, Inkscape-friendly resolution - see OUT_DPI), using
data produced by scripts/spatial_physics_ablation.py and scripts/ismrm/figure_physics_informed_cg.py,
both rerun with REAL HCP anatomy (--anatomy-source hcp) into outputs/overview/hcp_anatomy_ablation
and outputs/overview/hcp_anatomy_cg_recon - kept separate from outputs/spatial_physics_ablation/ and
outputs/ismrm_cg_recon_matrix128/, which remain the Shepp-Logan science-figure outputs.

Only panels backed by real arrays are generated. Panels with no corresponding real data yet
(scanner_b1minus_map, tissue_contrast_maps_thumb, sequence_waveform_schematic,
source/target_field_strength_image, app_training_benchmark_data) are intentionally skipped rather
than faked - see README_decomposition.md "Which parts are easy to generate in Python?".

Run with the python3.13 conda env (has torch/matplotlib/nibabel/gradunwarp):
  /home/johnlamaster/miniconda3/envs/python3.13/bin/python outputs/overview/generate_real_panels.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # repo root, for augmentrum.*
sys.path.insert(0, str(HERE.parent.parent / "scripts"))  # for spatial_physics_ablation
from overview_parts_starter import load_manifest  # noqa: E402

ABLATION = HERE / "hcp_anatomy_ablation"
CG = HERE / "hcp_anatomy_cg_recon"
OUT = HERE / "generated"
OUT.mkdir(exist_ok=True, parents=True)

# Figures are saved at the manifest's crop ASPECT RATIO but at this many pixels per "manifest
# unit" (the manifest was authored assuming 100px/unit) - i.e. full resolution for Inkscape to
# scale down, not pre-shrunk to the small placeholder size.
OUT_DPI = 400


def mag(x):
    return np.abs(x)


def norm01(a, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def dims():
    man = load_manifest()
    return {c["name"]: (c["width"], c["height"]) for c in man["crops"]}


def _new_fig(name, d, facecolor="white"):
    w, h = d[name]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=OUT_DPI, facecolor=facecolor)
    return fig, (w, h)


def save_sized(array, name, cmap, d, vmin=None, vmax=None, facecolor="white", mask=None):
    """mask: optional boolean array, True = keep, False -> rendered as facecolor (NaN'd out),
    for brain-only maps so the background matches the page instead of the colormap's zero-color."""
    fig, _ = _new_fig(name, d, facecolor)
    ax = plt.axes([0, 0, 1, 1])
    arr = array.astype(float)
    if mask is not None:
        arr = np.where(mask > 0, arr, np.nan)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(facecolor)
    ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_axis_off()
    fig.savefig(OUT / f"{name}.png", dpi=OUT_DPI, facecolor=facecolor)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def _girf_module(outdir, matrix, fov_m):
    """Rebuilds GIRF-Sim's own GIRFModule (shot-indexed k_nominal/k_actual/dk_error, shape
    [S, 3, L]) via the SAME RepoAdapter.get_spiral/load_bacon_measured_girf the real pipeline
    uses - reuses the cached .seq file in outdir, so this is cheap. Per-shot, un-flattened arrays
    straight from GIRF-Sim - no manual reshape/chunking of our own to get wrong."""
    import spatial_physics_ablation as spa
    adapter = spa.RepoAdapter(None, None, outdir=outdir, field_strength_t=3.0, device="cpu",
                              slice_offset_m=0.03)
    adapter.get_spiral(matrix, fov_m)
    return adapter.load_bacon_measured_girf()


def save_trajectory_single_shot(mod, name, d, shot=0):
    # Same data/convention as GIRF-Sim's own fig1_forward_model.py panel_b_plot (one real shot,
    # nominal vs GIRF-actual/measured) - but WITHOUT its zoom inset, which only works at a full
    # journal-panel size; at this tile's small footprint an inset + legend just overlaps into
    # illegible clutter, so this keeps the two-color trajectory overlay and a compact legend only.
    k_nom = mod.k_nominal[shot].numpy()  # [D, L]
    k_act = mod.k_actual[shot].numpy()

    fig, _ = _new_fig(name, d)
    ax = plt.axes([0.06, 0.06, 0.88, 0.88])
    ax.plot(k_nom[0], k_nom[1], lw=1.3, color="tab:blue", label="Nominal")
    ax.plot(k_act[0], k_act[1], lw=1.3, color="tab:red", alpha=0.85, label="Realized")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(OUT / f"{name}.png", dpi=OUT_DPI, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def save_trajectory_undersampled(mod, name, d, n_show=24):
    # An actually plausible undersampled multi-shot trajectory: the real sequence's 200 shots are
    # ordered by golden angle (see spatial_physics_ablation.py's RepoAdapter.get_spiral docstring
    # - "every shot ... at its own distinct golden-angle azimuth"), so a PREFIX of shots in
    # storage order is evenly distributed in angle (that's the whole point of golden-angle
    # ordering: any truncation gives near-uniform coverage). A STRIDED subset of a golden-angle
    # sequence is NOT evenly distributed and isn't how any real undersampled trajectory is built -
    # this plots the former, nominal-only (this panel isn't about GIRF deviation).
    k_nom = mod.k_nominal.numpy()  # [S, D, L]

    fig, _ = _new_fig(name, d)
    ax = plt.axes([0.04, 0.04, 0.92, 0.92])
    for s in range(n_show):
        kn = k_nom[s]
        ax.plot(kn[0], kn[1], '-', lw=0.7, color='#4C78A8', alpha=0.9)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(OUT / f"{name}.png", dpi=OUT_DPI, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def save_girf_realized_flower(name, d, n_shots=28, first_shot=20,
                              girfsim_root="/run/media/johnlamaster/data/Postdoc/GIRF-Sim"):
    # Matches the original mockup's own GIRF icon (a dense rosette of thin orange radiating arms)
    # as an actual plot of real data: the GIRF-distorted (realized) trajectory, INCLUDING the
    # rewinder (seq_girf.load_all_shots_batched/GIRFModule clip to the ADC window only, dropping
    # the rewinder where deviation is actually largest - confirmed numerically: ~2.3 cyc/m at the
    # rewinder vs ~0.2-0.4 cyc/m inside the ADC window), for n_shots golden-angle-ordered shots (a
    # PREFIX of the real shot order, not a stride - see the trajectory-panel fix earlier this
    # session for why a prefix is the only valid way to pick an evenly-distributed subset).
    # first_shot=20: shots 0-19 are dummy/steady-state-prep TRs with no ADC (confirmed by
    # checking adc_mask per shot), so 20 is the first real acquiring shot.
    import sys
    if girfsim_root not in sys.path:
        sys.path.insert(0, girfsim_root)
    import seq_girf as sqg
    import bacon_girf as bg

    seq_path = str(ABLATION / "spatial_physics_ablation.seq")
    sg = sqg.load_seq_gradients(seq_path)
    bacon = bg.load_bacon_girf(str(Path(girfsim_root) / "local_data" / "zenodo" / "zenodo" / "GIRFs"),
                               order=1)

    fig, _ = _new_fig(name, d)
    ax = plt.axes([0.02, 0.02, 0.96, 0.96])
    for i in range(n_shots):
        rep = sqg.trim_to_active(sqg.extract_repetition(sg, index=first_shot + i), margin_s=0.0)
        grad_mT = rep.gradients_t_per_m * 1000.0
        k_nom = rep.kspace_cyc_per_m
        dk = bg.bacon_kspace_error_batch(bacon, grad_mT.unsqueeze(0), k_nom.unsqueeze(0), rep.dt)[0]
        k_act = (k_nom + dk).numpy()
        ax.plot(k_act[0], k_act[1], '-', lw=0.7, color='#F28E2B', alpha=0.9)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(OUT / f"{name}.png", dpi=OUT_DPI, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def save_b1minus_map(name, d, nx, ny, mask):
    # B1- (receive sensitivity): Augmentrum has no separate scalar "B1-" field - receive
    # sensitivity IS the coil array's own sensitivity map. Uses Augmentrum's default 8-channel
    # Birdcage array (augmentrum.sampling.coil_sampling.Birdcage, radius=1.5, seed=0 - the exact
    # same call this session's own multi-coil CG reconstruction uses). RSS across all 8 channels
    # was tried first and is WRONG: Birdcage normalizes its array so RSS=1.0 everywhere by
    # construction (confirmed numerically - std ~5e-8, i.e. floating-point noise, which is exactly
    # the speckle artifact that showed up when stretched by percentile). Showing one representative
    # element's own magnitude instead - smooth, real 1/r-type spatial falloff, same masked/
    # coolwarm convention as scanner_b0_map/scanner_b1plus_map.
    from augmentrum.sampling.coil_sampling import Birdcage

    coil_grid = Birdcage(n_coils=8, radius=1.5).maps((nx, ny, 1), rng=np.random.default_rng(0))[:, :, 0, :]
    coil0_mag = np.abs(coil_grid[:, :, 0])  # (nx, ny)

    lo, hi = np.percentile(coil0_mag[mask > 0], [2, 98]) if np.any(mask > 0) else (float(coil0_mag.min()), float(coil0_mag.max()))
    save_sized(coil0_mag, name, "coolwarm", d, lo, hi, mask=mask)


def save_kspace_image(k_nominal, y, name, d, n_grid=220):
    # Real measured k-space signal, gridded onto a Cartesian raster (mean |signal| per bin, not
    # just sample density) - shows the actual reference-image k-space, not merely the sampling
    # trajectory/mask.
    kx, ky = k_nominal[:, 0], k_nominal[:, 1]
    mag_y = np.abs(y)
    h_sum, xedges, yedges = np.histogram2d(kx, ky, bins=n_grid, weights=mag_y)
    h_cnt, _, _ = np.histogram2d(kx, ky, bins=n_grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        img = h_sum / np.maximum(h_cnt, 1)
    img = np.log1p(img)
    img = img / (img.max() + 1e-12)
    # Real k-space energy falls off very fast with radius - a plain log leaves only a bright dot
    # at DC with everything else near-black. Gamma-stretch so the faint outer spiral structure
    # (visible in a real k-space magnitude image) actually shows instead of just the center.
    img = img ** 0.35

    fig, _ = _new_fig(name, d, facecolor="black")
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(img.T, origin="lower", cmap="gray",
             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto")
    ax.set_axis_off()
    fig.savefig(OUT / f"{name}.png", dpi=OUT_DPI, facecolor="black")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def save_gnl_warp_grid(name, d, fov_m=0.22, slice_offset_m=0.03, gnl_coefficient_scale=15.0,
                       n_lines=26, n_samples=220):
    # Real gradient-nonlinearity displacement field (same augmentrum.physics.gradient_nonlinearity
    # API and coefficient scale the ablation figure's own GNL condition uses), drawn as a warped
    # grid - matches the original mockup's style. Finer grid (26 lines, not 11) and color CODES
    # THE DISTORTION MAGNITUDE along each line segment (via a LineCollection + colormap), not an
    # arbitrary fixed color per line (which just reads as a plaid).
    from augmentrum.physics import gradient_nonlinearity as gnl
    from matplotlib.collections import LineCollection

    coeffs = gnl.load_coefficients()
    if gnl_coefficient_scale != 1.0:
        coeffs = coeffs._replace(**{
            f: getattr(coeffs, f) * gnl_coefficient_scale
            for f in coeffs._fields if f != 'R0_m'
        })
    term = gnl.GradientNonlinearityDisplacement(coeffs)

    half = fov_m / 2.0
    lines = np.linspace(-half, half, n_lines)
    t = np.linspace(-half, half, n_samples)

    segments, seg_mag = [], []

    def _add_line(pts):
        warped = term.displace(pts)
        disp = np.linalg.norm(warped[:, :2] - pts[:, :2], axis=-1)
        xy = warped[:, :2]
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        segments.append(segs)
        seg_mag.append(0.5 * (disp[:-1] + disp[1:]))

    for y0 in lines:
        _add_line(np.stack([t, np.full_like(t, y0), np.full_like(t, slice_offset_m)], axis=-1))
    for x0 in lines:
        _add_line(np.stack([np.full_like(t, x0), t, np.full_like(t, slice_offset_m)], axis=-1))

    segments = np.concatenate(segments, axis=0)
    seg_mag = np.concatenate(seg_mag, axis=0)

    fig, _ = _new_fig(name, d)
    ax = plt.axes([0.04, 0.04, 0.92, 0.92])
    # viridis: blue/purple -> turquoise -> bright yellow, matching the requested palette while
    # also being a proper sequential "magnitude" colormap (dark=small distortion, bright=large).
    lc = LineCollection(segments, array=seg_mag, cmap='viridis', linewidths=1.3)
    ax.add_collection(lc)
    ax.set_xlim(-half * 1.05, half * 1.05)
    ax.set_ylim(-half * 1.05, half * 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(OUT / f"{name}.png", dpi=OUT_DPI, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def main():
    d = dims()

    from scipy.ndimage import binary_fill_holes

    prep = np.load(ABLATION / "prepared_inputs.npz")
    anatomy, b0_hz, b1_rel = prep["anatomy"], prep["b0_hz"], prep["b1_rel"]
    # Display-only: the real intensity-threshold brain mask has small interior holes at low-signal
    # anatomy (ventricles, sinuses) - fine for the science pipeline (b0_hz/b1_rel are already
    # defined there as 0/1 respectively) but shows as stray white NaN-holes in these masked map
    # renders, so fill them for the figure.
    mask = binary_fill_holes(prep["mask"] > 0).astype(float)

    extra = np.load(ABLATION / "real_effect_results.npz", allow_pickle=True)
    k_nominal, k_girf = extra["k_nominal"], extra["k_girf"]
    image_ideal = mag(extra["image_ideal"])

    gt = mag(np.load(CG / "ground_truth.npz")["x_true"])
    y_ref = np.load(CG / "kspace_reference.npz")["y_ref"]
    if y_ref.ndim == 2:  # (n_coils, T) -> coil-combined magnitude for display
        y_ref_disp = np.sqrt((np.abs(y_ref) ** 2).sum(axis=0))
    else:
        y_ref_disp = y_ref
    recon_nominal = mag(np.load(CG / "recon_R0_nominal.npz")["image"])
    recon_full = mag(np.load(CG / "recon_R5_full.npz")["image"])

    gt_lo, gt_hi = np.percentile(gt, [1, 99])

    # --- B0: diverging, brain-masked. vmax set from the 90th-percentile |value| (not the true
    # max) so one sinus hotspot doesn't wash out the colormap - the rest of the brain gets more
    # visible color variation; the hotspot itself just saturates to the colormap's end color. ---
    b0_v = np.percentile(np.abs(b0_hz[mask > 0]), 90) if np.any(mask > 0) else np.max(np.abs(b0_hz))
    save_sized(b0_hz, "scanner_b0_map", "coolwarm", d, -b0_v, b0_v, mask=mask)
    save_sized(b0_hz, "operator_b0_map", "coolwarm", d, -b0_v, b0_v, mask=mask)

    # --- B1+: symmetric around nominal (=1); 90th-percentile deviation (not the true max, which
    # is dominated by a thin rim of exactly-0 registration-edge pixels) for visible mid-brain
    # color variation instead of a washed-out single hue. ---
    b1_dev = np.percentile(np.abs(b1_rel[mask > 0] - 1.0), 90) if np.any(mask > 0) else 0.2
    save_sized(b1_rel, "scanner_b1plus_map", "coolwarm", d, 1 - b1_dev, 1 + b1_dev, mask=mask)
    save_sized(b1_rel, "operator_b1_maps", "turbo", d, 1 - b1_dev, 1 + b1_dev, mask=mask)
    save_b1minus_map("scanner_b1minus_map", d, anatomy.shape[0], anatomy.shape[1], mask)

    # --- spatially varying phase (generic operator tile): B0 has real anatomical structure
    # (susceptibility), a much closer visual match to the original's textured rainbow brain map
    # than a smooth instrument-only field like concomitant/GNL. ---
    save_sized(b0_hz, "operator_spatial_phase_map", "turbo", d, -b0_v, b0_v, mask=mask)

    # --- concomitant: genuinely flat at 3T off-isocenter for this trajectory (see metrics.json
    # nrmse_concomitant) - not a rendering bug. Centered with a small symmetric window (rather
    # than autoscaled vmin==vmax, which matplotlib maps to the colormap's dark low end) + a
    # vivid colormap, so the flat value renders as a bright color instead of near-black. ---
    c_val = float(np.mean(extra["concomitant_phase_map"]))
    c_span = max(abs(c_val), 1e-12) * 3
    save_sized(extra["concomitant_phase_map"], "scanner_concomitant_map", "plasma", d,
              c_val - c_span, c_val + c_span)

    # --- GNL: real displacement-field warped grid, not a scalar colormap (README + user request) ---
    save_gnl_warp_grid("scanner_gnl_map", d)

    # --- anatomy: the real HCP T1w slice. digital_phantom_thumb reverts to the original (first-
    # generated) Shepp-Logan rendering per request - a quantized real-brain image "makes no
    # sense" as a stand-in for a digital phantom; Shepp-Logan is the deliberately abstract,
    # digitized test object, pulled from the untouched Shepp-Logan ablation run. ---
    save_sized(norm01(anatomy), "anatomy_thumb", "gray", d, 0, 1)
    shepp_extra = np.load(Path("outputs/spatial_physics_ablation/real_effect_results.npz"))
    digital_phantom = mag(shepp_extra["image_ideal"])
    save_sized(norm01(digital_phantom), "digital_phantom_thumb", "gray", d, 0, 1)

    # --- trajectories/GIRF: all three rebuilt directly from GIRF-Sim's own GIRFModule (shot-
    # indexed k_nominal/k_actual/dk_error), reusing its own fig1_forward_model.py plotting code/
    # conventions rather than a from-scratch reimplementation. ---
    mod = _girf_module(ABLATION, matrix=anatomy.shape[0], fov_m=float(prep["fov_m"]))
    save_trajectory_single_shot(mod, "operator_realized_trajectory", d)
    save_trajectory_undersampled(mod, "spiral_trajectory_input", d)
    save_girf_realized_flower("scanner_girf_map", d)

    # --- k-space: the real reference signal gridded onto a raster, not a sampling-mask scatter ---
    save_kspace_image(k_nominal, y_ref_disp, "operator_encoding_interface", d)
    save_kspace_image(k_nominal, y_ref_disp, "app_synthetic_kspace", d)
    save_kspace_image(k_nominal, y_ref_disp, "app_measured_kspace", d)

    # --- application panel: forward model -> CG solve -> reconstruction -> improved, all from the
    # real LinearMRIOperator/CG pipeline on real HCP anatomy; normalized against the SAME
    # ground-truth dynamic range so "improved" isn't independently auto-contrast-stretched.
    # Artifact map boosted (true residual is small relative to anatomy intensity) with the factor
    # labeled directly on the image so it's never ambiguous how much visual amplification was
    # applied. ---
    artifact_boost = 10.0
    w, h = d["app_artifact_realizations"]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=OUT_DPI, facecolor="white")
    ax = plt.axes([0, 0, 1, 1])
    vis = np.clip(np.abs(recon_full - gt) / (gt_hi - gt_lo) * artifact_boost, 0, 1)
    ax.imshow(vis, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
    ax.text(0.03, 0.03, f"×{artifact_boost:g}", transform=ax.transAxes, color="white",
           fontsize=9, ha="left", va="bottom", fontweight="bold")
    ax.set_axis_off()
    fig.savefig(OUT / "app_artifact_realizations.png", dpi=OUT_DPI, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "app_artifact_realizations.png")
    save_sized(gt, "app_reconstructed_image", "gray", d, gt_lo, gt_hi)
    save_sized(recon_nominal, "app_reconstruction_result", "gray", d, gt_lo, gt_hi)
    save_sized(recon_full, "app_improved_result", "gray", d, gt_lo, gt_hi)

    print("\nSkipped (no real data yet): tissue_contrast_maps_thumb, "
          "sequence_waveform_schematic, source_field_strength_image, target_field_strength_image, "
          "app_training_benchmark_data")


if __name__ == "__main__":
    main()
