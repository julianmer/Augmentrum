#!/usr/bin/env python3
"""
Generate the Augmentrum+ spatial-physics ablation figure.

Scientific goal
---------------
Use one common anatomy and one fully sampled 2D spiral acquisition to isolate:
    ideal, +B0, +B1+, +concomitant, +measured GIRF, +gradient nonlinearity,
    and combined effects.

The script is intentionally split into:
1) generic data preparation + plotting code that is runnable now, and
2) a small repository adapter section that Claude Code should wire to the
   current Augmentrum / GIRF-Sim APIs.

The script NEVER silently substitutes fake physics for the scientific panels.
If the repository adapter is not implemented, run with --prepare-only to
generate anatomy/maps/layout inputs and a handoff bundle.

Recommended usage after repository wiring:
    python spatial_physics_ablation.py \
        --augmentrum-root /path/to/Augmentrum \
        --girfsim-root /path/to/GIRF-Sim \
        --outdir outputs/spatial_ablation

Optional:
    --anatomy /path/to/brainweb_or_other.nii.gz
    --b0-map /path/to/b0_hz.nii.gz
    --b1-map /path/to/b1plus_relative.nii.gz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter, zoom
    from scipy.special import ive
except ImportError as e:
    raise SystemExit("Install scipy: pip install scipy") from e

try:
    import nibabel as nib
except ImportError:
    nib = None


# ----------------------------
# Configuration / data classes
# ----------------------------

CAPTION = (
    "Spatially resolved acquisition effects in Augmentrum+. A common digital brain "
    "and fully sampled 2D spiral acquisition were used to isolate B0 off-resonance, "
    "B1+ modulation, concomitant fields, measured GIRF response, and gradient "
    "nonlinearity. Each column shows the input map or trajectory perturbation, "
    "reconstructed image, and matched difference from the ideal acquisition. "
    "The final column combines all effects to illustrate scanner-dependent spatial variability."
)

EFFECT_KEYS = ["ideal", "b0", "b1", "concomitant", "girf", "gnl", "combined"]
EFFECT_TITLES = [
    "Ideal",
    r"$B_0$",
    r"$B_1^+$",
    "Concomitant",
    "GIRF",
    "Gradient\nnonlinearity",
    "Combined",
]


@dataclass
class FigureInputs:
    anatomy: np.ndarray
    mask: np.ndarray
    b0_hz: np.ndarray
    b1_rel: np.ndarray
    fov_m: float
    source_anatomy: str
    source_b0: str
    source_b1: str


@dataclass
class ForwardResult:
    images: Dict[str, np.ndarray]
    extras: Dict[str, np.ndarray]
    metadata: Dict[str, object]


# ----------------------------
# Generic utilities
# ----------------------------

def robust_norm(x: np.ndarray, pct: float = 99.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x)
    lo = np.percentile(x, 0.5)
    hi = np.percentile(x, pct)
    if hi <= lo:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    n = min(h, w)
    y0 = (h - n) // 2
    x0 = (w - n) // 2
    return img[y0:y0+n, x0:x0+n]


def resize_2d(img: np.ndarray, n: int = 192, order: int = 1) -> np.ndarray:
    img = np.asarray(img)
    fy = n / img.shape[0]
    fx = n / img.shape[1]
    return zoom(img, (fy, fx), order=order)


def load_nifti_slice(path: Path, target_n: int = 192) -> np.ndarray:
    if nib is None:
        raise RuntimeError("nibabel is required for NIfTI input: pip install nibabel")
    vol = np.asarray(nib.load(str(path)).get_fdata())
    vol = np.squeeze(vol)
    if vol.ndim == 2:
        sl = vol
    elif vol.ndim == 3:
        # choose axial slice with the largest nonzero/high-signal area
        scores = []
        for z in range(vol.shape[2]):
            s = vol[:, :, z]
            scores.append(np.count_nonzero(s > np.percentile(vol, 55)))
        sl = vol[:, :, int(np.argmax(scores))]
    else:
        raise ValueError(f"Expected 2D or 3D NIfTI, got shape {vol.shape}")
    sl = np.rot90(sl)
    sl = center_crop_square(sl)
    sl = resize_2d(sl, target_n)
    return robust_norm(sl)


# Shepp-Logan geometry/intensities (Kak & Slaney modified/Toft contrast) - a clean synthetic
# anatomy with no baked-in scanner artifacts, which is the property this figure actually needs
# (see ANATOMY_SOURCE below): a real scanned T1w already carries whatever real B1+/B0 that
# scanner produced, which would contaminate the B0/B1+ panels this script exists to isolate.
_SHEPP_LOGAN_GEOMETRY = (
    ( 0.00,  0.0000, 0.6900, 0.920,   0.0),
    ( 0.00, -0.0184, 0.6624, 0.874,   0.0),
    ( 0.22,  0.0000, 0.1100, 0.310, -18.0),
    (-0.22,  0.0000, 0.1600, 0.410,  18.0),
    ( 0.00,  0.3500, 0.2100, 0.250,   0.0),
    ( 0.00,  0.1000, 0.0460, 0.046,   0.0),
    ( 0.00, -0.1000, 0.0460, 0.046,   0.0),
    (-0.08, -0.6050, 0.0460, 0.023,   0.0),
    ( 0.00, -0.6050, 0.0230, 0.023,   0.0),
    ( 0.06, -0.6050, 0.0230, 0.046,   0.0),
)
_SHEPP_LOGAN_INTENSITIES = (1.00, -0.80, -0.20, -0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10)


def load_shepp_logan_anatomy(target_n: int = 192) -> Tuple[np.ndarray, str]:
    """The default anatomy: no download, no license to check, no baked-in scanner artifacts."""
    ax = np.linspace(-1.0, 1.0, target_n)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    img = np.zeros((target_n, target_n), dtype=np.float32)
    for (x0, y0, a, b, angle_deg), value in zip(_SHEPP_LOGAN_GEOMETRY, _SHEPP_LOGAN_INTENSITIES):
        t = np.deg2rad(angle_deg)
        xr = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
        yr = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += value
    img = np.clip(img, 0.0, None)
    return robust_norm(img), "Shepp-Logan phantom (modified/Toft contrast) - clean synthetic anatomy"


_SHEPP_LOGAN_3D_CACHE = Path(__file__).resolve().parent / "data" / "shepp_logan_3d_ellipsoids.npz"


def build_shepp_logan_3d_ellipsoids(size: int = 96, cache_path: Path = _SHEPP_LOGAN_3D_CACHE,
                                    force: bool = False) -> np.ndarray:
    """
    The 2-D Shepp-Logan phantom's own 10 ellipses, extended into 3-D ellipsoids - a genuinely
    3-D, still-synthetic (no baked-in scanner artifacts) phantom for off-isocenter/3-D-trajectory
    work (needed for a spatially-structured concomitant/GNL demonstration - see
    SPATIAL_PHYSICS_FIGURE_STATUS.md; a flat 2-D slice, even off-isocenter, cannot show one).

    Each ellipse already specifies two real measurements - its in-plane semi-axes "(a, b)" from
    "_SHEPP_LOGAN_GEOMETRY" ("width"/"length"). The third (through-slice) semi-axis isn't
    measured by a 2-D phantom at all, so it has to be chosen, not derived: this uses the
    geometric mean "c = sqrt(a*b)" - the one choice that actually earns the name "deformed
    sphere" (c == a == b, a true sphere, exactly when the in-plane ellipse is already a circle;
    partial Kak & Slaney's own hand-tuned "SHEPP_LOGAN_3D" table, which assigns an independently
    chosen c has no 2-D measurement to derive from). Centered at z=0; in-plane (x0, y0) and
    rotation reused unchanged from the 2-D table.

    Cached to "cache_path" (an .npz with "volume" and the geometry/size used to build it) so this
    is built once and reused, not regenerated per run - pass "force=True" to rebuild (e.g. after
    changing "size").
    """
    if cache_path.exists() and not force:
        data = np.load(cache_path)
        if int(data["size"]) == size:
            return data["volume"]

    ax = np.linspace(-1.0, 1.0, size)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    vol = np.zeros((size, size, size), dtype=np.float32)
    for (x0, y0, a, b, angle_deg), value in zip(_SHEPP_LOGAN_GEOMETRY, _SHEPP_LOGAN_INTENSITIES):
        c = float(np.sqrt(a * b))
        t = np.deg2rad(angle_deg)
        xr = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
        yr = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
        zr = zz
        vol[(xr / a) ** 2 + (yr / b) ** 2 + (zr / c) ** 2 <= 1.0] += value
    vol = np.clip(vol, 0.0, None)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, volume=vol, size=size)
    return vol


def load_brainweb_anatomy(target_n: int = 192, subject: int = 4) -> Tuple[np.ndarray, str]:
    """
    BrainWeb anatomy via the "brainweb" pip package.

    Not the default (see ANATOMY_SOURCE) - BrainWeb's own server
    (brainweb.bic.mni.mcgill.ca, which this package's own brainweb.LINKS points at too) has been
    unreachable from this environment every time it's been tried; this path exists so swapping in
    a working BrainWeb fetch (or a manually-downloaded file via --anatomy) later is a one-flag
    change (--anatomy-source brainweb), not a code change.
    """
    try:
        import brainweb
    except ImportError as e:
        raise RuntimeError("pip install brainweb") from e

    files = brainweb.get_files(".", subjects=[subject])
    vol = brainweb.load_file(files[0])   # label volume; see brainweb docs for T1-weighted synth
    if vol.ndim == 3:
        scores = [np.count_nonzero(vol[z] > 0) for z in range(vol.shape[0])]
        img = vol[int(np.argmax(scores))]
    else:
        img = vol
    img = center_crop_square(np.asarray(img, dtype=np.float32))
    img = resize_2d(img, target_n)
    return robust_norm(img), f"BrainWeb subject {subject:02d}"


def make_mask(anatomy: np.ndarray) -> np.ndarray:
    sm = gaussian_filter(anatomy, 1.5)
    thresh = max(0.05, np.percentile(sm[sm > 0], 12) if np.any(sm > 0) else 0.05)
    mask = sm > thresh
    mask = gaussian_filter(mask.astype(float), 1.0) > 0.45
    return mask.astype(float)


def synthetic_b0_map(mask: np.ndarray, max_abs_hz: float = 90.0, seed: int = 17) -> np.ndarray:
    """
    Representative synthetic B0 map for development/figure prototyping only.

    This is not a susceptibility simulation. It creates a smooth field with
    anterior/inferior lobes and weak low-frequency variation so the B0 forward
    model can be exercised before a measured/validated field map is supplied.
    """
    rng = np.random.default_rng(seed)
    n = mask.shape[0]
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]

    field = (
        0.65 * np.exp(-((xx - 0.30)**2 / 0.05 + (yy + 0.55)**2 / 0.04))
        -0.70 * np.exp(-((xx + 0.28)**2 / 0.06 + (yy + 0.52)**2 / 0.05))
        +0.18 * xx
        -0.10 * yy
    )
    noise = gaussian_filter(rng.normal(size=mask.shape), sigma=n/18)
    noise /= (np.max(np.abs(noise)) + 1e-12)
    field += 0.12 * noise

    field *= mask
    denom = np.max(np.abs(field)) + 1e-12
    return max_abs_hz * field / denom


def synthetic_b1_map(mask: np.ndarray) -> np.ndarray:
    """
    Representative relative B1+ modulation map for development only.

    Values are dimensionless B1+/nominal-B1. Replace with a measured or
    validated field map for the final figure if available.
    """
    n = mask.shape[0]
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    b1 = (
        0.92
        + 0.22 * np.exp(-((xx - 0.35)**2 + (yy + 0.05)**2) / 0.35)
        - 0.18 * np.exp(-((xx + 0.45)**2 + (yy - 0.10)**2) / 0.25)
        + 0.08 * xx
    )
    b1 = gaussian_filter(b1, 5)
    b1 = np.clip(b1, 0.65, 1.25)
    return 1.0 + (b1 - 1.0) * mask


# ----------------------------
# Real HCP-derived B0/B1+ maps
# ----------------------------
# Real data, confirmed this session (see scripts/SPATIAL_PHYSICS_FIGURE_STATUS.md for the full
# provenance/citations) - not measured for THIS anatomy (Shepp-Logan, or whichever subject
# BrainWeb eventually supplies), so these are real field maps applied to a different, clean
# object, not a subject-matched acquisition. Both conversions are one-time arithmetic into the
# units the generic --b0-map/--b1-map NIfTI loaders already expect (Hz, and B1+/nominal), cached
# under the run's own outdir so repeat runs don't redo the I/O.
_HCP_B0_FIELDMAP_PHASE = (
    "/run/media/johnlamaster/data/datasets/MRSI_Challenge_Data/dev/connectome/Data/"
    "105923_3T/scans/108_FieldMap/NIFTI/105923_3T_FieldMap_Phase.nii.gz"
)
_HCP_B0_FIELDMAP_MAGNITUDE = (
    "/run/media/johnlamaster/data/datasets/MRSI_Challenge_Data/dev/connectome/Data/"
    "105923_3T/scans/107_FieldMap/NIFTI/105923_3T_FieldMap_Magnitude.nii.gz"
)
# HCP Young Adult 3T GRE field map: TE1=4.92ms (HCP_S1200_Release_Appendix_I.pdf scanner
# protocol card), deltaTE=2.46ms (HCPpipelines/Examples/Scripts/PreFreeSurferPipelineBatch.sh,
# documented for 3T). Phase is the standard Siemens 12-bit phase-difference convention
# (empirically confirmed: raw values run exactly -4096..4092 in steps of 2 = -pi..pi).
_HCP_B0_DELTA_TE_S = 2.46e-3

_HCP_B1_RAFI = (
    "/run/media/johnlamaster/data/datasets/hcp/100408_StructuralRecommended/"
    "100408/T1w/rAFI.nii.gz"
)
# Already an HCP-computed actual-flip-angle map, in degrees (not raw AFI - no TR1/TR2 ratio
# needed). Nominal flip angle 50 deg, from the same HCP Appendix I AFI scanner protocol card
# (NOT AFI_stats.txt's "45.002", which disagreed with the protocol card and wasn't trusted).
_HCP_B1_NOMINAL_FLIP_DEG = 50.0

# Each subject's OWN T1w, used only as the rigid-registration target to bring that subject's own
# field map into the phantom's coordinate frame (see _rigid_register_t1w_to_phantom). Assumes
# rAFI.nii.gz/the field-map phase+magnitude are already on the same native grid as their own
# subject's T1w (true for rAFI.nii.gz by HCP's own "r" = resampled-to-T1w naming convention; true
# for 105923's field map only to the extent that it was acquired in the same session/scanner
# frame as its own T1w - not independently re-verified beyond that being the standard assumption).
_HCP_B0_T1W = (
    "/run/media/johnlamaster/data/datasets/MRSI_Challenge_Data/dev/connectome/Data/"
    "105923_3T/scans/103_T1w/NIFTI/105923_3T_T1w_MPR1.nii.gz"
)
_HCP_B1_T1W = (
    "/run/media/johnlamaster/data/datasets/hcp/100408_StructuralRecommended/"
    "100408/T1w/T1w_acpc_dc_restore_brain.nii.gz"
)

_HCP_B0_ACQUISITION_FIELD_T = 3.0   # the field strength 105923's own GRE field map was acquired at

_REGISTRATION_CACHE_DIR = Path(__file__).resolve().parent / "data" / "registration_cache"
_PHANTOM_3D_NIFTI_CACHE = Path(__file__).resolve().parent / "data" / "shepp_logan_3d_phantom.nii.gz"


def _phantom_3d_nifti(fov_mm: float, size: int = 96,
                      cache_path: Path = _PHANTOM_3D_NIFTI_CACHE) -> Path:
    """
    "build_shepp_logan_3d_ellipsoids" written out as a NIfTI with a real physical affine
    (isotropic "fov_mm/size" mm voxels, centered on the origin) so ANTs can use it as a
    registration target.

    At z=0 this volume's own axial slice is EXACTLY the 2-D Shepp-Logan phantom already used
    everywhere else in this script: every one of the 3-D phantom's 10 ellipsoids reduces to its
    original 2-D ellipse at z=0 (the "c=sqrt(a*b)" through-slice semi-axis only changes the shape
    away from z=0). So a subject registered into this volume's space, then re-sliced at its own
    z=0 (which is exactly what "load_or_prepare_inputs"'s existing "raw[:, :, raw.shape[2]//2]"
    mid-slice logic already picks, given this affine centers the volume symmetrically) lands
    registered against the SAME anatomy the rest of the figure shows - no changes needed to that
    downstream loading code.
    """
    if cache_path.exists():
        existing = nib.load(str(cache_path))
        if existing.shape == (size, size, size) and abs(existing.header.get_zooms()[0] - fov_mm / size) < 1e-6:
            return cache_path

    vol = build_shepp_logan_3d_ellipsoids(size=size)
    voxel_mm = fov_mm / size
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0]).astype(np.float64)
    affine[:3, 3] = -0.5 * fov_mm   # centers the volume at the origin, so index size//2 is z=0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), str(cache_path))
    return cache_path


def _rigid_register_t1w_to_phantom(t1w_path: str, phantom_nifti_path: Path, label: str) -> list:
    """
    Rigid-only (6 DOF: rotation + translation, no scaling/shearing/deformation) registration of a
    subject's own T1w onto the phantom via ANTs.

    Rigid, not deformable/SyN: the phantom's own hand-specified ellipse geometry has no real brain
    texture for a deformable algorithm to anatomically lock onto - a SyN warp trying to match real
    T1w gyral/sulcal intensity to an abstract 10-ellipse phantom has no well-posed target to
    converge on. A rigid transform only aligns position/orientation/center, which stays well-posed
    regardless of what the fixed image's own intensities represent.

    Cached under scripts/data/registration_cache/ (keyed by label + phantom size, NOT by the
    run's own --outdir) since a full 3-D rigid T1w registration is real work that repeated runs
    (different outdir, different --field-strength-t) shouldn't have to redo.
    """
    import ants
    _REGISTRATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    outprefix = str(_REGISTRATION_CACHE_DIR / f"{label}_rigid_to_phantom_")
    cached_mat = Path(outprefix + "0GenericAffine.mat")
    if cached_mat.exists():
        return [str(cached_mat)]

    t1w = ants.image_read(str(t1w_path))
    phantom = ants.image_read(str(phantom_nifti_path))
    reg = ants.registration(fixed=phantom, moving=t1w, type_of_transform="Rigid",
                            outprefix=outprefix)
    return reg["fwdtransforms"]


def _apply_rigid_transform(map_path: str, phantom_nifti_path: Path, transformlist: list,
                           interpolator: str = "linear"):
    """Applies a subject's own T1w->phantom rigid transform to one of that subject's own field
    maps - see _HCP_B0_T1W/_HCP_B1_T1W's docstring comment for the same-native-space assumption
    this relies on. Returns an in-memory ANTsImage (caller writes/converts as needed)."""
    import ants
    moving = ants.image_read(str(map_path))
    phantom = ants.image_read(str(phantom_nifti_path))
    return ants.apply_transforms(fixed=phantom, moving=moving, transformlist=transformlist,
                                 interpolator=interpolator)


def _derive_hcp_b0_map(cache_path: Path, field_strength_t: float = _HCP_B0_ACQUISITION_FIELD_T,
                       fov_mm: float = 220.0, phantom_size: int = 96) -> Path:
    """
    Real HCP 105923_3T GRE field map -> rigidly registered onto the phantom -> Hz, written as a
    NIfTI the generic loader can read.

    The field map's own native-space phase/magnitude are first rigidly registered into the
    phantom's own coordinate frame (via 105923's own T1w - see _rigid_register_t1w_to_phantom and
    _HCP_B0_T1W's docstring for the same-native-space assumption this relies on) - previously
    this just took the raw native-space slice and hoped a generic center-crop/resize happened to
    land it roughly where the phantom's own anatomy was, with no real correspondence guaranteed.

    "field_strength_t" rescales the *same* real shim/susceptibility residual (a field-independent
    ppm pattern) to a different target field strength: off-resonance in Hz is
    "ppm * gamma * B0", so "Hz_target = Hz_measured * (field_strength_t / 3.0)" - e.g. simulating
    what this real residual would look like at 55 mT rather than claiming a measured 55 mT map.
    """
    if cache_path.exists():
        return cache_path
    phantom_nifti = _phantom_3d_nifti(fov_mm, size=phantom_size)
    transformlist = _rigid_register_t1w_to_phantom(_HCP_B0_T1W, phantom_nifti, label="b0_105923")
    # The magnitude file is 2 echoes (4-D); ANTs refuses to apply_transforms on a non-3-D volume
    # without being told it's a time series ("imagetype"), which this isn't - extract the first
    # echo to its own 3-D NIfTI first rather than fight that flag.
    mag_3d_path = _REGISTRATION_CACHE_DIR / "b0_105923_magnitude_echo1.nii.gz"
    if not mag_3d_path.exists():
        mag_img = nib.load(_HCP_B0_FIELDMAP_MAGNITUDE)
        mag_echo1 = np.asarray(mag_img.dataobj)[..., 0]
        nib.save(nib.Nifti1Image(mag_echo1, mag_img.affine), str(mag_3d_path))

    phase_reg = _apply_rigid_transform(_HCP_B0_FIELDMAP_PHASE, phantom_nifti, transformlist)
    mag_reg = _apply_rigid_transform(str(mag_3d_path), phantom_nifti, transformlist)

    phase_raw = phase_reg.numpy().astype(np.float64)
    mag = mag_reg.numpy().astype(np.float64)

    phase_rad = phase_raw * (np.pi / 4096.0)
    b0_hz = phase_rad / (2.0 * np.pi * _HCP_B0_DELTA_TE_S)
    b0_hz *= field_strength_t / _HCP_B0_ACQUISITION_FIELD_T
    mask = mag > np.percentile(mag[mag > 0], 20.0) if np.any(mag > 0) else np.ones_like(mag, dtype=bool)
    b0_hz = (b0_hz * mask).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(b0_hz, nib.load(str(phantom_nifti)).affine), str(cache_path))
    return cache_path


def _derive_hcp_anatomy_map(cache_path: Path, fov_mm: float = 220.0, phantom_size: int = 96) -> Path:
    """Real HCP 105923 T1w, rigidly registered onto the SAME phantom grid as that subject's own
    B0 map (reuses the "b0_105923" registration cache - same transform, same subject) so
    anatomy_source='hcp' stays pixel-aligned with the B0/B1+ maps instead of loading independently
    like a bare --anatomy <path> would."""
    if cache_path.exists():
        return cache_path
    phantom_nifti = _phantom_3d_nifti(fov_mm, size=phantom_size)
    transformlist = _rigid_register_t1w_to_phantom(_HCP_B0_T1W, phantom_nifti, label="b0_105923")
    t1w_reg = _apply_rigid_transform(_HCP_B0_T1W, phantom_nifti, transformlist)
    vol = robust_norm(t1w_reg.numpy().astype(np.float64)).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, nib.load(str(phantom_nifti)).affine), str(cache_path))
    return cache_path


def _derive_hcp_b1_map(cache_path: Path, fov_mm: float = 220.0, phantom_size: int = 96) -> Path:
    """Real HCP 100408 actual-flip-angle map -> rigidly registered onto the phantom (via 100408's
    own T1w) -> relative B1+, written as a NIfTI. See _derive_hcp_b0_map's docstring for why
    registration replaces the earlier raw-slice-and-hope approach."""
    if cache_path.exists():
        return cache_path
    phantom_nifti = _phantom_3d_nifti(fov_mm, size=phantom_size)
    transformlist = _rigid_register_t1w_to_phantom(_HCP_B1_T1W, phantom_nifti, label="b1_100408")
    rafi_reg = _apply_rigid_transform(_HCP_B1_RAFI, phantom_nifti, transformlist)

    b1_rel = (rafi_reg.numpy().astype(np.float64) / _HCP_B1_NOMINAL_FLIP_DEG).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(b1_rel, nib.load(str(phantom_nifti)).affine), str(cache_path))
    return cache_path


def _resolve_map_path(requested: Optional[str], derive_fn, cache_path: Path, label: str) -> Optional[str]:
    """"auto" derives the real HCP map (falling back to synthetic with a warning if that fails);
    "synthetic"/"none" explicitly disables it; any other value is an explicit path, used as-is."""
    if requested in (None, "synthetic", "none"):
        return None
    if requested != "auto":
        return requested
    try:
        return str(derive_fn(cache_path))
    except Exception as exc:
        warnings.warn(f"Could not derive the real HCP {label} map ({exc}); using synthetic instead.")
        return None


def _rotate90_ccw(arr: np.ndarray, n: int) -> np.ndarray:
    """
    "n" rotations of 90 degrees counterclockwise, as displayed by this script's own
    "_imshow_clean(..., origin='lower')" convention.

    Empirically verified (not assumed) before use: under "origin='lower'", "np.rot90(arr, k=1)"
    actually appears CLOCKWISE (N-arm moves to E), so a visual CCW turn needs "k=-1" per step.
    """
    return np.ascontiguousarray(np.rot90(arr, k=-n)) if n else arr


def load_or_prepare_inputs(args) -> FigureInputs:
    # Single source of truth for which axial slice of the (shared) phantom-registered grid
    # anatomy/B0/B1+ all come from - anatomy, B0, and B1+ are ALL registered onto the SAME 3-D
    # phantom grid (_phantom_3d_nifti) regardless of anatomy_source, so they MUST use the same
    # z-index or they silently show different, misaligned slices of the brain.
    hcp_z_offset = getattr(args, "hcp_slice_offset_vox", 0)

    def _hcp_slice(raw3d: np.ndarray) -> np.ndarray:
        z = int(np.clip(raw3d.shape[2] // 2 + hcp_z_offset, 0, raw3d.shape[2] - 1))
        return raw3d[:, :, z]

    # ANATOMY_SOURCE: the one flag to flip once a BrainWeb volume is reachable.
    # --anatomy <path> (an already-downloaded file) always wins if given; otherwise
    # --anatomy-source picks the generator. Default is Shepp-Logan - no download, no
    # license to check. "hcp" uses 105923's own real T1w, rigidly registered onto the
    # SAME phantom grid its own B0 map already uses (see _derive_hcp_anatomy_map) so
    # anatomy/B0/B1+ all stay pixel-aligned - unlike a bare --anatomy <path>, which
    # loads independently via load_nifti_slice and would NOT line up with the
    # phantom-registered B0/B1+ maps.
    if args.anatomy:
        anatomy = load_nifti_slice(Path(args.anatomy), args.matrix)
        anatomy_source = str(Path(args.anatomy).resolve())
    elif args.anatomy_source == "brainweb":
        anatomy, anatomy_source = load_brainweb_anatomy(args.matrix, args.brainweb_subject)
    elif args.anatomy_source == "hcp":
        anat_path = _resolve_map_path(
            "auto", lambda p: _derive_hcp_anatomy_map(p, args.fov_mm),
            Path(args.outdir) / "hcp_derived" / "anatomy_105923.nii.gz", "Anatomy")
        raw = np.asarray(nib.load(anat_path).get_fdata()).squeeze()
        # --hcp-slice-offset-vox (via _hcp_slice/hcp_z_offset above) picks which axial index is
        # shown - independent of --slice-offset-mm, which is a separate physics parameter
        # (off-isocenter position fed to the concomitant/GNL terms).
        raw = _hcp_slice(raw)
        raw = center_crop_square(raw)
        anatomy = robust_norm(resize_2d(raw, args.matrix)).astype(np.float32)
        anatomy_source = f"HCP 105923 T1w (rigid-registered onto phantom grid): {_HCP_B0_T1W}"
    else:
        anatomy, anatomy_source = load_shepp_logan_anatomy(args.matrix)

    # Rotation is applied once, at the very end, to anatomy/mask/b0_hz/b1_rel together - doing it
    # here (before b0_hz/b1_rel are masked against "mask" below) would misalign an
    # already-rotated mask against still-unrotated maps.
    mask = make_mask(anatomy)

    outdir = Path(args.outdir)
    b0_derive = lambda cache_path: _derive_hcp_b0_map(cache_path, args.field_strength_t, args.fov_mm)
    b1_derive = lambda cache_path: _derive_hcp_b1_map(cache_path, args.fov_mm)
    args.b0_map = _resolve_map_path(
        args.b0_map, b0_derive,
        outdir / "hcp_derived" / f"b0_hz_105923_{args.field_strength_t:g}T.nii.gz", "B0")
    args.b1_map = _resolve_map_path(args.b1_map, b1_derive,
                                    outdir / "hcp_derived" / "b1_rel_100408.nii.gz", "B1+")

    if args.b0_map:
        # Preserve sign/units rather than robust-normalizing a quantitative NIfTI map.
        raw = np.asarray(nib.load(args.b0_map).get_fdata()).squeeze()
        if raw.ndim == 3:
            raw = _hcp_slice(raw)
        # NO blind np.rot90 here (there used to be one) - this array is now the rigid-registered,
        # phantom-aligned map produced by _derive_hcp_b0_map, not an arbitrary raw native-space
        # NIfTI slice, so it is already in the phantom's own array orientation (confirmed
        # empirically: the registered T1w's own aspect ratio/anatomy only looks correct without
        # an extra rotation - see SPATIAL_PHYSICS_FIGURE_STATUS.md). The old unconditional rot90
        # was a leftover from before the registration pipeline existed and was actively WRONG once
        # registration was added - any orientation correction now belongs solely in the single,
        # user-controlled --rotate90-ccw applied uniformly to anatomy/mask/b0/b1 together below.
        raw = center_crop_square(raw)
        b0_hz = resize_2d(raw, args.matrix)
        b0_hz *= mask
        b0_source = str(Path(args.b0_map).resolve())
    else:
        b0_hz = synthetic_b0_map(mask, args.synthetic_b0_hz, args.seed)
        b0_source = "representative synthetic B0 map (replace for final figure if desired)"

    if args.b1_map:
        raw = np.asarray(nib.load(args.b1_map).get_fdata()).squeeze()
        if raw.ndim == 3:
            raw = _hcp_slice(raw)
        # See the matching comment in the --b0-map branch above - same reasoning applies here.
        raw = center_crop_square(raw)
        b1_rel = resize_2d(raw, args.matrix)
        med = np.median(b1_rel[mask > 0]) if np.any(mask > 0) else 1.0
        if med != 0:
            b1_rel = b1_rel / med
        b1_rel = np.where(mask > 0, b1_rel, 1.0)
        b1_source = str(Path(args.b1_map).resolve())
    else:
        b1_rel = synthetic_b1_map(mask)
        b1_source = "representative synthetic relative B1+ map (replace for final figure if desired)"

    if args.rotate90_ccw:
        # Applied last, uniformly, to every spatial array - after b0_hz/b1_rel's own masking/
        # normalization above (which must happen against the still-unrotated mask they were
        # actually computed against), so anatomy/mask/b0_hz/b1_rel all stay mutually registered.
        anatomy = _rotate90_ccw(anatomy, args.rotate90_ccw)
        mask = _rotate90_ccw(mask, args.rotate90_ccw)
        b0_hz = _rotate90_ccw(b0_hz, args.rotate90_ccw)
        b1_rel = _rotate90_ccw(b1_rel, args.rotate90_ccw)

    return FigureInputs(
        anatomy=anatomy.astype(np.float32),
        mask=mask.astype(np.float32),
        b0_hz=b0_hz.astype(np.float32),
        b1_rel=b1_rel.astype(np.float32),
        fov_m=float(args.fov_mm) / 1000.0,
        source_anatomy=anatomy_source,
        source_b0=b0_source,
        source_b1=b1_source,
    )


def save_inputs(inputs: FigureInputs, outdir: Path):
    np.savez_compressed(
        outdir / "prepared_inputs.npz",
        anatomy=inputs.anatomy,
        mask=inputs.mask,
        b0_hz=inputs.b0_hz,
        b1_rel=inputs.b1_rel,
        fov_m=np.array(inputs.fov_m),
    )
    with open(outdir / "prepared_inputs_metadata.json", "w") as f:
        json.dump({
            "source_anatomy": inputs.source_anatomy,
            "source_b0": inputs.source_b0,
            "source_b1": inputs.source_b1,
            "fov_m": inputs.fov_m,
        }, f, indent=2)


# ----------------------------
# Repository adapter
# ----------------------------

def _patch_girf_sim_for_device(gmx):
    """
    Monkey-patches GIRF-Sim's "ForwardOperator"/"SpectralPhantom"/"CoilSensitivity" IN THIS
    PROCESS ONLY - never touches the GIRF-Sim checkout on disk - so the exact-NUDFT encode (this
    script's dominant cost: O(N_positions * T_samples) per condition) can actually run on GPU.

    Why this is needed: GIRF-Sim's own "ForwardOperator.forward"/"total_phase" and
    "SpectralPhantom.single_component"/"CoilSensitivity.get" create several internal tensors
    ("t", "phi", "d", "t2star_q_s", "freq_q_hz", the default coil map) with no "device=" argument
    at all, i.e. hardcoded to CPU - regardless of what device "positions"/"k_nominal"/"rho" are
    on. Passing CUDA tensors into the unpatched classes does not run slow on GPU, it crashes
    (PyTorch refuses to mix CPU and CUDA tensors in one op) the moment a phase term or the
    default coil touches them. This patch creates those same internal tensors on the REFERENCE
    tensor's own device instead - same formulas, unchanged otherwise.
    """
    import torch

    def total_phase(self, t):
        # Deliberately CPU, regardless of self.positions'/t's own device - see forward()'s own
        # docstring for why: some phase terms (GradientNonlinearityPhase) refuse a sub-chunked
        # "t" outright (it validates "t.shape[0] == k_traj.shape[1]"), so the full (N, T) phi
        # tensor can never be avoided - keeping it on the CPU's much larger RAM budget instead of
        # scarce GPU VRAM is what actually avoids the OOM, not the chunking below (which only
        # bounds the GPU-resident slice at any one time).
        r = self.effective_positions()
        r_cpu = r.cpu() if hasattr(r, 'cpu') else r
        t_cpu = t.cpu() if hasattr(t, 'cpu') else t
        n = self.positions.shape[0]
        phi = torch.zeros(n, t_cpu.numel(), dtype=torch.float32)
        for term in self.phase_terms:
            phi = phi + term.phase(r_cpu, t_cpu).cpu()
        return phi

    def forward(self, phantom, dt, noise_std=0.0, chunk_size=4096):
        # Chunked over T throughout, not just the NUDFT step: GIRF-Sim's own "waveform" materializes
        # the FULL (N, T) tensor up front (no chunk_size awareness at all), which at this script's
        # T~45000+ samples is multiple GB per tensor - enough to OOM a 12GB card well before the
        # (correctly chunked) NUDFT step even runs. Extending the same chunking GIRF-Sim already
        # applies to the encode step to the waveform construction too keeps only one chunk's worth
        # of (N, chunk_size) tensors alive on the GPU at a time - mathematically identical result,
        # bounded GPU peak memory regardless of T.
        #
        # "total_phase" itself is a SEPARATE memory problem from the above: at least one phase
        # term (GradientNonlinearityPhase) requires its own full-length (N, T) output and refuses
        # a chunked "t" (see its own "t.shape[0] == k_traj.shape[1]" check) - there is no way to
        # avoid materializing the whole thing. The fix is "total_phase" computing and returning it
        # on CPU (ample system RAM, e.g. ~2GB for N=6400/T=91000 float32) instead of GPU VRAM; only
        # each chunk's own slice is ever uploaded to the GPU below, so GPU memory stays bounded by
        # chunk_size even though the real-valued phase buffer itself is not. Discovered via: a
        # doubled shot count (200 shots, T~91000) OOM'd even with the GPU-device patch already in
        # place and the complex waveform/signal construction already chunked - the full-length
        # phi tensor living on the GPU was the remaining non-chunked allocation.
        device = self.k_nominal.device
        T = self.k_nominal.shape[-1]
        t_full = torch.arange(T, dtype=torch.float32, device=device) * dt
        k_actual = gmx.apply_girf_trajectory_error(self.k_nominal, self.dk_girf)
        r_eff = self.effective_positions()
        C = self.coil.get(self.positions.shape[0])[0].to(device)

        phi_cpu = self.total_phase(t_full)   # (N, T) float32, CPU - see total_phase's own comment

        d = torch.zeros(T, dtype=torch.complex64, device=device)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            t_chunk = t_full[start:end]
            phi_chunk = phi_cpu[:, start:end].to(device)
            M_chunk = phantom.waveform(t_chunk)
            signal_chunk = C.reshape(-1, 1).to(M_chunk.dtype) * M_chunk \
                * torch.exp(1j * phi_chunk.to(M_chunk.dtype))
            E_chunk = gmx.nudft_encode_matrix(k_actual[:, start:end], r_eff)
            d[start:end] = (E_chunk * signal_chunk.transpose(0, 1)).sum(dim=1)

        if noise_std > 0:
            noise = torch.randn_like(d.real) + 1j * torch.randn_like(d.real)
            d = d + (noise_std / math.sqrt(2.0)) * noise

        return gmx.JointKTMRSIResult(d=d, k_nominal=self.k_nominal, k_actual=k_actual,
                                     phase_ho=phi_cpu, t=t_full)

    @staticmethod
    def single_component(rho, t2star_s: float, freq_hz: float = 0.0):
        n = rho.shape[0]
        device = rho.device
        return gmx.SpectralPhantom(
            rho_q=rho.reshape(n, 1).to(torch.complex64),
            t2star_q_s=torch.full((n, 1), float(t2star_s), device=device),
            freq_q_hz=torch.full((n, 1), float(freq_hz), device=device),
        )

    def coil_get(self, n_positions: int):
        if self.maps is not None:
            return self.maps
        return torch.ones(1, n_positions, dtype=torch.complex64)   # device fixed up by caller

    gmx.ForwardOperator.total_phase = total_phase
    gmx.ForwardOperator.forward = forward
    gmx.SpectralPhantom.single_component = single_component
    gmx.CoilSensitivity.get = coil_get


class _MultiShotConcomitantPhase:
    """
    phi_Maxwell(r,t) across a flattened multi-shot trajectory.

    "augmentrum.physics.concomitant_field.ConcomitantFieldPhase" assumes one shot's gradient
    waveform and accumulates phase (via a cumulative sum) over its own "t" with no shot-boundary
    resets - correct for one shot, wrong for several concatenated ones (phase would wrongly keep
    accumulating across TRs that each actually start fresh after their own RF excitation). This
    composes the SAME real primitives (`concomitant_field_coefficients`, `concomitant_field_basis`
    - no physics reimplemented) the way "GIRFArtifacts._severity_coefficients" already does:
    accumulate each shot's own coefficients separately, then concatenate along time - never
    the raw gradients.
    """
    name = 'concomitant_maxwell'

    def __init__(self, gradients_per_shot, dt: float, b0_tesla: float):
        from augmentrum.physics.concomitant_field import concomitant_field_coefficients
        self.coeffs = np.concatenate(
            [concomitant_field_coefficients(np.asarray(g), dt, b0_tesla)
             for g in gradients_per_shot], axis=-1)   # (4, T_total)

    def phase(self, positions, t):
        import torch
        from augmentrum.physics.concomitant_field import concomitant_field_basis
        positions_np = (positions.detach().cpu().numpy() if hasattr(positions, 'detach')
                       else np.asarray(positions, dtype=np.float64))
        basis = concomitant_field_basis(positions_np)       # (N, 4)
        phi = basis @ self.coeffs                             # (N, T_total)
        device = positions.device if hasattr(positions, 'device') else 'cpu'
        return torch.from_numpy(phi.astype(np.float32)).to(device)


class _MultiShotStaticOffResonancePhase:
    """
    phi_B0(r,t) = 2*pi*f(r)*t_local across a flattened multi-shot trajectory, where "t_local"
    resets to 0 at the start of every shot.

    GIRF-Sim's own "StaticOffResonancePhase" implements the correct SINGLE-shot formula
    "phi = 2*pi*f*t" but takes "t" literally - correct for one shot, wrong once several shots are
    concatenated into one global flattened time axis (this is exactly what
    "ForwardOperator.forward"'s own "t_full" is): by the N-th shot, "t_full" has already grown to
    "N*samples_per_shot*dt" (up to ~1.8s for this script's 100 shots x 455 samples), so even a
    modest ~100 Hz off-resonance accumulates many FULL CYCLES of spurious, effectively random
    phase before that shot's own readout has even started. Physically, every shot's own RF
    excitation resets transverse-plane phase to zero, so B0 dephasing must reset with it too -
    the exact same shot-boundary problem "_MultiShotConcomitantPhase" above already handles for
    the Maxwell term. (GIRF-Sim's class itself is not wrong - "phi=2*pi*f*t" is the correct
    single-shot formula - it is simply being fed the wrong "t" by this multi-shot script; the fix
    belongs here, not in GIRF-Sim.)

    Discovered via: the "b0"/"combined" panels reconstructing as near-total noise (NRMSE ~78%,
    unrelated to the slice-offset/GNL-scale changes - confirmed by reproducing it at matrix=48
    with both of those at their defaults) rather than the expected mild off-resonance blur from a
    real ~100 Hz field map over an ~18ms spiral readout.
    """
    name = 'static_b0_multishot'

    def __init__(self, freq_map_hz, n_shots: int, samples_per_shot: int, dt: float):
        import torch
        self.freq_map_hz = freq_map_hz
        t_shot = torch.arange(samples_per_shot, dtype=torch.float32) * dt
        self.t_local = t_shot.repeat(n_shots)   # (T,) - resets to 0 every samples_per_shot samples

    def phase(self, positions, t):
        import torch
        device = positions.device if hasattr(positions, 'device') else 'cpu'
        t_local = self.t_local.to(device)
        return 2.0 * math.pi * self.freq_map_hz.reshape(-1, 1).to(device) * t_local.reshape(1, -1)


class RepoAdapter:
    """
    Wires the figure to real Augmentrum + GIRF-Sim physics.

    Every condition is ONE "ForwardOperator.forward()" call: B1+ is a pre-multiplication of the
    object (never a phase term - "complex-valued multiplication, not Bloch", per the handover),
    B0/concomitant/GNL are real "PhaseTerm"s, and GIRF is real trajectory error ("dk_girf"). This
    is what makes "combined" genuinely single-pass rather than five reconstructions composed
    afterward. See scripts/SPATIAL_PHYSICS_FIGURE_STATUS.md and the approved plan at
    ~/.claude/plans/parallel-cooking-beacon.md for the full per-condition table and provenance of
    every physical constant used here.

    Positions/flatten-order convention (empirically verified against a point-source round trip
    before writing this class - see git history for the validation script): a 2-D image array
    "arr" of shape "(nx, ny)" is encoded as "arr.T.reshape(-1)" against
    "girf_synthetic.make_grid_2d(nx, ny, ...)"'s own (Y-outer, X-inner) position order; everything
    that reconstructs through "GriddingNUFFT.adjoint" comes back already in "arr"'s own
    orientation (no further transform needed); anything evaluated directly from "positions"
    (the phase-map extras, never reconstructed) needs ".reshape(ny, nx).T" to match.
    """

    def __init__(self, augmentrum_root: Optional[Path], girfsim_root: Optional[Path],
                outdir: Optional[Path] = None, field_strength_t: float = 3.0,
                gnl_coefficient_scale: float = 1.0, slice_offset_m: float = 0.0,
                device: str = 'auto'):
        self.augmentrum_root = augmentrum_root
        self.girfsim_root = girfsim_root
        self.seq_cache_dir = Path(outdir) if outdir is not None else Path('/tmp')
        self.seq_cache_dir.mkdir(parents=True, exist_ok=True)
        self._field_strength_t = float(field_strength_t)
        # Illustrative amplification of gradunwarp's bundled dummy coil - NOT a real vendor
        # coefficient fit. Picked empirically (not guessed): flat multiplicative coefficient
        # scaling is linear and well-behaved (10x -> 1.67 voxels, 50x -> 8.35 voxels, 100x ->
        # 16.7 voxels of displacement at a 0.22m FOV / 80-matrix); shrinking gradunwarp's R0_m
        # instead was tried and rejected - it diverges explosively once positions sit outside the
        # (now-shrunk) nominal radius in normalized units (0.5x -> 3.85 voxels, but 0.25x -> 84
        # voxels, 0.1x -> 3534 voxels), i.e. extrapolating a polynomial expansion outside its
        # design region - not a controllable dial.
        self._gnl_coefficient_scale = float(gnl_coefficient_scale)
        # Off-isocenter slice position, meters along the main-field (z) axis. The concomitant
        # Maxwell term's leading contribution for a pure in-plane 2-D readout (no concurrent
        # through-slice gradient) is "(Gx^2+Gy^2)*z^2/(2*B0)" - IDENTICALLY zero at z=0 (isocenter)
        # regardless of field strength, confirmed both analytically and empirically this session.
        # A nonzero offset activates that term, but because z is one constant value across the
        # whole 2-D slice, the result is a spatially UNIFORM phase/signal shift (same at every
        # in-plane position), not within-slice geometric distortion - that needs a true 3-D or
        # oblique acquisition with a concurrent through-slice gradient instead. Do not present a
        # nonzero slice_offset_m result as "spatial concomitant distortion" - it isn't one.
        self._slice_offset_m = float(slice_offset_m)

        import torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("--device cuda requested but torch.cuda.is_available() is False; "
                         "falling back to cpu.")
            device = 'cpu'
        self.device = device

        # Import every augmentrum.* module BEFORE girf_module/mrs_seq is ever touched (building
        # a .seq file first and importing augmentrum afterward reproducibly breaks
        # augmentrum.augmentation's import chain - a narrow ordering quirk in mrs_seq's own
        # trajectory adapter, confirmed twice this session; not an augmentrum bug).
        import augmentrum                                                     # noqa: F401
        from augmentrum.physics.concomitant_field import GAMMA_HZ_PER_T       # noqa: F401
        from augmentrum.physics.gnl_field import GradientNonlinearityPhase    # noqa: F401
        from augmentrum.physics import gradient_nonlinearity                  # noqa: F401
        from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT    # noqa: F401
        self._GradientNonlinearityPhase = GradientNonlinearityPhase
        self._GradientNonlinearityDisplacement = gradient_nonlinearity.GradientNonlinearityDisplacement
        self._load_gnl_coefficients = gradient_nonlinearity.load_coefficients
        self._GriddingNUFFT = GriddingNUFFT

        for root in [augmentrum_root, girfsim_root]:
            if root and str(root) not in sys.path:
                sys.path.insert(0, str(root))
        if str(self.girfsim_root or "") not in sys.path:
            pass   # mrs_seq is found via girf_module.__file__ below if not already on sys.path

        import girf_module as gmod
        import girf_mrsi_extensions as gmx
        import girf_synthetic as gs
        self.gmod, self.gmx, self.gs = gmod, gmx, gs
        if self.device != 'cpu':
            _patch_girf_sim_for_device(gmx)

        self._seq_path: Optional[str] = None
        self._mod_nominal = None          # GIRFModule.from_synthetic - defines k_nominal/gradients
        self._mod_measured = None         # GIRFModule.from_measured - defines dk_error (lazy)

    #*****************#
    #   the spiral    #
    #*****************#
    def get_spiral(self, matrix: int, fov_m: float, dt_s: float = 4e-5) -> Dict[str, np.ndarray]:
        """
        Builds ONE real, fully sampled 2-D spiral ".seq" via GIRF-Sim's mrs_seq
        (mrs_seq.imaging.build_spiral_gre, acquisition_mode='mri2d' - a spoiled spiral GRE, not
        MRSI/STEAM), which itself calls Augmentrum's own spiral_2d trajectory generator
        (mrs_seq/trajectory/augmentrum_adapter.py) - "Augmentrum's spiral" and "GIRF-Sim's
        sequence" are the same generator, not two competing ones. Loaded via
        GIRFModule.from_synthetic for the nominal trajectory every non-GIRF condition shares.
        """
        import sys as _sys
        from pathlib import Path as _Path
        try:
            import mrs_seq   # noqa: F401
        except ImportError:
            repo_root = str(_Path(self.gmod.__file__).resolve().parent)
            if repo_root not in _sys.path:
                _sys.path.insert(0, repo_root)

        from mrs_seq.imaging import build_spiral_gre
        from mrs_seq.system import build_system

        fov_mm = fov_m * 1000.0
        system = build_system('generic_3T', grad_raster_time=dt_s)
        # Explicit, user-specified shot count (200 interleaves x 455 samples/shot).
        n_shots = 200
        samples_per_shot = 455
        # Spoiled spiral GRE (mrs_seq.imaging.build_spiral_gre), not STEAM: this figure is a pure
        # SPATIAL ablation (no real spectral dimension - the phantom is built with t2star_s=1e12,
        # freq_hz=0), and GIRF-Sim now ships a dedicated single-pass imaging sequence family for
        # exactly this case (added this session specifically so Augmentrum+GIRF-Sim could serve
        # MRI, not only MRSI, use cases). The earlier approach - reusing the STEAM/MRSI builder
        # with a near-zero target_fid_duration to defeat its spectral-tiling logic - worked but was
        # a workaround, not a real fix; it also silently produced 3x too many samples per shot
        # before being caught (see SPATIAL_PHYSICS_FIGURE_STATUS.md). build_spiral_gre's own
        # build_mri_readout has NO tiling at all (n_repeats is hardcoded to 1) and its ADC covers
        # only the real shot samples (no pad-in/rewinder in the acquired window), so this is a
        # clean fix, not another workaround.
        #
        # Consequence: spoiled GRE has NO volume-selective localization (voi_mm=None in its own
        # metadata - a single slice-selective pulse sets slice THICKNESS only, not any in-plane
        # box) - unlike STEAM, which prescribed a VOI box later (ab)used for the GNL
        # excitation-side effect. That excitation-side VOI-warping model no longer has a physical
        # analog under GRE and has been removed from simulate_all (see its own comment) - GNL is
        # now modeled only via its readout-encoding phase term, same as b0/concomitant/girf.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            seq, info = build_spiral_gre(
                system, acquisition_mode='mri2d',
                fov_mm=(fov_mm, fov_mm), matrix=(matrix, matrix),
                slice_thickness_mm=20.0, n_shots=n_shots, samples_per_shot=samples_per_shot,
                # TR must comfortably exceed the readout alone (455 samples * 40us = 18.2ms) plus
                # RF/prephase/spoiler overhead - 50ms leaves ample margin regardless of matrix size.
                flip_angle_deg=12.0, TE=None, TR=0.05, n_dummy=20,
                seq_name='spatial_physics_ablation',
            )
        seq_path = str(self.seq_cache_dir / 'spatial_physics_ablation.seq')
        seq.write(seq_path)
        self._seq_path = seq_path

        self._mod_nominal = self.gmod.GIRFModule.from_synthetic(seq_path, seed=0)
        ndim = int(self._mod_nominal.k_nominal.shape[1])   # 3 even for a 2-D acquisition (z=0)

        # NOT self._mod_nominal.k_nominal: GIRFModule's k_nominal comes from
        # seq_girf.load_all_shots_batched, which re-references every shot's k-space to k=0 at the
        # first ADC sample ("k = k - k[:, :, 0:1]"), discarding the true, nonzero k-space position
        # the readout actually starts at (the moment accrued by the spiral's own prewinder before
        # the ADC turns on) - a constant 0.4005 cycles/m, identical across all 200 shots to 1e-17,
        # confirmed empirically (see SPATIAL_PHYSICS_FIGURE_STATUS.md). Using the zero-referenced
        # trajectory manufactures a fake 200-way exact coincidence at k=(0,0) that has no physical
        # counterpart - every shot actually starts at the SAME small radius at its own distinct
        # golden-angle azimuth, not at the origin.
        #
        # info['readouts'][i].k_achieved carries the true absolute, post-realizability trajectory
        # in the same per-shot order as GIRFModule's own batched loader (verified empirically:
        # shot-wise last-sample k-angle matches self._mod_nominal.k_nominal to within 0.003 rad /
        # 0.18 deg over all 200 shots). dk_error (below, from the measured-GIRF GIRFModule) is
        # unaffected by this swap - it is a cumsum-vs-cumsum difference against the SAME
        # zero-referenced baseline internally, so any constant offset added to that baseline
        # cancels out exactly; same for gradients_per_shot (gradients are rates of change, never
        # touched by the k re-reference, which only subtracts a constant from the integrated k).
        k_ach = np.stack([np.asarray(r.k_achieved) for r in info['readouts']]).astype(np.float64)  # [S,2,L]
        S, _, L = k_ach.shape
        k_full = np.zeros((S, ndim, L), dtype=np.float64)
        k_full[:, :2, :] = k_ach                         # z row stays 0 for a 2-D acquisition
        k_nominal = k_full.transpose(0, 2, 1).reshape(-1, ndim).T   # (ndim, S*L), same flatten order as before
        gradients_per_shot = list(self._mod_nominal.batched.gradients_t_per_m)   # n_shots x (3, L) - unaffected

        return {
            'k_nominal': k_nominal,
            'gradients_per_shot': gradients_per_shot,
            'dt_s': float(self._mod_nominal.dt),
            'sequence_path': seq_path,
            'ndim': ndim,
            'n_shots': len(gradients_per_shot),
        }

    #**************************#
    #   measured bacon girf    #
    #**************************#
    def load_bacon_measured_girf(self):
        """
        GIRF-Sim's own canonical measured-GIRF loader (girf_module.GIRFModule.from_measured),
        pointed at the real Bacon et al. released spherical-harmonic GIRF data already present in
        this GIRF-Sim checkout - not downloaded or reparsed here.
        """
        if self._seq_path is None:
            raise RuntimeError("call get_spiral() before load_bacon_measured_girf()")
        bacon_data_dir = str(Path(self.girfsim_root or
                                  '/run/media/johnlamaster/data/Postdoc/GIRF-Sim')
                             / 'local_data' / 'zenodo' / 'zenodo' / 'GIRFs')
        self._mod_measured = self.gmod.GIRFModule.from_measured(
            self._seq_path, bacon_data_dir=bacon_data_dir)
        return self._mod_measured

    #*****************#
    #   simulation    #
    #*****************#
    @staticmethod
    def _flatten(arr: np.ndarray) -> np.ndarray:
        """(nx, ny) -> (nx*ny,), matching make_grid_2d's (Y-outer, X-inner) position order."""
        return np.asarray(arr).T.reshape(-1)

    @staticmethod
    def _unflatten_to_grid(flat: np.ndarray, nx: int, ny: int) -> np.ndarray:
        """The inverse of "_flatten", for extras evaluated directly from positions (never
        reconstructed through the adjoint NUFFT, which needs no such transform - see the class
        docstring)."""
        return np.asarray(flat).reshape(ny, nx).T

    def simulate_all(self, inputs: FigureInputs) -> ForwardResult:
        import torch

        nx = ny = inputs.anatomy.shape[0]
        b0_tesla = float(getattr(self, '_field_strength_t', 3.0))

        spiral = self.get_spiral(nx, inputs.fov_m)
        k_nominal = spiral['k_nominal']                       # (ndim, T)
        ndim, T = k_nominal.shape
        dt_s = spiral['dt_s']
        gradients_per_shot = spiral['gradients_per_shot']

        measured = self.load_bacon_measured_girf()
        dk_error = (measured.dk_error.permute(0, 2, 1).reshape(-1, ndim)
                   .numpy().astype(np.float64).T)             # (ndim, T), measured-GIRF error
        k_actual = k_nominal + dk_error

        positions_cpu = self.gs.make_grid_2d(nx, ny, inputs.fov_m, inputs.fov_m, z=self._slice_offset_m)
        positions_np = positions_cpu.numpy().astype(np.float64)   # (N, 3), z=slice_offset_m - CPU/
                                                                   # numpy, reused for the RMS
                                                                   # phase-map extras below
        positions = positions_cpu.to(self.device)                # the torch tensor ForwardOperator uses
        rho_anatomy_full = self._flatten(inputs.anatomy).astype(np.complex64)
        rho_b1_full = self._flatten(inputs.anatomy * inputs.b1_rel).astype(np.complex64)
        # float32, not float64: StaticOffResonancePhase's "freq_map_hz * t" promotes to whichever
        # dtype is wider, and a float64 (N,T) intermediate at this script's N/T is multiple GB -
        # float32 is plenty of precision for a phase term and halves that allocation.
        b0_hz_flat = self._flatten(inputs.b0_hz).astype(np.float32)

        k_nominal_t = torch.from_numpy(k_nominal).float().to(self.device)
        dk_zero_t = torch.zeros(1, 3, T, device=self.device)
        dk_measured_t = torch.from_numpy(dk_error[None, :, :]).float().to(self.device)

        gnl_coeffs = self._load_gnl_coefficients()   # warns: gradunwarp's bundled dummy coil
        if self._gnl_coefficient_scale != 1.0:
            gnl_coeffs = gnl_coeffs._replace(**{
                f: getattr(gnl_coeffs, f) * self._gnl_coefficient_scale
                for f in gnl_coeffs._fields if f != 'R0_m'
            })

        # No excitation-side GNL model anymore: that mechanism relied on STEAM's volume-selective
        # VOI box (a voxel's GNL-displaced position landing outside a nominal excitation box).
        # Now that get_spiral() builds a spoiled spiral GRE (mrs_seq.imaging.build_spiral_gre)
        # instead of STEAM, there IS no in-plane VOI box to warp (its own metadata reports
        # "voi_mm": None - a single slice-selective pulse sets only slice THICKNESS, not any
        # in-plane excitation region) - "what part of the FOV gets excited" is physically just
        # "the whole FOV, uniformly" under GRE, so there is nothing left for GNL to distort on the
        # excitation side. GNL is modeled only via its readout-encoding phase term below, the same
        # way b0/concomitant/girf are - this is a real reduction from the STEAM-specific modeling
        # added earlier this session, not an oversight; see SPATIAL_PHYSICS_FIGURE_STATUS.md.
        rho_anatomy = rho_anatomy_full
        rho_b1 = rho_b1_full

        def encode_reconstruct(rho_np, phase_terms, dk_girf_t):
            phantom = self.gmx.SpectralPhantom.single_component(
                torch.from_numpy(rho_np).to(self.device), t2star_s=1e12, freq_hz=0.0)
            op = self.gmx.ForwardOperator(positions, k_nominal_t, dk_girf_t,
                                          phase_terms=phase_terms)
            result = op.forward(phantom, dt_s)
            d = result.d.cpu().numpy().astype(np.complex64)
            # Free this condition's GPU tensors (phantom, the (N,T) phase/signal intermediates
            # inside ForwardOperator.forward, result) before the next condition runs - otherwise
            # they accumulate across the 7 conditions in one process and OOM a 12GB card well
            # before anything is actually that large on its own.
            del phantom, op, result
            if self.device != 'cpu':
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            kmax_xy = nx / (2.0 * inputs.fov_m)
            coords = (k_nominal.T[:, :2] / (2.0 * kmax_xy)).astype(np.float32)
            nufft = self._GriddingNUFFT((nx, ny), osf=2.0)
            w = nufft.density_weights(coords)
            return nufft.adjoint((d * w)[None, :], coords)[0]

        samples_per_shot = T // spiral['n_shots']
        b0_term = _MultiShotStaticOffResonancePhase(
            torch.from_numpy(b0_hz_flat).to(self.device), spiral['n_shots'], samples_per_shot, dt_s)

        images: Dict[str, np.ndarray] = {}
        images['ideal'] = encode_reconstruct(rho_anatomy, [], dk_zero_t)
        images['b0'] = encode_reconstruct(rho_anatomy, [b0_term], dk_zero_t)
        images['b1'] = encode_reconstruct(rho_b1, [], dk_zero_t)

        concomitant_term = _MultiShotConcomitantPhase(gradients_per_shot, dt_s, b0_tesla)
        images['concomitant'] = encode_reconstruct(rho_anatomy, [concomitant_term], dk_zero_t)

        images['girf'] = encode_reconstruct(rho_anatomy, [], dk_measured_t)

        # GNL: readout-encoding phase term only now (see the comment above the rho_anatomy/rho_b1
        # assignment for why the excitation-side VOI-warping mechanism was removed).
        gnl_term_nominal = self._GradientNonlinearityPhase(gnl_coeffs, k_nominal, max_order=9)
        images['gnl'] = encode_reconstruct(rho_anatomy, [gnl_term_nominal], dk_zero_t)

        gnl_term_actual = self._GradientNonlinearityPhase(gnl_coeffs, k_actual, max_order=9)
        combined_terms = [
            b0_term,
            _MultiShotConcomitantPhase(gradients_per_shot, dt_s, b0_tesla),
            gnl_term_actual,
        ]
        images['combined'] = encode_reconstruct(rho_b1, combined_terms, dk_measured_t)

        # Top-row extras: RMS-over-readout phase maps (this file's own documented convention,
        # per the handover's "prefer RMS or max-over-readout because it is less arbitrary").
        # Computed on CPU (positions_np/t_axis_cpu), same reasoning as ForwardOperator.total_phase
        # above - a standalone (N, T) GPU tensor at T~91000+ has no need to compete for scarce
        # VRAM when this is a one-off, non-performance-critical plotting extra.
        t_axis_cpu = torch.arange(T, dtype=torch.float32) * dt_s
        positions_cpu_t = torch.from_numpy(positions_np).float()
        conc_phase = concomitant_term.phase(positions_cpu_t, t_axis_cpu).cpu().numpy()
        gnl_phase = gnl_term_nominal.phase(positions_cpu_t, t_axis_cpu).cpu().numpy()
        concomitant_phase_map = self._unflatten_to_grid(
            np.sqrt(np.mean(conc_phase ** 2, axis=1)), nx, ny)
        gnl_phase_map = self._unflatten_to_grid(
            np.sqrt(np.mean(gnl_phase ** 2, axis=1)), nx, ny)

        extras = {
            'k_nominal': k_nominal.T[:, :2],
            'k_girf': k_actual.T[:, :2],
            'concomitant_phase_map': concomitant_phase_map,
            'gnl_phase_map': gnl_phase_map,
        }
        metadata = {
            'sequence': spiral['sequence_path'],
            'trajectory_units': 'cycles/m',
            'field_strength_T': b0_tesla,
            'slice_offset_m': self._slice_offset_m,
            'girf_source': 'Bacon measured GIRF (order 1)',
            'concomitant_map_definition': 'RMS phase across ADC samples',
            'gnl_map_definition': 'RMS phase across ADC samples',
            'reconstruction': 'GriddingNUFFT adjoint on nominal trajectory, Pipe DCF',
            'n_shots': spiral['n_shots'],
            'gnl_source': ('gradunwarp bundled test coil - NOT a real scanner calibration; '
                          'see augmentrum.physics.gradient_nonlinearity.load_coefficients'),
            'gnl_coefficient_scale': self._gnl_coefficient_scale,
            'gnl_excitation_model': ('none - readout-encoding phase term only; the earlier '
                                    'STEAM-VOI excitation-side warping model was removed when '
                                    'get_spiral() switched to a spoiled spiral GRE sequence '
                                    '(mrs_seq.imaging.build_spiral_gre), which has no '
                                    'volume-selective VOI to warp'),
        }
        return ForwardResult(images=images, extras=extras, metadata=metadata)


# ----------------------------
# Metrics
# ----------------------------

def magnitude(x: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(x))


def nrmse(ref: np.ndarray, x: np.ndarray) -> float:
    ref = magnitude(ref)
    x = magnitude(x)
    return float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-12))


def estimate_background_sigma(ideal_img: np.ndarray, mask: np.ndarray) -> float:
    """
    Rician noise-scale estimate from the "ideal" reconstruction's own background (outside
    "mask", where true signal is exactly zero). Any nonzero magnitude there is attributable
    entirely to the reconstruction's own noise-like floor (see
    SPATIAL_PHYSICS_FIGURE_STATUS.md's streaking investigation - this floor is real,
    reproducible, and not removable by DCF/trajectory fixes, so NCE below treats it the same
    way genuine thermal noise would be treated: not something a reconstruction should be
    penalized for failing to reproduce exactly).

    For background magnitude data (true signal = 0), Rician reduces to Rayleigh(sigma), for
    which E[R^2] = 2*sigma^2 - the standard estimator used here.
    """
    bg = magnitude(ideal_img)[mask <= 0]
    if bg.size == 0:
        return 1e-8
    return float(np.sqrt(np.mean(bg.astype(np.float64) ** 2) / 2.0)) or 1e-8


def nce(x_img: np.ndarray, y_img: np.ndarray, sigma: float, eps: float = 1e-12) -> float:
    """
    Noncentral Chi Error (see nce_handover.md), specialized to L=1 (single virtual coil - this
    simulation does not do multi-coil root-sum-of-squares combination, so the noncentral chi
    distribution's L=1 case applies, which is exactly the Rician distribution).

    NCE(x, y) = -ln p(y|x) + ln p(y|y), under y_n ~ Rice(x_n, sigma). Evaluates reconstruction
    quality against "y" (here, the "ideal" condition, standing in as this script's best
    available reference) without penalizing "x" for failing to reproduce the reference's own
    noise-like texture exactly - unlike NRMSE, which treats every last bit of that texture as
    error. NCE(y, y) = 0 by construction (derived and confirmed below).

    Derivation (L=1, so the general formula's "(L-1)*ln(x/y)" term vanishes and
    "x^(1-L)=1"): expanding and cancelling the shared "2*ln(sigma)" and "ln(y)" terms between
    "-ln p(y|x)" and "+ln p(y|y)" leaves
        NCE = (x^2 - y^2) / (2*sigma^2) + ln I_0(y^2/sigma^2) - ln I_0(x*y/sigma^2)
    "log I_0(z) = log(ive(0,z)) + z" (ive = exponentially-scaled I_0, "ive(0,z) = I_0(z)*exp(-z)"
    for z>=0) keeps this finite for the large z typical at this script's SNR, instead of
    overflowing through the raw modified Bessel function.
    """
    x = np.maximum(magnitude(x_img), eps)
    y = np.maximum(magnitude(y_img), eps)
    sigma2 = max(float(sigma), eps) ** 2
    z_xy = x * y / sigma2
    z_yy = y * y / sigma2
    log_i0_xy = np.log(np.maximum(ive(0, z_xy), eps)) + z_xy
    log_i0_yy = np.log(np.maximum(ive(0, z_yy), eps)) + z_yy
    per_voxel = (x ** 2 - y ** 2) / (2.0 * sigma2) + log_i0_yy - log_i0_xy
    return float(np.mean(per_voxel))


def compute_metrics(result: ForwardResult, inputs: Optional["FigureInputs"] = None) -> Dict[str, float]:
    ref = result.images["ideal"]
    metrics = {}
    sigma = (estimate_background_sigma(ref, inputs.mask) if inputs is not None
            else estimate_background_sigma(ref, np.ones_like(magnitude(ref))))
    metrics["nce_sigma"] = sigma
    for k in EFFECT_KEYS[1:]:
        metrics[f"nrmse_{k}"] = nrmse(ref, result.images[k])
        metrics[f"nce_{k}"] = nce(result.images[k], ref, sigma)
        metrics[f"max_abs_diff_{k}"] = float(
            np.max(np.abs(magnitude(result.images[k]) - magnitude(ref)))
        )

    if "k_nominal" in result.extras and "k_girf" in result.extras:
        a = np.asarray(result.extras["k_nominal"])
        b = np.asarray(result.extras["k_girf"])
        n = min(len(a), len(b))
        d = np.linalg.norm(a[:n, :2] - b[:n, :2], axis=1)
        metrics["girf_trajectory_rms"] = float(np.sqrt(np.mean(d**2)))
        metrics["girf_trajectory_max"] = float(np.max(d))
    return metrics


# ----------------------------
# Plotting
# ----------------------------

def _imshow_clean(ax, image, cmap="gray", vmin=None, vmax=None):
    im = ax.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_top_row(axs, inputs: FigureInputs, result: ForwardResult):
    # Reference/anatomy
    _imshow_clean(axs[0], inputs.anatomy, "gray", 0, 1)

    # B0
    vmax_b0 = max(1.0, np.max(np.abs(inputs.b0_hz)))
    im = _imshow_clean(axs[1], inputs.b0_hz, "coolwarm", -vmax_b0, vmax_b0)
    axs[1].text(0.03, 0.03, r"$\Delta B_0$ [Hz]", transform=axs[1].transAxes,
                fontsize=9, color="black", ha="left", va="bottom")

    # B1+ - same diverging convention as B0, centered at 1.0 (no modulation) instead of 0.
    b1_span = max(1e-6, np.max(np.abs(inputs.b1_rel - 1.0)))
    im = _imshow_clean(axs[2], inputs.b1_rel, "coolwarm", 1.0 - b1_span, 1.0 + b1_span)
    axs[2].text(0.03, 0.03, r"$B_1^+/B_{1,\mathrm{nom}}^+$",
                transform=axs[2].transAxes, fontsize=9,
                color="black", ha="left", va="bottom")

    # Concomitant - RMS-over-readout phase (see metadata["concomitant_map_definition"]):
    # non-negative by construction, so a sequential colormap from 0, not a diverging one.
    if "concomitant_phase_map" in result.extras:
        p = np.asarray(result.extras["concomitant_phase_map"])
        vmax = max(1e-8, np.max(p))
        vmin = np.min(p)
        _imshow_clean(axs[3], p, "magma", 0, vmax)
        # This panel is often spatially UNIFORM by construction (a uniform slice offset with no
        # concurrent through-slice gradient gives a position-INdependent phase - see
        # slice_offset_m's docstring in RepoAdapter), so color alone carries no information; the
        # actual value(s) must be printed, not just implied by a vmin/vmax color scale.
        label = (f"RMS phase: {vmax:.3g} rad (uniform)" if (vmax - vmin) < 1e-3 * max(vmax, 1e-12)
                else f"RMS phase: {vmin:.3g}–{vmax:.3g} rad")
        axs[3].text(0.03, 0.03, label, transform=axs[3].transAxes,
                    fontsize=8.5, color="white", ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.55))
    else:
        axs[3].text(0.5, 0.5, "Concomitant\nphase map", ha="center", va="center")
        axs[3].axis("off")

    # GIRF trajectory overlay
    if "k_nominal" in result.extras and "k_girf" in result.extras:
        kn_full = np.asarray(result.extras["k_nominal"])
        kg_full = np.asarray(result.extras["k_girf"])
        # BUG FIX: this used to plot the full flattened (n_shots*samples_per_shot, 2) arrays as
        # ONE continuous line - with n_shots=200 that's unreadable for two reasons: (1) every
        # shot boundary draws a spurious straight connecting line to the next shot's start, and
        # (2) 200 overlaid interleaves saturate into a solid blob regardless. Fix: split by shot
        # (using metadata["n_shots"], always populated by get_spiral) and plot only a few
        # individual shots as separate line segments. Shots are golden-angle ordered (see
        # RepoAdapter.get_spiral's own docstring), so the first few (a PREFIX) are evenly
        # distributed in angle - unlike a strided subset, which is not.
        n_shots = int(result.metadata.get("n_shots", 1))
        samples_per_shot = kn_full.shape[0] // n_shots
        n_show = min(16, n_shots)
        for i in range(n_show):
            sl = slice(i * samples_per_shot, (i + 1) * samples_per_shot)
            axs[4].plot(kn_full[sl, 0], kn_full[sl, 1], lw=0.9, color="tab:blue",
                       label="Nominal" if i == 0 else None)
            axs[4].plot(kg_full[sl, 0], kg_full[sl, 1], lw=0.9, color="tab:orange", alpha=0.85,
                       label="GIRF" if i == 0 else None)
        axs[4].set_aspect("equal")
        axs[4].set_xticks([])
        axs[4].set_yticks([])
        axs[4].legend(fontsize=7.5, frameon=False, loc="best")
    else:
        axs[4].text(0.5, 0.5, "Nominal vs\nGIRF trajectory", ha="center", va="center")
        axs[4].axis("off")

    # GNL - same RMS-over-readout convention as concomitant (see
    # metadata["gnl_map_definition"]); non-negative, sequential colormap.
    if "gnl_phase_map" in result.extras:
        p = np.asarray(result.extras["gnl_phase_map"])
        vmax = max(1e-8, np.max(p))
        _imshow_clean(axs[5], p, "magma", 0, vmax)
        axs[5].text(0.03, 0.03, f"RMS phase: 0–{vmax:.3g} rad", transform=axs[5].transAxes,
                    fontsize=8.5, color="white", ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.55))
    else:
        axs[5].text(0.5, 0.5, "GNL spatial\nencoding map", ha="center", va="center")
        axs[5].axis("off")

    # Combined schematic
    axs[6].axis("off")
    axs[6].text(
        0.5, 0.57,
        r"$B_0$  +  $B_1^+$" + "\n+\nConcomitant  +  GIRF\n+\nGNL",
        ha="center", va="center", fontsize=9
    )
    axs[6].text(0.5, 0.08, "Composed in forward model",
                ha="center", va="bottom", fontsize=8.5)


def plot_figure(inputs: FigureInputs, result: ForwardResult, outdir: Path, dpi: int = 350):
    images = {k: magnitude(result.images[k]) for k in EFFECT_KEYS}
    ref = images["ideal"]

    # Common reconstruction window.
    vmax_img = np.percentile(ref[inputs.mask > 0], 99.7) if np.any(inputs.mask > 0) else np.percentile(ref, 99.7)
    vmax_img = max(vmax_img, 1e-8)

    diffs = {k: np.abs(images[k] - ref) for k in EFFECT_KEYS[1:]}
    diff_vmax = max(np.percentile(d, 99.5) for d in diffs.values())
    diff_vmax = max(diff_vmax, 1e-10)

    metrics = compute_metrics(result, inputs)

    fig, axes = plt.subplots(
        3, 7, figsize=(15.5, 7.1),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]}
    )
    plt.subplots_adjust(left=0.045, right=0.995, top=0.91, bottom=0.07,
                        wspace=0.055, hspace=0.12)

    for j, title in enumerate(EFFECT_TITLES):
        axes[0, j].set_title(title, fontsize=10, pad=6)

    # Top row: inputs / physical drivers
    plot_top_row(axes[0], inputs, result)

    # Middle row: reconstructions
    for j, k in enumerate(EFFECT_KEYS):
        _imshow_clean(axes[1, j], images[k], "gray", 0, vmax_img)
        if k != "ideal":
            axes[1, j].text(
                0.03, 0.04, f"NCE {metrics[f'nce_{k}']:.3g}",
                transform=axes[1, j].transAxes,
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.65)
            )

    # Bottom row: matched differences. Every panel shares one color scale (diff_vmax) so
    # brightness is comparable across effects - but an effect whose own difference is much
    # smaller than the scale (e.g. GNL/concomitant next to B0) would then just read as black,
    # which looks like "no effect" rather than "small effect". Where a panel's own 99.5th
    # percentile is well under the shared scale, multiply the DISPLAYED values by a round factor
    # (still against the same diff_vmax, so panels stay comparable) and label it, rather than
    # silently rescaling or - worse - leaving it looking empty.
    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, "Matched\nabsolute difference\nfrom ideal",
                    ha="center", va="center", fontsize=8)
    last_im = None
    for j, k in enumerate(EFFECT_KEYS[1:], start=1):
        own_max = np.percentile(diffs[k], 99.5)
        mult = 1
        if own_max > 0 and own_max < diff_vmax / 3.0:
            mult = int(10 ** max(0, round(math.log10(diff_vmax / own_max))))
        last_im = _imshow_clean(axes[2, j], diffs[k] * mult, "magma", 0, diff_vmax)
        if mult > 1:
            axes[2, j].text(0.03, 0.03, f"×{mult:g}", transform=axes[2, j].transAxes,
                            fontsize=9, color="white", ha="left", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.55))

    # Row labels
    fig.text(0.012, 0.755, "Physics input", rotation=90, va="center", fontsize=9)
    fig.text(0.012, 0.475, "Reconstruction", rotation=90, va="center", fontsize=9)
    fig.text(0.012, 0.185, "Difference", rotation=90, va="center", fontsize=9)

    if last_im is not None:
        cax = fig.add_axes([0.925, 0.025, 0.065, 0.012])
        cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=6)
        cb.set_label("|effect − ideal| [a.u.]", fontsize=6.5)

    fig.suptitle(
        "Augmentrum+: spatially structured acquisition-physics ablations",
        fontsize=12, y=0.975
    )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(outdir / f"spatial_physics_ablation.{ext}",
                    dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(outdir / "figure_caption.txt", "w") as f:
        f.write(CAPTION + "\n")

    with open(outdir / "forward_metadata.json", "w") as f:
        json.dump(result.metadata, f, indent=2, default=str)

    return metrics


def plot_preparation_preview(inputs: FigureInputs, outdir: Path, dpi: int = 220):
    """
    Creates a NON-SCIENTIFIC preparation preview so the data pipeline/layout can
    be inspected before the actual Augmentrum/GIRF-Sim forward adapter is wired.
    """
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.1))
    _imshow_clean(ax[0], inputs.anatomy, "gray", 0, 1)
    ax[0].set_title("Anatomy")
    vmax = max(1, np.max(np.abs(inputs.b0_hz)))
    _imshow_clean(ax[1], inputs.b0_hz, "coolwarm", -vmax, vmax)
    ax[1].set_title(r"$\Delta B_0$ [Hz]")
    _imshow_clean(ax[2], inputs.b1_rel, "viridis")
    ax[2].set_title(r"Relative $B_1^+$")
    fig.suptitle("Preparation preview only — no acquisition physics applied", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "prepared_inputs_preview.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--augmentrum-root", type=str, default=os.getenv("AUGMENTRUM_ROOT"))
    p.add_argument("--girfsim-root", type=str, default=os.getenv("GIRFSIM_ROOT"))
    p.add_argument("--anatomy", type=str, default=None,
                   help="Explicit NIfTI anatomy file - overrides --anatomy-source when given. "
                        "Point this at a manually-downloaded BrainWeb volume once you have one.")
    p.add_argument("--anatomy-source", choices=["shepp_logan", "brainweb", "hcp"], default="shepp_logan",
                   help="Synthetic anatomy source when --anatomy is not given. Default "
                        "'shepp_logan' (no download, no baked-in scanner artifacts); flip to "
                        "'brainweb' once a BrainWeb fetch actually reaches real data - as of "
                        "2026-10-23 brainweb.bic.mni.mcgill.ca (and the 'brainweb' pip package, "
                        "which points at the same host) was unreachable from this environment.")
    p.add_argument("--hcp-slice-offset-vox", type=int, default=0,
                   help="Only used with --anatomy-source hcp: axial voxel offset from the "
                        "registered volume's own mid-slice (index shape[2]//2), for picking a "
                        "different 2D anatomy slice to image - NOT the same as --slice-offset-mm "
                        "(a separate physics parameter for the concomitant/GNL terms).")
    p.add_argument("--brainweb-subject", type=int, default=4,
                   help="BrainWeb subject number, used only with --anatomy-source brainweb.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Device for the exact-NUDFT encode (this script's dominant cost). "
                        "'auto' (default) uses CUDA if available. GIRF-Sim's own ForwardOperator "
                        "hardcodes several internal tensors to CPU; this script monkey-patches "
                        "those (in-process only, never touching the GIRF-Sim checkout on disk) "
                        "so CUDA actually works rather than crashing on a device mismatch.")
    p.add_argument("--gnl-coefficient-scale", type=float, default=1.0,
                   help="Illustrative multiplicative amplification of gradunwarp's bundled dummy "
                        "coil's coefficients (NOT a real vendor calibration) - the default "
                        "coefficients give a sub-pixel (~0.17 voxel) displacement at a typical "
                        "FOV/matrix, invisible regardless of display scaling. ~20-50 gives a "
                        "clearly visible, non-absurd effect (empirically checked, see "
                        "SPATIAL_PHYSICS_FIGURE_STATUS.md); do not shrink gradunwarp's R0_m "
                        "instead - it diverges explosively rather than scaling smoothly.")
    p.add_argument("--slice-offset-mm", type=float, default=0.0,
                   help="Off-isocenter slice position along the main-field (z) axis, in mm. "
                        "At the default 0 (isocenter), the concomitant-field term is analytically "
                        "zero for this in-plane-only 2-D acquisition regardless of field strength "
                        "- a nonzero offset activates it, but ONLY as a spatially uniform phase/"
                        "signal shift (same at every in-plane position), not geometric distortion "
                        "- that needs a true 3-D/oblique acquisition instead. A few cm (e.g. 30) "
                        "is a typical off-center axial slice.")
    p.add_argument("--rotate90-ccw", type=int, default=0,
                   help="Number of 90-degree counterclockwise rotations applied to the anatomy "
                        "and every map together (as displayed - verified empirically against "
                        "this script's own origin='lower' convention, not assumed from numpy's "
                        "own rot90 direction). E.g. 1 for a single 90-degree CCW turn.")
    p.add_argument("--b0-map", type=str, default="auto",
                   help="Quantitative B0/DeltaB0 NIfTI in Hz. 'auto' (default) derives it from "
                        "the real HCP subject 105923 GRE field map; 'synthetic' disables it in "
                        "favor of the representative synthetic map; any other value is an "
                        "explicit path to your own file.")
    p.add_argument("--b1-map", type=str, default="auto",
                   help="B1+ NIfTI (script normalizes brain median to 1). 'auto' (default) "
                        "derives it from the real HCP subject 100408 actual-flip-angle map; "
                        "'synthetic' disables it in favor of the representative synthetic map; "
                        "any other value is an explicit path to your own file.")
    p.add_argument("--outdir", type=str, default="outputs/spatial_physics_ablation")
    p.add_argument("--matrix", type=int, default=192)
    p.add_argument("--fov-mm", type=float, default=220.0)
    p.add_argument("--field-strength-t", type=float, default=3.0)
    p.add_argument("--synthetic-b0-hz", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dpi", type=int, default=350)
    p.add_argument("--prepare-only", action="store_true",
                   help="Prepare anatomy/maps and preview; do not call repo physics.")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = load_or_prepare_inputs(args)
    save_inputs(inputs, outdir)
    plot_preparation_preview(inputs, outdir)

    with open(outdir / "figure_caption.txt", "w") as f:
        f.write(CAPTION + "\n")

    if args.prepare_only:
        print(f"Prepared inputs in: {outdir.resolve()}")
        print("No scientific effect images were generated.")
        print("Next: wire RepoAdapter.simulate_all() using Augmentrum + GIRF-Sim.")
        return

    adapter = RepoAdapter(
        Path(args.augmentrum_root).resolve() if args.augmentrum_root else None,
        Path(args.girfsim_root).resolve() if args.girfsim_root else None,
        outdir=outdir, field_strength_t=args.field_strength_t,
        gnl_coefficient_scale=args.gnl_coefficient_scale,
        slice_offset_m=args.slice_offset_mm / 1000.0,
        device=args.device,
    )
    result = adapter.simulate_all(inputs)

    missing = [k for k in EFFECT_KEYS if k not in result.images]
    if missing:
        raise RuntimeError(f"Repository adapter did not return required image keys: {missing}")

    np.savez_compressed(
        outdir / "real_effect_results.npz",
        **{f"image_{k}": result.images[k] for k in EFFECT_KEYS},
        **{k: np.asarray(v) for k, v in result.extras.items()},
    )

    metrics = plot_figure(inputs, result, outdir, args.dpi)
    print(json.dumps(metrics, indent=2))
    print(f"Figure written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
