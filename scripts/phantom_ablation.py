####################################################################################################
#                                     phantom_ablation.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-07-29                                                                              #
#                                                                                                  #
# Purpose: Ablates Augmentrum's spatial augmentation and k-space sampling on phantoms, where       #
#          ground truth is exact and every effect can be isolated and measured.                    #
#                                                                                                  #
####################################################################################################

"""
Ablate spatial augmentation and k-space sampling against exact ground truth.

On in-vivo data an augmentation can only be eyeballed. A Shepp-Logan or a GE
phantom has known geometry and no noise, so each effect can be turned on ALONE
and scored against the untouched original — which is what turns "it looks
plausible" into a number.

Ablations
---------
1. Spatial   each augmentation in isolation (translate, rotate, zoom, shear,
             flip, scale)
2. Sampling  acceleration sweep, Cartesian masks and trajectory-derived masks
3. Recon     zero-filled reconstruction error vs acceleration, and the
             anisotropy correction
4. NUFFT     the same measured samples reconstructed by nearest-bin gridding and
             by the reconstructor, plus how much oversampling the cubic
             interpolator needs
5. Panel     a full-page sweep of every panel trajectory against acceleration,
             2-D and 3-D side by side, reconstructed by NUFFT throughout

Output ("results/phantom_ablation/")
--------------------------------------
===================================  =========================================
"<phantom>_spatial_ablation"       one panel per augmentation, applied alone
"<phantom>_sampling_ablation"      masks and their zero-filled recons
"<phantom>_acceleration_curve"     reconstruction error vs acceleration
"<phantom>_trajectories_2d"        2-D trajectories at matched bin coverage
"<phantom>_trajectories_3d"        3-D trajectories at matched bin coverage
"<phantom>_nufft_ablation"         nearest-bin gridding vs NUFFT, same samples
"<phantom>_interpolation_error"    interpolator error vs k-space oversampling
"<phantom>_nufft_panel"            full page: trajectory vs acceleration
"<phantom>_nufft_panel_inset"      the same, with a zoomed window onto the
                                     Cartesian bins on each 2-D pattern
"<phantom>_nufft_panel_paired"     the same, with what was sampled above each
                                     reconstruction
"<phantom>_nufft_panel_paired_inset"  paired, with the zoom windows as well
"anisotropy_ablation"              rotation with and without the voxel-size
                                     correction
"metrics.csv"                      every number behind those figures
===================================  =========================================

Every figure is written as both PNG and PDF. Ablations 4 to 5 need the optional
"torchkbnufft" dependency and are skipped with a note when it is absent; the
3-D NUFFTs in the panel dominate the runtime.
"""

#*************#
#   imports   #
#*************#
import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # Ensure relative paths like 'data/...' resolve correctly

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from augmentrum.augmentation.spatial_augmentations import SpatialAugmentations
from augmentrum.sampling import GridMask, KspaceSampler, KspaceUndersampling


#************#
#   config   #
#************#
DEFAULT_CONFIG = {
    'phantom_dir':  'data/Phantom',
    # One phantom per entry. Only the 100 um volume is listed: block-averaged to
    # the same output matrix the two sources agree to a correlation of 0.99886
    # (mean absolute difference 0.006), because the 400 um file is already an
    # averaged version of the same brain. Running both would cost a second full
    # ablation for a duplicate figure. The 100 um is the better source of the
    # two, and its extra 45 s of streaming is nothing against the run.
    'bigbrain':     ('data/BigBrainMR/BigBrainMR_T1weighted_100um.nii.gz',),
    'save_dir':     'results/phantom_ablation',
    'size':         128,        # in-plane matrix for the generated Shepp-Logan
    'n_slices':     32,         # slab thickness; 32 resolves the c=0.05 tumours
    'seed':         0,
    'accelerations': (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0),
    'voxel_mm':     (2.8, 3.5, 4.0),   # deliberately anisotropic, as in MRSI
    'sl_variant':   'modified',        # Toft contrast; 'original' hides the tumours
    'coverage_target': 0.25,           # matched retained-bin fraction (~4x)
    'traj_shots':   256,               # starting shot count before coverage matching
    'nufft_osf':    2.0,               # NUFFT grid oversampling
    'interp_osf':   (1, 2, 3, 4, 6),   # k-space oversampling sweep for the interpolator
    'panel_accelerations': (1.0, 2.0, 4.0, 6.0, 8.0),   # rows of the full-page panel
}


#**************#
#   phantoms   #
#**************#
# Shepp-Logan geometry. The classic phantom (data/Phantom/SheppLogan.java) is
# strictly 2-D — ten ellipses in a plane — so a stack of copies has no
# through-plane structure and every z-direction ablation is vacuous.
#
# SHEPP_LOGAN_3D is the Kak & Slaney extension: the same ten features as
# ellipsoids, with z centres and a semi-axis c. The two tumours sit at z = 0.25
# and the upper inclusion at z = -0.15, so slices genuinely differ and flip z,
# translate z and z-shear have something real to act on.
#
# Columns (2-D): x centre, y centre, a, b, rotation [deg]
SHEPP_LOGAN_GEOMETRY = (
    ( 0.00,  0.0000, 0.6900, 0.920,   0.0),   # skull
    ( 0.00, -0.0184, 0.6624, 0.874,   0.0),   # brain
    ( 0.22,  0.0000, 0.1100, 0.310, -18.0),   # right ventricle
    (-0.22,  0.0000, 0.1600, 0.410,  18.0),   # left ventricle
    ( 0.00,  0.3500, 0.2100, 0.250,   0.0),   # upper inclusion
    ( 0.00,  0.1000, 0.0460, 0.046,   0.0),   # tumour, upper
    ( 0.00, -0.1000, 0.0460, 0.046,   0.0),   # tumour, lower
    (-0.08, -0.6050, 0.0460, 0.023,   0.0),   # basal inclusions
    ( 0.00, -0.6050, 0.0230, 0.023,   0.0),
    ( 0.06, -0.6050, 0.0230, 0.046,   0.0),
)

# Columns (3-D): x centre, y centre, z centre, a, b, c, rotation about z [deg]
SHEPP_LOGAN_3D = (
    ( 0.00,  0.0000,  0.00, 0.6900, 0.920, 0.810,   0.0),   # skull
    ( 0.00, -0.0184,  0.00, 0.6624, 0.874, 0.780,   0.0),   # brain
    ( 0.22,  0.0000,  0.00, 0.1100, 0.310, 0.220, -18.0),   # right ventricle
    (-0.22,  0.0000,  0.00, 0.1600, 0.410, 0.280,  18.0),   # left ventricle
    ( 0.00,  0.3500, -0.15, 0.2100, 0.250, 0.410,   0.0),   # upper inclusion
    ( 0.00,  0.1000,  0.25, 0.0460, 0.046, 0.050,   0.0),   # tumour, upper
    ( 0.00, -0.1000,  0.25, 0.0460, 0.046, 0.050,   0.0),   # tumour, lower
    (-0.08, -0.6050,  0.00, 0.0460, 0.023, 0.050,   0.0),   # basal inclusions
    ( 0.00, -0.6060,  0.00, 0.0230, 0.023, 0.020,   0.0),
    ( 0.06, -0.6050,  0.00, 0.0230, 0.046, 0.020,   0.0),
)

# Two intensity sets over that geometry.
#
# 'original' is Shepp & Logan's, and it is deliberately near-invisible: the
# inclusions sit 0.01 above a brain of 1.02 while the skull is 2.00, so every
# internal feature is a 0.5% step across the display range. That is the point of
# the original — it was designed as a hard reconstruction test, not a picture.
#
# 'modified' is Toft's rescaling of the same ellipses. The features become a 50%
# step against the brain, which is what makes the ventricles and the two tumours
# actually readable, and is what most people mean by "the Shepp-Logan phantom".
SHEPP_LOGAN_INTENSITIES = {
    'original': (2.00, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
    'modified': (1.00, -0.80, -0.20, -0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10),
}


def shepp_logan(size=128, n_slices=1, variant='modified', z_extent=0.8):
    """
    Generate the Shepp-Logan phantom as (size, size, n_slices).

    A single slice uses the classic 2-D ellipses. More than one uses the 3-D
    ellipsoid table, so slices actually differ — stacking copies of the 2-D
    phantom would make every z-direction augmentation a no-op, and the ablation
    would report a meaningless 0% error for flip z and z-shear.

    Args:
        variant: 'modified' (default, Toft contrast — internal structure and the
                two tumours are visible) or 'original' (Shepp & Logan's, whose
                features are 1% steps and will not show on a linear grey scale
                windowed to the skull).
        z_extent: the slab spans z in [-z_extent, +z_extent] of the phantom's
                normalised height. The default 0.8 covers essentially the whole
                head (the brain ellipsoid has c = 0.78), so the stack is a real
                volume rather than a thin slab. Resolving the tumours needs
                enough slices: their semi-axis is c = 0.05, so slice spacing
                2 * z_extent / n_slices should be below that — n_slices >= 32.
    """
    if variant not in SHEPP_LOGAN_INTENSITIES:
        raise ValueError(f"variant must be one of {sorted(SHEPP_LOGAN_INTENSITIES)}, "
                         f"got {variant!r}")
    intensities = SHEPP_LOGAN_INTENSITIES[variant]
    ax = np.linspace(-1.0, 1.0, size)

    if n_slices == 1:
        xx, yy = np.meshgrid(ax, ax, indexing='ij')
        img = np.zeros((size, size), dtype=np.float32)
        for (x0, y0, a, b, angle_deg), value in zip(SHEPP_LOGAN_GEOMETRY, intensities):
            t = np.deg2rad(angle_deg)
            xr = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
            yr = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
            img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += value
        return img[:, :, None]

    az = np.linspace(-z_extent, z_extent, n_slices)
    xx, yy, zz = np.meshgrid(ax, ax, az, indexing='ij')
    vol = np.zeros((size, size, n_slices), dtype=np.float32)
    for (x0, y0, z0, a, b, c, angle_deg), value in zip(SHEPP_LOGAN_3D, intensities):
        t = np.deg2rad(angle_deg)
        dx, dy, dz = xx - x0, yy - y0, zz - z0
        xr = dx * np.cos(t) + dy * np.sin(t)
        yr = -dx * np.sin(t) + dy * np.cos(t)
        vol[(xr / a) ** 2 + (yr / b) ** 2 + (dz / c) ** 2 <= 1.0] += value
    return vol


def load_ge_phantom(phantom_dir, size=128, n_slices=8):
    """
    Load the GE phantom and crop a centred slab to *size* x *size* x *n_slices*.

    The file is a 192 x 192 x 176 int16 magnitude volume, so it is normalised to
    a unit peak to make the error metrics comparable with the Shepp-Logan.
    """
    import nibabel as nib

    path = os.path.join(phantom_dir, 'GE.nii.gz')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GE phantom not found at {path}")

    vol = np.asarray(nib.load(path).dataobj).astype(np.float32)
    out = []
    for n, axis in zip((size, size, n_slices), (0, 1, 2)):
        start = max(0, (vol.shape[axis] - n) // 2)
        out.append(slice(start, start + min(n, vol.shape[axis])))
    vol = vol[tuple(out)]

    peak = vol.max()
    return vol / peak if peak > 0 else vol


def load_bigbrain(path, size=128, n_slices=8):
    """
    Load the BigBrainMR T1-weighted volume, block-averaged down to the matrix.

    Downsampled rather than cropped, unlike "load_ge_phantom". The volume
    is 388 x 480 x 408 at 0.4 mm and the brain fills it, so a centred crop of
    128 voxels would return a 51 mm block of white matter instead of a head.
    Averaging whole blocks — as opposed to striding — is what keeps that
    downsampling from aliasing the cortical detail it is being kept for.

    Windowed to the 99.9th percentile, not the maximum: T1w volumes carry
    isolated bright voxels that would otherwise compress the entire brain into
    the bottom of the range.
    """
    import gzip

    import nibabel as nib

    if not os.path.isfile(path):
        raise FileNotFoundError(f"BigBrainMR volume not found at {path}")

    img = nib.load(path)
    nx, ny, nz = img.shape
    dtype = img.header.get_data_dtype()
    target = (size, size, n_slices)
    factor = [max(1, s // t) for s, t in zip(img.shape, target)]
    keep = [f * t for f, t in zip(factor, target)]
    start = [(s - k) // 2 for s, k in zip(img.shape, keep)]

    # Streamed in one sequential pass rather than sliced out of "dataobj".
    # The 100 um volume is 1550 x 1920 x 1630 — 4.85 G voxels, 19 GB as float32,
    # so it cannot be materialised. Slicing it instead is worse: without
    # indexed_gzip every seek decompresses from the top of the file, and one
    # eight-slice read at z = 800 measured 27.7 s. Reading forward once and
    # folding each xy plane into its output slice as it arrives costs a single
    # pass and one plane of memory.
    #
    # The data offset comes from the array proxy, not from "vox_offset": these
    # files store 0 there, and reading from byte 0 silently parses the 352-byte
    # header as voxels.
    per_plane = nx * ny * np.dtype(dtype).itemsize
    acc = np.zeros((size, size, n_slices), np.float64)
    with gzip.open(path, 'rb') as fh:
        fh.read(int(img.dataobj.offset))
        for z in range(nz):
            plane = fh.read(per_plane)
            if not start[2] <= z < start[2] + keep[2]:
                continue
            # NIfTI stores x fastest, so a plane reads back as [y][x].
            sl = np.frombuffer(plane, dtype=dtype).reshape(ny, nx).T
            sl = sl[start[0]:start[0] + keep[0], start[1]:start[1] + keep[1]]
            acc[:, :, (z - start[2]) // factor[2]] += sl.astype(np.float32).reshape(
                size, factor[0], size, factor[1]).sum(axis=(1, 3))
    vol = acc / float(factor[0] * factor[1] * factor[2])

    hi = np.percentile(vol, 99.9)
    out = np.clip(vol / hi, 0.0, 1.0) if hi > 0 else vol
    return out.astype(np.float32)


#*************#
#   metrics   #
#*************#
#******************#
#   figure style   #
#******************#
# Appendix-figure conventions: no metrics printed on the panels (those live in
# metrics.csv and belong in a table, not burned into a PNG), lower-case (a) (b)
# panel letters so a caption can refer to them, no coloured chrome, and a
# consistent grey ramp. Titles stay short enough to read at column width.
def _bare_axes(ax):
    """Strip an image axis down to the image."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig, save_path, name, dpi=200, pdf_dpi=600):
    """
    PNG for viewing plus PDF (vector) for the paper.

    Line art and text come out as vectors in the PDF; image panels and any
    artist marked "rasterized=True" — the trajectory scatters — do not, and
    are resampled at save time. savefig defaults to "dpi='figure'", i.e. 100,
    so omitting it here emitted those parts of the PDF at half the resolution of
    the PNG and made the vector file look the softer of the two. Hence a dpi
    for the PDF as well, and a higher one, since it is the print target.
    """
    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, f'{name}.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(save_path, f'{name}.pdf'), dpi=pdf_dpi,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


def nrmse(reference, test):
    """Normalised RMSE in percent, relative to the reference's dynamic range."""
    reference, test = np.asarray(reference), np.asarray(test)
    denom = np.ptp(np.abs(reference))
    if denom == 0:
        return float('nan')
    return float(100.0 * np.sqrt(np.mean(np.abs(test - reference) ** 2)) / denom)


def ssim_like(reference, test):
    """
    Global structural similarity: the correlation of the two images.

    Not windowed SSIM — this is a single scalar for the whole image, which is what
    an ablation table wants. 1.0 means identical structure.
    """
    a = np.asarray(np.abs(reference), dtype=np.float64).ravel()
    b = np.asarray(np.abs(test), dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / denom) if denom > 0 else float('nan')


def _display_slice(vol):
    """
    The centre slice. Nothing else.

    Every "smarter" pick tried here has misfired on real data: "most distinct
    intensity levels" landed on a noisy end slice, and the energy-argmax
    fallback chose whichever slice happened to be brightest — on the GE
    phantom, whose intensity ramps along z, that was again an end slice
    instead of the structured middle. The centre is predictable, comparable
    across phantoms, and the full through-plane structure is in the slice
    montage figure anyway.
    """
    return np.abs(np.asarray(vol)).shape[2] // 2


def _to_nifti_layout(vol):
    """(X, Y, Z) real -> (1, X, Y, Z, 1) complex, the layout the modules expect."""
    return torch.as_tensor(vol, dtype=torch.float32)[None, ..., None].to(torch.complex64)


def _from_nifti_layout(t):
    return t[0, ..., 0].numpy()


#***************#
#   ablations   #
#***************#
# Each entry turns on exactly ONE augmentation with a fixed magnitude, so the
# measured change is attributable to that effect alone. Magnitudes are chosen to
# be clearly visible rather than subtle — this is a diagnostic, not a training
# config.
SPATIAL_ABLATIONS = (
    ('translate x',   dict(do_translate=True, tx=0.15)),
    ('translate y',   dict(do_translate=True, ty=0.15)),
    ('translate z',   dict(do_translate=True, tz=0.25)),
    ('rotate 20 deg', dict(do_z_rot=True, z_angle_deg=20.0)),
    ('zoom in 1.25x', dict(do_zoom=True, zoom_xyz=(1.25, 1.25, 1.25))),
    ('zoom out 0.8x', dict(do_zoom=True, zoom_xyz=(0.8, 0.8, 0.8))),
    ('shear xy 0.15', dict(do_shear=True, shear_xy=(0.15, 0.0))),
    ('flip x',        dict(do_flip=True, flip_x=True)),
    ('flip y',        dict(do_flip=True, flip_y=True)),
    ('flip z',        dict(do_flip=True, flip_z=True)),
    ('anisotropic',   dict(do_anisotropic=True, zoom_xyz=(1.2, 0.85, 1.0))),
)


def ablate_spatial(vol, config, pixdim=None):
    """Apply each spatial augmentation alone. Returns [(label, image, metrics)]."""
    aug = SpatialAugmentations(dim=3, prob=0.0, padding_mode='zeros',
                               pixdim=pixdim, allow_rot90=False)
    x = _to_nifti_layout(vol)
    mid = _display_slice(vol)
    results = [('original', vol[:, :, mid], dict(nrmse=0.0, corr=1.0))]

    for label, overrides in SPATIAL_ABLATIONS:
        spec = aug.sample_augmentations(1)[0]     # prob=0 -> an all-off spec
        spec.update(overrides)
        out, _ = aug.apply(x, aug_spec_list=[spec])
        img = _from_nifti_layout(out).real
        results.append((label, img[:, :, mid],
                        dict(nrmse=nrmse(vol, img), corr=ssim_like(vol, img))))
    return results


def ablate_sampling(vol, config):
    """Undersample at a range of accelerations. Returns [(label, image, metrics)]."""
    x = _to_nifti_layout(vol)
    mid = _display_slice(vol)
    results = [('fully sampled', vol[:, :, mid], dict(nrmse=0.0, corr=1.0, kept=1.0))]

    for accel in config['accelerations']:
        if accel <= 1.0:
            continue
        us = KspaceUndersampling(ksp_mode='cartesian', acceleration_factor=accel,
                                 acs_frac=0.06, us_seed=config['seed'])
        out, _ = us.process_tensor(x.clone())
        img = np.abs(_from_nifti_layout(out))
        kept = float(us.last_masks_[0].mean())
        results.append((f'{accel:g}x', img[:, :, mid],
                        dict(nrmse=nrmse(vol, img), corr=ssim_like(vol, img), kept=kept)))
    return results


# Every trajectory in the library, split by the dimensionality of the coordinates
# it generates. 2-D trajectories describe one plane and are replicated along z;
# 3-D ones sample the volume directly.
TRAJECTORIES_2D = (
    'cartesian_2d', 'radial_2d', 'spiral_2d',
    'rosette_2d_petals', 'concentric_rings_2d',
)
# The 3-D family mirrors the 2-D one: cartesian<->cartesian, stacks carry the
# 2-D patterns along kz (stars<->radial, spirals, ECCENTRIC<->rings,
# rosettes), and the true 3-D members are the volumetric analogues — spokes
# (phyllotaxis-ordered), cones (spiral arms on cone surfaces) and rosettes
# (petals through the origin on cone surfaces). floret_3d and 3d_egg_rosette
# are exotic variants that add nothing to the comparison, and stack_of_cones
# was removed from the library (it was a byte-identical alias for
# stack_of_spirals — a stack of 2-D spirals is not a cones acquisition).
# Two 3-D families, each mirroring the 2-D one member-for-member.
#
# STACK: the literal 2-D pattern repeated along kz — cartesian<->cartesian,
# stars<->radial, spirals, rosettes, rings<->concentric rings.
#
# OTHER: every remaining 3-D trajectory in the library — the true-3-D
# patterns that fill the ball radially (cones_3d = 3-D spiral,
# cones_3d_rosette = 3-D rosette, concentric_shells_3d = rings nested on
# spheres, floret_3d, 3d_egg_rosette) plus phyllotaxis spokes (3-D radial),
# placed last. Cartesian appears only in the stack family.
TRAJECTORIES_3D_STACK = (
    'cartesian_3d', 'stack_of_stars', 'stack_of_spirals',
    'stack_of_rosettes', 'stack_of_rings',
)
# The user-selected non-stack 3-D family: cones, cones-rosette, the (fixed,
# petal-based) egg rosette, floret, and phyllotaxis last. concentric_shells_3d
# and cartesian_3d remain available in the library.
TRAJECTORIES_3D_OTHER = (
    'cones_3d', 'cones_3d_rosette', '3d_egg_rosette',
    'floret_3d', '3d_phyllotaxis',
)


def _grid_coverage(shots, shot_mask, meta):
    """Fraction of Cartesian bins the retained shots actually touch."""
    matrix = tuple(int(v) for v in meta['matrix'])
    fov_m = tuple(f / 1000.0 for f in meta['fov_mm'])
    grid = GridMask.rasterize_shots_to_grid(shots, shot_mask, fov_m, matrix)
    return grid, float(grid.mean())


# Trajectories whose successive shots advance by the golden angle. For these a
# PREFIX of shots is near-uniform — that is the design property of golden
# ordering — while an every-Mth stride multiplies the golden step by M, which
# lands near a rational multiple of pi and clusters the kept shots into a few
# wedges (stride 5 of the golden sequence bunches into ~11 angular groups).
# Linearly-ordered trajectories are the mirror image: a prefix keeps a
# contiguous wedge and even decimation stays uniform. So the decimation rule
# must follow the ordering, per trajectory.
GOLDEN_ORDERED = {'3d_phyllotaxis', '3d_egg_rosette'}
GOLDEN_INPLANE = set()

# Per-trajectory generation parameters for this comparison. Spirals use linear
# interleaving here: arm i rotated by 2*pi*i/n gives IDENTICAL spacing between
# adjacent arms, and even decimation preserves that exactly. (Golden ordering
# is the right choice for prefix-style dynamic imaging, but this figure asks
# what the geometry does at fixed density, so uniform spacing is the fairer
# and more readable variant.)
TRAJ_PARAMS = {
    'spiral_2d':        {'ordering': 'linear'},
    'stack_of_spirals': {'ordering': 'linear'},
}


def _even_indices(count, keep):
    """*keep* indices spread evenly over range(*count*)."""
    return (np.arange(keep) * count / keep).astype(int)


def _decimation_mask(name, meta, n_total, accel):
    """
    Shot mask keeping ~n/accel shots, respecting the trajectory's ordering.

    Stacks are decimated per kz-slice. Their shot list is slice-major, so any
    global rule couples the two orderings: a global prefix keeps only the
    bottom slices, and a global stride walks the in-plane index by M with a
    slice-dependent offset — on stack_of_stars that turned the spokes into
    drifting angular clumps instead of a homogeneous star on every slice.
    Decimating each slice independently, with the in-plane ordering rule,
    keeps the same spoke pattern on every slice by construction.
    """
    tt = str(meta.get('trajectory_type', name))
    if 'shots_per_shell' in meta:
        per = list(meta['shots_per_shell'])
        n_shells = len(per)
        keep = max(1, min(n_shells, int(round(n_shells / accel))))
        starts = np.concatenate([[0], np.cumsum(per)])
        mask = np.zeros(n_total, dtype=bool)
        for si in _even_indices(n_shells, keep):
            mask[starts[si]:starts[si + 1]] = True
        return mask
    if tt.startswith('stack_of_'):
        n_inplane = int(meta['n_inplane_shots'])
        n_slices = int(meta['n_kz_slices'])
        keep = max(1, min(n_inplane, int(round(n_inplane / accel))))
        if meta.get('inplane_trajectory') in GOLDEN_INPLANE:
            idx = np.arange(keep)
        else:
            idx = _even_indices(n_inplane, keep)
        mask = np.zeros(n_total, dtype=bool)
        for sl in range(n_slices):
            mask[sl * n_inplane + idx] = True
        return mask

    keep = max(1, min(n_total, int(round(n_total / accel))))
    if name in GOLDEN_ORDERED:
        idx = np.arange(keep)
    else:
        idx = _even_indices(n_total, keep)
    mask = np.zeros(n_total, dtype=bool)
    mask[idx] = True
    return mask


def _match_coverage(name, header, target, config,
                    max_shots=4096, tol=0.02, min_shots=32):
    """
    Find a trajectory + shot mask whose retained BIN coverage is near *target*.

    A shared "acceleration_factor" does not make trajectories comparable: it
    undersamples shots, and at full sampling these cover anywhere from 0.9% of
    the grid (floret_3d at 256 shots) to 38% (cartesian_2d). Comparing them at
    one nominal acceleration compares mostly how many bins each happened to
    reach. So instead: grow the shot count until the trajectory *can* reach the
    target, then bisect the acceleration factor until the bins it retains land
    on it.

    Returns (shots, mask, meta, achieved_coverage, n_shots, acceleration).
    """
    from augmentrum.sampling import ShotUndersampler, TrajectoryRegistry

    # Start from the generator's OWN default shot count and only grow if that
    # cannot reach the target. Forcing a shot count breaks trajectories whose
    # geometry fixes it: cartesian_2d puts one shot per phase-encode line, so
    # asking for 256 on a 96-wide grid sends 160 of them off the grid entirely,
    # and the rasteriser silently drops every one.
    n_shots = None
    shots = meta = None
    prev_actual = -1
    while True:
        params = dict(TRAJ_PARAMS.get(name, {}))
        if n_shots is not None:
            params['n_shots'] = n_shots
        shots, meta = TrajectoryRegistry.generate(name, header, params)
        full = np.ones(len(shots), dtype=bool)
        _, cov_full = _grid_coverage(shots, full, meta)
        actual = len(shots)

        # Two conditions have to hold before the bisection can do anything.
        #
        #   coverage >= target  — obvious: you cannot decimate down to 25% of
        #       the grid from a trajectory that only ever touches 8% of it.
        #
        #   shots >= min_shots  — less obvious, and it is what made spiral_2d
        #       four times denser than everything else. Undersampling drops
        #       whole shots, so a single-shot trajectory has exactly two
        #       reachable states: all of it or none of it. spiral_2d defaults to
        #       one arm covering 65% of the grid, the bisection could not move
        #       it, and the panel came out at 65% next to everyone else's 25%.
        #       Ask for enough shots that decimation has something to bite on.
        if actual >= max_shots or actual <= prev_actual:
            n_shots = actual
            break
        if cov_full >= target and actual >= min_shots:
            n_shots = actual
            break

        prev_actual = actual
        want = actual
        if actual < min_shots:
            want = max(want, min_shots)
        if cov_full < target:
            # Coverage grows sublinearly (shots overlap), so step generously.
            want = max(want, int(actual * max(2.0, target / max(cov_full, 1e-6))))
        n_shots = min(max_shots, max(want, actual + 1))

    if cov_full < target:      # cannot reach it even fully sampled
        return shots, np.ones(len(shots), bool), meta, cov_full, n_shots, 1.0

    # Bisect on acceleration: more acceleration -> fewer shots -> less coverage.
    lo, hi = 1.0, 64.0
    best = (np.ones(len(shots), bool), cov_full, 1.0)
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        mask = _decimation_mask(name, meta, len(shots), mid)
        _, cov = _grid_coverage(shots, mask, meta)
        if abs(cov - target) < abs(best[1] - target):
            best = (mask, cov, mid)
        if abs(cov - target) <= tol:
            break
        if cov > target:
            lo = mid
        else:
            hi = mid
    mask, cov, accel = best
    return shots, mask, meta, cov, n_shots, accel


def ablate_trajectory_family(vol, config, names, family):
    """
    Compare a family of trajectories at MATCHED bin coverage.

    Returns [(label, image, mask, metrics)].
    """
    from augmentrum.sampling import KspaceGeometry

    target = float(config['coverage_target'])
    x = _to_nifti_layout(vol)
    mid = _display_slice(vol)
    nx, ny, nz = vol.shape
    header = {
        "dim":    [4, nx, ny, nz, 1, 1, 1, 1],
        "pixdim": [1.0] + list(config['voxel_mm']) + [1.0, 1.0, 1.0, 1.0],
        "DwellTime": 0.83e-3,
        "SpectrometerFrequency": [127.732434e6],
    }

    # The fully-sampled phantom leads each family, so every panel is read
    # against the thing it is trying to reproduce.
    coord_entries = []
    results = [('fully sampled', vol[:, :, mid], np.ones(vol.shape[:2], bool),
                dict(nrmse=0.0, corr=1.0, kept=1.0, shots=0, accel=1.0, family=family))]
    for name in names:
        try:
            shots, mask, meta, cov, n_shots, accel = _match_coverage(
                name, header, target, config)
            grid, _ = _grid_coverage(shots, mask, meta)
            while grid.ndim < 3:                     # 2-D mask -> replicate along z
                grid = grid[..., None]
            grid = np.broadcast_to(grid, (nx, ny, nz))

            k = np.fft.fftshift(np.fft.fftn(x[0, ..., 0].numpy(), axes=(0, 1, 2),
                                            norm='ortho'), axes=(0, 1, 2))
            img = np.abs(np.fft.ifftn(np.fft.ifftshift(k * grid, axes=(0, 1, 2)),
                                      axes=(0, 1, 2), norm='ortho'))
        except Exception as exc:
            print(f"     {name:22s} skipped: {type(exc).__name__}: {str(exc)[:50]}")
            continue

        keep = np.flatnonzero(np.asarray(mask, dtype=bool))
        pts = np.concatenate([np.atleast_2d(np.asarray(shots[int(k)])) for k in keep]) \
            if keep.size else np.zeros((1, 3))
        coord_entries.append((name, pts))

        results.append((name, img[:, :, mid], grid,
                        dict(nrmse=nrmse(vol, img), corr=ssim_like(vol, img),
                             kept=cov, shots=n_shots, accel=accel, family=family)))
    return results, coord_entries


def ablate_trajectories(vol, config):
    """
    Compare Cartesian masks against masks rasterised from real trajectories.

    Reported acceleration is the SHOT-level figure requested; the retained bin
    fraction is measured separately because for a non-Cartesian trajectory the
    two are different numbers.
    """
    x = _to_nifti_layout(vol)
    mid = _display_slice(vol)
    trajectories = [
        ('cartesian VD',  dict(ksp_mode='cartesian')),
        ('golden radial', dict(ksp_mode='gridded', trajectory='golden_radial_2d',
                               undersampling='prefix')),
        ('spiral',        dict(ksp_mode='gridded', trajectory='spiral_2d',
                               undersampling='variable_density_spiral')),
        ('spiral linear', dict(ksp_mode='gridded', trajectory='spiral_2d',
                               undersampling='variable_density_spiral',
                               traj_params_extra={'ordering': 'linear'})),
        ('rosette',       dict(ksp_mode='gridded', trajectory='rosette_2d',
                               undersampling='prefix')),
        ('conc. rings',   dict(ksp_mode='gridded', trajectory='concentric_rings_2d',
                               undersampling='prefix')),
    ]
    results = []
    for label, kwargs in trajectories:
        # Only n_shots is set here. samples_per_shot is left to each generator's
        # own default, which is tuned per trajectory — forcing one value on all of
        # them under-samples the ones that need more (rosette defaults to 512).
        traj_params = {'n_shots': 256}
        traj_params.update(kwargs.pop('traj_params_extra', {}))
        us = KspaceUndersampling(acceleration_factor=4.0, acs_frac=0.06,
                                 us_seed=config['seed'], traj_params=traj_params,
                                 **kwargs)
        try:
            out, _ = us.process_tensor(x.clone())
        except Exception as exc:                  # incompatible traj/method pairs
            print(f"    {label:16s} skipped: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        img = np.abs(_from_nifti_layout(out))
        mask = us.last_masks_[0]
        results.append((label, img[:, :, mid], mask,
                        dict(nrmse=nrmse(vol, img), corr=ssim_like(vol, img),
                             kept=float(mask.mean()))))
    return results


#*******************#
#   nufft support   #
#*******************#
# The reconstructor needs torchkbnufft, which is optional. Everything below is
# skipped rather than fatal when it is absent, so the rest of the figures still
# build on a machine without it.
def _nufft_available():
    try:
        import torchkbnufft  # noqa: F401
    except ImportError:
        return False
    return True


def _retained_coords(shots, mask, meta):
    """
    Retained samples as normalised "[1, 1, D, K]" coordinates in the unit box.

    Every retained shot is concatenated into a single readout. Shots are kept
    separate elsewhere because undersampling drops whole shots, but that has
    already happened by the time we get here, and trajectories like
    "concentric_rings_2d" give each shot a different length — a ring near the
    centre needs fewer samples than one at the edge — so there is no rectangular
    "[S, L]" to stack them into. The NUFFT flattens the two axes anyway.
    """
    from augmentrum.sampling import KspaceReconstructor

    keep = np.flatnonzero(np.asarray(mask, dtype=bool))
    pts = np.concatenate([np.atleast_2d(np.asarray(shots[int(i)])) for i in keep])
    coords = torch.from_numpy(pts.T).float()[None, None]           # [1, 1, D, K]
    ndim = coords.shape[2]
    return KspaceReconstructor.normalise_trajectory(coords, meta['kmax'][:ndim])


def _nearest_bin_recon(coords, kdata, n):
    """
    Reconstruct by snapping every measured sample to its nearest Cartesian bin.

    This is the honest counterpart to the NUFFT: identical measurements, but each
    one is rounded onto the grid and bins that collect several samples average
    them. It is what "GridMask.rasterize_shots_to_grid" implies, and the
    approximation the reconstructor exists to avoid.

    Note this is NOT what "ksp_mode='gridded'" does today. That path uses the
    trajectory only to decide which Cartesian bins to keep and then masks the
    image's own FFT, so it never leaves the grid and never sees an off-grid
    sample value at all.
    """
    idx = torch.round(coords[0, 0] * (n / 2.0) + n / 2.0).long().clamp(0, n - 1)
    flat = (idx[0] * n + idx[1]).numpy()
    values = kdata.reshape(-1).numpy()

    grid = np.zeros(n * n, dtype=np.complex128)
    count = np.zeros(n * n, dtype=np.float64)
    np.add.at(grid, flat, values)
    np.add.at(count, flat, 1.0)
    grid[count > 0] /= count[count > 0]

    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid.reshape(n, n)))))


def _measure(image, coords, osf):
    """
    Forward NUFFT: the k-space samples a scanner would collect along *coords*.

    Using the exact NUFFT rather than our own interpolator is deliberate — it
    keeps this an honest test of the RECONSTRUCTION, instead of folding the
    interpolator's error into the measurement. The interpolator is characterised
    separately in "ablate_interpolator".
    """
    import torchkbnufft as tkbn
    from augmentrum.sampling import KspaceReconstructor

    im_size = tuple(int(s) for s in image.shape)
    if len(im_size) != coords.shape[2]:
        raise ValueError(f"image is {len(im_size)}-D but coords are {coords.shape[2]}-D.")
    grid = tuple(int(osf * s) for s in im_size)
    ktraj, _ = KspaceReconstructor.flatten(coords)
    flat = tkbn.KbNufft(im_size=im_size, grid_size=grid)(
        torch.from_numpy(image).to(torch.complex64)[None, None], ktraj)
    return flat.reshape(1, 1, coords.shape[1], coords.shape[3]).permute(0, 2, 1, 3)


def _ls_scale(rec, reference):
    """
    Least-squares scale factor between a reconstruction and its reference.

    Gridding recovers the image only up to a constant, and that constant differs
    per trajectory (it depends on the density weights). Scaling by the max
    instead would let a single ringing overshoot compress the whole image and
    show up as reconstruction error that is really a windowing artefact.
    """
    rec, reference = np.asarray(rec), np.asarray(reference)
    denom = float(np.vdot(rec.ravel(), rec.ravel()).real)
    if denom <= 0:
        return rec
    return rec * (float(np.vdot(rec.ravel(), reference.ravel()).real) / denom)


def ablate_nufft(vol, config):
    """
    Nearest-bin gridding against a true NUFFT, from the same measured samples.

    This is the seam between the two halves of the module. "KspaceUndersampling"
    in "gridded" mode snaps every trajectory sample to its nearest Cartesian bin
    and runs an ordinary inverse FFT; the reconstructor instead evaluates the
    adjoint at the coordinates the samples were actually taken at. Both start
    from an identical set of measurements here, so the difference between the two
    panels is the cost of that snapping and nothing else.

    Returns [(label, gridded, nufft, metrics)].
    """
    from augmentrum.sampling import KspaceGeometry, KspaceReconstructor

    target = float(config['coverage_target'])
    osf = float(config['nufft_osf'])
    mid = _display_slice(vol)
    img = np.abs(vol[:, :, mid]).astype(np.float32)
    nx, ny, nz = vol.shape
    header = {
        "dim":    [4, nx, ny, nz, 1, 1, 1, 1],
        "pixdim": [1.0] + list(config['voxel_mm']) + [1.0, 1.0, 1.0, 1.0],
        "DwellTime": 0.83e-3,
        "SpectrometerFrequency": [127.732434e6],
    }

    results = []
    for name in TRAJECTORIES_2D:
        try:
            shots, mask, meta, cov, n_shots, accel = _match_coverage(
                name, header, target, config)
            coords = _retained_coords(shots, mask, meta)
            kdata = _measure(img, coords, osf)

            # Same measurements, two reconstructions: snap to the nearest bin,
            # or evaluate the adjoint where the samples were actually taken.
            gridded = _nearest_bin_recon(coords, kdata, nx)
            recon = KspaceReconstructor(image_size=(nx, ny), oversampling_factor=osf)
            nufft = np.abs(recon(coords, kdata)[0, 0].numpy())
        except Exception as exc:
            print(f"     {name:22s} skipped: {type(exc).__name__}: {str(exc)[:50]}")
            continue

        gridded = _ls_scale(gridded, img)
        nufft = _ls_scale(nufft, img)
        pattern = coords[0, 0].numpy().T          # [K, D], normalised to the unit box
        results.append((name, pattern, gridded, nufft,
                        dict(nrmse=nrmse(img, gridded), nrmse_nufft=nrmse(img, nufft),
                             corr=ssim_like(img, gridded), corr_nufft=ssim_like(img, nufft),
                             kept=cov, shots=n_shots, accel=accel)))
    return results


def ablate_interpolator(vol, config):
    """
    How accurately the cubic interpolator resamples k-space, against oversampling.

    The interpolator is the forward half of the pair, so its job is to read a
    gridded volume at off-grid trajectory positions. On a critically sampled
    k-space that is a hard ask: k-space of a sharp-edged object is not smooth at
    the bin scale, which is exactly why a NUFFT grids onto an oversampled lattice
    with a Kaiser-Bessel kernel rather than interpolating the raw grid. Zero-pad
    the image and the same interpolator converges quickly, so the sweep below is
    really a statement about how much oversampling the kernel needs.

    Returns [(oversampling, relative_error)].
    """
    import torchkbnufft as tkbn
    from augmentrum.processing import BicubicHermiteMAkima2D
    from augmentrum.sampling import KspaceReconstructor, TrajectoryRegistry

    mid = _display_slice(vol)
    img = np.abs(vol[:, :, mid]).astype(np.float32)
    n = img.shape[0]
    header = {
        "dim":    [4, n, n, 1, 1, 1, 1, 1],
        "pixdim": [1.0] + list(config['voxel_mm']) + [1.0, 1.0, 1.0, 1.0],
        "DwellTime": 0.83e-3,
    }
    shots, meta = TrajectoryRegistry.generate('radial_2d', header,
                                              {'n_shots': 201})
    coords = _retained_coords(shots, np.ones(len(shots), bool), meta)
    exact = _measure(img, coords, float(config['nufft_osf']))

    rows = []
    for osf in config['interp_osf']:
        m = int(n * osf)
        pad = (m - n) // 2
        padded = np.zeros((m, m), np.float32)
        padded[pad:pad + n, pad:pad + n] = img
        kgrid = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))).astype(np.complex64)

        # Normalised k in [-1, 1] spans +-kmax. On the padded grid the bin
        # spacing shrinks by osf, so that half-extent is m/2 bins, not n/2 —
        # and the interpolator's own axis runs over m points, hence the (m - 1).
        axis = 2.0 * (coords * (m / 2.0) + m / 2.0) / (m - 1.0) - 1.0
        got = BicubicHermiteMAkima2D(torch.from_numpy(kgrid)[None, None])(axis)
        got = got.reshape(exact.shape)
        rows.append((osf, float((got - exact).abs().mean() / exact.abs().mean())))
    return rows


#*******************#
#   nufft panel     #
#*******************#
# The trajectories the full-page panel sweeps, and the labels it prints. Labels
# are set here rather than reusing the registry keys: a printed figure reads
# better with words than identifiers, and the underscores survive badly at panel
# size. Shot counts are per trajectory because a shot count is tied to geometry
# — cartesian_2d puts one shot on each phase-encode line, so asking it for more
# than the matrix holds sends the surplus off the grid to be dropped. The 3-D
# entries are sized to stay affordable; stack_of_stars is deliberately absent,
# since its count is per kz slice and reaches 24576 shots on this grid.
# Shot counts are multiples of 24, the lowest common multiple of the panel's
# acceleration factors. Undersampling keeps every (n / R)-th shot, so when R
# does not divide the shot count the retained gaps come out mixed — 201 spokes
# at R = 6 alternates between 5 and 6, which shows up as visibly uneven spacing
# in the sampling row and is a defect in the experiment, not just the picture.
# Ordered by how far each departs from a Cartesian raster: straight lines, then
# closed rings, then lines through the centre, then one continuous curve, then
# overlapping petals. Reading left to right is reading up that progression.
PANEL_2D = (
    ('cartesian_2d',        'cartesian',     {}),
    ('concentric_rings_2d', 'rings',         {'n_shots': 48}),
    ('radial_2d',           'radial',        {'n_shots': 192}),
    ('spiral_2d',           'spiral',        {'n_shots': 96, 'ordering': 'linear'}),
    ('rosette_2d_petals',   'rosette',       {'n_shots': 96}),
)
# The three ways of taking a 2-D pattern to 3-D, all built from the rosette that
# already has a column in the 2-D block: stack it along kz, wrap it onto cone
# surfaces, or distribute its petals over a sphere. Picking three variants of one
# in-plane pattern rather than three unrelated trajectories is what makes the
# comparison mean something — the only thing differing is how kz is covered.
PANEL_3D = (
    ('stack_of_rosettes',   'stacked',       {'n_shots': 24}),
    ('cones_3d_rosette',    'on cones',      {'n_shots': 864}),
    ('3d_egg_rosette',      'on a sphere',   {'n_shots': 984}),
)
PANEL_TRAJECTORIES = PANEL_2D + PANEL_3D

def _pattern_shots(shots, mask, meta):
    """
    Every retained sample, undrawn and unthinned.

    Thinning for legibility was tried three ways and all three lied about the
    sampling density. A fixed cap made every acceleration draw the same 30 shots
    for the 3-D trajectories, so the sampling row never changed while the
    reconstructions below it visibly degraded. Scaling the cap with the retained
    fraction fixed that but still drew only 32 of Cartesian's 96 lines at R = 1,
    so a FULLY SAMPLED trajectory came out looking undersampled. And any
    fixed-count spread aliases against the shot index — 32 of 48 rings gives gaps
    of 1, 2, 1, 2 and renders the rings in pairs.

    Drawing everything removes the question. Density on the page is then density
    in k-space: Cartesian at R = 1 is a solid square, and each doubling of R
    visibly halves it. The panels are rasterised, so the point count costs file
    size rather than vector complexity.
    """
    return _retained_coords(shots, mask, meta)[0, 0].numpy().T


def _nufft_cell(volume, shots, mask, meta, osf):
    """
    NUFFT reconstruction of *volume* from the retained shots of one trajectory.

    A 2-D trajectory says nothing about kz, so it is measured and reconstructed
    on the single displayed slice; a 3-D one samples the volume directly and the
    same slice is taken afterwards. Both return a 2-D image.
    """
    from augmentrum.sampling import KspaceReconstructor

    coords = _retained_coords(shots, mask, meta)
    ndim = coords.shape[2]
    nx, ny, nz = volume.shape
    mid = _display_slice(volume)

    if ndim == 2:
        target = np.abs(volume[:, :, mid]).astype(np.float32)
        im_size = (nx, ny)
    else:
        target = np.abs(volume).astype(np.float32)
        im_size = (nx, ny, nz)

    rec = np.abs(KspaceReconstructor(im_size, osf)(
        coords, _measure(target, coords, osf))[0, 0].numpy())
    rec = _ls_scale(rec, target)
    return (rec, target) if ndim == 2 else (rec[:, :, mid], target[:, :, mid])


def ablate_nufft_panel(vol, config):
    """
    Every panel trajectory at every acceleration, reconstructed by NUFFT.

    Returns (columns, cells): columns carries each trajectory's label and its
    fully-sampled pattern, cells maps (acceleration, trajectory) to the
    reconstruction, the pattern actually retained at that acceleration, and the
    metrics.
    """
    from augmentrum.sampling import TrajectoryRegistry

    nx, ny, nz = vol.shape
    header = {
        "dim":    [4, nx, ny, nz, 1, 1, 1, 1],
        "pixdim": [1.0] + list(config['voxel_mm']) + [1.0, 1.0, 1.0, 1.0],
        "DwellTime": 0.83e-3,
        "SpectrometerFrequency": [127.732434e6],
    }
    osf = float(config['nufft_osf'])

    columns, cells = [], {}
    bar = tqdm(total=len(PANEL_TRAJECTORIES) * len(config['panel_accelerations']),
               desc='     measuring', unit='cell', leave=False)
    for name, label, params in PANEL_TRAJECTORIES:
        shots, meta = TrajectoryRegistry.generate(name, header, dict(params))
        full = np.ones(len(shots), bool)
        columns.append((name, label, _pattern_shots(shots, full, meta)))

        for accel in config['panel_accelerations']:
            mask = full if accel <= 1.0 else _decimation_mask(name, meta, len(shots), accel)
            rec, target = _nufft_cell(vol, shots, mask, meta, osf)
            _, cov = _grid_coverage(shots, mask, meta)
            err = nrmse(target, rec)
            cells[(accel, name)] = (rec, _pattern_shots(shots, mask, meta),
                                    dict(trajectory=name, acceleration=accel,
                                         nrmse=err, kept=cov, shots=int(mask.sum())))
            bar.set_postfix_str(f"{label} R={accel:g}")
            bar.update(1)
            bar.write(f"     {label:<14} {accel:>4.0f}x  shots {int(mask.sum()):>5}  "
                      f"bins {100 * cov:5.1f}%  NRMSE {err:6.2f}%")
    bar.close()
    return columns, cells


# The panels are saved at a higher resolution than the other figures, and the
# reason is entirely the sampling row. An image panel holds --size real samples,
# so extra dpi only resamples data that is not there; the trajectory scatter is
# continuous geometry and does get sharper. Measured on one rosette panel, the
# imshow accounts for 7% of the file and the scatter for the rest, so the cost
# of the extra resolution buys the dots and almost nothing else.
_PANEL_DPI = {'dpi': 300, 'pdf_dpi': 1200}


def _pattern_axes(fig, gs_cell, pattern):
    """
    Draw one sampling pattern, in 2-D or 3-D according to what it is.

    A 3-D trajectory projected onto kx-ky is close to useless — an egg rosette
    and a cone rosette both collapse to a filled disc — so 3-D columns get real
    3-D axes.
    """
    pts = np.asarray(pattern, dtype=np.float64)

    if pts.shape[1] < 3:
        ax = fig.add_subplot(gs_cell)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.30, c='#333333',
                   alpha=0.60, linewidths=0, rasterized=True)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax = fig.add_subplot(gs_cell, projection='3d')
        # Lighter than the 2-D panels because a 3-D trajectory carries an
        # order of magnitude more samples; matching their alpha turns the
        # fully-sampled sphere into a solid disc.
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.16, c='#333333',
                   alpha=0.16, linewidths=0, rasterized=True)
        for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            lim(-1.05, 1.05)
        ax.view_init(elev=22, azim=38)
        # 3-D axes reserve a wide internal margin, so at the default zoom the
        # cloud sits noticeably smaller than the 2-D panels beside it. zoom
        # fills the cell without touching the data limits.
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.set_axis_off()          # hides panes, grid AND the axis lines
    return ax


def _panel_grid_spec(fig, n_row, hspace=0.05):
    """Gridspec with a narrow spacer between the 2-D and 3-D trajectory blocks."""
    n_2d = len(PANEL_2D)
    widths = [1.0] * n_2d + [0.22] + [1.0] * len(PANEL_3D)
    return fig.add_gridspec(n_row, len(PANEL_TRAJECTORIES) + 1,
                            width_ratios=widths, hspace=hspace, wspace=0.04)


def _panel_column(c):
    """Grid column for trajectory *c*, skipping the spacer."""
    return c if c < len(PANEL_2D) else c + 1


def _label_families(fig, gs, pad_pt=17.0):
    """
    Write '2-D' and '3-D' over their column blocks.

    The gap between the blocks already separates them, but a reader meeting the
    figure cold should not have to infer what the gap means.

    The offset is given in points and converted, not as a figure fraction: the
    two panels differ in height by more than 50%, and a fixed fraction that
    clears the column titles on one of them lands on top of them on the other.
    """
    n_2d = len(PANEL_2D)
    y_pad = pad_pt / (fig.get_figheight() * 72.0)
    # The 3-D block is three variants of ONE in-plane pattern, so the family
    # header names it and the column labels carry only what distinguishes them.
    for label, lo, hi in (('2-D', 0, n_2d),
                          ('3-D rosette', n_2d + 1, len(PANEL_TRAJECTORIES) + 1)):
        box = gs[0, lo:hi].get_position(fig)
        fig.text(0.5 * (box.x0 + box.x1), box.y1 + y_pad, label,
                 ha='center', va='bottom', fontsize=11)


def plot_nufft_panel(columns, cells, config, reference, save_path, name, insets=False):
    """
    Trajectory across, acceleration down, sampling pattern on top.

    Every reconstruction shares one display window taken from the reference, so
    a cell that looks worse is worse rather than windowed differently. Metrics
    stay out of the panels and go to metrics.csv — a paper figure that has to be
    read at 100% to see its own annotations is not doing its job.

    With *insets*, each 2-D sampling panel carries a zoomed window onto a few
    Cartesian bins with the grid drawn in, as in the nufft_ablation figure. The
    3-D columns are left alone: their panels are a projection of a shell, and a
    bin grid means nothing there.
    """
    accels = list(config['panel_accelerations'])
    n_row = len(accels) + 1
    fig = plt.figure(figsize=(1.5 * (len(columns) + 0.25), 1.58 * n_row))
    gs = _panel_grid_spec(fig, n_row)
    vmax = np.percentile(np.abs(reference), 99.9) or 1.0
    n_bins = int(np.asarray(reference).shape[0])

    for c, (traj, label, pattern) in enumerate(
            tqdm(columns, desc=f'     drawing {name}', unit='col', leave=False)):
        gc = _panel_column(c)
        ax = _pattern_axes(fig, gs[0, gc], pattern)
        ax.set_title(label, fontsize=9, pad=4)
        if insets and np.asarray(pattern).shape[1] < 3:
            _draw_zoom_inset(ax, np.asarray(pattern), n_bins)
        if c == 0:
            ax.set_ylabel('sampling', fontsize=9)
            ax.yaxis.set_visible(True)

        for r, accel in enumerate(accels, start=1):
            ax = fig.add_subplot(gs[r, gc])
            ax.imshow(np.rot90(np.abs(cells[(accel, traj)][0])), cmap='gray',
                      vmin=0, vmax=vmax, interpolation='nearest')
            _bare_axes(ax)
            if c == 0:
                ax.set_ylabel(f'$R$ = {accel:g}', fontsize=10)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

    _label_families(fig, gs)
    return _save(fig, save_path, name, **_PANEL_DPI)


def plot_nufft_panel_paired(columns, cells, config, reference, save_path, name,
                            insets=False):
    """
    The same sweep, but each acceleration shows what was sampled AND what came back.

    Two rows per acceleration: the shots actually retained, then the image they
    reconstruct. Taller than the compact panel, and worth it when the point is
    *why* a trajectory degrades — the rings thinning out, the spokes fanning
    apart — rather than only that it does.
    """
    accels = list(config['panel_accelerations'])
    n_row = 2 * len(accels)
    fig = plt.figure(figsize=(1.5 * (len(columns) + 0.25), 1.5 * n_row))
    gs = _panel_grid_spec(fig, n_row, hspace=0.04)
    vmax = np.percentile(np.abs(reference), 99.9) or 1.0
    n_bins = int(np.asarray(reference).shape[0])

    for c, (traj, label, _full) in enumerate(
            tqdm(columns, desc=f'     drawing {name}', unit='col', leave=False)):
        gc = _panel_column(c)
        for i, accel in enumerate(accels):
            rec, pattern, _m = cells[(accel, traj)]

            ax = _pattern_axes(fig, gs[2 * i, gc], pattern)
            if insets and np.asarray(pattern).shape[1] < 3:
                _draw_zoom_inset(ax, np.asarray(pattern), n_bins)
            if i == 0:
                ax.set_title(label, fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(f'$R$ = {accel:g}\nk-space', fontsize=8)
                ax.yaxis.set_visible(True)

            ax = fig.add_subplot(gs[2 * i + 1, gc])
            ax.imshow(np.rot90(np.abs(rec)), cmap='gray', vmin=0, vmax=vmax,
                      interpolation='nearest')
            _bare_axes(ax)
            if c == 0:
                ax.set_ylabel(f'$R$ = {accel:g}\nimage', fontsize=8)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

    _label_families(fig, gs)
    return _save(fig, save_path, name, **_PANEL_DPI)


def _rms_extents(img):
    """Intensity-weighted RMS width along x and y, in voxels."""
    w = np.abs(np.asarray(img, dtype=np.float64))
    total = w.sum()
    if total <= 0:
        return float('nan'), float('nan')
    xs, ys = np.arange(w.shape[0]), np.arange(w.shape[1])
    px, py = w.sum(axis=1) / total, w.sum(axis=0) / total
    cx, cy = (px * xs).sum(), (py * ys).sum()
    return (float(np.sqrt((px * (xs - cx) ** 2).sum())),
            float(np.sqrt((py * (ys - cy) ** 2).sum())))


def ablate_anisotropy(config):
    """
    Test the voxel-size correction on a disc that is CIRCULAR IN PHYSICAL SPACE.

    Comparing a rotated image against the unrotated original would only measure
    how much rotation moves things, which is large whether or not the correction
    is applied — it says nothing about correctness. The discriminating question
    is whether the rotation is a rotation at all.

    With anisotropic voxels a physical circle is an ellipse on the voxel grid.
    Rotate it 90 degrees:

      * corrected   -> it is a real rotation, the physical circle maps onto
                       itself, so the VOXEL aspect ratio is unchanged
      * uncorrected -> the affine acts in per-axis normalised coordinates, so the
                       ellipse's axes swap and the physical shape is destroyed

    So the metric is the aspect ratio, before and after.
    """
    size, n_slices = config['size'], config['n_slices']
    vx, vy = config['voxel_mm'][0], config['voxel_mm'][1]

    # Circular in mm, hence elliptical in voxels by exactly vy/vx.
    ax = np.arange(size) - size / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    radius_mm = 0.30 * size * min(vx, vy)
    disc = (np.exp(-((xx * vx) ** 2 + (yy * vy) ** 2) / (2 * (radius_mm / 2) ** 2))
            .astype(np.float32))
    vol = np.repeat(disc[:, :, None], n_slices, axis=2)

    x = _to_nifti_layout(vol)
    mid = n_slices // 2   # the disc is uniform in z
    ex0, ey0 = _rms_extents(vol[:, :, mid])
    results = [('original', vol[:, :, mid],
                dict(nrmse=0.0, corr=1.0, ex=ex0, ey=ey0, aspect=ex0 / ey0))]

    for label, pixdim in (('uncorrected', None), ('corrected', config['voxel_mm'])):
        aug = SpatialAugmentations(dim=3, prob=0.0, padding_mode='zeros',
                                   pixdim=pixdim, allow_rot90=False)
        spec = aug.sample_augmentations(1)[0]
        spec.update(dict(do_z_rot=True, z_angle_deg=90.0))
        rotated, _ = aug.apply(x, aug_spec_list=[spec])
        img = _from_nifti_layout(rotated).real
        ex, ey = _rms_extents(img[:, :, mid])
        results.append((label, img[:, :, mid],
                        dict(nrmse=nrmse(vol, img), corr=ssim_like(vol, img),
                             ex=ex, ey=ey, aspect=ex / ey if ey else float('nan'))))
    return results


#*************#
#   figures   #
#*************#
def _panel_grid(results, title, save_path, name, n_cols=4, reference=None):
    """Grid of image panels. Metrics are not drawn — they go to metrics.csv."""
    n = len(results)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    window_src = reference if reference is not None else results[0][1]
    vmax = np.percentile(np.abs(window_src), 99.5) or 1.0

    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis('off')
            continue
        label, img = results[i][0], results[i][1]
        ax.imshow(np.rot90(np.abs(img)), cmap='gray', vmin=0, vmax=vmax,
                  interpolation='nearest')
        ax.set_title(label, fontsize=10, pad=6)
        _bare_axes(ax)

    fig.tight_layout(h_pad=2.2, w_pad=0.5)
    return _save(fig, save_path, name)


def plot_slice_montage(vol, title, save_path, name, n_show=12):
    """Montage through z, so the volume's 3-D structure is visible."""
    n_z = vol.shape[2]
    idx = np.unique(np.linspace(0, n_z - 1, min(n_show, n_z)).astype(int))
    n_cols = 6
    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows),
                             squeeze=False)
    vmax = np.percentile(np.abs(vol), 99.5) or 1.0
    for k, ax in enumerate(axes.ravel()):
        if k >= len(idx):
            ax.axis('off')
            continue
        z = int(idx[k])
        ax.imshow(np.rot90(np.abs(vol[:, :, z])), cmap='gray', vmin=0, vmax=vmax,
                  interpolation='nearest')
        ax.set_title(f'$z$ = {z}', fontsize=10, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(h_pad=2.2, w_pad=0.5)
    return _save(fig, save_path, name)


def plot_trajectory_3d(entries, title, save_path, name, max_pts=6000):
    """
    Show 3-D trajectories as coordinates in 3-D, not as a flattened mask.

    A 3-D k-space mask viewed as one 2-D slice or a through-plane projection is
    close to useless: a stack-of-stars and a phyllotaxis both collapse to a
    filled disc, so the figure says nothing about how they differ. Plotting the
    retained sample coordinates on 3-D axes shows the thing that actually
    distinguishes them — how each one distributes shots over the sphere.

    Points are subsampled to *max_pts* per panel purely for legibility.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    n = len(entries)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(3.2 * n_cols, 3.4 * n_rows))

    for i, (label, coords) in enumerate(entries):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        c = np.asarray(coords, dtype=np.float64)
        # Deduplicate to one point per k-space bin: the figure is about which
        # bins a trajectory covers, and without this a centre-out family
        # (spokes, cones) piles ~1/r^2 of its samples at k=0 and renders as a
        # blob with an invisible rim.
        if c.shape[0]:
            step = np.abs(c).max(axis=0)
            step[step == 0] = 1.0
            # Normalised per-axis coordinates (k / k_max). In physical 1/m the
            # anisotropic FOV squashes every panel into the same flat ellipsoid
            # (kz_max is 0.70 of kx_max here), which dominates the figure and
            # hides the between-trajectory differences it exists to show.
            c = np.unique(np.round(c / step * 47.5).astype(int), axis=0) / 47.5
        if c.shape[0] > max_pts:
            # Random (seeded) subsample, NOT an evenly-strided one. np.unique
            # above returns bins in lexicographic order, so striding it walks
            # through kx in sequence and renders a spatially biased slab rather
            # than the trajectory: phyllotaxis was the only panel over the cap
            # and it alone came out looking like a flat lens, while its actual
            # bin coverage is 51% of the grid, 87% of it in the outer half.
            rng = np.random.default_rng(0)
            c = c[rng.choice(c.shape[0], size=max_pts, replace=False)]
        if c.shape[1] < 3:                      # a 2-D trajectory: pin kz = 0
            c = np.column_stack([c, np.zeros(len(c))])
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=0.4, c='#333333',
                   alpha=0.35, linewidths=0, rasterized=True)
        ax.set_title(label, fontsize=10, pad=6)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlabel('$k_x/k_{x,max}$', fontsize=8, labelpad=-12)
        ax.set_ylabel('$k_y/k_{y,max}$', fontsize=8, labelpad=-12)
        ax.set_zlabel('$k_z/k_{z,max}$', fontsize=8, labelpad=-12)
        ax.grid(False)
        ax.view_init(elev=22, azim=38)
        # Equal aspect so a sphere reads as a sphere.
        lim = float(np.abs(c).max()) or 1.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    fig.tight_layout(h_pad=2.2, w_pad=0.5)
    return _save(fig, save_path, name)


def plot_acceleration_curve(rows, title, save_path, name):
    """
    Reconstruction error against acceleration.

    One y-axis only. NRMSE and correlation have different units and ranges, and
    a twin axis invites reading a crossing point that means nothing — the
    correlation is in metrics.csv for anyone who wants it.

    *title* is accepted and ignored, as in the other plotters here: these figures
    are captioned in LaTeX, so a drawn-in title would be duplicated in print.
    """
    accel = [r['acceleration'] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(accel, [r['nrmse'] for r in rows], 'o-', color='#333333',
            linewidth=1.6, markersize=5)
    ax.set_xlabel('acceleration factor', fontsize=10)
    ax.set_ylabel('NRMSE (%)', fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(h_pad=2.2, w_pad=0.5)
    return _save(fig, save_path, name)


def plot_trajectory_ablation(results, title, save_path, name, reference=None):
    """
    Two rows: the sampling mask, and the image it reconstructs.

    *reference* is the ground-truth slice and sets the display window. Taking it
    from the first panel instead would window every reconstruction to whichever
    trajectory was listed first — if that one is dark, all the others saturate
    and the figure looks broken.
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.4), squeeze=False)
    window_src = reference if reference is not None else results[0][1]
    vmax = np.percentile(np.abs(window_src), 99.9) or 1.0

    for i, (label, img, mask, _m) in enumerate(results):
        m = np.asarray(mask)
        while m.ndim > 2:
            m = m[..., m.shape[-1] // 2]
        axes[0][i].imshow(np.rot90(m.astype(float)), cmap='gray', vmin=0, vmax=1,
                          interpolation='nearest')
        axes[0][i].set_title(label, fontsize=10, pad=6)
        axes[1][i].imshow(np.rot90(np.abs(img)), cmap='gray', vmin=0, vmax=vmax,
                          interpolation='nearest')
        for r in (0, 1):
            _bare_axes(axes[r][i])

    axes[0][0].set_ylabel('k-space mask', fontsize=10)
    axes[1][0].set_ylabel('reconstruction', fontsize=10)
    for r in (0, 1):
        axes[r][0].yaxis.set_visible(True)
        axes[r][0].set_yticks([])

    fig.tight_layout(h_pad=2.2, w_pad=0.5)
    return _save(fig, save_path, name)


def _zoom_patch(pts, n_bins, radius=0.35, half_width=3.0):
    """
    A small k-space window, in normalised units, that is guaranteed to hold samples.

    Anchored on whichever sample sits closest to *radius* rather than on a fixed
    location: a patch pinned near k=0 fills with the centre-out pile-up that every
    trajectory shares, and a patch at a fixed off-centre spot misses concentric
    rings entirely whenever it lands between two rings.
    """
    step = 2.0 / n_bins                                  # bin spacing, normalised
    r = np.hypot(pts[:, 0], pts[:, 1])
    centre = pts[np.argmin(np.abs(r - radius))]
    return centre, half_width * step


def _draw_zoom_inset(ax, pts, n_bins):
    """
    Inset showing the Cartesian bin grid with the actual samples on top of it.

    The panel behind it can only show that a trajectory is curved; at 96 bins a
    few thousand dots visually fill in and read as a mask. Drawing the grid the
    samples are being rounded onto is what makes the argument legible: the dots
    sit between the lines, and the row below rounds each one to the nearest
    crossing.
    """
    (cx, cy), half = _zoom_patch(pts, n_bins)
    axins = ax.inset_axes([0.62, 0.62, 0.38, 0.38])
    step = 2.0 / n_bins

    # Bin centres are at (2i - n)/n; draw the ones that fall inside the window.
    lo_i = int(np.floor((cx - half + 1.0) * n_bins / 2.0))
    hi_i = int(np.ceil((cx + half + 1.0) * n_bins / 2.0))
    for i in range(lo_i, hi_i + 1):
        axins.axvline((2 * i - n_bins) / n_bins, color='#bbbbbb', linewidth=0.4, zorder=0)
    lo_j = int(np.floor((cy - half + 1.0) * n_bins / 2.0))
    hi_j = int(np.ceil((cy + half + 1.0) * n_bins / 2.0))
    for j in range(lo_j, hi_j + 1):
        axins.axhline((2 * j - n_bins) / n_bins, color='#bbbbbb', linewidth=0.4, zorder=0)

    inside = ((np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 1] - cy) <= half))
    axins.scatter(pts[inside, 0], pts[inside, 1], s=6.0, c='#333333',
                  linewidths=0, zorder=2)
    axins.set_xlim(cx - half, cx + half)
    axins.set_ylim(cy - half, cy + half)
    axins.set_aspect('equal')
    axins.set_xticks([]); axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_color('#888888')
        spine.set_linewidth(0.6)
    ax.indicate_inset_zoom(axins, edgecolor='#888888', linewidth=0.6, alpha=1.0)
    return step


def plot_nufft_ablation(results, save_path, name, reference, max_pts=6000):
    """
    Where the samples fall, then the two reconstructions built from them.

    The top row is the continuous trajectory rather than a rasterised mask, and
    that is the whole argument of the figure: those points do not sit on grid
    bins. Each panel carries an inset that zooms onto a few bins with the grid
    drawn in, because at full scale a few thousand dots fill in and read as a
    mask — for Cartesian that reading is correct, and the inset is what shows
    the other four are not. The row below rounds every sample to its nearest
    bin, the row under that evaluates the adjoint where the sample actually is,
    and the difference between them is what the rounding costs.

    The window comes from the reference so both image rows share a scale —
    otherwise the noisier row sets its own and the two stop being comparable.
    """
    n = len(results)
    fig = plt.figure(figsize=(2.4 * (n + 1), 7.8))
    gs = fig.add_gridspec(3, n + 1)
    vmax = np.percentile(np.abs(reference), 99.9) or 1.0
    rng = np.random.default_rng(0)

    # The reference spans all three rows: it is the target for each of them, and
    # a mostly-empty first column reads as missing panels.
    ax_ref = fig.add_subplot(gs[:, 0])
    ax_ref.imshow(np.rot90(np.abs(reference)), cmap='gray', vmin=0, vmax=vmax,
                  interpolation='nearest')
    ax_ref.set_title('fully sampled', fontsize=10, pad=6)
    _bare_axes(ax_ref)

    n_bins = int(np.asarray(reference).shape[0])
    for i, (label, pattern, gridded, nufft, _m) in enumerate(results, start=1):
        ax = fig.add_subplot(gs[0, i])
        full = np.asarray(pattern, dtype=np.float64)
        pts = full
        if pts.shape[0] > max_pts:               # seeded, so panels stay stable
            pts = pts[rng.choice(pts.shape[0], size=max_pts, replace=False)]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.8, c='#333333',
                   alpha=0.55, linewidths=0, rasterized=True)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=10, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # The inset uses every sample, not the thinned set: the window is only a
        # few bins wide and subsampling can empty it.
        _draw_zoom_inset(ax, full, n_bins)
        if i == 1:
            ax.set_ylabel('sampling pattern', fontsize=10)
            ax.yaxis.set_visible(True)

        for r, img in ((1, gridded), (2, nufft)):
            ax = fig.add_subplot(gs[r, i])
            ax.imshow(np.rot90(np.abs(img)), cmap='gray', vmin=0, vmax=vmax,
                      interpolation='nearest')
            _bare_axes(ax)
            if i == 1:
                ax.set_ylabel('nearest-bin gridding' if r == 1 else 'NUFFT', fontsize=10)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

    fig.tight_layout(h_pad=2.2, w_pad=0.5)
    return _save(fig, save_path, name)


def plot_interpolation_curve(rows, save_path, name):
    """Interpolator error against k-space oversampling, on a log axis."""
    osf = [r[0] for r in rows]
    err = [100.0 * r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.semilogy(osf, err, 'o-', color='#333333', linewidth=1.6, markersize=5)
    ax.set_xlabel('k-space grid oversampling', fontsize=10)
    ax.set_ylabel('sampling error (%)', fontsize=10)
    ax.set_xticks(osf)
    ax.set_xticklabels([f'{o:g}x' for o in osf])
    ax.grid(True, alpha=0.25, linewidth=0.6, which='both')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(h_pad=2.2, w_pad=0.5)
    return _save(fig, save_path, name)


#************#
#   driver   #
#************#
def run_ablation(config):
    warnings.filterwarnings("ignore")
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()
    metric_rows = []

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Loading phantoms                                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    phantoms = {'shepp_logan': shepp_logan(config['size'], config['n_slices'],
                                           variant=config['sl_variant'])}
    loaders = [('ge', lambda: load_ge_phantom(config['phantom_dir'],
                                              config['size'], config['n_slices']))]
    for path in config['bigbrain']:
        # 'BigBrainMR_T1weighted_400um.nii.gz' -> 'bigbrain_400um'
        stem = os.path.basename(path).replace('.nii.gz', '')
        loaders.append(('bigbrain_' + stem.rsplit('_', 1)[-1],
                        lambda p=path: load_bigbrain(p, config['size'], config['n_slices'])))

    for key, loader in loaders:
        try:
            phantoms[key] = loader()
        except FileNotFoundError as exc:
            print(f"  ! {exc} — skipping {key}")
    for name, vol in phantoms.items():
        n_unique = len({vol[:, :, z].tobytes() for z in range(vol.shape[2])})
        print(f"  {name:12s} {vol.shape}  range {vol.min():.2f}..{vol.max():.2f}  "
              f"{n_unique}/{vol.shape[2]} distinct slices")
        print(f"               -> {plot_slice_montage(vol, f'{name}: slices through z', save_dir, f'{name}_slices')}")

    for pname, vol in phantoms.items():
        print(f"\n╔══════════════════════════════════════════════════════════════════╗")
        print(f"║  Ablating: {pname:53s} ║")
        print("╚══════════════════════════════════════════════════════════════════╝")

        print("  1. spatial augmentations, one at a time")
        spatial = ablate_spatial(vol, config, pixdim=config['voxel_mm'])
        for label, _, m in spatial:
            print(f"     {label:16s} NRMSE {m['nrmse']:6.2f}%   corr {m['corr']:.4f}")
            metric_rows.append(dict(phantom=pname, group='spatial', case=label, **m))
        p = _panel_grid(spatial, f'{pname}: spatial augmentations in isolation',
                        save_dir, f'{pname}_spatial_ablation',
                        reference=vol[:, :, _display_slice(vol)])
        print(f"     -> {p}")

        print("  2. acceleration sweep (Cartesian variable-density)")
        sampling = ablate_sampling(vol, config)
        curve = []
        for label, _, m in sampling:
            print(f"     {label:16s} NRMSE {m['nrmse']:6.2f}%   corr {m['corr']:.4f}"
                  f"   bins kept {100 * m['kept']:.0f}%")
            metric_rows.append(dict(phantom=pname, group='sampling', case=label, **m))
            curve.append(dict(acceleration=1.0 / m['kept'] if m['kept'] else np.nan,
                              nrmse=m['nrmse'], corr=m['corr']))
        p = _panel_grid(sampling, f'{pname}: zero-filled recon vs acceleration',
                        save_dir, f'{pname}_sampling_ablation')
        print(f"     -> {p}")
        p = plot_acceleration_curve(curve, f'{pname}: reconstruction error vs acceleration',
                                    save_dir, f'{pname}_acceleration_curve')
        print(f"     -> {p}")

        target = config['coverage_target']
        # A 2-D trajectory has nothing to say about kz, so comparing them on a
        # slab means every panel is really the mid slice with the rest of the
        # volume along for the ride. Give the 2-D family a single slice: the
        # classic 2-D Shepp-Logan for the generated phantom, the mid slice for
        # the measured one.
        vol_2d = vol[:, :, _display_slice(vol):_display_slice(vol) + 1]
        if pname == 'shepp_logan':
            vol_2d = shepp_logan(vol.shape[0], 1, variant=config['sl_variant'])
        for i, (family, names, fvol) in enumerate(
                (('2D', TRAJECTORIES_2D, vol_2d),
                 ('3D_stack', TRAJECTORIES_3D_STACK, vol),
                 ('3D_other', TRAJECTORIES_3D_OTHER, vol))):
            print(f"  3{'abc'[i]}. {family} trajectories at matched "
                  f"{100 * target:.0f}% bin coverage")
            traj, coord_entries = ablate_trajectory_family(fvol, config, names, family)
            for label, _, _, m in traj:
                if m['shots'] == 0:
                    print(f"     {label:22s} {'reference':>34s}")
                    continue
                missed = '' if abs(m['kept'] - target) < 0.05 else '  (target unreachable)'
                print(f"     {label:22s} shots {m['shots']:5d}  accel {m['accel']:5.2f}  "
                      f"bins {100 * m['kept']:5.1f}%  NRMSE {m['nrmse']:6.2f}%  "
                      f"corr {m['corr']:.4f}{missed}")
                metric_rows.append(dict(phantom=pname, group=f'trajectory_{family}',
                                        case=label, **m))
            if traj:
                p = plot_trajectory_ablation(
                    traj,
                    f'{pname}: {family} trajectories at ~{100 * target:.0f}% of k-space bins',
                    save_dir, f'{pname}_trajectories_{family.lower()}',
                    reference=fvol[:, :, _display_slice(fvol)])
                print(f"     -> {p}")
            if coord_entries and family != '2D':
                p = plot_trajectory_3d(
                    coord_entries, '',
                    save_dir, f'{pname}_trajectories_{family.lower()}_coords')
                print(f"     -> {p}")

        if not _nufft_available():
            print("  4-6. NUFFT figures — skipped (torchkbnufft not installed)")
            continue

        print(f"  4. NUFFT vs nearest-bin gridding at matched "
              f"{100 * target:.0f}% bin coverage")
        nuf = ablate_nufft(vol_2d, config)
        for label, _, _, _, m in nuf:
            print(f"     {label:22s} shots {m['shots']:5d}  bins {100 * m['kept']:5.1f}%  "
                  f"NRMSE  gridded {m['nrmse']:6.2f}%  NUFFT {m['nrmse_nufft']:6.2f}%")
            metric_rows.append(dict(phantom=pname, group='nufft', case=label, **m))
        if nuf:
            p = plot_nufft_ablation(nuf, save_dir, f'{pname}_nufft_ablation',
                                    reference=vol_2d[:, :, _display_slice(vol_2d)])
            print(f"     -> {p}")

        print("  5. cubic interpolator accuracy vs k-space oversampling")
        interp = ablate_interpolator(vol_2d, config)
        for osf, rel in interp:
            print(f"     {osf}x oversampled grid     sampling error {100 * rel:7.3f}%")
            metric_rows.append(dict(phantom=pname, group='interpolator',
                                    case=f'{osf}x', nrmse=100 * rel))
        p = plot_interpolation_curve(interp, save_dir, f'{pname}_interpolation_error')
        print(f"     -> {p}")

        print("  6. full-page NUFFT panel: trajectory vs acceleration "
              "(slow — 3-D NUFFTs dominate)")
        pcolumns, pcells = ablate_nufft_panel(vol, config)
        for _, _, m in pcells.values():
            metric_rows.append(dict(phantom=pname, group='nufft_panel',
                                    case=f"{m['trajectory']}@{m['acceleration']:g}x",
                                    nrmse=m['nrmse'], kept=m['kept'], shots=m['shots'],
                                    accel=m['acceleration']))
        panel_ref = vol[:, :, _display_slice(vol)]
        p = plot_nufft_panel(pcolumns, pcells, config, panel_ref, save_dir,
                             f'{pname}_nufft_panel')
        print(f"     -> {p}")
        p = plot_nufft_panel(pcolumns, pcells, config, panel_ref, save_dir,
                             f'{pname}_nufft_panel_inset', insets=True)
        print(f"     -> {p}")
        p = plot_nufft_panel_paired(pcolumns, pcells, config, panel_ref, save_dir,
                                    f'{pname}_nufft_panel_paired')
        print(f"     -> {p}")
        p = plot_nufft_panel_paired(pcolumns, pcells, config, panel_ref, save_dir,
                                    f'{pname}_nufft_panel_paired_inset', insets=True)
        print(f"     -> {p}")

    # Anisotropy is a property of the affine, not of a particular phantom, so it
    # runs once on a purpose-built disc that is circular in physical space.
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Anisotropic voxel correction (90 deg rotation of a physical circle) ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    aniso = ablate_anisotropy(config)
    base_aspect = aniso[0][-1]['aspect']
    print(f"  voxel size {config['voxel_mm'][:2]} mm -> a physical circle is an "
          f"ellipse of aspect {base_aspect:.3f} on the voxel grid")
    for label, _, m in aniso:
        verdict = ''
        if label != 'original':
            verdict = ('  <- shape preserved' if abs(m['aspect'] - base_aspect) < 0.1 * base_aspect
                       else '  <- shape destroyed (axes swapped)')
        print(f"     {label:16s} x {m['ex']:5.2f}  y {m['ey']:5.2f}  "
              f"aspect {m['aspect']:.3f}{verdict}")
        metric_rows.append(dict(phantom='disc', group='anisotropy', case=label, **m))
    p = _panel_grid(aniso, f'90 deg rotation of a physical circle, voxel '
                           f'{config["voxel_mm"][0]}x{config["voxel_mm"][1]} mm',
                    save_dir, 'anisotropy_ablation', n_cols=3,
)
    print(f"     -> {p}")

    csv_path = os.path.join(save_dir, 'metrics.csv')
    fields = ['phantom', 'group', 'case', 'nrmse', 'corr', 'kept', 'shots',
              'accel', 'family', 'ex', 'ey', 'aspect', 'nrmse_nufft', 'corr_nufft']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"\n  metrics -> {csv_path}  ({len(metric_rows)} rows)")
    print(f"  ✓ finished in {time.time() - start:.1f}s")
    return metric_rows


#*********#
#   cli   #
#*********#
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablate Augmentrum's spatial augmentation and k-space sampling on phantoms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/phantom_ablation.py
  python scripts/phantom_ablation.py --size 256 --n-slices 16
  python scripts/phantom_ablation.py --accelerations 2 4 8 16
        """)
    d = DEFAULT_CONFIG
    parser.add_argument('--phantom-dir', default=d['phantom_dir'])
    parser.add_argument('--bigbrain', nargs='*', default=list(d['bigbrain']),
                        help='BigBrainMR volumes, one phantom each; block-averaged '
                             'down to the matrix')
    parser.add_argument('--save', default=d['save_dir'])
    parser.add_argument('--size', type=int, default=d['size'],
                        help='in-plane matrix size')
    parser.add_argument('--n-slices', type=int, default=d['n_slices'])
    parser.add_argument('--seed', type=int, default=d['seed'])
    parser.add_argument('--accelerations', type=float, nargs='+',
                        default=list(d['accelerations']))
    parser.add_argument('--voxel-mm', type=float, nargs=3, default=list(d['voxel_mm']),
                        help='voxel size for the anisotropy ablation')
    parser.add_argument('--sl-variant', default=d['sl_variant'],
                        choices=sorted(SHEPP_LOGAN_INTENSITIES),
                        help="Shepp-Logan contrast: 'modified' (Toft, features visible) "
                             "or 'original' (1%% steps, features effectively invisible)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config.update({
        'phantom_dir':   args.phantom_dir,
        'bigbrain':      tuple(args.bigbrain),
        'save_dir':      args.save,
        'size':          args.size,
        'n_slices':      args.n_slices,
        'seed':          args.seed,
        'accelerations': tuple(args.accelerations),
        'voxel_mm':      tuple(args.voxel_mm),
        'sl_variant':    args.sl_variant,
    })
    run_ablation(config)
