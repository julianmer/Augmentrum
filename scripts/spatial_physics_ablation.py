####################################################################################################
#                                 spatial_physics_ablation.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (jlamaste@gmail.com)                                                     #
#                                                                                                  #
# Created: 2026-10-22                                                                              #
#                                                                                                  #
# Purpose: One anatomically recognizable slice, pushed through the same spiral acquisition with    #
#          B0, B1+, concomitant-field, GIRF/eddy-current and gradient-nonlinearity effects turned   #
#          on one at a time, at two field strengths - so the figure shows not just "artifacts",     #
#          but that different scanner fields leave distinct, physically-motivated spatial           #
#          signatures. A non-Cartesian (spiral) readout is used throughout so the B0 panel gets the #
#          classic off-resonance blur that frequency-segmented modeling (FieldInhomogeneity's own   #
#          b0_n_segments) exists to capture.                                                        #
#                                                                                                  #
####################################################################################################

"""
Spatial-physics ablation figure: ideal / +B0 / +B1 / +concomitant / +GIRF / +GNL / combined.

Run this directly:

    python scripts/spatial_physics_ablation.py

Everything that shapes the figure is a CLI flag (see --help) or a constant in
the "config" section below - edit and rerun rather than hand-editing results.
Needs the optional GIRF-Sim dependency (girf_module, girf_synthetic,
girf_mrsi_extensions, mrs_seq) for the concomitant/GIRF/GNL panels; B0, B1 and
the "ideal" baseline run with only Augmentrum + gradunwarp.

Two independent non-Cartesian trajectories are used, both deliberately
off-resonance-sensitive but NOT numerically identical:
  - "ideal"/"+B0"/"+B1" all go through FieldInhomogeneity itself
    (trajectory='spiral_2d'), with whichever of b0_map/b1_map is not the
    condition under test forced to an explicit all-zero array rather than
    "None" - see _field_inhomogeneity()'s docstring for why that matters.
  - "+concomitant"/"+GIRF"/"+GNL" go through a synthetic Pulseq ".seq" file
    built with GIRF-Sim's mrs_seq (acquisition_mode='mrsi2d'), which also
    resolves to a spiral readout.
"combined" chains both: FieldInhomogeneity (B1+B0, analytic trajectory) then
GIRFArtifacts (GIRF+concomitant+GNL, .seq trajectory) on its output. This is a
sequential composition for a readable illustration, not a single unified
forward model - see the module docstrings if you need the latter.

A caveat worth knowing before reading ANY panel: this repo's non-Cartesian
gridding NUFFT has a known, pre-existing, large (roughly 50-65%) baseline
reconstruction error at modest matrix sizes/shot counts - confirmed here
across spiral/radial/rosette alike, insensitive to shot count or NUFFT
oversampling, and already the documented reason
tests/augmentation/test_field_inhomogeneity.py uses 'cartesian_2d' for its
exact-tolerance checks. "ideal" is therefore not a lossless reference - every
condition's panel is normalized to its OWN display range (see
plot_ablation_grid), precisely so this baseline loss (and "combined"'s two
compounded lossy NUFFT passes) does not masquerade as physics and does not
hide real structure behind a too-dark panel.

A second caveat, specific to the "+concomitant" panel: the standard
symmetric-coil concomitant-field term only has spatial structure (the "x^2+y^2",
"xz", "yz" basis terms) when a partition/through-slice gradient is active
*during* the readout. This script's acquisition is a single on-axis 2-D slice
with no concurrent Gz, so only the "z^2" term could ever be nonzero, and with
the slice at isocenter (z=0, which is what GIRFArtifacts always assumes for a
single 2-D slice - see its own _positions()) even that term is analytically
zero. The panel is kept in the figure anyway, labeled accordingly, because
showing a genuine null result honestly is better than faking one: a visibly
structured concomitant artifact needs either an off-isocenter/oblique slice or
a true 3-D (e.g. cones/floret) acquisition, which is future work, not
something to fake here.
"""

#*************#
#   imports   #
#*************#
import argparse
import os
import sys
import time
import warnings

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import every Augmentrum class BEFORE anything from GIRF-Sim/mrs_seq is ever
# touched. Building a ".seq" file via mrs_seq (see build_seq_file() below)
# otherwise leaves a stale "Domain" bound in augmentrum.augmentation's import
# chain the first time it is imported afterwards (a cross-package import-order
# quirk in mrs_seq's own trajectory adapter) - importing Augmentrum first
# avoids it entirely, at zero cost.
from augmentrum.augmentation.field_inhomogeneity import FieldInhomogeneity
from augmentrum.augmentation.girf_artifacts import GIRFArtifacts
from augmentrum.physics.concomitant_field import GAMMA_HZ_PER_T

GIRF_SIM_AVAILABLE = True
try:
    import girf_module as _girf_module   # noqa: F401
except ImportError:
    GIRF_SIM_AVAILABLE = False


#************#
#   config   #
#************#
# Edit these directly for a quick rerun, or override from the CLI (--help).
DEFAULT_CONFIG = {
    'out_dir':            'results/spatial_physics_ablation',
    'size':                48,     # in-plane matrix, both trajectories
    'fov_mm':              160.0,
    'n_spectral_points':   48,
    'spectral_bandwidth_hz': 2000.0,
    'decay_hz':            10.0,   # single-compartment T2*-like FID decay
    'field_strengths_T':   (0.55, 3.0),
    'spiral_n_shots':       1,     # single long arm - maximal off-resonance blur
    'b0_n_segments':        16,
    # Peak-to-peak 2nd-order shim residual, in ppm - deliberately strong (a
    # well-shimmed brain is closer to 0.1-0.2 ppm) so the spiral off-resonance
    # blur this is meant to showcase is actually visible at 3 T, while still
    # being barely-there at 0.55 T (B0 inhomogeneity in Hz scales with field).
    'b0_shim_ppm':          0.80,
    'b1_depth_per_tesla':   0.08,  # illustrative B1+ dropout-with-field coefficient
    # Illustrative "gradient coil quality" stand-in for compact/low-field vs
    # whole-body/high-field platforms - NOT a literature-calibrated value,
    # see run_gnl()'s docstring.
    'gnl_stochasticity':   {0.55: 0.15, 3.0: 0.03},
    'seq_n_shots':          8,
    'seq_samples_per_shot': 256,
    'seq_spectral_points':  8,
    'seed':                 0,
    # Phantom source. 'shepp_logan' needs no external data and is the default;
    # 'nifti' is wired up for later use against a real anatomical scan - see
    # load_phantom_slice().
    'phantom':              'shepp_logan',
    'nifti_path':            None,
    'nifti_slice':            None,
}


#*********************************#
#   phantom (anatomical slice)    #
#*********************************#
# Shepp-Logan geometry/intensities, copied from scripts/phantom_ablation.py's
# own tables (not imported from it) so this file stays a single, independently
# runnable script - see that file if the two drift and you want to reconcile.
SHEPP_LOGAN_GEOMETRY = (
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
SHEPP_LOGAN_INTENSITIES = (1.00, -0.80, -0.20, -0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10)


def _shepp_logan_slice(size):
    """A single modified-Toft-contrast Shepp-Logan slice, (size, size) float32."""
    ax = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    img = np.zeros((size, size), dtype=np.float32)
    for (x0, y0, a, b, angle_deg), value in zip(SHEPP_LOGAN_GEOMETRY, SHEPP_LOGAN_INTENSITIES):
        t = np.deg2rad(angle_deg)
        xr = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
        yr = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += value
    return np.clip(img, 0.0, None)


def _nifti_slice(path, slice_index, size):
    """
    A single slice from a real NIfTI anatomical volume, resized to "size".

    This is the hook for later use against an actual subject scan: point
    "--phantom nifti --nifti-path ... [--nifti-slice ...]" at any T1w/T2w/
    FLAIR volume "nibabel" can read. No windowing beyond a unit-peak
    normalization is applied - supply an already-preprocessed (skull-stripped/
    bias-corrected) volume if you want that.
    """
    import nibabel as nib
    from scipy.ndimage import zoom

    img = nib.load(path)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    idx = slice_index if slice_index is not None else vol.shape[2] // 2
    sl = vol[:, :, idx]

    if sl.shape != (size, size):
        sl = zoom(sl, (size / sl.shape[0], size / sl.shape[1]), order=1)

    peak = sl.max()
    return sl / peak if peak > 0 else sl


def load_phantom_slice(config):
    """The (size, size) anatomical slice this figure runs on, per config['phantom']."""
    if config['phantom'] == 'shepp_logan':
        return _shepp_logan_slice(config['size'])
    if config['phantom'] == 'nifti':
        if not config['nifti_path']:
            raise ValueError("config['phantom'] == 'nifti' needs config['nifti_path'] set.")
        return _nifti_slice(config['nifti_path'], config['nifti_slice'], config['size'])
    raise ValueError(f"Unknown phantom source {config['phantom']!r}.")


#*******************************#
#   synthetic FID + field maps  #
#*******************************#
def make_fid_volume(img, n_points, bandwidth_hz, decay_hz):
    """"(1, X, Y, 1, T)" complex64: the slice's own intensity times one decaying FID per voxel."""
    dwell = 1.0 / bandwidth_hz
    t = np.arange(n_points) * dwell
    fid = np.exp(-decay_hz * t).astype(np.complex64)
    vol = img[None, :, :, None, None].astype(np.complex64) * fid[None, None, None, None, :]
    return vol, dwell


def make_b0_map_hz(size, b0_tesla, shim_ppm):
    """
    A smooth 2nd-order shim-residual B0 map, in Hz.

    Illustrative, not measured: a quadrupolar (z^2-like in-plane) pattern with
    peak-to-peak amplitude "shim_ppm" (same residual at every field strength),
    converted to Hz via "delta_f = ppm * 1e-6 * gamma * B0" - the only
    field-strength dependence that matters physically, since a given shim
    *fraction* of B0 is roughly what shimming achieves regardless of field.
    """
    ax = (np.arange(size) - size / 2.0) / (size / 2.0)
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    shim_pattern = 0.6 * (xx ** 2 - yy ** 2) + 0.4 * xx * yy   # peak-to-peak ~1 in these units
    shim_pattern *= shim_ppm / (shim_pattern.max() - shim_pattern.min())
    return (shim_pattern * 1e-6 * GAMMA_HZ_PER_T * b0_tesla).astype(np.float64)[:, :, None]


def make_b1_map(size, b0_tesla, depth_per_tesla):
    """
    A radial B1+ magnitude dropout, worsening with field strength.

    Illustrative, not measured: dielectric-resonance-driven B1+ inhomogeneity
    is well known to worsen at higher field, but this specific quadratic
    radial falloff and the "depth_per_tesla" coefficient are a stand-in for
    that trend, not a fitted model - see the class docstring note in
    FieldInhomogeneity about not synthesizing maps itself.
    """
    ax = (np.arange(size) - size / 2.0) / (size / 2.0)
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    r2 = xx ** 2 + yy ** 2
    depth = min(0.6, depth_per_tesla * b0_tesla)
    magnitude = np.clip(1.0 - depth * r2, 0.2, None)
    return magnitude.astype(np.complex128)[:, :, None]


#***************************#
#   the .seq fixture file   #
#***************************#
def build_seq_file(config, out_path):
    """
    A spiral MRSI2D ".seq" file for the concomitant/GIRF/GNL panels.

    One field-strength-independent file is enough: GIRFArtifacts takes
    "b0_tesla" as its own argument (overriding whatever the ".seq" system
    preset declares), so the trajectory/timing never needs rebuilding per row.
    Needs GIRF-Sim's "mrs_seq" (not part of its installed package surface -
    found via girf_module.__file__, same fallback tests/augmentation/
    test_girf_artifacts.py's fixture uses).
    """
    import sys as _sys
    from pathlib import Path as _Path
    try:
        import mrs_seq   # noqa: F401
    except ImportError:
        repo_root = str(_Path(_girf_module.__file__).resolve().parent)
        if repo_root not in _sys.path:
            _sys.path.insert(0, repo_root)

    from mrs_seq.localization.steam import build_steam
    from mrs_seq.system import build_system

    system = build_system('generic_3T', grad_raster_time=4e-5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seq, _ = build_steam(
            system, acquisition_mode='mrsi2d',
            voi_mm=(config['fov_mm'] * 0.75, config['fov_mm'] * 0.75, 20.0),
            TE=0.020, TM=0.020, TR=0.3,
            fov_mm=(config['fov_mm'], config['fov_mm']), matrix=(config['size'], config['size']),
            n_shots=config['seq_n_shots'], samples_per_shot=config['seq_samples_per_shot'],
            spectral_bandwidth_hz=config['spectral_bandwidth_hz'],
            spectral_points=config['seq_spectral_points'],
            target_fid_duration=config['n_spectral_points'] / config['spectral_bandwidth_hz'],
            seq_name='spatial_physics_ablation',
        )
        seq.write(str(out_path))
    return str(out_path)


#************************#
#   per-condition runs   #
#************************#
# All seven conditions share this signature: (vol, dwell, b0_tesla, config) -> (X, Y) magnitude
# image, taken at the LAST spectral sample (where B0/off-resonance dephasing has had the whole
# readout to accumulate - at t=0 every condition looks identical, which would be a misleading
# first panel to display).
def _late_frame(out):
    return np.abs(out[0, :, :, 0, -1])


def _field_inhomogeneity(b0_map_hz, b1_map, config, dwell):
    """
    A `FieldInhomogeneity` instance sharing one NUFFT/segmentation config.

    Used for every condition below, including "ideal" (a literal all-zero
    "b0_map") - `FieldInhomogeneity(b0_map=None, b1_map=None)` is a true
    identity (no NUFFT at all, see its own docstring), which would make
    "ideal" exact-by-construction and every other condition's apparent
    "damage" partly just this module's own gridding/NUFFT reconstruction
    error relative to a *different* pipeline. Passing an explicit all-zero
    array instead forces the same NUFFT roundtrip with trivial physics, so
    "ideal" and "+B0"/"+B1"/"combined" differ only by the physics term that is
    actually on.
    """
    return FieldInhomogeneity(b0_map=b0_map_hz, b1_map=b1_map,
                              b0_n_segments=config['b0_n_segments'],
                              b0_segment_strategy='adaptive', trajectory='spiral_2d',
                              traj_params={'n_shots': config['spiral_n_shots']},
                              acceleration_factor=1.0, traj_seed=config['seed'],
                              dwell_time_s=dwell)


def run_ideal(vol, dwell, b0_tesla, config, size):
    """Baseline: the same spiral trajectory and gridding NUFFT, no physics effects at all."""
    zero_map = np.zeros((size, size, 1), dtype=np.float64)
    mod = _field_inhomogeneity(zero_map, None, config, dwell)
    out, _ = mod.process_tensor(vol)
    return _late_frame(out)


def run_b0(vol, dwell, b0_tesla, config, b0_map_hz):
    """
    Segmented off-resonance, on the same spiral trajectory.

    The spiral is exactly the geometry frequency-segmented B0 correction was
    invented for (Noll/Meyer-style conjugate-phase reconstruction): a long,
    slowly-sampled readout accumulates real off-resonance phase along its own
    arm, which a Cartesian "shift-only" baseline never shows - see the module
    docstring's "b0_n_segments" discussion for the segmentation itself.
    """
    mod = _field_inhomogeneity(b0_map_hz, None, config, dwell)
    out, _ = mod.process_tensor(vol)
    return _late_frame(out)


def run_b1(vol, dwell, b0_tesla, config, b1_map, size):
    """B1+ magnitude modulation, through the same spiral/NUFFT pipeline (b0_map forced to zero)."""
    zero_map = np.zeros((size, size, 1), dtype=np.float64)
    mod = _field_inhomogeneity(zero_map, b1_map, config, dwell)
    out, _ = mod.process_tensor(vol)
    return _late_frame(out)


def run_concomitant(vol, dwell, b0_tesla, config, seq_file):
    """
    The Maxwell/concomitant phase, isolated (see this file's module docstring
    caveat: analytically zero for this on-axis 2-D single-slice geometry -
    kept in the figure as an honest null result, not hidden).
    """
    if not GIRF_SIM_AVAILABLE:
        return None
    mod = GIRFArtifacts(seq_file=seq_file, b0_tesla=b0_tesla,
                        include_girf_phase=False, include_trajectory_error=False,
                        include_concomitant=True, include_gradient_nonlinearity=False)
    out, _ = mod.process_tensor(vol)
    return _late_frame(out)


def run_girf(vol, dwell, b0_tesla, config, seq_file):
    """GIRF-induced trajectory error + phase (eddy-current-like gradient infidelity)."""
    if not GIRF_SIM_AVAILABLE:
        return None
    mod = GIRFArtifacts(seq_file=seq_file, b0_tesla=b0_tesla, girf_seed=config['seed'],
                        include_girf_phase=True, include_trajectory_error=True,
                        include_concomitant=False, include_gradient_nonlinearity=False)
    out, _ = mod.process_tensor(vol)
    return _late_frame(out)


def run_gnl(vol, dwell, b0_tesla, config, seq_file):
    """
    Gradient-coil nonlinearity, as a pure image-domain geometric warp.

    "gnl_stochasticity" stands in for "a more compact/lower-fidelity gradient
    coil distorts more" (true in general for portable/low-field platforms)
    without claiming to be any particular vendor's calibrated coefficients -
    see GradientNonlinearityDisplacement's own docstring for the real thing.
    """
    if not GIRF_SIM_AVAILABLE:
        return None
    stochasticity = config['gnl_stochasticity'].get(b0_tesla, 0.05)
    mod = GIRFArtifacts(seq_file=seq_file, b0_tesla=b0_tesla,
                        include_girf_phase=False, include_trajectory_error=False,
                        include_concomitant=False, include_gradient_nonlinearity=True,
                        gradient_nonlinearity_stochasticity=stochasticity,
                        gradient_nonlinearity_seed=config['seed'])
    out, _ = mod.process_tensor(vol)
    return _late_frame(out)


def run_combined(vol, dwell, b0_tesla, config, b0_map_hz, b1_map, seq_file):
    """
    Everything at once: FieldInhomogeneity (B1+B0, analytic spiral) then
    GIRFArtifacts (GIRF+concomitant+GNL, .seq spiral) on its output.

    A sequential composition of two independent acquisition models, not a
    single unified forward model (see this file's module docstring) - good
    enough to show cumulative, field-strength-distinct visual degradation,
    which is the point of this panel.
    """
    field_mod = _field_inhomogeneity(b0_map_hz, b1_map, config, dwell)
    stage1, _ = field_mod.process_tensor(vol)

    if not GIRF_SIM_AVAILABLE:
        return _late_frame(stage1)

    stochasticity = config['gnl_stochasticity'].get(b0_tesla, 0.05)
    girf_mod = GIRFArtifacts(seq_file=seq_file, b0_tesla=b0_tesla, girf_seed=config['seed'],
                             include_girf_phase=True, include_trajectory_error=True,
                             include_concomitant=True, include_gradient_nonlinearity=True,
                             gradient_nonlinearity_stochasticity=stochasticity,
                             gradient_nonlinearity_seed=config['seed'])
    stage2, _ = girf_mod.process_tensor(stage1)
    return _late_frame(stage2)


#******************#
#   figure style   #
#******************#
# Mirrors scripts/phantom_ablation.py's own panel-grid conventions (bare axes,
# percentile-clipped grayscale, no colored chrome, PNG+PDF) rather than
# importing it, so this script stays independently runnable/editable.
def _bare_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig, out_dir, name, dpi=200, pdf_dpi=600):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=dpi, bbox_inches='tight',
               facecolor='white')
    fig.savefig(os.path.join(out_dir, f'{name}.pdf'), dpi=pdf_dpi, bbox_inches='tight',
               facecolor='white')


#: Column kinds drive display scaling: the two maps are each normalized against
#: their own range (wildly different units/scales from both each other and
#: from a reconstructed image), while every acquisition condition shares one
#: vmax taken from that row's own "ideal" panel - so the conditions stay
#: directly comparable to each other, and the maps are still legible rather
#: than swamped by them (or swamping them).
_COLUMN_KINDS = ('b0map', 'b1map') + ('condition',) * 7


def plot_ablation_grid(rows, out_dir, name='ablation'):
    """
    rows: list of (row_label, [(title, image_or_None), ...]) - one row per
    field strength, columns in "_COLUMN_KINDS" order. A "None" image (a
    GIRF-Sim-dependent panel when GIRF-Sim is unavailable) is rendered as a
    blank panel with a note, not silently skipped, so the grid never shifts.

    Every condition panel is normalized to its OWN 99.5th-percentile value,
    not one vmax shared across the row. This script's point is the spatial
    STRUCTURE each effect leaves behind, not relative brightness: a condition
    that also happens to lose signal (B0 dephasing, or "combined" running two
    separate lossy non-Cartesian NUFFT passes back to back - see this file's
    module docstring) would otherwise render as a near-black, uninformative
    panel under a shared scale, hiding the very structure the figure exists to
    show.
    """
    n_rows = len(rows)
    n_cols = len(rows[0][1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows), squeeze=False)

    for r, (row_label, panels) in enumerate(rows):
        for c, ((title, img), kind) in enumerate(zip(panels, _COLUMN_KINDS)):
            ax = axes[r, c]
            if img is None:
                ax.text(0.5, 0.5, 'needs\nGIRF-Sim', ha='center', va='center',
                       fontsize=8, color='0.5', transform=ax.transAxes)
            elif kind == 'b0map':
                vmax = np.abs(img).max() or 1.0
                ax.imshow(np.rot90(img), cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                         interpolation='nearest')
            elif kind == 'b1map':
                ax.imshow(np.rot90(img), cmap='gray', vmin=0, vmax=1.0, interpolation='nearest')
            else:
                vmax = np.percentile(img, 99.5) or 1.0
                ax.imshow(np.rot90(img), cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
            if r == 0:
                ax.set_title(title, fontsize=10, pad=6)
            _bare_axes(ax)
            if c == 0:
                ax.text(-0.08, 0.5, row_label, fontsize=10, rotation=90,
                       ha='center', va='center', transform=ax.transAxes)

    fig.tight_layout(h_pad=1.8, w_pad=0.4)
    _save(fig, out_dir, name)
    plt.close(fig)


#************#
#   driver   #
#************#
def run(config):
    if not GIRF_SIM_AVAILABLE:
        print("GIRF-Sim not importable: +concomitant / +GIRF / +GNL / combined's "
             "GIRF-side terms will render as blank 'needs GIRF-Sim' panels. "
             "B0, B1 and the ideal baseline are unaffected.")

    img = load_phantom_slice(config)
    seq_file = None
    if GIRF_SIM_AVAILABLE:
        seq_path = os.path.join(config['out_dir'], 'spatial_physics_ablation.seq')
        os.makedirs(config['out_dir'], exist_ok=True)
        t0 = time.time()
        seq_file = build_seq_file(config, seq_path)
        print(f"built .seq fixture in {time.time() - t0:.2f}s -> {seq_file}")

    rows = []
    for b0_tesla in config['field_strengths_T']:
        print(f"--- {b0_tesla} T ---")
        vol, dwell = make_fid_volume(img, config['n_spectral_points'],
                                     config['spectral_bandwidth_hz'], config['decay_hz'])
        b0_map_hz = make_b0_map_hz(config['size'], b0_tesla, config['b0_shim_ppm'])
        b1_map = make_b1_map(config['size'], b0_tesla, config['b1_depth_per_tesla'])

        panels = [
            ('B0 map', b0_map_hz[:, :, 0]),
            ('B1+ map', np.abs(b1_map[:, :, 0])),
        ]
        conditions = [
            ('ideal', lambda: run_ideal(vol, dwell, b0_tesla, config, config['size'])),
            ('+B0', lambda: run_b0(vol, dwell, b0_tesla, config, b0_map_hz)),
            ('+B1', lambda: run_b1(vol, dwell, b0_tesla, config, b1_map, config['size'])),
            ('+concomitant', lambda: run_concomitant(vol, dwell, b0_tesla, config, seq_file)),
            ('+GIRF', lambda: run_girf(vol, dwell, b0_tesla, config, seq_file)),
            ('+GNL', lambda: run_gnl(vol, dwell, b0_tesla, config, seq_file)),
            ('combined', lambda: run_combined(vol, dwell, b0_tesla, config,
                                              b0_map_hz, b1_map, seq_file)),
        ]
        for title, fn in conditions:
            t0 = time.time()
            panels.append((title, fn()))
            print(f"  {title:<14s} {time.time() - t0:5.2f}s")

        rows.append((f"{b0_tesla:g} T", panels))

    plot_ablation_grid(rows, config['out_dir'], name='spatial_physics_ablation')
    print(f"\nwrote {config['out_dir']}/spatial_physics_ablation.png (+ .pdf)")


#**********#
#   CLI    #
#**********#
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_CONFIG['out_dir'])
    p.add_argument('--size', type=int, default=DEFAULT_CONFIG['size'])
    p.add_argument('--fov-mm', type=float, default=DEFAULT_CONFIG['fov_mm'])
    p.add_argument('--n-spectral-points', type=int, default=DEFAULT_CONFIG['n_spectral_points'])
    p.add_argument('--field-strengths', type=float, nargs='+',
                   default=list(DEFAULT_CONFIG['field_strengths_T']))
    p.add_argument('--b0-n-segments', type=int, default=DEFAULT_CONFIG['b0_n_segments'])
    p.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'])
    p.add_argument('--phantom', choices=['shepp_logan', 'nifti'], default=DEFAULT_CONFIG['phantom'])
    p.add_argument('--nifti-path', default=DEFAULT_CONFIG['nifti_path'])
    p.add_argument('--nifti-slice', type=int, default=DEFAULT_CONFIG['nifti_slice'])
    return p.parse_args()


def main():
    args = parse_args()
    config = dict(DEFAULT_CONFIG)
    config.update(
        out_dir=args.out_dir, size=args.size, fov_mm=args.fov_mm,
        n_spectral_points=args.n_spectral_points,
        field_strengths_T=tuple(args.field_strengths),
        b0_n_segments=args.b0_n_segments, seed=args.seed,
        phantom=args.phantom, nifti_path=args.nifti_path, nifti_slice=args.nifti_slice,
    )
    run(config)


if __name__ == '__main__':
    main()
