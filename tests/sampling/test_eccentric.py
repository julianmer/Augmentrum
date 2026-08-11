"""
Tests for the ECCENTRIC trajectory and center-crossing undersampling.

Tests cover (on an asymmetric matrix, measuring quantities rather than eyeballing):
- Coordinates staying inside the anisotropic kmax box
- The constructed center-crossing subset, and the measured predicate finding it
- All crossing shots retained under acceleration; achieved AF near requested
- Per-kz redraw in the stack, seeded reproducibility of both classes
- k-space coverage of the full trajectory
"""

import numpy as np
import pytest

from augmentrum.sampling.kspace_sampling import (
    Eccentric2D, GridMask, ShotUndersampler, TrajectoryRegistry)


GEOM_2D = {"matrix": (48, 64, 1), "fov_mm": (192.0, 256.0, 1.0),
           "ndim": 2, "inferred": []}
GEOM_3D = {"matrix": (48, 64, 6), "fov_mm": (192.0, 256.0, 60.0),
           "ndim": 3, "inferred": []}


def build_2d(**params):
    params.setdefault("n_shots", 128)
    params.setdefault("seed", 3)
    return Eccentric2D(**params).generate(GEOM_2D)


#**************************************************************************************************#
#                                          geometry                                                #
#**************************************************************************************************#
def test_coordinates_stay_inside_anisotropic_kmax():
    shots, meta = build_2d()
    pts = np.concatenate([np.asarray(s) for s in shots])
    kmax_x = (48 / 2.0) / 0.192
    kmax_y = (64 / 2.0) / 0.256
    assert np.abs(pts[:, 0]).max() <= kmax_x * (1 + 1e-6)
    assert np.abs(pts[:, 1]).max() <= kmax_y * (1 + 1e-6)
    assert meta["n_shots"] == 128 and len(shots) == 128


def test_center_crossing_construction_and_predicate():
    shots, meta = build_2d(center_frac=0.15)
    constructed = meta["center_crossing"]
    assert constructed.sum() == round(0.15 * 128)

    detected = ShotUndersampler._us_center_crossing(len(shots), 1e9, {}, shots)
    assert detected[constructed].all(), \
        "every constructed crossing circle must be detected by the predicate"
    # Free circles may legitimately pass within a sample step of the center,
    # so extras are allowed — but not many.
    assert detected.sum() <= constructed.sum() + 0.1 * len(shots)


def test_full_trajectory_covers_the_kspace_disk():
    shots, meta = build_2d(n_shots=192)
    keep_all = np.ones(len(shots), dtype=bool)
    grid = GridMask.rasterize_shots_to_grid(
        shots, keep_all, (0.192, 0.256), (48, 64))

    x = (np.arange(48) - 24) / 24.0
    y = (np.arange(64) - 32) / 32.0
    inner = (x[:, None] ** 2 + y[None, :] ** 2) <= 0.9 ** 2
    coverage = grid[inner].mean()
    assert coverage > 0.9, f"only {coverage:.2f} of the central disk is covered"


#**************************************************************************************************#
#                                        undersampling                                             #
#**************************************************************************************************#
def test_center_crossing_scheme_keeps_crossing_shots():
    shots, meta = build_2d()
    mask, _ = ShotUndersampler.undersample_shots(
        shots, "center_crossing", 3.0, {"seed": 0},
        trajectory_name="eccentric_2d")
    mask = np.asarray(mask).astype(bool)
    assert mask[meta["center_crossing"]].all(), \
        "acceleration must never drop a center-crossing circle"
    achieved = len(shots) / mask.sum()
    assert abs(achieved - 3.0) / 3.0 < 0.15


def test_center_crossing_seed_reproduces():
    shots, _ = build_2d()
    m1 = ShotUndersampler._us_center_crossing(len(shots), 3.0, {"seed": 5}, shots)
    m2 = ShotUndersampler._us_center_crossing(len(shots), 3.0, {"seed": 5}, shots)
    m3 = ShotUndersampler._us_center_crossing(len(shots), 3.0, {"seed": 6}, shots)
    assert (m1 == m2).all()
    assert not (m1 == m3).all()


def test_incompatible_method_rejected():
    shots, _ = build_2d()
    with pytest.raises(ValueError, match="not compatible"):
        ShotUndersampler.undersample_shots(
            shots, "shell_based", 2.0, {}, trajectory_name="eccentric_2d")


#**************************************************************************************************#
#                                           stack                                                  #
#**************************************************************************************************#
def test_stack_redraws_per_partition_and_reproduces():
    make = lambda: TrajectoryRegistry.create(
        'stack_of_eccentric', n_shots=32, seed=11).generate(GEOM_3D)
    shots, meta = make()

    n_ip = meta["n_inplane_shots"]
    assert meta["n_shots"] == meta["n_kz_slices"] * n_ip == len(shots)
    assert meta["center_crossing"].shape == (meta["n_shots"],)

    kz0 = np.asarray(shots[0])
    kz1 = np.asarray(shots[n_ip])
    assert not np.allclose(kz0[:, :2], kz1[:, :2]), \
        "partitions must not share one circle layout"
    assert np.allclose(kz0[:, 2], kz0[0, 2]), "kz constant within a shot"

    shots_again, _ = make()
    assert np.allclose(np.asarray(shots[5]), np.asarray(shots_again[5])), \
        "seeded stack must reproduce"


def test_stack_crossing_predicate_ignores_kz_offset():
    shots, meta = TrajectoryRegistry.create(
        'stack_of_eccentric', n_shots=32, seed=11).generate(GEOM_3D)
    detected = ShotUndersampler._us_center_crossing(len(shots), 1e9, {}, shots)
    assert detected[meta["center_crossing"]].all(), \
        "in-plane crossing must be detected on every kz partition"


def test_stack_requires_3d_geometry():
    with pytest.raises(ValueError, match="3D geometry"):
        TrajectoryRegistry.create('stack_of_eccentric', n_shots=8).generate(
            {**GEOM_2D, "ndim": 2})
