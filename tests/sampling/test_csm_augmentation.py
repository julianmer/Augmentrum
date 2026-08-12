####################################################################################################
#                                   test_csm_augmentation.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds array placement to being a placement - the same coils seen from a different       #
#          angle - rather than a degradation dressed up as one.                                    #
#                                                                                                  #
####################################################################################################

"""
Tests for moving a receive array relative to the object.

A coil array is not bolted to the subject: between sessions it sits at a
different angle, so the same array gives different sensitivities. Training on
one fixed placement teaches a model that geometry rather than the anatomy under
it.

The thing to guard is that moving the array is still a *placement*. Maps that
have been smeared into mush would also "differ from the original", and would
also pass any check that only asks whether something changed - while quietly
destroying the encoding that makes parallel imaging work.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.sampling import Augmented, Birdcage, CoilSampler


def _rank(maps):
    """Independent channels among *maps*, counting only voxels they see."""
    flat = maps.reshape(-1, maps.shape[-1])
    return np.linalg.matrix_rank(flat[np.abs(flat).sum(1) > 0])


#***************#
#   placement   #
#***************#
def test_the_array_moves():
    """Otherwise there is no augmentation at all."""
    plain = Birdcage(n_coils=8).maps((16, 16, 4))
    moved = Augmented(Birdcage(n_coils=8), rotation_deg=25.0).maps((16, 16, 4))

    assert not np.allclose(np.abs(moved), np.abs(plain), atol=1e-3)


def test_asking_for_no_movement_changes_nothing():
    """A placement of zero should cost nothing and do nothing."""
    source = Birdcage(n_coils=6)
    still = Augmented(source, rotation_deg=0.0, shift_frac=0.0)

    assert np.allclose(still.maps((8, 8, 2)), source.maps((8, 8, 2)))


def test_every_call_places_it_anew():
    """A batch should not meet the same placement every time."""
    mover = Augmented(Birdcage(n_coils=4), rotation_deg=30.0)

    assert not np.allclose(mover.maps((8, 8, 2)), mover.maps((8, 8, 2)))


#********************#
#   still an array   #
#********************#
def test_the_moved_maps_keep_unit_sensitivity():
    """
    The convention every source shares, so the two stay interchangeable.

    Turning brings in voxels the array never covered, which breaks it at the
    edges unless the result is renormalized.
    """
    moved = Augmented(Birdcage(n_coils=8), rotation_deg=25.0,
                      shift_frac=0.05).maps((16, 16, 4))

    rss = np.sqrt((np.abs(moved) ** 2).sum(-1))
    assert np.allclose(rss[rss > 1e-6], 1.0, atol=1e-4)


def test_moving_the_array_does_not_destroy_its_encoding():
    """
    The claim that matters, and the one a "did it change" test would miss.

    Maps smeared into mush would differ from the original just as much, and
    would also renormalize cleanly - while losing the independence between
    channels that makes an accelerated acquisition recoverable at all.
    """
    plain = Birdcage(n_coils=8).maps((16, 16, 4))
    moved = Augmented(Birdcage(n_coils=8), rotation_deg=25.0,
                      shift_frac=0.05).maps((16, 16, 4))

    assert _rank(moved) >= _rank(plain) - 1, (
        f"placement cost {_rank(plain) - _rank(moved)} independent channels"
    )


def test_the_maps_stay_smooth():
    """
    A sensitivity map is low-frequency, and a placement does not change that.

    Interpolation artifacts from the turn would show up here as energy spread
    away from the center of k-space.
    """
    moved = Augmented(Birdcage(n_coils=8), rotation_deg=25.0).maps((32, 32, 8))

    energy = np.abs(np.fft.fftshift(np.fft.fftn(moved[..., 0], axes=(0, 1, 2)),
                                    axes=(0, 1, 2))) ** 2
    cx, cy = energy.shape[0] // 2, energy.shape[1] // 2
    core = energy[cx - 4:cx + 5, cy - 4:cy + 5, :].sum() / energy.sum()

    assert core > 0.8, f"only {core:.1%} of the moved map is low-frequency"


#***************#
#   composing   #
#***************#
def test_it_is_a_source_like_any_other():
    """It wraps a source and is one, so it goes wherever they go."""
    volume = np.ones((1, 16, 16, 4, 8), np.complex64)
    out, _ = CoilSampler(mode='synthesize',
                         source=Augmented(Birdcage(n_coils=8))).process_tensor(volume)

    assert out.shape == volume.shape + (8,)


def test_it_can_move_a_supplied_array():
    """Measured maps are worth moving too - more so, being one real placement."""
    from augmentrum.sampling import Supplied

    measured = Birdcage(n_coils=6).maps((12, 12, 4))
    moved = Augmented(Supplied(measured), rotation_deg=20.0).maps((12, 12, 4))

    assert moved.shape == measured.shape
    assert not np.allclose(np.abs(moved), np.abs(measured), atol=1e-3)


@pytest.mark.parametrize("n_coils", [4, 8])
def test_the_coil_count_is_preserved(n_coils):
    """Moving an array does not add or lose elements."""
    moved = Augmented(Birdcage(n_coils=n_coils), rotation_deg=15.0).maps((8, 8, 2))

    assert moved.shape[-1] == n_coils


#*****************#
#   reweighting   #
#*****************#
def _with_array(n_coils=6, seed=0):
    """Rank-1 multi-coil data built by synthesis, plus its combined source."""
    rng = np.random.default_rng(seed)
    combined = (rng.standard_normal((2, 4, 4, 2, 16))
                + 1j * rng.standard_normal((2, 4, 4, 2, 16))).astype(np.complex64)
    coiled, _ = CoilSampler(mode='synthesize',
                            source=Birdcage(n_coils=n_coils)).process_tensor(combined)
    return combined, np.asarray(coiled)


def _rss(x, axis=-1):
    return np.sqrt((np.abs(x) ** 2).sum(axis=axis))


def test_reweighting_with_the_same_array_returns_the_data():
    """Estimate, combine, reapply the same maps: rank-1 data comes back."""
    _, coiled = _with_array(n_coils=6)
    out, _ = CoilSampler(mode='reweight',
                         source=Birdcage(n_coils=6)).process_tensor(coiled)

    np.testing.assert_allclose(np.abs(np.asarray(out)), np.abs(coiled),
                               rtol=1e-3, atol=1e-4)


def test_reweighting_can_resize_the_array():
    """8 elements in, 3 out — with the total sensitivity preserved."""
    _, coiled = _with_array(n_coils=8)
    out, _ = CoilSampler(mode='reweight', n_coils=3,
                         source=Birdcage(n_coils=8)).process_tensor(coiled)

    out = np.asarray(out)
    assert out.shape[-1] == 3
    np.testing.assert_allclose(_rss(out), _rss(coiled), rtol=1e-3, atol=1e-4)


def test_reweighting_moves_a_real_array_in_one_step():
    """CSM augmentation of multi-coil data, no separate combination step."""
    _, coiled = _with_array(n_coils=6)
    sampler = CoilSampler(mode='reweight',
                          source=Augmented(Birdcage(n_coils=6), rotation_deg=20.0),
                          seed=0)
    out, _ = sampler.process_tensor(coiled)

    out = np.asarray(out)
    assert out.shape == coiled.shape
    assert not np.allclose(np.abs(out), np.abs(coiled), atol=1e-3), \
        "the moved array must weight the volume differently"
    np.testing.assert_allclose(_rss(out), _rss(coiled), rtol=1e-3, atol=1e-4)
