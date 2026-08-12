####################################################################################################
#                                    test_spatial_kspace.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds the k-space form of spatial augmentation to the image-domain one, which is the    #
#          only thing that makes it the same augmentation rather than a different operation.       #
#                                                                                                  #
####################################################################################################

"""
Tests that augmenting in k-space matches augmenting the image.

Three things here fail silently if they are wrong, and each was wrong at some
point while this was written:

- the linear part is the *inverse transpose*, which for a rotation equals the
  matrix itself, so a rotation-only test cannot tell a correct implementation
  from a broken one;
- the shift is a phase ramp whose sign and axis pairing are easy to invert, and
  an inverted one still returns a plausible volume;
- the relation holds between voxel coordinates, while sampling grids normalize
  every axis independently, so a square grid hides the aspect ratio entirely.

Every case below therefore uses an anisotropic map on a rectangular grid unless
it is specifically testing the case that hides a bug.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.augmentation import SpatialAugmentations


#*************#
#   helpers   #
#*************#
def _to_kspace(x):
    """Image to k-space over the two spatial axes, DC at the center."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(1, 2)),
                                       axes=(1, 2), norm='ortho'), axes=(1, 2))


def _to_image(x):
    """The way back."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(1, 2)),
                                        axes=(1, 2), norm='ortho'), axes=(1, 2))


def _object(shape):
    """An asymmetric object, so any wrong transform shows up."""
    height, width = shape
    x = np.zeros((1, height, width, 1), np.complex64)
    x[0, height // 5:height // 2, width // 3:3 * width // 4, 0] = 1.0
    x[0, height // 5:height // 5 + 3, width // 3:width // 3 + 4, 0] = 2.0
    return x


def _agreement(shape, linear, shift=(0.0, 0.0)):
    """How well the two domains agree on the same augmentation."""
    aug = SpatialAugmentations(dim=2, prob=1.0)
    x = _object(shape)

    theta = np.zeros((1, 2, 3), np.float32)
    theta[0, :, :2] = linear
    theta[0, :, 2] = shift

    in_image = aug._apply_affine_batch(x, theta)
    in_kspace = _to_image(aug._apply_affine_batch(_to_kspace(x), theta, domain='kspace'))

    return abs(np.vdot(in_image.ravel(), in_kspace.ravel())) / (
        np.linalg.norm(in_image) * np.linalg.norm(in_kspace))


def _rotation(degrees):
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]], np.float32)


#: Square hides the aspect ratio, even hides the half-voxel centring, so the
#: shapes are chosen to expose both.
SHAPES = [(33, 33), (32, 32), (32, 48), (48, 32), (33, 49)]

#: Interpolating k-space is not the same operator as interpolating an image, so
#: a resampled augmentation agrees to about a percent and no better.
RESAMPLED = 0.97

#: A shift and a resampling together are worse than either alone: the ramp makes
#: k-space oscillate, and bilinear interpolation of a fast oscillation is poor.
#: The image path has no equivalent difficulty, so this is the honest ceiling.
COMBINED = 0.94

#: A shift is exact in k-space - it is a phase ramp, no interpolation at all -
#: so the residual here is the *image* path interpolating. A shift of 0.25 is
#: four voxels on a 32-grid and agrees exactly, but 4.125 on a 33-grid, where
#: grid_sample has to interpolate and the k-space answer is the better one.
SHIFTED = 0.998


#****************#
#   the affine   #
#****************#
@pytest.mark.parametrize("shape", SHAPES)
def test_a_rotation_matches(shape):
    """The simplest case, and the one that cannot discriminate the matrix."""
    assert _agreement(shape, _rotation(25.0)) > RESAMPLED


@pytest.mark.parametrize("shape", SHAPES)
def test_an_anisotropic_zoom_matches(shape):
    """
    The case that pins the matrix down.

    For a rotation the inverse transpose equals the matrix, so only an
    anisotropic map separates a correct implementation from three wrong ones.
    """
    assert _agreement(shape, _rotation(25.0) @ np.diag([1.4, 0.7]).astype(np.float32)) > RESAMPLED


@pytest.mark.parametrize("shape", SHAPES)
def test_a_translation_matches(shape):
    """A shift is a phase ramp, and lands on samples, so it should be near exact."""
    assert _agreement(shape, np.eye(2, dtype=np.float32), (0.25, -0.125)) > SHIFTED


@pytest.mark.parametrize("shape", SHAPES)
def test_everything_at_once(shape):
    """
    Rotation, anisotropic zoom and shift together.

    Worth its own case because the composition is not obvious: the displacement
    rides on the transformed coordinate, so the ramp has to be applied before
    the resampling reads it. Getting that backwards leaves rotation alone and
    shift alone both looking perfect while the two together fall to 0.69.
    """
    linear = _rotation(25.0) @ np.diag([1.4, 0.7]).astype(np.float32)
    assert _agreement(shape, linear, (0.2, 0.1)) > COMBINED


#************************#
#   what would hide it   #
#************************#
def test_the_wrong_matrix_would_be_caught():
    """
    The anisotropic case has to actually discriminate.

    If it did not, a rotation-only suite would pass with the matrix that gets
    every anisotropic zoom wrong - which is exactly what happened while this
    was being written.
    """
    aug = SpatialAugmentations(dim=2, prob=1.0)
    shape, linear = (32, 48), _rotation(25.0) @ np.diag([1.4, 0.7]).astype(np.float32)
    x = _object(shape)

    theta = np.zeros((1, 2, 3), np.float32)
    theta[0, :, :2] = linear
    reference = aug._apply_affine_batch(x, theta)

    # the plausible-but-wrong choice: the matrix itself rather than its inverse transpose
    wrong = np.zeros((1, 2, 3), np.float32)
    wrong[0, :, :2] = linear
    got = _to_image(aug._apply_affine_batch(_to_kspace(x), wrong))
    score = abs(np.vdot(reference.ravel(), got.ravel())) / (
        np.linalg.norm(reference) * np.linalg.norm(got))

    assert score < 0.9, "this case no longer separates right from wrong"


def test_it_is_agnostic_so_it_forces_no_transform():
    """Working in both domains means never making the pipeline move the data."""
    assert SpatialAugmentations(dim=2, prob=1.0).DOMAIN is None
