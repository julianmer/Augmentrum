####################################################################################################
#                                    test_noise_physics.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds the noise module to what a scanner actually produces, rather than to the shape    #
#          of its output - which is the only part that would notice if it were wrong.              #
#                                                                                                  #
####################################################################################################

"""
Tests for noise that works out what it should be from where it is.

Everything here is a statistical claim rather than a value, because noise has no
values worth asserting. The claims are the ones that would let a study draw the
wrong conclusion if they failed: averaging has to pay off as sqrt(N), a
correlated array must not look as good as an independent one, magnitude data
must not go negative, and noise must not be added to an undersampled image as
though it were still white there.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs_plus.core import DataState

from augmentrum.augmentation.noise import (
    FromSensitivity, Independent, Noise, SuppliedCovariance,
)
from augmentrum.sampling import Birdcage


#: A coil axis sitting where NIfTI-MRS puts it.
COILS = ['DIM_COIL', None, None]

#: Coils and averages together.
COILS_AND_AVERAGES = ['DIM_COIL', 'DIM_DYN', None]


#*******************#
#   averaging       #
#*******************#
@pytest.mark.parametrize("n_averages", [1, 4, 16, 64])
def test_averaging_pays_off_as_the_root_of_the_count(n_averages):
    """
    The trade every protocol makes.

    Only true if each average gets its own draw. Adding one noise realization
    and repeating it would leave the residual unchanged however many averages
    were taken, and the module would look fine on every shape check.
    """
    signal = np.ones((1, 1, 1, 1, 256, 2, n_averages), np.complex64)
    noisy, _ = Noise(sigma=0.1, seed=0).process_tensor(
        signal, dim_tags=COILS_AND_AVERAGES)

    combined = np.asarray(noisy).mean(axis=-1)
    expected = 0.1 * np.sqrt(2) / np.sqrt(n_averages)

    assert np.isclose(np.std(combined - 1.0), expected, rtol=0.1)


#*******************#
#   coil coupling   #
#*******************#
def test_independent_channels_stay_independent():
    """The default model, and the baseline the correlated one is measured against."""
    signal = np.ones((1, 4, 4, 2, 300, 6), np.complex64)
    noisy, _ = Noise(covariance=Independent(), sigma=0.2, seed=0).process_tensor(
        signal, dim_tags=COILS)

    drawn = np.corrcoef((np.asarray(noisy) - 1.0).reshape(-1, 6).T.real)

    assert np.abs(drawn - np.eye(6)).max() < 0.1


def test_the_drawn_noise_has_the_covariance_it_was_asked_for():
    """
    A requested psi has to actually come out.

    Drawing white noise and calling it correlated would make an array look
    better than it is, because independent channels carry more information than
    coupled ones at the same level.
    """
    maps = Birdcage(n_coils=6).maps((8, 8, 4))
    covariance = FromSensitivity(maps)

    signal = np.ones((1, 8, 8, 4, 400, 6), np.complex64)
    noisy, _ = Noise(covariance=covariance, sigma=0.2, seed=0).process_tensor(
        signal, dim_tags=COILS)

    drawn = np.corrcoef((np.asarray(noisy) - 1.0).reshape(-1, 6).T.real)
    wanted = np.real(covariance.matrix(6))

    assert np.abs(np.abs(drawn) - np.abs(wanted)).mean() < 0.1


def test_a_supplied_covariance_is_used_as_given():
    """A caller who measured psi should get psi."""
    wanted = np.array([[1.0, 0.8], [0.8, 1.0]], np.complex64)

    signal = np.ones((1, 4, 4, 2, 4000, 2), np.complex64)
    noisy, _ = Noise(covariance=SuppliedCovariance(wanted), sigma=0.2,
                     seed=0).process_tensor(signal, dim_tags=COILS)

    drawn = np.corrcoef((np.asarray(noisy) - 1.0).reshape(-1, 2).T.real)

    assert abs(drawn[0, 1] - 0.8) < 0.1


def test_a_covariance_must_match_the_array():
    """Maps for a different array are a mistake, not something to broadcast."""
    with pytest.raises(ValueError, match="channels"):
        FromSensitivity(Birdcage(n_coils=4).maps((4, 4, 2))).matrix(8)


def test_data_without_coils_is_left_alone():
    """Nothing named DIM_COIL means there is nothing to correlate."""
    signal = np.ones((1, 1, 1, 1, 512), np.complex64)
    noisy, _ = Noise(covariance=FromSensitivity(Birdcage(n_coils=4).maps((4, 4, 2))),
                     sigma=0.1, seed=0).process_tensor(signal)

    assert np.asarray(noisy).shape == signal.shape


#*******************#
#   magnitude data  #
#*******************#
def test_magnitude_data_never_goes_negative():
    """
    Rician, not Gaussian.

    A magnitude is the length of a complex number, so noise cannot push it below
    zero. Adding a symmetric perturbation would, and on bright data nobody would
    notice - it only shows where the signal is near the noise floor.
    """
    signal = np.full((1, 1, 1, 1, 20000), 0.5, np.float32)
    noisy, _ = Noise(sigma=1.0, seed=0).process_tensor(signal)

    assert np.asarray(noisy).min() >= 0.0


def test_magnitude_noise_biases_upward():
    """
    The Rician noise floor, which is what makes it the right model.

    Taking the magnitude of a complex signal in noise gives a mean *above* the
    true value, and that bias is exactly why quantifying from magnitude spectra
    at low signal is hard. Gaussian noise would leave the mean unchanged.
    """
    signal = np.full((1, 1, 1, 1, 40000), 1.0, np.float32)
    noisy, _ = Noise(sigma=1.0, seed=0).process_tensor(signal)

    assert np.asarray(noisy).mean() > 1.05


def test_complex_data_is_not_biased():
    """The counterpart: complex noise is symmetric and shifts nothing."""
    signal = np.ones((1, 1, 1, 1, 40000), np.complex64)
    noisy, _ = Noise(sigma=1.0, seed=0).process_tensor(signal)

    assert abs(np.asarray(noisy).mean() - 1.0) < 0.05


#***********************#
#   where noise enters  #
#***********************#
def test_undersampled_image_data_is_noised_in_kspace():
    """
    The reason the module reads the state at all.

    Noise enters at the receiver. While the data is fully sampled the transform
    is orthonormal so it does not matter which side it is added on - but a
    zero-filled reconstruction of undersampled k-space has correlated noise, and
    adding white noise there would resemble nothing a scanner produces.
    """
    signal = np.zeros((1, 16, 16, 1, 64), np.complex64)
    signal[0, 4:12, 4:12, 0, :] = 1.0

    in_image, _ = Noise(sigma=0.1, seed=0).process_tensor(
        signal, state=DataState(spatial='image', sampling='full'))
    routed, _ = Noise(sigma=0.1, seed=0).process_tensor(
        signal, state=DataState(spatial='image', sampling='undersampled'))

    assert not np.allclose(np.asarray(in_image), np.asarray(routed)), (
        "the undersampled case took the same path as the fully sampled one"
    )
    # same level either way - it is where it was added that differs
    assert np.isclose((np.asarray(routed) - signal).std(),
                      (np.asarray(in_image) - signal).std(), rtol=0.1)


def test_it_is_droppable_anywhere():
    """It never forces the pipeline to move the data."""
    assert Noise(snr=30).DOMAIN is None


#***********************#
#   how loud where      #
#***********************#
# Noise is not flat across a volume: sensitivity falls off and parallel imaging
# amplifies unevenly. A profile says where it is louder and averages to one, so
# the level stays whatever was asked for.

def test_a_flat_profile_changes_nothing():
    """The default, and the baseline the others are measured against."""
    from augmentrum.augmentation.noise import Flat

    signal = np.ones((1, 8, 8, 4, 300), np.complex64)
    noisy, _ = Noise(profile=Flat(), sigma=0.2, seed=0).process_tensor(signal)

    drawn = (np.asarray(noisy) - 1.0)[0]
    assert np.isclose(drawn[7].std() / drawn[0].std(), 1.0, rtol=0.15)


def test_the_drawn_level_follows_the_profile():
    """
    What a profile is for.

    Asserted on the level actually drawn rather than on the profile, because a
    profile that is computed and then ignored would pass any check of itself.
    """
    from augmentrum.augmentation.noise import SuppliedProfile

    ramp = np.linspace(0.5, 1.5, 8)[:, None, None] * np.ones((8, 8, 4))
    signal = np.ones((1, 8, 8, 4, 400), np.complex64)
    noisy, _ = Noise(profile=SuppliedProfile(ramp), sigma=0.2, seed=0).process_tensor(signal)

    drawn = (np.asarray(noisy) - 1.0)[0]
    assert np.isclose(drawn[7].std() / drawn[0].std(), 3.0, rtol=0.15)


def test_a_profile_averages_to_one():
    """It says where the noise is louder, never how loud overall."""
    from augmentrum.augmentation.noise import SuppliedProfile

    ramp = np.linspace(0.5, 1.5, 8)[:, None, None] * np.ones((8, 8, 4))
    assert np.isclose(SuppliedProfile(ramp).sigma((8, 8, 4)).mean(), 1.0, rtol=1e-5)


def test_a_profile_must_cover_the_grid():
    """A profile for a different matrix is a mistake, not something to stretch."""
    from augmentrum.augmentation.noise import SuppliedProfile

    with pytest.raises(ValueError, match="covers"):
        SuppliedProfile(np.ones((4, 4, 2))).sigma((8, 8, 4))


def test_a_spectrum_with_no_extent_is_left_alone():
    """A single-voxel acquisition has nowhere for the level to vary."""
    from augmentrum.augmentation.noise import SuppliedProfile

    signal = np.ones((1, 1, 1, 1, 512), np.complex64)
    noisy, _ = Noise(profile=SuppliedProfile(np.ones((1, 1, 1))), sigma=0.1,
                     seed=0).process_tensor(signal)

    assert np.asarray(noisy).shape == signal.shape
