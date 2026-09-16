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


#***************#
#   averaging   #
#***************#
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


#********************#
#   magnitude data   #
#********************#
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


#************************#
#   where noise enters   #
#************************#
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


#********************#
#   how loud where   #
#********************#
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


#*****************#
#   the level     #
#*****************#
# A level is defined in the unitary spectrum: sigma is the per-channel SD of the
# time-domain noise, snr the spectrum peak over that SD, sigma_frac its inverse.
# None of them may depend on where in the pipeline the module sits, or on
# whether it meets the data per coil, per voxel or combined.

def _lorentzian(n=4096, sw=4000.0, batch=4, amp=1.0):
    """A clean FID, so everything added to it is the noise under test."""
    t = np.arange(n) / sw
    fid = amp * np.exp(-t / 0.05) * np.exp(2j * np.pi * -300.0 * t)
    return np.tile(fid.astype(np.complex64), (batch, 1, 1, 1, 1))


def _spectrum(fid):
    """The unitary spectrum the level is defined in."""
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm='ortho'), axes=-1)


def _to_frequency(fid):
    """What a DomainTransform hands a module placed in the frequency domain."""
    return np.fft.fftshift(np.fft.ifft(fid, axis=-1), axes=-1)


def _to_time(spec):
    return np.fft.fft(np.fft.ifftshift(spec, axes=-1), axis=-1)


def test_snr_is_the_spectrum_peak_over_the_noise_sd():
    """
    The definition, on a clean signal.

    Peak height over the real-part noise SD is how MRS reports SNR, and it is
    what index files and QC tables hold - so it is what the parameter must
    mean, or a training set labelled "SNR 20" is not.
    """
    signal = _lorentzian()
    noisy, _ = Noise(snr=10.0, seed=0).process_tensor(signal)

    added = _spectrum(np.asarray(noisy) - signal).real
    peak = np.abs(_spectrum(signal[0])).max()

    assert np.isclose(peak / added.std(), 10.0, rtol=0.03)


def test_sigma_is_the_time_domain_sd_per_channel():
    """Absolute, and the same on the real and imaginary channel."""
    signal = _lorentzian()
    noisy, _ = Noise(sigma=0.3, seed=0).process_tensor(signal)

    added = np.asarray(noisy) - signal
    assert np.isclose(added.real.std(), 0.3, rtol=0.03)
    assert np.isclose(added.imag.std(), 0.3, rtol=0.03)


def test_sigma_frac_is_one_over_snr():
    """Two spellings of one quantity draw the same noise."""
    signal = _lorentzian()
    a, _ = Noise(sigma_frac=0.1, seed=0).process_tensor(signal)
    b, _ = Noise(snr=10.0, seed=0).process_tensor(signal)

    assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_snr_db_is_twenty_log_ten():
    """A peak over a SD is an amplitude ratio, so its decibels are 20 log10."""
    signal = _lorentzian()
    a, _ = Noise(snr_db=20.0, seed=0).process_tensor(signal)
    b, _ = Noise(snr=10.0, seed=0).process_tensor(signal)

    assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_the_noise_already_there_is_not_subtracted():
    """
    The parameter describes what is added.

    Real data always carries noise of its own, and taking it out of the
    request would make the result depend on how well it could be measured.
    Variances add, so the outcome is predictable instead.
    """
    signal = _lorentzian()
    # Not seed 1: a SeedGenerator seeded 0 keys its first draw with exactly
    # that, and the "existing" noise would be the module's own.
    rng = np.random.default_rng(12345)
    already = 0.2 * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    noisy, _ = Noise(sigma=0.3, seed=0).process_tensor((signal + already).astype(np.complex64))

    total = (np.asarray(noisy) - signal).real.std()
    assert np.isclose(total, np.sqrt(0.2 ** 2 + 0.3 ** 2), rtol=0.03)


#****************************#
#   placement in a pipeline  #
#****************************#
@pytest.mark.parametrize("level", [dict(sigma=0.3), dict(snr=10.0),
                                   dict(sigma_frac=0.1), dict(snr_db=20.0)])
def test_the_level_means_the_same_in_either_domain(level):
    """
    The reason the module reads the spectral state.

    The pipeline's spectral transform is not unitary, so the same numbers
    added on either side of it are different noise on the FID. Every level
    has to come out identical once the data is back in the time domain.
    """
    signal = _lorentzian()

    in_time, _ = Noise(seed=0, **level).process_tensor(
        signal, state=DataState(spectral='time'))
    in_frequency, _ = Noise(seed=0, **level).process_tensor(
        _to_frequency(signal), state=DataState(spectral='frequency'))

    added_time = np.asarray(in_time) - signal
    added_frequency = _to_time(np.asarray(in_frequency)) - signal

    assert np.isclose(added_frequency.real.std(), added_time.real.std(), rtol=0.03)


def test_a_pipeline_placement_in_the_frequency_domain_adds_the_same_noise():
    """
    End to end, so the state actually reaches the module.

    A DomainTransform on either side is exactly what a pipeline inserts for a
    frequency-domain neighbour; the noise module in between must not notice.
    """
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    from nifti_mrs_plus import Backend, NIfTI_MRS_Plus

    from augmentrum.core.pipeline import AugmentationPipeline
    from augmentrum.processing.domain import DomainTransform

    signal = _lorentzian(batch=1)[0]
    niftis = [gen_nifti_mrs(signal.copy(), 1 / 4000.0, 123.2) for _ in range(4)]

    def run(steps):
        data = NIfTI_MRS_Plus([n.copy() for n in niftis], backend=Backend.NUMPY, volatile=True)
        out, _ = AugmentationPipeline(steps)(data, None)
        return np.asarray(out.get_data(Backend.NUMPY)) - signal[None]

    direct = run([Noise(snr=10.0, seed=0)])
    moved = run([DomainTransform(spectral='frequency'), Noise(snr=10.0, seed=0),
                 DomainTransform(spectral='time')])

    assert np.isclose(moved.real.std(), direct.real.std(), rtol=0.03)
    assert np.isclose(np.abs(_spectrum(signal)).max() / direct.real.std(), 10.0, rtol=0.05)


def test_kspace_placement_references_the_image():
    """
    A spectrum peak is an image-domain quantity.

    The spatial transform is orthonormal, so the noise level carries over to
    k-space unchanged - but the peak of a k-space trace is a sum over voxels
    and means nothing. Placed in k-space, the module must still add what the
    image asked for.
    """
    volume = np.zeros((1, 8, 8, 1, 256), np.complex64)
    volume[0, 2:6, 2:6, 0, :] = _lorentzian(n=256, batch=1)[0, 0, 0, 0]
    axes = (1, 2, 3)
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume, axes=axes), axes=axes,
                                         norm='ortho'), axes=axes)

    in_image, _ = Noise(snr=10.0, seed=0).process_tensor(
        volume, state=DataState(spatial='image'))
    in_kspace, _ = Noise(snr=10.0, seed=0).process_tensor(
        kspace, state=DataState(spatial='kspace'))
    back = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(np.asarray(in_kspace), axes=axes),
                                        axes=axes, norm='ortho'), axes=axes)

    assert np.isclose((back - volume).real.std(), (np.asarray(in_image) - volume).real.std(),
                      rtol=0.03)


def test_undersampled_data_gets_one_level_from_its_image():
    """The k-space path, with a relative level: referenced to the image, once."""
    volume = np.zeros((1, 8, 8, 1, 256), np.complex64)
    volume[0, 2:6, 2:6, 0, :] = _lorentzian(n=256, batch=1)[0, 0, 0, 0]

    routed, _ = Noise(snr=10.0, seed=0).process_tensor(
        volume, state=DataState(spatial='image', sampling='undersampled'))

    peak = np.abs(_spectrum(volume)).max()
    added = _spectrum(np.asarray(routed) - volume).real
    assert np.isclose(peak / added.std(), 10.0, rtol=0.05)


def test_a_profile_is_not_applied_in_kspace():
    """Position means nothing in k-space, and saying so beats a silent no-op."""
    from augmentrum.augmentation.noise import SuppliedProfile

    ramp = np.linspace(0.5, 1.5, 8)[:, None, None] * np.ones((8, 8, 4))
    signal = np.ones((1, 8, 8, 4, 64), np.complex64)

    with pytest.warns(RuntimeWarning, match="k-space"):
        Noise(profile=SuppliedProfile(ramp), sigma=0.2, seed=0).process_tensor(
            signal, state=DataState(spatial='kspace'))


#****************************#
#   coils, voxels, batches   #
#****************************#
def test_a_coil_array_shares_one_level():
    """
    The noise level is a property of the receiver, not of what a coil sees.

    A far element sees a weak signal at the same noise as a near one. A
    per-coil reference would give it almost no noise, which makes the array
    look far better than it is once combined.
    """
    amplitudes = np.array([1.0, 0.5, 0.1, 0.01], np.float32)
    signal = (_lorentzian(n=1024, batch=1)[..., None] * amplitudes).astype(np.complex64)

    noisy, _ = Noise(snr=10.0, seed=0).process_tensor(signal, dim_tags=COILS)
    per_coil = (np.asarray(noisy) - signal).real.std(axis=(0, 1, 2, 3, 4))

    peak = np.abs(_spectrum(signal[..., 0])).max()
    assert np.allclose(per_coil, peak / 10.0, rtol=0.1)


def test_per_trace_is_still_there_on_request():
    """Explicitly per trace, each coil gets its own reference."""
    amplitudes = np.array([1.0, 0.5, 0.1, 0.01], np.float32)
    signal = (_lorentzian(n=1024, batch=1)[..., None] * amplitudes).astype(np.complex64)

    noisy, _ = Noise(snr=10.0, seed=0, global_scale=False).process_tensor(
        signal, dim_tags=COILS)
    per_coil = (np.asarray(noisy) - signal).real.std(axis=(0, 1, 2, 3, 4))

    assert np.allclose(per_coil / per_coil[0], amplitudes, rtol=0.1)


def test_a_volume_gives_the_background_no_free_mask():
    """
    Every voxel of an MRSI grid sits in the same receiver noise.

    A per-voxel reference would leave the background nearly silent, tracing
    the anatomy and handing a network a brain mask it never had to learn.
    """
    volume = np.zeros((1, 4, 4, 1, 1024), np.complex64)
    volume[0, 1:3, 1:3, 0, :] = _lorentzian(n=1024, batch=1)[0, 0, 0, 0]

    noisy, _ = Noise(snr=10.0, seed=0).process_tensor(volume)
    added = (np.asarray(noisy) - volume).real.std(axis=-1)[0, :, :, 0]

    assert np.isclose(added[0, 0], added[1, 1], rtol=0.1)


def test_each_batch_element_gets_its_own_reference():
    """One level per subject, never one per batch: a weak subject is not drowned."""
    signal = np.concatenate([_lorentzian(n=1024, batch=1, amp=1.0),
                             _lorentzian(n=1024, batch=1, amp=0.1)])

    noisy, _ = Noise(snr=10.0, seed=0).process_tensor(signal)
    per_subject = (np.asarray(noisy) - signal).real.std(axis=-1).ravel()

    assert np.isclose(per_subject[1] / per_subject[0], 0.1, rtol=0.1)


def test_the_list_engine_agrees_with_the_tensor_engine():
    """One subject is one batch element, whichever engine handles it."""
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    signal = _lorentzian(n=1024, batch=1)[0]
    nifti = gen_nifti_mrs(signal.copy(), 1 / 4000.0, 123.2)

    listed, _ = Noise(snr=10.0, seed=0).process_nifti_list([nifti])
    tensored, _ = Noise(snr=10.0, seed=0).process_tensor(signal[None])

    assert np.allclose(np.asarray(listed[0][:]), np.asarray(tensored)[0], rtol=1e-5, atol=1e-6)


#**************#
#   backends   #
#**************#
def test_numpy_and_torch_agree_on_the_level():
    """
    The level is a deterministic function of the data, on any backend.

    The samples themselves come from each framework's own generator, so a
    seed reproduces a run on a backend rather than across backends; what must
    agree is how loud it is.
    """
    torch = pytest.importorskip("torch")
    signal = _lorentzian()

    with_numpy, _ = Noise(snr=10.0, seed=0).process_tensor(signal)
    with_torch, _ = Noise(snr=10.0, seed=0).process_tensor(torch.as_tensor(signal))
    again, _ = Noise(snr=10.0, seed=0).process_tensor(torch.as_tensor(signal))

    added_numpy = (np.asarray(with_numpy) - signal).real.std()
    added_torch = (with_torch.numpy() - signal).real.std()

    assert np.isclose(added_torch, added_numpy, rtol=0.03)
    assert torch.equal(with_torch, again)
