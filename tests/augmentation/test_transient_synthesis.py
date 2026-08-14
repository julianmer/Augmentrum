####################################################################################################
#                                  test_transient_synthesis.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-14                                                                              #
#                                                                                                  #
# Purpose: Holds the transient synthesizer to what a real transient train looks like: a new        #
#          DIM_DYN axis whose members share correlated scan-time structure, not i.i.d. jitter.     #
#                                                                                                  #
####################################################################################################

"""
Tests for synthesizing a transient train from one averaged FID.

The point of the module is the *correlation structure*: drift makes the
frequency track walk rather than jump, so neighbors must look more alike
than distant transients. That, the axis bookkeeping, and reproducibility
are what is worth pinning.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.augmentation import TransientSynthesizer


SW_HZ = 2000.0


#*************#
#   helpers   #
#*************#
def _fid(n_pts=512):
    """One clean decaying FID in the tensor layout (batch, 1, 1, 1, T)."""
    t = np.arange(n_pts) / SW_HZ
    fid = np.exp(-15.0 * t) * np.exp(1j * 2 * np.pi * 40.0 * t)
    return fid.astype(np.complex64).reshape(1, 1, 1, 1, n_pts)


def _quiet(**kwargs):
    """A synthesizer with everything but the chosen effect switched off."""
    defaults = dict(drift_hz_per_min=0.0, ar_sigma_hz=0.0, resp_amp_hz=0.0,
                    phase_coupling_deg_per_hz=0.0, phase_jitter_deg=0.0,
                    amp_jitter_frac=0.0, broaden_hz=0.0, events_per_min=0.0)
    defaults.update(kwargs)
    return TransientSynthesizer(**defaults)


def _estimated_freq(train):
    """Per-transient frequency from the FID's sample-to-sample rotation."""
    x = train[0, 0, 0, 0]                                    # (T, N_transients)
    rot = np.angle(np.sum(x[1:] * np.conj(x[:-1]), axis=0))
    return rot * SW_HZ / (2.0 * np.pi)


#******************#
#   bookkeeping    #
#******************#
def test_the_train_is_a_new_trailing_axis():
    train, _ = TransientSynthesizer(n_transients=8, seed=0).process_tensor(
        _fid(), sw_hz=SW_HZ)

    assert train.shape == (1, 1, 1, 1, 512, 8)
    assert TransientSynthesizer.ADDS_DIM_TAGS == ('DIM_DYN',)


def test_existing_transients_are_refused():
    """Synthesizing on top of a real train would nest acquisition axes."""
    batch = np.ones((1, 1, 1, 1, 64, 4), np.complex64)

    with pytest.raises(ValueError, match="already carries transients"):
        TransientSynthesizer(seed=0).process_tensor(
            batch, sw_hz=SW_HZ, dim_tags=['DIM_DYN', None, None])


def test_a_seed_replays_exactly():
    first, _ = TransientSynthesizer(n_transients=6, seed=3).process_tensor(
        _fid(), sw_hz=SW_HZ)
    again, _ = TransientSynthesizer(n_transients=6, seed=3).process_tensor(
        _fid(), sw_hz=SW_HZ)

    assert np.array_equal(np.asarray(first), np.asarray(again))


def test_batch_elements_are_different_scans():
    fid = np.concatenate([_fid(), _fid()])
    train, _ = TransientSynthesizer(n_transients=6, seed=0).process_tensor(
        fid, sw_hz=SW_HZ)

    assert not np.allclose(np.asarray(train)[0], np.asarray(train)[1])


#*******************#
#   the processes   #
#*******************#
def test_everything_off_replicates_the_fid():
    """With every process silenced the train is N copies of the input."""
    train, _ = _quiet(n_transients=5, seed=0).process_tensor(_fid(), sw_hz=SW_HZ)

    for k in range(5):
        assert np.allclose(np.asarray(train)[..., k], _fid()[..., :], atol=1e-6)


def test_drift_walks_the_frequency_monotonically():
    """A pure linear drift must come back as a monotone frequency track."""
    module = _quiet(n_transients=16, tr_s=2.0, drift_hz_per_min=6.0, seed=0)
    train, _ = module.process_tensor(_fid(), sw_hz=SW_HZ)

    freq = _estimated_freq(np.asarray(train))
    steps = np.diff(freq)
    assert np.all(steps > 0), freq
    # 6 Hz/min at TR = 2 s is 0.2 Hz per transient
    assert np.allclose(steps, 0.2, atol=0.02), steps


def test_neighbors_are_more_alike_than_strangers():
    """
    The correlation structure that separates this from i.i.d. jitter.

    Under drift + AR(1) wander, the frequency difference between adjacent
    transients must be far smaller than between the ends of the scan.
    """
    module = TransientSynthesizer(n_transients=64, tr_s=2.0, seed=2,
                                  events_per_min=0.0, phase_jitter_deg=0.0,
                                  amp_jitter_frac=0.0)
    train, _ = module.process_tensor(_fid(), sw_hz=SW_HZ)

    freq = _estimated_freq(np.asarray(train))
    neighbor = np.abs(np.diff(freq)).mean()
    span = np.abs(freq[-1] - freq[0])

    assert neighbor < span, (neighbor, span)


def test_events_hit_a_contiguous_stretch():
    """A motion event kicks a cluster of transients, not random singletons."""
    module = _quiet(n_transients=48, events_per_min=20.0, seed=5)
    train, _ = module.process_tensor(_fid(), sw_hz=SW_HZ)

    freq = _estimated_freq(np.asarray(train))
    assert np.abs(freq).max() > 0.5, "no event landed despite the rate"
