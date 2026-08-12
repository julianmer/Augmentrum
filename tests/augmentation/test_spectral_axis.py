####################################################################################################
#                                     test_spectral_axis.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds every spectral module to acting along the spectral axis, including when a coil    #
#          or average axis sits behind it and the spectral points are no longer last.              #
#                                                                                                  #
####################################################################################################

"""
Tests that spectral modules act along the spectrum, not along whatever is last.

Every spectral module is written against the last axis, because that is where
the points sit in the usual "(batch, X, Y, Z, T)". Put a coil axis behind it and
the last axis is a coil - so a line broadening decays across channels instead of
along the FID, and returns a perfectly plausible volume while doing it.

Nothing raises when this is wrong, and the shape is right either way, so the
only way to see it is to look at *which* axis changed. That is what these do.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

from augmentrum.core import Backend, NIfTI_MRS_Plus
from augmentrum.core.pipeline import AugmentationPipeline
from augmentrum.augmentation import (
    ArtificialPeaks, Apodization, LineBroadening, ResidualWater,
)
from augmentrum.sampling import CoilSampler

from tests.module_specs import SPECS


#**************#
#   fixtures   #
#**************#
def _fid(n_points=64, sw_hz=2000.0):
    """
    A decaying resonance rather than a constant.

    A flat FID is degenerate for several modules - its spectrum is a single
    spike, so a phase ramp changes only the phase and a replica lands on top of
    itself. Anything measured on it would say more about the probe than the
    module.
    """
    t = np.arange(n_points) / sw_hz
    return (np.exp(2j * np.pi * 120.0 * t) * np.exp(-t / 0.15)).astype(np.complex64)


@pytest.fixture
def with_coils():
    """A batch that has been given a receive array, so T is no longer last."""
    volume = np.broadcast_to(_fid(), (2, 2, 1, 64)).astype(np.complex64).copy()
    plus = NIfTI_MRS_Plus(nifti_list=[gen_nifti_mrs(volume, 1 / 2000.0, 123.0)],
                          backend=Backend.NUMPY, volatile=True)
    coiled, _ = CoilSampler(mode='synthesize', n_coils=4)(plus)
    return coiled


def _values(plus):
    return np.asarray(plus.get_data(Backend.NUMPY))


#***********************#
#   the spectral axis   #
#***********************#
def test_broadening_decays_the_fid_not_the_coils(with_coils):
    """
    The case that showed the bug.

    A 20 Hz broadening over 64 points at 2 kHz should leave the last point at
    about 0.135 of the first. Applied to a four-point coil axis instead it
    barely decays at all - and nothing about the result looks wrong.
    """
    before = _values(with_coils)
    after = _values(LineBroadening(lb_hz=20.0)(with_coils)[0])

    envelope = np.abs(after / before)[0, 0, 0, 0]

    assert np.isclose(envelope[-1, 0], 0.135, atol=0.02), "the FID was not broadened"
    assert np.allclose(envelope[:, 0], envelope[:, 3], atol=1e-5), (
        "the coils were given different envelopes"
    )


def test_the_envelope_is_the_same_on_every_channel(with_coils):
    """
    A receive array measures one signal, so one envelope applies to all of it.

    Broadening along the coil axis would give each channel a different scale,
    which is the signature of the axis mix-up.
    """
    before = _values(with_coils)
    after = _values(Apodization(mode='exponential', lb_hz=10.0)(with_coils)[0])

    envelope = np.abs(after / before)[0, 0, 0, 0]
    spread = envelope.std(axis=1).max()

    assert spread < 1e-5, f"channels got different envelopes, spread {spread:.2e}"


@pytest.mark.parametrize("module", [ResidualWater(), ArtificialPeaks()],
                         ids=lambda m: type(m).__name__)
def test_additions_have_the_same_shape_on_every_channel(with_coils, module):
    """
    Water and peaks are features of a spectrum, so their shape along it is
    the same for every channel.

    How *much* is added does differ per channel, and legitimately so: the
    amplitude is taken from each channel's own signal, and a channel that sees
    more signal gets proportionally more water. What cannot differ is the
    profile, because that comes from the ppm axis alone. Added along the coil
    axis instead, no two channels would share a profile.
    """
    added = _values(module(with_coils)[0]) - _values(with_coils)
    trace = added[0, 0, 0, 0]

    assert np.abs(trace).max() > 0, "nothing was added at all"

    # scale each channel by its own peak, so only the shape is compared
    peaks = np.abs(trace).max(axis=0, keepdims=True)
    shapes = np.abs(trace) / np.where(peaks > 0, peaks, 1.0)

    assert abs(np.corrcoef(shapes[:, 0], shapes[:, -1])[0, 1]) > 0.99, (
        "the channels were given different spectral profiles"
    )


#***************#
#   the sweep   #
#***************#
@pytest.mark.parametrize(
    "spec", [s for s in SPECS if not s.spatial and not s.volume
             and not s.needs_multicoil and not s.changes_length],
    ids=lambda s: s.label)
def test_no_module_acts_on_the_wrong_axis(spec, with_coils):
    """
    Swept from the registry, so a new module cannot quietly reintroduce this.

    The signature of the bug is that the FID is left untouched: a module acting
    on the coil axis scales each trace uniformly, so its change is *constant*
    along the spectrum. That is what is checked, rather than anything about the
    channels - several modules draw independently per trace, which is a separate
    question and a legitimate one.
    """
    module = spec.build()
    if getattr(module, 'DOMAIN', None) is None or module.DOMAIN.spectral is None:
        pytest.skip(f"{spec.label} does not read along the spectral axis")

    change = _values(module(with_coils)[0]) - _values(with_coils)
    if np.abs(change).max() == 0:
        pytest.skip(f"{spec.label} left the data untouched")

    trace = np.abs(change[0, 0, 0, 0])
    level = trace.mean(axis=0)
    varies = trace.std(axis=0) / np.where(level > 0, level, 1.0)

    assert varies.max() > 0.01, (
        f"{spec.label} changed the data by a constant along the spectrum, which "
        f"is what acting on the coil axis looks like"
    )


def test_a_pipeline_that_adds_coils_then_broadens(with_coils):
    """End to end, since that is the order a real pipeline uses."""
    volume = np.broadcast_to(_fid(), (2, 2, 1, 64)).astype(np.complex64).copy()
    plus = NIfTI_MRS_Plus(nifti_list=[gen_nifti_mrs(volume, 1 / 2000.0, 123.0)],
                          backend=Backend.NUMPY, volatile=True)

    out, _ = AugmentationPipeline([CoilSampler(mode='synthesize', n_coils=4),
                                   LineBroadening(lb_hz=20.0)])(plus)
    result = _values(out)

    assert result.shape == (1, 2, 2, 1, 64, 4)

    # the FID already decays, so compare against the same pipeline without it
    plain, _ = AugmentationPipeline([CoilSampler(mode='synthesize', n_coils=4)])(plus)
    extra = (np.abs(result[0, 0, 0, 0, :, 0])
             / np.abs(_values(plain)[0, 0, 0, 0, :, 0]))
    assert np.isclose(extra[-1] / extra[0], 0.135, atol=0.02)
