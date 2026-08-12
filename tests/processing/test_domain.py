####################################################################################################
#                                        test_domain.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds domain routing to the two things that make it safe: a module always runs in the   #
#          domain it declared, and the pipeline gets there in as few transforms as it can.         #
#                                                                                                  #
####################################################################################################

"""
Tests for which domain a module works in, and how the data gets there.

The failure this guards against is silent. Line broadening multiplies the FID
by a decay; hand it a spectrum and it still returns a number, just the wrong
one. So the tests check that each module is actually put where it said it
works, not merely that a pipeline runs to completion.
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
    ArtificialPeaks, Noise, LineBroadening, PhaseShift, ResidualWater,
)
from augmentrum.processing import Domain, DomainError, DomainTransform


#**************#
#   fixtures   #
#**************#
@pytest.fixture
def batch():
    """One volume, in the canonical time-domain image-space form."""
    rng = np.random.default_rng(0)
    vol = (rng.standard_normal((6, 6, 2, 32))
           + 1j * rng.standard_normal((6, 6, 2, 32))).astype(np.complex64)
    return NIfTI_MRS_Plus(nifti_list=[gen_nifti_mrs(vol, 1 / 2000.0, 123.0)],
                          backend=Backend.NUMPY, volatile=True)


def _values(plus):
    """The batch as a plain array."""
    return np.asarray(plus.get_data(Backend.NUMPY))


#***************#
#   transform   #
#***************#
@pytest.mark.parametrize("axis,there,back", [('spectral', 'frequency', 'time'),
                                             ('spatial', 'kspace', 'image')])
def test_a_transform_round_trips(batch, axis, there, back):
    """Going and coming back has to leave the data as it was."""
    start = _values(batch)

    out, _ = DomainTransform(**{axis: there})(batch)
    assert getattr(out.state, axis) == there

    returned, _ = DomainTransform(**{axis: back})(out)
    assert np.abs(_values(returned) - start).max() < 1e-5
    assert getattr(returned.state, axis) == back


def test_the_two_axes_commute(batch):
    """
    Which order the transforms happen in cannot matter.

    They act on different axes of the same array, which is why the state keeps
    them apart instead of collapsing them into a single flag.
    """
    first, _ = DomainTransform(spectral='frequency')(batch)
    first, _ = DomainTransform(spatial='kspace')(first)

    other, _ = DomainTransform(spatial='kspace')(batch)
    other, _ = DomainTransform(spectral='frequency')(other)

    assert np.allclose(_values(first), _values(other), atol=1e-5)


def test_a_transform_needs_a_target():
    """Asking for nothing is a mistake, not a no-op."""
    with pytest.raises(ValueError, match="needs a spectral or a spatial target"):
        DomainTransform()


#******************#
#   declarations   #
#******************#
def test_a_module_runs_in_the_domain_it_declared(batch):
    """
    The guarantee everything else rests on.

    Checked by watching what the module is handed, because one in the wrong
    domain fails silently - it still returns a number.
    """
    seen = {}
    module = ResidualWater()
    original = type(module).process_tensor

    def spy(self, data_array, water_array=None, backend=None, **kwargs):
        seen['spectral'] = kwargs.get('state').spectral
        return original(self, data_array, water_array, backend, **kwargs)

    type(module).process_tensor = spy
    try:
        module(batch)
    finally:
        type(module).process_tensor = original

    assert seen['spectral'] == 'frequency', "the module was not put where it declared"


def test_calling_a_module_directly_puts_the_data_back(batch):
    """Outside a pipeline there is nothing to plan against, so it round-trips."""
    out, _ = ResidualWater()(batch)

    assert out.state.spectral == 'time'
    assert out.state.last == 'ResidualWater'


def test_a_module_that_needs_nothing_forces_nothing():
    """A scalar multiply commutes with the transform, so it should not force one."""
    assert Noise(snr=30).DOMAIN is None
    assert PhaseShift(zero_order_deg=30.0).DOMAIN is None


def test_a_domain_is_only_declared_when_it_is_needed():
    """A first-order phase needs a spectrum; a zero-order one does not."""
    assert PhaseShift(first_order_deg=45.0).DOMAIN == Domain(spectral='frequency')
    assert PhaseShift(zero_order_deg=45.0).DOMAIN is None


#**************#
#   the plan   #
#**************#
def test_consecutive_modules_share_one_transform():
    """
    Why the pipeline plans instead of letting each module round-trip.

    Three modules that all want a spectrum should be moved there once, not
    three times.
    """
    pipeline = AugmentationPipeline([ResidualWater(), ArtificialPeaks(),
                                     PhaseShift(first_order_deg=45.0)])
    inserted = [step for index, step in pipeline.domain_plan() if index < 0]

    assert len(inserted) == 2, "expected one move in and one back"
    assert inserted[0].target.spectral == 'frequency'
    assert inserted[-1].target.spectral == 'time'


def test_a_time_domain_module_is_brought_back():
    """
    The bug this whole thing exists to prevent.

    Line broadening multiplies the FID by a decay. Left in the frequency domain
    after a spectral module it would apply that decay to a spectrum, which is a
    different operation entirely and would not raise.
    """
    pipeline = AugmentationPipeline([ResidualWater(), LineBroadening(lb_hz=2.0)])

    ran_in, current = [], 'time'
    for index, step in pipeline.domain_plan():
        if index < 0:
            current = step.target.spectral or current
        else:
            ran_in.append((type(step).__name__, current))

    assert ('ResidualWater', 'frequency') in ran_in
    assert ('LineBroadening', 'time') in ran_in


def test_the_plan_leaves_the_data_where_it_can_be_written(batch):
    """A pipeline ends in the canonical form unless told otherwise."""
    out, _ = AugmentationPipeline([ResidualWater()])(batch)

    assert (out.state.spectral, out.state.spatial) == ('time', 'image')


def test_the_end_domain_can_be_overridden(batch):
    """Staying in k-space is a legitimate thing to ask for."""
    pipeline = AugmentationPipeline([Noise(snr=30, seed=0)],
                                    end_domain=Domain(spatial='kspace'))
    out, _ = pipeline(batch)

    assert out.state.spatial == 'kspace'


def test_a_strict_module_refuses_rather_than_guessing():
    """
    Some modules define the domain they work in.

    Being handed another is a mistake in the pipeline, and saying so is more
    use than a transform the caller never asked for.
    """
    class OnlyInKspace(Noise):
        DOMAIN = Domain(spatial='kspace')
        STRICT = True

    pipeline = AugmentationPipeline([OnlyInKspace(snr=30)])

    with pytest.raises(DomainError, match="Add a DomainTransform"):
        pipeline.domain_plan()


#*********************#
#   strict planning   #
#*********************#
def test_strict_planning_raises_instead_of_inserting():
    """A user who places every transform themselves wants mistakes surfaced."""
    pipeline = AugmentationPipeline([ResidualWater()], domain_planning='strict')

    with pytest.raises(DomainError, match="Add a DomainTransform"):
        pipeline.domain_plan()


def test_strict_planning_accepts_a_correctly_placed_chain():
    """Hand-placed transforms count; a correct chain gets nothing added."""
    pipeline = AugmentationPipeline(
        [DomainTransform(spectral='frequency'), ResidualWater(),
         DomainTransform(spectral='time')],
        domain_planning='strict')

    planned = pipeline.domain_plan()
    assert all(index >= 0 for index, _ in planned), "strict must insert nothing"


#**********************************#
#   parameters decide the domain   #
#**********************************#
def test_an_injected_first_order_phase_gets_its_spectrum(batch):
    """
    Range-sampled parameters must reach the plan.

    The pipeline builds PhaseShift with defaults and injects the sampled
    first-order value per batch — the domain has to follow the value that
    actually runs, or the ramp lands on the FID and shifts it instead.
    """
    shift = PhaseShift()
    assert shift.DOMAIN is None

    out_injected, _ = AugmentationPipeline([PhaseShift()])(
        batch, batch_params={0: {'first_order_deg': 45.0}})

    reference = PhaseShift(first_order_deg=45.0)
    assert reference.DOMAIN == Domain(spectral='frequency')
    out_direct, _ = AugmentationPipeline([reference])(batch)

    np.testing.assert_allclose(_values(out_injected), _values(out_direct),
                               rtol=1e-5, atol=1e-6)


#****************************#
#   k-space native modules   #
#****************************#
def test_undersampling_declares_the_domain_it_masks_in():
    """The mask modes consume k-space; only the NUFFT measures off the image."""
    from augmentrum.sampling import KspaceUndersampling
    from augmentrum.sampling.coil_sampling import CoilSampler

    assert KspaceUndersampling(ksp_mode='cartesian').DOMAIN == Domain(spatial='kspace')
    assert KspaceUndersampling(ksp_mode='gridded').DOMAIN == Domain(spatial='kspace')
    assert KspaceUndersampling(ksp_mode='nufft').DOMAIN == Domain(spatial='image')
    assert KspaceUndersampling(ksp_mode='off').DOMAIN is None
    assert CoilSampler(mode='synthesize', n_coils=2).DOMAIN == Domain(spatial='image')


def test_undersampling_masks_without_transforming(batch):
    """
    Zeroing bins is a multiply on data already in k-space.

    Called on k-space directly, the module must touch nothing but the masked
    bins — no FFT round-trip of its own; and through the module call, the
    result must equal masking the centered k-space the DomainTransform
    convention produces.
    """
    from augmentrum.sampling import KspaceUndersampling

    us = KspaceUndersampling(ksp_mode='cartesian', acceleration_factor=2.0,
                             us_seed=0)
    out, _ = us(batch)

    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(
        _values(batch), axes=(1, 2, 3)), axes=(1, 2, 3), norm='ortho'), axes=(1, 2, 3))
    expected = k * us.last_masks_[..., None]
    expected = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(
        expected, axes=(1, 2, 3)), axes=(1, 2, 3), norm='ortho'), axes=(1, 2, 3))

    np.testing.assert_allclose(_values(out), expected, rtol=1e-4, atol=1e-5)


#*******************************#
#   per-trace noise statistic   #
#*******************************#
def test_noise_statistic_runs_along_the_spectrum_not_across_coils():
    """
    With a trailing coil axis, per-trace SNR must still be measured along T.

    Two coils with wildly different amplitudes: each coil's noise must track
    its own peak, which only happens if the statistic reduces the spectral
    axis (axis 4), not whatever axis happens to be last.
    """
    rng = np.random.default_rng(1)
    weak = rng.standard_normal((1, 1, 1, 1, 256)) + 1j * rng.standard_normal((1, 1, 1, 1, 256))
    data = np.stack([weak, 1000.0 * weak], axis=-1)             # (1, 1, 1, 1, T, C)

    out, _ = Noise(sigma_frac=0.01, seed=0).process_tensor(data)
    added = np.asarray(out) - data

    ratio = added[..., 1].std() / added[..., 0].std()
    assert 100 < ratio < 10000, f"noise must scale per coil trace, got ratio {ratio:.1f}"
