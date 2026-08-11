"""
Tests for named pipeline taps and the outputs spec.

Tests cover:
- Snapshot ownership: the two aliasing regressions found in design review
  (in-place list-path writes; stale restack through the shared nifti list)
- Domain alignment of taps that fire mid-chain
- The nested outputs structure across frameworks
- Fixed-mode iteration leaving the subject pool untouched
- Provenance frozen at the tap point
- Construction-time errors: duplicate tap names, unknown output tokens
"""

import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from augmentrum.core import Augmentrum, Backend, NIfTI_MRS_Plus, Tap
from augmentrum.core.pipeline import AugmentationPipeline
from augmentrum.augmentation import Noise, PhaseShift


def make_niftis(n=4, pts=256, seed=0):
    """Small single-voxel subjects."""
    rng = np.random.default_rng(seed)
    return [gen_nifti_mrs(
        (rng.standard_normal((1, 1, 1, pts))
         + 1j * rng.standard_normal((1, 1, 1, pts))).astype(np.complex64),
        1 / 2000, 123.2) for _ in range(n)]


def make_volumes(n=2, seed=0):
    """Small volumes with real spatial extent, for k-space modules."""
    rng = np.random.default_rng(seed)
    return [gen_nifti_mrs(
        (rng.standard_normal((8, 8, 4, 64))
         + 1j * rng.standard_normal((8, 8, 4, 64))).astype(np.complex64),
        1 / 2000, 123.2) for _ in range(n)]


#**************************************************************************************************#
#                                     snapshot ownership                                           #
#**************************************************************************************************#
def test_tap_owns_snapshot_on_nifti_list_backend():
    """List-path modules write in place; the tap must not follow them."""
    aug = Augmentrum(data=make_niftis(), pipeline=['noise', 'tap:t', 'broadening'],
                     sigma_frac=0.001, lb_hz=8.0, outputs=('data', 't'),
                     batch_size=2, backend='nifti_list', volatile=False)
    x, y = next(aug.dataloader(framework='numpy'))
    assert not np.allclose(x, y), "tap aliased the broadened output"


def test_tap_survives_materialization_on_pytorch():
    """A tap without its own tensor would restack from the shared nifti list."""
    aug = Augmentrum(data=make_niftis(), pipeline=['tap:pre', 'noise'],
                     sigma_frac=0.3, outputs=('data', 'pre'),
                     batch_size=2, backend='pytorch', volatile=True)
    import torch
    x, y = next(aug.dataloader(framework='pytorch'))
    assert not torch.allclose(x, y), "tap re-materialized as the noisy output"


def test_post_tap_range_changes_output_not_tap():
    aug = Augmentrum(data=make_niftis(), pipeline=['noise', 'tap:clean', 'phase'],
                     sigma_frac=0.02, zero_order_deg=(30.0, 60.0),
                     outputs=('data', 'clean'),
                     batch_size=2, backend='numpy', volatile=True)
    x, y = next(aug.dataloader(framework='numpy'))
    assert not np.allclose(x, y)
    # a pure post-tap phase leaves magnitudes identical
    assert np.allclose(np.abs(x), np.abs(y), atol=1e-5)
    angle = np.rad2deg(np.abs(np.angle(np.sum(x * np.conj(y)))))
    assert 30.0 - 1.0 <= angle <= 60.0 + 1.0


#**************************************************************************************************#
#                                      domain alignment                                            #
#**************************************************************************************************#
def test_tap_before_undersampling_is_fully_sampled():
    """The tap fires before the k-space move; the pair must still align."""
    aug = Augmentrum(data=make_volumes(), pipeline=['noise', 'tap:full', 'undersampling'],
                     sigma_frac=0.001, acceleration_factor=3.0, us_seed=0,
                     outputs=('data', 'full'),
                     batch_size=2, backend='numpy', volatile=True)
    x, y = next(aug.dataloader(framework='numpy'))
    assert x.shape == y.shape

    def zero_fraction(vol):
        # Relative threshold: the pipeline's own FFT round trip leaves masked
        # bins at float32 epsilon rather than exactly zero.
        k = np.fft.fftshift(np.fft.fftn(vol, axes=(1, 2)), axes=(1, 2))
        return float((np.abs(k) < 1e-5 * np.abs(k).max()).mean())

    assert zero_fraction(x) > 0.3, "input should carry exact k-space zeros"
    assert zero_fraction(y) < 0.01, "tap should be fully sampled"


def test_tap_after_kspace_step_is_end_aligned():
    """A tap firing while the data sits in k-space is brought back at the end."""
    aug = Augmentrum(data=make_volumes(), pipeline=['undersampling', 'tap:t'],
                     acceleration_factor=2.0, us_seed=0, outputs=('data', 't'),
                     batch_size=2, backend='numpy', volatile=True)
    x, y = next(aug.dataloader(framework='numpy'))
    assert x.shape == y.shape
    assert np.allclose(x, y, atol=1e-4), \
        "identity-after-tap must give an identical, end-aligned pair"


#**************************************************************************************************#
#                                       outputs structure                                          #
#**************************************************************************************************#
def test_nested_outputs_structure_and_python_framework():
    outputs = (('data', 'water'), ('a', 'b'))
    aug = Augmentrum(data=make_niftis(), pipeline=['tap:a', 'noise', 'tap:b'],
                     sigma_frac=0.05, outputs=outputs,
                     batch_size=2, backend='numpy', volatile=True)
    (x, xw), (a, b) = next(aug.dataloader(framework='numpy'))
    assert x.shape == a.shape == b.shape
    assert xw is None
    assert np.allclose(x, b), "tap after last module equals the output"
    assert not np.allclose(a, b), "taps around noise must differ"

    lists = next(aug.dataloader(framework='python'))
    (x_l, _), (a_l, _) = lists
    assert isinstance(x_l, list) and isinstance(a_l, list)
    assert x_l[0] is not a_l[0], "python framework must yield distinct objects"


def test_as_torch_dataloader_yields_pairs():
    import torch
    aug = Augmentrum(data=make_niftis(), pipeline=['noise', 'tap:clean', 'phase'],
                     sigma_frac=0.02, zero_order_deg=45.0,
                     outputs=(('data', 'water'), ('clean', 'clean.water')),
                     batch_size=2, backend='pytorch', volatile=True)
    (x, xw), (y, yw) = next(iter(aug.as_torch_dataloader('train')))
    assert x.is_complex() and y.is_complex()
    assert x.shape == y.shape
    assert xw is None and yw is None


def test_pipeline_without_taps_keeps_pair_contract():
    aug = Augmentrum(data=make_niftis(), pipeline=['noise'], sigma_frac=0.02,
                     batch_size=2, backend='numpy', volatile=True)
    x, water = next(aug.dataloader(framework='numpy'))
    assert x.shape[0] == 2 and water is None


#**************************************************************************************************#
#                                    fixed mode / provenance                                       #
#**************************************************************************************************#
def test_fixed_mode_leaves_pool_untouched():
    niftis = make_niftis()
    before = [np.asarray(n[:]).copy() for n in niftis]
    aug = Augmentrum(data=niftis, pipeline=['noise', 'tap:t', 'broadening'],
                     sigma_frac=0.05, lb_hz=6.0, outputs=('data', 't'),
                     mode='fixed', batch_size=2, backend='nifti_list', volatile=False)
    for _ in range(2):                                 # two epochs
        for _ in aug.dataloader(framework='numpy', shuffle=False):
            pass
    after = [np.asarray(n[:]) for n in niftis]
    for b, a in zip(before, after):
        assert np.allclose(b, a), "iteration polluted the subject pool"


def test_provenance_frozen_at_tap():
    pipe = AugmentationPipeline([Noise(sigma_frac=0.02), Tap(name='t'),
                                 PhaseShift(zero_order_deg=30.0)])
    data = NIfTI_MRS_Plus(nifti_list=make_niftis(2), backend=Backend.NIFTI_LIST,
                          volatile=False)
    out, _, taps = pipe(data, None)

    def provenance(plus):
        hdr = plus.list()[0].hdr_ext
        return str(hdr.to_dict() if hasattr(hdr, 'to_dict') else hdr)

    tap_prov = provenance(taps['t'][0])
    out_prov = provenance(out)
    assert 'Noise' in tap_prov and 'Tap' in tap_prov
    assert 'PhaseShift' not in tap_prov, "tap provenance leaked a post-tap step"
    assert 'PhaseShift' in out_prov


#**************************************************************************************************#
#                                    construction-time errors                                      #
#**************************************************************************************************#
def test_duplicate_tap_name_raises():
    with pytest.raises(ValueError, match="Duplicate tap"):
        Augmentrum(data=make_niftis(), pipeline=['tap:a', 'noise', 'tap:a'],
                   sigma_frac=0.02, backend='numpy', volatile=True)


def test_unknown_output_token_raises():
    with pytest.raises(ValueError, match="unknown stage"):
        Augmentrum(data=make_niftis(), pipeline=['noise', 'tap:a'],
                   sigma_frac=0.02, outputs=('data', 'nonexistent'),
                   backend='numpy', volatile=True)


def test_direct_call_returns_triple_only_with_taps():
    tapped = AugmentationPipeline([Noise(sigma_frac=0.02), Tap(name='t')])
    plain = AugmentationPipeline([Noise(sigma_frac=0.02)])
    data = NIfTI_MRS_Plus(nifti_list=make_niftis(2), backend=Backend.NUMPY,
                          volatile=True)
    assert len(tapped(data, None)) == 3
    assert len(plain(data.copy(), None)) == 2
