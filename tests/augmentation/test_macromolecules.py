"""
Tests for the Macromolecules module and its MM sources.

Tests cover:
- Parametrized components landing at the stated ppm positions, measured on an
  asymmetric spectral axis (never a symmetric probe)
- The mm_scale contract relative to the spectrum's real maximum
- Supplied: array round-trip and the COWS .mat layout (when present locally)
- Every source being the spectrum of a causal signal
- Measured: closest-field / species selection against a fake collection
  (no network involved)
- Seeded reproducibility of the randomized sources
"""

from pathlib import Path

import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from augmentrum.core import Backend, NIfTI_MRS_Plus
from augmentrum.augmentation import (Macromolecules, Parametrized,
                                     SemiParametrized, Measured, Supplied)


N_PTS, SW_HZ, SF_MHZ = 1000, 2000.0, 127.7      # deliberately non-power-of-two

COWS_MM_MAT = Path(__file__).resolve().parents[2] / (
    'data/openneuro_ds006812/derivatives/mrs_mat/sub-01_mat/PARIETAL/'
    'sub-01_COWS7_MM.mat')


def ppm_axis(n=N_PTS, sw=SW_HZ, sf=SF_MHZ):
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sw))
    return 4.7 - freq / sf


#**************************************************************************************************#
#                                         parametrized                                             #
#**************************************************************************************************#
def test_parametrized_peaks_land_at_stated_ppms():
    components = ((0.92, 0.10, 1.0), (2.04, 0.10, 0.6), (3.75, 0.10, 0.4))
    ppm = ppm_axis()
    profile = Parametrized(components=components).profile(ppm, np.random.default_rng(0))

    resolution = abs(ppm[1] - ppm[0])
    for center, _, _ in components:
        window = np.abs(ppm - center) < 0.3
        peak_ppm = ppm[window][np.argmax(np.real(profile[window]))]
        assert abs(peak_ppm - center) <= 2 * resolution, \
            f"component at {center} ppm peaked at {peak_ppm:.3f}"


def test_parametrized_unit_normalization_and_jitter_bounds():
    ppm = ppm_axis()
    rng = np.random.default_rng(1)
    source = Parametrized(amp_jitter=0.3, ppm_jitter=0.02, fwhm_jitter=0.2)
    profile = source.profile(ppm, rng)
    assert np.isclose(np.max(np.abs(np.real(profile))), 1.0)


def test_mm_scale_contract():
    """The added MM peaks at mm_scale x the spectrum's real maximum."""
    rng = np.random.default_rng(2)
    t = np.arange(N_PTS) / SW_HZ
    fid = (np.exp(2j * np.pi * (-2.0 * SF_MHZ) * t) * np.exp(-t * 8 * np.pi))
    nifti = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64),
                          1 / SW_HZ, SF_MHZ)
    data = NIfTI_MRS_Plus(nifti_list=[nifti], backend=Backend.NUMPY, volatile=True)

    module = Macromolecules(mm_scale=0.2, seed=0)
    out, _ = module(data, None)

    to_spec = lambda x: np.fft.fftshift(np.fft.ifft(np.asarray(x).squeeze()), axes=-1)
    spec_in = to_spec(fid)
    added = to_spec(out.get_data(Backend.NUMPY)) - spec_in
    ratio = np.max(np.abs(np.real(added))) / np.max(np.abs(np.real(spec_in)))
    assert np.isclose(ratio, 0.2, rtol=1e-3)


#**************************************************************************************************#
#                                     supplied / measured                                          #
#**************************************************************************************************#
def test_supplied_array_roundtrip():
    ppm = ppm_axis()
    template = Parametrized().profile(ppm, np.random.default_rng(0))
    regridded = Supplied(spectrum=template, ppm=ppm).profile(ppm, np.random.default_rng(0))
    assert np.allclose(regridded, template, atol=1e-10)


def test_supplied_regrids_and_zeros_outside_coverage():
    """
    The absorption part is not extrapolated beyond the source's coverage.

    A real curve is the absorption part of a causal signal, so its dispersion
    part is its Hilbert transform (Kramers-Kronig), and that reaches beyond
    the coverage the way the dispersion of anything confined must.
    """
    from scipy.signal import hilbert

    ppm_src = np.linspace(0.5, 4.5, 400)
    spectrum = np.exp(-0.5 * ((ppm_src - 2.0) / 0.1) ** 2).astype(complex)
    profile = Supplied(spectrum=spectrum, ppm=ppm_src).profile(
        ppm_axis(), np.random.default_rng(0))
    ppm = ppm_axis()
    assert np.max(np.real(profile[np.abs(ppm - 2.0) < 0.05])) > 0.9
    assert np.allclose(np.real(profile[ppm > 4.6]), 0.0), "no extrapolation outside coverage"
    assert np.allclose(profile, hilbert(np.real(profile))), "dispersion from absorption"
    assert np.abs(np.imag(profile[ppm > 4.6])).max() > 1e-3, "which reaches beyond it"


@pytest.mark.skipif(not COWS_MM_MAT.exists(), reason="local COWS release not present")
def test_supplied_reads_cows_mat():
    profile = Supplied(path=str(COWS_MM_MAT)).profile(ppm_axis(), np.random.default_rng(0))
    assert profile.shape == (N_PTS,)
    assert np.isclose(np.max(np.abs(np.real(profile))), 1.0, atol=0.1)


def test_measured_selects_closest_field_and_species(tmp_path, monkeypatch):
    for name in ('3T_MM_human_STEAM_CMRR.fid', '7T_MM_human_STEAM_CMRR.fid',
                 '9.4T_MM_rat_STEAM_CMRR.fid'):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(Measured, '_database', lambda self: tmp_path)

    assert Measured()._select(123.2 / Measured.GAMMA_1H).name.startswith('3T')
    assert Measured()._select(7.0).name.startswith('7T')
    assert Measured(species='rat')._select(9.0).name.startswith('9.4T')
    with pytest.raises(FileNotFoundError):
        Measured(sequence='sLASER')._select(3.0)


#**************************************************************************************************#
#                                       reproducibility                                            #
#**************************************************************************************************#
@pytest.mark.parametrize("source_params", [
    {'amp_jitter': 0.3, 'ppm_jitter': 0.02},
    None,
])
def test_seeded_draws_reproduce_and_advance(source_params):
    kwargs = dict(source_params=source_params, seed=7) if source_params else dict(seed=7)
    ppm = ppm_axis()

    first = Macromolecules(**kwargs)
    second = Macromolecules(**kwargs)
    p1 = first.source.profile(ppm, first.rng.numpy_rng())
    p2 = second.source.profile(ppm, second.rng.numpy_rng())
    assert np.allclose(p1, p2), "same seed must reproduce the draw"

    if source_params:                                   # randomized source
        p3 = first.source.profile(ppm, first.rng.numpy_rng())
        assert not np.allclose(p1, p3), "the next draw must differ"


def test_semi_parametrized_broadens_and_reproduces():
    ppm = ppm_axis()
    narrow = Parametrized(components=((2.0, 0.05, 1.0),))
    source = SemiParametrized(base=narrow, broaden_ppm=(0.2, 0.2), amp_mod=0.0)

    sharp = narrow.profile(ppm, np.random.default_rng(0))
    broad = source.profile(ppm, np.random.default_rng(0))

    def fwhm(profile):
        real = np.real(profile)
        above = np.flatnonzero(real >= 0.5 * real.max())
        return abs(ppm[above[-1]] - ppm[above[0]])

    assert fwhm(broad) > 2 * fwhm(sharp)
    again = source.profile(ppm, np.random.default_rng(0))
    assert np.allclose(broad, again)


#**************************************************************************************************#
#                                        per-sample draws                                          #
#**************************************************************************************************#
def _batch_of(fid, n_subjects, backend=Backend.NUMPY):
    niftis = [gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), 1 / SW_HZ, SF_MHZ)
              for _ in range(n_subjects)]
    return NIfTI_MRS_Plus(niftis, backend=backend, volatile=True)


def _naa_fid():
    t = np.arange(N_PTS) / SW_HZ
    return np.exp(2j * np.pi * ((2.01 - 4.65) * SF_MHZ) * t) * np.exp(-t * 8 * np.pi)


def _added(module, data):
    before = np.asarray(data.get_data(Backend.NUMPY))
    after = np.asarray(module(data)[0].get_data(Backend.NUMPY))
    return (after - before)[:, 0, 0, 0, :]


def test_a_randomized_source_is_drawn_per_sample():
    added = _added(Macromolecules(mm_source='semi_parametrized', seed=0), _batch_of(_naa_fid(), 4))
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(added[i], added[j], atol=1e-6)


def test_a_fixed_source_is_shared_by_the_batch():
    assert not Parametrized().varies
    assert Parametrized(amp_jitter=0.1).varies
    assert SemiParametrized().varies
    added = _added(Macromolecules(seed=0), _batch_of(_naa_fid(), 3))
    assert np.allclose(added[0], added[1]) and np.allclose(added[0], added[2])


def test_mm_scale_is_drawn_per_sample_in_a_pipeline():
    from augmentrum.core.pipeline import AugmentationPipeline

    assert 'mm_scale' in Macromolecules.PER_SAMPLE_PARAMS
    data = _batch_of(_naa_fid(), 4)
    pipe = AugmentationPipeline([Macromolecules(seed=0)], user_kwargs={'mm_scale': (0.05, 0.3)})
    params = pipe.sample_batch_parameters(4)
    out, _ = pipe(data, None, batch_params=params)

    to_spec = lambda x: np.fft.fftshift(np.fft.ifft(x, axis=-1), axes=-1)
    before = to_spec(np.asarray(data.get_data(Backend.NUMPY)))[:, 0, 0, 0, :]
    added = to_spec(np.asarray(out.get_data(Backend.NUMPY)))[:, 0, 0, 0, :] - before
    ratio = np.max(np.abs(np.real(added)), axis=-1) / np.max(np.abs(np.real(before)), axis=-1)
    assert np.allclose(ratio, params[0]['mm_scale'], rtol=1e-3)
    assert len(np.unique(np.round(ratio, 6))) == 4


def test_components_land_on_the_fsl_axis():
    """A component at 2.04 ppm shows at 2.04 ppm on an FSL-MRS plot."""
    from fsl_mrs.core import MRS

    source = Parametrized(components=((2.04, 0.08, 1.0),))
    added = _added(Macromolecules(mm_source=source, mm_scale=0.5), _batch_of(_naa_fid(), 1))[0]
    mrs = MRS(FID=added, cf=SF_MHZ, bw=SW_HZ, nucleus='1H')
    axis, spectrum = mrs.getAxes(), mrs.get_spec()
    assert abs(axis[np.argmax(np.real(spectrum))] - 2.04) <= SW_HZ / (N_PTS - 1) / SF_MHZ


def test_seeded_numpy_and_torch_agree():
    pytest.importorskip('torch')
    first = _added(Macromolecules(mm_source='semi_parametrized', seed=9), _batch_of(_naa_fid(), 3))
    torch_out = _added(Macromolecules(mm_source='semi_parametrized', seed=9),
                       _batch_of(_naa_fid(), 3, backend=Backend.PYTORCH))
    other = _added(Macromolecules(mm_source='semi_parametrized', seed=10), _batch_of(_naa_fid(), 3))
    assert np.allclose(first, torch_out, atol=1e-5)
    assert not np.allclose(first, other, atol=1e-6)


def test_semi_parametrized_copes_with_the_nyquist_alias():
    """The FSL-referenced axis puts the Nyquist alias first; the bin width must not read it."""
    from augmentrum.processing.utils import ppm_axis

    ppm = ppm_axis(N_PTS, SW_HZ, SF_MHZ)
    source = SemiParametrized(base=Parametrized(components=((2.0, 0.05, 1.0),)),
                              broaden_ppm=(0.2, 0.2), amp_mod=0.0)
    profile = source.profile(ppm, np.random.default_rng(0))
    real = np.real(profile)
    above = np.flatnonzero(real >= 0.5 * real.max())
    assert 0.15 < abs(ppm[above[-1]] - ppm[above[0]]) < 0.3


#**************************************************************************************************#
#                                           causality                                              #
#**************************************************************************************************#
def _second_half(fid):
    """The fraction of the FID's energy in its second half: ~0 causal, 0.5 white."""
    fid = np.asarray(fid).ravel()
    return float(np.sum(np.abs(fid[fid.size // 2:]) ** 2) / np.sum(np.abs(fid) ** 2))


def _supplied_absorption():
    """A real MM template on its own axis: only its absorption part is given."""
    ppm = np.linspace(0.5, 4.5, 400)
    return Supplied(spectrum=sum(np.exp(-0.5 * ((ppm - c) / 0.08) ** 2)
                                 for c in (0.92, 2.04, 3.0)), ppm=ppm)


@pytest.mark.parametrize("make", [
    lambda: Macromolecules(mm_source='parametrized', mm_scale=0.3),
    lambda: Macromolecules(mm_source='parametrized', mm_scale=0.3,
                           source_params={'amp_jitter': 0.3, 'ppm_jitter': 0.02,
                                          'fwhm_jitter': 0.2}, seed=0),
    lambda: Macromolecules(mm_source='semi_parametrized', mm_scale=0.3, seed=0),
    lambda: Macromolecules(mm_source=_supplied_absorption(), mm_scale=0.3),
], ids=['parametrized', 'jittered', 'semi_parametrized', 'supplied_real'])
def test_what_is_added_is_causal(make):
    """
    MM signal is tissue signal: it starts at the first point of the FID and
    decays. A profile drawn on the axis as a real curve has a two-sided FID
    instead, half of it wrapped to the end of the acquisition.
    """
    from augmentrum.core.pipeline import AugmentationPipeline

    data = _batch_of(_naa_fid(), 1)
    pipe = AugmentationPipeline([make()])
    out, _ = pipe(data, None, batch_params=pipe.sample_batch_parameters(1))
    added = (np.asarray(out.get_data(Backend.NUMPY)) - np.asarray(data.get_data(Backend.NUMPY)))
    assert _second_half(added[0, 0, 0, 0]) <= 0.02


def test_a_component_is_a_gaussian_of_the_stated_fwhm():
    """The Gaussian damping of the FID gives a Gaussian line of FWHM_ppm in the spectrum."""
    ppm = ppm_axis()
    profile = Parametrized(components=((2.04, 0.12, 1.0),)).profile(ppm, np.random.default_rng(0))
    real = np.real(profile)
    above = np.flatnonzero(real >= 0.5)
    resolution = abs(ppm[1] - ppm[0])
    assert abs(abs(ppm[above[-1]] - ppm[above[0]]) + resolution - 0.12) <= resolution
    fid = np.fft.fft(np.fft.ifftshift(profile))
    assert _second_half(fid) < 1e-6
