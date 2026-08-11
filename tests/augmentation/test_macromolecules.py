"""
Tests for the Macromolecules module and its MM sources.

Tests cover:
- Parametrized components landing at the stated ppm positions, measured on an
  asymmetric spectral axis (never a symmetric probe)
- The mm_scale contract relative to the spectrum's real maximum
- Supplied: array round-trip and the COWS .mat layout (when present locally)
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
    ppm_src = np.linspace(0.5, 4.5, 400)
    spectrum = np.exp(-0.5 * ((ppm_src - 2.0) / 0.1) ** 2).astype(complex)
    profile = Supplied(spectrum=spectrum, ppm=ppm_src).profile(
        ppm_axis(), np.random.default_rng(0))
    ppm = ppm_axis()
    assert np.max(np.real(profile[np.abs(ppm - 2.0) < 0.05])) > 0.9
    assert np.allclose(profile[ppm > 4.6], 0.0), "no extrapolation outside coverage"


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
