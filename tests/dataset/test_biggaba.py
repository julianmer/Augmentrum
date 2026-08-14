####################################################################################################
#                                       test_biggaba.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-14                                                                              #
#                                                                                                  #
# Purpose: Holds the Big GABA loader to one contract across three vendors: every scan arrives as   #
#          NIfTI-MRS with a DIM_EDIT axis of length two, ready for edited augmentation work.       #
#                                                                                                  #
####################################################################################################

"""
Tests for loading Big GABA MEGA-PRESS data.

The repository mixes GE P-files, Siemens TWIX, and Philips SDAT, each hiding
the edit loop somewhere else. What is worth pinning is the contract the rest
of the package relies on: DIM_EDIT exists, has length two, and the spectral
axis arrives at the protocol's true 2000 Hz / 2048 points.
"""

#*************#
#   imports   #
#*************#
import os
import pytest

from augmentrum.dataset.biggaba import BigGABAModule


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'BigGABA')

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATA_DIR, 'MEGA_PRESS')),
    reason="Big GABA data not present")


#*************#
#   helpers   #
#*************#
def _one_scan(site):
    loader = BigGABAModule(DATA_DIR, sites=(site,), limit=1)
    data, water, names = loader.load()
    assert not loader.load_failures, loader.load_failures
    assert len(data) == 1, f"expected one scan from {site}, got {len(data)}"
    return data[0], water[0], names[0]


def _edit_axis(nifti):
    assert 'DIM_EDIT' in nifti.dim_tags, nifti.dim_tags
    return nifti.shape[4 + nifti.dim_tags.index('DIM_EDIT')]


#***************#
#   the three   #
#***************#
@pytest.mark.parametrize("site", ["G6", "P5", "S5"])
def test_every_vendor_arrives_with_an_edit_axis(site):
    """One contract across three raw formats."""
    met, ref, _ = _one_scan(site)

    assert _edit_axis(met) == 2
    assert met.shape[3] == 2048
    assert 1.0 / met.dwelltime == pytest.approx(2000.0)
    if ref is not None:
        assert 'DIM_EDIT' in ref.dim_tags


def test_philips_edit_axis_carries_the_edited_signature():
    """
    The reshape assumption, held to the data twice over.

    The Philips edit conditions interleave row by row, the second stored
    phase-inverted. Two independent fingerprints pin the axis: (1) the two
    conditions differ in *magnitude* far more than alternate dynamics do —
    a pure phase cycle could not do that; (2) after undoing the inversion,
    their coherent difference peaks at the MEGA-PRESS landmarks (GABA+ at
    ~3.0 ppm or Glx at ~3.7 ppm), not at some arbitrary frequency.
    """
    import numpy as np

    met, _, _ = _one_scan("P5")
    array = np.asarray(met[:])[0, 0, 0]                      # (T, dyn, edit)

    # (1) magnitude fingerprint
    spectra = np.abs(np.fft.fftshift(np.fft.fft(array, axis=0), axes=0))
    on_off = np.linalg.norm(spectra[..., 0].mean(-1) - spectra[..., 1].mean(-1))
    drift = np.linalg.norm(spectra[:, 0::2, :].mean((-2, -1))
                           - spectra[:, 1::2, :].mean((-2, -1)))
    assert on_off > 2.0 * drift, (on_off, drift)

    # (2) spectral fingerprint of the coherent difference
    aligned = array.copy()
    aligned[..., 1] *= -1.0                                  # undo the inversion
    diff = np.abs(np.fft.fftshift(np.fft.fft(
        aligned[..., 0].mean(-1) - aligned[..., 1].mean(-1))))

    sw_hz = 1.0 / float(met.dwelltime)
    freq = np.fft.fftshift(np.fft.fftfreq(array.shape[0], d=1.0 / sw_hz))
    ppm = 4.68 - freq / float(met.spectrometer_frequency[0])
    away_from_water = (ppm > 0.5) & (ppm < 4.2)

    peaks = ppm[away_from_water][np.argsort(diff[away_from_water])[-3:]]
    landmark = any((2.85 < p < 3.15) or (3.55 < p < 3.85) for p in peaks)
    assert landmark, f"difference peaks at {np.round(np.sort(peaks), 2)} ppm"


def test_the_te_filter_selects_the_other_acquisition():
    loader = BigGABAModule(DATA_DIR, te='80', sites=("P5",), limit=1)
    data, _, names = loader.load()

    if not data:
        pytest.skip("this site has no TE 80 acquisition")
    assert names[0].endswith("TE80")
