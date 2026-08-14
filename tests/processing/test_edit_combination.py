####################################################################################################
#                                  test_edit_combination.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-14                                                                              #
#                                                                                                  #
# Purpose: Holds the edit combiner to its contract: the two conditions collapse into one spectrum, #
#          the DIM_EDIT tag goes with the axis, and the sign convention is the user's to state.    #
#                                                                                                  #
####################################################################################################

"""
Tests for combining edited (ON/OFF) acquisitions.

The arithmetic is trivial; what is worth pinning is the bookkeeping — the
right axis found by tag, the tag removed with the axis, unedited data left
alone — and, on real Big GABA data, that the combination actually produces
an edited difference spectrum with GABA where GABA belongs.
"""

#*************#
#   imports   #
#*************#
import os
import numpy as np
import pytest

from augmentrum.processing import EditCombiner


TAGS = ['DIM_DYN', 'DIM_EDIT', None]


#*************#
#   helpers   #
#*************#
def _batch(n_dyn=8, n_pts=64):
    """Distinct conditions: edit 0 is ones, edit 1 is twos."""
    batch = np.ones((1, 1, 1, 1, n_pts, n_dyn, 2), np.complex64)
    batch[..., 1] = 2.0
    return batch


#*****************#
#   the algebra   #
#*****************#
def test_diff_and_sum_follow_the_fid_a_convention():
    """(a ± b) / 2, so the result keeps the scale of one condition."""
    batch = _batch()

    diff, _ = EditCombiner(mode='diff').process_tensor(batch, dim_tags=TAGS)
    added, _ = EditCombiner(mode='sum').process_tensor(batch, dim_tags=TAGS)

    assert diff.shape == batch.shape[:-1]
    assert np.allclose(np.asarray(diff), -0.5)
    assert np.allclose(np.asarray(added), 1.5)


def test_the_edit_axis_is_found_by_tag_not_by_position():
    """DIM_EDIT first instead of second must combine the other axis."""
    batch = np.ones((1, 1, 1, 1, 64, 2, 8), np.complex64)
    batch[..., 1, :] = 2.0

    diff, _ = EditCombiner(mode='diff').process_tensor(
        batch, dim_tags=['DIM_EDIT', 'DIM_DYN', None])

    assert diff.shape == (1, 1, 1, 1, 64, 8)
    assert np.allclose(np.asarray(diff), -0.5)


def test_unedited_data_passes_through():
    """The combiner must be safe to leave in a mixed pipeline."""
    batch = np.ones((1, 1, 1, 1, 64, 8), np.complex64)

    out, _ = EditCombiner().process_tensor(batch, dim_tags=['DIM_DYN', None, None])
    assert out is batch


def test_more_than_two_conditions_are_refused():
    batch = np.ones((1, 1, 1, 1, 64, 3), np.complex64)

    with pytest.raises(ValueError, match="exactly two"):
        EditCombiner().process_tensor(batch, dim_tags=['DIM_EDIT', None, None])


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be"):
        EditCombiner(mode='subtract')


def test_the_tag_leaves_with_the_axis():
    """Through the NIfTI route, the output must not claim a DIM_EDIT."""
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    from augmentrum.core import Backend, NIfTI_MRS_Plus

    nifti = gen_nifti_mrs(_batch()[0], dwelltime=1 / 2000.0, spec_freq=127.7,
                          nucleus='1H', dim_tags=['DIM_DYN', 'DIM_EDIT', None])
    plus = NIfTI_MRS_Plus(nifti_list=[nifti], backend=Backend.NUMPY, volatile=True)

    out, _ = EditCombiner(mode='sum')(plus)
    combined = out.list()[0]

    assert 'DIM_EDIT' not in (combined.dim_tags or [])
    assert combined.shape[-1] == 8


#*****************#
#   in vivo       #
#*****************#
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'BigGABA')


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_DIR, 'MEGA_PRESS')),
                    reason="Big GABA data not present")
def test_big_gaba_philips_combination_reveals_gaba():
    """
    The whole point, on real data.

    Philips stores the second condition phase-inverted, so mode='sum' is the
    edited difference there — and its spectrum must peak at the MEGA-PRESS
    landmarks (GABA+ ~3.0 ppm, Glx ~3.7 ppm) outside the water region.
    """
    from augmentrum.core import Backend, NIfTI_MRS_Plus
    from augmentrum.dataset.biggaba import BigGABAModule

    met = BigGABAModule(DATA_DIR, sites=("P5",), limit=1).load()[0][0]
    plus = NIfTI_MRS_Plus(nifti_list=[met], backend=Backend.NUMPY, volatile=True)

    out, _ = EditCombiner(mode='sum')(plus)
    combined = out.list()[0]
    fid = np.asarray(combined[:])[0, 0, 0].mean(-1)          # over dynamics

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid)))
    freq = np.fft.fftshift(np.fft.fftfreq(fid.size, d=float(combined.dwelltime)))
    ppm = 4.68 - freq / float(combined.spectrometer_frequency[0])
    away_from_water = (ppm > 0.5) & (ppm < 4.2)

    peaks = ppm[away_from_water][np.argsort(spectrum[away_from_water])[-3:]]
    landmark = any((2.85 < p < 3.15) or (3.55 < p < 3.85) for p in peaks)
    assert landmark, f"difference spectrum peaks at {np.round(np.sort(peaks), 2)} ppm"
