####################################################################################################
#                                      test_data_state.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds every module to carrying the data state forward, so that a later module can ask   #
#          where it is and what has already been done rather than assume.                          #
#                                                                                                  #
####################################################################################################

"""
Tests that state reaches modules, and that modules keep it honest.

A module that quietly drops the state is worse than one that never had it: the
next module would read a stale answer and act on it. So the guard is that every
registered module signs its name, and that the one operation which genuinely
changes what is legitimate downstream - undersampling - says so.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from nifti_mrs_plus.core import DataState

from augmentrum.core import Backend, NIfTI_MRS_Plus
from augmentrum.augmentation import Noise
from augmentrum.sampling import KspaceUndersampling

from tests.module_specs import SPECS


#**************#
#   fixtures   #
#**************#
@pytest.fixture
def volume():
    """One MRSI volume, as a batch of one."""
    def build(volatile=False):
        vol = np.ones((8, 8, 2, 16), np.complex64)
        return NIfTI_MRS_Plus(nifti_list=[gen_nifti_mrs(vol, 1 / 2000.0, 123.0)],
                              backend=Backend.NUMPY, volatile=volatile)
    return build


#*******************#
#   carried along   #
#*******************#
@pytest.mark.parametrize("volatile", [True, False])
def test_a_module_signs_its_name(volume, volatile):
    """Whoever ran last is recorded, on the fast path as well as the slow one."""
    out, _ = Noise(snr=30, seed=0)(volume(volatile))

    assert out.state.last == 'Noise'


@pytest.mark.parametrize("volatile", [True, False])
def test_undersampling_records_that_it_happened(volume, volatile):
    """
    The one fact that changes what is legitimate afterwards.

    A zero-filled reconstruction of undersampled k-space no longer has white
    noise, so anything added in the image domain after this point is not being
    added to what a scanner would have measured. Later modules need to be able
    to notice.
    """
    out, _ = KspaceUndersampling(ksp_mode='cartesian', acceleration_factor=2.0,
                                 us_seed=0)(volume(volatile))

    assert out.state.sampling == 'undersampled'


def test_undersampling_switched_off_changes_nothing(volume):
    """Claiming the data is undersampled when nothing was dropped would be a lie."""
    out, _ = KspaceUndersampling(ksp_mode='off')(volume())

    assert out.state.sampling == 'full'


def test_state_accumulates_across_a_pipeline(volume):
    """Each step updates what it changed and leaves the rest as it found it."""
    plus = volume()
    plus, _ = KspaceUndersampling(ksp_mode='cartesian', acceleration_factor=2.0,
                                  us_seed=0)(plus)
    plus, _ = Noise(snr=30, seed=0)(plus)

    assert plus.state.last == 'Noise', "the later module should sign it"
    assert plus.state.sampling == 'undersampled', "the earlier fact must survive"


#**************#
#   registry   #
#**************#
@pytest.mark.parametrize("spec", [s for s in SPECS if not s.spatial and not s.volume
                                  and not s.needs_multicoil], ids=lambda s: s.label)
def test_no_module_drops_the_state(spec, volume):
    """
    Every module must carry the state forward, not reset it.

    Swept from the registry so a new module cannot quietly opt out: a dropped
    state reads as "nothing has happened yet", which a later module would act on.
    """
    plus = volume()
    plus.set_state(plus.state.having(sampling='undersampled'))

    out, _ = spec.build()(plus)

    assert out.state.sampling == 'undersampled', f"{spec.label} reset the state"
    assert out.state.last, f"{spec.label} did not record itself"
