####################################################################################################
#                                       test_precision.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-18                                                                              #
#                                                                                                  #
# Purpose: The working precision every module takes: the helpers, that every registered module    #
#          accepts it, that None follows the data and 'single' / 'double' set it, and that a      #
#          pipeline-wide "precision" reaches every step that does not set its own.                 #
#                                                                                                  #
####################################################################################################

import numpy as np
import pytest
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from augmentrum import Augmentrum
from augmentrum.augmentation import PhaseShift
from augmentrum.core import precision as prec
from augmentrum.core.pipeline import constructor_params
from tests.module_specs import registered_classes

try:
    import torch
except ImportError:                       # pragma: no cover
    torch = None


def _niftis(dtype, n=3, n_pts=256, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        fid = (rng.standard_normal((1, 1, 1, n_pts))
               + 1j * rng.standard_normal((1, 1, 1, n_pts))).astype(dtype)
        out.append(gen_nifti_mrs(fid, 1 / 2000, 123.0))
    return out


#*************#
#   helpers   #
#*************#
def test_of_names_the_precision_of_floating_data_only():
    assert prec.of(np.zeros(2, np.complex64)) == 'single'
    assert prec.of(np.zeros(2, np.float64)) == 'double'
    assert prec.of(np.zeros(2, np.int64)) is None
    if torch is not None:
        assert prec.of(torch.zeros(2, dtype=torch.complex128)) == 'double'


def test_cast_keeps_real_and_complex_apart_and_leaves_the_rest():
    x = np.ones(3, np.complex64)
    assert prec.cast(x, 'double').dtype == np.complex128
    assert prec.cast(np.ones(3, np.float64), 'single').dtype == np.float32
    assert prec.cast(x, None) is x                   # None follows the data
    assert prec.cast(x, 'single') is x               # already there: the same object
    idx = np.arange(3)
    assert prec.cast(idx, 'double') is idx           # integers are not a precision


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError, match='precision'):
        prec.check('half')
    with pytest.raises(ValueError, match='precision'):
        PhaseShift(precision='half')


#*************#
#   modules   #
#*************#
@pytest.mark.parametrize('name, cls', sorted(registered_classes().items()))
def test_every_module_takes_a_precision(name, cls):
    """Accepted by the pipeline (its routing reads constructor_params) and recorded."""
    assert 'precision' in constructor_params(cls)


def test_a_module_records_its_precision_in_its_parameters():
    assert PhaseShift(precision='double').params['precision'] == 'double'
    assert PhaseShift().precision is None and 'precision' not in PhaseShift().params


BACKENDS = [Backend.NUMPY] + ([Backend.PYTORCH] if torch is not None else [])


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('stored, precision, expected', [
    (np.complex64, None, 'single'), (np.complex128, None, 'double'),
    (np.complex64, 'double', 'double'), (np.complex128, 'single', 'single'),
])
def test_none_follows_the_data_and_a_setting_decides(backend, stored, precision, expected):
    data = NIfTI_MRS_Plus(nifti_list=_niftis(stored), backend=backend)
    out, _ = PhaseShift(zero_order_deg=10.0, first_order_deg=0.0, precision=precision)(data)
    assert prec.of(out.get_data(backend)) == expected


#**************#
#   pipeline   #
#**************#
def test_a_pipeline_wide_precision_reaches_every_step_without_its_own():
    aug = Augmentrum(data=_niftis(np.complex64), backend='numpy',
                     pipelines={'train': ['phase_shift', {'noise': {'precision': 'single'}}]},
                     precision='double')
    steps = aug.pipelines['train'].steps
    assert [s.precision for s in steps] == ['double', 'single']
