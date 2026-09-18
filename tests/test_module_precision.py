"""
Every registered module in single and in double precision.

A module computes in its working precision ("precision"; None follows the data),
so single-precision data come out in single precision and double-precision data
in double, unless the module is told otherwise. Swept from the registry, so a
module that fixes a dtype of its own cannot hide.
"""

import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend

from augmentrum.core import precision as prec
from tests.module_specs import SPECS

try:
    import torch                                                   # noqa: F401
    BACKENDS = [Backend.NUMPY, Backend.PYTORCH]
except ImportError:                                                # pragma: no cover
    BACKENDS = [Backend.NUMPY]

N_PTS = 128

CASES = [(np.complex64, None, 'single'), (np.complex128, None, 'double'),
         (np.complex64, 'double', 'double'), (np.complex128, 'single', 'single')]


def _kind(spec):
    if spec.spatial or spec.volume:
        return 'volume'
    if spec.needs_multicoil:
        return 'multicoil'
    return 'coiled' if spec.coiled else 'spectral'


def _batch(kind, dtype, backend, n=2):
    """*n* scans of the layout *kind* needs, stored in *dtype*."""
    rng = np.random.default_rng(0)
    shape, tags = {'spectral': ((1, 1, 1, N_PTS), ()),
                   'coiled': ((1, 1, 1, N_PTS, 4), ('DIM_COIL',)),
                   'multicoil': ((1, 1, 1, N_PTS, 4, 4), ('DIM_COIL', 'DIM_DYN')),
                   'volume': ((8, 8, 2, 16), ())}[kind]
    niftis = []
    for _ in range(1 if kind == 'volume' else n):
        data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
        nifti = gen_nifti_mrs(data, 1 / 2000, 123.0)
        for axis, tag in enumerate(tags, start=4):
            nifti.set_dim_tag(axis, tag)
        niftis.append(nifti)
    return NIfTI_MRS_Plus(nifti_list=niftis, backend=backend, volatile=True)


def _module(spec, precision):
    kwargs = {**spec.kwargs, **(spec.nifti_kwargs if spec.spatial else {})}
    return spec.cls(**kwargs, precision=precision)


@pytest.mark.parametrize('stored, precision, expected', CASES,
                         ids=lambda v: getattr(v, '__name__', str(v)))
@pytest.mark.parametrize('backend', BACKENDS, ids=lambda b: b.value)
@pytest.mark.parametrize('spec', SPECS, ids=lambda s: s.label)
def test_a_module_works_in_its_precision(spec, backend, stored, precision, expected):
    data = _batch(_kind(spec), stored, backend)
    out, _ = _module(spec, precision)(data)
    result = out.get_data(backend)
    assert prec.of(result) == expected, (
        f"{spec.label} on {backend.value}: {np.dtype(stored).name} data with precision="
        f"{precision!r} came out as {getattr(result, 'dtype', None)}, not {expected}. "
        f"Something in it fixes a dtype instead of following its input.")
