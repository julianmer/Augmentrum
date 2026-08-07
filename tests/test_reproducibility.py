####################################################################################################
#                                     test_reproducibility.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-07                                                                              #
#                                                                                                  #
# Purpose: Guards the two halves of seeded augmentation: a seeded run must reproduce exactly,      #
#          while still drawing a fresh perturbation for every batch.                               #
#                                                                                                  #
####################################################################################################

"""
Reproducibility tests for stochastic modules.

Seeded augmentation has to satisfy two things at once. A seeded run must
reproduce exactly, or an experiment cannot be repeated. It must also draw a
fresh perturbation for every batch, or on-the-fly training sees one fixed
pattern on every sample. These tests hold every stochastic module to both.

Which modules are stochastic is measured rather than listed, so a module that
gains or loses randomness is covered without anyone editing a list.
"""

#*************#
#   imports   #
#*************#
import inspect

import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend

from tests.module_specs import SPECS

N_PTS = 128

#***********************#
#   specs under test    #
#***********************#
# Only modules that actually accept a seed, and only the spectral ones: the
# spatial / volume modules and the NIfTI-only samplers have their own suites.
_EXCLUDED = {"NIfTI_RawProcessor", "CoilAverageSampler"}


def _takes_seed(spec) -> bool:
    try:
        return "seed" in inspect.signature(spec.cls.__init__).parameters
    except (TypeError, ValueError):
        return False


SEEDED_SPECS = [s for s in SPECS
                if not (s.spatial or s.volume)
                and s.label.split("[")[0] not in _EXCLUDED
                and _takes_seed(s)]


#***************#
#   helpers     #
#***************#

def _batch():
    """A fixed single-voxel batch, identical on every call."""
    rng = np.random.default_rng(0)
    d = (rng.standard_normal((1, 1, 1, N_PTS))
         + 1j * rng.standard_normal((1, 1, 1, N_PTS))).astype(np.complex64)
    return NIfTI_MRS_Plus([gen_nifti_mrs(d, 1 / 2000, 123.0)],
                          backend=Backend.NUMPY, volatile=True)


def _run(spec, seed):
    module = spec.cls(**{**spec.kwargs, "seed": seed})
    return module, lambda: np.asarray(module(_batch())[0].numpy())


def _is_stochastic(spec) -> bool:
    """Measured, not declared: does this module vary at all without a seed?"""
    _, draw = _run(spec, None)
    return not np.allclose(draw(), draw())


#***********************************#
#   a seeded run must reproduce     #
#***********************************#

@pytest.mark.parametrize("spec", SEEDED_SPECS, ids=lambda s: s.label)
def test_same_seed_reproduces(spec):
    """Two modules built with the same seed produce the same first batch."""
    _, draw_a = _run(spec, 1234)
    _, draw_b = _run(spec, 1234)
    assert np.allclose(draw_a(), draw_b()), (
        f"{spec.label}: seed=1234 did not reproduce. An experiment using this "
        f"module cannot be repeated."
    )


@pytest.mark.parametrize("spec", SEEDED_SPECS, ids=lambda s: s.label)
def test_seeded_batches_still_vary(spec):
    """A seeded module must still draw fresh values for every batch."""
    if not _is_stochastic(spec):
        pytest.skip(f"{spec.label} is deterministic by design")

    _, draw = _run(spec, 1234)
    assert not np.allclose(draw(), draw()), (
        f"{spec.label}: consecutive batches are identical under a seed. "
        f"On-the-fly training would see one fixed perturbation on every sample. "
        f"The module is probably rebuilding its generator per call instead of "
        f"holding a SeedGenerator."
    )


@pytest.mark.parametrize("spec", SEEDED_SPECS, ids=lambda s: s.label)
def test_different_seeds_differ(spec):
    """Different seeds must give different streams."""
    if not _is_stochastic(spec):
        pytest.skip(f"{spec.label} is deterministic by design")

    _, draw_a = _run(spec, 1)
    _, draw_b = _run(spec, 2)
    assert not np.allclose(draw_a(), draw_b()), (
        f"{spec.label}: seeds 1 and 2 gave the same result, so the seed is "
        f"not reaching the generator."
    )


def test_unseeded_runs_differ():
    """Without a seed, two runs of the same module must differ."""
    stochastic = [s for s in SEEDED_SPECS if _is_stochastic(s)]
    assert stochastic, "expected at least one stochastic module to exist"

    for spec in stochastic:
        _, draw_a = _run(spec, None)
        _, draw_b = _run(spec, None)
        assert not np.allclose(draw_a(), draw_b()), (
            f"{spec.label}: unseeded runs were identical"
        )
