####################################################################################################
#                                     test_gradient_flow.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-07                                                                              #
#                                                                                                  #
# Purpose: Guards the differentiable path: an augmentation must not sever the autograd graph       #
#          between its input and its output, so a pipeline can sit inside a training loop.         #
#                                                                                                  #
####################################################################################################

"""
Gradient-flow regression tests.

An augmentation must not sever the autograd graph between its input and its
output, or a pipeline cannot sit inside a training loop.

No module is exempt, including those that resize the spectrum or add a
dimension. Fitting the NIfTI objects to a new shape is deferred to
materialization, so the NumPy round-trip happens once at the end rather than
inside the operation that changed the shape.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend

from tests.module_specs import SPECS

torch = pytest.importorskip("torch")


#*******************#
#   spec selection  #
#*******************#
# Spatial / volume modules and the NIfTI-list-only samplers are out of scope
# here: they are covered by their own suites and are the subject of the separate
# torch-port work.
_EXCLUDED = {"NIfTI_RawProcessor", "CoilSampler[draw]", "AverageSampler"}

SPECTRAL_SPECS = [s for s in SPECS
                  if not (s.spatial or s.volume)
                  and s.label.split("[")[0] not in _EXCLUDED]


#**************#
#   fixtures   #
#**************#

# 512 points, chosen so the resizing specs genuinely resize: truncate and crop
# target 256 and pad targets 1024. At 256 the first two would be no-ops and the
# test would pass without exercising the rebuild path at all.
N_PTS = 512


@pytest.fixture
def seeded_batch():
    """Two single-voxel subjects, fixed values, as a PyTorch-backed batch."""
    rng = np.random.default_rng(0)
    objs = []
    for _ in range(2):
        d = (rng.standard_normal((1, 1, 1, N_PTS))
             + 1j * rng.standard_normal((1, 1, 1, N_PTS))).astype(np.complex64)
        objs.append(gen_nifti_mrs(d, 1 / 2000, 123.0))
    return NIfTI_MRS_Plus(nifti_list=objs, backend=Backend.PYTORCH, volatile=True)


def _grad_reaches_input(plus, module):
    """Run *module* and report whether a gradient gets back to a leaf scalar."""
    leaf = torch.tensor(2.0, requires_grad=True)
    seeded = plus.get_data(Backend.PYTORCH) * torch.complex(leaf, torch.zeros_like(leaf))
    plus.set_data(seeded, Backend.PYTORCH)

    out, _ = module(plus)
    result = out.get_data(Backend.PYTORCH)

    if not torch.is_tensor(result) or result.grad_fn is None:
        return False
    torch.abs(result).sum().backward()
    return leaf.grad is not None and float(leaf.grad.abs().sum()) != 0.0


#*******************************#
#   per-module gradient flow    #
#*******************************#

@pytest.mark.parametrize("spec", SPECTRAL_SPECS, ids=lambda s: s.label)
def test_gradient_survives_module(spec, seeded_batch):
    """Every module must pass gradients through, whether or not it resizes."""
    assert _grad_reaches_input(seeded_batch, spec.build()), (
        f"{spec.label} severed the autograd graph. Something in its "
        f"process_tensor path converted the data (to_numpy, .numpy(), or a "
        f"NumPy kernel applied to the data itself) instead of staying on the "
        f"tensor's own backend."
    )


#***********************************#
#   whole-pipeline gradient flow    #
#***********************************#

def test_gradient_survives_a_chained_pipeline(seeded_batch):
    """The point of the exercise: a whole pipeline stays differentiable."""
    from augmentrum.augmentation.line_broadening import LineBroadening
    from augmentrum.augmentation.phase_frequency import PhaseShift
    from augmentrum.augmentation.gaussian_noise import GaussianNoise
    from augmentrum.core.pipeline import AugmentationPipeline

    pipeline = AugmentationPipeline([
        LineBroadening(lb_hz=5.0, mode="lorentzian"),
        PhaseShift(zero_order_deg=30.0),
        GaussianNoise(sigma=0.01),
    ])

    leaf = torch.tensor(2.0, requires_grad=True)
    seeded = seeded_batch.get_data(Backend.PYTORCH) * torch.complex(
        leaf, torch.zeros_like(leaf))
    seeded_batch.set_data(seeded, Backend.PYTORCH)

    out, _ = pipeline(seeded_batch)
    result = out.get_data(Backend.PYTORCH)

    assert result.grad_fn is not None, "pipeline severed the autograd graph"
    torch.abs(result).sum().backward()
    assert leaf.grad is not None and float(leaf.grad.abs().sum()) != 0.0


def test_data_stays_on_its_backend(seeded_batch):
    """A torch batch must come back as torch, never converted behind the caller."""
    from augmentrum.augmentation.line_broadening import LineBroadening

    out, _ = LineBroadening(lb_hz=5.0, mode="lorentzian")(seeded_batch)
    assert torch.is_tensor(out.get_data(Backend.PYTORCH))
