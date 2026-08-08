####################################################################################################
#                                         test_nufft.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-07                                                                              #
#                                                                                                  #
# Purpose: Holds the backend-agnostic gridding NUFFT to the torchkbnufft reference, and to the     #
#          two properties that justify it: it runs natively on every backend and passes            #
#          gradients back to the samples.                                                          #
#                                                                                                  #
####################################################################################################

"""
Tests for the Kaiser-Bessel gridding NUFFT.

A gridding NUFFT is an approximation, so the reference comparison is on
correlation and relative error rather than exact equality. torchkbnufft is the
reference because it is the implementation this replaces.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs_plus import ops
from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchkbnufft  # noqa: F401
    TKBN_AVAILABLE = True
except ImportError:
    TKBN_AVAILABLE = False

N = 32


#**************#
#   fixtures   #
#**************#
def _radial(n_spokes=48, n_read=N):
    """Radial trajectory in [-0.5, 0.5), shaped [K, 2]."""
    angles = np.linspace(0, np.pi, n_spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, n_read, endpoint=False)
    kx = (radius[None, :] * np.cos(angles)[:, None]).ravel()
    ky = (radius[None, :] * np.sin(angles)[:, None]).ravel()
    return np.stack([kx, ky], axis=-1).astype(np.float32)


def _samples(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)[None]


#*******************#
#   Class TestGeometry   #
#*******************#
class TestGeometry:
    """Grid sizing and shapes."""

    def test_grid_is_oversampled(self):
        assert GriddingNUFFT((16, 16), osf=2.0).grid_size == (32, 32)
        assert GriddingNUFFT((16, 16, 8), osf=1.5).grid_size == (24, 24, 12)

    def test_rejects_unsupported_rank(self):
        with pytest.raises(ValueError, match="2-D and 3-D"):
            GriddingNUFFT((8, 8, 8, 8))

    def test_adjoint_returns_the_image_matrix(self):
        coords = _radial()
        out = GriddingNUFFT((N, N)).adjoint(_samples(len(coords)), coords)
        assert ops.shape(out) == (1, N, N)


#***********************************#
#   Class TestAgainstTorchKbNufft   #
#***********************************#
@pytest.mark.skipif(not (TORCH_AVAILABLE and TKBN_AVAILABLE),
                    reason="torchkbnufft not installed")
class TestAgainstTorchKbNufft:
    """The reference this implementation replaces."""

    def test_adjoint_matches_reference(self):
        import torchkbnufft as tkbn

        coords = _radial()
        kdata = _samples(len(coords))

        mine = np.asarray(GriddingNUFFT((N, N)).adjoint(kdata, coords))[0]

        # torchkbnufft takes the trajectory in radians per voxel
        ktraj = torch.from_numpy((2 * np.pi * coords.T).astype(np.float32))
        ref = tkbn.KbNufftAdjoint(im_size=(N, N))(
            torch.from_numpy(kdata)[None], ktraj[None]).numpy()[0, 0]

        def unit(a):
            return a / (np.abs(a).max() + 1e-30)

        a, b = unit(mine), unit(ref)
        correlation = np.abs(np.vdot(a.ravel(), b.ravel())) / (
            np.linalg.norm(a) * np.linalg.norm(b))

        assert correlation > 0.999, f"correlation with torchkbnufft is only {correlation:.6f}"
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 0.01


#*******************************#
#   Class TestBackendAgnostic   #
#*******************************#
class TestBackendAgnostic:
    """Why this exists: it must run on every backend and stay differentiable."""

    def test_runs_natively_on_every_backend(self):
        coords = _radial()
        kdata = _samples(len(coords))
        nufft = GriddingNUFFT((N, N))

        cases = [("numpy", lambda a: a, lambda v: isinstance(v, np.ndarray))]
        if TORCH_AVAILABLE:
            cases.append(("torch", torch.from_numpy, ops.is_torch))
        try:
            import jax.numpy as jnp
            cases.append(("jax", jnp.array, ops.is_jax))
        except ImportError:
            pass
        try:
            import tensorflow as tf
            cases.append(("tensorflow", tf.constant, ops.is_tf))
        except ImportError:
            pass

        reference = None
        for name, convert, is_native in cases:
            out = nufft.adjoint(convert(kdata), coords)
            assert is_native(out), f"{name}: result left its backend"

            result = ops.to_numpy(out)
            if reference is None:
                reference = result
            else:
                assert np.allclose(result, reference, atol=1e-5), (
                    f"{name} disagrees with the NumPy result"
                )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_gradient_reaches_the_samples(self):
        coords = _radial()
        kdata = torch.from_numpy(_samples(len(coords))).requires_grad_(True)

        torch.abs(GriddingNUFFT((N, N)).adjoint(kdata, coords)).sum().backward()

        assert kdata.grad is not None
        assert float(kdata.grad.abs().sum()) > 0


#***************************#
#   Class TestForwardPair   #
#***************************#
class TestForwardPair:
    """The forward operator, and its consistency with the adjoint."""

    def test_forward_returns_one_value_per_sample(self):
        coords = _radial()
        rng = np.random.default_rng(0)
        image = (rng.standard_normal((1, N, N))
                 + 1j * rng.standard_normal((1, N, N))).astype(np.complex64)

        out = GriddingNUFFT((N, N)).forward(image, coords)
        assert ops.shape(out) == (1, len(coords))

    def test_adjoint_property(self):
        """
        <Ax, y> == <x, A^H y>.

        The one test that catches a convention error in either direction:
        scaling, fftshift placement or index ordering that disagrees between
        forward and adjoint breaks this even when each looks reasonable alone.
        """
        coords = _radial(n_spokes=64)
        nufft = GriddingNUFFT((N, N))
        rng = np.random.default_rng(0)

        x = (rng.standard_normal((1, N, N))
             + 1j * rng.standard_normal((1, N, N))).astype(np.complex64)
        y = (rng.standard_normal((1, len(coords)))
             + 1j * rng.standard_normal((1, len(coords)))).astype(np.complex64)

        lhs = np.vdot(np.asarray(nufft.forward(x, coords)).ravel(), y.ravel())
        rhs = np.vdot(x.ravel(), np.asarray(nufft.adjoint(y, coords)).ravel())

        assert np.allclose(lhs, rhs, rtol=2e-3), (
            f"forward and adjoint disagree: {lhs} vs {rhs}"
        )

    def test_forward_is_native_on_every_backend(self):
        coords = _radial()
        rng = np.random.default_rng(0)
        image = (rng.standard_normal((1, N, N))
                 + 1j * rng.standard_normal((1, N, N))).astype(np.complex64)
        nufft = GriddingNUFFT((N, N))

        cases = [("numpy", lambda a: a, lambda v: isinstance(v, np.ndarray))]
        if TORCH_AVAILABLE:
            cases.append(("torch", torch.from_numpy, ops.is_torch))
        try:
            import jax.numpy as jnp
            cases.append(("jax", jnp.array, ops.is_jax))
        except ImportError:
            pass
        try:
            import tensorflow as tf
            cases.append(("tensorflow", tf.constant, ops.is_tf))
        except ImportError:
            pass

        reference = None
        for name, convert, is_native in cases:
            out = nufft.forward(convert(image), coords)
            assert is_native(out), f"{name}: result left its backend"
            result = ops.to_numpy(out)
            if reference is None:
                reference = result
            else:
                assert np.allclose(result, reference, atol=1e-4), f"{name} disagrees"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_forward_gradient_reaches_the_image(self):
        coords = _radial()
        rng = np.random.default_rng(0)
        image = torch.from_numpy(
            (rng.standard_normal((1, N, N))
             + 1j * rng.standard_normal((1, N, N))).astype(np.complex64)
        ).requires_grad_(True)

        torch.abs(GriddingNUFFT((N, N)).forward(image, coords)).sum().backward()

        assert image.grad is not None
        assert float(image.grad.abs().sum()) > 0
