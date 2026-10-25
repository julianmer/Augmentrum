"""
Small-phantom CG regression test, per AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md section 18: on a
small, noiseless, fully sampled grid, x -> A_full -> y -> CG(A_full) should recover x tightly,
and CG(A_nominal) on the SAME data should show materially larger error when physics effects are
intentionally nonzero - proving the earlier adjoint-correctness tests (test_linear_operator.py)
compose into an actual working reconstruction, not just a passing dot-product check.
"""
import torch

from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT
from augmentrum.reconstruction.linear_operator import LinearMRIOperator, PhysicsConfig
from augmentrum.reconstruction.cg import cg_solve

IM_SIZE = (32, 32)
N = IM_SIZE[0] * IM_SIZE[1]


def _smooth_test_image(n):
    """
    A smooth (band-limited-friendly) test image - deliberately NOT a sharp-edged Shepp-Logan
    phantom. A truly discontinuous phantom has Fourier energy beyond any finite k-space
    bandwidth, so recovering it from band-limited data carries an unavoidable ~15-20% Gibbs-
    phenomenon residual REGARDLESS of algorithm (confirmed empirically: an exact brute-force
    NUDFT + CG, with no gridding approximation at all, showed the identical ~18% floor as
    GriddingNUFFT on a sharp phantom at this matrix size) - that floor is real signal-processing
    physics, not something this test should be asserting against, since the point here is
    validating the OPERATOR/CG CODE, not Gibbs phenomena. A smooth image has no such floor, so
    near-exact recovery is the correct expected outcome and actually tests the code.
    """
    ax = torch.linspace(-1.0, 1.0, n)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    img = (1.0 * torch.exp(-((xx - 0.1) ** 2 + (yy + 0.1) ** 2) / 0.15)
          + 0.6 * torch.exp(-((xx + 0.3) ** 2 + (yy - 0.2) ** 2) / 0.08)
          + 0.4 * torch.exp(-((xx - 0.35) ** 2 + (yy - 0.35) ** 2) / 0.05))
    return img


def _fully_sampled_cartesian_like_coords(n, oversample=1.5):
    """A dense (not Cartesian, but dense/near-isotropic) non-Cartesian coordinate set - enough
    samples to fully constrain an n x n image, so CG(A_full) recovering x is a meaningful,
    non-trivial check rather than an underdetermined-system fluke."""
    n_samples = int(oversample * N)
    g = torch.Generator().manual_seed(123)
    r = torch.sqrt(torch.rand(n_samples, generator=g)) * 0.49
    theta = torch.rand(n_samples, generator=g) * 2 * torch.pi
    coords = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    t = torch.linspace(0, 0.02, n_samples)
    return coords, t


class _SmoothB0:
    def __init__(self, n):
        ax = torch.linspace(-1.0, 1.0, n)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        self._f = (40.0 * xx - 25.0 * yy).reshape(-1)

    def phase(self, positions, t):
        return 2 * torch.pi * self._f.reshape(-1, 1) * t.reshape(1, -1)


def _build_operator(config, positions, coords, t, b0_term):
    nufft = GriddingNUFFT(IM_SIZE, osf=2.0)
    return LinearMRIOperator(
        positions=positions, t=t, k_nominal=coords, gridding_nufft=nufft, im_size=IM_SIZE,
        b0_term=b0_term, config=config,
    ).prepare()


def test_cg_full_operator_recovers_phantom_tightly():
    x_true = _smooth_test_image(IM_SIZE[0]).to(torch.complex64).reshape(-1)
    ax = torch.linspace(-0.5, 0.5, IM_SIZE[0])
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    positions = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(N)], dim=-1)
    coords, t = _fully_sampled_cartesian_like_coords(IM_SIZE[0])
    b0_term = _SmoothB0(IM_SIZE[0])

    op_full = _build_operator(PhysicsConfig(b0=True, n_segments=10), positions, coords, t, b0_term)
    y = op_full.forward(x_true)

    rhs = op_full.adjoint(y)
    result = cg_solve(lambda v: op_full.normal(v, lam=1e-6), rhs, max_iter=60, rtol=1e-5)
    assert result.converged

    err = torch.linalg.vector_norm(result.image - x_true) / torch.linalg.vector_norm(x_true)
    assert err < 0.05, f"CG(A_full) did not tightly recover the phantom: rel_err={err}"


def test_cg_nominal_shows_larger_error_than_full_on_full_physics_data():
    x_true = _smooth_test_image(IM_SIZE[0]).to(torch.complex64).reshape(-1)
    ax = torch.linspace(-0.5, 0.5, IM_SIZE[0])
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    positions = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(N)], dim=-1)
    coords, t = _fully_sampled_cartesian_like_coords(IM_SIZE[0])
    b0_term = _SmoothB0(IM_SIZE[0])

    op_full = _build_operator(PhysicsConfig(b0=True, n_segments=10), positions, coords, t, b0_term)
    y_full_physics = op_full.forward(x_true)   # the one reference dataset, per handover section 11

    def recon_with(op):
        rhs = op.adjoint(y_full_physics)
        result = cg_solve(lambda v: op.normal(v, lam=1e-6), rhs, max_iter=60, rtol=1e-5)
        err = torch.linalg.vector_norm(result.image - x_true) / torch.linalg.vector_norm(x_true)
        return float(err)

    err_full = recon_with(op_full)
    op_nominal = _build_operator(PhysicsConfig(), positions, coords, t, b0_term)
    err_nominal = recon_with(op_nominal)

    assert err_nominal > err_full, (
        f"nominal reconstruction of full-physics data should be WORSE than the matched-physics "
        f"reconstruction: err_nominal={err_nominal}, err_full={err_full}"
    )
    # Not just numerically worse - the model-mismatch gap should be substantial, matching the
    # handover's own expected result (nominal-model image error plateaus at a higher level).
    assert err_nominal > 2 * err_full
