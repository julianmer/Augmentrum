"""
Mandatory adjoint-correctness tests for LinearMRIOperator, per
AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md section 5: adjoint/inner-product test (5.1, run for nominal
and every individual physics term plus the full operator), linearity test (5.2), and
normal-operator test (5.3). Deliberately independent of GIRF-Sim/RepoAdapter (synthetic small
trajectory + synthetic phase terms) so these stay fast, deterministic, and decoupled from the
full simulation stack - they test the OPERATOR's linear algebra, not any specific physics model.
"""
import torch
import pytest

from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT
from augmentrum.reconstruction.linear_operator import LinearMRIOperator, PhysicsConfig


IM_SIZE = (16, 16)
N = IM_SIZE[0] * IM_SIZE[1]
T = 300


def _vdot(a, b):
    return torch.vdot(a.reshape(-1), b.reshape(-1))


def _make_positions_and_trajectory(seed=0):
    g = torch.Generator().manual_seed(seed)
    ax = torch.linspace(-0.5, 0.5, IM_SIZE[0])
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    positions = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(N)], dim=-1)

    # Random but bounded-in-[-0.5,0.5) non-Cartesian coords, roughly uniform coverage.
    coords = (torch.rand(T, 2, generator=g) - 0.5) * 0.98
    t = torch.linspace(0, 0.02, T)
    return positions, coords, t


class _LinearB0Phase:
    def __init__(self, scale=30.0, seed=1):
        g = torch.Generator().manual_seed(seed)
        self._scale = scale
        self._seed = seed
        self._freq = None

    def freq(self, positions):
        if self._freq is None:
            g = torch.Generator().manual_seed(self._seed)
            self._freq = self._scale * (torch.rand(positions.shape[0], generator=g) - 0.5) * 2
        return self._freq

    def phase(self, positions, t):
        f = self.freq(positions)
        return 2 * torch.pi * f.reshape(-1, 1) * t.reshape(1, -1)


class _QuadraticPhase:
    """A different functional form (position^2, not linear-in-t) - stands in for
    concomitant/GNL-style terms without depending on their real implementations."""

    def __init__(self, scale=5.0):
        self._scale = scale

    def phase(self, positions, t):
        r2 = (positions[:, 0] ** 2 + positions[:, 1] ** 2).reshape(-1, 1)
        return self._scale * r2 * (t.reshape(1, -1) ** 1.3)


def _build_operator(config: PhysicsConfig, with_b1plus=False, with_girf=False):
    positions, coords, t = _make_positions_and_trajectory()
    nufft = GriddingNUFFT(IM_SIZE, osf=2.0)

    k_actual = None
    if with_girf or config.girf:
        g = torch.Generator().manual_seed(7)
        k_actual = coords + 0.01 * (torch.rand(*coords.shape, generator=g) - 0.5)

    b1plus_map = None
    if with_b1plus or config.b1plus:
        g = torch.Generator().manual_seed(3)
        b1plus_map = (0.8 + 0.4 * torch.rand(N, generator=g)) * torch.exp(
            1j * 2 * torch.pi * torch.rand(N, generator=g))
        b1plus_map = b1plus_map.to(torch.complex64)

    b0_term = _LinearB0Phase() if config.b0 else None
    concomitant_term = _QuadraticPhase(scale=4.0) if config.concomitant else None
    gnl_term = _QuadraticPhase(scale=2.0) if config.gnl else None

    op = LinearMRIOperator(
        positions=positions, t=t, k_nominal=coords, k_actual=k_actual,
        gridding_nufft=nufft, im_size=IM_SIZE, b1plus_map=b1plus_map,
        b0_term=b0_term, concomitant_term=concomitant_term, gnl_term=gnl_term,
        config=config,
    ).prepare()
    return op


CONFIGS = {
    "nominal": PhysicsConfig(),
    "b0_only": PhysicsConfig(b0=True),
    "b1plus_only": PhysicsConfig(b1plus=True),
    "girf_only": PhysicsConfig(girf=True),
    "concomitant_only": PhysicsConfig(concomitant=True),
    "gnl_only": PhysicsConfig(gnl=True),
    "full": PhysicsConfig(b0=True, b1plus=True, girf=True, concomitant=True, gnl=True),
}


@pytest.mark.parametrize("name", list(CONFIGS.keys()))
def test_adjoint_inner_product(name):
    op = _build_operator(CONFIGS[name])
    g = torch.Generator().manual_seed(42)
    x = torch.randn(N, dtype=torch.complex64, generator=g)
    y = torch.randn(T, dtype=torch.complex64, generator=g)

    lhs = _vdot(op.forward(x), y)
    rhs = _vdot(x, op.adjoint(y))
    rel_error = (lhs - rhs).abs() / max(lhs.abs(), rhs.abs(), 1e-12)
    assert rel_error < 1e-4, f"[{name}] adjoint mismatch: lhs={lhs}, rhs={rhs}, rel_err={rel_error}"


def test_linearity():
    op = _build_operator(CONFIGS["full"])
    g = torch.Generator().manual_seed(11)
    x1 = torch.randn(N, dtype=torch.complex64, generator=g)
    x2 = torch.randn(N, dtype=torch.complex64, generator=g)
    a = torch.tensor(0.7 - 0.2j, dtype=torch.complex64)
    b = torch.tensor(-1.1 + 0.4j, dtype=torch.complex64)

    lhs = op.forward(a * x1 + b * x2)
    rhs = a * op.forward(x1) + b * op.forward(x2)
    rel_error = torch.linalg.vector_norm(lhs - rhs) / torch.linalg.vector_norm(rhs)
    assert rel_error < 1e-4, f"linearity violated: rel_err={rel_error}"


def test_normal_operator_hermitian_and_psd():
    op = _build_operator(CONFIGS["full"])
    g = torch.Generator().manual_seed(99)
    x1 = torch.randn(N, dtype=torch.complex64, generator=g)
    x2 = torch.randn(N, dtype=torch.complex64, generator=g)

    lhs = _vdot(op.normal(x1), x2)
    rhs = _vdot(x1, op.normal(x2))
    rel_error = (lhs - rhs).abs() / max(lhs.abs(), rhs.abs(), 1e-12)
    assert rel_error < 1e-4, f"A^H A not Hermitian: rel_err={rel_error}"

    quad = _vdot(x1, op.normal(x1))
    assert quad.real >= -1e-3 * quad.real.abs().clamp(min=1.0)
    assert abs(quad.imag) < 1e-3 * max(abs(quad.real), 1.0)
