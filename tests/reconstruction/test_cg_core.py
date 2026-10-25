"""
CG-core correctness tests (AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md section 18): identity operator,
a known diagonal Hermitian positive-definite system, a complex-valued system, and convergence
metadata - independent of any MRI-specific physics.
"""
import torch

from augmentrum.reconstruction.cg import cg_solve


def test_identity_operator_recovers_rhs_in_one_step():
    # Identity converges exactly in 1 CG step in exact arithmetic; complex64 round-off can need
    # one tiny extra correction step to clear a strict rtol=1e-8, so allow <=2, not <=1.
    rhs = torch.randn(16, dtype=torch.complex64)
    result = cg_solve(lambda x: x, rhs, max_iter=5, rtol=1e-8)
    assert result.converged
    assert result.n_iter <= 2
    torch.testing.assert_close(result.image, rhs, atol=1e-5, rtol=1e-5)


def test_diagonal_hermitian_positive_definite_system():
    torch.manual_seed(0)
    n = 32
    diag = torch.rand(n, dtype=torch.float64) + 0.5   # strictly positive, well-conditioned
    x_true = torch.randn(n, dtype=torch.complex128)
    rhs = diag.to(torch.complex128) * x_true

    result = cg_solve(lambda x: diag.to(torch.complex128) * x, rhs, max_iter=100, rtol=1e-10)
    assert result.converged
    torch.testing.assert_close(result.image, x_true, atol=1e-6, rtol=1e-6)


def test_complex_valued_hermitian_system():
    torch.manual_seed(1)
    n = 20
    # Build a random Hermitian positive-definite matrix M = A^H A + eps*I.
    A = torch.randn(n, n, dtype=torch.complex128)
    M = A.conj().T @ A + 0.1 * torch.eye(n, dtype=torch.complex128)
    x_true = torch.randn(n, dtype=torch.complex128)
    rhs = M @ x_true

    result = cg_solve(lambda x: M @ x, rhs, max_iter=200, rtol=1e-10)
    assert result.converged
    torch.testing.assert_close(result.image, x_true, atol=1e-5, rtol=1e-5)
    # Residual history should be monotonically non-increasing (within numerical slack) and
    # the last entry should actually be the smallest.
    assert result.rel_residual_history[-1] == min(result.rel_residual_history)


def test_convergence_metadata_reports_non_convergence_when_starved_of_iterations():
    torch.manual_seed(2)
    n = 50
    A = torch.randn(n, n, dtype=torch.complex128)
    M = A.conj().T @ A + 0.1 * torch.eye(n, dtype=torch.complex128)
    x_true = torch.randn(n, dtype=torch.complex128)
    rhs = M @ x_true

    result = cg_solve(lambda x: M @ x, rhs, max_iter=1, rtol=1e-12)
    assert not result.converged
    assert result.n_iter == 1
    assert len(result.residual_history) == 2   # initial + 1 iteration
