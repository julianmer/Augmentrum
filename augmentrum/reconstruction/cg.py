"""
A small, reusable, torch-native conjugate-gradient solver for Hermitian positive-(semi)definite
normal operators ("A^H A + lambda I"), per AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md section 7.

Deliberately unpreconditioned by default (handover: "First make unpreconditioned CG correct.
Add simple preconditioning only if necessary, and use the same strategy for all model
comparisons.") and deliberately not TV/wavelet/learned-prior regularized (handover section 3:
those would obscure whether improvement comes from a better encoding model, which is the whole
point of the experiment this solver serves).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


@dataclass
class CGResult:
    image: torch.Tensor
    residual_history: List[float] = field(default_factory=list)
    rel_residual_history: List[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


def _vdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex-safe inner product <a, b> = sum(conj(a) * b), flattened."""
    return torch.vdot(a.reshape(-1), b.reshape(-1))


def cg_solve(
    normal_op: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 30,
    rtol: float = 1e-6,
    atol: float = 0.0,
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    callback: Optional[Callable[[int, torch.Tensor, float], None]] = None,
) -> CGResult:
    """
    Solves "normal_op(x) = rhs" (intended use: normal_op = A^H A + lambda*I, rhs = A^H y) via
    (optionally preconditioned) conjugate gradients, for complex-valued tensors.

    "normal_op" must be Hermitian positive-(semi)definite for CG to be well-posed - the caller
    (LinearMRIOperator.normal) is responsible for that, not this function.

    Stops when "||r|| <= max(atol, rtol * ||rhs||)" (standard relative-residual criterion) or
    after "max_iter" iterations. "converged" reports which of those actually triggered.
    """
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    r = rhs - normal_op(x)
    z = r if preconditioner is None else preconditioner(r)
    p = z.clone()
    rz_old = _vdot(r, z)

    rhs_norm = torch.linalg.vector_norm(rhs.reshape(-1)).item()
    threshold = max(atol, rtol * rhs_norm)

    residual_history: List[float] = []
    rel_residual_history: List[float] = []
    converged = False
    n_iter = 0

    r_norm = torch.linalg.vector_norm(r.reshape(-1)).item()
    residual_history.append(r_norm)
    rel_residual_history.append(r_norm / rhs_norm if rhs_norm > 0 else r_norm)
    if callback is not None:
        callback(0, x, r_norm)
    if r_norm <= threshold:
        return CGResult(image=x, residual_history=residual_history,
                        rel_residual_history=rel_residual_history, n_iter=0, converged=True)

    for it in range(1, max_iter + 1):
        Ap = normal_op(p)
        pAp = _vdot(p, Ap)
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.linalg.vector_norm(r.reshape(-1)).item()
        residual_history.append(r_norm)
        rel_residual_history.append(r_norm / rhs_norm if rhs_norm > 0 else r_norm)
        n_iter = it
        if callback is not None:
            callback(it, x, r_norm)
        if r_norm <= threshold:
            converged = True
            break

        z = r if preconditioner is None else preconditioner(r)
        rz_new = _vdot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return CGResult(image=x, residual_history=residual_history,
                    rel_residual_history=rel_residual_history, n_iter=n_iter,
                    converged=converged)
