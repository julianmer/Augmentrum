"""
A composable linear MRI forward/adjoint operator built from the SAME physics primitives as the
spatial-physics ablation figure (scripts/spatial_physics_ablation.py), reused rather than
reimplemented, per AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md sections 1 and 6.

Toggleable physics (PhysicsConfig) compose as:

    x(r) --[B1+]--[coil]--> per-segment masked object --[fast NUFFT]--> per-segment k-space
        --[x exp(i*phi_segment(t))]--> summed over segments --> d(k,t)

B0/concomitant/GNL phase terms are combined and segmented TOGETHER as one phase field (see
_SumPhaseTerms), not segmented independently and multiplied afterward: exp(i*phiA)*exp(i*phiB) =
exp(i*(phiA+phiB)) is a property of the SUM, not something a per-term segmentation product would
correctly reproduce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from .segmented_phase import SegmentedPhaseTerm


@dataclass
class PhysicsConfig:
    b0: bool = False
    b1plus: bool = False
    girf: bool = False          # selects k_actual vs k_nominal for the NUFFT trajectory
    concomitant: bool = False
    gnl: bool = False
    n_segments: int = 8


class _SumPhaseTerms:
    """Combines several PhaseTerm-like objects (".phase(positions, t) -> (N,T)") into one by
    SUMMING their phase outputs - the correct way to jointly segment several simultaneously
    active phase terms (see module docstring)."""

    def __init__(self, terms: Sequence[object]):
        self.terms = list(terms)

    def phase(self, positions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        total: Optional[torch.Tensor] = None
        for term in self.terms:
            p = term.phase(positions, t)
            total = p if total is None else total + p
        if total is None:
            return torch.zeros(positions.shape[0], t.numel(), device=t.device)
        return total


class LinearMRIOperator:
    """
    Reconstruction-facing linear operator: ".forward"/".adjoint"/".normal", per
    AUGMENTRUM_PLUS_CG_RECON_HANDOVER.md section 4.

    Single-coil for this first implementation (coil_map defaults to all-ones) but kept
    multi-coil-shaped (coil_map is a per-voxel complex array like b1plus_map, not hardcoded away)
    so adding real coil maps later does not change this class's structure.

    Density compensation is NEVER applied inside forward/adjoint here (handover section 8) - if
    useful for CG initialization/preconditioning, apply it explicitly outside this class.
    """

    def __init__(
        self,
        positions: torch.Tensor,            # (N, 3), real
        t: torch.Tensor,                    # (T,), real - absolute sample times
        k_nominal: torch.Tensor,            # (2, T) or (3, T) cycles/m
        gridding_nufft,                     # augmentrum.sampling.kspace_reconstructor.GriddingNUFFT
        im_size: tuple,
        k_actual: Optional[torch.Tensor] = None,
        b1plus_map: Optional[torch.Tensor] = None,     # (N,) complex
        coil_map: Optional[torch.Tensor] = None,        # (N,) complex, single coil for now
        b0_term: Optional[object] = None,               # PhaseTerm, built against whichever k the
        concomitant_term: Optional[object] = None,       # caller already chose consistently with
        gnl_term: Optional[object] = None,               # config.girf (see class docstring)
        config: PhysicsConfig = PhysicsConfig(),
        unflatten_order: str = "xy",        # matches RepoAdapter._flatten's own convention
    ):
        self.positions = positions
        self.t = t
        self.k_nominal = k_nominal
        self.k_actual = k_actual
        self.nufft = gridding_nufft
        self.im_size = im_size
        self.config = config
        self.device = positions.device

        n = positions.shape[0]
        self.b1plus_map = (b1plus_map if b1plus_map is not None
                           else torch.ones(n, dtype=torch.complex64, device=self.device))
        self.coil_map = (coil_map if coil_map is not None
                         else torch.ones(n, dtype=torch.complex64, device=self.device))

        if config.girf and k_actual is None:
            raise ValueError("config.girf=True requires k_actual (the realized/GIRF-perturbed "
                            "trajectory) - forward and adjoint must use the same realized "
                            "trajectory (handover section 6).")
        self._coords_t = (k_actual if config.girf else k_nominal)

        active_terms = []
        if config.b0 and b0_term is not None:
            active_terms.append(b0_term)
        if config.concomitant and concomitant_term is not None:
            active_terms.append(concomitant_term)
        if config.gnl and gnl_term is not None:
            active_terms.append(gnl_term)
        self._active_terms = active_terms
        self.segmented: Optional[SegmentedPhaseTerm] = None

    def prepare(self) -> "LinearMRIOperator":
        """One-time expensive setup (handover section 16: cache everything static). Must be
        called before forward/adjoint/normal."""
        if self._active_terms:
            combined = _SumPhaseTerms(self._active_terms)
            self.segmented = SegmentedPhaseTerm(n_segments=self.config.n_segments,
                                                binning="quantile")
            self.segmented.prepare(combined, self.positions, self.t)
        else:
            self.segmented = None
        return self

    def _coords_for_nufft(self) -> torch.Tensor:
        """(T, 2) normalized [-0.5, 0.5) coords, the convention GriddingNUFFT.forward/adjoint
        expect - matches RepoAdapter.simulate_all's own "k_nominal.T[:,:2] / (2*kmax_xy)"."""
        nx = self.im_size[0]
        # kmax is implicit in how k_nominal/k_actual were built (cycles/m); callers already scale
        # consistently elsewhere in this repo via "nx / (2*fov_m)" - accept pre-normalized coords
        # directly here to avoid baking a second, possibly-inconsistent FOV assumption into this
        # class. See build_linear_operator() for the one place this scaling actually happens.
        return self._coords_t

    def _apply_sensitivity(self, x: torch.Tensor) -> torch.Tensor:
        return self.b1plus_map * self.coil_map * x

    def _apply_sensitivity_adjoint(self, x: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.b1plus_map) * torch.conj(self.coil_map) * x

    def _nufft_forward(self, x_flat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Flat (N,) image -> (T,) k-space, via GriddingNUFFT's own (C, *im_size) convention."""
        grid = x_flat.reshape(*self.im_size)
        return self.nufft.forward(grid[None, ...], coords)[0]

    def _nufft_adjoint(self, d: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(T,) k-space -> flat (N,) image, via GriddingNUFFT's own (C, *im_size) convention."""
        grid = self.nufft.adjoint(d[None, :], coords)[0]
        return grid.reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image (N,) complex -> k-space data (T,) complex."""
        x_mod = self._apply_sensitivity(x)
        coords = self._coords_for_nufft()
        T = self.t.numel()

        if self.segmented is None:
            return self._nufft_forward(x_mod, coords)

        d = torch.zeros(T, dtype=torch.complex64, device=self.device)
        for l in range(self.config.n_segments):
            mask_l = self.segmented.bin_masks[l]
            if torch.count_nonzero(mask_l) == 0:
                continue
            x_l = x_mod * mask_l.to(x_mod.dtype)
            d_l = self._nufft_forward(x_l, coords)
            d = d + self.segmented.bin_phase_exp[l] * d_l
        return d

    def adjoint(self, d: torch.Tensor) -> torch.Tensor:
        """k-space data (T,) complex -> image (N,) complex, the mathematical adjoint of
        .forward() for the SAME segmentation/trajectory (no density compensation - section 8)."""
        coords = self._coords_for_nufft()
        n = self.positions.shape[0]

        if self.segmented is None:
            x = self._nufft_adjoint(d, coords)
            return self._apply_sensitivity_adjoint(x)

        x = torch.zeros(n, dtype=torch.complex64, device=self.device)
        for l in range(self.config.n_segments):
            mask_l = self.segmented.bin_masks[l]
            if torch.count_nonzero(mask_l) == 0:
                continue
            d_l = torch.conj(self.segmented.bin_phase_exp[l]) * d
            x_l = self._nufft_adjoint(d_l, coords)
            x = x + mask_l.to(x_l.dtype) * x_l
        return self._apply_sensitivity_adjoint(x)

    def normal(self, x: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
        out = self.adjoint(self.forward(x))
        if lam != 0.0:
            out = out + lam * x
        return out
