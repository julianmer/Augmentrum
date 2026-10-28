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
    girf: bool = False          # toggles the GIRF trajectory-error PHASE TERM (see
                                # _GIRFTrajectoryErrorPhase) - the NUFFT itself always grids on
                                # k_nominal regardless of this flag (see class docstring)
    concomitant: bool = False
    gnl: bool = False
    n_segments: int = 8


class _GIRFTrajectoryErrorPhase:
    """
    phi_GIRF(r, t) = -2*pi * dk(t) . r, where dk(t) = k_actual(t) - k_nominal(t) is the measured
    GIRF trajectory error (cycles/m), and r is in meters - so phi is in radians, matching the
    standard Fourier phase convention exp(-i*2*pi*k.r).

    This is an EXACT algebraic reformulation, not an approximation: exp(-i*2*pi*k_actual(t).r) =
    exp(-i*2*pi*(k_nominal(t)+dk(t)).r) = exp(-i*2*pi*k_nominal(t).r) * exp(-i*2*pi*dk(t).r).
    Applying this as an ordinary phase term on top of a NUFFT that always grids on k_nominal is
    therefore mathematically identical to gridding directly onto k_actual, while keeping the
    NUFFT (and its segmentation) structurally IDENTICAL regardless of whether GIRF is enabled -
    unlike switching which trajectory the gridding itself uses, which changes the sampling
    geometry altogether between GIRF-on and GIRF-off configurations. That structural difference
    was suspected (not just theorized - see CG_RECON_STATUS.md) to be the cause of severe CG
    instability in "B0 without GIRF" configurations once GIRF was still implemented as a
    trajectory swap.
    """

    def __init__(self, dk: torch.Tensor):
        self.dk = dk   # (D, T) cycles/m, D matching positions' own first D columns

    def phase(self, positions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        d = self.dk.shape[0]
        r = positions[:, :d].to(self.dk.dtype)        # (M, D)
        return -2.0 * torch.pi * (r @ self.dk.to(r.device))   # (M, T)


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
        k_nominal,                          # (T, 2) numpy, normalized [-0.5,0.5) - GriddingNUFFT's
                                             # own convention; MUST be numpy/CPU, not a device
                                             # tensor (GriddingNUFFT.forward/adjoint call
                                             # np.asarray(coords) internally). The NUFFT ALWAYS
                                             # grids on this trajectory - see class/PhysicsConfig
                                             # docstrings for why GIRF is a phase term instead of
                                             # a second trajectory.
        gridding_nufft,                     # augmentrum.sampling.kspace_reconstructor.GriddingNUFFT
        im_size: tuple,
        b1plus_map: Optional[torch.Tensor] = None,     # (N,) complex
        coil_map: Optional[torch.Tensor] = None,        # (N,) complex, single coil for now
        b0_term: Optional[object] = None,
        concomitant_term: Optional[object] = None,
        gnl_term: Optional[object] = None,               # built against whichever k_traj the
                                                           # caller judges correct for GNL's own
                                                           # k(t).Delta_r(r) formula - independent
                                                           # of this class's NUFFT trajectory choice
        girf_term: Optional[object] = None,               # _GIRFTrajectoryErrorPhase - the
                                                           # trajectory-error phase term; see
                                                           # PhysicsConfig.girf
        config: PhysicsConfig = PhysicsConfig(),
        unflatten_order: str = "xy",        # matches RepoAdapter._flatten's own convention
    ):
        self.positions = positions
        self.t = t
        self.k_nominal = k_nominal
        self.nufft = gridding_nufft
        self.im_size = im_size
        self.config = config
        self.device = positions.device

        n = positions.shape[0]
        self.b1plus_map = (b1plus_map if b1plus_map is not None
                           else torch.ones(n, dtype=torch.complex64, device=self.device))

        # coil_map: (N,) for single-coil (backward compatible - internally promoted to (1,N)) or
        # (C,N) for real multi-coil (handover section 6: "forward: C_c(r)*x(r); adjoint:
        # sum_c conj(C_c(r))*x_c(r)"). GriddingNUFFT.forward/adjoint already accept a batched
        # leading channel dimension ([C, *im_size]), so every coil is vectorized through ONE
        # NUFFT call per segment rather than looping per-coil x per-segment (which would multiply
        # the already-dominant NUFFT cost by n_coils).
        coil_map = (coil_map if coil_map is not None
                   else torch.ones(n, dtype=torch.complex64, device=self.device))
        if coil_map.ndim == 1:
            coil_map = coil_map.unsqueeze(0)
        self.coil_map = coil_map              # (C, N)
        self.n_coils = coil_map.shape[0]

        # ALWAYS k_nominal - see class/PhysicsConfig docstrings. GIRF trajectory error enters as
        # a phase term (girf_term) instead of changing which trajectory the NUFFT grids onto.
        self._coords_t = k_nominal

        active_terms = []
        if config.b0 and b0_term is not None:
            active_terms.append(b0_term)
        if config.concomitant and concomitant_term is not None:
            active_terms.append(concomitant_term)
        if config.gnl and gnl_term is not None:
            active_terms.append(gnl_term)
        if config.girf and girf_term is not None:
            active_terms.append(girf_term)
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
        """Image (N,) -> per-coil images (C, N). B1+ is a TRANSMIT effect, shared by every
        receive channel; coil_map carries the per-channel RECEIVE sensitivity."""
        return self.b1plus_map.unsqueeze(0) * self.coil_map * x.unsqueeze(0)

    def _apply_sensitivity_adjoint(self, x_c: torch.Tensor) -> torch.Tensor:
        """Per-coil images (C, N) -> image (N,), summed over coils (handover section 6)."""
        return (torch.conj(self.b1plus_map).unsqueeze(0) * torch.conj(self.coil_map) * x_c).sum(dim=0)

    def _nufft_forward(self, x_batch: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(C, N) per-coil images -> (C, T) k-space, via GriddingNUFFT's native (C, *im_size)
        channel batching - every coil goes through ONE NUFFT call, not a per-coil loop."""
        grid = x_batch.reshape(self.n_coils, *self.im_size)
        return self.nufft.forward(grid, coords)

    def _nufft_adjoint(self, d_batch: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(C, T) k-space -> (C, N) per-coil images, via GriddingNUFFT's native channel batching."""
        grid = self.nufft.adjoint(d_batch, coords)
        return grid.reshape(self.n_coils, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image (N,) complex -> k-space data (C, T) complex (C=1 dim present even for a single
        coil, so callers get a consistent shape; cg_solve/metrics flatten regardless)."""
        x_mod = self._apply_sensitivity(x)   # (C, N)
        coords = self._coords_for_nufft()
        T = self.t.numel()

        if self.segmented is None:
            return self._nufft_forward(x_mod, coords)

        d = torch.zeros(self.n_coils, T, dtype=torch.complex64, device=self.device)
        for l in range(self.config.n_segments):
            mask_l = self.segmented.bin_masks[l]
            if torch.count_nonzero(mask_l) == 0:
                continue
            x_l = x_mod * mask_l.to(x_mod.dtype).unsqueeze(0)             # (C, N)
            d_l = self._nufft_forward(x_l, coords)                        # (C, T)
            d = d + self.segmented.bin_phase_exp[l].unsqueeze(0) * d_l
        return d

    def adjoint(self, d: torch.Tensor) -> torch.Tensor:
        """k-space data (C, T) [or (T,) for single-coil convenience] complex -> image (N,)
        complex, the mathematical adjoint of .forward() for the SAME segmentation/trajectory/coil
        maps (no density compensation - section 8)."""
        if d.ndim == 1:
            d = d.unsqueeze(0)
        coords = self._coords_for_nufft()
        n = self.positions.shape[0]

        if self.segmented is None:
            x_c = self._nufft_adjoint(d, coords)
            return self._apply_sensitivity_adjoint(x_c)

        x_c = torch.zeros(self.n_coils, n, dtype=torch.complex64, device=self.device)
        for l in range(self.config.n_segments):
            mask_l = self.segmented.bin_masks[l]
            if torch.count_nonzero(mask_l) == 0:
                continue
            d_l = torch.conj(self.segmented.bin_phase_exp[l]).unsqueeze(0) * d   # (C, T)
            x_l = self._nufft_adjoint(d_l, coords)                               # (C, N)
            x_c = x_c + mask_l.to(x_l.dtype).unsqueeze(0) * x_l
        return self._apply_sensitivity_adjoint(x_c)

    def normal(self, x: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
        out = self.adjoint(self.forward(x))
        if lam != 0.0:
            out = out + lam * x
        return out
