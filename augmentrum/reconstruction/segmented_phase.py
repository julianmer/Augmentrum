"""
Generic severity/frequency-segmented approximation of a spatially- AND temporally-varying phase
term exp(i*phi(r,t)), for use inside a fast (gridding-NUFFT-based) forward/adjoint operator pair.

Why this exists: B0, concomitant-field, and gradient-nonlinearity (GNL) phase terms in this repo
(StaticOffResonancePhase, _MultiShotConcomitantPhase, GradientNonlinearityPhase - see
scripts/spatial_physics_ablation.py and girf_mrsi_extensions.py) are all evaluated EXACTLY, per
voxel per time sample - correct for one-shot forward simulation, but each one individually costs
O(N*T) and a CG solve needs dozens of forward+adjoint calls, making that cost prohibitive.
A real frequency-segmented B0 implementation already exists (augmentrum/augmentation/
field_inhomogeneity.py's FieldInhomogeneity) but is fused into that module's own augmentation
pipeline, not available as a reusable forward/adjoint pair, and nothing equivalent exists at all
for concomitant/GNL (confirmed by direct code inspection, not assumed).

This module generalizes the classical frequency-segmented off-resonance correction (Noll et al.)
to an ARBITRARY phase term, by exploiting the fact that every PhaseTerm-like object here already
implements ".phase(positions, t) -> (N, T) real tensor": voxels are binned by a scalar "severity"
(by default, RMS-over-t of that same term's own phase - already exactly what this repo's own
RMS-over-readout phase maps use for the ablation figure), and each bin is represented not by a
fitted/parametric curve but by the EXACT phase-vs-time curve at one representative voxel drawn
from that bin. Full phi(r, t) is then approximated as piecewise-constant over bins in r. More
segments -> closer to the exact per-voxel phase; the number of segments is this scheme's only
free accuracy/cost knob (consistent with the handover's own "frequency-segments"/"severity-bins"
language), and the SAME representative-voxel-curve machinery applies unchanged to B0, concomitant,
or GNL, so there is no per-physics-term special-casing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SegmentedPhaseTerm:
    """
    A piecewise-constant-in-r approximation of a PhaseTerm's exp(i*phi(r,t)), ready to be used
    as "per-bin spatial mask" x "per-bin temporal exp(i*phi) curve" inside a fast NUFFT-based
    forward/adjoint pair (see LinearMRIOperator).

    Call ".prepare(phase_term, positions, t)" once (an expensive, exact evaluation - reuse the
    result across every CG iteration/every forward-adjoint call, per the handover's own "cache
    every static quantity possible").
    """
    n_segments: int
    binning: str = "quantile"   # "quantile" (equal-population bins) or "uniform" (equal-width)

    bin_masks: Optional[torch.Tensor] = None       # (L, N) float {0,1}, assigned AFTER prepare()
    bin_phase_exp: Optional[torch.Tensor] = None   # (L, T) complex, exp(i*phi_l(t))
    severity: Optional[torch.Tensor] = None        # (N,) float, for diagnostics/reuse
    representative_voxel: Optional[torch.Tensor] = None   # (L,) long, for diagnostics

    def prepare(self, phase_term, positions: torch.Tensor, t: torch.Tensor,
               severity: Optional[torch.Tensor] = None) -> "SegmentedPhaseTerm":
        device = positions.device if hasattr(positions, "device") else t.device
        n = positions.shape[0]

        if severity is None:
            # Signed phase at the LAST sample, not RMS-over-t: RMS is sign-blind (+f and -f
            # off-resonance give identical RMS severity despite opposite-sign phase curves,
            # which would wrongly bin them together and substitute one voxel's curve for the
            # other's) - confirmed empirically to produce near-pi representation errors
            # regardless of segment count before this fix. Using the final sample's signed,
            # exact phase is the natural generalization of classical frequency-segmented
            # reconstruction, which bins by the SIGNED field value itself, not a magnitude proxy.
            phi_full = phase_term.phase(positions, t)   # (N, T) - the one expensive exact call
            severity = phi_full[:, -1].to(torch.float64)
        severity = severity.to(device=device, dtype=torch.float64)

        L = self.n_segments
        if self.binning == "quantile":
            edges = torch.quantile(severity, torch.linspace(0.0, 1.0, L + 1, dtype=torch.float64,
                                                             device=device))
        elif self.binning == "uniform":
            edges = torch.linspace(severity.min(), severity.max(), L + 1, dtype=torch.float64,
                                   device=device)
        else:
            raise ValueError(f"Unknown binning {self.binning!r} (expected 'quantile' or 'uniform')")
        edges[0] -= 1.0    # make the lowest/highest edges inclusive of the true extrema
        edges[-1] += 1.0

        masks = torch.zeros(L, n, dtype=torch.float32, device=device)
        rep_voxel = torch.zeros(L, dtype=torch.long, device=device)
        rep_curves = torch.zeros(L, t.numel(), dtype=torch.float32, device=device)
        for l in range(L):
            in_bin = (severity > edges[l]) & (severity <= edges[l + 1])
            if not torch.any(in_bin):
                # Empty bin (can happen with "uniform" binning on a skewed severity
                # distribution) - give it a harmless representative voxel (voxel 0) and zero
                # mask, so it contributes nothing rather than raising.
                rep_voxel[l] = 0
                continue
            masks[l, in_bin] = 1.0
            idx_in_bin = torch.nonzero(in_bin, as_tuple=True)[0]
            bin_sev = severity[idx_in_bin]
            median_target = torch.median(bin_sev)
            closest = idx_in_bin[torch.argmin(torch.abs(bin_sev - median_target))]
            rep_voxel[l] = closest
            rep_curves[l] = phase_term.phase(positions[closest:closest + 1], t)[0]

        self.bin_masks = masks
        self.bin_phase_exp = torch.exp(1j * rep_curves.to(torch.complex64))
        self.severity = severity
        self.representative_voxel = rep_voxel
        return self

    def max_abs_representation_error(self, phase_term, positions, t) -> float:
        """Diagnostic only (expensive - do not call inside a CG loop): the true per-voxel exact
        phase minus this segmentation's piecewise-constant approximation, max over (r, t)."""
        phi_exact = phase_term.phase(positions, t).to(torch.float64)
        phi_approx = (self.bin_masks.to(torch.float64).T
                     @ torch.angle(self.bin_phase_exp).to(torch.float64))
        # angle() wraps to (-pi,pi]; compare via the wrapped difference, not the raw one.
        diff = torch.remainder(phi_exact - phi_approx + torch.pi, 2 * torch.pi) - torch.pi
        return float(torch.max(torch.abs(diff)))
