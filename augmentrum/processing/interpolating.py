####################################################################################################
#                                        interpolating.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-22                                                                              #
#                                                                                                  #
# Purpose: Hermite modified-Akima interpolation of gridded volumes at arbitrary off-grid           #
#          coordinates, for sampling k-space along non-Cartesian trajectories. The forward half    #
#          of the pair whose adjoint is sampling.kspace_reconstructor; coordinates are             #
#          normalised to [-1, 1] per axis, as they are there.                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


__all__ = ['HermiteMAkimaInterpolator', 'BicubicHermiteMAkima2D', 'TricubicHermiteMAkima3D']


#**************************************************************************************************#
#                                 Class HermiteMAkimaInterpolator                                  #
#**************************************************************************************************#
#                                                                                                  #
# Base for N-dimensional Hermite modified-Akima interpolators.                                     #
#                                                                                                  #
#**************************************************************************************************#
class HermiteMAkimaInterpolator(nn.Module, ABC):
    """
    Base for N-dimensional Hermite modified-Akima interpolators.

    Holds everything the 2-D and 3-D interpolators share: axis construction,
    modified-Akima slope estimation, the Hermite basis, and the index/local-
    coordinate lookup. A subclass supplies its dimension count and the tensor
    contraction over the neighbourhood.

    Signal layout is ``[B, S, *grid]``; a signal given without the shot axis
    gets ``S = 1`` inserted. Coordinates are ``[B, S, NDIM, L]``, normalised to
    ``[-1, 1]``, and the result is ``[B, S, 1, L]``.

    >>> interp = BicubicHermiteMAkima2D(torch.zeros(1, 1, 8, 8))
    >>> tuple(interp(torch.zeros(1, 2, 2, 5)).shape)
    (1, 2, 1, 5)
    """

    #: Number of spatial dimensions the subclass interpolates over.
    NDIM: int = 0

    def __init__(self, signal: torch.Tensor):
        """
        Args:
            signal: gridded values, ``[B, S, *grid]`` or ``[B, *grid]``
                (the shot axis is inserted when absent).
        """
        super().__init__()
        expected = self.NDIM + 2
        if signal.dim() == expected - 1:
            signal = signal.unsqueeze(1)
        elif signal.dim() != expected:
            raise ValueError(
                f"{type(self).__name__} expects a {expected - 1}-D [B, *grid] or "
                f"{expected}-D [B, S, *grid] signal, got {signal.dim()}-D."
            )

        self.dtype = signal.dtype
        self.device = signal.device
        self.signal = signal.to(self.device)

        self.B, self.S = signal.shape[0], signal.shape[1]
        self.grid_shape = tuple(signal.shape[2:])

        # One normalised axis per spatial dimension, and the cell width on it.
        # The axes are uniform by construction, so a single scalar width is exact.
        self.axes = [
            torch.linspace(-1.0, 1.0, steps=n, device=self.device)
            for n in self.grid_shape
        ]
        self.cell = [ax[1] - ax[0] for ax in self.axes]

        # Slopes along each spatial axis, at signal dimension 2 + d.
        self.slopes = [
            self._prepare_slopes(self.signal, self._axis_view(d), dim=2 + d)
            for d in range(self.NDIM)
        ]

    #*******************#
    #   grid geometry   #
    #*******************#

    def _axis_view(self, d: int) -> torch.Tensor:
        """Axis *d* broadcast against the signal, i.e. ``[1, 1, ..., n_d, ..., 1]``."""
        shape = [1, 1] + [1] * self.NDIM
        shape[2 + d] = self.grid_shape[d]
        return self.axes[d].view(*shape)

    def _prepare_slopes(self, signal: torch.Tensor, axis: torch.Tensor,
                        dim: int) -> torch.Tensor:
        """
        Modified-Akima slopes of *signal* along *dim*.

        The 3-D copy of this used to call ``axis.unsqueeze(0).diff(dim=dim)``,
        which differences the wrong dimension and raised on any input — the
        tricubic interpolator could never be constructed. The axis is already
        broadcast to the signal's rank by :meth:`_axis_view`, so it is
        differenced directly.
        """
        def slc(start=None, end=None):
            s = [slice(None)] * signal.dim()
            s[dim] = slice(start, end)
            return tuple(s)

        m = signal.diff(dim=dim) / axis.diff(dim=dim)

        # Extrapolate two slopes past each end, as modified Akima requires.
        d0 = 2 * m.select(dim, 0) - m.select(dim, 1)
        dn = 2 * m.select(dim, -1) - m.select(dim, -2)
        m0 = 2 * d0 - m.select(dim, 0)
        mn = 2 * dn - m.select(dim, -1)
        m = torch.cat([m0.unsqueeze(dim), d0.unsqueeze(dim), m,
                       dn.unsqueeze(dim), mn.unsqueeze(dim)], dim=dim)

        m_prev, m_next = m[slc(None, -1)], m[slc(1, None)]
        weights = torch.abs(m_next - m_prev) + torch.abs((m_prev + m_next) / 2.0)

        w1, w2 = weights[slc(None, -2)], weights[slc(2, None)]
        delta1, delta2 = m[slc(1, -2)], m[slc(2, -1)]
        w_sum = w1 + w2 + 1e-6
        slopes = (w2 / w_sum) * delta1 + (w1 / w_sum) * delta2
        slopes[w_sum == 0] = 0.0
        return slopes

    @staticmethod
    def h_poly(t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Cubic Hermite basis functions ``h00, h10, h01, h11``."""
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return h00, h10, h01, h11

    def _basis(self, t: torch.Tensor) -> torch.Tensor:
        """
        Hermite basis stacked to match the coefficient tensor's block layout.

        The basis functions come in interleaved order — value, derivative,
        value, derivative — while ``P`` is built in blocks: function values at
        ``0:2`` and scaled derivatives at ``2:4``. Both interpolators used to
        stack the basis interleaved and contract it against the blocked ``P``,
        so the two orderings disagreed and the contraction picked the wrong
        coefficients: sampled exactly on a grid node the result came out as the
        cross-derivative term rather than the node's value. Stacking as
        ``(h00, h01, h10, h11)`` puts the value weights against the value block
        and the derivative weights against the derivative block.
        """
        h00, h10, h01, h11 = self.h_poly(t)
        return torch.stack([h00, h01, h10, h11], dim=-1).to(self.dtype)

    def _locate(self, coord: torch.Tensor, d: int):
        """Cell index, local coordinate and cell width for *coord* on axis *d*."""
        ax = self.axes[d]
        idx = (torch.searchsorted(ax, coord.clamp(ax.min(), ax.max())) - 1)
        idx = idx.clamp(0, self.grid_shape[d] - 2)
        width = self.cell[d].expand_as(idx)
        return idx, (coord - ax[idx]) / width, width

    def _neighbours(self, idx: torch.Tensor, d: int) -> torch.Tensor:
        """The four sample indices ``i-1, i, i+1, i+2`` clamped to axis *d*."""
        return torch.stack([idx - 1, idx, idx + 1, idx + 2],
                           dim=-1).clamp(0, self.grid_shape[d] - 1)

    #*******************#
    #   interpolation   #
    #*******************#
    @abstractmethod
    def _interpolate(self, idx, t, width) -> torch.Tensor:
        """Contract the Hermite bases against the coefficient tensor. ``[B, S, L]``."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Interpolate the signal at *coords*.

        Args:
            coords: ``[B, S, NDIM, L]``, normalised to ``[-1, 1]``.

        Returns:
            ``[B, S, 1, L]`` interpolated values.
        """
        if coords.dim() != 4 or coords.shape[2] != self.NDIM:
            raise ValueError(
                f"{type(self).__name__} expects coords [B, S, {self.NDIM}, L], "
                f"got {tuple(coords.shape)}."
            )
        batch, n_shots, ndim, length = coords.shape

        # Fold the shots into the sample axis so one pass covers them all. The
        # signal's own shot axis is indexed with S=1, so a multi-shot signal is
        # broadcast rather than indexed per shot.
        flat = coords.transpose(1, 2).contiguous().view(batch, 1, ndim, n_shots * length)
        per_axis = [c.squeeze(2) for c in flat.split(1, dim=2)]

        idx, t, width = [], [], []
        for d, coord in enumerate(per_axis):
            i, tt, w = self._locate(coord, d)
            idx.append(i)
            t.append(tt.to(self.dtype))
            width.append(w)

        values = self._interpolate(idx, t, width)
        return values.view(batch, 1, n_shots, length).transpose(1, 2).contiguous()

    def _gather(self, neighbours):
        """Signal and slope sub-grids at the neighbour indices."""
        b_idx = torch.arange(self.B, device=self.device, dtype=torch.long).view(
            self.B, *([1] * (1 + 1 + self.NDIM)))
        s_idx = torch.arange(1, device=self.device, dtype=torch.long).view(
            1, 1, *([1] * (1 + self.NDIM)))
        index = (b_idx, s_idx) + tuple(neighbours)
        return self.signal[index], [m[index] for m in self.slopes]


#**************************************************************************************************#
#                                   Class BicubicHermiteMAkima2D                                   #
#**************************************************************************************************#
#                                                                                                  #
# Bicubic Hermite modified-Akima interpolation on a 2-D grid ``[B, S, Nx, Ny]``.                   #
#                                                                                                  #
#**************************************************************************************************#
class BicubicHermiteMAkima2D(HermiteMAkimaInterpolator):
    """Bicubic Hermite modified-Akima interpolation on a 2-D grid ``[B, S, Nx, Ny]``."""

    NDIM = 2

    def _interpolate(self, idx, t, width):
        I_x, I_y = idx
        t_x, t_y = t
        dx_cell, dy_cell = width

        H_x = self._basis(t_x)
        H_y = self._basis(t_y)

        ix_n = self._neighbours(I_x, 0)
        iy_n = self._neighbours(I_y, 1)
        signal_sub, (m_x_sub, m_y_sub) = self._gather(
            (ix_n.unsqueeze(-1), iy_n.unsqueeze(-2)))

        # Cross-derivative fxy by central difference of fx across y.
        denom_y = (self.axes[1][iy_n[..., 2]] - self.axes[1][iy_n[..., 0]]).clamp(min=1e-12)
        fxy_base = (m_x_sub[..., 1, 2] - m_x_sub[..., 1, 0]) / denom_y

        P = torch.zeros(signal_sub.shape, dtype=self.dtype, device=self.device)
        dx = dx_cell.unsqueeze(-1)
        dy = dy_cell.unsqueeze(-1)
        P[..., 0:2, 0:2] = signal_sub[..., 1:3, 1:3]
        P[..., 2:4, 0:2] = m_x_sub[..., 1:3, 1:3] * dx.unsqueeze(-1)
        P[..., 0:2, 2:4] = m_y_sub[..., 1:3, 1:3] * dy.unsqueeze(-2)
        P[..., 2:4, 2:4] = fxy_base.unsqueeze(-1).unsqueeze(-1) * (dx * dy).unsqueeze(-1)

        return torch.einsum('bsli, bslj, bslij -> bsl', H_x, H_y, P)


#**************************************************************************************************#
#                                  Class TricubicHermiteMAkima3D                                   #
#**************************************************************************************************#
#                                                                                                  #
# Tricubic Hermite modified-Akima interpolation on a 3-D grid ``[B, S, Nx, Ny, Nz]``.              #
#                                                                                                  #
#**************************************************************************************************#
class TricubicHermiteMAkima3D(HermiteMAkimaInterpolator):
    """Tricubic Hermite modified-Akima interpolation on a 3-D grid ``[B, S, Nx, Ny, Nz]``."""

    NDIM = 3

    def _interpolate(self, idx, t, width):
        I_x, I_y, I_z = idx
        t_x, t_y, t_z = t
        dx_cell, dy_cell, dz_cell = width

        H_x = self._basis(t_x)
        H_y = self._basis(t_y)
        H_z = self._basis(t_z)

        ix_n = self._neighbours(I_x, 0)
        iy_n = self._neighbours(I_y, 1)
        iz_n = self._neighbours(I_z, 2)
        signal_sub, (m_x_sub, m_y_sub, m_z_sub) = self._gather((
            ix_n.unsqueeze(-1).unsqueeze(-1),
            iy_n.unsqueeze(-2).unsqueeze(-1),
            iz_n.unsqueeze(-2).unsqueeze(-2),
        ))

        denom_y = (self.axes[1][iy_n[..., 2]] - self.axes[1][iy_n[..., 0]]
                   ).clamp(min=1e-12).to(self.dtype)
        denom_z = (self.axes[2][iz_n[..., 2]] - self.axes[2][iz_n[..., 0]]
                   ).clamp(min=1e-12).to(self.dtype)

        fxy_base = (m_x_sub[..., 1, 2, 1] - m_x_sub[..., 1, 0, 1]) / denom_y
        fxz_base = (m_x_sub[..., 1, 1, 2] - m_x_sub[..., 1, 1, 0]) / denom_z
        fyz_base = (m_y_sub[..., 1, 1, 2] - m_y_sub[..., 1, 1, 0]) / denom_z
        fxy_z_plus = (m_x_sub[..., 1, 2, 2] - m_x_sub[..., 1, 0, 2]) / denom_y
        fxy_z_minus = (m_x_sub[..., 1, 2, 0] - m_x_sub[..., 1, 0, 0]) / denom_y
        fxyz_base = (fxy_z_plus - fxy_z_minus) / denom_z

        P = torch.zeros(signal_sub.shape, dtype=self.dtype, device=self.device)
        dx = dx_cell.unsqueeze(-1).unsqueeze(-1).to(self.dtype)
        dy = dy_cell.unsqueeze(-1).unsqueeze(-1).to(self.dtype)
        dz = dz_cell.unsqueeze(-1).unsqueeze(-1).to(self.dtype)

        P[..., 0:2, 0:2, 0:2] = signal_sub[..., 1:3, 1:3, 1:3]
        P[..., 2:4, 0:2, 0:2] = m_x_sub[..., 1:3, 1:3, 1:3] * dx.unsqueeze(-1)
        P[..., 0:2, 2:4, 0:2] = m_y_sub[..., 1:3, 1:3, 1:3] * dy.unsqueeze(-1)
        P[..., 0:2, 0:2, 2:4] = m_z_sub[..., 1:3, 1:3, 1:3] * dz.unsqueeze(-1)

        dxdy = (dx * dy).unsqueeze(-1)
        dxdz = (dx * dz).unsqueeze(-1)
        dydz = (dy * dz).unsqueeze(-1)
        dxdydz = dxdy * dz.unsqueeze(-1)

        def corner(base):
            return base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        P[..., 2:4, 2:4, 0:2] = corner(fxy_base) * dxdy
        P[..., 2:4, 0:2, 2:4] = corner(fxz_base) * dxdz
        P[..., 0:2, 2:4, 2:4] = corner(fyz_base) * dydz
        P[..., 2:4, 2:4, 2:4] = corner(fxyz_base) * dxdydz

        return torch.einsum('bsli, bslj, bslk, bslijk -> bsl', H_x, H_y, H_z, P)
