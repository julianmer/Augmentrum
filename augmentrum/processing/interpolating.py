####################################################################################################
#                                        interpolating.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-22                                                                              #
#                                                                                                  #
# Purpose: Sampling gridded arrays at arbitrary off-grid coordinates. Defines the Interpolator     #
#          contract and its implementations - linear, Kaiser-Bessel and Hermite modified-Akima -   #
#          which are what a non-Cartesian trajectory needs to read k-space off a grid.             #
#          Coordinates are normalized to [-1, 1] per axis throughout.                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations           # so torch annotations never evaluate

import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import numpy as np

from nifti_mrs_plus import ops


__all__ = ['Interpolator', 'LinearInterpolator', 'GriddingKernel', 'KaiserBesselInterpolator',
           'HermiteMAkimaInterpolator', 'BicubicHermiteMAkima2D', 'TricubicHermiteMAkima3D']


#**************************************************************************************************#
#                                        Class Interpolator                                        #
#**************************************************************************************************#
#                                                                                                  #
# Contract for sampling a gridded array at arbitrary coordinates.                                  #
#                                                                                                  #
#**************************************************************************************************#
class Interpolator(ABC):
    """
    Contract for sampling a gridded array at arbitrary coordinates.

    Reading a non-Cartesian trajectory off a Cartesian grid is one job with
    several answers, differing in accuracy and cost. Stating it as a contract
    lets a caller swap between them - which is exactly what an accuracy study
    of the k-space forward operator varies.

    Coordinates are normalized to [-1, 1] per axis. Implementations take
    "grid" shaped "[C, *grid_shape]" and return "[C, K]".
    """

    @abstractmethod
    def sample(self, grid, coords):
        """
        Sample *grid* at *coords*.

        Args:
            grid: Gridded values, "[C, *grid_shape]", on any backend.
            coords: Normalized coordinates in [-1, 1], "[K, ndim]" (NumPy).

        Returns:
            Values "[C, K]" on grid's backend.
        """


#**************************************************************************************************#
#                                     Class LinearInterpolator                                     #
#**************************************************************************************************#
#                                                                                                  #
# Bilinear or trilinear sampling. The cheapest option, and the least accurate.                     #
#                                                                                                  #
#**************************************************************************************************#
class LinearInterpolator(Interpolator):
    """Bilinear or trilinear sampling. The cheapest option, and the least accurate."""

    def sample(self, grid, coords):
        """Sample by linear interpolation between neighboring bins."""
        n_chan = int(ops.shape(grid)[0])
        coords = np.asarray(coords)

        # grid_sample wants [N, C, ...] and coordinates ordered x, y[, z],
        # which is the reverse of the spatial axes
        stack = ops.reshape(grid, (1,) + tuple(ops.shape(grid)))
        view = (1,) * (len(ops.shape(grid)) - 1) + (len(coords), coords.shape[1])
        sample_grid = ops.match_backend(
            np.ascontiguousarray(coords[..., ::-1]).astype(np.float32).reshape(view),
            ops.real(grid))

        if ops.is_complex(grid):
            out = ops.complex_from(
                ops.grid_sample(ops.real(stack), sample_grid, padding_mode="border"),
                ops.grid_sample(ops.imag(stack), sample_grid, padding_mode="border"))
        else:
            out = ops.grid_sample(stack, sample_grid, padding_mode="border")

        return ops.reshape(out, (n_chan, len(coords)))


#**************************************************************************************************#
#                                       Class GriddingKernel                                       #
#**************************************************************************************************#
#                                                                                                  #
# Interpolator that can also write to the grid, and so can drive a NUFFT.                          #
#                                                                                                  #
#**************************************************************************************************#
class GriddingKernel(Interpolator):
    """
    Interpolator that can also write to the grid, and so can drive a NUFFT.

    Gathering from a grid is enough to read a trajectory, but a NUFFT also needs
    the reverse - scattering a sample across the bins it touches - and a
    correction for the taper that leaves in the image. The two travel together:
    a kernel wide enough to spread with is wide enough to taper, so a caller
    that has one has the other.

    Unlike a plain interpolator these are tied to a grid geometry, because
    kernel width and shape are chosen against the oversampling factor.

    Args:
        grid_size: Oversampled grid the kernel addresses.
    """

    def __init__(self, grid_size: Sequence[int]):
        self.grid_size = tuple(int(n) for n in grid_size)
        self.ndim = len(self.grid_size)

    @abstractmethod
    def neighbors(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bins each sample touches, and the kernel weight for each.

        The common ground under gathering and scattering alike, and what a
        density estimate counts.

        Args:
            coords: Trajectory in [-0.5, 0.5), "[K, ndim]" (NumPy).

        Returns:
            "(indices, weights)", both "[K, taps**ndim]", indices flat into the grid.
        """

    @abstractmethod
    def spread(self, values, coords):
        """
        Scatter samples onto the grid: the reverse of :meth:"sample".

        Args:
            values: Samples "[C, K]", on any backend.
            coords: Trajectory in [-0.5, 0.5), "[K, ndim]" (NumPy).

        Returns:
            Grid "[C, *grid_size]" on values' backend.
        """

    @abstractmethod
    def deapodization(self, im_size: Sequence[int]) -> np.ndarray:
        """
        Image-domain correction for the taper this kernel leaves behind.

        Args:
            im_size: Image matrix the oversampled grid is cropped to.

        Returns:
            Multiplicative correction "[*im_size]" (NumPy).
        """


#**************************************************************************************************#
#                                  Class KaiserBesselInterpolator                                  #
#**************************************************************************************************#
#                                                                                                  #
# Kaiser-Bessel gridding kernel: the interpolator a NUFFT is built on.                             #
#                                                                                                  #
#**************************************************************************************************#
class KaiserBesselInterpolator(GriddingKernel):
    """
    Kaiser-Bessel gridding kernel: the interpolator a NUFFT is built on.

    Kernel and taper depend only on the grid geometry, never on the data, so
    both are built once in NumPy and promoted to the caller's backend.

    Args:
        grid_size: Oversampled grid the kernel addresses.
        osf: Oversampling factor the grid represents.
        width: Kernel width in oversampled grid units.
    """

    def __init__(self, grid_size: Sequence[int], osf: float = 2.0, width: float = 4.0):
        super().__init__(grid_size)
        self.osf = float(osf)
        self.width = float(width)
        self.beta = self.beatty_beta(self.width, self.osf)

    #************#
    #   kernel   #
    #************#
    @staticmethod
    def beatty_beta(width: float, osf: float) -> float:
        """
        Kaiser-Bessel shape parameter, from Beatty et al. (2005).

        Chosen to minimize aliasing energy for a given kernel width and
        oversampling factor.
        """
        return math.pi * math.sqrt((width / osf) ** 2 * (osf - 0.5) ** 2 - 0.8)

    def kernel(self, offsets: np.ndarray) -> np.ndarray:
        """Kernel evaluated at *offsets* in grid units."""
        from scipy.special import i0

        r = 2.0 * offsets / self.width
        inside = np.abs(r) <= 1.0
        arg = self.beta * np.sqrt(np.clip(1.0 - r ** 2, 0.0, None))
        return np.where(inside, i0(arg) / self.width, 0.0)

    def deapodization(self, im_size: Sequence[int]) -> np.ndarray:
        """
        Reciprocal of the kernel's transform, as an outer product over axes.

        Gridding convolves with the kernel, which tapers the image; dividing by
        the kernel's transform undoes exactly that taper.
        """
        deapod = None
        for n, grid in zip(im_size, self.grid_size):
            axis = self._axis_deapodization(int(n), int(grid))
            deapod = axis if deapod is None else np.multiply.outer(deapod, axis)
        return deapod.astype(np.float32)

    def _axis_deapodization(self, n: int, grid: int) -> np.ndarray:
        """The correction along one image axis of length *n*."""
        x = (np.arange(n) - n // 2) / float(grid)
        arg = (math.pi * self.width * x) ** 2 - self.beta ** 2

        # Real above the root, imaginary below it, where sin becomes sinh
        positive = arg > 0
        root_pos = np.sqrt(np.clip(arg, 0.0, None))
        root_neg = np.sqrt(np.clip(-arg, 0.0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(positive,
                           np.sin(root_pos) / np.where(root_pos == 0, 1.0, root_pos),
                           np.sinh(root_neg) / np.where(root_neg == 0, 1.0, root_neg))
        return np.where(root_pos + root_neg == 0, 1.0, out)

    #**************#
    #   sampling   #
    #**************#
    def neighbors(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The bins within half a kernel width of each sample, and their weights."""
        half = int(math.ceil(self.width / 2.0))
        taps = np.arange(-half, half + 1)

        per_axis_idx, per_axis_w = [], []
        for d in range(self.ndim):
            center = (coords[:, d] + 0.5) * self.grid_size[d]      # bins
            base = np.floor(center)
            idx = base[:, None] + taps[None, :]                    # [K, T]
            per_axis_w.append(self.kernel(center[:, None] - idx))
            per_axis_idx.append(np.mod(idx, self.grid_size[d]).astype(np.int64))

        # Outer product over axes, flattened to [K, T**ndim]
        flat_idx, flat_w = None, None
        for d in range(self.ndim):
            stride = int(np.prod(self.grid_size[d + 1:])) if d + 1 < self.ndim else 1
            idx_d = per_axis_idx[d] * stride
            if flat_idx is None:
                flat_idx, flat_w = idx_d, per_axis_w[d]
            else:
                flat_idx = (flat_idx[:, :, None] + idx_d[:, None, :]).reshape(len(coords), -1)
                flat_w = (flat_w[:, :, None] * per_axis_w[d][:, None, :]).reshape(len(coords), -1)
        return flat_idx, flat_w.astype(np.float32)

    def sample(self, grid, coords):
        """Gather each sample from the bins its kernel touches."""
        idx, weight = self.neighbors(np.asarray(coords))
        n_chan = int(ops.shape(grid)[0])
        n_taps = idx.shape[1]

        flat = ops.reshape(grid, (n_chan, -1))
        weight_b = ops.cast_like(ops.match_backend(weight, flat), flat)

        rows = []
        for c in range(n_chan):
            plane = ops.reshape(ops.take(flat, np.array([c]), axis=0), (-1,))
            taps = ops.reshape(ops.take(plane, idx.reshape(-1), axis=0),
                               (len(coords), n_taps))
            rows.append(ops.sum(taps * weight_b, axis=-1))
        return ops.stack(rows, axis=0)

    def spread(self, values, coords):
        """Scatter each sample across the bins its kernel touches."""
        idx, weight = self.neighbors(np.asarray(coords))
        n_chan = int(ops.shape(values)[0])
        n_cells = int(np.prod(self.grid_size))
        n_taps = idx.shape[1]

        weight_b = ops.cast_like(ops.match_backend(weight.reshape(-1), values), values)
        idx_flat = idx.reshape(-1)

        planes = []
        for c in range(n_chan):
            # One sample feeds every tap it touches, so repeat it across them
            samples = ops.reshape(ops.take(values, np.array([c]), axis=0), (-1,))
            spread = ops.reshape(ops.stack([samples] * n_taps, axis=-1), (-1,))
            planes.append(ops.scatter_add((n_cells,), idx_flat, spread * weight_b))

        return ops.reshape(ops.stack(planes, axis=0), (n_chan,) + self.grid_size)


#**************************************************************************************************#
#                                 Class HermiteMAkimaInterpolator                                  #
#**************************************************************************************************#
#                                                                                                  #
# Base for N-dimensional Hermite modified-Akima interpolators.                                     #
#                                                                                                  #
#**************************************************************************************************#
class HermiteMAkimaInterpolator(Interpolator):
    """
    Base for N-dimensional Hermite modified-Akima interpolators.

    Holds everything the 2-D and 3-D interpolators share: axis construction,
    modified-Akima slope estimation, the Hermite basis, and the index/local-
    coordinate lookup. A subclass supplies its dimension count and the tensor
    contraction over the neighborhood.

    Signal layout is "[B, S, *grid]"; a signal given without the shot axis
    gets "S = 1" inserted. Coordinates are "[B, S, NDIM, L]", normalized to
    "[-1, 1]", and the result is "[B, S, 1, L]".

    >>> interp = BicubicHermiteMAkima2D(torch.zeros(1, 1, 8, 8))
    >>> tuple(interp(torch.zeros(1, 2, 2, 5)).shape)
    (1, 2, 1, 5)
    """

    #: Number of spatial dimensions the subclass interpolates over.
    NDIM: int = 0

    def __init__(self, signal: Optional[torch.Tensor] = None):
        """
        Args:
            signal: gridded values, "[B, S, *grid]" or "[B, *grid]" (the shot
                axis is inserted when absent). Optional - an instance built
                without one binds a grid per call in :meth:"sample" instead.
        """
        import torch                       # deferred: this branch is torch-only
        self._torch = torch
        self.signal = None
        if signal is None:
            return

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

        # One normalized axis per spatial dimension, and the cell width on it.
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

    def sample(self, grid, coords):
        """
        Sample *grid* at *coords*, satisfying the :class:"Interpolator" contract.

        This class binds its signal at construction, because the modified-Akima
        slopes are a property of the grid and are worth computing once. The
        contract passes the grid per call, so an instance bound to it is made
        here; construct the class directly to reuse one across many coordinate
        sets.
        """
        import torch

        signal = grid if ops.is_torch(grid) else torch.as_tensor(ops.to_numpy(grid))
        n_chan = int(signal.shape[0])
        bound = type(self)(signal[:, None])                       # [C, 1, *grid]

        pts = torch.as_tensor(np.asarray(coords).T[None, None])   # [1, 1, ndim, K]
        pts = pts.expand(n_chan, 1, -1, -1).to(signal.real.dtype)

        return bound.forward(pts).reshape(n_chan, -1)

    #*******************#
    #   grid geometry   #
    #*******************#
    def _axis_view(self, d: int) -> torch.Tensor:
        """Axis *d* broadcast against the signal, i.e. "[1, 1, ..., n_d, ..., 1]"."""
        import torch
        shape = [1, 1] + [1] * self.NDIM
        shape[2 + d] = self.grid_shape[d]
        return self.axes[d].view(*shape)

    def _prepare_slopes(self, signal: torch.Tensor, axis: torch.Tensor,
                        dim: int) -> torch.Tensor:
        """
        Modified-Akima slopes of *signal* along *dim*.

        The axis is already broadcast to the signal's rank by "_axis_view",
        so it is differenced directly.
        """
        import torch
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
        """Cubic Hermite basis functions "h00, h10, h01, h11"."""
        import torch
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return h00, h10, h01, h11

    def _basis(self, t: torch.Tensor) -> torch.Tensor:
        """
        Hermite basis stacked to match the coefficient tensor's block layout.

        The basis functions come in interleaved order - value, derivative,
        value, derivative - while "P" is built in blocks: function values at
        "0:2" and scaled derivatives at "2:4". Stacking as
        "(h00, h01, h10, h11)" puts the value weights against the value block
        and the derivative weights against the derivative block.
        """
        import torch
        h00, h10, h01, h11 = self.h_poly(t)
        return torch.stack([h00, h01, h10, h11], dim=-1).to(self.dtype)

    def _locate(self, coord: torch.Tensor, d: int):
        """Cell index, local coordinate and cell width for *coord* on axis *d*."""
        import torch
        ax = self.axes[d]
        idx = (torch.searchsorted(ax, coord.clamp(ax.min(), ax.max())) - 1)
        idx = idx.clamp(0, self.grid_shape[d] - 2)
        width = self.cell[d].expand_as(idx)
        return idx, (coord - ax[idx]) / width, width

    def _neighbors(self, idx: torch.Tensor, d: int) -> torch.Tensor:
        """The four sample indices "i-1, i, i+1, i+2" clamped to axis *d*."""
        import torch
        return torch.stack([idx - 1, idx, idx + 1, idx + 2],
                           dim=-1).clamp(0, self.grid_shape[d] - 1)

    #*******************#
    #   interpolation   #
    #*******************#
    @abstractmethod
    def _interpolate(self, idx, t, width) -> torch.Tensor:
        """Contract the Hermite bases against the coefficient tensor. "[B, S, L]"."""
        import torch

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Interpolate the signal at *coords*.

        Args:
            coords: "[B, S, NDIM, L]", normalized to "[-1, 1]".

        Returns:
            "[B, S, 1, L]" interpolated values.
        """
        import torch
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

    def __call__(self, coords):
        """
        Interpolate at *coords*, in this class's bound-signal form.
        """
        return self.forward(coords)

    def _gather(self, neighbors):
        """Signal and slope sub-grids at the neighbor indices."""
        import torch
        b_idx = torch.arange(self.B, device=self.device, dtype=torch.long).view(
            self.B, *([1] * (1 + 1 + self.NDIM)))
        s_idx = torch.arange(1, device=self.device, dtype=torch.long).view(
            1, 1, *([1] * (1 + self.NDIM)))
        index = (b_idx, s_idx) + tuple(neighbors)
        return self.signal[index], [m[index] for m in self.slopes]


#**************************************************************************************************#
#                                   Class BicubicHermiteMAkima2D                                   #
#**************************************************************************************************#
#                                                                                                  #
# Bicubic Hermite modified-Akima interpolation on a 2-D grid "[B, S, Nx, Ny]".                     #
#                                                                                                  #
#**************************************************************************************************#
class BicubicHermiteMAkima2D(HermiteMAkimaInterpolator):
    """Bicubic Hermite modified-Akima interpolation on a 2-D grid "[B, S, Nx, Ny]"."""

    NDIM = 2

    def _interpolate(self, idx, t, width):
        import torch
        I_x, I_y = idx
        t_x, t_y = t
        dx_cell, dy_cell = width

        H_x = self._basis(t_x)
        H_y = self._basis(t_y)

        ix_n = self._neighbors(I_x, 0)
        iy_n = self._neighbors(I_y, 1)
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
# Tricubic Hermite modified-Akima interpolation on a 3-D grid "[B, S, Nx, Ny, Nz]".                #
#                                                                                                  #
#**************************************************************************************************#
class TricubicHermiteMAkima3D(HermiteMAkimaInterpolator):
    """Tricubic Hermite modified-Akima interpolation on a 3-D grid "[B, S, Nx, Ny, Nz]"."""

    NDIM = 3

    def _interpolate(self, idx, t, width):
        import torch
        I_x, I_y, I_z = idx
        t_x, t_y, t_z = t
        dx_cell, dy_cell, dz_cell = width

        H_x = self._basis(t_x)
        H_y = self._basis(t_y)
        H_z = self._basis(t_z)

        ix_n = self._neighbors(I_x, 0)
        iy_n = self._neighbors(I_y, 1)
        iz_n = self._neighbors(I_z, 2)
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
