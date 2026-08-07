####################################################################################################
#                                       gridding_nufft.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-07                                                                              #
#                                                                                                  #
# Purpose: Non-uniform FFT by Kaiser-Bessel gridding, written on the backend-agnostic op           #
#          namespace so non-Cartesian sampling runs on NumPy, PyTorch, JAX or TensorFlow and       #
#          keeps its gradients. The forward maps an image to samples along a trajectory; the       #
#          adjoint grids samples back onto an image.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math
from typing import Sequence, Tuple

import numpy as np
from scipy.special import i0

from nifti_mrs_plus import ops


#***********************#
#   kaiser-bessel       #
#***********************#
# The kernel and its transform depend only on the grid geometry, never on the
# data, so they are built once in NumPy and promoted to the caller's backend.

def _beatty_beta(width: float, osf: float) -> float:
    """
    Kaiser-Bessel shape parameter, from Beatty et al. (2005).

    Chosen to minimise aliasing energy for a given kernel width and
    oversampling factor.
    """
    return math.pi * math.sqrt((width / osf) ** 2 * (osf - 0.5) ** 2 - 0.8)


def _kb_weights(offsets: np.ndarray, width: float, beta: float) -> np.ndarray:
    """Kaiser-Bessel kernel evaluated at *offsets* in grid units."""
    r = 2.0 * offsets / width
    inside = np.abs(r) <= 1.0
    arg = beta * np.sqrt(np.clip(1.0 - r ** 2, 0.0, None))
    return np.where(inside, i0(arg) / width, 0.0)


def _deapodization(n: int, grid: int, width: float, beta: float) -> np.ndarray:
    """
    Reciprocal of the kernel's transform over the image axis.

    Gridding convolves with the Kaiser-Bessel kernel, which tapers the image;
    dividing by the kernel's transform undoes exactly that taper.
    """
    x = (np.arange(n) - n // 2) / float(grid)
    arg = (math.pi * width * x) ** 2 - beta ** 2

    # Real above the root, imaginary below it, where sin becomes sinh
    out = np.empty_like(arg)
    positive = arg > 0
    root_pos = np.sqrt(np.clip(arg, 0.0, None))
    root_neg = np.sqrt(np.clip(-arg, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(positive,
                       np.sin(root_pos) / np.where(root_pos == 0, 1.0, root_pos),
                       np.sinh(root_neg) / np.where(root_neg == 0, 1.0, root_neg))
    out = np.where(root_pos + root_neg == 0, 1.0, out)
    return out


#**************************************************************************************************#
#                                       Class GriddingNUFFT                                        #
#**************************************************************************************************#
#                                                                                                  #
# Kaiser-Bessel gridding NUFFT, backend-agnostic and differentiable.                               #
#                                                                                                  #
#**************************************************************************************************#
class GriddingNUFFT:
    """
    Kaiser-Bessel gridding NUFFT, backend-agnostic and differentiable.

    Everything a non-Cartesian acquisition needs, in one place:

    "forward"
        image -> samples along a trajectory, by gridding.
    "forward_interp"
        image -> samples by interpolating an oversampled Cartesian grid. The
        cheap approximation, whose accuracy is set by the oversampling.
    "adjoint"
        samples -> image.
    "density_weights"
        how much each sample should count, given how densely the trajectory
        covers k-space.

    Both directions are a gather or scatter against a separable kernel, so they
    run wherever "nifti_mrs_plus.ops" runs and gradients reach the data.

    Trajectory coordinates are normalised to [-0.5, 0.5) per axis, the
    convention that maps directly onto grid bins.

    Args:
        im_size: Image matrix, "(nx, ny)" or "(nx, ny, nz)".
        osf: Grid oversampling factor.
        width: Kernel width in oversampled grid units.

    Examples:
        >>> nufft = GriddingNUFFT((16, 16))
        >>> nufft.grid_size
        (32, 32)
    """

    def __init__(self, im_size: Sequence[int], osf: float = 2.0, width: float = 4.0):
        self.im_size = tuple(int(n) for n in im_size)
        self.ndim = len(self.im_size)
        if self.ndim not in (2, 3):
            raise ValueError(f"GriddingNUFFT supports 2-D and 3-D, got {self.ndim}-D.")

        self.osf = float(osf)
        self.width = float(width)
        self.grid_size = tuple(int(math.ceil(self.osf * n)) for n in self.im_size)
        self.beta = _beatty_beta(self.width, self.osf)

        # Separable deapodization, one factor per axis, outer-multiplied
        factors = [_deapodization(n, g, self.width, self.beta)
                   for n, g in zip(self.im_size, self.grid_size)]
        deapod = factors[0]
        for f in factors[1:]:
            deapod = np.multiply.outer(deapod, f)
        self._deapod = deapod.astype(np.float32)

    #*******************#
    #   kernel table    #
    #*******************#

    def _neighbours(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grid bins touched by each sample, and the separable kernel weights.

        Returns "(indices, weights)" with shapes "[K, W**ndim]", where the flat
        index addresses the oversampled grid.
        """
        half = int(math.ceil(self.width / 2.0))
        taps = np.arange(-half, half + 1)

        per_axis_idx, per_axis_w = [], []
        for d in range(self.ndim):
            centre = (coords[:, d] + 0.5) * self.grid_size[d]      # bins
            base = np.floor(centre)
            idx = base[:, None] + taps[None, :]                    # [K, T]
            per_axis_w.append(_kb_weights(centre[:, None] - idx, self.width, self.beta))
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

    #***************#
    #   adjoint     #
    #***************#

    def adjoint(self, kdata, coords: np.ndarray):
        """
        Grid non-Cartesian samples back onto an image.

        Args:
            kdata: Samples, "[C, K]", on any backend.
            coords: Trajectory in [-0.5, 0.5), "[K, ndim]" (NumPy).

        Returns:
            Image "[C, *im_size]" on kdata's backend.
        """
        idx, weight = self._neighbours(np.asarray(coords))
        n_chan = int(ops.shape(kdata)[0])
        n_cells = int(np.prod(self.grid_size))

        weight_b = ops.cast_like(ops.match_backend(weight.reshape(-1), kdata), kdata)
        idx_flat = idx.reshape(-1)

        n_taps = idx.shape[1]
        planes = []
        for c in range(n_chan):
            # One sample feeds every tap it touches, so repeat it across them
            samples = ops.reshape(ops.take(kdata, np.array([c]), axis=0), (-1,))
            spread = ops.reshape(ops.stack([samples] * n_taps, axis=-1), (-1,))
            planes.append(ops.scatter_add((n_cells,), idx_flat, spread * weight_b))

        grid = ops.reshape(ops.stack(planes, axis=0), (n_chan,) + self.grid_size)

        axes = tuple(range(1, self.ndim + 1))
        image = ops.ifftn(ops.ifftshift(grid, axis=axes), axes, norm="ortho")
        image = ops.fftshift(image, axis=axes)

        image = self._crop(image)
        deapod = ops.cast_like(ops.match_backend(self._deapod, image), image)
        return image / deapod

    def _pad(self, image, grid_n=None):
        """Centre-pad the image up to *grid_n*, or to this object's grid."""
        grid_n = self.grid_size if grid_n is None else grid_n
        out = image
        for d, (n, g) in enumerate(zip(self.im_size, grid_n)):
            if g == n:
                continue
            before = (g - n) // 2
            width = [(0, 0)] * (self.ndim + 1)
            width[d + 1] = (before, g - n - before)
            out = ops.pad(out, width)
        return out

    def _crop(self, grid):
        """Centre-crop the oversampled grid back to the image matrix."""
        out = grid
        for d, (n, g) in enumerate(zip(self.im_size, self.grid_size)):
            start = (g - n) // 2
            index = np.arange(start, start + n)
            out = ops.take(out, index, axis=d + 1)
        return out

    #***************#
    #   forward     #
    #***************#

    def forward(self, image, coords: np.ndarray):
        """
        Sample an image along a trajectory.

        The adjoint's mirror: deapodize, transform to an oversampled grid, then
        gather each sample from the bins the kernel touches.

        Args:
            image: "[C, *im_size]", on any backend.
            coords: Trajectory in [-0.5, 0.5), "[K, ndim]" (NumPy).

        Returns:
            Samples "[C, K]" on image's backend.
        """
        idx, weight = self._neighbours(np.asarray(coords))
        n_chan = int(ops.shape(image)[0])
        n_taps = idx.shape[1]

        deapod = ops.cast_like(ops.match_backend(self._deapod, image), image)
        padded = self._pad(image / deapod)

        axes = tuple(range(1, self.ndim + 1))
        grid = ops.fftshift(
            ops.fftn(ops.ifftshift(padded, axis=axes), axes, norm="ortho"), axis=axes)
        flat = ops.reshape(grid, (n_chan, -1))

        weight_b = ops.cast_like(ops.match_backend(weight, flat), flat)
        rows = []
        for c in range(n_chan):
            plane = ops.reshape(ops.take(flat, np.array([c]), axis=0), (-1,))
            taps = ops.reshape(ops.take(plane, idx.reshape(-1), axis=0),
                               (len(coords), n_taps))
            rows.append(ops.sum(taps * weight_b, axis=-1))
        return ops.stack(rows, axis=0)

    #***********************#
    #   density weights     #
    #***********************#

    def density_weights(self, coords: np.ndarray, iterations: int = 15) -> np.ndarray:
        """
        Pipe-style density compensation weights for *coords*.

        Non-Cartesian trajectories visit the centre of k-space far more often
        than the edges, so an unweighted adjoint is dominated by the centre.
        The Pipe iteration drives the weights towards those that make gridding
        followed by de-gridding the identity.

        The weights depend only on the trajectory, never on the data, so this is
        computed in NumPy once and promoted wherever it is applied.

        Args:
            coords: Trajectory in [-0.5, 0.5), "[K, ndim]".
            iterations: Pipe iterations; converges quickly, 10-20 is plenty.

        Returns:
            Weights "[K]".
        """
        idx, weight = self._neighbours(np.asarray(coords))
        n_cells = int(np.prod(self.grid_size))
        flat_idx = idx.reshape(-1)

        w = np.ones(len(coords), dtype=np.float64)
        for _ in range(iterations):
            spread = np.repeat(w[:, None], idx.shape[1], axis=1) * weight
            grid = np.zeros(n_cells, dtype=np.float64)
            np.add.at(grid, flat_idx, spread.reshape(-1))
            back = (grid[flat_idx].reshape(idx.shape) * weight).sum(axis=1)
            w = w / np.maximum(back, 1e-12)

        return (w / w.max()).astype(np.float32)

    #***********************#
    #   interpolated form   #
    #***********************#

    def forward_interp(self, image, coords: np.ndarray, osf: float = None):
        """
        Sample by interpolating an oversampled Cartesian k-space grid.

        The cheap alternative to :meth:"forward": zero-pad, transform once, then
        read the trajectory off the grid by interpolation. Its accuracy is set by
        how far the grid is oversampled, which is what makes it worth measuring
        against the exact operator.

        Args:
            image: "[C, *im_size]", on any backend.
            coords: Trajectory in [-0.5, 0.5), "[K, ndim]" (NumPy).
            osf: Grid oversampling; defaults to this object's.

        Returns:
            Samples "[C, K]" on image's backend.
        """
        if self.ndim != 2:
            raise NotImplementedError(
                "forward_interp currently covers 2-D trajectories only."
            )

        osf = float(self.osf if osf is None else osf)
        grid_n = [int(round(osf * n)) for n in self.im_size]

        padded = self._pad(image, grid_n)
        grid = ops.fftshift(ops.fftn(ops.ifftshift(padded, axis=(1, 2)), (1, 2),
                                     norm="ortho"), axis=(1, 2))

        # grid_sample wants [N, C, H, W] and normalised [-1, 1] coordinates,
        # ordered x then y
        stack = ops.reshape(grid, (1,) + tuple(ops.shape(grid)))
        sample_grid = ops.match_backend(
            (2.0 * coords[None, None, :, ::-1]).astype(np.float32), ops.real(grid))

        real = ops.grid_sample(ops.real(stack), sample_grid, padding_mode="border")
        imag = ops.grid_sample(ops.imag(stack), sample_grid, padding_mode="border")
        out = ops.complex_from(real, imag)

        # An orthonormal transform scales with the grid it runs on. Referencing
        # this object's own grid rather than the image makes the result
        # independent of osf and puts it on the same scale as forward(), so an
        # accuracy-versus-oversampling comparison measures the interpolation
        # rather than the normalisation.
        growth = math.sqrt(float(np.prod(grid_n)) / float(np.prod(self.grid_size)))
        out = out * growth      # a Python float multiplies on any backend

        return ops.reshape(out, (int(ops.shape(image)[0]), len(coords)))
