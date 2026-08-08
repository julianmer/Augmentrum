####################################################################################################
#                                     kspace_reconstructor.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-22                                                                              #
#                                                                                                  #
# Purpose: Non-Cartesian k-space reconstruction. Holds the gridding NUFFT itself - forward,        #
#          adjoint and density compensation - and the batched adapter that drives it from the      #
#          shot-organized layout the pipeline uses. Backend-agnostic throughout; the grid is       #
#          read by an interpolator from processing.interpolating.                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math

import numpy as np

from nifti_mrs_plus import ops
from augmentrum.processing.interpolating import (
    GriddingKernel, Interpolator, KaiserBesselInterpolator,
)
from typing import Any, Optional, Sequence, Tuple, Union



__all__ = ['KspaceReconstructor']


#**************************************************************************************************#
#                                       Class GriddingNUFFT                                        #
#**************************************************************************************************#
#                                                                                                  #
# Non-uniform FFT by gridding, backend-agnostic and differentiable.                                #
#                                                                                                  #
#**************************************************************************************************#
class GriddingNUFFT:
    """
    Non-uniform FFT by gridding, backend-agnostic and differentiable.

    Everything a non-Cartesian acquisition needs:

    "forward"
        image -> samples along a trajectory.
    "adjoint"
        samples -> image.
    "density_weights"
        how much each sample should count, given how densely the trajectory
        covers k-space.

    The transform itself is only a pad, an FFT and a crop. What distinguishes
    one NUFFT from another is the interpolator that reads the trajectory off the
    grid, so that is supplied rather than built in: Kaiser-Bessel by default,
    or a cheaper one when the trade is worth making.

    Both directions are a gather or scatter against the kernel, so they run
    wherever "nifti_mrs_plus.ops" runs and gradients reach the data.

    Trajectory coordinates are normalized to [-0.5, 0.5) per axis, the
    convention that maps directly onto grid bins.

    Args:
        im_size: Image matrix, "(nx, ny)" or "(nx, ny, nz)".
        osf: Grid oversampling factor.
        width: Kaiser-Bessel kernel width, when the default interpolator is used.
        interpolator: Interpolator to read the grid with. Defaults to
            Kaiser-Bessel at this object's oversampling.

    Examples:
        >>> nufft = GriddingNUFFT((16, 16))
        >>> nufft.grid_size
        (32, 32)
    """

    def __init__(self, im_size: Sequence[int], osf: float = 2.0, width: float = 4.0,
                 interpolator: Optional[Interpolator] = None):
        self.im_size = tuple(int(n) for n in im_size)
        self.ndim = len(self.im_size)
        if self.ndim not in (2, 3):
            raise ValueError(f"GriddingNUFFT supports 2-D and 3-D, got {self.ndim}-D.")

        self.osf = float(osf)
        self.width = float(width)
        self.grid_size = tuple(int(math.ceil(self.osf * n)) for n in self.im_size)

        self.interpolator = interpolator or KaiserBesselInterpolator(
            self.grid_size, osf=self.osf, width=self.width)

        # Gridding tapers the image by the kernel's transform. Only a kernel
        # that tapers knows how to undo it, so this is asked of the interpolator
        # rather than assumed.
        self._deapod = (self.interpolator.deapodization(self.im_size)
                        if isinstance(self.interpolator, GriddingKernel) else None)

    #*************#
    #   adjoint   #
    #*************#
    def adjoint(self, kdata, coords: np.ndarray):
        """
        Grid non-Cartesian samples back onto an image.

        Args:
            kdata: Samples, "[C, K]", on any backend.
            coords: Trajectory in [-0.5, 0.5), "[K, ndim]" (NumPy).

        Returns:
            Image "[C, *im_size]" on kdata's backend.
        """
        if not isinstance(self.interpolator, GriddingKernel):
            raise TypeError(
                f"{type(self.interpolator).__name__} can read the grid but not write to "
                f"it, so it cannot run the adjoint. Use a GriddingKernel."
            )

        grid = self.interpolator.spread(kdata, np.asarray(coords))

        axes = tuple(range(1, self.ndim + 1))
        image = ops.ifftn(ops.ifftshift(grid, axis=axes), axes, norm="ortho")
        image = ops.fftshift(image, axis=axes)

        return self._deapodize(self._crop(image))

    #*************#
    #   forward   #
    #*************#
    def forward(self, image, coords: np.ndarray):
        """
        Sample an image along a trajectory.

        The adjoint's mirror: correct for the kernel, transform to an
        oversampled grid, then read the trajectory off it.

        Args:
            image: "[C, *im_size]", on any backend.
            coords: Trajectory in [-0.5, 0.5), "[K, ndim]" (NumPy).

        Returns:
            Samples "[C, K]" on image's backend.
        """
        padded = self._pad(self._deapodize(image))

        axes = tuple(range(1, self.ndim + 1))
        grid = ops.fftshift(
            ops.fftn(ops.ifftshift(padded, axis=axes), axes, norm="ortho"), axis=axes)

        return self.interpolator.sample(grid, self._for_interpolator(coords))

    def _for_interpolator(self, coords: np.ndarray) -> np.ndarray:
        """
        Coordinates in the convention this object's interpolator expects.

        A gridding kernel addresses grid bins directly and so takes [-0.5, 0.5);
        every other interpolator works in the [-1, 1] box that
        :class:"Interpolator" specifies.
        """
        coords = np.asarray(coords)
        if isinstance(self.interpolator, GriddingKernel):
            return coords
        return 2.0 * coords

    #*******************#
    #   grid geometry   #
    #*******************#
    def _deapodize(self, image):
        """Undo the kernel's image-domain taper, if it has one."""
        if self._deapod is None:
            return image
        deapod = ops.cast_like(ops.match_backend(self._deapod, image), image)
        return image / deapod

    def _pad(self, image, grid_n=None):
        """Center-pad the image up to *grid_n*, or to this object's grid."""
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
        """Center-crop the oversampled grid back to the image matrix."""
        out = grid
        for d, (n, g) in enumerate(zip(self.im_size, self.grid_size)):
            start = (g - n) // 2
            index = np.arange(start, start + n)
            out = ops.take(out, index, axis=d + 1)
        return out

    #*********************#
    #   density weights   #
    #*********************#
    def density_weights(self, coords: np.ndarray, iterations: int = 15) -> np.ndarray:
        """
        Pipe-style density compensation weights for *coords*.

        Non-Cartesian trajectories visit the center of k-space far more often
        than the edges, so an unweighted adjoint is dominated by the center.
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
        kernel = self.interpolator
        if not isinstance(kernel, GriddingKernel):
            # Density is a property of the trajectory and the grid, not of
            # whichever interpolator happens to be reading it.
            kernel = KaiserBesselInterpolator(self.grid_size, osf=self.osf, width=self.width)

        idx, weight = kernel.neighbors(np.asarray(coords))
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


#**************************************************************************************************#
#                                    Class KspaceReconstructor                                     #
#**************************************************************************************************#
#                                                                                                  #
# Regrid non-Cartesian k-space samples onto an image grid.                                         #
#                                                                                                  #
#**************************************************************************************************#
class KspaceReconstructor:
    """
    Regrid non-Cartesian k-space samples onto an image grid.

    Reconstruction is the three steps below, which "__call__" runs in order
    and which are also usable individually:

    1. zero out the shots an undersampling mask discards,
    2. weight the survivors by a Pipe-style density compensation function, since
       most trajectories sample the center of k-space far more densely than the
       edge,
    3. apply the adjoint NUFFT onto an oversampled grid.

    Shapes follow the rest of the package: coordinates are "[B, S, D, L]"
    (batch, shots, dimensions, samples per shot), k-space data is "[B, S, C, L]"
    with "C" coils, and the reconstruction is "[B, C, *image_size]".

    Coordinates must be normalized to "[-1, 1]"; "normalize_trajectory"
    converts the cycles-per-meter output of "kspace_sampling" for you.

    Args:
        image_size: target volume shape, 2- or 3-D.
        oversampling_factor: NUFFT grid oversampling, scalar or per axis.
        device: unused; results stay on the data's own backend.

    Note:
        Runs on any backend "nifti_mrs_plus.ops" supports
        so that importing this module never fails on its account.

    Note:
        Deliberately NOT a "BaseModule". A module is a pipeline stage,
        called as "module(nifti_plus, water)" and returning a dataset of the
        same shape; this is an operator over k-space samples, called as
        "(coords, kdata, mask)" and returning an image. Inheriting would mean
        implementing "process_tensor(data_array, water_array, ...)" and
        smuggling the trajectory in through "**kwargs", which would satisfy
        the pipeline's isinstance check while remaining unusable in a pipeline.
        It sits with "KspaceGeometry", "GridMask" and "KspaceSampler" —
        building blocks that modules are assembled from. The module that reaches
        for this one is "KspaceUndersampling".
    """

    def __init__(self, image_size: Sequence[int],
                 oversampling_factor: Union[float, Sequence[float]] = 1.5,
                 device=None):
        self.image_size = [int(n) for n in image_size]
        if len(self.image_size) not in (2, 3):
            raise ValueError(
                f"image_size must be 2- or 3-D, got {len(self.image_size)} axes."
            )

        if isinstance(oversampling_factor, (int, float)):
            factors = [float(oversampling_factor)] * len(self.image_size)
        else:
            factors = [float(f) for f in oversampling_factor]
        if len(factors) != len(self.image_size):
            raise ValueError(
                f"oversampling_factor must be a scalar or one value per axis; got "
                f"{len(factors)} for a {len(self.image_size)}-D image."
            )
        if any(f < 1.0 for f in factors):
            raise ValueError(f"oversampling factors must be >= 1, got {factors}.")

        self.oversampling_factor = factors
        self.grid_size = [int(n * f) for n, f in zip(self.image_size, factors)]
        self.device = device

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(image_size={self.image_size}, "
                f"oversampling_factor={self.oversampling_factor})")

    #*************#
    #   backend   #
    #*************#
    def _gridder(self):
        """The backend-agnostic NUFFT this reconstructor is built on."""
        osf = max(self.oversampling_factor)
        return GriddingNUFFT(self.image_size, osf=osf)

    @staticmethod
    def _tkbn():
        """
        The torchkbnufft module, with an actionable error when it is absent.

        This reconstructor no longer needs it; it remains for the comparison
        path in KspaceUndersampling(nufft_impl='torchkbnufft').
        """
        try:
            import torchkbnufft as tkbn
        except ImportError as exc:                                # pragma: no cover
            raise ImportError(
                "KspaceReconstructor needs the optional dependency torchkbnufft. "
                "Install it with `pip install torchkbnufft`."
            ) from exc
        return tkbn

    #*****************#
    #   coordinates   #
    #*****************#
    @staticmethod
    def normalize_trajectory(coords: Any,
                            k_max: Union[float, Sequence[float]]) -> Any:
        """
        Scale a trajectory from cycles per meter to the "[-1, 1]" box.

        Trajectories are generated in physical units, where the Nyquist edge sits
        at "k_max = 1 / (2 * voxel_size)". Both this module and
        "processing.interpolating" work in normalized coordinates, so this is
        the bridge between them.

        "k_max" is per axis whenever the voxels are anisotropic, which is how
        trajectory metadata reports it. A scalar is accepted and applied to every
        axis; a sequence must give one value per dimension of *coords*.

        Args:
            coords: trajectory coordinates in cycles per meter, "[B, S, D, L]".
            k_max: k-space half-extent, scalar or one value per axis. Trajectory
                metadata carries it as "meta['kmax']".
        """
        ndim = ops.shape(coords)[2]
        values = [float(k_max)] * ndim if isinstance(k_max, (int, float)) \
            else [float(k) for k in k_max]
        if len(values) != ndim:
            raise ValueError(
                f"k_max must be a scalar or one value per axis; got {len(values)} "
                f"for {ndim}-D coordinates."
            )
        if any(k <= 0 for k in values):
            raise ValueError(f"k_max values must be positive, got {values}.")

        scale = ops.match_backend(
            np.asarray(values, dtype=np.float32).reshape(1, 1, ndim, 1), coords)
        return coords / ops.cast_like(scale, coords)

    @staticmethod
    def flatten(coords: Any,
                kdata: Optional[Any] = None
                ) -> Tuple[Any, Optional[Any]]:
        """
        Collapse the shot axis into the sample axis.

        Coordinates go "[B, S, D, L] -> [B, D, S * L]" and, when given, k-space
        data goes "[B, S, C, L] -> [B, C, S * L]". Coordinates are also scaled
        from "[-1, 1]" to the "[-pi, pi]" convention the NUFFT uses.
        """
        if len(ops.shape(coords)) != 4:
            raise ValueError(f"coords must be [B, S, D, L], got {tuple(coords.shape)}.")

        B, S, D, L = ops.shape(coords)
        k_traj = ops.reshape(ops.transpose(coords * np.pi, (0, 2, 1, 3)), (B, D, S * L))

        kdata_flat = None
        if kdata is not None:
            if len(ops.shape(kdata)) != 4:
                raise ValueError(
                    f"kdata must be [B, S, C, L], got {tuple(kdata.shape)}."
                )
            if (ops.shape(kdata)[0] != B or ops.shape(kdata)[1] != S
                    or ops.shape(kdata)[3] != L):
                raise ValueError(
                    f"kdata {tuple(kdata.shape)} does not match coords "
                    f"{tuple(coords.shape)} on the batch, shot or sample axis."
                )
            C = ops.shape(kdata)[2]
            kdata_flat = ops.reshape(ops.transpose(kdata, (0, 2, 1, 3)), (B, C, S * L))

        return k_traj, kdata_flat

    #********************************#
    #   density compensation (dcf)   #
    #********************************#
    def density_weights(self, coords: Any) -> Any:
        """
        Pipe-style iterative density compensation weights, shaped "[B, S, L]".

        Computed from the *full* trajectory rather than the retained shots: the
        weights describe how densely the trajectory was designed to cover k-space,
        which undersampling does not change.
        """
        B, S, _, L = ops.shape(coords)

        # The weights depend only on the trajectory, so they are computed once
        # in NumPy from the first batch element and broadcast.
        flat = ops.to_numpy(self.flatten(coords)[0])[0].T / (2.0 * np.pi)
        weights = self._gridder().density_weights(flat.astype(np.float32))

        return ops.reshape(ops.match_backend(weights, coords), (1, S, L)) \
            + ops.cast_like(ops.full_like_shape(coords, (B, S, L), 0.0), coords)

    #**************************#
    #   regridding (adjoint)   #
    #**************************#
    def adjoint(self, kdata: Any, coords: Any) -> Any:
        """
        Adjoint NUFFT of *kdata* along *coords*, giving "[B, C, *image_size]".

        Args:
            kdata: density-compensated k-space samples, "[B, S, C, L]".
            coords: normalized trajectory coordinates, "[B, S, D, L]".
        """
        ndim = ops.shape(coords)[2]
        if ndim != len(self.image_size):
            raise ValueError(
                f"coords are {ndim}-D but image_size is {len(self.image_size)}-D."
            )

        k_traj, kdata_flat = self.flatten(coords, kdata)
        gridder = self._gridder()

        # One trajectory per batch element; coordinates in [-0.5, 0.5)
        planes = []
        for b in range(ops.shape(kdata_flat)[0]):
            coords_b = ops.to_numpy(k_traj)[b].T / (2.0 * np.pi)
            samples = ops.reshape(
                ops.take(kdata_flat, np.array([b]), axis=0),
                ops.shape(kdata_flat)[1:])
            planes.append(gridder.adjoint(samples, coords_b.astype(np.float32)))

        return ops.stack(planes, axis=0)

    #********************#
    #   reconstruction   #
    #********************#
    def __call__(self, coords: Any, kdata: Any,
                 mask: Optional[Any] = None) -> Any:
        """
        Mask, density-compensate and regrid in one call.

        Args:
            coords: normalized trajectory coordinates, "[B, S, D, L]".
            kdata: k-space samples, "[B, S, C, L]".
            mask: boolean shots to keep, "[B, S]"; all shots when omitted.

        Returns:
            Complex reconstruction, "[B, C, *image_size]".
        """
        if mask is not None:
            if tuple(ops.shape(mask)) != tuple(ops.shape(kdata))[:2]:
                raise ValueError(
                    f"mask must be [B, S] = {tuple(ops.shape(kdata))[:2]}, got "
                    f"{tuple(ops.shape(mask))}."
                )
            keep = ops.reshape(ops.cast_like(mask, ops.real(kdata)),
                               ops.shape(mask) + (1, 1))
            kdata = kdata * ops.cast_like(keep, kdata)

        weights = self.density_weights(coords)
        weighted = kdata * ops.cast_like(
            ops.reshape(weights, ops.shape(weights)[:2] + (1,) + ops.shape(weights)[2:]),
            kdata)
        return self.adjoint(weighted, coords)
