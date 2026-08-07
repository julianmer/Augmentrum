####################################################################################################
#                                     kspace_reconstructor.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-22                                                                              #
#                                                                                                  #
# Purpose: Regrids non-Cartesian k-space samples back onto an image grid: density compensation     #
#          followed by an adjoint NUFFT. The adjoint half of the pair whose forward half is        #
#          processing.interpolating. Backend-agnostic: built on the gridding NUFFT.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from nifti_mrs_plus import ops
from typing import Any, Optional, Sequence, Tuple, Union



__all__ = ['KspaceReconstructor']


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
       most trajectories sample the centre of k-space far more densely than the
       edge,
    3. apply the adjoint NUFFT onto an oversampled grid.

    Shapes follow the rest of the package: coordinates are "[B, S, D, L]"
    (batch, shots, dimensions, samples per shot), k-space data is "[B, S, C, L]"
    with "C" coils, and the reconstruction is "[B, C, *image_size]".

    Coordinates must be normalised to "[-1, 1]"; "normalise_trajectory"
    converts the cycles-per-metre output of "kspace_sampling" for you.

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
        from augmentrum.sampling.gridding_nufft import GriddingNUFFT

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
    def normalise_trajectory(coords: Any,
                            k_max: Union[float, Sequence[float]]) -> Any:
        """
        Scale a trajectory from cycles per metre to the "[-1, 1]" box.

        Trajectories are generated in physical units, where the Nyquist edge sits
        at "k_max = 1 / (2 * voxel_size)". Both this module and
        "processing.interpolating" work in normalised coordinates, so this is
        the bridge between them.

        "k_max" is per axis whenever the voxels are anisotropic, which is how
        trajectory metadata reports it. A scalar is accepted and applied to every
        axis; a sequence must give one value per dimension of *coords*.

        Args:
            coords: trajectory coordinates in cycles per metre, "[B, S, D, L]".
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
            coords: normalised trajectory coordinates, "[B, S, D, L]".
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
            coords: normalised trajectory coordinates, "[B, S, D, L]".
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
