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
#          processing.interpolating. Torch only; needs the optional torchkbnufft dependency.       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Optional, Sequence, Tuple, Union

import torch


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

    Reconstruction is the three steps below, which :meth:`__call__` runs in order
    and which are also usable individually:

    1. zero out the shots an undersampling mask discards,
    2. weight the survivors by a Pipe-style density compensation function, since
       most trajectories sample the centre of k-space far more densely than the
       edge,
    3. apply the adjoint NUFFT onto an oversampled grid.

    Shapes follow the rest of the package: coordinates are ``[B, S, D, L]``
    (batch, shots, dimensions, samples per shot), k-space data is ``[B, S, C, L]``
    with ``C`` coils, and the reconstruction is ``[B, C, *image_size]``.

    Coordinates must be normalised to ``[-1, 1]``; :meth:`normalise_trajectory`
    converts the cycles-per-metre output of ``kspace_sampling`` for you.

    Args:
        image_size: target volume shape, 2- or 3-D.
        oversampling_factor: NUFFT grid oversampling, scalar or per axis.
        device: torch device for the NUFFT operators; defaults to the data's.

    Note:
        Requires the optional ``torchkbnufft`` dependency, imported on first use
        so that importing this module never fails on its account.

    Note:
        Deliberately NOT a :class:`BaseModule`. A module is a pipeline stage,
        called as ``module(nifti_plus, water)`` and returning a dataset of the
        same shape; this is an operator over k-space samples, called as
        ``(coords, kdata, mask)`` and returning an image. Inheriting would mean
        implementing ``process_tensor(data_array, water_array, ...)`` and
        smuggling the trajectory in through ``**kwargs``, which would satisfy
        the pipeline's isinstance check while remaining unusable in a pipeline.
        It sits with ``KspaceGeometry``, ``GridMask`` and ``KspaceSampler`` —
        building blocks that modules are assembled from. The module that reaches
        for this one is ``KspaceUndersampling``.
    """

    def __init__(self, image_size: Sequence[int],
                 oversampling_factor: Union[float, Sequence[float]] = 1.5,
                 device: Optional[torch.device] = None):
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

    @staticmethod
    def _tkbn():
        """The torchkbnufft module, with an actionable error when it is absent."""
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
    def normalise_trajectory(coords: torch.Tensor,
                            k_max: Union[float, Sequence[float]]) -> torch.Tensor:
        """
        Scale a trajectory from cycles per metre to the ``[-1, 1]`` box.

        Trajectories are generated in physical units, where the Nyquist edge sits
        at ``k_max = 1 / (2 * voxel_size)``. Both this module and
        ``processing.interpolating`` work in normalised coordinates, so this is
        the bridge between them.

        ``k_max`` is per axis whenever the voxels are anisotropic, which is how
        trajectory metadata reports it. A scalar is accepted and applied to every
        axis; a sequence must give one value per dimension of *coords*.

        Args:
            coords: trajectory coordinates in cycles per metre, ``[B, S, D, L]``.
            k_max: k-space half-extent, scalar or one value per axis. Trajectory
                metadata carries it as ``meta['kmax']``.
        """
        ndim = coords.shape[2]
        values = [float(k_max)] * ndim if isinstance(k_max, (int, float)) \
            else [float(k) for k in k_max]
        if len(values) != ndim:
            raise ValueError(
                f"k_max must be a scalar or one value per axis; got {len(values)} "
                f"for {ndim}-D coordinates."
            )
        if any(k <= 0 for k in values):
            raise ValueError(f"k_max values must be positive, got {values}.")

        scale = torch.tensor(values, dtype=coords.dtype, device=coords.device)
        return coords / scale.view(1, 1, ndim, 1)

    @staticmethod
    def flatten(coords: torch.Tensor,
                kdata: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Collapse the shot axis into the sample axis, as torchkbnufft expects.

        Coordinates go ``[B, S, D, L] -> [B, D, S * L]`` and, when given, k-space
        data goes ``[B, S, C, L] -> [B, C, S * L]``. Coordinates are also scaled
        from ``[-1, 1]`` to the ``[-pi, pi]`` convention the NUFFT uses.
        """
        if coords.dim() != 4:
            raise ValueError(f"coords must be [B, S, D, L], got {tuple(coords.shape)}.")

        B, S, D, L = coords.shape
        k_traj = (coords * torch.pi).permute(0, 2, 1, 3).reshape(B, D, S * L)

        kdata_flat = None
        if kdata is not None:
            if kdata.dim() != 4:
                raise ValueError(
                    f"kdata must be [B, S, C, L], got {tuple(kdata.shape)}."
                )
            if kdata.shape[0] != B or kdata.shape[1] != S or kdata.shape[3] != L:
                raise ValueError(
                    f"kdata {tuple(kdata.shape)} does not match coords "
                    f"{tuple(coords.shape)} on the batch, shot or sample axis."
                )
            C = kdata.shape[2]
            kdata_flat = kdata.permute(0, 2, 1, 3).reshape(B, C, S * L)

        return k_traj, kdata_flat

    #********************************#
    #   density compensation (dcf)   #
    #********************************#

    def density_weights(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Pipe-style iterative density compensation weights, shaped ``[B, S, L]``.

        Computed from the *full* trajectory rather than the retained shots: the
        weights describe how densely the trajectory was designed to cover k-space,
        which undersampling does not change.
        """
        B, S, _, L = coords.shape
        k_traj, _ = self.flatten(coords)

        with torch.no_grad():
            dcf = self._tkbn().calc_density_compensation_function(
                k_traj, im_size=self.image_size, grid_size=self.grid_size,
            ).detach()

        return dcf.view(B, S, L)

    #**************************#
    #   regridding (adjoint)   #
    #**************************#

    def adjoint(self, kdata: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Adjoint NUFFT of *kdata* along *coords*, giving ``[B, C, *image_size]``.

        Args:
            kdata: density-compensated k-space samples, ``[B, S, C, L]``.
            coords: normalised trajectory coordinates, ``[B, S, D, L]``.
        """
        ndim = coords.shape[2]
        if ndim != len(self.image_size):
            raise ValueError(
                f"coords are {ndim}-D but image_size is {len(self.image_size)}-D."
            )

        k_traj, kdata_flat = self.flatten(coords, kdata)
        device = self.device or kdata.device

        adjoint = self._tkbn().KbNufftAdjoint(
            im_size=self.image_size, grid_size=self.grid_size,
        ).to(device)

        return adjoint(kdata_flat.to(device), k_traj.to(device))

    #********************#
    #   reconstruction   #
    #********************#

    def __call__(self, coords: torch.Tensor, kdata: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Mask, density-compensate and regrid in one call.

        Args:
            coords: normalised trajectory coordinates, ``[B, S, D, L]``.
            kdata: k-space samples, ``[B, S, C, L]``.
            mask: boolean shots to keep, ``[B, S]``; all shots when omitted.

        Returns:
            Complex reconstruction, ``[B, C, *image_size]``.
        """
        if mask is not None:
            if mask.shape != kdata.shape[:2]:
                raise ValueError(
                    f"mask must be [B, S] = {tuple(kdata.shape[:2])}, got "
                    f"{tuple(mask.shape)}."
                )
            keep = mask.to(dtype=kdata.real.dtype, device=kdata.device)
            kdata = kdata * keep.unsqueeze(2).unsqueeze(3)

        weighted = kdata * self.density_weights(coords).unsqueeze(2)
        return self.adjoint(weighted, coords)
