####################################################################################################
#                                spatial_augmentations.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-11-28                                                                              #
#                                                                                                  #
# Purpose: Defines SpatialAugmentations, a modular class that samples and defines multiple         #
#          augmentations with parameters and prbabilities. It returns a list of partially defined  #
#          augmentation functions and a list of dictionaries describing the provenance for each    #
#          sample in the mini-batch.                                                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus
from augmentrum.utils.geometry import Affine


__all__ = ['SpatialAugmentations']


#**************************************************************************************************#
#                                    Class SpatialAugmentations                                    #
#**************************************************************************************************#
#                                                                                                  #
# Affine spatial augmentation (translation, rotation, zoom, shear, flip, anisotropic scaling) for  #
# volumetric MRS/MRSI data and coil sensitivity maps.                                              #
#                                                                                                  #
#**************************************************************************************************#
class SpatialAugmentations(BaseModule):
    """
    Affine spatial augmentation (translation, rotation, zoom, shear, flip,
    anisotropic scaling) for volumetric MRS/MRSI data and coil sensitivity maps.

    Data layout
    -----------
    Input and output follow the **NIfTI-MRS convention**: spatial axes first,
    the channel-like axis (spectral points, or coils for the CSM pipeline) last.

        dim=3 :  (batch, X, Y, Z, C)
        dim=2 :  (batch, X, Y, C)

    This is exactly what ``NIfTI_MRS_Plus.get_data()`` produces by stacking
    ``(X, Y, Z, T)`` NIfTI arrays along a new leading batch axis, so tensors flow
    through unchanged. Internally the batch is permuted to PyTorch's
    ``(N, C, D, H, W)`` image layout for ``grid_sample`` and permuted back; the
    permutation reverses axes 1..n-1 and is its own inverse.

    Under that mapping the affine's axes are (X, Y, Z) in order, so
    ``z_angle_deg`` rotates in the XY plane about Z and ``flip_z`` is a
    through-slice flip.

    All channels of a sample receive the *same* transform — the spatial
    augmentation of an MRSI volume must not vary with spectral point.

    Fully-singleton spatial axes (e.g. single-voxel spectroscopy, shaped
    ``(B, 1, 1, 1, N)``) are passed through untouched rather than resampled
    along the spectral axis.
    """

    # NIfTI_LIST and NUMPY inputs are converted to PyTorch internally.
    # TensorFlow, JAX, and Keras are NOT natively supported — the base-module
    # fallback will route them through the NIfTI-list path (implicit conversion).
    SUPPORTED_BACKENDS = [Backend.NIFTI_LIST, Backend.NUMPY, Backend.PYTORCH]

    def __init__(self,
                 dim: int = 3,
                 prob: float = 0.5,
                 device: Optional[torch.device] = None,
                 # data-pipeline ranges (sampleable as flat kwargs from Augmentrum)
                 translation_frac: float = 0.10,
                 max_z_angle_deg: float = 30.0,
                 max_random_angle_deg: float = 359.0,
                 zoom_min: float = 0.9,
                 zoom_max: float = 1.1,
                 shear_max: float = 0.15,
                 scale_min: float = 0.9,
                 scale_max: float = 1.1,
                 translation_prob: Optional[float] = None,
                 # csm-pipeline overrides (if None, fall back to the data-pipeline value)
                 csm_max_z_angle_deg: Optional[float] = 359.0,
                 # coil subsampling
                 min_coils: int = 6,
                 max_coils: Optional[int] = None,
                 # geometry / resampling
                 pixdim: Optional[Tuple[float, ...]] = None,
                 padding_mode: str = 'zeros',
                 chunk_channels: Optional[int] = 64,
                 allow_rot90: bool = True,
                 # additional bulk-override dicts (still supported for convenience)
                 data_ranges: Optional[Dict[str, Any]] = None,
                 csm_ranges: Optional[Dict[str, Any]] = None,
                 pipeline: str = 'data'):
        """
        dim: 2 or 3
        prob: Bernoulli probability to activate each augmentation
        device: torch device.  Leave as None to operate on tensors wherever they
                already live (important on GPU); set it to force a device.
        translation_frac: max translation as fraction of axis length
        max_z_angle_deg: max z-axis rotation for the data pipeline (degrees)
        max_random_angle_deg: max random rotation angle (degrees)
        zoom_min, zoom_max: bounds of the zoom factor range
        shear_max: max shear magnitude
        scale_min, scale_max: bounds of the anisotropic scale factor range
        translation_prob: probability of applying translation. None (default) uses
                `prob`, like every other augmentation. Setting it makes translation
                more or less likely than the rest; `prob=0` still disables
                everything, so a zero probability always means "identity".
        csm_max_z_angle_deg: max z-axis rotation for the CSM pipeline (degrees).
                             Defaults to 359 (unrestricted). Set to None to mirror
                             max_z_angle_deg.
        min_coils, max_coils: integer range of coils to keep when coil subsampling is active
        pixdim: voxel size per spatial axis, e.g. (2.8, 3.5, 4.0) in mm. When given,
                the affine is corrected for anisotropic voxels — see
                ``_correct_anisotropy``. Leave as None for isotropic data or when
                the axes are not physically comparable.
        padding_mode: how ``grid_sample`` fills coordinates that fall outside the
                volume. 'zeros' (default) is correct for a finite object; 'border'
                extrudes the edge voxels, which fabricates anatomy beyond a slab.
        chunk_channels: resample this many channels at a time to bound peak memory.
                None processes all at once. Only affects memory, not the result.
        allow_rot90: permit 90° in-plane rotations. Set False when the in-plane
                field of view is not square — a true 90° rotation then maps one
                axis onto a shorter one and content leaves the volume.
        data_ranges: optional dict to bulk-override any data-pipeline range key
        csm_ranges: optional dict to bulk-override any csm-pipeline range key
        pipeline: default pipeline to use ('data' or 'csm')

        Pipeline-specific ranges allow different augmentation constraints:
        - Data pipeline: typically has restricted z-axis rotation
        - CSM pipeline: allows unrestricted rotation (full 360°)

        Every scalar parameter can be passed as a (min, max) tuple to Augmentrum
        for per-batch sampling in on-the-fly mode — e.g. zoom_min=(0.85, 0.95)
        means the lower zoom bound is sampled uniformly from [0.85, 0.95] each batch.
        """
        super().__init__(dim=dim, prob=prob, min_coils=min_coils,
                         max_coils=max_coils, pipeline=pipeline)
        assert dim in (2, 3)
        self.dim = dim
        self.prob = float(prob)
        # Remember whether a device was actually requested. Defaulting self.device
        # to cpu and then unconditionally moving inputs would silently drag GPU
        # tensors back to the host.
        self._device_explicit = device is not None
        self.device = device or torch.device('cpu')
        self.pipeline = pipeline

        if pixdim is not None:
            pixdim = tuple(float(v) for v in pixdim)
            if len(pixdim) < dim:
                raise ValueError(f"pixdim must have at least {dim} entries, got {pixdim}")
        self.pixdim = pixdim
        self.padding_mode = padding_mode
        self.chunk_channels = chunk_channels
        self.allow_rot90 = bool(allow_rot90)
        self._warned_rot90 = False

        # Every attribute here is a plain scalar.  The pipeline can sample each one
        # independently — e.g. zoom_min=(0.85, 0.95) → sampled to 0.91 per-batch.
        self.translation_frac = translation_frac
        self.max_z_angle_deg = max_z_angle_deg
        self.max_random_angle_deg = max_random_angle_deg
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max
        self.shear_max = shear_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.translation_prob = translation_prob
        self.csm_max_z_angle_deg = csm_max_z_angle_deg if csm_max_z_angle_deg is not None else max_z_angle_deg

        self.min_coils = int(min_coils) if min_coils is not None else None
        self.max_coils = int(max_coils) if max_coils is not None else None

        # Bulk overrides merged in last so explicit kwargs always win
        self._extra_data_ranges = data_ranges or {}
        self._extra_csm_ranges = csm_ranges or {}

        # Populated after each forward pass — available for logging / reproducibility
        self.aug_specs_: Optional[List[Dict[str, Any]]] = None

    #*************************#
    #   range dict builders   #
    #*************************#
    @property
    def data_ranges(self) -> Dict[str, Any]:
        """Live view of data-pipeline ranges, rebuilt from the flat scalar attributes."""
        ranges = {
            'translation_frac': self.translation_frac,
            'max_z_angle_deg': self.max_z_angle_deg,
            'max_random_angle_deg': self.max_random_angle_deg,
            'zoom_range': (self.zoom_min, self.zoom_max),
            'shear_max': self.shear_max,
            'scale_range': (self.scale_min, self.scale_max),
            'translation_prob': self.translation_prob,
        }
        ranges.update(self._extra_data_ranges)
        return ranges

    @property
    def csm_ranges(self) -> Dict[str, Any]:
        """Live view of CSM-pipeline ranges, rebuilt from the flat scalar attributes."""
        ranges = {
            'translation_frac': self.translation_frac,
            'max_z_angle_deg': self.csm_max_z_angle_deg,
            'max_random_angle_deg': self.max_random_angle_deg,
            'zoom_range': (self.zoom_min, self.zoom_max),
            'shear_max': self.shear_max,
            'scale_range': (self.scale_min, self.scale_max),
            'translation_prob': self.translation_prob,
        }
        ranges.update(self._extra_csm_ranges)
        return ranges

    #***************************************#
    #   sampling augmentations per-sample   #
    #***************************************#
    def sample_augmentations(self, batch_size: int,
                           pipeline: str = 'data',
                           rng: Optional[torch.Generator] = None) -> List[Dict[str, Any]]:
        """
        Returns a list of augmentation specification dicts (one per sample).
        Each dict contains booleans 'do_*' keys and corresponding parameter keys.

        Args:
            batch_size: Number of samples to generate augmentations for
            pipeline: 'data' or 'csm' - determines which range configuration to use
            rng: Optional random number generator
        """
        ranges = self.csm_ranges if pipeline == 'csm' else self.data_ranges
        rng = rng or torch.default_generator
        info_list = []

        for _ in range(batch_size):
            def bern(p=None):
                # `prob` is the master switch: prob=0 means identity, whatever the
                # per-augmentation override says.
                if self.prob <= 0.0:
                    return False
                p = self.prob if p is None else float(p)
                return (torch.rand(1, generator=rng).item() < p)

            # decide per-augmentation activation
            do_translate = bern(ranges.get('translation_prob'))
            do_z_rot = bern()
            do_rot90 = bern() if self.allow_rot90 else False
            do_random_csm_rot = bern()
            do_zoom = bern()
            do_shear = bern()
            do_flip = bern()
            do_anisotropic = bern()
            do_coil_sub = bern()

            # sample magnitudes
            tx = 0.0
            ty = 0.0
            tz = 0.0
            if do_translate:
                frac = ranges['translation_frac']
                tx = (torch.rand(1, generator=rng).item() * 2 - 1) * frac
                ty = (torch.rand(1, generator=rng).item() * 2 - 1) * frac
                if self.dim == 3:
                    tz = (torch.rand(1, generator=rng).item() * 2 - 1) * frac

            # z-rotation angle
            z_angle = 0.0
            if do_z_rot:
                max_deg = ranges['max_z_angle_deg']
                z_angle = (torch.rand(1, generator=rng).item() * 2 - 1) * max_deg

            # random full rotation angle, plus the axis it turns about.  The axis
            # belongs in the spec: sampling it at apply() time would make a saved
            # spec un-replayable, so the same "reused" augmentation would rotate
            # the data one way and its label maps another.
            random_rot_deg = 0.0
            rot_axis = (0.0, 0.0, 1.0)
            if do_random_csm_rot:
                random_rot_deg = torch.rand(1, generator=rng).item() * ranges['max_random_angle_deg']
                if self.dim == 3:
                    ax = torch.randn(3, generator=rng)
                    rot_axis = tuple((ax / (ax.norm() + 1e-12)).tolist())

            # 90-degree rotations (k=1,2,3 for 90,180,270)
            k_rot90 = 0
            if do_rot90:
                k_rot90 = torch.randint(1, 4, (1,), generator=rng).item()

            # zoom factor
            zoom_factor = 1.0
            if do_zoom:
                lo, hi = ranges['zoom_range']
                zoom_factor = lo + (hi - lo) * torch.rand(1, generator=rng).item()

            # shear
            shx = 0.0
            shy = 0.0
            shz = 0.0
            if do_shear:
                mx = ranges['shear_max']
                shx = (torch.rand(1, generator=rng).item() * 2 - 1) * mx
                shy = (torch.rand(1, generator=rng).item() * 2 - 1) * mx
                if self.dim == 3:
                    shz = (torch.rand(1, generator=rng).item() * 2 - 1) * mx

            # flip
            flip_x = False
            flip_y = False
            flip_z = False
            if do_flip:
                flip_x = bool(torch.rand(1, generator=rng).item() < 0.5)
                flip_y = bool(torch.rand(1, generator=rng).item() < 0.5)
                if self.dim == 3:
                    flip_z = bool(torch.rand(1, generator=rng).item() < 0.5)

            # anisotropic scaling
            scale_x = 1.0
            scale_y = 1.0
            scale_z = 1.0
            if do_anisotropic:
                lo, hi = ranges['scale_range']
                scale_x = lo + (hi - lo) * torch.rand(1, generator=rng).item()
                scale_y = lo + (hi - lo) * torch.rand(1, generator=rng).item()
                if self.dim == 3:
                    scale_z = lo + (hi - lo) * torch.rand(1, generator=rng).item()

            # coil subsampling
            coil_keep = None
            if do_coil_sub and self.min_coils is not None:
                if self.max_coils is not None:
                    coil_keep = torch.randint(self.min_coils, self.max_coils + 1, (1,), generator=rng).item()
                else:
                    coil_keep = self.min_coils

            info = {
                'do_translate': do_translate,
                'tx': tx, 'ty': ty, 'tz': tz,
                'do_z_rot': do_z_rot,
                'z_angle_deg': z_angle,
                'do_rot90': do_rot90,
                'k_rot90': k_rot90,
                'do_random_csm_rot': do_random_csm_rot,
                'random_rot_deg': random_rot_deg,
                'rot_axis': rot_axis,
                'do_zoom': do_zoom,
                'zoom_factor': zoom_factor,
                'do_shear': do_shear,
                'shear_xy': (shx, shy),
                'shear_z': shz,
                'do_flip': do_flip,
                'flip_x': flip_x, 'flip_y': flip_y, 'flip_z': flip_z,
                'do_anisotropic': do_anisotropic,
                'scale_xyz': (scale_x, scale_y, scale_z),
                'do_coil_sub': do_coil_sub,
                'coil_keep': coil_keep,
                'pipeline': pipeline,  # Track which pipeline was used
            }
            info_list.append(info)
        return info_list

    #************************#
    #   affine composition   #
    #************************#
    def _compose_affines_for_batch(self, aug_spec_list: List[Dict[str, Any]],
                                   data_shape: torch.Size) -> torch.Tensor:
        """
        Build affine matrices for an entire batch given augmentation specs.
        Returns (N, 2, 3) or (N, 3, 4) depending on dim.
        """
        batch_size = len(aug_spec_list)
        thetas = []
        for spec in aug_spec_list:
            if self.dim == 2:
                # 2D
                rot_rad = math.radians(spec['z_angle_deg']) if spec['do_z_rot'] else 0.0
                # incorporate 90-deg rotation
                if spec['do_rot90']:
                    rot_rad += spec['k_rot90'] * math.pi / 2.0
                # random rotation
                if spec['do_random_csm_rot']:
                    rot_rad += math.radians(spec['random_rot_deg'])
                sx, sy, _ = spec['scale_xyz']
                sx *= spec['zoom_factor']
                sy *= spec['zoom_factor']
                shx, shy = spec['shear_xy']
                tx = spec['tx']
                ty = spec['ty']
                fx = spec['flip_x']
                fy = spec['flip_y']
                theta = Affine.build_2d(rot_rad, (sx, sy), (shx, shy), tx, ty, fx, fy)
                thetas.append(theta)
            else:
                # 3D
                # Build rotation matrix
                R = torch.eye(3, dtype=torch.float32)
                # z-axis rotation
                if spec['do_z_rot']:
                    z_rad = math.radians(spec['z_angle_deg'])
                    q_z = Affine.quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0]), z_rad, device=None)
                    R = Affine.quat_to_rotmat(q_z)
                # random rotation — axis comes from the spec so the transform is
                # exactly reproducible from it (see sample_augmentations)
                if spec['do_random_csm_rot']:
                    rand_rad = math.radians(spec['random_rot_deg'])
                    axis_rand = torch.tensor(spec.get('rot_axis', (0.0, 0.0, 1.0)),
                                             dtype=torch.float32)
                    axis_rand = axis_rand / (axis_rand.norm() + 1e-12)
                    q_rand = Affine.quat_from_axis_angle(axis_rand, rand_rad, device=None)
                    R_rand = Affine.quat_to_rotmat(q_rand)
                    R = R @ R_rand
                # 90-degree rotations around z
                if spec['do_rot90']:
                    k = spec['k_rot90']
                    angle_90 = k * math.pi / 2.0
                    q90 = Affine.quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0]), angle_90, device=None)
                    R90 = Affine.quat_to_rotmat(q90)
                    R = R @ R90

                # shear
                shx, shy = spec['shear_xy']
                shz = spec['shear_z']
                SH = torch.tensor([
                    [1.0, shx, 0.0],
                    [shy, 1.0, 0.0],
                    [0.0, shz, 1.0]
                ], dtype=torch.float32)
                # scale
                sx, sy, sz = spec['scale_xyz']
                sx *= spec['zoom_factor']
                sy *= spec['zoom_factor']
                sz *= spec['zoom_factor']
                tx = spec['tx']
                ty = spec['ty']
                tz = spec['tz']
                fx = spec['flip_x']
                fy = spec['flip_y']
                fz = spec['flip_z']
                theta = Affine.build_3d(R, (sx, sy, sz), SH, (tx, ty, tz), fx, fy, fz)
                thetas.append(theta)
        # Stack
        theta_batch = torch.stack(thetas, dim=0)  # Nx2x3 or Nx3x4

        if self.pixdim is not None and data_shape is not None:
            theta_batch = self._correct_anisotropy(theta_batch, data_shape, aug_spec_list)
        return theta_batch

    def _correct_anisotropy(self, theta: torch.Tensor, data_shape,
                            aug_spec_list: Optional[List[Dict[str, Any]]] = None) -> torch.Tensor:
        """
        Rewrite the affine so rotations are physical when voxels are anisotropic.

        ``grid_sample`` works in coordinates normalised to [-1, 1] *per axis*, so a
        rotation matrix applied there is conjugated by the wrong metric whenever
        the field of view is not isotropic.  For a 179.2 x 224.0 mm FOV, a nominal
        30 degree rotation comes out as a rotation composed with a 1.25x stretch —
        a shear, not a rotation, and the anatomy is visibly skewed.

        With physical coordinates ``p = D n`` where ``D = diag(L_i / 2)`` and
        ``L_i = N_i * pixdim_i``, requiring ``p_in = R p_out`` gives
        ``n_in = D^-1 R D n_out``.  So the linear block is conjugated by
        ``S = diag(L_i)`` — the factor of 2 cancels.  The translation column is
        left alone: ``translation_frac`` already means "fraction of axis length",
        which is the per-axis semantics we want.
        """
        d = self.dim
        lengths = torch.tensor(
            [float(data_shape[1 + i]) * float(self.pixdim[i]) for i in range(d)],
            dtype=theta.dtype, device=theta.device,
        )

        if (not self._warned_rot90) and self.allow_rot90 and d >= 2 \
                and abs(float(lengths[0]) - float(lengths[1])) > 1e-6 \
                and aug_spec_list is not None \
                and any(s.get('do_rot90') and s.get('k_rot90', 0) % 2 == 1 for s in aug_spec_list):
            self._warned_rot90 = True
            import warnings as _warnings
            _warnings.warn(
                f"90 degree rotation requested on a non-square field of view "
                f"({float(lengths[0]):.1f} x {float(lengths[1]):.1f} mm): the rotated "
                f"content cannot fit and will be cropped. Pass allow_rot90=False.",
                RuntimeWarning, stacklevel=3,
            )

        S = torch.diag(lengths)
        S_inv = torch.diag(1.0 / lengths)
        linear = S_inv @ theta[:, :, :d] @ S
        return torch.cat([linear, theta[:, :, d:]], dim=2)

    @staticmethod
    def _axis_perm(ndim: int) -> Tuple[int, ...]:
        """
        Permutation between the NIfTI layout ``(B, X, Y, [Z,] C)`` and the
        ``grid_sample`` layout ``(B, C, [Z,] Y, X)``.

        It reverses axes 1..ndim-1, so it is its own inverse and the same tuple
        converts in both directions.
        """
        return (0,) + tuple(range(ndim - 1, 0, -1))

    def _apply_affine_batch(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        x: NIfTI layout — (N, X, Y, Z, C) for dim=3, (N, X, Y, C) for dim=2.
        theta: (N, 2, 3) or (N, 3, 4)
        returns the transformed tensor in the same layout.
        """
        perm = self._axis_perm(x.dim())
        xg = x.permute(*perm)                       # (N, C, [Z,] Y, X)
        n_batch, n_chan = xg.shape[0], xg.shape[1]
        spatial = tuple(xg.shape[2:])

        real_dtype = xg.real.dtype if torch.is_complex(xg) else xg.dtype
        if real_dtype not in (torch.float32, torch.float64):
            real_dtype = torch.float32

        # The sampling grid depends only on (theta, spatial) — never on C — so it
        # is built once and reused for every chunk and for both the real and the
        # imaginary pass.
        grid = F.affine_grid(
            theta.to(device=xg.device, dtype=real_dtype),
            (n_batch, 1) + spatial,
            align_corners=False,
        )

        chunk = int(self.chunk_channels) if self.chunk_channels else n_chan
        chunk = max(1, min(chunk, n_chan))

        outs = []
        for c0 in range(0, n_chan, chunk):
            block = xg[:, c0:c0 + chunk]
            if torch.is_complex(block):
                outs.append(torch.complex(
                    self._grid_sample(block.real.to(real_dtype), grid),
                    self._grid_sample(block.imag.to(real_dtype), grid),
                ))
            else:
                outs.append(self._grid_sample(block.to(real_dtype), grid))

        return torch.cat(outs, dim=1).permute(*perm)

    def _grid_sample(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(x.contiguous(), grid, mode='bilinear',
                             padding_mode=self.padding_mode, align_corners=False)

    #******************#
    #   coil sampler   #
    #******************#
    def default_coil_sampler(self, x: torch.Tensor, keep: int, coils_axis: int = -1) -> torch.Tensor:
        """
        Subsample coil maps along the coil axis (last axis in the NIfTI layout).
        Keep 'keep' coils selected uniformly at random.
        """
        if keep is None:
            return x
        C = x.shape[coils_axis]
        keep = min(max(1, int(keep)), C)
        idx = torch.randperm(C, device=x.device)[:keep]
        return torch.index_select(x, dim=coils_axis, index=idx)

    #***************************#
    #   public apply function   #
    #***************************#
    def apply(self, x: torch.Tensor,
             pipeline: str = 'data',
             aug_spec_list: Optional[List[Dict[str, Any]]] = None,
             coils_axis: int = -1) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Apply augmentations to input batch x.

        Args:
            x: Input tensor in the NIfTI-MRS layout — (bS, X, Y, Z, C) for dim=3
               or (bS, X, Y, C) for dim=2. The channel-like axis (spectral points,
               or coils for the CSM pipeline) is LAST.
            pipeline: 'data' or 'csm' - determines augmentation ranges if sampling new augs
            aug_spec_list: Optional pre-specified augment list (length bS). If None, will sample.
            coils_axis: Axis along which coils are stored (for coil subsampling)

        Returns:
            (x_aug, aug_info_list): Augmented tensor and list of augmentation specifications
        """
        expected = self.dim + 2
        if x.dim() != expected:
            raise ValueError(
                f"SpatialAugmentations(dim={self.dim}) expects a {expected}-D tensor in the "
                f"NIfTI layout "
                f"{'(batch, X, Y, Z, channels)' if self.dim == 3 else '(batch, X, Y, channels)'}, "
                f"got shape {tuple(x.shape)}."
            )
        batch_size = x.shape[0]
        is_csm = (pipeline == 'csm')

        if aug_spec_list is None:
            aug_spec_list = self.sample_augmentations(batch_size, pipeline=pipeline)

        # Single-voxel spectroscopy has no spatial extent to transform.  Resampling
        # a (B, 1, 1, 1, N) volume would interpolate along the spectral axis, which
        # is meaningless, so pass it through untouched.
        if all(int(s) == 1 for s in x.shape[1:1 + self.dim]):
            self.aug_specs_ = aug_spec_list
            return x, aug_spec_list

        # Prepare theta batch
        theta = self._compose_affines_for_batch(aug_spec_list, x.shape)

        # Skip resampling entirely when nothing was drawn.  Beyond saving the work,
        # this keeps "no augmentation" bit-exact: grid_sample through an identity
        # affine still blends neighbours at the 1e-6 level because the normalised
        # sampling coordinates are not exactly on voxel centres in float32.
        identity = torch.eye(self.dim, self.dim + 1,
                             dtype=theta.dtype, device=theta.device)
        if torch.allclose(theta, identity.expand_as(theta), atol=1e-7, rtol=0.0):
            x_aug = x
        else:
            # Apply affine to entire batch at once
            x_aug = self._apply_affine_batch(x, theta)

        # If coil subsampling requested and is CSM pipeline, do it per-sample
        if is_csm:
            out_list = []
            for i, aug in enumerate(aug_spec_list):
                sample = x_aug[i:i+1]  # keep batch dim
                if aug['do_coil_sub'] and aug['coil_keep'] is not None:
                    sample = self.default_coil_sampler(sample, keep=aug['coil_keep'], coils_axis=coils_axis)
                out_list.append(sample)
            # pad/stack back into a tensor: if coil counts differ we will return a list instead
            chans = [s.shape[coils_axis] for s in out_list]
            if all(c == chans[0] for c in chans):
                x_aug = torch.cat(out_list, dim=0)
            else:
                # Return as list in this unusual case
                x_aug = out_list

        return x_aug, aug_spec_list

    #******************************#
    #   nifti conversion helpers   #
    #******************************#
    def _nifti_to_tensor(self, nifti_list: List[NIfTI_MRS_Plus]) -> torch.Tensor:
        """
        Convert list of NIfTI_MRS_Plus objects to tensor format.

        Returns:
            torch.Tensor: Shape [batch_size, channels, ...spatial dims]
        """
        tensors = []
        for nifti in nifti_list:
            # Extract data from NIFTI object
            data = nifti[:]  # Assuming this returns the data array

            # Convert to torch tensor
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            else:
                tensor = data

            # Ensure proper dtype
            if torch.is_complex(tensor):
                tensor = tensor.to(dtype=torch.complex64)
            else:
                tensor = tensor.to(dtype=torch.float32)

            tensors.append(tensor)

        # Stack into batch
        batch_tensor = torch.stack(tensors, dim=0)
        return batch_tensor.to(self.device)

    def _tensor_to_nifti(self, tensor: torch.Tensor,
                        original_nifti_list: List[NIfTI_MRS_Plus]) -> List[NIfTI_MRS_Plus]:
        """
        Convert tensor back to list of NIfTI_MRS_Plus objects.

        Args:
            tensor: Augmented data tensor
            original_nifti_list: Original NIFTI objects to preserve metadata

        Returns:
            List of new NIfTI_MRS_Plus objects with augmented data
        """
        result_list = []

        # Handle case where tensor is actually a list (due to varying coil counts)
        if isinstance(tensor, list):
            tensor_list = tensor
        else:
            # Split batch dimension
            tensor_list = [tensor[i] for i in range(tensor.shape[0])]

        for i, (data_tensor, original_nifti) in enumerate(zip(tensor_list, original_nifti_list)):
            # Remove batch dimension if present
            if data_tensor.dim() > len(original_nifti.shape):
                data_tensor = data_tensor.squeeze(0)

            # Convert to numpy
            if data_tensor.is_cuda:
                data_tensor = data_tensor.cpu()
            data_array = data_tensor.numpy()

            # Create new NIFTI object with augmented data
            nifti_new = original_nifti.copy()
            nifti_new[:] = data_array

            result_list.append(nifti_new)

        return result_list

    #**************************#
    #   basemodule interface   #
    #**************************#
    def process_nifti_list(self,
                          data_list: List[NIfTI_MRS_Plus],
                          water_list: Optional[List[NIfTI_MRS_Plus]] = None,
                          **kwargs) -> Tuple[List[NIfTI_MRS_Plus], Optional[List[NIfTI_MRS_Plus]]]:
        """
        Process list of NIFTI_MRS objects with spatial augmentations.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects (passed through unchanged)
            **kwargs: pipeline, aug_spec_list forwarded to apply()

        Returns:
            Tuple of (augmented_data_list, water_list_unchanged)
            Augmentation specs are stored in self.aug_specs_ after each call.
        """
        pipeline = kwargs.get('pipeline', self.pipeline)
        aug_spec_list = kwargs.get('aug_spec_list', None)

        # Convert NIFTI to tensor
        data_tensor = self._nifti_to_tensor(data_list)

        # Apply augmentations
        data_aug_tensor, aug_list = self.apply(
            data_tensor,
            pipeline=pipeline,
            aug_spec_list=aug_spec_list,
            coils_axis=-1
        )

        self.aug_specs_ = aug_list  # store for provenance / reproducibility

        # Convert back to NIFTI
        augmented_data_list = self._tensor_to_nifti(data_aug_tensor, data_list)

        # Water references unchanged
        return augmented_data_list, water_list

    def process_tensor(self,
                      data_array,
                      water_array=None,
                      backend: Backend = Backend.NUMPY,
                      **kwargs) -> Tuple:
        """
        Process tensor data with spatial augmentations.

        Args:
            data_array: Input tensor or array in the NIfTI layout —
                        (batch, X, Y, Z, channels) for dim=3,
                        (batch, X, Y, channels) for dim=2.
            water_array: Optional water reference (passed through unchanged)
            backend: Backend enum (unused, kept for BaseModule signature compatibility)
            **kwargs: pipeline, aug_spec_list forwarded to apply()

        Returns:
            Tuple of (augmented_data, water_unchanged)
            Augmentation specs are stored in self.aug_specs_ after each call.
        """
        pipeline = kwargs.get('pipeline', self.pipeline)
        aug_spec_list = kwargs.get('aug_spec_list', None)

        # Convert to torch tensor if needed
        return_numpy = False
        if isinstance(data_array, np.ndarray):
            return_numpy = True
            data_array = torch.from_numpy(data_array)

        # Ensure proper dtype
        if torch.is_complex(data_array):
            data_array = data_array.to(dtype=torch.complex64)
        else:
            data_array = data_array.to(dtype=torch.float32)

        # Only relocate when a device was explicitly requested — otherwise leave
        # the tensor where the caller put it (moving it would drag GPU tensors
        # back to the host, since self.device defaults to cpu).
        if self._device_explicit:
            data_array = data_array.to(self.device)

        # Apply augmentations
        data_aug, aug_list = self.apply(
            data_array,
            pipeline=pipeline,
            aug_spec_list=aug_spec_list,
            coils_axis=-1
        )

        self.aug_specs_ = aug_list  # store for provenance / reproducibility

        # Convert back to numpy if needed
        if return_numpy:
            if isinstance(data_aug, list):
                # Handle list of tensors (variable coil counts)
                data_aug = [d.cpu().numpy() for d in data_aug]
            else:
                data_aug = data_aug.cpu().numpy()

        return data_aug, water_array

    #**********************#
    #   pipeline routing   #
    #**********************#
    def route_pipeline(self,
                       data_list: Optional[Union[List[NIfTI_MRS_Plus], torch.Tensor, np.ndarray]] = None,
                       water_list: Optional[Union[List[NIfTI_MRS_Plus], torch.Tensor, np.ndarray]] = None,
                       csm_list: Optional[Union[List[NIfTI_MRS_Plus], torch.Tensor, np.ndarray]] = None,
                       data_aug_list: Optional[List[Dict[str, Any]]] = None,
                       csm_aug_list: Optional[List[Dict[str, Any]]] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Route data through appropriate pipelines based on data type.
        Automatically detects whether data is NIFTI_MRS or tensor/array format.

        Args:
            data_list: Spectroscopy data (NIFTI list or tensor/array)
            water_list: Optional water reference (NIFTI list or tensor/array)
            csm_list: Coil sensitivity maps (NIFTI list or tensor/array)
            data_aug_list: Optional pre-computed augmentations for data
            csm_aug_list: Optional pre-computed augmentations for CSMs
            **kwargs: Additional arguments passed to processing methods

        Returns:
            Dictionary containing processed data and augmentation lists:
            {
                'data': processed data,
                'water': processed water,
                'csm': processed CSM,
                'data_augmentations': augmentation specs used for data,
                'csm_augmentations': augmentation specs used for CSMs
            }
        """
        results = {}

        def _is_nifti_list(x):
            return isinstance(x, list) and len(x) > 0 and isinstance(x[0], NIfTI_MRS_Plus)

        # Process data pipeline
        if data_list is not None:
            if _is_nifti_list(data_list):
                data_proc, water_proc = self.process_nifti_list(
                    data_list, water_list, pipeline='data', aug_spec_list=data_aug_list, **kwargs)
            else:
                data_proc, water_proc = self.process_tensor(
                    data_list, water_list, pipeline='data', aug_spec_list=data_aug_list, **kwargs)
            results['data'] = data_proc
            results['water'] = water_proc
            results['data_augmentations'] = self.aug_specs_

        # Process CSM pipeline
        if csm_list is not None:
            if _is_nifti_list(csm_list):
                csm_proc, _ = self.process_nifti_list(
                    csm_list, None, pipeline='csm', aug_spec_list=csm_aug_list, **kwargs)
            else:
                csm_proc, _ = self.process_tensor(
                    csm_list, None, pipeline='csm', aug_spec_list=csm_aug_list, **kwargs)
            results['csm'] = csm_proc
            results['csm_augmentations'] = self.aug_specs_

        return results


#*********************#
#   usage / testing   #
#*********************#
if __name__ == "__main__":
    # All tensors below are in the NIfTI-MRS layout: spatial axes first,
    # channel-like axis (spectral points / coils) LAST.
    device = torch.device('cpu')

    print("=" * 80)
    print("Testing SpatialAugmentations with dual pipelines (NIfTI layout)")
    print("=" * 80)

    # Test 1: 2D data with data pipeline (restricted rotation)
    print("\n1. Testing 2D data pipeline (restricted rotation):")
    bS = 2
    img2d = torch.randn(bS, 64, 64, 1, dtype=torch.float32, device=device) \
            + 1j * torch.randn(bS, 64, 64, 1, dtype=torch.float32, device=device)

    aug = SpatialAugmentations(dim=2, prob=0.6, device=device, min_coils=6, max_coils=12)
    x2d_aug, info2d = aug.apply(img2d, pipeline='data')
    print(f"   (batch, X, Y, C): {tuple(img2d.shape)} -> {tuple(x2d_aug.shape)}")
    print(f"   Z-rotation (restricted): {info2d[0]['z_angle_deg']:.2f}°")
    print(f"   Pipeline: {info2d[0]['pipeline']}")

    # Test 2: 3D CSM with CSM pipeline (unrestricted rotation)
    print("\n2. Testing 3D CSM pipeline (unrestricted rotation, coils last):")
    csm3d = torch.randn(1, 64, 64, 16, 8, dtype=torch.float32, device=device) \
            + 1j * torch.randn(1, 64, 64, 16, 8, dtype=torch.float32, device=device)

    aug3d = SpatialAugmentations(dim=3, prob=0.7, device=device, min_coils=6, max_coils=8)
    csm3d_aug, info3d = aug3d.apply(csm3d, pipeline='csm')
    print(f"   (batch, X, Y, Z, coils): {tuple(csm3d.shape)} -> {tuple(csm3d_aug.shape)}")
    print(f"   Coil subsampling: {info3d[0]['do_coil_sub']}, keep={info3d[0]['coil_keep']}")
    print(f"   Pipeline: {info3d[0]['pipeline']}")

    # Test 3: replaying a stored spec must reproduce the transform exactly.
    print("\n3. Testing spec replay (data and labels must transform identically):")
    vol = torch.randn(2, 32, 32, 8, 4, dtype=torch.complex64, device=device)
    specs = aug3d.sample_augmentations(batch_size=2, pipeline='data')
    out_a, _ = aug3d.apply(vol, pipeline='data', aug_spec_list=specs)
    out_b, _ = aug3d.apply(vol, pipeline='data', aug_spec_list=specs)
    print(f"   Replay is bit-identical: {torch.equal(out_a, out_b)}")

    # Test 4: MRSI volume — 384 spectral points ride along as channels
    print("\n4. Testing an MRSI volume (spectral axis last):")
    mrsi = torch.randn(1, 32, 32, 8, 384, dtype=torch.complex64, device=device)
    mrsi_aug, _ = aug3d.apply(mrsi, pipeline='data')
    print(f"   (batch, X, Y, Z, T): {tuple(mrsi.shape)} -> {tuple(mrsi_aug.shape)}")

    # Test 5: SVS is passed through untouched
    print("\n5. Testing SVS passthrough:")
    svs = torch.randn(3, 1, 1, 1, 2048, dtype=torch.complex64, device=device)
    svs_aug, _ = aug3d.apply(svs, pipeline='data')
    print(f"   {tuple(svs.shape)} unchanged: {torch.equal(svs, svs_aug)}")

    # Test 6: anisotropic voxels
    print("\n6. Testing anisotropic voxel correction:")
    aug_aniso = SpatialAugmentations(dim=3, prob=1.0, pixdim=(2.8, 3.5, 4.0),
                                     allow_rot90=False, device=device)
    out_aniso, _ = aug_aniso.apply(vol, pipeline='data')
    print(f"   pixdim={aug_aniso.pixdim} -> {tuple(out_aniso.shape)}")

    # Test 7: process_tensor + route_pipeline
    print("\n7. Testing process_tensor and route_pipeline:")
    data_np = np.random.randn(2, 32, 32, 32, 1).astype(np.float32)
    data_proc, _ = aug3d.process_tensor(data_np, pipeline='data')
    print(f"   numpy in -> {type(data_proc).__name__} out, shape {data_proc.shape}")

    results = aug3d.route_pipeline(
        data_list=torch.randn(2, 32, 32, 32, 1, dtype=torch.float32),
        csm_list=torch.randn(2, 32, 32, 32, 8, dtype=torch.complex64),
    )
    print(f"   Processed data shape: {tuple(results['data'].shape)}")
    print(f"   Data augmentations: {len(results['data_augmentations'])} specs")
    print(f"   CSM augmentations: {len(results['csm_augmentations'])} specs")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
