####################################################################################################
#                                spatial_augmentations.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2025-11-28                                                                              #
#                                                                                                  #
# Purpose: Defines SpatialAugmentations, a modular class that samples and defines multiple         #
#          augmentations with parameters and prbabilities. It returns a list of partially defined  #
#          augmentation functions and a list of dictionaries describing the provenance for each    #
#          sample in the mini-batch.                                                               #
#                                                                                                  #
####################################################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, List, Dict, Any


# ----------------------
# Quaternion utilities
# ----------------------
def quat_from_axis_angle(axis: torch.Tensor, angle_rad: float, device=None) -> torch.Tensor:
    axis = axis.to(device) if device is not None else axis
    axis = axis / (axis.norm() + 1e-12)
    s = math.sin(angle_rad / 2.0)
    w = math.cos(angle_rad / 2.0)
    x, y, z = (axis * s).tolist()
    return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Hamilton product q = q1 * q2
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z])


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    # q: [4] tensor [w, x, y, z]
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    R = torch.stack([
        torch.stack([ww + xx - yy - zz, 2 * (xy - wz),       2 * (xz + wy)]),
        torch.stack([2 * (xy + wz),       ww - xx + yy - zz, 2 * (yz - wx)]),
        torch.stack([2 * (xz - wy),       2 * (yz + wx),     ww - xx - yy + zz])
    ])  # 3x3
    return R


# ----------------------
# Builder helpers
# ----------------------
def build_affine_2d_from_components(rotation_rad: float,
                                    scale_xy: Tuple[float, float],
                                    shear_xy: Tuple[float, float],
                                    tx: float,
                                    ty: float,
                                    flip_x: bool,
                                    flip_y: bool) -> torch.Tensor:
    """
    Build a 2x3 affine matrix that maps normalized output coords to normalized input coords.
    flips are implemented as -scale on axis when True.
    tx,ty are normalized translations (-1..1).
    """
    # rotation
    c = math.cos(rotation_rad)
    s = math.sin(rotation_rad)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    # shear matrix
    shx, shy = shear_xy
    SH = torch.tensor([[1.0, shx], [shy, 1.0]], dtype=torch.float32)
    sx = -scale_xy[0] if flip_x else scale_xy[0]
    sy = -scale_xy[1] if flip_y else scale_xy[1]
    Sc = torch.diag(torch.tensor([sx, sy], dtype=torch.float32))
    A = R @ SH @ Sc
    T = torch.tensor([tx, ty], dtype=torch.float32).unsqueeze(1)
    mat = torch.cat([A, T], dim=1)  # 2x3
    return mat


def build_affine_3d_from_components(R: torch.Tensor,
                                    scale_xyz: Tuple[float, float, float],
                                    shear_mat: Optional[torch.Tensor],
                                    t_xyz: Tuple[float, float, float],
                                    flip_x: bool, flip_y: bool, flip_z: bool) -> torch.Tensor:
    """
    R: 3x3 rotation matrix (torch)
    scale_xyz: (sx, sy, sz)
    shear_mat: 3x3 shear matrix (diagonal should be 1)
    t_xyz: translations normalized
    flips: convert to negative scale when True
    returns 3x4 affine
    """
    sx, sy, sz = scale_xyz
    sx = -sx if flip_x else sx
    sy = -sy if flip_y else sy
    sz = -sz if flip_z else sz
    Sc = torch.diag(torch.tensor([sx, sy, sz], dtype=torch.float32))
    SH = shear_mat if shear_mat is not None else torch.eye(3, dtype=torch.float32)
    linear = R @ SH @ Sc
    T = torch.tensor(t_xyz, dtype=torch.float32).unsqueeze(1)
    mat = torch.cat([linear, T], dim=1)  # 3x4
    return mat


# ----------------------
# Module
# ----------------------
class SpatialAugmentations(nn.Module):
    def __init__(self,
                 dim: int = 3,
                 prob: float = 0.5,
                 device: Optional[torch.device] = None,
                 ranges: Optional[Dict[str, Any]] = None,
                 min_coils: int = 6,
                 max_coils: Optional[int] = None):
        """
        dim: 2 or 3
        prob: Bernoulli probability to activate each augmentation
        device: torch device
        ranges: optional dict to override default ranges:
            'translation_frac', 'max_z_angle_deg', 'max_random_angle_deg',
            'zoom_range', 'shear_max', 'scale_range'
        min_coils, max_coils: integer range of coils to keep when coil subsampling is active
        """
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.prob = float(prob)
        self.device = device or torch.device('cpu')
        defaults = {
            'translation_frac': 0.10,        # fraction of axis length
            'max_z_angle_deg': 30.0,
            'max_random_angle_deg': 359.0,
            'zoom_range': (0.9, 1.1),
            'shear_max': 0.15,
            'scale_range': (0.9, 1.1),
            'translation_prob': 0.1,
        }
        if ranges:
            defaults.update(ranges)
        self.ranges = defaults
        self.min_coils = int(min_coils) if min_coils is not None else None
        self.max_coils = int(max_coils) if max_coils is not None else None

    # ----------------------
    # Sampling augmentations per-sample
    # ----------------------
    def sample_augmentations(self, batch_size: int, rng: Optional[torch.Generator] = None) -> List[Dict[str, Any]]:
        """
        Returns a list of augmentation specification dicts (one per sample).
        Each dict contains booleans 'do_*' keys and corresponding parameter keys.
        """
        rng = rng or torch.default_generator
        info_list = []
        for _ in range(batch_size):
            def bern():
                return (torch.rand(1, generator=rng).item() < self.prob)

            # decide per-augmentation activation
            do_translate = bern()
            do_z_rot = bern()
            do_rot90 = bern()
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
                frac = self.ranges['translation_frac']
                tx = (torch.rand(1, generator=rng).item() * 2 - 1) * frac
                ty = (torch.rand(1, generator=rng).item() * 2 - 1) * frac
                if self.dim == 3:
                    tz = (torch.rand(1, generator=rng).item() * 2 - 1) * frac

            z_angle = 0.0
            if do_z_rot:
                z_angle = (torch.rand(1, generator=rng).item() * 2 - 1) * self.ranges['max_z_angle_deg']

            # 90-degree multiples
            k90 = 0
            if do_rot90:
                k90 = int(torch.randint(0, 4, (1,), generator=rng).item())

            rnd_csm_angle = 0.0
            rnd_csm_axis = torch.tensor([0.0, 0.0, 1.0])
            if do_random_csm_rot and self.dim == 3:
                rnd_csm_angle = (torch.rand(1, generator=rng).item() * 2 - 1) * self.ranges['max_random_angle_deg']
                # sample random axis uniformly on sphere (simple method)
                u = torch.rand(1, generator=rng).item() * 2 - 1
                phi = torch.rand(1, generator=rng).item() * 2 * math.pi
                x = math.sqrt(max(0.0, 1 - u * u)) * math.cos(phi)
                y = math.sqrt(max(0.0, 1 - u * u)) * math.sin(phi)
                z = u
                rnd_csm_axis = torch.tensor([x, y, z], dtype=torch.float32)

            zoom = 1.0
            if do_zoom:
                zmin, zmax = self.ranges['zoom_range']
                zoom = float(zmin + (zmax - zmin) * torch.rand(1, generator=rng).item())

            shear_x = shear_y = shear_z = 0.0
            if do_shear:
                smax = self.ranges['shear_max']
                shear_x = (torch.rand(1, generator=rng).item() * 2 - 1) * smax
                shear_y = (torch.rand(1, generator=rng).item() * 2 - 1) * smax
                if self.dim == 3:
                    shear_z = (torch.rand(1, generator=rng).item() * 2 - 1) * smax

            flip_x = bool(torch.rand(1, generator=rng).item() < 0.5) if do_flip else False
            flip_y = bool(torch.rand(1, generator=rng).item() < 0.5) if do_flip else False
            flip_z = bool(torch.rand(1, generator=rng).item() < 0.5) if (do_flip and self.dim == 3) else False

            sx = sy = sz = 1.0
            if do_anisotropic:
                smin, smax = self.ranges['scale_range']
                sx = float(smin + (smax - smin) * torch.rand(1, generator=rng).item())
                sy = float(smin + (smax - smin) * torch.rand(1, generator=rng).item())
                if self.dim == 3:
                    sz = float(smin + (smax - smin) * torch.rand(1, generator=rng).item())

            coil_keep = None
            if do_coil_sub:
                if self.max_coils is not None:
                    # pick random integer in [min_coils, max_coils]
                    low = max(self.min_coils or 0, 1)
                    high = max(self.max_coils, low)
                    coil_keep = int(torch.randint(low, high + 1, (1,), generator=rng).item())
                else:
                    coil_keep = int(self.min_coils)

            # Compose augmentation dict
            aug = dict(
                do_translate=do_translate, tx=tx, ty=ty, tz=tz,
                do_z_rot=do_z_rot, z_angle_deg=z_angle,
                do_rot90=do_rot90, k90=k90,
                do_random_csm_rot=do_random_csm_rot, rnd_csm_angle_deg=rnd_csm_angle, rnd_csm_axis=rnd_csm_axis,
                do_zoom=do_zoom, zoom=zoom,
                do_shear=do_shear, shear_x=shear_x, shear_y=shear_y, shear_z=shear_z,
                do_flip=do_flip, flip_x=flip_x, flip_y=flip_y, flip_z=flip_z,
                do_anisotropic=do_anisotropic, sx=sx, sy=sy, sz=sz,
                do_coil_sub=do_coil_sub, coil_keep=coil_keep
            )
            info_list.append(aug)
        return info_list

    # ----------------------
    # Compose affine and apply to complex tensors
    # ----------------------
    def _compose_affines_for_batch(self, aug_list: List[Dict[str, Any]], tensor_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        For batch of augment dicts and target tensor shape build batch theta (N x 2 x 3) or (N x 3 x 4)
        that maps output coords to input coords (to be passed to affine_grid).
        tensor_shape expected to be shape of x (N, C, ...) where ... = X,Y(,Z)
        """
        N = len(aug_list)
        # need spatial sizes
        if self.dim == 2:
            # expect x shape (N, C, H, W)
            _, _, H, W = tensor_shape
            thetas = []
            for aug in aug_list:
                # rotation from quaternion or k90
                # We'll build a combined rotation angle in-plane = z_angle + 90*k
                angle_deg = 0.0
                if aug['do_z_rot']:
                    angle_deg += aug['z_angle_deg']
                if aug['do_rot90']:
                    angle_deg += 90.0 * aug['k90']
                angle_rad = math.radians(angle_deg)
                # compose scale (anisotropic * zoom)
                sx = aug['sx'] if aug['do_anisotropic'] else 1.0
                sy = aug['sy'] if aug['do_anisotropic'] else 1.0
                if aug['do_zoom']:
                    sx *= aug['zoom']
                    sy *= aug['zoom']
                # flips -> negative scale
                flip_x = aug['flip_x']
                flip_y = aug['flip_y']
                # shear
                shx = aug['shear_x'] if aug['do_shear'] else 0.0
                shy = aug['shear_y'] if aug['do_shear'] else 0.0
                # translation normalized - we sampled tx as fraction of axis length; leave as-is
                tx = aug['tx'] if aug['do_translate'] else 0.0
                ty = aug['ty'] if aug['do_translate'] else 0.0
                # Build 2x3 matrix
                A = build_affine_2d_from_components(rotation_rad=angle_rad,
                                                   scale_xy=(1.0 / sx, 1.0 / sy),
                                                   shear_xy=(shx, shy),
                                                   tx=tx, ty=ty,
                                                   flip_x=flip_x, flip_y=flip_y)
                # Note: here scale_x is 1/sx because we want to map output coords -> input coords such that
                # zoom >1 should sample smaller input area => using 1/zoom in linear part.
                thetas.append(A.to(self.device))
            theta_batch = torch.stack(thetas, dim=0)  # Nx2x3
            return theta_batch
        else:
            # 3D case: x shape (N, C, D, H, W)
            _, _, D, H, W = tensor_shape
            thetas = []
            for aug in aug_list:
                # Compose quaternion from multiple rotations:
                # Start with identity quaternion
                q_total = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                # if z rotation
                if aug['do_z_rot']:
                    qz = quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0], device=self.device),
                                              math.radians(aug['z_angle_deg']), device=self.device)
                    q_total = quat_mul(qz.to(self.device), q_total)
                # if rot90 (around z axis)
                if aug['do_rot90'] and (aug['k90'] % 4 != 0):
                    angle90 = math.radians(90.0 * aug['k90'])
                    q90 = quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0], device=self.device), angle90, device=self.device)
                    q_total = quat_mul(q90.to(self.device), q_total)
                # random csm rotation
                if aug['do_random_csm_rot']:
                    qrc = quat_from_axis_angle(aug['rnd_csm_axis'].to(self.device), math.radians(aug['rnd_csm_angle_deg']), device=self.device)
                    q_total = quat_mul(qrc.to(self.device), q_total)
                # obtain rotation matrix
                R = quat_to_rotmat(q_total).to(self.device)
                # scales: anisotropic * zoom
                sx = aug['sx'] if aug['do_anisotropic'] else 1.0
                sy = aug['sy'] if aug['do_anisotropic'] else 1.0
                sz = aug['sz'] if aug['do_anisotropic'] else 1.0
                if aug['do_zoom']:
                    sx *= aug['zoom']
                    sy *= aug['zoom']
                    sz *= aug['zoom']
                # shear matrix
                SH = torch.eye(3, dtype=torch.float32, device=self.device)
                if aug['do_shear']:
                    SH[0, 1] = aug['shear_x']
                    SH[0, 2] = aug['shear_y']
                    SH[1, 2] = aug['shear_z']
                    # keep diagonal ones
                # flips as negative scale
                flip_x = aug['flip_x']
                flip_y = aug['flip_y']
                flip_z = aug['flip_z']
                # translation normalized
                tx = aug['tx'] if aug['do_translate'] else 0.0
                ty = aug['ty'] if aug['do_translate'] else 0.0
                tz = aug['tz'] if aug['do_translate'] else 0.0
                # We use inverse scaling in building affine, so set scale factors accordingly
                A = build_affine_3d_from_components(R=R,
                                                   scale_xyz=(1.0 / sx, 1.0 / sy, 1.0 / sz),
                                                   shear_mat=SH,
                                                   t_xyz=(tx, ty, tz),
                                                   flip_x=flip_x, flip_y=flip_y, flip_z=flip_z)
                thetas.append(A.to(self.device))
            theta_batch = torch.stack(thetas, dim=0)  # Nx3x4
            return theta_batch

    def _apply_affine_batch(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W) or (N, C, D, H, W)
        theta: Nx2x3 or Nx3x4
        returns transformed tensor
        """
        # detect complex dtype and split if necessary
        is_complex = torch.is_complex(x)
        if is_complex:
            real = self._apply_affine_batch(x.real, theta)
            imag = self._apply_affine_batch(x.imag, theta)
            return torch.complex(real, imag)

        # real path
        # Ensure theta and x on same device
        theta = theta.to(x.device)
        # Note: F.affine_grid expects theta shape (N, 2, 3) for 4D or (N, 3, 4) for 5D and target size
        if self.dim == 2:
            out = F.grid_sample(x, F.affine_grid(theta, x.size(), align_corners=False),
                                mode='bilinear', padding_mode='border', align_corners=False)
            return out
        else:
            out = F.grid_sample(x, F.affine_grid(theta, x.size(), align_corners=False),
                                mode='bilinear', padding_mode='border', align_corners=False)
            return out

    # ----------------------
    # Coil sampler
    # ----------------------
    def default_coil_sampler(self, x: torch.Tensor, keep: int, coils_axis: int = 1) -> torch.Tensor:
        """
        Subsample coil maps along channel axis. Keep 'keep' coils selected uniformly at random.
        """
        if keep is None:
            return x
        C = x.shape[coils_axis]
        keep = min(max(1, int(keep)), C)
        idx = torch.randperm(C, device=x.device)[:keep]
        return torch.index_select(x, dim=coils_axis, index=idx)

    # ----------------------
    # Public apply function
    # ----------------------
    def apply(self, x: torch.Tensor, is_csm: bool = False, aug_spec_list: Optional[List[Dict[str, Any]]] = None,
              coils_axis: int = 1) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Apply augmentations to input batch x.
        x shape: [bS, channels, X, Y, Z] for 3D or [bS, channels, X, Y] for 2D
        is_csm: whether x are coil sensitivity maps (affects coil subsampling applicability)
        aug_spec_list: optional pre-specified augment list (length bS). If None, will sample using sample_augmentations.
        Returns (x_aug, aug_info_list)
        """
        assert x.dim() in (4, 5), "Input must be 4D (2D images) or 5D (3D images)."
        batch_size = x.shape[0]
        if aug_spec_list is None:
            aug_spec_list = self.sample_augmentations(batch_size)

        # Prepare theta batch
        theta = self._compose_affines_for_batch(aug_spec_list, x.shape)

        # Apply affine to entire batch at once
        x_aug = self._apply_affine_batch(x, theta)

        # If coil subsampling requested and is_csm True, do it per-sample (we do it by concatenating results)
        if is_csm:
            out_list = []
            for i, aug in enumerate(aug_spec_list):
                sample = x_aug[i:i+1]  # keep batch dim
                if aug['do_coil_sub'] and aug['coil_keep'] is not None:
                    sample = self.default_coil_sampler(sample, keep=aug['coil_keep'], coils_axis=coils_axis)
                out_list.append(sample)
            # pad/stack back into a tensor: if coil counts differ we will return a list instead
            # simplest: if all have same channel count, stack; else return list
            chans = [s.shape[1] for s in out_list]
            if all(c == chans[0] for c in chans):
                x_aug = torch.cat(out_list, dim=0)
            else:
                # Return as list in this unusual case
                x_aug = out_list

        return x_aug, aug_spec_list


# ----------------------
# Usage / testing
# ----------------------
if __name__ == "__main__":
    # Quick test -- 2D imaging (complex) and 3D csm example
    device = torch.device('cpu')

    # 2D image: batch 2, channels=1, H=64, W=64, complex
    bS = 2
    img2d = torch.randn(bS, 1, 64, 64, dtype=torch.float32, device=device) \
            + 1j * torch.randn(bS, 1, 64, 64, dtype=torch.float32, device=device)
    aug = SpatialAugmentations(dim=2, prob=0.6, device=device, min_coils=6, max_coils=12)
    x2d_aug, info2d = aug.apply(img2d, is_csm=False)
    print("2D augmented shape:", x2d_aug.shape)
    print("2D aug info[0]:", info2d[0])

    # 3D coil sensitivity maps: batch 1, channels=8 (coils), D=16, H=64, W=64, complex
    bS = 1
    csm3d = torch.randn(bS, 8, 16, 64, 64, dtype=torch.float32, device=device) \
            + 1j * torch.randn(bS, 8, 16, 64, 64, dtype=torch.float32, device=device)
    aug3d = SpatialAugmentations(dim=3, prob=0.7, device=device, min_coils=6, max_coils=8)
    csm3d_aug, info3d = aug3d.apply(csm3d, is_csm=True)
    print("3D CSM result type:", type(csm3d_aug))
    print("3D aug info[0]:", info3d[0])
