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
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
from augmentrum.core.base_module import BaseModule
from augmentrum.core.nifti_mrs_plus import Backend, NIfTI_MRS_Plus


__all__ = ['SpatialAugmentations']


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
class SpatialAugmentations(BaseModule):

    SUPPORTED_BACKENDS = [Backend.NIFTI_LIST, Backend.NUMPY, Backend.PYTORCH,
                         Backend.TENSORFLOW, Backend.JAX, Backend.KERAS]
    
    def __init__(self,
                 dim: int = 3,
                 prob: float = 0.5,
                 device: Optional[torch.device] = None,
                 data_ranges: Optional[Dict[str, Any]] = None,
                 csm_ranges: Optional[Dict[str, Any]] = None,
                 min_coils: int = 6,
                 max_coils: Optional[int] = None):
        """
        dim: 2 or 3
        prob: Bernoulli probability to activate each augmentation
        device: torch device
        data_ranges: augmentation ranges for data pipeline (spectroscopy data)
        csm_ranges: augmentation ranges for CSM pipeline (coil sensitivity maps)
        min_coils, max_coils: integer range of coils to keep when coil subsampling is active
        
        Pipeline-specific ranges allow different augmentation constraints:
        - Data pipeline: typically has restricted z-axis rotation
        - CSM pipeline: allows unrestricted rotation (full 360°)
        """
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.prob = float(prob)
        self.device = device or torch.device('cpu')
        
        # Default ranges for data pipeline (restricted rotation)
        default_data_ranges = {
            'translation_frac': 0.10,        # fraction of axis length
            'max_z_angle_deg': 30.0,         # restricted z-axis rotation
            'max_random_angle_deg': 359.0,
            'zoom_range': (0.9, 1.1),
            'shear_max': 0.15,
            'scale_range': (0.9, 1.1),
            'translation_prob': 0.1,
        }
        
        # Default ranges for CSM pipeline (unrestricted rotation)
        default_csm_ranges = {
            'translation_frac': 0.10,
            'max_z_angle_deg': 359.0,        # unrestricted z-axis rotation
            'max_random_angle_deg': 359.0,
            'zoom_range': (0.9, 1.1),
            'shear_max': 0.15,
            'scale_range': (0.9, 1.1),
            'translation_prob': 0.1,
        }
        
        # Update with user-provided ranges
        if data_ranges:
            default_data_ranges.update(data_ranges)
        if csm_ranges:
            default_csm_ranges.update(csm_ranges)
            
        self.data_ranges = default_data_ranges
        self.csm_ranges = default_csm_ranges
        
        self.min_coils = int(min_coils) if min_coils is not None else None
        self.max_coils = int(max_coils) if max_coils is not None else None

    # ----------------------
    # Sampling augmentations per-sample
    # ----------------------
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

            # random full rotation angle
            random_rot_deg = 0.0
            if do_random_csm_rot:
                random_rot_deg = torch.rand(1, generator=rng).item() * ranges['max_random_angle_deg']

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

    # ----------------------
    # Affine composition
    # ----------------------
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
                theta = build_affine_2d_from_components(rot_rad, (sx, sy), (shx, shy), tx, ty, fx, fy)
                thetas.append(theta)
            else:
                # 3D
                # Build rotation matrix
                R = torch.eye(3, dtype=torch.float32)
                # z-axis rotation
                if spec['do_z_rot']:
                    z_rad = math.radians(spec['z_angle_deg'])
                    q_z = quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0]), z_rad, device=None)
                    R = quat_to_rotmat(q_z)
                # random rotation
                if spec['do_random_csm_rot']:
                    rand_rad = math.radians(spec['random_rot_deg'])
                    axis_rand = torch.randn(3)
                    axis_rand = axis_rand / (axis_rand.norm() + 1e-12)
                    q_rand = quat_from_axis_angle(axis_rand, rand_rad, device=None)
                    R_rand = quat_to_rotmat(q_rand)
                    R = R @ R_rand
                # 90-degree rotations around z
                if spec['do_rot90']:
                    k = spec['k_rot90']
                    angle_90 = k * math.pi / 2.0
                    q90 = quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0]), angle_90, device=None)
                    R90 = quat_to_rotmat(q90)
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
                theta = build_affine_3d_from_components(R, (sx, sy, sz), SH, (tx, ty, tz), fx, fy, fz)
                thetas.append(theta)
        # Stack
        theta_batch = torch.stack(thetas, dim=0)  # Nx2x3 or Nx3x4
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
    def apply(self, x: torch.Tensor, 
             pipeline: str = 'data',
             aug_spec_list: Optional[List[Dict[str, Any]]] = None,
             coils_axis: int = 1) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Apply augmentations to input batch x.
        
        Args:
            x: Input tensor [bS, channels, X, Y, Z] for 3D or [bS, channels, X, Y] for 2D
            pipeline: 'data' or 'csm' - determines augmentation ranges if sampling new augs
            aug_spec_list: Optional pre-specified augment list (length bS). If None, will sample.
            coils_axis: Axis along which coils are stored (for coil subsampling)
            
        Returns:
            (x_aug, aug_info_list): Augmented tensor and list of augmentation specifications
        """
        assert x.dim() in (4, 5), "Input must be 4D (2D images) or 5D (3D images)."
        batch_size = x.shape[0]
        is_csm = (pipeline == 'csm')
        
        if aug_spec_list is None:
            aug_spec_list = self.sample_augmentations(batch_size, pipeline=pipeline)

        # Prepare theta batch
        theta = self._compose_affines_for_batch(aug_spec_list, x.shape)

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
            chans = [s.shape[1] for s in out_list]
            if all(c == chans[0] for c in chans):
                x_aug = torch.cat(out_list, dim=0)
            else:
                # Return as list in this unusual case
                x_aug = out_list

        return x_aug, aug_spec_list

    # ----------------------
    # NIFTI conversion helpers
    # ----------------------
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

    # ----------------------
    # Pipeline routing
    # ----------------------
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
                'data_augmentations': augmentation list for data,
                'csm_augmentations': augmentation list for CSMs
            }
        """
        results = {}
        
        # Process data pipeline
        if data_list is not None:
            if isinstance(data_list, list) and len(data_list) > 0 and isinstance(data_list[0], NIfTI_MRS_Plus):
                # NIFTI format
                data_proc, water_proc, data_augs = self.process_nifti_list(
                    data_list=data_list,
                    water_list=water_list,
                    pipeline='data',
                    aug_spec_list=data_aug_list,
                    **kwargs
                )
                results['data'] = data_proc
                results['water'] = water_proc
                results['data_augmentations'] = data_augs
            else:
                # Tensor/array format
                data_proc, water_proc, data_augs = self.process_tensor(
                    data=data_list,
                    water=water_list,
                    pipeline='data',
                    aug_spec_list=data_aug_list,
                    **kwargs
                )
                results['data'] = data_proc
                results['water'] = water_proc
                results['data_augmentations'] = data_augs
        
        # Process CSM pipeline
        if csm_list is not None:
            if isinstance(csm_list, list) and len(csm_list) > 0 and isinstance(csm_list[0], NIfTI_MRS_Plus):
                # NIFTI format
                csm_proc, _, csm_augs = self.process_nifti_list(
                    data_list=csm_list,
                    water_list=None,
                    pipeline='csm',
                    aug_spec_list=csm_aug_list,
                    **kwargs
                )
                results['csm'] = csm_proc
                results['csm_augmentations'] = csm_augs
            else:
                # Tensor/array format
                csm_proc, _, csm_augs = self.process_tensor(
                    data=csm_list,
                    water=None,
                    pipeline='csm',
                    aug_spec_list=csm_aug_list,
                    **kwargs
                )
                results['csm'] = csm_proc
                results['csm_augmentations'] = csm_augs
        
        return results

    # ----------------------
    # NIFTI processing
    # ----------------------
    def process_nifti_list(self,
                          data_list: List[NIfTI_MRS_Plus],
                          water_list: Optional[List[NIfTI_MRS_Plus]] = None,
                          pipeline: str = 'data',
                          aug_spec_list: Optional[List[Dict[str, Any]]] = None,
                          **kwargs) -> Tuple[List[NIfTI_MRS_Plus], Optional[List[NIfTI_MRS_Plus]], List[Dict[str, Any]]]:
        """
        Process list of NIFTI_MRS objects with spatial augmentations.

        Args:
            data_list: List of NIFTI_MRS objects
            water_list: Optional list of water reference NIFTI_MRS objects
            pipeline: 'data' or 'csm' - determines augmentation ranges
            aug_spec_list: Optional pre-specified augmentations. If None, will sample new ones.
            **kwargs: Additional arguments

        Returns:
            Tuple of (augmented_data_list, water_list_unchanged, augmentation_list)
        """
        # Convert NIFTI to tensor
        data_tensor = self._nifti_to_tensor(data_list)
        
        # Apply augmentations
        data_aug_tensor, aug_list = self.apply(
            data_tensor, 
            pipeline=pipeline,
            aug_spec_list=aug_spec_list,
            coils_axis=1
        )
        
        # Convert back to NIFTI
        augmented_data_list = self._tensor_to_nifti(data_aug_tensor, data_list)
        
        # Water references unchanged (or process separately if needed)
        return augmented_data_list, water_list, aug_list

    # ----------------------
    # Tensor processing
    # ----------------------
    def process_tensor(self,
                      data: Union[torch.Tensor, np.ndarray],
                      water: Optional[Union[torch.Tensor, np.ndarray]] = None,
                      pipeline: str = 'data',
                      aug_spec_list: Optional[List[Dict[str, Any]]] = None,
                      **kwargs) -> Tuple[Union[torch.Tensor, np.ndarray], 
                                        Optional[Union[torch.Tensor, np.ndarray]], 
                                        List[Dict[str, Any]]]:
        """
        Process tensor data with spatial augmentations.

        Args:
            data: Input tensor (batch, channels, ...spatial dims)
            water: Optional water reference tensor
            pipeline: 'data' or 'csm' - determines augmentation ranges
            aug_spec_list: Optional pre-specified augmentations. If None, will sample new ones.
            **kwargs: Additional arguments

        Returns:
            Tuple of (augmented_data, water_unchanged, augmentation_list)
        """
        # Convert to torch tensor if needed
        return_numpy = False
        if isinstance(data, np.ndarray):
            return_numpy = True
            data = torch.from_numpy(data)
        
        # Ensure proper dtype
        if torch.is_complex(data):
            data = data.to(dtype=torch.complex64)
        else:
            data = data.to(dtype=torch.float32)
        
        # Move to device
        data = data.to(self.device)
        
        # Apply augmentations
        data_aug, aug_list = self.apply(
            data,
            pipeline=pipeline,
            aug_spec_list=aug_spec_list,
            coils_axis=1
        )
        
        # Convert back to numpy if needed
        if return_numpy:
            if isinstance(data_aug, list):
                # Handle list of tensors (variable coil counts)
                data_aug = [d.cpu().numpy() for d in data_aug]
            else:
                data_aug = data_aug.cpu().numpy()
        
        # Water unchanged
        return data_aug, water, aug_list


# ----------------------
# Usage / testing
# ----------------------
if __name__ == "__main__":
    # Quick test -- 2D imaging (complex) and 3D csm example
    device = torch.device('cpu')

    print("=" * 80)
    print("Testing SpatialAugmentations with dual pipelines")
    print("=" * 80)
    
    # Test 1: 2D data with data pipeline (restricted rotation)
    print("\n1. Testing 2D data pipeline (restricted rotation):")
    bS = 2
    img2d = torch.randn(bS, 1, 64, 64, dtype=torch.float32, device=device) \
            + 1j * torch.randn(bS, 1, 64, 64, dtype=torch.float32, device=device)
    
    aug = SpatialAugmentations(dim=2, prob=0.6, device=device, min_coils=6, max_coils=12)
    x2d_aug, info2d = aug.apply(img2d, pipeline='data')
    print(f"   Shape: {x2d_aug.shape}")
    print(f"   Z-rotation (restricted): {info2d[0]['z_angle_deg']:.2f}°")
    print(f"   Pipeline: {info2d[0]['pipeline']}")

    # Test 2: 3D CSM with CSM pipeline (unrestricted rotation)
    print("\n2. Testing 3D CSM pipeline (unrestricted rotation):")
    bS = 1
    csm3d = torch.randn(bS, 8, 16, 64, 64, dtype=torch.float32, device=device) \
            + 1j * torch.randn(bS, 8, 16, 64, 64, dtype=torch.float32, device=device)
    
    aug3d = SpatialAugmentations(dim=3, prob=0.7, device=device, min_coils=6, max_coils=8)
    csm3d_aug, info3d = aug3d.apply(csm3d, pipeline='csm')
    print(f"   Result type: {type(csm3d_aug)}")
    print(f"   Z-rotation (unrestricted): {info3d[0]['z_angle_deg']:.2f}°")
    print(f"   Coil subsampling: {info3d[0]['do_coil_sub']}, keep={info3d[0]['coil_keep']}")
    print(f"   Pipeline: {info3d[0]['pipeline']}")
    
    # Test 3: Pre-specified augmentations
    print("\n3. Testing with pre-specified augmentations:")
    aug_specs = aug.sample_augmentations(batch_size=2, pipeline='data')
    print(f"   Sampled {len(aug_specs)} augmentation specs")
    img2d_2 = torch.randn(2, 1, 64, 64, dtype=torch.complex64, device=device)
    x2d_aug_2, info2d_2 = aug.apply(img2d_2, pipeline='data', aug_spec_list=aug_specs)
    print(f"   Applied pre-specified augmentations")
    print(f"   Augmentations match: {aug_specs[0] == info2d_2[0]}")
    
    # Test 4: process_tensor
    print("\n4. Testing process_tensor:")
    data_np = np.random.randn(2, 1, 32, 32, 32).astype(np.float32)
    data_proc, _, augs = aug3d.process_tensor(data_np, pipeline='data')
    print(f"   Input type: {type(data_np)}, Output type: {type(data_proc)}")
    print(f"   Returned {len(augs)} augmentation specs")
    
    # Test 5: route_pipeline
    print("\n5. Testing route_pipeline with automatic type detection:")
    data_tensor = torch.randn(2, 1, 32, 32, 32, dtype=torch.float32)
    csm_tensor = torch.randn(2, 8, 32, 32, 32, dtype=torch.complex64)
    
    results = aug3d.route_pipeline(
        data_list=data_tensor,
        csm_list=csm_tensor
    )
    print(f"   Processed data shape: {results['data'].shape}")
    print(f"   Processed CSM type: {type(results['csm'])}")
    print(f"   Data augmentations: {len(results['data_augmentations'])} specs")
    print(f"   CSM augmentations: {len(results['csm_augmentations'])} specs")
    print(f"   Data pipeline: {results['data_augmentations'][0]['pipeline']}")
    print(f"   CSM pipeline: {results['csm_augmentations'][0]['pipeline']}")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
