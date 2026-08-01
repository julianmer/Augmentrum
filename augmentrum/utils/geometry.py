####################################################################################################
#                                          geometry.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-07-31                                                                              #
#                                                                                                  #
# Purpose: Quaternion and affine matrix construction for spatial resampling. Builds the theta      #
#          matrices consumed by torch's affine_grid, in the normalised [-1, 1] coordinate system.  #
#          Torch-typed, because their only consumer (F.affine_grid) is torch-only.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math
from typing import Optional, Tuple

import torch


#**************************************************************************************************#
#                                           Class Affine                                           #
#**************************************************************************************************#
#                                                                                                  #
# Builders for the ``theta`` matrices that drive :func:`torch.nn.functional.affine_grid`.          #
#                                                                                                  #
#**************************************************************************************************#
class Affine:
    """
    Builders for the ``theta`` matrices that drive :func:`torch.nn.functional.affine_grid`.

    Both builders map *normalised output* coordinates to *normalised input*
    coordinates, which is the direction ``affine_grid`` expects, and both encode
    a flip as a negative scale on that axis.

    >>> theta = Affine.build_2d(0.0, (1.0, 1.0), (0.0, 0.0), 0.0, 0.0, False, False)
    >>> tuple(theta.shape)
    (2, 3)
    """

    #*****************#
    #   quaternions   #
    #*****************#

    @staticmethod
    def quat_from_axis_angle(axis: torch.Tensor, angle_rad: float, device=None) -> torch.Tensor:
        """
        Unit quaternion ``[w, x, y, z]`` for a rotation of *angle_rad* about *axis*.

        The half-angle is evaluated with Python floats, so the result is not
        differentiable with respect to *angle_rad*. That is deliberate: the
        angle comes from an augmentation spec, never from a graph.
        """
        axis = axis.to(device) if device is not None else axis
        axis = axis / (axis.norm() + 1e-12)
        s = math.sin(angle_rad / 2.0)
        w = math.cos(angle_rad / 2.0)
        x, y, z = (axis * s).tolist()
        return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)

    @staticmethod
    def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """Convert a ``[w, x, y, z]`` quaternion to a 3x3 rotation matrix."""
        w, x, y, z = q
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        return torch.stack([
            torch.stack([ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy)]),
            torch.stack([2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx)]),
            torch.stack([2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz]),
        ])

    #*********************#
    #   affine builders   #
    #*********************#

    @staticmethod
    def build_2d(rotation_rad: float,
                 scale_xy: Tuple[float, float],
                 shear_xy: Tuple[float, float],
                 tx: float,
                 ty: float,
                 flip_x: bool,
                 flip_y: bool) -> torch.Tensor:
        """
        Build the 2x3 affine mapping normalised output coords to normalised input coords.

        Args:
            rotation_rad: in-plane rotation angle.
            scale_xy: per-axis zoom factors.
            shear_xy: off-diagonal shear terms.
            tx, ty: normalised translations in [-1, 1].
            flip_x, flip_y: applied as a negative scale on that axis.
        """
        c, s = math.cos(rotation_rad), math.sin(rotation_rad)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        shx, shy = shear_xy
        SH = torch.tensor([[1.0, shx], [shy, 1.0]], dtype=torch.float32)
        sx = -scale_xy[0] if flip_x else scale_xy[0]
        sy = -scale_xy[1] if flip_y else scale_xy[1]
        Sc = torch.diag(torch.tensor([sx, sy], dtype=torch.float32))
        A = R @ SH @ Sc
        T = torch.tensor([tx, ty], dtype=torch.float32).unsqueeze(1)
        return torch.cat([A, T], dim=1)

    @staticmethod
    def build_3d(R: torch.Tensor,
                 scale_xyz: Tuple[float, float, float],
                 shear_mat: Optional[torch.Tensor],
                 t_xyz: Tuple[float, float, float],
                 flip_x: bool,
                 flip_y: bool,
                 flip_z: bool) -> torch.Tensor:
        """
        Build the 3x4 affine mapping normalised output coords to normalised input coords.

        Args:
            R: 3x3 rotation matrix, typically from :meth:`quat_to_rotmat`.
            scale_xyz: per-axis zoom factors.
            shear_mat: 3x3 shear matrix with unit diagonal; ``None`` means identity.
            t_xyz: normalised translations in [-1, 1].
            flip_x, flip_y, flip_z: applied as a negative scale on that axis.
        """
        sx, sy, sz = scale_xyz
        sx = -sx if flip_x else sx
        sy = -sy if flip_y else sy
        sz = -sz if flip_z else sz
        Sc = torch.diag(torch.tensor([sx, sy, sz], dtype=torch.float32))
        SH = shear_mat if shear_mat is not None else torch.eye(3, dtype=torch.float32)
        linear = R @ SH @ Sc
        T = torch.tensor(t_xyz, dtype=torch.float32).unsqueeze(1)
        return torch.cat([linear, T], dim=1)
