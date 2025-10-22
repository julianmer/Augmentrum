####################################################################################################
#                                 tricubic_interpolator.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2025-10-22                                                                              #
#                                                                                                  #
# Purpose: Implements TricubicHermiteMAkima3D for sampling 3D k-space/image domain volumes         #
#          using non-Cartesian trajectories.                                                       #
#                                                                                                  #
####################################################################################################

import torch
import torch.nn as nn


__all__ = ['TricubicHermiteMAkima3D']


class TricubicHermiteMAkima3D(nn.Module):
    def __init__(self, signal: torch.Tensor):
        """
        Tricubic Hermite Modified Akima Interpolation for 3D data.

        Args:
            signal (torch.Tensor): 3D tensor of signal values, shape (bS, Nx, Ny, Nz).

        Note: The class assumes the input signal shape is [bS, Nx, Ny, Nz]
              and initializes axes based on Nx, Ny, Nz.
        """
        super().__init__()
        print('Initialize...')
        self.device = signal.device
        # signal.shape is [bS, Nx, Ny, Nz]
        _, xaxis_size, yaxis_size, zaxis_size = signal.shape
        
        # Store axes and signal
        self.xaxis = torch.linspace(start=-1., end=1., steps=xaxis_size, device=self.device)
        self.yaxis = torch.linspace(start=-1., end=1., steps=yaxis_size, device=self.device)
        self.zaxis = torch.linspace(start=-1., end=1., steps=zaxis_size, device=self.device)
        self.signal = signal.to(self.device)

        # Compute slopes for Modified Akima interpolation
        self.m_x = self._prepare_slopes(signal, self.xaxis.unsqueeze(-1).unsqueeze(-1), dim=1)
        self.m_y = self._prepare_slopes(signal, self.yaxis.unsqueeze(0).unsqueeze(-1), dim=2)
        self.m_z = self._prepare_slopes(signal, self.zaxis.unsqueeze(0).unsqueeze(0), dim=3)

    def _prepare_slopes(self, signal, axis, dim):
        """
        Compute Modified Akima slopes along a given axis. (Original implementation retained)
        """
        # helper to build a tuple-of-slices selecting slice(start, end) on axis `dim`
        def slc(dim, start=None, end=None):
            s = [slice(None)] * signal.dim()
            s[dim] = slice(start, end)
            return tuple(s)

        m = (signal.diff(dim=dim) / axis.unsqueeze(0).diff(dim=dim))
        d0 = 2 * m.select(dim, 0) - m.select(dim, 1)
        dn = 2 * m.select(dim, -1) - m.select(dim, -2)
        m0 = 2 * d0 - m.select(dim, 0)
        mn = 2 * dn - m.select(dim, -1)

        m = torch.cat([m0.unsqueeze(dim), d0.unsqueeze(dim), m, dn.unsqueeze(dim), mn.unsqueeze(dim)], dim=dim)
        m_prev, m_next = m[slc(dim, None, -1)], m[slc(dim, 1, None)]
        weights = torch.abs(m_next - m_prev) + torch.abs((m_prev + m_next) / 2.0)

        w1, w2 = weights[slc(dim, None, -2)], weights[slc(dim, 2, None)]
        delta1, delta2 = m[slc(dim, 1, -2)], m[slc(dim, 2, -1)]
        w_sum = w1 + w2 + 1e-6
        slopes = (w2 / w_sum) * delta1 + (w1 / w_sum) * delta2
        slopes[w_sum == 0] = 0.0
        return slopes

    @staticmethod
    def h_poly(t):
        """
        Compute cubic Hermite basis functions.
        """
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00, h10, h01, h11

    def _interpolate_tricubic_hermite(self, B, S, L, I_x, I_y, I_z, t_x, t_y, t_z, dx_cell, dy_cell, dz_cell):
        """
        Vectorized core logic for tricubic Hermite interpolation.
        
        Input indices and local coordinates have shape [B, S, L].
        """
        # Compute Hermite basis functions and stack them [B,S,L,4]
        H_x = torch.stack(self.h_poly(t_x), dim=-1)
        H_y = torch.stack(self.h_poly(t_y), dim=-1)
        H_z = torch.stack(self.h_poly(t_z), dim=-1)

        # --- Subgrid Indexing ---
        ix_n = torch.stack([I_x - 1, I_x, I_x + 1, I_x + 2], dim=-1).clamp(0, self.xaxis.numel() - 1)
        iy_n = torch.stack([I_y - 1, I_y, I_y + 1, I_y + 2], dim=-1).clamp(0, self.yaxis.numel() - 1)
        iz_n = torch.stack([I_z - 1, I_z, I_z + 1, I_z + 2], dim=-1).clamp(0, self.zaxis.numel() - 1)

        target_shape = (B, S, L, 4, 4, 4)
        batch_idx = torch.arange(B, device=self.device, dtype=torch.long).view(B, 1, 1, 1, 1, 1).expand(target_shape)
        ix_idx = ix_n.unsqueeze(-1).unsqueeze(-1).expand(target_shape)
        iy_idx = iy_n.unsqueeze(-2).unsqueeze(-1).expand(target_shape)
        iz_idx = iz_n.unsqueeze(-2).unsqueeze(-2).expand(target_shape)

        signal_subgrid = self.signal[batch_idx, ix_idx, iy_idx, iz_idx]   # [B,S,L,4,4,4]
        m_x_subgrid = self.m_x[batch_idx, ix_idx, iy_idx, iz_idx]
        m_y_subgrid = self.m_y[batch_idx, ix_idx, iy_idx, iz_idx]
        m_z_subgrid = self.m_z[batch_idx, ix_idx, iy_idx, iz_idx]

        # --- 1. Compute Cross-Derivatives (Central Difference Approximation) ---
        y_plus_coords  = self.yaxis[iy_n[..., 2]] # index i+1
        y_minus_coords = self.yaxis[iy_n[..., 0]] # index i-1
        denom_y = (y_plus_coords - y_minus_coords).clamp(min=1e-12) # [B,S,L]
        
        z_plus_coords  = self.zaxis[iz_n[..., 2]] # index i+1
        z_minus_coords = self.zaxis[iz_n[..., 0]] # index i-1
        denom_z = (z_plus_coords - z_minus_coords).clamp(min=1e-12) # [B,S,L]
        
        # Central difference at grid point (I_x, I_y, I_z) - which is index 1 in the 4-neighbour array
        
        # fxy
        fx_y_plus  = m_x_subgrid[..., 1, 2, 1]; fx_y_minus = m_x_subgrid[..., 1, 0, 1]
        fxy_base = (fx_y_plus - fx_y_minus) / denom_y
        
        # fxz
        fx_z_plus  = m_x_subgrid[..., 1, 1, 2]; fx_z_minus = m_x_subgrid[..., 1, 1, 0]
        fxz_base = (fx_z_plus - fx_z_minus) / denom_z

        # fyz
        fy_z_plus  = m_y_subgrid[..., 1, 1, 2]; fy_z_minus = m_y_subgrid[..., 1, 1, 0]
        fyz_base = (fy_z_plus - fy_z_minus) / denom_z
        
        # fxyz
        fx_yplus_zplus  = m_x_subgrid[..., 1, 2, 2]; fx_yminus_zplus = m_x_subgrid[..., 1, 0, 2]
        fxy_z_plus  = (fx_yplus_zplus - fx_yminus_zplus) / denom_y
        fx_yplus_zminus  = m_x_subgrid[..., 1, 2, 0]; fx_yminus_zminus = m_x_subgrid[..., 1, 0, 0]
        fxy_z_minus = (fx_yplus_zminus - fx_yminus_zminus) / denom_y
        fxyz_base = (fxy_z_plus - fxy_z_minus) / denom_z

        # --- 2. Build the $4 \times 4 \times 4$ Coefficient Tensor $\mathbf{P}$ ---
        P = torch.zeros(target_shape, device=self.device)
        
        # Helper variables for broadcasting cell widths
        dx = dx_cell.unsqueeze(-1).unsqueeze(-1)
        dy = dy_cell.unsqueeze(-1).unsqueeze(-1)
        dz = dz_cell.unsqueeze(-1).unsqueeze(-1)
        
        # Extract the $2 \times 2 \times 2$ cell corner values (indices 1:3)
        f_c = signal_subgrid[..., 1:3, 1:3, 1:3]
        mx_c = m_x_subgrid[..., 1:3, 1:3, 1:3]
        my_c = m_y_subgrid[..., 1:3, 1:3, 1:3]
        mz_c = m_z_subgrid[..., 1:3, 1:3, 1:3]

        # print("mx_c.shape: {}; dx.shape: {}; P[..., 0:2, 0:2, 0:2].shape: {}".format(mx_c.shape, dx.unsqueeze(-1).shape, P[..., 0:2, 0:2, 0:2].shape))
        # print("my_c.shape: {}; dy.shape: {}; P[..., 0:2, 2:4, 0:2].shape: {}".format(my_c.shape, dy.unsqueeze(-1).unsqueeze(-1).shape, P[..., 0:2, 2:4, 0:2].shape))
        # Populate the 8 sub-blocks of P (indices 0:2 and 2:4 correspond to value/derivative in basis)
        P[..., 0:2, 0:2, 0:2] = f_c                                       # f
        P[..., 2:4, 0:2, 0:2] = mx_c * dx.unsqueeze(-1)                   # fx * dx
        P[..., 0:2, 2:4, 0:2] = my_c * dy.unsqueeze(-1)#.unsqueeze(-1)     # fy * dy
        P[..., 0:2, 0:2, 2:4] = mz_c * dz.unsqueeze(-1)#.unsqueeze(-1)     # fz * dz
        
        # print("fxy_base.shape: {}; dx.shape: {}; dy.shape: {}; P[..., 2:4, 2:4, 0:2].shape: {}".format(fxy_base.unsqueeze(-1).unsqueeze(-1).shape, dx.unsqueeze(-1).shape, dy.unsqueeze(-1).shape, P[..., 2:4, 2:4, 0:2].shape))
        # Use the central difference approximations for mixed derivatives, broadcasting to $2 \times 2 \times 2$
        P[..., 2:4, 2:4, 0:2] = fxy_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (dx * dy).unsqueeze(-1)    # fxy * dx * dy
        P[..., 2:4, 0:2, 2:4] = fxz_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (dx * dz).unsqueeze(-1)    # fxz * dx * dz
        P[..., 0:2, 2:4, 2:4] = fyz_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (dy * dz).unsqueeze(-1)    # fyz * dy * dz
        P[..., 2:4, 2:4, 2:4] = fxyz_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (dx * dy * dz).unsqueeze(-1) # fxyz * dx * dy * dz

        # --- 3. Compute the final interpolation result using tensor contraction (einsum) ---
        interpolated = torch.einsum('bsli, bslj, bslk, bslijk -> bsl', H_x, H_y, H_z, P)
        
        return interpolated
    
    def forward(self, coords):
        """
        Perform tricubic interpolation for a batch of coordinates.
        This simplified version only handles pre-processing and post-processing.
        
        Args:
            coords (torch.Tensor): Tensor of shape [batchSize, num_shots, 3, shot_length].

        Returns:
            torch.Tensor: Interpolated values, shape [batchSize, num_shots, 1, shot_length].
        """
        print('Forward pass (Cleaned)')
        batch_size, num_shots, _, shot_length = coords.shape
        x_coords, y_coords, z_coords = coords.split(1, dim=2)
        
        # --- PRE-PROCESSING: Reshape and Find Indices/Local Coords ---
        x_coords = x_coords.squeeze(2)  # -> [B, S, L]
        y_coords = y_coords.squeeze(2)
        z_coords = z_coords.squeeze(2)

        B, S, L = x_coords.shape

        # Find grid cell indices I and clamp
        I_x = (torch.searchsorted(self.xaxis, x_coords.clamp(self.xaxis.min(), self.xaxis.max())) - 1).clamp(0, self.xaxis.numel() - 2)
        I_y = (torch.searchsorted(self.yaxis, y_coords.clamp(self.yaxis.min(), self.yaxis.max())) - 1).clamp(0, self.yaxis.numel() - 2)
        I_z = (torch.searchsorted(self.zaxis, z_coords.clamp(self.zaxis.min(), self.zaxis.max())) - 1).clamp(0, self.zaxis.numel() - 2)

        # Compute cell widths (denominator for t)
        dx_cell = (self.xaxis[I_x + 1] - self.xaxis[I_x])
        dy_cell = (self.yaxis[I_y + 1] - self.yaxis[I_y])
        dz_cell = (self.zaxis[I_z + 1] - self.zaxis[I_z])

        # Compute local coordinates t (0 <= t < 1)
        t_x = (x_coords - self.xaxis[I_x]) / dx_cell
        t_y = (y_coords - self.yaxis[I_y]) / dy_cell
        t_z = (z_coords - self.zaxis[I_z]) / dz_cell

        # --- CORE LOGIC: Call the private interpolation method ---
        interpolated_vals = self._interpolate_tricubic_hermite(
            B, S, L, I_x, I_y, I_z, t_x, t_y, t_z, dx_cell, dy_cell, dz_cell
        )

        # --- POST-PROCESSING: Reshape back to API output format ---
        result = interpolated_vals.view(batch_size, num_shots, 1, shot_length)
        
        return result


if __name__ == "__ main __":
    # Example grid
    signal = torch.rand(4, 10, 10, 10)  # 3D signal

    # Example trajectory coordinates
    coords = torch.rand(4, 10, 3, 100)  # Batch of 3D trajectories (batchSize=4, num_shots=2, dim=3, shot_length=100)

    # Instantiate and interpolate
    #     interpolator = TricubicHermiteMAkima3D(xaxis, yaxis, zaxis, signal)
    interpolator = TricubicHermiteMAkima3D(signal)
    output = interpolator(coords)

    # Output shape: [4, 2, 1, 100]
    print(output.shape)
