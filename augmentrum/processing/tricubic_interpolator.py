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
            signal (torch.Tensor): 4D tensor of signal values, shape (B, Nx, Ny, Nz).
                                   It is internally reshaped to [B, S=1, Nx, Ny, Nz].
        """
        super().__init__()        
        # --- Dimension Check and Reshape to 5D [B, S, Nx, Ny, Nz] ---
        if signal.dim() == 4:
            # Assume shape is [B, Nx, Ny, Nz] and insert S=1 dimension
            signal = signal.unsqueeze(1)
        elif signal.dim() != 5:
            raise ValueError(f"Input signal must be 4D [B, Nx, Ny, Nz] or 5D [B, S, Nx, Ny, Nz], but got {signal.dim()}D tensor.")

        self.dtype = signal.dtype
        self.device = signal.device
        
        # signal.shape is now [B, S, Nx, Ny, Nz]
        B, S, Nx, Ny, Nz = signal.shape
        # S = 1
        self.B, self.S, self.Nx, self.Ny, self.Nz = B, S, Nx, Ny, Nz
        
        # Store axes and signal
        self.xaxis = torch.linspace(start=-1., end=1., steps=Nx, device=self.device)
        self.yaxis = torch.linspace(start=-1., end=1., steps=Ny, device=self.device)
        self.zaxis = torch.linspace(start=-1., end=1., steps=Nz, device=self.device)
        self.signal = signal.to(self.device)

        # Compute slopes for Modified Akima interpolation (axes are dimensions 2, 3, 4)
        # The axis tensor needs to be unsqueezed to match the 5D signal structure for broadcasting
        self.m_x = self._prepare_slopes(signal, self.xaxis.view(1, 1, Nx, 1, 1), dim=2)
        self.m_y = self._prepare_slopes(signal, self.yaxis.view(1, 1, 1, Ny, 1), dim=3)
        self.m_z = self._prepare_slopes(signal, self.zaxis.view(1, 1, 1, 1, Nz), dim=4)

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
        # FIX 1: Cast local coordinates (t) to self.dtype (e.g., complex64) for multiplication
        # The basis functions are computed based on the real coordinates, but the result must match P's dtype.
        H_x = torch.stack(self.h_poly(t_x), dim=-1).to(self.dtype)
        H_y = torch.stack(self.h_poly(t_y), dim=-1).to(self.dtype)
        H_z = torch.stack(self.h_poly(t_z), dim=-1).to(self.dtype)

        # --- Subgrid Indexing ---
        # The 4 neighbours for interpolation: i-1, i, i+1, i+2
        ix_n = torch.stack([I_x - 1, I_x, I_x + 1, I_x + 2], dim=-1).clamp(0, self.xaxis.numel() - 1)
        iy_n = torch.stack([I_y - 1, I_y, I_y + 1, I_y + 2], dim=-1).clamp(0, self.yaxis.numel() - 1)
        iz_n = torch.stack([I_z - 1, I_z, I_z + 1, I_z + 2], dim=-1).clamp(0, self.zaxis.numel() - 1)

        # Target coefficient shape [B, S, L, 4, 4, 4]
        target_shape = (B, S, L, 4, 4, 4)
        
        # Batch and Shot indexing: [B, S, L, 4, 4, 4]
        b_idx = torch.arange(B, device=self.device, dtype=torch.long).view(B, 1, 1, 1, 1, 1) # [B, 1, 1, 1, 1, 1]
        s_idx = torch.arange(S, device=self.device, dtype=torch.long).view(1, S, 1, 1, 1, 1) # [1, S, 1, 1, 1, 1]

        # X, Y, Z grid indices: [B, S, L, 4, 4, 4] - relying on broadcasting
        ix_idx = ix_n.unsqueeze(-1).unsqueeze(-1) # [B, S, L, 4, 1, 1]
        iy_idx = iy_n.unsqueeze(-2).unsqueeze(-1) # [B, S, L, 1, 4, 1]
        iz_idx = iz_n.unsqueeze(-2).unsqueeze(-2) # [B, S, L, 1, 1, 4]

        # FIX 2: Correct indexing using B and S=1 indices (5D signal tensor)
        signal_subgrid = self.signal[b_idx, s_idx, ix_idx, iy_idx, iz_idx]   # [B,S,L,4,4,4]
        m_x_subgrid = self.m_x[b_idx, s_idx, ix_idx, iy_idx, iz_idx]
        m_y_subgrid = self.m_y[b_idx, s_idx, ix_idx, iy_idx, iz_idx]
        m_z_subgrid = self.m_z[b_idx, s_idx, ix_idx, iy_idx, iz_idx]

        # --- 1. Compute Cross-Derivatives (Central Difference Approximation) ---
        
        # Denominators are real floats, cast to complex dtype for division if signal is complex
        dtype_for_division = self.signal.dtype
        denom_y = (self.yaxis[iy_n[..., 2]] - self.yaxis[iy_n[..., 0]]).clamp(min=1e-12).to(dtype_for_division) # [B,S,L]
        denom_z = (self.zaxis[iz_n[..., 2]] - self.zaxis[iz_n[..., 0]]).clamp(min=1e-12).to(dtype_for_division) # [B,S,L]
        
        # Central difference at grid point (I_x, I_y, I_z) - which is index 1 in the 4-neighbour array
        
        # fxy
        fx_y_plus  = m_x_subgrid[..., 1, 2, 1]; 
        fx_y_minus = m_x_subgrid[..., 1, 0, 1]
        fxy_base = (fx_y_plus - fx_y_minus) / denom_y
        
        # fxz
        fx_z_plus  = m_x_subgrid[..., 1, 1, 2]; 
        fx_z_minus = m_x_subgrid[..., 1, 1, 0]
        fxz_base = (fx_z_plus - fx_z_minus) / denom_z

        # fyz
        fy_z_plus  = m_y_subgrid[..., 1, 1, 2]; 
        fy_z_minus = m_y_subgrid[..., 1, 1, 0]
        fyz_base = (fy_z_plus - fy_z_minus) / denom_z
        
        # fxyz
        fx_yplus_zplus  = m_x_subgrid[..., 1, 2, 2]; 
        fx_yminus_zplus = m_x_subgrid[..., 1, 0, 2]
        fxy_z_plus  = (fx_yplus_zplus - fx_yminus_zplus) / denom_y
        fx_yplus_zminus  = m_x_subgrid[..., 1, 2, 0]; 
        fx_yminus_zminus = m_x_subgrid[..., 1, 0, 0]
        fxy_z_minus = (fx_yplus_zminus - fx_yminus_zminus) / denom_y
        fxyz_base = (fxy_z_plus - fxy_z_minus) / denom_z

        # --- 2. Build the $4 \times 4 \times 4$ Coefficient Tensor $\mathbf{P}$ ---
        P = torch.zeros(target_shape, dtype=self.dtype, device=self.device)
        
        # Helper variables for broadcasting cell widths (must be same dtype as P)
        dx_val = dx_cell.unsqueeze(-1).unsqueeze(-1).to(self.dtype)
        dy_val = dy_cell.unsqueeze(-1).unsqueeze(-1).to(self.dtype)
        dz_val = dz_cell.unsqueeze(-1).unsqueeze(-1).to(self.dtype)
        
        # Extract the $2 \times 2 \times 2$ cell corner values (indices 1:3)
        f_c = signal_subgrid[..., 1:3, 1:3, 1:3]
        mx_c = m_x_subgrid[..., 1:3, 1:3, 1:3]
        my_c = m_y_subgrid[..., 1:3, 1:3, 1:3]
        mz_c = m_z_subgrid[..., 1:3, 1:3, 1:3]

        # Populate the 8 sub-blocks of P (indices 0:2 and 2:4 correspond to value/derivative in basis)
        P[..., 0:2, 0:2, 0:2] = f_c                                         # f
        P[..., 2:4, 0:2, 0:2] = mx_c * dx_val.unsqueeze(-1)                   # fx * dx
        P[..., 0:2, 2:4, 0:2] = my_c * dy_val.unsqueeze(-1)                   # fy * dy
        P[..., 0:2, 0:2, 2:4] = mz_c * dz_val.unsqueeze(-1)                   # fz * dz
        
        # Use the central difference approximations for mixed derivatives, broadcasting to $2 \times 2 \times 2$
        dxdy_val = (dx_val * dy_val).unsqueeze(-1)
        dxdz_val = (dx_val * dz_val).unsqueeze(-1)
        dydz_val = (dy_val * dz_val).unsqueeze(-1)
        dxdydz_val = (dxdy_val * dz_val.unsqueeze(-1))
        
        fxy_broadcast = fxy_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fxz_broadcast = fxz_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fyz_broadcast = fyz_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fxyz_broadcast = fxyz_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        P[..., 2:4, 2:4, 0:2] = fxy_broadcast * dxdy_val                      # fxy * dx * dy
        P[..., 2:4, 0:2, 2:4] = fxz_broadcast * dxdz_val                      # fxz * dx * dz
        P[..., 0:2, 2:4, 2:4] = fyz_broadcast * dydz_val                      # fyz * dy * dz
        P[..., 2:4, 2:4, 2:4] = fxyz_broadcast * dxdydz_val                   # fxyz * dx * dy * dz

        # --- 3. Compute the final interpolation result using tensor contraction (einsum) ---      
        return torch.einsum('bsli, bslj, bslk, bslijk -> bsl', H_x, H_y, H_z, P)
    
    def forward(self, coords):
        """
        Perform tricubic interpolation for a batch of coordinates.
        This simplified version only handles pre-processing and post-processing.
        
        Args:
            coords (torch.Tensor): Tensor of shape [batchSize, num_shots, 3, shot_length].

        Returns:
            torch.Tensor: Interpolated values, shape [batchSize, num_shots, 1, shot_length].
        """
        batch_size, num_shots, D, shot_length = coords.shape
        coords = coords.transpose(1,2).contiguous().view(batch_size, 1, D, num_shots*shot_length)
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
            B, S, L, I_x, I_y, I_z, t_x.to(self.dtype), t_y.to(self.dtype), t_z.to(self.dtype), dx_cell, dy_cell, dz_cell
        )

        # --- POST-PROCESSING: Reshape back to API output format ---
        return interpolated_vals.view(batch_size, 1, num_shots, shot_length).transpose(1,2)
        

if __name__ == "__ main __":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Define parameters and complex data
    B, Nx, Ny, Nz = 4, 10, 10, 10 # Batch=4, Grid=10x10x10
    S, L, D = 8, 100, 3          # Shots=2, Length=100, Dim=3
    
    # Example signal: 4D signal [B, Nx, Ny, Nz] which will become [B, S=1, Nx, Ny, Nz] internally
    # Use complex64 to test type compatibility
    signal = torch.randn(B, Nx, Ny, Nz, dtype=torch.complex64, device=device)
    
    # Example trajectory coordinates: [B, S, D, L]
    coords = (torch.rand(B, S, D, L) * 2 - 1).to(dtype=torch.float32, device=device) 
    # coords = coords.transpose(1,2).contiguous().view(B, 1, D, S*L)
    
    # Ensure coordinates require grad for autograd test
    coords.requires_grad_(True)
    
    # 2. Instantiate and interpolate
    interpolator = TricubicHermiteMAkima3D(signal)
    
    print('\n--- Running Forward Pass ---')
    output = interpolator(coords)
    output = output.view(B, S, 1, L)
    
    print(f"Input Signal Shape (B, Nx, Ny, Nz): {signal.shape}")
    print(f"Internal Signal Shape (B, S, Nx, Ny, Nz): {interpolator.signal.shape}")
    print(f"Input Coords Shape (B, S, D, L): {coords.shape}")
    print(f"Interpolated Output Shape (B, S, C=1, L): {output.shape}")
    print(f"Output Dtype: {output.dtype}")
    
    # 3. Test Autograd Compatibility
    dummy_loss = torch.sum(torch.abs(output))
    print(f"\n--- Checking Autograd ---")
    try:
        # Perform backward pass
        dummy_loss.backward()
        print("Backward pass successful. The interpolator is differentiable.")
        
        # Check if gradients were computed for the coordinates
        grad_exists = coords.grad is not None and torch.any(coords.grad != 0)
        print(f"Gradient w.r.t input coords exists: {grad_exists}")
    except RuntimeError as e:
        print(f"Backward pass failed: {e}")
