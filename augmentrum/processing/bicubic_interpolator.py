####################################################################################################
#                                 bicubic_interpolator.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2025-10-22                                                                              #
#                                                                                                  #
# Purpose: Implements BicubicHermiteMAkima3D for sampling 2D k-space/image domain volumes          #
#          using non-Cartesian trajectories.                                                       #
#                                                                                                  #
####################################################################################################

import torch
from torch import nn
from typing import Tuple

__all__ = ['BicubicHermiteMAkima2D']


class BicubicHermiteMAkima2D(nn.Module):
    """
    Bicubic Hermite Modified Akima Interpolation for 2D data.
    
    The module respects the input signal dimensions [B, S, Nx, Ny] throughout 
    initialization and interpolation, avoiding the flattening of B and S.

    B = Batch size, S = number of shots, Nx, Ny = Image dimensions
    
    NOTE: If the input signal is 3D [B, Nx, Ny], it is internally treated as 
    4D [B, S=1, Nx, Ny].
    """
    def __init__(self, signal: torch.Tensor):
        """
        Args:
            signal (torch.Tensor): 4D tensor of signal values, shape (B, S, Nx, Ny),
                                   or 3D tensor (B, Nx, Ny) if S=1.
        """
        super().__init__()
        print('Initialize Bicubic Akima 2D Interpolator...')
        
        self.dtype = signal.dtype
        self.device = signal.device
        
        # --- Dimension Check and Reshape ---
        if signal.dim() == 3:
            # If 3D, assume shape is [B, Nx, Ny] and insert S=1 dimension
            signal = signal.unsqueeze(1)
        elif signal.dim() != 4:
            raise ValueError(f"Input signal must be 3D [B, Nx, Ny] or 4D [B, S, Nx, Ny], but got {signal.dim()}D tensor.")

        # signal.shape is guaranteed to be [B, S, Nx, Ny]
        B, S, Nx, Ny = signal.shape
        self.B, self.S, self.Nx, self.Ny = B, S, Nx, Ny
        
        # Store axes and signal
        self.xaxis = torch.linspace(start=-1., end=1., steps=Nx, device=self.device)
        self.yaxis = torch.linspace(start=-1., end=1., steps=Ny, device=self.device)
        
        # Store the 4D signal [B, S, Nx, Ny]
        self.signal = signal.to(self.device)
        
        # Pre-calculate cell widths (constant for uniform axes)
        self.dx = self.xaxis[1] - self.xaxis[0]
        self.dy = self.yaxis[1] - self.yaxis[0]

        # Compute Modified Akima slopes for X and Y on the 4D signal.
        # X-slopes are computed along dim=2 (Nx). Axis shape needed: [1, 1, Nx, 1]
        self.m_x = self._prepare_slopes(
            self.signal, 
            self.xaxis.view(1, 1, Nx, 1), 
            dim=2
        )
        # Y-slopes are computed along dim=3 (Ny). Axis shape needed: [1, 1, 1, Ny]
        self.m_y = self._prepare_slopes(
            self.signal, 
            self.yaxis.view(1, 1, 1, Ny), 
            dim=3
        )
        
    def _prepare_slopes(self, signal, axis, dim):
        """
        Compute Modified Akima slopes along a given axis (dim) for a 4D tensor [B, S, Nx, Ny].
        """
        # helper to build a tuple-of-slices selecting slice(start, end) on axis `dim`
        def slc(dim, start=None, end=None):
            s = [slice(None)] * signal.dim()
            s[dim] = slice(start, end)
            return tuple(s)

        # Finite difference approximation of slope (m)
        m = (signal.diff(dim=dim) / axis.diff(dim=dim))
        
        # Extrapolate slopes at boundaries (Modified Akima 1D logic)
        d0 = 2 * m.select(dim, 0) - m.select(dim, 1)
        dn = 2 * m.select(dim, -1) - m.select(dim, -2)
        m0 = 2 * d0 - m.select(dim, 0)
        mn = 2 * dn - m.select(dim, -1)

        # Concatenate boundaries: [m0, d0, m, dn, mn]
        m = torch.cat([m0.unsqueeze(dim), d0.unsqueeze(dim), m, dn.unsqueeze(dim), mn.unsqueeze(dim)], dim=dim)
        
        # Calculate weights based on previous and next slopes
        m_prev, m_next = m[slc(dim, None, -1)], m[slc(dim, 1, None)]
        weights = torch.abs(m_next - m_prev) + torch.abs((m_prev + m_next) / 2.0)

        # Compute weighted average of slopes
        w1, w2 = weights[slc(dim, None, -2)], weights[slc(dim, 2, None)]
        delta1, delta2 = m[slc(dim, 1, -2)], m[slc(dim, 2, -1)]
        w_sum = w1 + w2 + 1e-6 # Add epsilon for stability
        
        slopes = (w2 / w_sum) * delta1 + (w1 / w_sum) * delta2
        slopes[w_sum == 0] = 0.0 # Handle zero weight case
        
        # The result shape is [B, S, Nx, Ny] (or [B, S, Nx+1, Ny] for X slopes)
        return slopes

    @staticmethod
    def h_poly(t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute cubic Hermite basis functions: h00, h10, h01, h11.
        """
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00, h10, h01, h11

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Interpolate the signal at the given k-space coordinates.
        
        Args:
            coords (torch.Tensor): K-space coordinates, shape [B, S, 2, L].
                                   Coordinates should be normalized to [-1, 1].
        
        Returns:
            torch.Tensor: Interpolated signal values, shape [B, S, C, L].
        """
        B, S, D, L = coords.shape
        coords = coords.transpose(1,2).contiguous().view(B,1,D,S*L)
        
        # Split and reshape to [B, S, L]
        x_coords, y_coords = coords.split(1, dim=2)
        x_coords = x_coords.squeeze(2) # [B, S, L]
        y_coords = y_coords.squeeze(2) # [B, S, L]
        
        # --- 1. Find indices (I) and local coordinates (t) ---
        
        # Find grid cell indices I and clamp (searchsorted is robust for uniform axes)
        I_x = (torch.searchsorted(self.xaxis, x_coords.clamp(self.xaxis.min(), self.xaxis.max())) - 1).clamp(0, self.Nx - 2)
        I_y = (torch.searchsorted(self.yaxis, y_coords.clamp(self.yaxis.min(), self.yaxis.max())) - 1).clamp(0, self.Ny - 2)
        
        # Cell widths are now pre-calculated scalars. Expand them to the required shape.
        dx_cell = self.dx.expand_as(I_x) # [B, S, L]
        dy_cell = self.dy.expand_as(I_y) # [B, S, L]
        
        # Compute local coordinates t (0 <= t < 1)
        t_x = (x_coords - self.xaxis[I_x]) / dx_cell
        t_y = (y_coords - self.yaxis[I_y]) / dy_cell
        
        # --- 2. Interpolation ---
        # Pass all dimensions to the core logic
        interpolated_vals = self._interpolate_bicubic_hermite(
            B, 1, L, I_x, I_y, t_x.to(self.dtype), t_y.to(self.dtype), dx_cell, dy_cell
        )
        
        # Reshape to final k-space data format [B, S, C, L] (C=1 implicitly)
        return interpolated_vals.view(B, 1, S, L).transpose(1,2).contiguous()


    def _interpolate_bicubic_hermite(self, B, S, L, I_x, I_y, t_x, t_y, dx_cell, dy_cell):
        """
        Vectorized core logic for bicubic Hermite interpolation.
        
        Input indices and local coordinates have shape [B, S, L].
        """
        # Compute Hermite basis functions and stack them [B, S, L, 4]
        H_x = torch.stack(self.h_poly(t_x), dim=-1)
        H_y = torch.stack(self.h_poly(t_y), dim=-1)

        # --- Subgrid Indexing ---
        # The 4 neighbours for interpolation: i-1, i, i+1, i+2
        ix_n = torch.stack([I_x - 1, I_x, I_x + 1, I_x + 2], dim=-1).clamp(0, self.Nx - 1)
        iy_n = torch.stack([I_y - 1, I_y, I_y + 1, I_y + 2], dim=-1).clamp(0, self.Ny - 1)

        # Target coefficient shape [B, S, L, 4, 4]
        
        # Batch and Shot indexing: [B, S, L, 4, 4]
        # Create index tensors with size 1 in broadcastable dimensions
        b_idx = torch.arange(B, device=self.device, dtype=torch.long).view(B, 1, 1, 1, 1) # [B, 1, 1, 1, 1]
        s_idx = torch.arange(S, device=self.device, dtype=torch.long).view(1, S, 1, 1, 1) # [1, S, 1, 1, 1]

        # X and Y grid indices: [B, S, L, 4, 4]
        ix_idx = ix_n.unsqueeze(-1) # [B, S, L, 4, 1]
        iy_idx = iy_n.unsqueeze(-2) # [B, S, L, 1, 4]

        # Signal and slope subgrids (self.signal shape is [B, S, Nx, Ny], 4D)
        # The indices will broadcast together to the target shape [B, S, L, 4, 4]
        signal_subgrid = self.signal[b_idx, s_idx, ix_idx, iy_idx]  # [B, S, L, 4, 4]
        m_x_subgrid = self.m_x[b_idx, s_idx, ix_idx, iy_idx]
        m_y_subgrid = self.m_y[b_idx, s_idx, ix_idx, iy_idx]

        # --- 1. Compute Cross-Derivative (Central Difference Approximation) ---
        
        # fxy (derivative of fx w.r.t y)
        y_plus_coords  = self.yaxis[iy_n[..., 2]] # index i+1 (y-neighbor 2/4)
        y_minus_coords = self.yaxis[iy_n[..., 0]] # index i-1 (y-neighbor 0/4)
        denom_y = (y_plus_coords - y_minus_coords).clamp(min=1e-12) # [B, S, L]
        
        # Central difference of m_x (fx) at index 1 (the cell's start point) in X, 
        # using y-neighbors (index 0 and 2)
        fx_y_plus  = m_x_subgrid[..., 1, 2] 
        fx_y_minus = m_x_subgrid[..., 1, 0]
        
        fxy_base = (fx_y_plus - fx_y_minus) / denom_y # [B, S, L]

        # --- 2. Build the $4 \times 4$ Coefficient Tensor $\mathbf{P}$ ---
        # P holds the values and scaled derivatives at the cell corners (f, fx*dx, fy*dy, fxy*dx*dy)
        P = torch.zeros(signal_subgrid.shape, dtype=self.dtype, device=self.device)
        
        # Helper variables for broadcasting cell widths
        dx = dx_cell.unsqueeze(-1) # [B, S, L, 1]
        dy = dy_cell.unsqueeze(-1) # [B, S, L, 1]
        
        # Extract the $2 \times 2$ cell corner values (indices 1:3 for a $4 \times 4$ neighborhood)
        f_c = signal_subgrid[..., 1:3, 1:3]     
        mx_c = m_x_subgrid[..., 1:3, 1:3]
        my_c = m_y_subgrid[..., 1:3, 1:3]

        # Populate the 4 sub-blocks of P 
        P[..., 0:2, 0:2] = f_c                                       # f
        P[..., 2:4, 0:2] = mx_c * dx.unsqueeze(-1)                    # fx * dx
        P[..., 0:2, 2:4] = my_c * dy.unsqueeze(-2)                    # fy * dy
        
        # fxy_base is [B, S, L]. Broadcast it to the 2x2 corners [B, S, L, 2, 2]
        P[..., 2:4, 2:4] = fxy_base.unsqueeze(-1).unsqueeze(-1) * (dx * dy).unsqueeze(-1) # fxy * dx * dy

        # --- 3. Compute the final interpolation result using tensor contraction (einsum) ---
        # Contract H_x (i), H_y (j), and P (ij)
        return torch.einsum('bsli, bslj, bslij -> bsl', H_x, H_y, P)


# --- Example Usage (similar to the 3D pipeline) ---

if __name__ == '__main__':
    # Define parameters for a 2D example
    B, S, D, L, C = 2, 5, 2, 100, 1 # Batch=2, Shots=5, D=2, Length=100, Channels=1 (implicitly)
    Nx, Ny = 32, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Create a dummy signal (2D complex image slice, single coil)
    # Shape: [B, S, Nx, Ny] -> [2, 5, 32, 32]. 
    signal_input_4d = torch.randn(B, 1, Nx, Ny, dtype=torch.complex64, device=device) 

    # Test case 2: 3D signal input [B, Nx, Ny]
    signal_input_3d = torch.randn(B, Nx, Ny, dtype=torch.complex64, device=device)

    # 2. Instantiate the Interpolator (accepts 4D signal)
    interpolator_4d = BicubicHermiteMAkima2D(signal_input_4d).to(device)
    interpolator_3d = BicubicHermiteMAkima2D(signal_input_3d).to(device)


    # 3. Dummy Coordinates (normalized k-space coordinates in [-1, 1])
    # Shape: [B, S, D, L] -> [2, 5, 2, 100]
    coords_in = (torch.rand(B, S, D, L) * 2 - 1).to(dtype=torch.float32, device=device) 

    # Ensure coordinates require grad if this is part of a training loop
    coords_in.requires_grad_(True)

    # 4. Interpolate
    print("\n--- Performing Interpolation (4D Signal Input) ---")
    kdata_interpolated_4d = interpolator_4d(coords_in) # Output shape: [B, S, C, L]

    print(f"Input Signal Shape (B, S, Nx, Ny): {signal_input_4d.shape}")
    print(f"Input Coords Shape (B, S, D, L): {coords_in.shape}")
    print(f"Interpolated K-data Shape (B, S, C=1, L): {kdata_interpolated_4d.shape}")

    # Test case 2: 3D signal input - must fail if S > 1 in coords
    coords_in_3d_test = (torch.rand(B, 1, D, L) * 2 - 1).to(dtype=torch.float32, device=device) # Must use S=1 in coords

    print("\n--- Performing Interpolation (3D Signal Input, S=1 in Coords) ---")
    kdata_interpolated_3d = interpolator_3d(coords_in_3d_test) # Output shape: [B, S=1, C, L]
    print(f"Input Signal Shape (B, Nx, Ny): {signal_input_3d.shape} -> Used as [B, S=1, Nx, Ny] internally")
    print(f"Input Coords Shape (B, S=1, D, L): {coords_in_3d_test.shape}")
    print(f"Interpolated K-data Shape (B, S=1, C=1, L): {kdata_interpolated_3d.shape}")

    # --- Check Autograd Compatibility (for training loops) ---
    dummy_loss = torch.sum(torch.abs(kdata_interpolated_4d))
    print(f"\n--- Checking Autograd ---")
    try:
        dummy_loss.backward()
        print("Backward pass successful. The interpolator is differentiable.")
        print(f"Gradient w.r.t input coords exists: {coords_in.grad is not None}")
    except RuntimeError as e:
        print(f"Backward pass failed: {e}")
