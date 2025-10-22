####################################################################################################
#                                 kspace_reconstructor.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2025-10-22                                                                              #
#                                                                                                  #
# Purpose: Implements reconstruct_with_masking which accepts subsampled k-space data and its       #
#          trajectory coordinates to perform density compesnation and regridding.                  #
#                                                                                                  #
####################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchkbnufft as tkbn
from typing import Union, List, Tuple


__all__ = ['reconstruct_with_masking']


# --- Constants for normalization ---
# TorchKbNufft expects coordinates in the range [-pi, pi]
PI = torch.pi


def prepare_data_and_coords(
    coords: torch.Tensor, 
    kdata: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reshapes coordinates and k-data to the format required by TorchKbNufft:
    Coords: [B, D, N_total], Kdata: [B, C, N_total]
    N_total = S * L (shots x length).
    """
    # Assuming input coords shape: [B, S, D, L] (e.g., [4, 10, 3, 100])
    B, S, D, L = coords.shape
    N_total = S * L

    # 1. Normalize Coordinates (if not already done)
    # The NUFFT forward/adjoint operators typically assume coordinates in [-pi, pi].
    # We assume your input coords are normalized to [-1, 1] relative to k-space extent.
    coords_normalized = coords * PI 

    # 2. Flatten Trajectory: [B, S, D, L] -> [B, D, S, L] -> [B, D, N_total]
    k_traj_flat = coords_normalized.permute(0, 2, 1, 3).reshape(B, D, N_total)

    kdata_flat = None
    if kdata is not None:
        # Assuming input kdata shape: [B, S, C, L] (e.g., [4, 10, 1, 100])
        B_k, S_k, C, L_k = kdata.shape
        
        # Flatten K-data: [B, S, C, L] -> [B, C, S, L] -> [B, C, N_total]
        kdata_flat = kdata.permute(0, 2, 1, 3).reshape(B, C, N_total)

    return k_traj_flat, kdata_flat

def calculate_nufft_dcf_weights(
    coords: torch.Tensor,
    image_size: Union[List[int], Tuple[int]],
    oversampling_factor: Union[List[float], Tuple[float]]
) -> torch.Tensor:
    """
    Calculates Density Compensation Function (DCF) weights using the
    iterative Pipe-style method from torchkbnufft.
    """
    device = coords.device
    B, S, D, L = coords.shape
    
    # 1. Prepare Trajectory Coordinates
    k_traj_flat, _ = prepare_data_and_coords(coords) # [B, D, N_total]

    # 2. Determine Grid Size based on Image Size and Oversampling
    # Ensure all sizes are determined on the CPU and converted to a list/tuple of native types
    
    # Get sizes as native types (no longer need to keep them as tensors)
    image_size_list = list(image_size)
    oversampling_factor_tensor = torch.tensor(oversampling_factor, dtype=torch.float32)
    
    # Grid size = floor(Image Size * Oversampling Factor)
    grid_size_tensor = torch.floor(torch.tensor(image_size) * oversampling_factor_tensor).to(torch.long)
    grid_size_list = grid_size_tensor.tolist() # <--- Pass native list/tuple

    # 3. Calculate DCF (Iterative Pipe-style)
    # Pass 'im_size' and 'grid_size' as separate arguments, not the KbNufft object.
    with torch.no_grad():
        dcf_flat = tkbn.calc_density_compensation_function(
            k_traj_flat, 
            im_size=image_size_list,     # <--- FIX 1: Pass im_size
            grid_size=grid_size_list     # <--- FIX 2: Pass grid_size
        ).detach() # [B, 1, N_total]

    # 4. Reshape DCF for easy multiplication with k-data: [B, 1, S*L] -> [B, S, L]
    dcf_weights = dcf_flat.view(B, S, L)

    return dcf_weights


def regrid_kdata_to_image(
    weighted_kdata: torch.Tensor,
    coords: torch.Tensor,
    image_size: Union[List[int], Tuple[int]],
    oversampling_factor: Union[List[float], Tuple[float]]
) -> torch.Tensor:
    """
    Performs Adjoint NUFFT (regridding) to map weighted k-space data to the image domain.

    Args:
        weighted_kdata (torch.Tensor): Density-compensated k-space data, shape [B, S, C, L].
        coords (torch.Tensor): K-space trajectory coordinates, shape [B, S, D, L].
        image_size (Union[List[int], Tuple[int]]): Target image dimensions (Nx, Ny, Nz).
        oversampling_factor (Union[List[float], Tuple[float]]): Oversampling factors 
                                                                for the regridding grid (rho_x, rho_y, rho_z).

    Returns:
        torch.Tensor: Reconstructed complex image volume, shape [B, C, Nx, Ny, Nz].
    """
    device = weighted_kdata.device
    
    # 1. Prepare Coords and K-data for NUFFT
    # k_traj_flat: [B, D, N_total], kdata_flat: [B, C, N_total]
    k_traj_flat, kdata_flat = prepare_data_and_coords(coords, weighted_kdata)
    
    # 2. Determine Grid Size
    image_size = torch.tensor(image_size, dtype=torch.long, device=device)
    oversampling_factor = torch.tensor(oversampling_factor, dtype=torch.float32, device=device)
    
    # Grid size = floor(Image Size * Oversampling Factor)
    grid_size = torch.floor(image_size * oversampling_factor).to(torch.long)

    # 3. Instantiate Adjoint NUFFT Operator (Regridding)
    adj_op = tkbn.KbNufftAdjoint(
        im_size=image_size.tolist(),  # [Nx, Ny, Nz]
        grid_size=grid_size.tolist()  # [Grid_Nx, Grid_Ny, Grid_Nz]
    ).to(device)

    # 4. Perform Regridding
    # The operation is: Adjoint(k_data) -> IFFT -> Image
    image_domain_output = adj_op(kdata_flat, k_traj_flat)

    # Output shape: [B, C, Nx, Ny, Nz]
    return image_domain_output


def reconstruct_with_masking(
    coords: torch.Tensor,
    kdata: torch.Tensor,
    undersampling_mask: torch.Tensor,
    image_size: Union[List[int], Tuple[int]],
    oversampling_factor: Union[List[float], Tuple[float]]
) -> torch.Tensor:
    """
    High-level wrapper to apply undersampling mask, calculate DCF, and perform regridding.

    Args:
        coords (torch.Tensor): K-space trajectory coordinates, shape [B, S, D, L].
        kdata (torch.Tensor): Raw k-space data, shape [B, S, C, L].
        undersampling_mask (torch.Tensor): Boolean mask indicating which shots to KEEP. 
                                           Shape: [B, S].
        image_size (Union[List[int], Tuple[int]]): Target image dimensions (Nx, Ny, Nz).
        oversampling_factor (Union[List[float], Tuple[float]]): Oversampling factors 
                                                                for the regridding grid (rho_x, rho_y, rho_z).

    Returns:
        torch.Tensor: Reconstructed complex image volume, shape [B, C, Nx, Ny, Nz].
    """    
    # --- 1. Apply Undersampling Mask (Zero-Filling) ---
    # The mask is [B, S]. We need to expand it to [B, S, C, L] to multiply with kdata.
    B, S, C, L = kdata.shape
    
    # Expand mask to match kdata shape: [B, S] -> [B, S, 1, 1] -> [B, S, C, L]
    # Invert the boolean mask to create a float mask (True -> 0, False -> 1)
    # Since the mask is for samples to KEEP (conventionally True), 
    # we zero the kdata samples where the mask is False (discard).
    
    # Convert mask to float tensor (0 or 1) and expand
    mask_float = undersampling_mask.to(dtype=kdata.real.dtype).unsqueeze(2).unsqueeze(3) 
    
    # Expand across Coil (C) and Length (L) dimensions
    mask_bcast = mask_float.expand(B, S, C, L)
    
    # Apply mask (zero out discarded k-space samples/shots)
    masked_kdata = kdata * mask_bcast

    # --- 2. Calculate DCF Weights ---
    # DCF calculation should use ALL coordinates to reflect the desired density.
    dcf_weights = calculate_nufft_dcf_weights(
        coords=coords, 
        image_size=image_size, 
        oversampling_factor=oversampling_factor
    )
    
    # Reshape DCF for multiplication: [B, S, L] -> [B, S, 1, L]
    dcf_bcast = dcf_weights.unsqueeze(2) 
    
    # --- 3. Apply DCF to Masked K-space Data ---
    weighted_kdata = masked_kdata * dcf_bcast 
    
    # --- 4. Perform Regridding (Adjoint NUFFT) ---
    reconstructed_image = regrid_kdata_to_image(
        weighted_kdata=weighted_kdata,
        coords=coords,
        image_size=image_size,
        oversampling_factor=oversampling_factor
    )
    
    return reconstructed_image


if __name__ == "__ main __":
    # Define parameters for a 3D example
    B, S, D, L, C = 4, 10, 3, 100, 1 # Batch, Shots, Dimensions, Length, Channels
    Target_Image_Size = (64, 64, 64)
    Oversampling_Factor = (1.5, 1.5, 1.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dummy Data: Coords must be float, Kdata must be complex
    coords_in = (torch.rand(B, S, D, L) * 2 - 1).to(dtype=torch.float32, device=device) # [-1, 1] range
    kdata_in = torch.randn(B, S, C, L, dtype=torch.complex64, device=device) 
    
    # --- NEW: Create an Undersampling Mask ---
    # Mask [B, S] (e.g., keeping only 50% of the shots randomly per batch sample)
    undersampling_mask_in = torch.rand(B, S) < 0.5
    undersampling_mask_in = undersampling_mask_in.to(device)

    # --- 1. Full Reconstruction Pipeline with Masking ---
    print("--- Starting Full Reconstruction Pipeline (Masked) ---")
    reconstructed_image_masked = reconstruct_with_masking(
        coords=coords_in,
        kdata=kdata_in,
        undersampling_mask=undersampling_mask_in,
        image_size=Target_Image_Size,
        oversampling_factor=Oversampling_Factor
    )
    
    print(f"Masked Reconstruction successful. Final Image Shape: {reconstructed_image_masked.shape}")
    print(f"Shots kept per batch: {undersampling_mask_in.sum(dim=1).tolist()}")
    print(f"Expected Final Image Shape: [{B}, {C}, {Target_Image_Size[0]}, {Target_Image_Size[1]}, {Target_Image_Size[2]}]")

    # The reconstructed image is now a complex tensor [B, C, Nx, Ny, Nz]

    # Define parameters for a 2D example
    B, S, D, L, C = 4, 10, 2, 100, 1 # Batch, Shots, Dimensions, Length, Channels
    Target_Image_Size = (64, 64)
    Oversampling_Factor = (1.5, 1.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy Data: Coords must be float, Kdata must be complex
    coords_in = (torch.rand(B, S, D, L) * 2 - 1).to(dtype=torch.float32, device=device) # [-1, 1] range
    kdata_in = torch.randn(B, S, C, L, dtype=torch.complex64, device=device) 
    
    # --- NEW: Create an Undersampling Mask ---
    # Mask [B, S] (e.g., keeping only 50% of the shots randomly per batch sample)
    undersampling_mask_in = torch.rand(B, S) < 0.5
    undersampling_mask_in = undersampling_mask_in.to(device)
    
    # --- 1. Full Reconstruction Pipeline with Masking ---
    print("--- Starting Full Reconstruction Pipeline (Masked) ---")
    reconstructed_image_masked = reconstruct_with_masking(
        coords=coords_in,
        kdata=kdata_in,
        undersampling_mask=undersampling_mask_in,
        image_size=Target_Image_Size,
        oversampling_factor=Oversampling_Factor
    )
    
    print(f"Masked Reconstruction successful. Final Image Shape: {reconstructed_image_masked.shape}")
    print(f"Shots kept per batch: {undersampling_mask_in.sum(dim=1).tolist()}")
    print("Expected Final Image Shape: [{}, {}, {}]".format(B, C, [x for x in Target_Image_Size]))
    
    # The reconstructed image is now a complex tensor [B, C, Nx, Ny]
