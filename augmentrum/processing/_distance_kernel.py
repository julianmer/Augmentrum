####################################################################################################
#                                       _distance_kernel.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: The Triton kernel behind the alignment target: every transient's distance to the mean  #
#          of its sample's valid ones, accumulated in double precision from the single-precision   #
#          values, in a fixed order.                                                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import torch
import triton
import triton.language as tl


#: Points each program sums over.
BLOCK_T = 128


#***************#
#   distances   #
#***************#
@triton.jit
def partial_kernel(x_ptr, mask_ptr, out_ptr, D, T, CHUNKS,
                   D_PAD: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    One program per (sample, block of points): sum over the block of |x[b, d, t] - mean_t|^2 for
    every transient d, mean_t over the valid transients, all in double precision.
    """
    b = tl.program_id(0)
    chunk = tl.program_id(1)
    d = tl.arange(0, D_PAD)
    t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = tl.load(mask_ptr + b * D + d, mask=d < D, other=0) != 0
    weight = valid.to(tl.float64)
    inside = (d[:, None] < D) & (t[None, :] < T)
    base = x_ptr + ((b.to(tl.int64) * D + d[:, None]) * T + t[None, :]) * 2
    re = tl.load(base, mask=inside, other=0.0).to(tl.float64)
    im = tl.load(base + 1, mask=inside, other=0.0).to(tl.float64)
    count = tl.sum(weight, axis=0)
    mean_re = tl.sum(re * weight[:, None], axis=0) / count
    mean_im = tl.sum(im * weight[:, None], axis=0) / count
    dr = re - mean_re[None, :]
    di = im - mean_im[None, :]
    part = tl.sum(tl.where(inside, dr * dr + di * di, 0.0), axis=1)
    tl.store(out_ptr + (b * D + d) * CHUNKS + chunk, part, mask=d < D)


def distances(x, mask):
    """
    Every transient's distance to its sample's mean of *mask* (B, D) transients: *x* (B, D, T)
    complex64. Returns (B, D) float64, sqrt of the double-precision sum of squares.
    """
    b, d, t = x.shape
    chunks = triton.cdiv(t, BLOCK_T)
    parts = torch.empty((b, d, chunks), dtype=torch.float64, device=x.device)
    partial_kernel[(b, chunks)](torch.view_as_real(x.contiguous()),
                                mask.to(torch.int8).contiguous(), parts, d, t, chunks,
                                D_PAD=triton.next_power_of_2(d), BLOCK_T=BLOCK_T)
    return torch.sqrt(parts.sum(dim=-1))
