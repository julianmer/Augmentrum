####################################################################################################
#                                        _median_kernel.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: The Triton kernel behind the outlier removal's medians: NumPy's median over each        #
#          sample's drawn transients, every row sorted in registers.                               #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import torch
import triton
import triton.language as tl


#: Rows each program takes.
BLOCK_R = 64

#: What an entry a sample does not keep sorts as: beyond every kept one, but finite - Triton's sort
#: (3.2) misorders rows that hold infinities.
BEYOND: tl.constexpr = 3.0e38


#************#
#   median   #
#************#
@triton.jit
def median_kernel(parts_ptr, mask_ptr, count_ptr, out_ptr, R, ROWS_PER_SAMPLE, D,
                  D_PAD: tl.constexpr, BLOCK_R: tl.constexpr):
    """
    One program per block of rows: each row's median over the entries its sample keeps, the mean
    of the two middle values for an even count, the others sorted behind every kept one (as
    BEYOND, never picked: the middle positions count the kept entries only).
    """
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, D_PAD)
    sample = r // ROWS_PER_SAMPLE
    rows = r < R
    keep = tl.load(mask_ptr + sample[:, None] * D + d[None, :],
                   mask=rows[:, None] & (d[None, :] < D), other=0) != 0
    values = tl.load(parts_ptr + r[:, None].to(tl.int64) * D + d[None, :],
                     mask=rows[:, None] & keep, other=BEYOND)
    ordered = tl.sort(tl.where(keep, values, BEYOND), dim=1)
    count = tl.load(count_ptr + sample, mask=rows, other=1)
    low = tl.sum(tl.where(d[None, :] == ((count - 1) // 2)[:, None], ordered, 0.0), axis=1)
    high = tl.sum(tl.where(d[None, :] == (count // 2)[:, None], ordered, 0.0), axis=1)
    tl.store(out_ptr + r, 0.5 * (low + high), mask=rows)


def median(parts, mask):
    """
    "torch_engine._real_median": *parts* (B, ..., D) float32, *mask* (B, D) bool, at least one
    kept entry per sample. Returns (B, ...).
    """
    b, d = mask.shape
    flat = parts.contiguous().reshape(-1, d)
    out = torch.empty(flat.shape[0], dtype=parts.dtype, device=parts.device)
    median_kernel[(triton.cdiv(flat.shape[0], BLOCK_R),)](
        flat, mask.to(torch.int8).contiguous(), mask.sum(dim=-1).to(torch.int32).contiguous(),
        out, flat.shape[0], flat.shape[0] // b, d, D_PAD=triton.next_power_of_2(d),
        BLOCK_R=BLOCK_R)
    return out.reshape(parts.shape[:-1])
