####################################################################################################
#                                        _combine_kernel.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: The Triton kernel behind the pooled coil combination: every sample's raw row read where #
#          the pool holds it, its drawn coils weighted and summed, in one pass over the batch.     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import torch
import triton
import triton.language as tl


#: Points each program combines.
BLOCK_T = 2


#*******************#
#   combine coils   #
#*******************#
@triton.jit
def combine_kernel(pool_ptr, index_ptr, w_ptr, out_ptr, V, T, C, D,
                   C_PAD: tl.constexpr, D_PAD: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    One program per (sample, voxel) and block of points: out[b, v, t, d] = sum_c w[b, v, c]
    pool[index[b], v, t, c, d], complex values as (re, im) float pairs. A point's coils and
    transients lie together, so each point is one contiguous read; a coil of zero weight (not
    drawn) is masked out of it.
    """
    bv = tl.program_id(0)
    b = bv // V
    v = bv % V
    t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.arange(0, C_PAD)
    d = tl.arange(0, D_PAD)
    k = tl.arange(0, 2)
    w = tl.load(w_ptr + (bv * C + c[:, None]) * 2 + k[None, :], mask=c[:, None] < C, other=0.0)
    w_re, w_im = tl.split(w)
    drawn = (c < C) & ((w_re != 0.0) | (w_im != 0.0))
    row = tl.load(index_ptr + b).to(tl.int64)
    src = pool_ptr + (row * V + v) * T * C * D * 2
    offset = ((t[:, None, None] * C + c[None, :, None]) * D + d[None, None, :]) * 2
    mask = (t[:, None, None] < T) & drawn[None, :, None] & (d[None, None, :] < D)
    p = tl.load(src + offset[:, :, :, None] + k[None, None, None, :], mask=mask[:, :, :, None],
                other=0.0)
    p_re, p_im = tl.split(p)
    re = tl.sum(w_re[None, :, None] * p_re - w_im[None, :, None] * p_im, axis=1)
    im = tl.sum(w_re[None, :, None] * p_im + w_im[None, :, None] * p_re, axis=1)
    dst = out_ptr + ((bv.to(tl.int64) * T + t[:, None]) * D + d[None, :]) * 2
    out_mask = (t[:, None] < T) & (d[None, :] < D)
    tl.store(dst[:, :, None] + k[None, None, :], tl.join(re, im), mask=out_mask[:, :, None])


def combine(pool, index, weights):
    """
    The coil combination of pool rows *index* (B,) with *weights* (B, V, C): *pool* complex64
    (S, X, Y, Z, T, C, D) contiguous. Returns (B, V, T, D) complex64.
    """
    s, x, y, z, t, c, d = pool.shape
    v, b = x * y * z, index.numel()
    out = torch.empty((b, v, t, d), dtype=pool.dtype, device=pool.device)
    combine_kernel[(b * v, triton.cdiv(t, BLOCK_T))](
        torch.view_as_real(pool), index.contiguous(), torch.view_as_real(weights.contiguous()),
        torch.view_as_real(out), v, t, c, d, C_PAD=triton.next_power_of_2(c),
        D_PAD=triton.next_power_of_2(d), BLOCK_T=BLOCK_T, num_warps=8)
    return out
