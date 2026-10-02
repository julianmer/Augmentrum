####################################################################################################
#                                         _wsvd_kernel.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: The Triton kernel behind the wSVD coil weights from a reference: every sample's noise   #
#          covariance from the cached moments of its drawn transients, its Cholesky factor, the    #
#          whitened reference Gram matrix, its principal vector and the weights, in one program    #
#          per sample, every matrix in registers.                                                  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import torch
import triton
import triton.language as tl


#*****************************#
#   complex tile arithmetic   #
#*****************************#
@triton.jit
def _load(ptr, i, n):
    """The (n, n) complex matrix at *ptr*, as (re, im) tiles, zero beyond n."""
    k = tl.arange(0, 2)
    inside = (i[:, None] < n) & (i[None, :] < n)
    offset = (i[:, None] * n + i[None, :]) * 2
    tile = tl.load(ptr + offset[:, :, None] + k[None, None, :], mask=inside[:, :, None], other=0.0)
    return tl.split(tile)


@triton.jit
def _matmul(a_re, a_im, b_re, b_im):
    """The complex product a b in single precision (no TF32)."""
    re = (tl.dot(a_re, b_re, input_precision='ieee')
          - tl.dot(a_im, b_im, input_precision='ieee'))
    im = (tl.dot(a_re, b_im, input_precision='ieee')
          + tl.dot(a_im, b_re, input_precision='ieee'))
    return re, im


@triton.jit
def _normalised(re, im, floor):
    """The matrix over its Frobenius norm (at least *floor*)."""
    norm = tl.maximum(tl.sqrt(tl.sum(tl.sum(re * re + im * im, axis=1), axis=0)), floor)
    return re / norm, im / norm


#******************#
#   wsvd weights   #
#******************#
@triton.jit
def wsvd_kernel(second_ptr, first_ptr, index_ptr, dyn_ptr, gram_ptr, gram_index_ptr, coil_ptr,
                out_ptr, V, C, D, samples, min_per_coil, floor,
                C_PAD: tl.constexpr, CONJ_NOISE: tl.constexpr, CONJ_GRAM: tl.constexpr,
                SQUARINGS: tl.constexpr, STEPS: tl.constexpr):
    """
    One program per (sample, voxel): "torch_engine.noise_covariance" over the sample's kept
    transients, then "torch_engine.wsvd_weights" with a reference, the same arithmetic in another
    order. The inverse Cholesky factor X stands in for the triangular solves: the whitened Gram
    matrix is X G X^H, and the weights, L^-H L^-1 (L v), are X^H v.
    """
    bv = tl.program_id(0)
    b = bv // V
    i = tl.arange(0, C_PAD)
    rows, cols = i[:, None], i[None, :]
    diagonal = rows == cols

    # the noise covariance of the kept transients, np.cov's
    row = tl.load(index_ptr + b).to(tl.int64)
    total_re = tl.zeros((C_PAD, C_PAD), dtype=tl.float32)
    total_im = tl.zeros((C_PAD, C_PAD), dtype=tl.float32)
    sum_re = tl.zeros((C_PAD,), dtype=tl.float32)
    sum_im = tl.zeros((C_PAD,), dtype=tl.float32)
    kept = 0.0
    for d in range(D):
        weight = (tl.load(dyn_ptr + b * D + d) != 0).to(tl.float32)
        re, im = _load(second_ptr + (row * D + d) * C * C * 2, i, C)
        total_re += weight * re
        total_im += weight * im
        vector = first_ptr + ((row * D + d) * C + i) * 2
        sum_re += weight * tl.load(vector, mask=i < C, other=0.0)
        sum_im += weight * tl.load(vector + 1, mask=i < C, other=0.0)
        kept += weight
    n = kept * samples
    outer_re = sum_re[:, None] * sum_re[None, :] + sum_im[:, None] * sum_im[None, :]
    outer_im = sum_im[:, None] * sum_re[None, :] - sum_re[:, None] * sum_im[None, :]
    cov_re = (total_re - outer_re / n) / tl.maximum(n - 1.0, 1.0)
    cov_im = (total_im - outer_im / n) / tl.maximum(n - 1.0, 1.0)
    if CONJ_NOISE:
        cov_im = -cov_im

    # the masked coils cut out (identity), or no prewhitening at all
    active = (tl.load(coil_ptr + b * C + i, mask=i < C, other=0) != 0) & (i < C)
    pair = active[:, None] & active[None, :]
    count = tl.sum(active.to(tl.int32), axis=0)
    whiten = n >= min_per_coil * count
    eye = diagonal.to(tl.float32)
    a_re = tl.where(whiten, tl.where(pair, cov_re, 0.0) + tl.where(diagonal & ~pair, 1.0, 0.0),
                    eye)
    a_im = tl.where(whiten & pair, cov_im, 0.0)

    # Cholesky, column by column, each subtracted from the block behind it
    l_re = tl.zeros((C_PAD, C_PAD), dtype=tl.float32)
    l_im = tl.zeros((C_PAD, C_PAD), dtype=tl.float32)
    for j in range(C_PAD):
        column_re = tl.sum(tl.where(cols == j, a_re, 0.0), axis=1)
        column_im = tl.sum(tl.where(cols == j, a_im, 0.0), axis=1)
        root = tl.sqrt(tl.sum(tl.where(i == j, column_re, 0.0), axis=0))
        c_re = tl.where(i > j, column_re / root, tl.where(i == j, root, 0.0))
        c_im = tl.where(i > j, column_im / root, 0.0)
        l_re = tl.where(cols == j, c_re[:, None], l_re)
        l_im = tl.where(cols == j, c_im[:, None], l_im)
        behind = (rows > j) & (cols > j)
        a_re = tl.where(behind, a_re - (c_re[:, None] * c_re[None, :]
                                        + c_im[:, None] * c_im[None, :]), a_re)
        a_im = tl.where(behind, a_im - (c_im[:, None] * c_re[None, :]
                                        - c_re[:, None] * c_im[None, :]), a_im)

    # X = L^-1, row by row
    pivots = tl.sum(tl.where(diagonal, l_re, 0.0), axis=1)
    x_re = eye
    x_im = tl.zeros((C_PAD, C_PAD), dtype=tl.float32)
    for k in range(C_PAD):
        pivot = tl.sum(tl.where(i == k, pivots, 0.0), axis=0)
        r_re = tl.sum(tl.where(rows == k, x_re, 0.0), axis=0) / pivot
        r_im = tl.sum(tl.where(rows == k, x_im, 0.0), axis=0) / pivot
        c_re = tl.sum(tl.where((cols == k) & (rows > k), l_re, 0.0), axis=1)
        c_im = tl.sum(tl.where((cols == k) & (rows > k), l_im, 0.0), axis=1)
        x_re = tl.where(rows == k, r_re[None, :],
                        x_re - (c_re[:, None] * r_re[None, :] - c_im[:, None] * r_im[None, :]))
        x_im = tl.where(rows == k, r_im[None, :],
                        x_im - (c_re[:, None] * r_im[None, :] + c_im[:, None] * r_re[None, :]))

    # the whitened reference Gram matrix X G X^H, of the active coils
    g_re, g_im = _load(gram_ptr + (tl.load(gram_index_ptr + b).to(tl.int64) * V + bv % V)
                       * C * C * 2, i, C)
    if CONJ_GRAM:
        g_im = -g_im
    g_re = tl.where(pair, g_re, 0.0)
    g_im = tl.where(pair, g_im, 0.0)
    h_re, h_im = _matmul(x_re, x_im, g_re, g_im)
    w_re, w_im = _matmul(h_re, h_im, tl.trans(x_re), -tl.trans(x_im))

    # its principal vector: repeated squaring, then power steps on the matrix itself
    p_re, p_im = _normalised(w_re, w_im, floor)
    for _ in range(SQUARINGS):
        p_re, p_im = _matmul(p_re, p_im, p_re, p_im)
        p_re, p_im = _normalised(p_re, p_im, floor)
    column = tl.argmax(tl.sqrt(tl.sum(p_re * p_re + p_im * p_im, axis=0)), axis=0)
    v_re = tl.sum(tl.where(cols == column, p_re, 0.0), axis=1)
    v_im = tl.sum(tl.where(cols == column, p_im, 0.0), axis=1)
    for _ in range(STEPS):
        u_re = tl.sum(w_re * v_re[None, :] - w_im * v_im[None, :], axis=1)
        u_im = tl.sum(w_re * v_im[None, :] + w_im * v_re[None, :], axis=1)
        norm = tl.maximum(tl.sqrt(tl.sum(u_re * u_re + u_im * u_im, axis=0)), floor)
        v_re = u_re / norm
        v_im = u_im / norm

    # the weights X^H v, scaled by |L v| and pinned in phase to the first active coil
    u_re = tl.where(active, tl.sum(l_re * v_re[None, :] - l_im * v_im[None, :], axis=1), 0.0)
    u_im = tl.where(active, tl.sum(l_re * v_im[None, :] + l_im * v_re[None, :], axis=1), 0.0)
    first = tl.min(tl.where(active, i, C_PAD), axis=0)
    u0_re = tl.sum(tl.where(i == first, u_re, 0.0), axis=0)
    u0_im = tl.sum(tl.where(i == first, u_im, 0.0), axis=0)
    scale = tl.sqrt(tl.sum(u_re * u_re + u_im * u_im, axis=0)) / tl.sqrt(u0_re * u0_re
                                                                          + u0_im * u0_im)
    s_re, s_im = scale * u0_re, -scale * u0_im
    y_re = tl.sum(x_re * v_re[:, None] + x_im * v_im[:, None], axis=0)
    y_im = tl.sum(x_re * v_im[:, None] - x_im * v_re[:, None], axis=0)
    out_re = tl.where(active, y_re * s_re - y_im * s_im, 0.0)
    out_im = tl.where(active, y_re * s_im + y_im * s_re, 0.0)
    out_re = tl.where(count == 1, tl.where(i == first, 1.0, 0.0), out_re)
    out_im = tl.where(count == 1, 0.0, out_im)
    dst = out_ptr + (bv * C + i) * 2
    tl.store(dst, out_re, mask=i < C)
    tl.store(dst + 1, out_im, mask=i < C)


def weights(second, first, index, dyn_mask, samples_per_transient, gram, gram_index, coil_mask,
            conj_noise, conj_gram, min_per_coil, squarings=12, steps=2):
    """
    "torch_engine.noise_covariance" and "torch_engine.wsvd_weights" with a reference, in one
    launch: noise moments *second* (S, D, C, C) and *first* (S, D, C) of rows *index* (B,),
    transients kept by *dyn_mask* (B, D), reference Gram matrices *gram* (R, V, C, C) of rows
    *gram_index* (B,), coils kept by *coil_mask* (B, C), all complex64 or bool; *conj_noise* and
    *conj_gram* conjugate the moments and the Gram matrices. Returns the weights (B, V, C).
    """
    s, d, c = first.shape
    b, v = index.numel(), gram.shape[1]
    if dyn_mask is None:
        dyn_mask = torch.ones((b, d), dtype=torch.bool, device=gram.device)
    if coil_mask is None:
        coil_mask = torch.ones((b, c), dtype=torch.bool, device=gram.device)
    out = torch.empty((b, v, c), dtype=gram.dtype, device=gram.device)
    wsvd_kernel[(b * v,)](
        torch.view_as_real(second.contiguous()), torch.view_as_real(first.contiguous()),
        index.contiguous(), dyn_mask.to(torch.int8).contiguous(),
        torch.view_as_real(gram.contiguous()), gram_index.contiguous(),
        coil_mask.to(torch.int8).contiguous(), torch.view_as_real(out), v, c, d,
        float(samples_per_transient), float(min_per_coil), torch.finfo(torch.float32).tiny,
        C_PAD=max(16, triton.next_power_of_2(c)), CONJ_NOISE=bool(conj_noise),
        CONJ_GRAM=bool(conj_gram), SQUARINGS=squarings, STEPS=steps)
    return out
