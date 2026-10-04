####################################################################################################
#                                            kernels.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-21                                                                              #
#                                                                                                  #
# Purpose: The Triton kernels of Augmentrum's torch paths, each in place of the many small         #
#          kernels its torch operations launch. RawProcessor's torch engine ("torch_engine"): the  #
#          wSVD coil weights, the coil combination of a pooled batch, the alignment target's       #
#          distances, the alignment cost's sums and the whole alignment search, and the outlier    #
#          medians and statistics. Noise: the spectrum peak of every sample, and white draws       #
#          scaled, coupled across the coils and added, in one pass each. A module of its own,      #
#          imported on first use, so that Triton stays optional.                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


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


def wsvd_weights(second, first, index, dyn_mask, samples_per_transient, gram, gram_index,
                 coil_mask, conj_noise, conj_gram, min_per_coil, squarings=12, steps=2):
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


#*******************#
#   combine coils   #
#*******************#
#: Points each program combines.
COMBINE_POINTS = 2


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
    combine_kernel[(b * v, triton.cdiv(t, COMBINE_POINTS))](
        torch.view_as_real(pool), index.contiguous(), torch.view_as_real(weights.contiguous()),
        torch.view_as_real(out), v, t, c, d, C_PAD=triton.next_power_of_2(c),
        D_PAD=triton.next_power_of_2(d), BLOCK_T=COMBINE_POINTS, num_warps=8)
    return out


#***************#
#   distances   #
#***************#
#: Points each program sums over.
DISTANCE_POINTS = 128


@triton.jit
def distance_kernel(x_ptr, mask_ptr, out_ptr, D, T, CHUNKS, SB, SD, ST,
                    D_PAD: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    One program per (sample, block of points): sum over the block of |x[b, d, t] - mean_t|^2 for
    every transient d, mean_t over the valid transients, all in double precision; x[b, d, t] at
    b SB + d SD + t ST (complex elements).
    """
    b = tl.program_id(0)
    chunk = tl.program_id(1)
    d = tl.arange(0, D_PAD)
    t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = tl.load(mask_ptr + b * D + d, mask=d < D, other=0) != 0
    weight = valid.to(tl.float64)
    inside = (d[:, None] < D) & (t[None, :] < T)
    base = x_ptr + (b.to(tl.int64) * SB + d[:, None] * SD + t[None, :] * ST) * 2
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
    complex64, read where it lies. Returns (B, D) float64, sqrt of the double-precision sum of
    squares.
    """
    b, d, t = x.shape
    chunks = triton.cdiv(t, DISTANCE_POINTS)
    parts = torch.empty((b, d, chunks), dtype=torch.float64, device=x.device)
    distance_kernel[(b, chunks)](torch.view_as_real(x), mask.to(torch.int8).contiguous(), parts,
                                 d, t, chunks, *x.stride(), D_PAD=triton.next_power_of_2(d),
                                 BLOCK_T=DISTANCE_POINTS)
    return torch.sqrt(parts.sum(dim=-1))


#****************#
#   shift sums   #
#****************#
#: Points each program reads per iteration.
SHIFT_POINTS = 256


@triton.jit
def shift_sums_kernel(rows_ptr, nu_ptr, cos_ptr, sin_ptr, n, two_pi,
                      R: tl.constexpr, R_PAD: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per transient m: "sum_t rows[m, r, t] cos(theta_t)" and the same with sin, for
    every row r, with theta_t = (two_pi t) nu[m] formed in float32 as torch forms it.
    """
    m = tl.program_id(0)
    nu = tl.load(nu_ptr + m)
    r = tl.arange(0, R_PAD)
    acc_c = tl.zeros([R_PAD, BLOCK], dtype=tl.float32)
    acc_s = tl.zeros([R_PAD, BLOCK], dtype=tl.float32)
    base = rows_ptr + m.to(tl.int64) * R * n
    for start in range(0, n, BLOCK):
        t = start + tl.arange(0, BLOCK)
        theta = (two_pi * t.to(tl.float32)) * nu
        mask = (r[:, None] < R) & (t[None, :] < n)
        vals = tl.load(base + r[:, None] * n + t[None, :], mask=mask, other=0.0)
        acc_c += vals * libdevice.cos(theta)[None, :]
        acc_s += vals * libdevice.sin(theta)[None, :]
    out = m * R + r
    tl.store(cos_ptr + out, tl.sum(acc_c, axis=1), mask=r < R)
    tl.store(sin_ptr + out, tl.sum(acc_s, axis=1), mask=r < R)


def shift_sums(rows, nu, cos, sin):
    """Run the kernel: *rows* (M, R, T) float32 contiguous, *nu* (M,), outputs (M, R)."""
    m, r, n = rows.shape
    shift_sums_kernel[(m,)](rows, nu, cos, sin, n, 6.283185307179586, R=r,
                            R_PAD=max(triton.next_power_of_2(r), 2), BLOCK=SHIFT_POINTS)


#*******************#
#   shift profile   #
#*******************#
#: Points each program of the profile rows takes.
PROFILE_POINTS = 64

#: Points each program of the power takes.
POWER_POINTS = 1024


@triton.jit
def profile_rows_kernel(x_ptr, window_ptr, plain_ptr, padded_ptr, D, N, SB, SD, ST,
                        D_PAD: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per (sample b, block of points): its transients x[b, d, t] (at b SB + d SD + t ST)
    with the first point halved, the rows h = x conj(window[b]) into plain[b, d, 0:2], and x into
    the first half of its zero-padded row; complex products rounded as torch's.
    """
    b = tl.program_id(0)
    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D_PAD)
    inside = (d[:, None] < D) & (t[None, :] < N)
    at = (b.to(tl.int64) * SB + d[:, None] * SD + t[None, :] * ST) * 2
    x_re = tl.load(x_ptr + at, mask=inside, other=0.0)
    x_im = tl.load(x_ptr + at + 1, mask=inside, other=0.0)
    first = t[None, :] == 0
    x_re, x_im = (tl.where(first, tl.fma(x_re, 0.5, -(x_im * 0.0)), x_re),
                  tl.where(first, tl.fma(x_re, 0.0, x_im * 0.5), x_im))
    w_re = tl.load(window_ptr + (b * N + t) * 2, mask=t < N, other=0.0)[None, :]
    w_im = -tl.load(window_ptr + (b * N + t) * 2 + 1, mask=t < N, other=0.0)[None, :]
    rows = (b.to(tl.int64) * D + d[:, None]) * 4 * N + t[None, :]
    tl.store(plain_ptr + rows, tl.fma(x_re, w_re, -(x_im * w_im)), mask=inside)
    tl.store(plain_ptr + rows + N, tl.fma(x_re, w_im, x_im * w_re), mask=inside)
    pad = ((b.to(tl.int64) * D + d[:, None]) * 2 * N + t[None, :]) * 2
    tl.store(padded_ptr + pad, x_re, mask=inside)
    tl.store(padded_ptr + pad + 1, x_im, mask=inside)
    tl.store(padded_ptr + pad + 2 * N, 0.0 * x_re, mask=inside)
    tl.store(padded_ptr + pad + 2 * N + 1, 0.0 * x_re, mask=inside)


@triton.jit
def power_kernel(z_ptr, out_ptr, M, BLOCK: tl.constexpr):
    """One program per block: |z|^2 = re^2 + im^2 of complex z, each square and the sum rounded."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    re = tl.load(z_ptr + i.to(tl.int64) * 2, mask=i < M, other=0.0)
    im = tl.load(z_ptr + i.to(tl.int64) * 2 + 1, mask=i < M, other=0.0)
    tl.store(out_ptr + i, re * re + im * im, mask=i < M)


@triton.jit
def autocorr_rows_kernel(a_ptr, kernel_ptr, plain_ptr, q0_ptr, N, CONJ: tl.constexpr,
                         BLOCK: tl.constexpr):
    """
    One program per (transient m, block of lags): q = a[m, :N] kernel (a complex product rounded
    as torch's; a conjugated first with CONJ), its lag 0 into q0[m] and zeroed, the rest into
    plain[m, 2:4].
    """
    m = tl.program_id(0)
    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = t < N
    a = a_ptr + (m.to(tl.int64) * 2 * N + t) * 2
    a_re = tl.load(a, mask=inside, other=0.0)
    a_im = tl.load(a + 1, mask=inside, other=0.0)
    if CONJ:
        a_im = -a_im
    k_re = tl.load(kernel_ptr + t * 2, mask=inside, other=0.0)
    k_im = tl.load(kernel_ptr + t * 2 + 1, mask=inside, other=0.0)
    q_re = tl.fma(a_re, k_re, -(a_im * k_im))
    q_im = tl.fma(a_re, k_im, a_im * k_re)
    tl.store(q0_ptr + m + t * 0, q_re, mask=t == 0)
    rows = m.to(tl.int64) * 4 * N + 2 * N + t
    tl.store(plain_ptr + rows, tl.where(t == 0, 0.0, q_re), mask=inside)
    tl.store(plain_ptr + rows + N, tl.where(t == 0, 0.0, q_im), mask=inside)


def shift_profile(x, window, kernel):
    """
    "torch_engine.ShiftProfile"'s rows of complex64 transients *x* (B, D, T), read where they lie,
    against their sample's time-domain *window* (B, T) and the window *kernel* (T,): "(plain
    (B, D, 4, T) float32, q0 (B, D))", bit for bit; the two transforms are torch's own.
    """
    b, d, n = x.shape
    plain = torch.empty((b, d, 4, n), dtype=torch.float32, device=x.device)
    padded = torch.empty((b, d, 2 * n), dtype=x.dtype, device=x.device)
    profile_rows_kernel[(b, triton.cdiv(n, PROFILE_POINTS))](
        torch.view_as_real(x), torch.view_as_real(window.contiguous()), plain,
        torch.view_as_real(padded), d, n, *x.stride(), D_PAD=triton.next_power_of_2(d),
        BLOCK=PROFILE_POINTS, enable_fp_fusion=False)
    spectrum = torch.fft.fft(padded, dim=-1)
    power = torch.empty(spectrum.shape, dtype=torch.float32, device=x.device)
    power_kernel[(triton.cdiv(power.numel(), POWER_POINTS),)](
        torch.view_as_real(spectrum), power, power.numel(), BLOCK=POWER_POINTS,
        enable_fp_fusion=False)
    autocorr = torch.fft.ifft(power, dim=-1)        # of a real input, possibly a conjugate view
    conj = autocorr.is_conj()
    q0 = torch.empty((b, d), dtype=torch.float32, device=x.device)
    autocorr_rows_kernel[(b * d, triton.cdiv(n, POWER_POINTS))](
        torch.view_as_real(autocorr.conj() if conj else autocorr),
        torch.view_as_real(kernel.contiguous()), plain, q0, n, CONJ=conj, BLOCK=POWER_POINTS,
        enable_fp_fusion=False)
    return plain, q0


#***********************#
#   apply the alignment   #
#***********************#
@triton.jit
def aligned_kernel(x_ptr, phi_ptr, eps_ptr, t_ptr, out_ptr, D, N, SB, SD, ST, OB, OD, OT,
                   two_pi, D_PAD: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per (sample b, block of points): every transient x[b, d] (at b SB + d SD + t ST)
    times e^{-i phi - 2 pi i t eps} on the time axis *t*, the angle, its cosine and sine and the
    complex product rounded as "torch_engine.alignment_phasor" and torch's product round them,
    into out (at b OB + d OD + t OT).
    """
    b = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D_PAD)
    inside = (d[:, None] < D) & (j[None, :] < N)
    phi = tl.load(phi_ptr + b * D + d, mask=d < D, other=0.0)
    eps = tl.load(eps_ptr + b * D + d, mask=d < D, other=0.0)
    t = tl.load(t_ptr + j, mask=j < N, other=0.0)
    angle = (-phi)[:, None] - (t * two_pi)[None, :] * eps[:, None]
    c = libdevice.cos(angle)
    s = libdevice.sin(angle)
    at = (b.to(tl.int64) * SB + d[:, None] * SD + j[None, :] * ST) * 2
    x_re = tl.load(x_ptr + at, mask=inside, other=0.0)
    x_im = tl.load(x_ptr + at + 1, mask=inside, other=0.0)
    to = (b.to(tl.int64) * OB + d[:, None] * OD + j[None, :] * OT) * 2
    tl.store(out_ptr + to, tl.fma(x_re, c, -(x_im * s)), mask=inside)
    tl.store(out_ptr + to + 1, tl.fma(x_re, s, x_im * c), mask=inside)


def aligned(x, phi, eps, t):
    """
    Complex64 transients *x* (B, D, T), read where they lie, times "alignment_phasor" of *phi*,
    *eps* (B, D) on the time axis *t* (T,), in one pass; the result laid out as torch's product
    lays it out (empty_like).
    """
    b, d, n = x.shape
    out = torch.empty_like(x)
    aligned_kernel[(b, triton.cdiv(n, PROFILE_POINTS))](
        torch.view_as_real(x), phi.contiguous(), eps.contiguous(), t, torch.view_as_real(out), d,
        n, *x.stride(), *out.stride(), 2 * math.pi, D_PAD=triton.next_power_of_2(d),
        BLOCK=PROFILE_POINTS, enable_fp_fusion=False)
    return out


#***************#
#   ecc phase   #
#***************#
#: Points each program of the ECC kernels takes.
ECC_POINTS = 512


@triton.jit
def unwrap_steps_kernel(ref_ptr, angle_ptr, correct_ptr, N, pi, two_pi, BLOCK: tl.constexpr):
    """
    One program per (row r, block of points): the angle of ref[r] and, for every step into a
    point, numpy.unwrap's correction ("torch_engine.unwrap"), each operation rounded as its torch
    kernel rounds it (the remainder as fmod, moved to the divisor's sign).
    """
    r = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    at = (r.to(tl.int64) * N + j) * 2
    angle = libdevice.atan2(tl.load(ref_ptr + at + 1, mask=j < N, other=0.0),
                            tl.load(ref_ptr + at, mask=j < N, other=1.0))
    tl.store(angle_ptr + r.to(tl.int64) * N + j, angle, mask=j < N)
    step = (j >= 1) & (j < N)
    before = libdevice.atan2(tl.load(ref_ptr + at - 1, mask=step, other=0.0),
                             tl.load(ref_ptr + at - 2, mask=step, other=1.0))
    diff = angle - before
    wrapped = libdevice.fmod(diff + pi, two_pi)
    wrapped = tl.where((wrapped != 0) & ((two_pi < 0) != (wrapped < 0)), wrapped + two_pi,
                       wrapped) - pi
    wrapped = tl.where((wrapped == -pi) & (diff > 0), pi, wrapped)
    correct = tl.where(tl.abs(diff) < pi, 0.0, wrapped - diff)
    tl.store(correct_ptr + r.to(tl.int64) * (N - 1) + j - 1, correct, mask=step)


@triton.jit
def unwrapped_kernel(angle_ptr, cum_ptr, out_ptr, N, ROW, LEFT, BLOCK: tl.constexpr):
    """
    One program per (row r, block of points): the unwrapped phase, angle[r, 0] and then
    angle[r, j] + cum[r, j - 1] ("torch_engine.unwrap"), into out[r, LEFT + j] (rows ROW apart).
    """
    r = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    angle = tl.load(angle_ptr + r.to(tl.int64) * N + j, mask=j < N, other=0.0)
    cum = tl.load(cum_ptr + r.to(tl.int64) * (N - 1) + j - 1, mask=(j >= 1) & (j < N),
                  other=0.0)
    tl.store(out_ptr + r.to(tl.int64) * ROW + LEFT + j, tl.where(j == 0, angle, angle + cum),
             mask=j < N)


@triton.jit
def rotated_kernel(x_ptr, phase_ptr, out_ptr, N, XR, XT, OR, OT, BLOCK: tl.constexpr):
    """
    One program per (row r, block of points): x[r] e^{-i phase[r]} (rows of x and out at r XR,
    r OR, points XT, OT apart), the polar factor and the product rounded as torch's.
    """
    r = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    p = -tl.load(phase_ptr + r.to(tl.int64) * N + j, mask=j < N, other=0.0)
    c = libdevice.cos(p)
    s = libdevice.sin(p)
    at = (r.to(tl.int64) * XR + j * XT) * 2
    x_re = tl.load(x_ptr + at, mask=j < N, other=0.0)
    x_im = tl.load(x_ptr + at + 1, mask=j < N, other=0.0)
    to = (r.to(tl.int64) * OR + j * OT) * 2
    tl.store(out_ptr + to, tl.fma(x_re, c, -(x_im * s)), mask=j < N)
    tl.store(out_ptr + to + 1, tl.fma(x_re, s, x_im * c), mask=j < N)


def unwrapped_padded(ref, left, right):
    """
    The unwrapped phase of complex64 rows *ref* (R, T) ("torch_engine.unwrap" of their angle)
    in the middle of rows (R, left + T + right), whose margins are left for the caller.
    """
    rows, n = ref.shape
    ref = ref.contiguous()
    angle = torch.empty((rows, n), dtype=torch.float32, device=ref.device)
    correct = torch.empty((rows, n - 1), dtype=torch.float32, device=ref.device)
    grid = (rows, triton.cdiv(n, ECC_POINTS))
    unwrap_steps_kernel[grid](torch.view_as_real(ref), angle, correct, n, math.pi, 2 * math.pi,
                              BLOCK=ECC_POINTS, enable_fp_fusion=False)
    cum = torch.cumsum(correct, dim=-1)
    padded = torch.empty((rows, left + n + right), dtype=torch.float32, device=ref.device)
    unwrapped_kernel[grid](angle, cum, padded, n, padded.shape[1], left, BLOCK=ECC_POINTS)
    return padded


def rotated(x, phase):
    """Complex64 rows *x* (R, T), any strides, times e^{-i phase} of float32 *phase* (R, T),
    contiguous, in one pass; the result laid out as x (empty_like)."""
    rows, n = x.shape
    out = torch.empty_like(x)
    rotated_kernel[(rows, triton.cdiv(n, ECC_POINTS))](
        torch.view_as_real(x), phase, torch.view_as_real(out), n, *x.stride(), *out.stride(),
        BLOCK=ECC_POINTS, enable_fp_fusion=False)
    return out


#*******************#
#   peak searches   #
#*******************#
@triton.jit
def halved_padded_kernel(x_ptr, out_ptr, N, M, XR, XT, BLOCK: tl.constexpr):
    """
    One program per (row r, block of points): x[r] (at r XR + t XT) with its first point halved
    as torch's complex product by 0.5 halves it, zero-filled to M points.
    """
    r = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    at = (r.to(tl.int64) * XR + j * XT) * 2
    re = tl.load(x_ptr + at, mask=j < N, other=0.0)
    im = tl.load(x_ptr + at + 1, mask=j < N, other=0.0)
    re, im = (tl.where(j == 0, tl.fma(re, 0.5, -(im * 0.0)), re),
              tl.where(j == 0, tl.fma(re, 0.0, im * 0.5), im))
    to = (r.to(tl.int64) * M + j) * 2
    tl.store(out_ptr + to, re, mask=j < M)
    tl.store(out_ptr + to + 1, im, mask=j < M)


@triton.jit
def _window_peak(spec_ptr, r, M, FIRST, LAST, W: tl.constexpr):
    """The first bin of the largest |.| in [FIRST, LAST) of row r's fftshifted spectrum (read
    unshifted), and that bin's value."""
    k = FIRST + tl.arange(0, W)
    valid = k < LAST
    at = (r.to(tl.int64) * M + (k + M // 2) % M) * 2
    re = tl.load(spec_ptr + at, mask=valid, other=0.0)
    im = tl.load(spec_ptr + at + 1, mask=valid, other=0.0)
    size = tl.where(valid, libdevice.hypot(re, im), -1.0)
    top = tl.max(size, axis=0)
    peak = tl.min(tl.where(size == top, k, LAST), axis=0)
    at = (r.to(tl.int64) * M + (peak + M // 2) % M) * 2
    return peak, tl.load(spec_ptr + at), tl.load(spec_ptr + at + 1)


@triton.jit
def peak_phased_kernel(spec_ptr, x_ptr, out_ptr, N, M, FIRST, LAST, XR, XT, OR, OT,
                       W: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per (row r, block of points): x[r] times e^{i phi}, phi minus the angle of the
    window's peak ("torch_engine.peak_phase", FSL-MRS phaseCorrect), rounded as torch's.
    """
    r = tl.program_id(0)
    _, p_re, p_im = _window_peak(spec_ptr, r, M, FIRST, LAST, W)
    phi = -libdevice.atan2(p_im, p_re)
    c = libdevice.cos(phi)
    s = libdevice.sin(phi)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    at = (r.to(tl.int64) * XR + j * XT) * 2
    x_re = tl.load(x_ptr + at, mask=j < N, other=0.0)
    x_im = tl.load(x_ptr + at + 1, mask=j < N, other=0.0)
    to = (r.to(tl.int64) * OR + j * OT) * 2
    tl.store(out_ptr + to, tl.fma(x_re, c, -(x_im * s)), mask=j < N)
    tl.store(out_ptr + to + 1, tl.fma(x_re, s, x_im * c), mask=j < N)


@triton.jit
def peak_shifted_kernel(spec_ptr, x_ptr, hz_ptr, sf_ptr, consts_ptr, t_ptr, out_ptr, N, M,
                        FIRST, LAST, XR, XT, OR, OT, minus_two_pi, W: tl.constexpr,
                        BLOCK: tl.constexpr):
    """
    One program per (row r, block of points): x[r] shifted onto the reference - the window's peak
    in Hz over the row's own frequency sf[r], plus the proton reference less the target ppm
    (consts), times sf[r] ("torch_engine.peak_shift_each"), then e^{-2 pi i t shift} in float64
    rounded to complex64 ("shift_phasor"), every operation as torch rounds it.
    """
    r = tl.program_id(0)
    peak, _, _ = _window_peak(spec_ptr, r, M, FIRST, LAST, W)
    sf = tl.load(sf_ptr + r)
    shift = (libdevice.div_rn(tl.load(hz_ptr + peak - FIRST), sf) + tl.load(consts_ptr)
             - tl.load(consts_ptr + 1)) * sf
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    angle = (tl.load(t_ptr + j, mask=j < N, other=0.0) * minus_two_pi).to(tl.float64) * shift
    c = libdevice.cos(angle).to(tl.float32)
    s = libdevice.sin(angle).to(tl.float32)
    at = (r.to(tl.int64) * XR + j * XT) * 2
    x_re = tl.load(x_ptr + at, mask=j < N, other=0.0)
    x_im = tl.load(x_ptr + at + 1, mask=j < N, other=0.0)
    to = (r.to(tl.int64) * OR + j * OT) * 2
    tl.store(out_ptr + to, tl.fma(x_re, c, -(x_im * s)), mask=j < N)
    tl.store(out_ptr + to + 1, tl.fma(x_re, s, x_im * c), mask=j < N)


def padded_spectra(x, factor=4):
    """"torch_engine._padded_window"'s spectra of complex64 rows *x* (R, T), any strides, before
    the fftshift: first point halved, zero-filled to factor T, the orthonormal transform."""
    rows, n = x.shape
    padded = torch.empty((rows, factor * n), dtype=x.dtype, device=x.device)
    halved_padded_kernel[(rows, triton.cdiv(factor * n, ECC_POINTS))](
        torch.view_as_real(x), torch.view_as_real(padded), n, factor * n, *x.stride(),
        BLOCK=ECC_POINTS, enable_fp_fusion=False)
    return torch.fft.fft(padded, dim=-1, norm='ortho')


def peak_phased(x, spectra, first, last):
    """Complex64 rows *x* (R, T) phased by their window [first, last) of *spectra*
    ("padded_spectra"), in one pass; laid out as x (empty_like)."""
    rows, n = x.shape
    out = torch.empty_like(x)
    peak_phased_kernel[(rows, triton.cdiv(n, ECC_POINTS))](
        torch.view_as_real(spectra), torch.view_as_real(x), torch.view_as_real(out), n,
        spectra.shape[-1], first, last, *x.stride(), *out.stride(),
        W=triton.next_power_of_2(last - first), BLOCK=ECC_POINTS, enable_fp_fusion=False)
    return out


def peak_shifted(x, spectra, first, last, hz, sf, consts, t):
    """
    Complex64 rows *x* (R, T) shifted to the reference by their window [first, last) of *spectra*
    ("padded_spectra") read against their own frequencies *sf* (R,) float64, in one pass: *hz* the
    window's bins in Hz, *consts* float64 (proton reference, target ppm), *t* (T,) the time axis.
    """
    rows, n = x.shape
    out = torch.empty_like(x)
    peak_shifted_kernel[(rows, triton.cdiv(n, ECC_POINTS))](
        torch.view_as_real(spectra), torch.view_as_real(x), hz, sf.contiguous(), consts, t,
        torch.view_as_real(out), n, spectra.shape[-1], first, last, *x.stride(), *out.stride(),
        -2 * math.pi, W=triton.next_power_of_2(last - first), BLOCK=ECC_POINTS,
        enable_fp_fusion=False)
    return out


#********************************#
#   the whole alignment search   #
#********************************#
# "torch_engine._align" at its default path (two Powell passes, one bracket expansion, two Brent
# steps, no locked Newton steps, 2 then 3 free ones) for one transient per program: the same
# float32 formulas in the same order, launched without FMA contraction so that every operation
# rounds as its own torch kernel does, with IEEE division and square root.
@triton.jit
def _sums(base, n, nu, two_pi, MOMENTS: tl.constexpr, R_PAD: tl.constexpr, BLOCK: tl.constexpr):
    """
    The (R_PAD,) dot products with cos and sin of theta of the rows (h, q) x (re, im) at *base*,
    each weighted 0 .. MOMENTS - 1 times (by n + 1 for h, by n for q) as "ShiftProfile.rows"
    lays them out, the products formed as torch forms them.
    """
    r = tl.arange(0, R_PAD)
    c = r // MOMENTS
    j = r % MOMENTS
    acc_c = tl.zeros([R_PAD, BLOCK], dtype=tl.float32)
    acc_s = tl.zeros([R_PAD, BLOCK], dtype=tl.float32)
    for start in range(0, n, BLOCK):
        t = start + tl.arange(0, BLOCK)
        theta = (two_pi * t.to(tl.float32)) * nu
        mask = (r[:, None] < 4 * MOMENTS) & (t[None, :] < n)
        vals = tl.load(base + c[:, None] * n + t[None, :], mask=mask, other=0.0)
        if MOMENTS > 1:
            w = (t[None, :] + tl.where(c[:, None] < 2, 1, 0)).to(tl.float32)
            vals = tl.where(j[:, None] == 1, vals * w,
                            tl.where(j[:, None] == 2, vals * (w * w), vals))
        acc_c += vals * libdevice.cos(theta)[None, :]
        acc_s += vals * libdevice.sin(theta)[None, :]
    return tl.sum(acc_c, axis=1), tl.sum(acc_s, axis=1)


@triton.jit
def _pick(vec, j, R_PAD: tl.constexpr):
    """Entry *j* of a (R_PAD,) vector."""
    return tl.sum(tl.where(tl.arange(0, R_PAD) == j, vec, 0.0), axis=0)


@triton.jit
def _at(base, n, nu, q0, two_pi, neg_two_pi, BLOCK: tl.constexpr):
    """ShiftProfile.at(nu): (K real, K imag, E) at one shift."""
    c, s = _sums(base, n, nu, two_pi, 1, 4, BLOCK)
    kr = _pick(c, 0, 4) + _pick(s, 1, 4)
    ki = _pick(c, 1, 4) - _pick(s, 0, 4)
    er = _pick(c, 2, 4) + _pick(s, 3, 4)
    angle = neg_two_pi * nu
    lr, li = libdevice.cos(angle), libdevice.sin(angle)
    return lr * kr - li * ki, lr * ki + li * kr, q0 + 2.0 * er


@triton.jit
def _cost(base, n, nu, phi, q0, y_energy, norm, two_pi, neg_two_pi, BLOCK: tl.constexpr):
    """ShiftProfile.cost(phi, *at(nu))."""
    kr, ki, e = _at(base, n, nu, q0, two_pi, neg_two_pi, BLOCK)
    cp, sp = libdevice.cos(-phi), libdevice.sin(-phi)
    locked = (e + y_energy) - 2.0 * (cp * kr - sp * ki)
    return libdevice.div_rn(libdevice.sqrt_rn(tl.maximum(locked, 0.0)), norm)


@triton.jit
def _shift(eps, base, sw_hz, HAS_BASE: tl.constexpr):
    """The shift a line search in Hz from *base* probes: eps / sw (+ base)."""
    nu = libdevice.div_rn(eps, sw_hz)
    if HAS_BASE:
        nu = base + nu
    return nu


@triton.jit
def _bracket(base_ptr, n, nu0, phi, q0, y_energy, norm, sw_hz, two_pi, neg_two_pi,
             p1_nu, p2_nu, p3_nu, HAS_BASE: tl.constexpr, BLOCK: tl.constexpr):
    """torch_engine._bracket with one expansion step: (xa, xb, xc, fa, fb, fc, expanding)."""
    GOLD: tl.constexpr = 1.618034
    GROW_LIMIT: tl.constexpr = 110.0
    if HAS_BASE:
        f0 = _cost(base_ptr, n, nu0 + libdevice.div_rn(0.0, sw_hz), phi, q0, y_energy, norm,
                   two_pi, neg_two_pi, BLOCK)
        f1 = _cost(base_ptr, n, nu0 + libdevice.div_rn(1.0, sw_hz), phi, q0, y_energy, norm,
                   two_pi, neg_two_pi, BLOCK)
    else:
        f0 = _cost(base_ptr, n, 0.0, phi, q0, y_energy, norm, two_pi, neg_two_pi, BLOCK)
        f1 = _cost(base_ptr, n, p1_nu, phi, q0, y_energy, norm, two_pi, neg_two_pi, BLOCK)
    swap = f0 < f1
    xa = tl.where(swap, 1.0, 0.0)
    xb = 1.0 - xa
    xc = 2.618034 - 4.236068 * xa
    fa = tl.where(swap, f1, f0)
    fb = tl.where(swap, f0, f1)
    if HAS_BASE:
        fc = _cost(base_ptr, n, _shift(xc, nu0, sw_hz, True), phi, q0, y_energy, norm, two_pi,
                   neg_two_pi, BLOCK)
    else:
        f_pos = _cost(base_ptr, n, p2_nu, phi, q0, y_energy, norm, two_pi, neg_two_pi, BLOCK)
        f_neg = _cost(base_ptr, n, p3_nu, phi, q0, y_energy, norm, two_pi, neg_two_pi, BLOCK)
        fc = tl.where(swap, f_neg, f_pos)
    expanding = fc < fb

    tmp1 = (xb - xa) * (fb - fc)
    tmp2 = (xb - xc) * (fb - fa)
    val = tmp2 - tmp1
    denom = tl.where(tl.abs(val) < 1e-21, 2e-21, 2.0 * val)
    w = xb - libdevice.div_rn((xb - xc) * tmp2 - (xb - xa) * tmp1, denom)
    wlim = xb + GROW_LIMIT * (xc - xb)
    inside = (w - xc) * (xb - w) > 0
    limit = (inside == 0) & ((w - wlim) * (wlim - xc) >= 0)
    beyond = (inside == 0) & (limit == 0) & ((w - wlim) * (xc - w) > 0)
    p1 = tl.where(inside | beyond, w, tl.where(limit, wlim, xc + GOLD * (xc - xb)))
    f_p1 = _cost(base_ptr, n, _shift(p1, nu0, sw_hz, HAS_BASE), phi, q0, y_energy, norm, two_pi,
                 neg_two_pi, BLOCK)
    take_b = inside & (f_p1 < fc)
    take_c = inside & (take_b == 0) & (f_p1 > fb)
    closed = take_b | take_c
    falling = beyond & (f_p1 < fc)
    second = (inside & (closed == 0)) | falling
    mid_b = tl.where(falling, xc, xb)
    mid_c = tl.where(falling, p1, xc)
    mid_fb = tl.where(falling, fc, fb)
    mid_fc = tl.where(falling, f_p1, fc)
    p2 = mid_c + GOLD * (mid_c - mid_b)
    f_p2 = _cost(base_ptr, n, _shift(p2, nu0, sw_hz, HAS_BASE), phi, q0, y_energy, norm, two_pi,
                 neg_two_pi, BLOCK)
    w = tl.where(second, p2, p1)
    fw = tl.where(second, f_p2, f_p1)
    na = tl.where(closed, tl.where(take_b, xb, xa), mid_b)
    nb = tl.where(closed, tl.where(take_b, p1, xb), mid_c)
    nc = tl.where(closed, tl.where(take_c, p1, xc), w)
    nfa = tl.where(closed, tl.where(take_b, fb, fa), mid_fb)
    nfb = tl.where(closed, tl.where(take_b, f_p1, fb), mid_fc)
    nfc = tl.where(closed, tl.where(take_c, f_p1, fc), fw)
    xa = tl.where(expanding, na, xa)
    xb = tl.where(expanding, nb, xb)
    xc = tl.where(expanding, nc, xc)
    fa = tl.where(expanding, nfa, fa)
    fb = tl.where(expanding, nfb, fb)
    fc = tl.where(expanding, nfc, fc)
    expanding = expanding & (closed == 0) & (fc < fb)
    return xa, xb, xc, fa, fb, fc, expanding


@triton.jit
def _brent_step(x, w, v, fx, fw, fv, a, b, deltax, rat, done, base_ptr, n, nu0, phi, q0,
                y_energy, norm, sw_hz, two_pi, neg_two_pi, HAS_BASE: tl.constexpr,
                BLOCK: tl.constexpr):
    """One step of torch_engine._brent."""
    CGOLD: tl.constexpr = 0.3819660
    tol1 = 1.0e-2 * tl.abs(x) + 1.0e-11
    tol2 = 2.0 * tol1
    xmid = 0.5 * (a + b)
    done = done | (tl.abs(x - xmid) < (tol2 - 0.5 * (b - a)))
    far = tl.where(x >= xmid, a - x, b - x)
    tmp1 = (x - w) * (fx - fv)
    tmp2 = (x - v) * (fx - fw)
    p = (x - v) * tmp2 - (x - w) * tmp1
    q = 2.0 * (tmp2 - tmp1)
    p = tl.where(q > 0, -p, p)
    q = tl.abs(q)
    usable = (p > q * (a - x)) & (p < q * (b - x)) & (tl.abs(p) < tl.abs(0.5 * q * deltax))
    golden_first = tl.abs(deltax) <= tol1
    parabolic = (golden_first == 0) & usable
    step = tl.where(parabolic, libdevice.div_rn(p, tl.where(q > 0, q, 1.0)), CGOLD * far)
    near_edge = ((x + step - a) < tol2) | ((b - x - step) < tol2)
    step = tl.where(parabolic & near_edge, tl.where(xmid - x >= 0, tol1, -tol1), step)
    deltax = tl.where(golden_first | (usable == 0), far, rat)
    rat = step
    u = x + tl.where(tl.abs(rat) < tol1, tl.where(rat >= 0, tol1, -tol1), rat)
    fu = _cost(base_ptr, n, _shift(u, nu0, sw_hz, HAS_BASE), phi, q0, y_energy, norm, two_pi,
               neg_two_pi, BLOCK)
    worse = fu > fx
    a_new = tl.where(worse, tl.where(u < x, u, a), tl.where(u >= x, x, a))
    b_new = tl.where(worse, tl.where(u < x, b, u), tl.where(u >= x, b, x))
    shift_w = worse & ((fu <= fw) | (w == x))
    shift_v = worse & (shift_w == 0) & ((fu <= fv) | (v == x) | (v == w))
    v_new = tl.where(worse, tl.where(shift_w, w, tl.where(shift_v, u, v)), w)
    fv_new = tl.where(worse, tl.where(shift_w, fw, tl.where(shift_v, fu, fv)), fw)
    w_new = tl.where(worse, tl.where(shift_w, u, w), x)
    fw_new = tl.where(worse, tl.where(shift_w, fu, fw), fx)
    x_new = tl.where(worse, x, u)
    fx_new = tl.where(worse, fx, fu)
    a = tl.where(done, a, a_new)
    b = tl.where(done, b, b_new)
    v = tl.where(done, v, v_new)
    fv = tl.where(done, fv, fv_new)
    w = tl.where(done, w, w_new)
    fw = tl.where(done, fw, fw_new)
    x = tl.where(done, x, x_new)
    fx = tl.where(done, fx, fx_new)
    return x, w, v, fx, fw, fv, a, b, deltax, rat, done


@triton.jit
def _newton_free(base, n, nu, low, high, cap, q0, two_pi, neg_two_pi, rate, rate2,
                 BLOCK: tl.constexpr):
    """One step of torch_engine._newton with the phase free (phi None)."""
    c, s = _sums(base, n, nu, two_pi, 3, 16, BLOCK)
    angle = neg_two_pi * nu
    lr, li = libdevice.cos(angle), libdevice.sin(angle)
    # K, K' and K'' (h rows 0-5), E' and E'' (q rows 6-11), as ShiftProfile.at forms them
    ar = _pick(c, 0, 16) + _pick(s, 3, 16)
    ai = _pick(c, 3, 16) - _pick(s, 0, 16)
    kr, ki = lr * ar - li * ai, lr * ai + li * ar
    ar = _pick(c, 1, 16) + _pick(s, 4, 16)
    ai = _pick(c, 4, 16) - _pick(s, 1, 16)
    br, bi = lr * ar - li * ai, lr * ai + li * ar           # lead K1, times -i rate:
    k1r, k1i = br * -0.0 - bi * -rate, br * -rate + bi * -0.0
    ar = _pick(c, 2, 16) + _pick(s, 5, 16)
    ai = _pick(c, 5, 16) - _pick(s, 2, 16)
    br, bi = lr * ar - li * ai, lr * ai + li * ar           # lead K2, times -rate^2:
    k2r, k2i = br * rate2, bi * rate2
    e1r = _pick(c, 7, 16) + _pick(s, 10, 16)
    e1i = _pick(c, 10, 16) - _pick(s, 7, 16)
    e1 = 2.0 * (e1r * -0.0 - e1i * -rate)
    e2 = (2.0 * (_pick(c, 8, 16) + _pick(s, 11, 16))) * rate2

    mag = tl.maximum(libdevice.hypot(kr, ki), 1.1754943508222875e-38)
    slope = libdevice.div_rn(kr * k1r - (-ki) * k1i, mag)
    k1abs = libdevice.hypot(k1r, k1i)
    curve = libdevice.div_rn((k1abs * k1abs + (kr * k2r - (-ki) * k2i)) - slope * slope, mag)
    grad = e1 - 2.0 * slope
    hess = e2 - 2.0 * curve
    sign = tl.where(grad > 0, 1.0, tl.where(grad < 0, -1.0, 0.0))
    step = tl.where(hess > 0, libdevice.div_rn(-grad, hess), -sign * cap)
    step = tl.minimum(tl.maximum(step, -cap), cap)
    return tl.minimum(tl.maximum(nu + step, low), high)


@triton.jit
def _pass(base, n, nu, q0, y_energy, norm, sw_hz, reach, cap, inv_n, two_pi, neg_two_pi, rate,
          rate2, p1_nu, p2_nu, p3_nu, HAS_BASE: tl.constexpr, FREE: tl.constexpr,
          BLOCK: tl.constexpr):
    """One Powell pass of torch_engine._align from shift *nu*: the shift it ends at."""
    kr, ki, _ = _at(base, n, nu, q0, two_pi, neg_two_pi, BLOCK)
    phi = libdevice.atan2(ki, kr)
    xa, xb, xc, fa, fb, fc, expanding = _bracket(base, n, nu, phi, q0, y_energy, norm, sw_hz,
                                                 two_pi, neg_two_pi, p1_nu, p2_nu, p3_nu,
                                                 HAS_BASE, BLOCK)
    valid = ((((fb < fc) & (fb <= fa)) | ((fb < fa) & (fb <= fc)))
             & (((xa < xb) & (xb < xc)) | ((xc < xb) & (xb < xa))))
    best = tl.where((fa <= fb) & (fa <= fc), xa, tl.where(fb <= fc, xb, xc))
    if HAS_BASE:
        offset = nu
    else:
        offset = nu * 0.0
    usable = valid | expanding
    start = offset + libdevice.div_rn(tl.where(usable, xb, best), sw_hz)
    low = tl.where(usable, offset + libdevice.div_rn(tl.minimum(xa, xc), sw_hz), start)
    high = tl.where(usable, offset + libdevice.div_rn(tl.maximum(xa, xc), sw_hz), start)
    low = tl.where(expanding & (xc < xa), -reach, low)
    high = tl.where(expanding & (xc > xa), reach, high)

    # Brent's two steps inside the bracket
    x = xb
    w = xb
    v = xb
    fx = fb
    fw = fb
    fv = fb
    a = tl.minimum(xa, xc)
    b = tl.maximum(xa, xc)
    deltax = xb * 0.0
    rat = xb * 0.0
    done = xb != xb
    for _ in tl.static_range(2):
        x, w, v, fx, fw, fv, a, b, deltax, rat, done = _brent_step(
            x, w, v, fx, fw, fv, a, b, deltax, rat, done, base, n, nu, phi, q0, y_energy, norm,
            sw_hz, two_pi, neg_two_pi, HAS_BASE, BLOCK)
    start = tl.where(valid, offset + libdevice.div_rn(x, sw_hz), start)
    low = tl.where(valid, offset + libdevice.div_rn(a, sw_hz), low)
    high = tl.where(valid, offset + libdevice.div_rn(b, sw_hz), high)

    nu = start
    low = nu - inv_n
    high = nu + inv_n
    for _ in tl.static_range(FREE):
        nu = _newton_free(base, n, nu, low, high, cap, q0, two_pi, neg_two_pi, rate, rate2, BLOCK)
    return nu


@triton.jit
def align_search_kernel(rows_ptr, q0_ptr, energy_ptr, norm_ptr, moving_ptr, phi_ptr, nu_ptr, n,
                        sw_hz, reach, cap, inv_n, two_pi, neg_two_pi, rate, rate2, p1_nu, p2_nu,
                        p3_nu, BLOCK: tl.constexpr):
    """
    The alignment search of one transient per program: its phase (rad) and shift (cycles); a
    transient that does not move (not drawn, or its sample's only one) is left at zero unsearched.
    """
    m = tl.program_id(0)
    nu = tl.load(nu_ptr + m) * 0.0
    phi = nu
    if tl.load(moving_ptr + m) != 0:
        base = rows_ptr + m.to(tl.int64) * 4 * n
        q0 = tl.load(q0_ptr + m)
        y_energy = tl.load(energy_ptr + m)
        norm = tl.load(norm_ptr + m)
        nu = _pass(base, n, nu, q0, y_energy, norm, sw_hz, reach, cap, inv_n, two_pi,
                   neg_two_pi, rate, rate2, p1_nu, p2_nu, p3_nu, False, 2, BLOCK)
        nu = _pass(base, n, nu, q0, y_energy, norm, sw_hz, reach, cap, inv_n, two_pi,
                   neg_two_pi, rate, rate2, p1_nu, p2_nu, p3_nu, True, 3, BLOCK)
        kr, ki, _ = _at(base, n, nu, q0, two_pi, neg_two_pi, BLOCK)
        phi = libdevice.atan2(ki, kr)
    tl.store(phi_ptr + m, phi)
    tl.store(nu_ptr + m, nu)


def align_search(rows, q0, energy, norm, sw_hz, moving):
    """
    Run the search: *rows* (M, 4, T) float32 contiguous ("ShiftProfile.plain"), *q0*, *energy*,
    *norm* (M,); only the transients *moving* (M,) bool marks are searched, the others are
    returned at zero.

    Returns:
        "(phi, nu)": (M,) phase in radians and shift in cycles per sample.
    """
    m, _, n = rows.shape
    phi = torch.empty(m, dtype=torch.float32, device=rows.device)
    nu = torch.zeros(m, dtype=torch.float32, device=rows.device)
    gold = 1.618034
    probes = (np.array([0.0, 1.0, 1.0 + gold, -gold]) / sw_hz).astype(np.float32)
    align_search_kernel[(m,)](
        rows, q0, energy, norm, moving.to(torch.int8).contiguous(), phi, nu, n, float(sw_hz),
        float(np.float32(sw_hz / 4 / sw_hz)), float(np.float32(0.5 / sw_hz)),
        float(np.float32(1.0 / n)), 6.283185307179586, -6.283185307179586, 6.283185307179586,
        -6.283185307179586 ** 2, float(probes[1]), float(probes[2]), float(probes[3]),
        BLOCK=SHIFT_POINTS, enable_fp_fusion=False, num_warps=2)
    return phi, nu


#************#
#   median   #
#************#
#: Rows each program takes.
MEDIAN_ROWS = 64

#: What an entry a sample does not keep sorts as: beyond every kept one, but finite - Triton's sort
#: (3.2) misorders rows that hold infinities.
BEYOND = tl.constexpr(3.0e38)


@triton.jit
def median_kernel(parts_ptr, mask_ptr, out_ptr, R, ROWS_PER_SAMPLE, D,
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
    count = tl.sum(keep.to(tl.int32), axis=1)
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
    median_kernel[(triton.cdiv(flat.shape[0], MEDIAN_ROWS),)](
        flat, mask.contiguous(), out, flat.shape[0], flat.shape[0] // b, d,
        D_PAD=triton.next_power_of_2(d), BLOCK_R=MEDIAN_ROWS, num_warps=8)
    return out.reshape(parts.shape[:-1])


#*************#
#   unalike   #
#*************#
#: Points (real and imaginary parts) each program sums over.
UNLIKE_POINTS = 256


@triton.jit
def unlike_sums_kernel(x_ptr, target_ptr, energy_ptr, cross_ptr, square_ptr, D, N, CHUNKS, SB,
                       SD, ST, D_PAD: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per (sample, block of points): the block's sums of x^2 and x y for every
    transient x and of y^2 for the target y, first points halved (as their spectra take them);
    the real and imaginary parts p of x[b, d] at 2 (b SB + d SD + (p // 2) ST) + p % 2.
    """
    b = tl.program_id(0)
    chunk = tl.program_id(1)
    d = tl.arange(0, D_PAD)
    p = chunk * BLOCK + tl.arange(0, BLOCK)
    at = 2 * (b.to(tl.int64) * SB + d[:, None] * SD + (p[None, :] // 2) * ST) + p[None, :] % 2
    x = tl.load(x_ptr + at, mask=(d[:, None] < D) & (p[None, :] < N), other=0.0)
    x = tl.where(p[None, :] < 2, 0.5 * x, x)
    y = tl.load(target_ptr + b * N + p, mask=p < N, other=0.0)
    y = tl.where(p < 2, 0.5 * y, y)
    out = (b * D + d) * CHUNKS + chunk
    tl.store(energy_ptr + out, tl.sum(x * x, axis=1), mask=d < D)
    tl.store(cross_ptr + out, tl.sum(x * y[None, :], axis=1), mask=d < D)
    tl.store(square_ptr + b * CHUNKS + chunk, tl.sum(y * y, axis=0))


@triton.jit
def unlike_keep_kernel(energy_ptr, cross_ptr, square_ptr, mask_ptr, keep_ptr, D, CHUNKS, sdlimit,
                       D_PAD: tl.constexpr, CHUNKS_PAD: tl.constexpr):
    """
    One program per sample: every transient's distance to the target from the block sums, the
    mean and standard deviation over the valid ones, and those within *sdlimit* of the mean kept.
    """
    b = tl.program_id(0)
    d = tl.arange(0, D_PAD)
    c = tl.arange(0, CHUNKS_PAD)
    valid = tl.load(mask_ptr + b * D + d, mask=d < D, other=0) != 0
    parts = (b * D + d[:, None]) * CHUNKS + c[None, :]
    inside = (d[:, None] < D) & (c[None, :] < CHUNKS)
    energy = tl.sum(tl.load(energy_ptr + parts, mask=inside, other=0.0), axis=1)
    cross = tl.sum(tl.load(cross_ptr + parts, mask=inside, other=0.0), axis=1)
    square = tl.sum(tl.load(square_ptr + b * CHUNKS + c, mask=c < CHUNKS, other=0.0), axis=0)
    metric = tl.sqrt(tl.maximum(energy - 2.0 * cross + square, 0.0))
    weight = valid.to(tl.float32)
    count = tl.sum(weight, axis=0)
    avg = tl.sum(metric * weight, axis=0) / count
    std = tl.sqrt(tl.sum((metric - avg) * (metric - avg) * weight, axis=0) / count)
    tl.store(keep_ptr + b * D + d, valid & (tl.abs(metric - avg) <= sdlimit * std), mask=d < D)


def unlike_step(fids, target, mask, sdlimit):
    """
    The transients of complex64 *fids* (B, D, T) that *mask* (B, D) marks and that lie within
    *sdlimit* standard deviations of their sample's mean distance to *target* (B, T, 2); *fids*
    are read where they lie.
    """
    b, d, t = fids.shape
    chunks = triton.cdiv(2 * t, UNLIKE_POINTS)
    energy = torch.empty((b, d, chunks), dtype=torch.float32, device=fids.device)
    cross = torch.empty_like(energy)
    square = torch.empty((b, chunks), dtype=torch.float32, device=fids.device)
    keep = torch.empty((b, d), dtype=torch.bool, device=fids.device)
    d_pad = triton.next_power_of_2(d)
    unlike_sums_kernel[(b, chunks)](torch.view_as_real(fids), target.contiguous(), energy, cross,
                                    square, d, 2 * t, chunks, *fids.stride(), D_PAD=d_pad,
                                    BLOCK=UNLIKE_POINTS)
    unlike_keep_kernel[(b,)](energy, cross, square, mask.contiguous(), keep, d, chunks,
                             float(sdlimit), D_PAD=d_pad, CHUNKS_PAD=triton.next_power_of_2(chunks))
    return keep


#***********#
#   noise   #
#***********#
#: Points each program takes the peak of.
PEAK_POINTS = 4096


@triton.jit
def peak_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    One program per (sample, block of points): the block's largest |x|, folded into the
    sample's maximum (exact in any order).
    """
    b = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    q = tl.arange(0, 2)
    z = tl.load(x_ptr + (b.to(tl.int64) * N + p[:, None]) * 2 + q[None, :],
                mask=(p < N)[:, None], other=0.0)
    re, im = tl.split(z)
    tl.atomic_max(out_ptr + b, tl.max(libdevice.hypot(re, im), axis=0))


def spectrum_peak(x, axis):
    """
    The largest |fft| along *axis* of every sample of complex64 *x* (B, ...), (B,) float32: each
    sample's transform taken where its points lie (a strided batch, not a transposed copy), and
    its |.| and maximum in one pass.
    """
    shape = x.shape
    out = torch.zeros(shape[0], dtype=torch.float32, device=x.device)
    for b in range(shape[0]):
        spectrum = torch.fft.fft(x[b].reshape(math.prod(shape[1:axis]), shape[axis], -1), dim=1)
        n = spectrum.numel()
        peak_kernel[(1, triton.cdiv(n, PEAK_POINTS))](torch.view_as_real(spectrum), out[b:], n,
                                                      BLOCK=PEAK_POINTS)
    return out


@triton.jit
def mixed_noise_kernel(data_ptr, index_ptr, scale_ptr, root_ptr, out_ptr, seed, M, C, K,
                       C_PAD: tl.constexpr, K_PAD: tl.constexpr, ROWS: tl.constexpr):
    """
    One program per (point m, sample b): out[b, m] = data[b, m] + root[b] s_b (real + i
    imag)[b, m], each (C, K) with the coils as rows - white draws at the sample's level, coupled
    by a root of its channels' covariance, on the data; with ROWS, sample b's data is row
    index[b] of the data. The real and imaginary draws of element e (its place in the output)
    are one Box-Muller pair of Philox's words for *seed* at e, in libdevice's functions: the same
    wherever and however the kernel runs.
    """
    m = tl.program_id(0)
    b = tl.program_id(1)
    i = tl.arange(0, C_PAD)
    k = tl.arange(0, K_PAD)
    q = tl.arange(0, 2)
    root = tl.load(root_ptr + ((b * C + i[:, None]) * C + i[None, :])[:, :, None] * 2
                   + q[None, None, :], mask=((i[:, None] < C) & (i[None, :] < C))[:, :, None],
                   other=0.0)
    r_re, r_im = tl.split(root)
    s = tl.load(scale_ptr + b)
    offset = ((b.to(tl.int64) * M + m) * C + i[:, None]) * K + k[None, :]
    inside = (i[:, None] < C) & (k[None, :] < K)
    w1, w2, _, _ = tl.randint4x(seed, offset.to(tl.uint32))
    radius = tl.sqrt_rn(-2.0 * libdevice.log(tl.maximum(1.0e-7, tl.uint_to_uniform_float(w1))))
    angle = 6.283185307179586 * tl.uint_to_uniform_float(w2)
    n_re = tl.where(inside, radius * libdevice.cos(angle), 0.0) * s
    n_im = tl.where(inside, radius * libdevice.sin(angle), 0.0) * s
    re = (tl.dot(r_re, n_re, input_precision='ieee')
          - tl.dot(r_im, n_im, input_precision='ieee'))
    im = (tl.dot(r_re, n_im, input_precision='ieee')
          + tl.dot(r_im, n_re, input_precision='ieee'))
    row = tl.load(index_ptr + b).to(tl.int64) if ROWS else b.to(tl.int64)
    at = ((row * M + m) * C + i[:, None]) * K + k[None, :]
    d = tl.load(data_ptr + at[:, :, None] * 2 + q[None, None, :], mask=inside[:, :, None],
                other=0.0)
    d_re, d_im = tl.split(d)
    tl.store(out_ptr + offset[:, :, None] * 2 + q[None, None, :], tl.join(d_re + re, d_im + im),
             mask=inside[:, :, None])


def mixed_noise(data, seed, scale, root, axis, index=None):
    """
    "Noise._add" with its channels coupled, in one pass: complex64 *data* (B, ...) with its coils
    at *axis*, white draws for *seed* (an integer below 2^63) at the level *scale* (one per
    sample, or one for all), mixed by the root *root* (B, C, C) or (C, C). Returns data + root
    (scale (real + i imag)) along the coil axis. With *index* (B,), *data* is a pool (S, ...)
    and sample b its row index[b], read where it lies.
    """
    shape = (len(index),) + tuple(data.shape[1:]) if index is not None else tuple(data.shape)
    b, c = shape[0], shape[axis]
    m, k = math.prod(shape[1:axis]), math.prod(shape[axis + 1:])
    root = root.to(data.dtype).expand(b, c, c).contiguous()
    scale = scale.to(torch.float32).reshape(-1).expand(b).contiguous()
    out = torch.empty(shape, dtype=data.dtype, device=data.device)
    mixed_noise_kernel[(m, b)](
        torch.view_as_real(data.contiguous()), index if index is not None else scale, scale,
        torch.view_as_real(root), torch.view_as_real(out), int(seed), m, c, k,
        C_PAD=max(16, triton.next_power_of_2(c)), K_PAD=max(16, triton.next_power_of_2(k)),
        ROWS=index is not None, num_warps=4)
    return out


#**********************#
#   artificial peaks   #
#**********************#
#: Points each program of the peak FIDs fills.
PEAK_FID_POINTS = 1024

#: Rows of the peak parameters, each (P, B) - one value per peak and sample.
PEAK_PARAMS = ('winding_re', 'winding_im', 'lorentz_ppm', 'gauss_ppm', 'amp', 'phase_re',
               'phase_im', 'keep')


@triton.jit
def peak_fids_kernel(params_ptr, consts_ptr, out_ptr, N, B, BLOCK: tl.constexpr):
    """
    One program per (peak p, sample b) row and block of points: "causal_lineshapes"' FID of the
    row in float64, rounded as its torch operations round - e^{w t} for its winding w (no real
    part), damped where its Lorentzian and Gaussian widths are positive. consts: pi, 1 / |step|,
    1 / N and 1 / (4 ln 2), the reciprocals torch multiplies by where it divides by a number.
    """
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = j.to(tl.float64)
    at = params_ptr + (row // B) * 8 * B + row % B
    w_re = tl.load(at)
    w_im = tl.load(at + B)
    lorentz = tl.load(at + 2 * B)
    gauss = tl.load(at + 3 * B)
    pi = tl.load(consts_ptr)
    inv_step = tl.load(consts_ptr + 1)
    inv_n = tl.load(consts_ptr + 2)
    inv_four_ln2 = tl.load(consts_ptr + 3)
    # w t as a complex product (its real part a signed zero), and exp of an imaginary argument
    angle = tl.fma(w_re, 0.0, w_im * t)
    re = libdevice.cos(angle)
    im = libdevice.sin(angle)
    if lorentz > 0:
        e = libdevice.exp(((-pi) * (lorentz * inv_step)) * t * inv_n)
        re, im = tl.fma(re, e, -(im * 0.0)), tl.fma(re, 0.0, im * e)
    if gauss > 0:
        g = (pi * (gauss * inv_step)) * t * inv_n
        e = libdevice.exp(-(g * g) * inv_four_ln2)
        re, im = tl.fma(re, e, -(im * 0.0)), tl.fma(re, 0.0, im * e)
    q = tl.arange(0, 2)
    tl.store(out_ptr + (row.to(tl.int64) * N + j[:, None]) * 2 + q[None, :],
             tl.join(re, im), mask=(j < N)[:, None])


@triton.jit
def add_peaks_kernel(spec_ptr, spectra_ptr, params_ptr, out_ptr, N, B, P, PER_SAMPLE,
                     REAL: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per row of the complex64 spectra *spec* (B * PER_SAMPLE, N): the row plus its
    sample's peaks - each peak's spectrum (the FIDs' ifft, fftshifted here by indexing) over its
    real maximum, times amp and the phase where the sample keeps the peak, summed over the peaks
    in complex128, cast to complex64 and scaled by the row's own peak (|.|, or |real| with REAL).
    Every product and quotient rounds as torch's: complex products fma(a_re, b_re, -a_im b_im) +
    i fma(a_re, b_im, a_im b_re), a quotient by a real number its reciprocal times.
    """
    r = tl.program_id(0)
    b = r // PER_SAMPLE
    j = tl.arange(0, BLOCK)
    inside = j < N
    q = tl.arange(0, 2)
    s = tl.load(spec_ptr + (r.to(tl.int64) * N + j[:, None]) * 2 + q[None, :],
                mask=inside[:, None], other=0.0)
    s_re, s_im = tl.split(s)
    level = tl.abs(s_re) if REAL else libdevice.hypot(s_re, s_im)
    ref = tl.max(tl.where(inside, level, 0.0), axis=0)
    ref = tl.where(ref > 0, ref, 1.0)

    c_re = tl.zeros([BLOCK], dtype=tl.float64)
    c_im = tl.zeros([BLOCK], dtype=tl.float64)
    shifted = (j + N - N // 2) % N
    for p in range(P):
        at = params_ptr + p * 8 * B + b
        amp = tl.load(at + 4 * B)
        ph_re = tl.load(at + 5 * B)
        ph_im = tl.load(at + 6 * B)
        keep = tl.load(at + 7 * B) > 0
        row = spectra_ptr + (p * B + b).to(tl.int64) * N * 2
        top = tl.max(tl.load(row + j * 2, mask=inside, other=-float('inf')), axis=0)
        scale = 1.0 / top
        x = tl.load(row + shifted[:, None] * 2 + q[None, :], mask=inside[:, None], other=0.0)
        x_re, x_im = tl.split(x)
        x_re = x_re * scale
        x_im = x_im * scale
        a_re = tl.fma(amp, x_re, -(0.0 * x_im))
        a_im = tl.fma(amp, x_im, 0.0 * x_re)
        t_re = tl.fma(a_re, ph_re, -(a_im * ph_im))
        t_im = tl.fma(a_re, ph_im, a_im * ph_re)
        c_re = tl.where(keep, c_re + t_re, c_re)
        c_im = tl.where(keep, c_im + t_im, c_im)
    c_re = c_re.to(tl.float32)
    c_im = c_im.to(tl.float32)
    o_re = s_re + tl.fma(c_re, ref, -(c_im * 0.0))
    o_im = s_im + tl.fma(c_re, 0.0, c_im * ref)
    tl.store(out_ptr + (r.to(tl.int64) * N + j[:, None]) * 2 + q[None, :],
             tl.join(o_re, o_im), mask=inside[:, None])


def artificial_peaks(spec, params, consts, real):
    """
    "ArtificialPeaks.process_tensor" on complex64 spectra *spec* (B, ..., N): *params* (P, 8, B)
    float64 on the device, one row per name in PEAK_PARAMS (keep 1.0 or 0.0), *consts* float64:
    pi, 1 / |ppm step|, 1 / N, 1 / (4 ln 2); *real* takes each row's peak from |real|.
    """
    n, (p, _, b) = spec.shape[-1], params.shape
    fids = torch.empty((p * b, n), dtype=torch.complex128, device=spec.device)
    peak_fids_kernel[(p * b, triton.cdiv(n, PEAK_FID_POINTS))](
        params, consts, torch.view_as_real(fids), n, b, BLOCK=PEAK_FID_POINTS,
        enable_fp_fusion=False)
    spectra = torch.fft.ifft(fids, dim=-1)
    flat = spec.contiguous().reshape(-1, n)
    out = torch.empty_like(flat)
    add_peaks_kernel[(flat.shape[0],)](
        torch.view_as_real(flat), torch.view_as_real(spectra), params, torch.view_as_real(out),
        n, b, p, flat.shape[0] // b, REAL=bool(real), BLOCK=triton.next_power_of_2(n),
        enable_fp_fusion=False, num_warps=8)
    return out.reshape(spec.shape)


#********************#
#   macromolecules   #
#********************#
#: Points each program of the macromolecule kernels fills.
MM_POINTS = 256


@triton.jit
def mm_envelope_kernel(params_ptr, consts_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    One program per (sample b, block of points): "SemiParametrized"'s envelope before its mean
    is taken, 1 + sum_k w_bk cos(k x) / k for k = 1, 2, 3, each step rounded as its torch
    operation rounds - a quotient by k its reciprocal times. params: the weights (B, 3) first;
    consts: the cosine rows (3, N), then 1 / k.
    """
    b = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = j < N
    envelope = tl.full([BLOCK], 1.0, tl.float64)
    for k in tl.static_range(3):
        w = tl.load(params_ptr + b * 3 + k)
        cos = tl.load(consts_ptr + k * N + j, mask=inside)
        envelope = envelope + (w * cos) * tl.load(consts_ptr + 3 * N + k)
    tl.store(out_ptr + b.to(tl.int64) * N + j, envelope, mask=inside)


@triton.jit
def mm_absorption_kernel(params_ptr, envelope_ptr, mean_ptr, consts_ptr, out_ptr, N, B, TAPS,
                         BLOCK: tl.constexpr):
    """
    One program per (sample b, block of points): the base profile's real part convolved with the
    sample's stencil as torch's depthwise convolution sums it - an fma per tap in the zero-padded
    input, from the first - times the envelope e as 1 + amp_mod (e - mean e), clipped at 0.
    params: the weights (B, 3), the stencils (B, TAPS) and the real part (N); consts: amp_mod
    after the cosines and 1 / k.
    """
    b = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = j < N
    stencil = params_ptr + 3 * B + b * TAPS
    base = params_ptr + 3 * B + B * TAPS
    acc = tl.zeros([BLOCK], dtype=tl.float64)
    for k in range(TAPS):
        i = j - TAPS // 2 + k
        valid = (i >= 0) & (i < N)
        acc = tl.where(valid, tl.fma(tl.load(stencil + k), tl.load(base + i, mask=valid), acc),
                       acc)
    envelope = tl.load(envelope_ptr + b.to(tl.int64) * N + j, mask=inside)
    amp_mod = tl.load(consts_ptr + 3 * N + 3)
    envelope = tl.maximum(1.0 + amp_mod * (envelope - tl.load(mean_ptr + b)), 0.0)
    tl.store(out_ptr + b.to(tl.int64) * N + j, acc * envelope, mask=inside)


@triton.jit
def mm_analytic_kernel(half_ptr, out_ptr, N, H, BLOCK: tl.constexpr):
    """
    One program per (row, block of bins): the row's full transform times "scipy.signal.hilbert"'s
    filter h (1, 2 ... 2, 1 where N is even, 0 beyond), from its one-sided transform (H bins) -
    each bin past H the conjugate of its mirror, as torch fills it - in torch's complex product,
    fma(x_re, h, -(x_im 0)) + i fma(x_re, 0, x_im h).
    """
    r = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = j < N
    upper = j >= H
    q = tl.arange(0, 2)
    x = tl.load(half_ptr + (r.to(tl.int64) * H + tl.where(upper, N - j, j)[:, None]) * 2
                + q[None, :], mask=inside[:, None], other=0.0)
    re, im = tl.split(x)
    im = tl.where(upper, -im, im)
    h = tl.where((j == 0) | (2 * j == N), 1.0, tl.where(j < (N + 1) // 2, 2.0, 0.0))
    h = h.to(tl.float64)
    tl.store(out_ptr + (r.to(tl.int64) * N + j[:, None]) * 2 + q[None, :],
             tl.join(tl.fma(re, h, -(im * 0.0)), tl.fma(re, 0.0, im * h)), mask=inside[:, None])


@triton.jit
def mm_unit_kernel(x_ptr, consts_ptr, out_ptr, N, ROW: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per (row, block of points) of the analytic signals: the unnormalised inverse
    transform times 1 / N as torch scales it (a product with 1 / N + 0i), over the row's peak
    |real| - its reciprocal times - where that is positive. The peak is 1 / N times the row's
    largest |real|, as rounding keeps order. consts: 1 / N after the cosines, 1 / k and amp_mod.
    """
    r = tl.program_id(0)
    row = x_ptr + r.to(tl.int64) * N * 2
    i = tl.arange(0, ROW)
    s = tl.load(consts_ptr + 3 * N + 4)
    peak = s * tl.max(tl.abs(tl.load(row + i * 2, mask=i < N, other=0.0)), axis=0)
    scale = libdevice.div_rn(tl.full([], 1.0, tl.float64), tl.where(peak > 0, peak, 1.0))
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    q = tl.arange(0, 2)
    re, im = tl.split(tl.load(row + j[:, None] * 2 + q[None, :], mask=(j < N)[:, None],
                              other=0.0))
    re, im = tl.fma(s, re, -(0.0 * im)), tl.fma(s, im, 0.0 * re)
    tl.store(out_ptr + (r.to(tl.int64) * N + j[:, None]) * 2 + q[None, :],
             tl.join(re * scale, im * scale), mask=(j < N)[:, None])


def semi_parametrized(params, taps, consts, batch, n):
    """
    "SemiParametrized._device_profiles" with an envelope, (B, N) complex128: *params* float64
    holds the envelope weights (B, 3), the stencils (B, taps) and the base profile's real part
    (N); *consts* the envelope's cosines (3, N), 1 / k for k = 1, 2, 3, amp_mod and 1 / N. The
    mean and both transforms stay torch's.
    """
    envelope = torch.empty((batch, n), dtype=torch.float64, device=params.device)
    grid = (batch, triton.cdiv(n, MM_POINTS))
    mm_envelope_kernel[grid](params, consts, envelope, n, BLOCK=MM_POINTS, enable_fp_fusion=False)
    absorption = torch.empty_like(envelope)
    mm_absorption_kernel[grid](params, envelope, envelope.mean(dim=-1), consts, absorption, n,
                               batch, taps, BLOCK=MM_POINTS, enable_fp_fusion=False)
    half = torch.fft.rfft(absorption, dim=-1)
    full = torch.empty((batch, n), dtype=torch.complex128, device=params.device)
    mm_analytic_kernel[grid](torch.view_as_real(half), torch.view_as_real(full), n,
                             half.shape[-1], BLOCK=MM_POINTS, enable_fp_fusion=False)
    signal = torch.fft.ifft(full, dim=-1, norm='forward')
    unit = torch.empty_like(signal)
    mm_unit_kernel[grid](torch.view_as_real(signal), consts, torch.view_as_real(unit), n,
                         ROW=triton.next_power_of_2(n), BLOCK=MM_POINTS, enable_fp_fusion=False)
    return unit


@triton.jit
def add_mm_kernel(spec_ptr, unit_ptr, scale_ptr, out_ptr, N, PER_SAMPLE, UNIT_ROW,
                  ROW: tl.constexpr, BLOCK: tl.constexpr):
    """
    One program per (row, block of points) of the complex64 spectra *spec* (B * PER_SAMPLE, N):
    the row plus its sample's unit profile (UNIT_ROW values after the previous sample's, 0 when
    all share one) cast to complex64, times the sample's scale in float32 and the row's peak
    |real|, the product rounded as torch's: fma(a_re, b_re, -a_im b_im) + i fma(a_re, b_im,
    a_im b_re).
    """
    r = tl.program_id(0)
    b = r // PER_SAMPLE
    row = spec_ptr + r.to(tl.int64) * N * 2
    i = tl.arange(0, ROW)
    peak = tl.max(tl.abs(tl.load(row + i * 2, mask=i < N, other=0.0)), axis=0)
    amp = tl.load(scale_ptr + b).to(tl.float32) * peak
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = (j < N)[:, None]
    q = tl.arange(0, 2)
    s_re, s_im = tl.split(tl.load(row + j[:, None] * 2 + q[None, :], mask=inside, other=0.0))
    u = tl.load(unit_ptr + b.to(tl.int64) * UNIT_ROW + j[:, None] * 2 + q[None, :], mask=inside,
                other=0.0)
    u_re, u_im = tl.split(u.to(tl.float32))
    tl.store(out_ptr + (r.to(tl.int64) * N + j[:, None]) * 2 + q[None, :],
             tl.join(s_re + tl.fma(u_re, amp, -(u_im * 0.0)), s_im + tl.fma(u_re, 0.0, u_im * amp)),
             mask=inside)


def macromolecules(spec, scale, units):
    """
    "Macromolecules.process_tensor" on complex64 spectra *spec* (B, ..., N): every row plus its
    sample's unit profile times the sample's *scale* ((B,) float64) and the row's peak |real|.
    *units*: the profiles, complex128 or float64 (re, im) pairs, a row of N per sample or one
    for all.
    """
    n = spec.shape[-1]
    units = (torch.view_as_real(units) if units.is_complex() else units).reshape(-1, 2 * n)
    flat = spec.contiguous().reshape(-1, n)
    out = torch.empty_like(flat)
    add_mm_kernel[(flat.shape[0], triton.cdiv(n, MM_POINTS))](
        torch.view_as_real(flat), units, scale, torch.view_as_real(out), n,
        flat.shape[0] // scale.shape[0], 2 * n if units.shape[0] > 1 else 0,
        ROW=triton.next_power_of_2(n), BLOCK=MM_POINTS, enable_fp_fusion=False)
    return out.reshape(spec.shape)
