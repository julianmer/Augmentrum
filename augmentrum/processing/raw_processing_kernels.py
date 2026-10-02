####################################################################################################
#                                     raw_processing_kernels.py                                    #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-21                                                                              #
#                                                                                                  #
# Purpose: The Triton kernels of RawProcessor's torch engine ("torch_engine"), each in place of    #
#          the many small kernels its torch operations launch: the wSVD coil weights, the coil     #
#          combination of a pooled batch, the alignment target's distances, the alignment cost's   #
#          sums and the whole alignment search, and the outlier medians. A module of its own,      #
#          imported on first use, so that Triton stays optional.                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
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
def distance_kernel(x_ptr, mask_ptr, out_ptr, D, T, CHUNKS,
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
    chunks = triton.cdiv(t, DISTANCE_POINTS)
    parts = torch.empty((b, d, chunks), dtype=torch.float64, device=x.device)
    distance_kernel[(b, chunks)](torch.view_as_real(x.contiguous()),
                                 mask.to(torch.int8).contiguous(), parts, d, t, chunks,
                                 D_PAD=triton.next_power_of_2(d), BLOCK_T=DISTANCE_POINTS)
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
        BLOCK=SHIFT_POINTS, enable_fp_fusion=False)
    return phi, nu


#************#
#   median   #
#************#
#: Rows each program takes.
MEDIAN_ROWS = 64

#: What an entry a sample does not keep sorts as: beyond every kept one, but finite - Triton's sort
#: (3.2) misorders rows that hold infinities.
BEYOND: tl.constexpr = 3.0e38


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
    median_kernel[(triton.cdiv(flat.shape[0], MEDIAN_ROWS),)](
        flat, mask.to(torch.int8).contiguous(), mask.sum(dim=-1).to(torch.int32).contiguous(),
        out, flat.shape[0], flat.shape[0] // b, d, D_PAD=triton.next_power_of_2(d),
        BLOCK_R=MEDIAN_ROWS)
    return out.reshape(parts.shape[:-1])
