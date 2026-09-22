####################################################################################################
#                                        _shift_kernel.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-21                                                                              #
#                                                                                                  #
# Purpose: The Triton kernel behind "torch_engine.shift_sums": the alignment cost's dot products   #
#          of every transient's weighted rows with cos and sin of its shift's angles, in one pass. #
#          A module of its own, imported on first use, so that Triton stays optional.              #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import triton
import triton.language as tl
from triton.language.extra import libdevice


#: Points each program reads per iteration.
BLOCK = 256


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


def launch(rows, nu, cos, sin):
    """Run the kernel: *rows* (M, R, T) float32 contiguous, *nu* (M,), outputs (M, R)."""
    m, r, n = rows.shape
    shift_sums_kernel[(m,)](rows, nu, cos, sin, n, 6.283185307179586, R=r,
                            R_PAD=max(triton.next_power_of_2(r), 2), BLOCK=BLOCK)


#********************************#
#   the whole alignment search   #
#********************************#
# "torch_engine._align" at its default path (two Powell passes, one bracket expansion, two Brent
# steps, no locked Newton steps, 2 then 3 free ones) for one transient per program: the same
# float32 formulas in the same order, launched without FMA contraction so that every operation
# rounds as its own torch kernel does, with IEEE division and square root.
@triton.jit
def _sums(base, n, nu, two_pi, STRIDE: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
          BLOCK: tl.constexpr):
    """The (R_PAD,) dot products of rows base[0], base[STRIDE], ... with cos and sin of theta."""
    r = tl.arange(0, R_PAD)
    acc_c = tl.zeros([R_PAD, BLOCK], dtype=tl.float32)
    acc_s = tl.zeros([R_PAD, BLOCK], dtype=tl.float32)
    for start in range(0, n, BLOCK):
        t = start + tl.arange(0, BLOCK)
        theta = (two_pi * t.to(tl.float32)) * nu
        mask = (r[:, None] < R) & (t[None, :] < n)
        vals = tl.load(base + (r[:, None] * STRIDE) * n + t[None, :], mask=mask, other=0.0)
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
    c, s = _sums(base, n, nu, two_pi, 3, 4, 4, BLOCK)
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
    c, s = _sums(base, n, nu, two_pi, 1, 12, 16, BLOCK)
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
def align_search_kernel(rows_ptr, q0_ptr, energy_ptr, norm_ptr, phi_ptr, nu_ptr, n, sw_hz, reach,
                        cap, inv_n, two_pi, neg_two_pi, rate, rate2, p1_nu, p2_nu, p3_nu,
                        BLOCK: tl.constexpr):
    """The alignment search of one transient per program: its phase (rad) and shift (cycles)."""
    m = tl.program_id(0)
    base = rows_ptr + m.to(tl.int64) * 12 * n
    q0 = tl.load(q0_ptr + m)
    y_energy = tl.load(energy_ptr + m)
    norm = tl.load(norm_ptr + m)
    nu = tl.load(nu_ptr + m) * 0.0
    nu = _pass(base, n, nu, q0, y_energy, norm, sw_hz, reach, cap, inv_n, two_pi, neg_two_pi,
               rate, rate2, p1_nu, p2_nu, p3_nu, False, 2, BLOCK)
    nu = _pass(base, n, nu, q0, y_energy, norm, sw_hz, reach, cap, inv_n, two_pi, neg_two_pi,
               rate, rate2, p1_nu, p2_nu, p3_nu, True, 3, BLOCK)
    kr, ki, _ = _at(base, n, nu, q0, two_pi, neg_two_pi, BLOCK)
    tl.store(phi_ptr + m, libdevice.atan2(ki, kr))
    tl.store(nu_ptr + m, nu)


def align_search(rows, q0, energy, norm, sw_hz):
    """
    Run the search: *rows* (M, 12, T) float32 contiguous, *q0*, *energy*, *norm* (M,).

    Returns:
        "(phi, nu)": (M,) phase in radians and shift in cycles per sample.
    """
    import numpy as np
    import torch
    m, _, n = rows.shape
    phi = torch.empty(m, dtype=torch.float32, device=rows.device)
    nu = torch.zeros(m, dtype=torch.float32, device=rows.device)
    gold = 1.618034
    probes = (np.array([0.0, 1.0, 1.0 + gold, -gold]) / sw_hz).astype(np.float32)
    align_search_kernel[(m,)](
        rows, q0, energy, norm, phi, nu, n, float(sw_hz), float(np.float32(sw_hz / 4 / sw_hz)),
        float(np.float32(0.5 / sw_hz)), float(np.float32(1.0 / n)), 6.283185307179586,
        -6.283185307179586, 6.283185307179586, -6.283185307179586 ** 2, float(probes[1]),
        float(probes[2]), float(probes[3]), BLOCK=BLOCK, enable_fp_fusion=False)
    return phi, nu
