"""numba kernels: engine F on the 1-min grid, the hysteresis state machine, and the band layer (forward/backward)."""
import numpy as np
from numba import njit


@njit(cache=True)
def hysteresis_positions(mu, dec, H, thr):
    """Positions per minute (held from close t to close t+1) from forecasts mu (bps; NaN = none) at decision minutes.
    Same logic as policy.HysteresisPolicy with immediate (nominal) fills."""
    n = mu.shape[0]
    x = np.zeros(n, np.float64)
    s = 0
    since = -10 ** 9
    for i in range(n):
        if dec[i]:
            m = mu[i]
            if m == m:                                     # not NaN
                if s == 0:
                    if abs(m) > thr:
                        s = 1 if m > 0 else -1
                        since = i
                elif i - since >= H:
                    if s * m < -thr:
                        s = -s
                        since = i
                    elif s * m <= 0:
                        s = 0
        x[i] = s
    return x


@njit(cache=True)
def engine_f_minutes(x, R1, F1, c_in, c_out, x0):
    """Per-minute engine-F net: x[t] held over (t, t+1]; R1[t] = c[t+1]/c[t]-1; F1[t] = funding charged at bar t+1."""
    n = x.shape[0]
    out = np.empty(n, np.float64)
    prev = x0
    for t in range(n):
        xt = x[t]
        dx = abs(xt - prev)
        inc = (abs(xt) + dx - abs(prev)) * 0.5
        red = (abs(prev) + dx - abs(xt)) * 0.5
        out[t] = xt * (R1[t] - F1[t]) - (c_in * inc + c_out * red)
        prev = xt
    return out


@njit(cache=True)
def band_forward(tau, b, p0, out_p, code):
    """p_k = clip(p_{k-1}, tau_k - b_k, tau_k + b_k), then clip to [-1, 1]. tau, b: (B, T). code: 0 inside, 1 low edge,
    2 high edge (the edge binds, so the gradient goes to tau and b), 3 clipped at +-1 (no gradient)."""
    B, T = tau.shape
    for j in range(B):
        prev = p0[j]
        for k in range(T):
            lo = tau[j, k] - b[j, k]
            hi = tau[j, k] + b[j, k]
            if prev < lo:
                v = lo
                c = 1
            elif prev > hi:
                v = hi
                c = 2
            else:
                v = prev
                c = 0
            if v > 1.0:
                v = 1.0
                c = 3
            elif v < -1.0:
                v = -1.0
                c = 3
            out_p[j, k] = v
            code[j, k] = c
            prev = v


@njit(cache=True)
def band_backward(g, code, gt, gb, gp0):
    B, T = g.shape
    for j in range(B):
        carry = 0.0
        for k in range(T - 1, -1, -1):
            tot = g[j, k] + carry
            c = code[j, k]
            if c == 0:
                gt[j, k] = 0.0
                gb[j, k] = 0.0
                carry = tot
            elif c == 1:
                gt[j, k] = tot
                gb[j, k] = -tot
                carry = 0.0
            elif c == 2:
                gt[j, k] = tot
                gb[j, k] = tot
                carry = 0.0
            else:
                gt[j, k] = 0.0
                gb[j, k] = 0.0
                carry = 0.0
        gp0[j] = carry


@njit(cache=True)
def band_chain(tau, b, p0, step):
    """Inference band on one chain, weights rounded to `step` after clipping (as in engine E). Returns positions."""
    n = tau.shape[0]
    p = np.empty(n, np.float64)
    prev = p0
    for k in range(n):
        lo = max(-1.0, tau[k] - b[k])
        hi = min(1.0, tau[k] + b[k])
        if prev < lo:
            v = round(lo / step) * step if step > 0 else lo
        elif prev > hi:
            v = round(hi / step) * step if step > 0 else hi
        else:
            v = prev
        p[k] = v
        prev = v
    return p
