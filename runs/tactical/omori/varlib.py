"""Intraday VaR/ES with the FZ0 loss (hypothesis #6, SPEC_var.md). Reuses omorilib's causal month data (HAR sigma_hat, shock stream).
Returns are the T-bar log return R_i = ln C_{i+T-1} - ln C_{i-1}; z = R / (sigma_hat sqrt(T)). Left tail: y = z, right tail: y = -z.
VaR_a and ES_a are quantile and tail mean of the LOWER tail of y (negative numbers); FZ0 (Patton-Ziegel-Chen 2019):
  L(y, v, e) = -1{y <= v}(v - y) / (a e) + v/e + log(-e) - 1,   v < 0, e < v."""
import os, sys, math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import omorilib as O  # noqa: E402

ALPHAS = (0.025, 0.05, 0.01)          # first two primary, 0.01 secondary
TV = (15, 60)
RIDGE = 1e-3


def fz0(y, v, e, a):
    return -(y <= v).astype(float) * (v - y) / (a * e) + v / e + np.log(-e) - 1.0


def fz0_grad_ve(y, v, e, a):
    hit = (y <= v).astype(float)
    dv = -hit / (a * e) + 1.0 / e
    de = hit * (v - y) / (a * e * e) - v / (e * e) + 1.0 / e
    return dv, de


def _unpack(th, k):
    return th[0], th[1:1 + k], th[1 + k], th[2 + k:2 + 2 * k]


def _ve(th, X, k):
    a0, a, b0, b = _unpack(th, k); u = a0 + X @ a; w = b0 + X @ b
    eu, ew = np.exp(u), np.exp(w); return -eu, -(eu + ew), eu, ew


def objective(th, X, y, al, k):
    v, e, eu, ew = _ve(th, X, k)
    L = fz0(y, v, e, al); dv, de = fz0_grad_ve(y, v, e, al)
    du = dv * (-eu) + de * (-eu); dw = de * (-ew)
    n = len(y); f = float(L.mean()) + RIDGE * (np.sum(th[1:1 + k] ** 2) + np.sum(th[2 + k:] ** 2))
    g = np.empty_like(th)
    g[0] = du.mean(); g[1:1 + k] = X.T @ du / n + 2 * RIDGE * th[1:1 + k]
    g[1 + k] = dw.mean(); g[2 + k:] = X.T @ dw / n + 2 * RIDGE * th[2 + k:]
    return f, g


def fhs(ytr, al):
    """filtered historical simulation in z units: empirical lower quantile and the mean below it."""
    q = float(np.quantile(ytr, al)); es = float(ytr[ytr <= q].mean()); return q, es


def fit_fz0(Xtr, ytr, al, q0, e0):
    """FZ0 fit of v = -exp(a0 + a x), e = v - exp(b0 + b x); x may have zero columns (then the constant FZ0 fit)."""
    k = Xtr.shape[1]; th0 = np.zeros(2 + 2 * k); th0[0] = math.log(-q0); th0[1 + k] = math.log(max(q0 - e0, 1e-3))
    res = minimize(objective, th0, args=(Xtr, ytr, al, k), jac=True, method="L-BFGS-B", options=dict(maxiter=300))
    return res.x


def predict(th, X):
    k = X.shape[1]; v, e, _, _ = _ve(th, X, k); return v, e


# ------------------------------------------------------------------ features and the month
_SV = {}


def semivar_ratio(B):
    if "sv" not in _SV or _SV["id"] != id(B):
        r2 = B.r * B.r; rp = O.S.roll_trailing(r2 * (B.r > 0), 60); rn = O.S.roll_trailing(r2 * (B.r < 0), 60)
        _SV["sv"] = np.log((rp + 1e-12) / (rn + 1e-12)); _SV["id"] = id(B)
    return _SV["sv"]


def month_arrays(B, month, T, p_star, c_star):
    """z for the T-window of each anchor and the shape features; training/test masks from omorilib.MonthData (kernels off, one kernel)."""
    d = O.MonthData(B, month, T, O.K_T[T], kernels=False)
    if not d.ok:
        return None
    pos = d.anch - d.seg_lo; K = O.power_kernel(d.m_seg, p_star, c_star)[pos]
    e = d.anch + T - 1; ok = e < B.n
    R = np.where(ok, B.lc[np.minimum(e, B.n - 1)] - B.lc[d.anch - 1], np.nan)
    z = R / (d.sig * math.sqrt(T))
    tod = B.tod[d.anch]
    harm = np.stack([f(2 * np.pi * h * tod / 1440.0) for h in (1, 2) for f in (np.sin, np.cos)], 1)
    sv = semivar_ratio(B)[d.anch - 1]
    shape = np.stack([np.log1p(d.tsl.astype(float)), np.log1p(K), d.fs[:, 0]], 1)
    return dict(d=d, z=z, sv=sv, shape=shape, harm=harm, anch=d.anch)


def tail_features(arr, tail):
    """x for the lower tail of y: the semivariance ratio enters with the sign that makes it 'adverse minus favourable'."""
    s = -arr["sv"] if tail == "left" else arr["sv"]          # left tail: adverse side = downward (RS-); sv = ln RS+/RS-
    return np.concatenate([s[:, None], arr["shape"], arr["harm"]], 1)


def score_month_var(B, month, T, p_star, c_star, alphas=ALPHAS):
    arr = month_arrays(B, month, T, p_star, c_star)
    if arr is None:
        return None
    d = arr["d"]; z = arr["z"]; ok = np.isfinite(z); tr = d.tr & ok; te = d.te & ok
    if tr.sum() < 5000 or te.sum() < 500:
        return None
    out = {"month": d.month, "T": T, "n": int(te.sum()), "n_eff": float(te.sum() * O.STRIDE / T), "cells": []}
    sub = np.zeros(te.sum(), bool); sub[::max(1, T // O.STRIDE)] = True          # non-overlapping windows, for the independence test
    for tail in ("left", "right"):
        y = z if tail == "left" else -z
        X = tail_features(arr, tail)
        for al in alphas:
            q, es = fhs(y[tr], al)
            Xs_tr = O.standardise(X[tr], X[tr]); Xs_te = O.standardise(X[tr], X[te])
            th_c = fit_fz0(Xs_tr, y[tr], al, q, es); th_k = fit_fz0(Xs_tr[:, :0], y[tr], al, q, es)
            v_c, e_c = predict(th_c, Xs_te); v_k, e_k = predict(th_k, Xs_te[:, :0]); v_f = np.full(te.sum(), q); e_f = np.full(te.sum(), es)
            yt = y[te]; rec = {"tail": tail, "alpha": al}
            for name, (v, e) in (("fhs", (v_f, e_f)), ("k", (v_k, e_k)), ("c", (v_c, e_c))):
                hit = yt <= v
                rec[f"L_{name}"] = float(fz0(yt, v, e, al).mean()); rec[f"hit_{name}"] = float(hit.mean())
                rec[f"z2_{name}"] = float(np.mean(yt * hit / (al * e)) - 1.0)
                h = hit[sub].astype(int); rec[f"tr_{name}"] = [int(((h[:-1] == a) & (h[1:] == b)).sum()) for a in (0, 1) for b in (0, 1)]
            out["cells"].append(rec)
    return out
