"""tdconfirm: confirmation of the neural TD tower (SPEC.md). Reuses tdvar_lib (data, features, HAR, metrics) and tdvar_nn.Net
(the exact network of the TD study; nn_td and nn_direct are not retuned)."""
import contextlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tdvar"))
import tdvar_lib as L  # noqa: E402
import tdvar_nn as N  # noqa: E402

WINDOWS = (30, 60, 90, 365)
HORIZONS = (240, 1440)
SEEDS = (0, 1, 2, 3, 4)
BASE_L2, BASE_STEPS = N.L2, N.STEPS            # 1e-3, 200: the TD study's constants (never changed)
BASELINES = ("nn_direct_reg", "midas_q", "naive", "har_direct", "har_iter")
REG_PATH = os.path.join(HERE, "reg_config.json")


def load_reg():
    """The frozen regularised-direct configuration (written once by tune_reg.py)."""
    with open(REG_PATH) as f:
        c = json.load(f)
    return dict(l2=float(c["l2"]), steps=int(c["steps"]))


@contextlib.contextmanager
def net_settings(l2, steps):
    """Temporarily set the L2 weight and step count that tdvar_nn.Net.fit reads; always restored."""
    old = (N.L2, N.STEPS)
    N.L2, N.STEPS = l2, steps
    try:
        yield
    finally:
        N.L2, N.STEPS = old


# ------------------------------------------------------------------ anchors
def anchors(D, t0, t1, W, H, end_limit_pos):
    """(training anchors, test anchors): training labels end before t0 (a + H <= pos0); same rule as tdvar."""
    pos0 = int(D.index.searchsorted(t0))
    lo = int(D.index.searchsorted(t0 - pd.Timedelta(days=W)))
    pos1 = int(D.index.searchsorted(t1))
    A = D.anch
    tr = (A >= lo) & (A + H <= pos0) & D.contiguous(A, H)
    te = (A >= pos0) & (A < pos1) & (A + H <= end_limit_pos) & D.contiguous(A, H)
    return A[tr], A[te], pos0, lo


# ------------------------------------------------------------------ TD target
def td_target(net_s, ys, zn, s, H):
    """Bootstrap target Z = (s rate_s + (H-s) g_s(x_(a+s))) / H: the realised s-rate plus the FROZEN stage-1 net at the later
    state. It reads nothing else (not rate_H)."""
    return (s * ys + (H - s) * net_s.predict(zn)) / H


# ------------------------------------------------------------------ QLIKE MIDAS
def _q_obj(c, z, y, f_floor):
    f = np.maximum(c[0] + c[1] * z, f_floor)
    x = y / f
    return float(np.mean(x - np.log(x) - 1.0))


def fit_midas_q(Bk, y, y_floor, f_floor):
    """Direct MIDAS with theta (20-point grid) and c0, c1 >= 0 all fitted by the QLIKE; returns (theta, c, train_qlike)."""
    yy = np.maximum(y, y_floor)
    best = None
    for th in L.MIDAS_GRID:
        z = Bk @ L.exp_almon(*th)
        c_ols = np.clip(L.ols(L.design(z[:, None]), yy), 0.0, None)
        starts = [c_ols, np.array([0.0, float(yy.mean() / max(z.mean(), 1e-12))])]
        for c0 in starts:
            r = minimize(_q_obj, c0, args=(z, yy, f_floor), method="L-BFGS-B", bounds=[(0, None), (0, None)])
            if best is None or r.fun < best[0]:
                best = (float(r.fun), th, np.asarray(r.x, float))
    return best[1], best[2], best[0]


def predict_midas_q(theta, c, Bk):
    return c[0] + c[1] * (Bk @ L.exp_almon(*theta))


# ------------------------------------------------------------------ fit everything for one (month, W, H)
class Fit:
    """All models fitted on the training anchors before t0. `params()` lists every fitted number (purge test)."""

    def __init__(self, D, t0, t1, W, H, end_limit_pos, reg, seeds=SEEDS):
        self.D, self.H, self.s = D, H, H // 2
        self.a_tr, self.a_te, self.pos0, self.lo = anchors(D, t0, t1, W, H, end_limit_pos)
        self.ok = len(self.a_tr) >= 20 and len(self.a_te) >= 3
        if not self.ok:
            return
        s = self.s
        a = self.a_tr
        X = D.feat(a)
        lx = np.log(X + N.EPS)
        self.mu, self.sd = lx.mean(0), lx.std(0) + 1e-9
        z = N.zfeat(X, self.mu, self.sd)
        zn = N.zfeat(D.feat(a + s), self.mu, self.sd)
        self.y_tr, self.ys = D.rate(a, H), D.rate(a, s)
        med = float(np.median(self.y_tr))
        self.ok = med > 0
        if not self.ok:
            return
        self.f_floor, self.y_floor = L.FLOOR_REL * med, L.Y_FLOOR_REL * med
        self.nets = {}
        for sd in seeds:
            nd = N.Net(z, self.y_tr, self.y_floor, seed=sd)
            with net_settings(reg["l2"], reg["steps"]):
                nr = N.Net(z, self.y_tr, self.y_floor, seed=sd)
            ns = N.Net(z, self.ys, self.y_floor * s / H, seed=sd)
            ns_before = [getattr(ns, k).copy() for k in ("W1", "b1", "w2", "wl")]
            Z = td_target(ns, self.ys, zn, s, H)
            nt = N.Net(z, Z, self.y_floor, seed=sd)
            assert all(np.array_equal(b, getattr(ns, k)) for b, k in zip(ns_before, ("W1", "b1", "w2", "wl")))
            self.nets[sd] = dict(nn_direct=nd, nn_direct_reg=nr, nn_td=nt, stage1=ns, Z=Z)
        Bk = D.blocks(a)
        self.midas_theta, self.midas_c, self.midas_train_q = fit_midas_q(Bk, self.y_tr, self.y_floor, self.f_floor)

    def params(self):
        out = {}
        for sd, d in self.nets.items():
            for k in ("nn_direct", "nn_direct_reg", "nn_td", "stage1"):
                n = d[k]
                out[f"{sd}/{k}"] = np.concatenate([n.W1.ravel(), n.b1, n.w2, n.wl, [n.b]])
            out[f"{sd}/Z"] = d["Z"]
        out["midas"] = np.concatenate([np.asarray(self.midas_theta, float), self.midas_c])
        return out

    def predict(self, a_te):
        """{model: forecast array [seeds, n_test] (nets) or [n_test]} on test anchors a_te (features read bars < a only)."""
        D = self.D
        zte = N.zfeat(D.feat(a_te), self.mu, self.sd)
        out = {}
        for name in ("nn_direct", "nn_direct_reg", "nn_td"):
            out[name] = np.stack([self.nets[sd][name].predict(zte) for sd in self.nets])
        out["midas_q"] = predict_midas_q(self.midas_theta, self.midas_c, D.blocks(a_te))
        return out


def cell_confirm(D, t0, t1, W, H, end_limit_pos, reg, seeds=SEEDS):
    """Rows (one per model) and info for one (month, W, H); None when the cell cannot be scored."""
    fit = Fit(D, t0, t1, W, H, end_limit_pos, reg, seeds)
    if not fit.ok:
        return None
    # HAR and naive, via tdvar's own cell (same anchors and floors); its SSE-MIDAS is kept for reference
    lin = L.cell(D, t0, t1, W, H, end_limit_pos, tag_models=("direct", "iterated", "midas", "naive"))
    if lin is None:
        return None
    lrows, linfo = lin
    assert linfo["n_train"] == len(fit.a_tr) and linfo["n_test"] == len(fit.a_te)
    y_te = D.rate(fit.a_te, H)
    pred = fit.predict(fit.a_te)
    rows = []
    for name in ("nn_td", "nn_direct", "nn_direct_reg"):
        qs, sp = [], []
        for p in pred[name]:
            f = np.maximum(p, fit.f_floor)
            qs.append(L.qlike(y_te, f, fit.y_floor, fit.f_floor))
            sp.append(L.spearman(f, y_te))
        rows.append(dict(model=name, qlike=float(np.mean(qs)), qlike_seeds=[float(q) for q in qs],
                         spearman=float(np.nanmean(sp)), mean_f=float(np.mean(pred[name])), mean_y=float(y_te.mean())))
    f = np.maximum(pred["midas_q"], fit.f_floor)
    rows.append(dict(model="midas_q", qlike=L.qlike(y_te, f, fit.y_floor, fit.f_floor), spearman=L.spearman(f, y_te),
                     mean_f=float(f.mean()), mean_y=float(y_te.mean())))
    rename = {"direct": "har_direct", "iterated": "har_iter", "midas": "midas_sse", "naive": "naive"}
    for r in lrows:
        rows.append(dict(r, model=rename[r["model"]]))
    info = dict(n_train=int(len(fit.a_tr)), n_test=int(len(fit.a_te)), n_eff_train=float(len(fit.a_tr) * D.stride / H),
                n_eff_test=float(len(fit.a_te) * D.stride / H), win_days_avail=float((fit.pos0 - fit.lo) / 1440.0),
                midas_q_theta=list(map(float, fit.midas_theta)), midas_q_c=list(map(float, fit.midas_c)))
    return rows, info
