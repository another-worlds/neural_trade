"""tdvar: TD / tower estimation of the realised-variance term structure against direct HAR, iterated HAR and MIDAS (SPEC.md).

Everything is in variance LEVELS (rate = mean squared 1-minute log return per bar, times SC). Anchor a (a bar position) sees
bars < a only; the forecast interval is bars a .. a+H-1. The data are cut at CUTOFF when loaded: no later bar is ever in memory.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.environ.get("TDVAR_CSV", "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv")
CUTOFF = pd.Timestamp("2025-07-21 00:00:00")        # exclusive: the last bar is 2025-07-20 23:59
SC = 1e6
STRIDE = 60                                         # anchors every 60 bars
LAGS = (60, 240, 1440, 10080)                       # HAR lags (bars): 1 h, 4 h, 1 d, 7 d
LAGMAX = max(LAGS)
MIDAS_K = 168                                       # hourly blocks (7 days) for MIDAS
WINDOWS = (30, 60, 180, 365, 1095)                  # training window, days
HORIZONS = (240, 1440, 60)                          # 240 and 1440 main, 60 secondary
FLOOR_REL = 0.02                                    # forecast floor: 2% of the median training target
Y_FLOOR_REL = 1e-4                                  # target floor in QLIKE (a flat tape has rate 0)


# ------------------------------------------------------------------ data
def load_bars(path=None, cutoff=CUTOFF):
    """Close prices and the time index; every row at or after `cutoff` is dropped right here."""
    s = pd.read_csv(path or CSV, usecols=["timestamp", "close"])
    t = pd.to_datetime(s["timestamp"])
    keep = (t < cutoff).to_numpy()
    s, t = s[keep], t[keep]
    assert len(s) > 0 and t.iloc[-1] < cutoff
    return s["close"].to_numpy(np.float64), pd.DatetimeIndex(t.to_numpy())


class Data:
    """Cumulative squared returns and per-anchor features/targets, all computed from bars < anchor (features) or the
    forecast interval (targets)."""

    def __init__(self, close, index, stride=STRIDE):
        self.close, self.index, self.n = close, index, len(close)
        self.lr = np.concatenate([[0.0], np.diff(np.log(close))])
        self.cs = np.concatenate([[0.0], np.cumsum(self.lr ** 2)])
        self.stride = stride
        self.tns = index.values.astype("datetime64[m]").astype(np.int64)   # minutes since epoch, for gap checks
        self.anch = np.arange(LAGMAX + 1, self.n - 1, stride)
        self.ta = index[self.anch]
        self._blocks = None

    # ---- causal features at arbitrary positions a (bars < a only)
    def feat(self, a):
        a = np.asarray(a)
        cols = [(self.cs[a] - self.cs[a - k]) / k * SC for k in LAGS]
        return np.stack(cols, 1)

    def blocks(self, a):
        """MIDAS regressors: hourly RV rates of the K hours before a, [len(a), K], newest first."""
        a = np.asarray(a)
        ks = np.arange(MIDAS_K + 1)
        edges = self.cs[a[:, None] - 60 * ks[None, :]]                       # cs at a, a-60, ...
        return (edges[:, :-1] - edges[:, 1:]) / 60.0 * SC

    def rate(self, a, h):
        """Realised variance rate over bars a .. a+h-1."""
        a = np.asarray(a)
        return (self.cs[a + h] - self.cs[a]) / h * SC

    def contiguous(self, a, h):
        """True where bars a-LAGMAX .. a+h-1 are consecutive minutes (no gap in the file)."""
        a = np.asarray(a)
        ok = (a + h <= self.n) & (a - LAGMAX >= 0)
        aa = np.where(ok, a, LAGMAX)
        hh = np.where(ok, a + h - 1, LAGMAX)
        span = self.tns[hh] - self.tns[aa - LAGMAX]
        return ok & (span == (hh - (aa - LAGMAX)))


# ------------------------------------------------------------------ models
def design(X):
    return np.column_stack([np.ones(len(X)), X])


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_direct(X, y):
    return ols(design(X), y)


def predict_lin(w, X):
    return design(X) @ w


def fit_var(X, Xn):
    """Level VAR(1): next-step features Xn (features at a+s) on [1, X] (columns by OLS)."""
    return ols(design(X), Xn)                     # [1+p, p]


def combine(s, H, w_s, w_r, X, B=None, Xn=None):
    """Forecast of the H-rate from a split s + (H-s): (s * g_s(x_a) + (H-s) * g_r(x_{a+s} or E[x_{a+s}|x_a])) / H."""
    first = predict_lin(w_s, X)
    nxt = predict_lin(w_r, Xn if Xn is not None else design(X) @ B)
    return (s * first + (H - s) * nxt) / H


def exp_almon(theta1, theta2, K=MIDAS_K):
    k = np.arange(1, K + 1)
    w = np.exp(theta1 * k + theta2 * k * k)
    return w / w.sum()


MIDAS_GRID = [(t1, t2) for t1 in (0.0, -0.01, -0.03, -0.08, -0.2) for t2 in (0.0, -2e-4, -1e-3, -4e-3)]


def fit_midas(Bk, y):
    """Direct MIDAS: y on 1 and sum_k w_k(theta) block_k, weights exponential Almon, theta by grid on the SSE."""
    best = None
    for t1, t2 in MIDAS_GRID:
        z = Bk @ exp_almon(t1, t2)
        c = ols(design(z[:, None]), y)
        sse = float(np.sum((y - design(z[:, None]) @ c) ** 2))
        if best is None or sse < best[0]:
            best = (sse, (t1, t2), c)
    return best[1], best[2]


def predict_midas(theta, c, Bk):
    return c[0] + c[1] * (Bk @ exp_almon(*theta))


# ------------------------------------------------------------------ metrics
def qlike(y, f, y_floor, f_floor):
    x = np.maximum(y, y_floor) / np.maximum(f, f_floor)
    return float(np.mean(x - np.log(x) - 1.0))


def spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def t_ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float("nan")] * 3
    m = v.mean()
    hw = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return [float(m), float(m - hw), float(m + hw)]


# ------------------------------------------------------------------ one (month, window, horizon) cell
def month_bounds(period):
    return period.start_time, (period + 1).start_time


def cell(D, t0, t1, W, H, end_limit_pos, tag_models=("direct", "iterated", "td", "td_recent", "midas", "naive")):
    """Fit on the window before t0 (labels end by t0), score on anchors in [t0, t1) whose label ends by end_limit_pos.
    Returns (rows, info). All models share the training anchors except td_recent (see SPEC)."""
    s = H // 2
    r_ = H - s
    pos0 = int(D.index.searchsorted(t0))
    lo = int(D.index.searchsorted(t0 - pd.Timedelta(days=W)))
    A = D.anch
    # training anchors: window start, label ends by t0 (bar a+H-1 < pos0), contiguous history
    tr = (A >= lo) & (A + H <= pos0) & D.contiguous(A, H)
    tr_recent = (A >= lo) & (A + s <= pos0) & D.contiguous(A, s) & ~tr & (A + H > pos0)   # extra anchors for td_recent
    a_tr = A[tr]
    pos1 = int(D.index.searchsorted(t1))
    te = (A >= pos0) & (A < pos1) & (A + H <= end_limit_pos) & D.contiguous(A, H)
    a_te = A[te]
    if len(a_tr) < 20 or len(a_te) < 3:
        return None
    X, Xte = D.feat(a_tr), D.feat(a_te)
    y_tr, y_te = D.rate(a_tr, H), D.rate(a_te, H)
    med = float(np.median(y_tr))
    f_floor, y_floor = FLOOR_REL * med, Y_FLOOR_REL * med
    preds = {}
    # direct HAR
    w_d = fit_direct(X, y_tr)
    preds["direct"] = predict_lin(w_d, Xte)
    # iterated HAR: g_s and g_r direct fits and a level VAR, same anchors
    ys_tr = D.rate(a_tr, s)
    Xn = D.feat(a_tr + s)
    w_s = fit_direct(X, ys_tr)
    w_r = fit_direct(X, y_tr if r_ == H else D.rate(a_tr, r_)) if r_ != s else w_s
    B = fit_var(X, Xn)
    preds["iterated"] = combine(s, H, w_s, w_r, Xte, B=B)
    # TD (linear fixed-horizon tower): bootstrap target from realised s-rate + g_r at the realised x_{a+s}; same anchors
    # (g_r is fit on the realised r-rate of the SAME anchors, i.e. horizon r_ starting at a, as in the iterated variant;
    # the bootstrap uses its value at the later state x_{a+s}).
    Z = (s * ys_tr + r_ * predict_lin(w_r, Xn)) / H
    w_td = ols(design(X), Z)
    preds["td"] = predict_lin(w_td, Xte)
    # td_recent: also uses anchors whose H-label is not complete but whose s-part is (bars < t0 only)
    a_rec = np.concatenate([a_tr, A[tr_recent]])
    Xr, Xnr = D.feat(a_rec), D.feat(a_rec + s)
    ysr = D.rate(a_rec, s)
    w_sr = fit_direct(Xr, ysr)
    w_rr = w_sr if r_ == s else fit_direct(Xr, D.rate(a_rec, r_))
    Zr = (s * ysr + r_ * predict_lin(w_rr, Xnr)) / H
    preds["td_recent"] = predict_lin(ols(design(Xr), Zr), Xte)
    # MIDAS direct
    Bk, Bte = D.blocks(a_tr), D.blocks(a_te)
    th, cm = fit_midas(Bk, y_tr)
    preds["midas"] = predict_midas(th, cm, Bte)
    # naive: trailing rate over the last H bars (the nearest lag available)
    k = LAGS[int(np.argmin([abs(l - H) for l in LAGS]))]
    preds["naive"] = (D.cs[a_te] - D.cs[a_te - k]) / k * SC
    rows = []
    for m in tag_models:
        f = np.maximum(preds[m], f_floor)
        rows.append(dict(model=m, qlike=qlike(y_te, f, y_floor, f_floor), spearman=spearman(f, y_te),
                         mean_f=float(f.mean()), mean_y=float(y_te.mean())))
    info = dict(n_train=int(len(a_tr)), n_train_recent=int(len(a_rec)), n_test=int(len(a_te)),
                n_eff_train=float(len(a_tr) * D.stride / H), n_eff_test=float(len(a_te) * D.stride / H),
                midas_theta=list(th), win_days_avail=float((pos0 - lo) / 1440.0))
    return rows, info
