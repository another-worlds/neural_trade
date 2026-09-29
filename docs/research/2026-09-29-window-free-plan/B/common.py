"""Shared helpers for the NT-053 part-B prototypes (read-only use of D:/neural_trade; CPU only).

* data: the bundled 30-day close series and the purged fold blocks, rebuilt with the package's own
  split_arrays (so anchor bars match the runs), CSV path absolute (no chdir into the repo).
* runs: served logits, the meta_adjust Dense (kernel [2, 18], bias [18]) and the target scale of a run.
* float64 references: windowed (cold-started, like utils/math.py) and full-history EWMAs, today's
  per-window context features, per-bar recurrences.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

REPO = "D:/neural_trade"
RUNS = f"{REPO}/runs"
CSV30 = f"{REPO}/binance_btcusdt_1min_ccxt.csv"
HERE = os.path.dirname(os.path.abspath(__file__))

# Logit order everywhere below = LearnableIndicators.get_indicator_trainable_variables() order,
# which is also the meta_adjust column order and the get_learned_parameters() key order.
NAMES = (["ma_period_0", "ma_period_1", "ma_period_2"]
         + [f"macd_{i}_{r}" for i in range(3) for r in ("fast", "slow", "signal")]
         + ["rsi_period_0", "rsi_period_1", "rsi_period_2", "bb_period_0", "bb_period_1", "bb_period_2"])
H5_NAMES = (["alpha_ma_0", "alpha_ma_1", "alpha_ma_2"]
            + [f"macd_{i}_{r}" for i in range(3) for r in ("fast", "slow", "signal")]
            + ["rsi_alpha_0", "rsi_alpha_1", "rsi_alpha_2", "bb_alpha_0", "bb_alpha_1", "bb_alpha_2"])
TEXTBOOK = dict(zip(NAMES, [5, 10, 30, 12, 26, 9, 5, 35, 5, 8, 17, 9, 9, 14, 21, 10, 20, 25]))
META_SCALE = 0.5          # learnable_indicators.py:33
EPS8 = 1e-8               # learnable_indicators.py:27


def logit_of_period(p):
    a = 2.0 / (np.asarray(p, np.float64) + 1.0)
    return np.log(a + EPS8) - np.log(1.0 - a + EPS8)


def period_of_logit(lg):
    a = 1.0 / (1.0 + np.exp(-np.asarray(lg, np.float64)))
    return np.maximum(2.0 / (a + EPS8) - 1.0, 0.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


# ----------------------------------------------------------------------------------------- data
def config(**kw):
    import neural_trade  # noqa: F401
    from neural_trade.core.config import Config
    cfg = Config(CSV_PATH=CSV30, **kw)
    return cfg


def blocks(fold_index=-1):
    """split_arrays of the default config at ``fold_index``: close series and anchor bars per block."""
    from neural_trade.data.processor import split_arrays
    cfg = config(FOLD_INDEX=fold_index)
    out = split_arrays(cfg)
    close = np.asarray(out["close"], np.float64)
    res = {"close": close, "fold": out["fold"]}
    for b in ("train", "val", "cal", "test"):
        res[b] = np.asarray(out[b]["anchor_bar"], np.int64)
    return res


def run_dirs():
    return sorted(glob.glob(f"{RUNS}/2026*/"))


def ablation_dirs():
    return sorted(glob.glob(f"{RUNS}/ablations/ablate_physics_v1-full/runs/*/"))


def run_info(run_dir):
    """Served logits (artifacts/weights.h5 if present, else weights.h5), meta Dense, target scale."""
    import h5py
    fn = run_dir + "artifacts/weights.h5"
    if not os.path.exists(fn):
        fn = run_dir + "weights.h5"
    with h5py.File(fn, "r") as f:
        W = np.array(f["dense/dense/kernel:0"], np.float64)
        b = np.array(f["dense/dense/bias:0"], np.float64)
        lg = np.array([float(np.array(f[f"learnable_indicators/learnable_indicators/{n}:0"])) for n in H5_NAMES])
    scale = None
    meta = run_dir + "artifacts/meta.json"
    if os.path.exists(meta):
        scale = json.load(open(meta)).get("pred_scale")
    if scale is None:  # older runs: the scaler.joblib next to the run
        import joblib
        sc = joblib.load(run_dir + "scaler.joblib")
        scale = float(sc.scale_[0])
    return {"run": os.path.basename(run_dir.rstrip("/\\")), "weights": fn, "W": W, "b": b, "logits": lg,
            "scale": float(scale)}


# ----------------------------------------------------------------------------------------- windows
def windows(close, anchors, L=60):
    idx = anchors[:, None] - (L - 1) + np.arange(L)[None, :]
    return close[idx]


def today_context(close, anchors, scale, L=60):
    """gru_attention.py:44-51 on the window_relative input (scaling.py:52): [mean, max] of (w - last)/scale."""
    w = windows(close, anchors, L)
    rel = (w - w[:, -1:]) / scale
    return np.stack([rel.mean(1), rel.max(1)], 1)


def meta_shift(ctx, W, b):
    """learnable_indicators.py:115: the logit shift = meta_scale * tanh(ctx @ W + b)  -> [N, 18]."""
    return META_SCALE * np.tanh(ctx @ W + b)


def rolling_context(close, scale, L=60):
    """Series-mode mirror of today's context at EVERY bar t >= L-1: trailing-L mean and max of
    (close_k - close_t)/scale, k in [t-L+1, t]. Rows t < L-1 use the expanding window [0, t]."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(close)
    out = np.full((n, 2), np.nan)
    sw = sliding_window_view(close, L)                      # [n-L+1, L], row r ends at bar r+L-1
    last = sw[:, -1:]
    out[L - 1:, 0] = ((sw - last) / scale).mean(1)
    out[L - 1:, 1] = ((sw - last) / scale).max(1)
    for t in range(min(L - 1, n)):                          # expanding (still causal) before bar L-1
        seg = (close[:t + 1] - close[t]) / scale
        out[t] = [seg.mean(), seg.max()]
    return out


def ew_context(close, scale, span=60, lam=None):
    """EW mirror (no fixed window): [EMA_span(close)_t - close_t, decayed max offset] / scale.
    Decayed max offset o_t = max(0, lam * (o_{t-1} - dx_t)) (a running max that relaxes toward the close
    with factor lam per bar; lam = 1 - 2/(span + 1) gives the EMA's mean lag)."""
    a = 2.0 / (span + 1.0)
    lam = (1.0 - a) if lam is None else lam
    n = len(close)
    d = np.zeros(n); o = np.zeros(n)
    dx = np.diff(close, prepend=close[0])
    for t in range(1, n):
        d[t] = (1 - a) * (d[t - 1] - dx[t])
        o[t] = max(0.0, lam * (o[t - 1] - dx[t]))
    return np.stack([d, o], 1) / scale


# ----------------------------------------------------------------------------------------- EWMAs
def ewma_windowed(w, alpha):
    """Cold-started EWMA over each window row (ema[0] = x[0]); alpha scalar or [N]. float64. [N, L]."""
    w = np.asarray(w, np.float64)
    a = np.broadcast_to(np.asarray(alpha, np.float64), (w.shape[0],))[:, None]
    out = np.empty_like(w)
    out[:, 0] = w[:, 0]
    for t in range(1, w.shape[1]):
        out[:, t] = a[:, 0] * w[:, t] + (1 - a[:, 0]) * out[:, t - 1]
    return out


def ewma_full(x, alpha):
    """Full-history EWMA (ema[0] = x[0]) with a constant alpha, float64, via scipy lfilter."""
    from scipy.signal import lfilter
    x = np.asarray(x, np.float64)
    zi = np.array([(1 - alpha) * x[0]])
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=zi)
    return y


def linrec64(a, b, h0=None):
    """float64 h_t = a_t h_{t-1} + b_t along the last axis; a, b [K, T]."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    h = np.empty_like(b)
    prev = np.zeros(b.shape[:-1]) if h0 is None else np.asarray(h0, np.float64)
    for t in range(b.shape[-1]):
        prev = a[..., t] * prev + b[..., t]
        h[..., t] = prev
    return h


def m_eps(alpha_min, eps):
    """Bars until the weight on the initial state, prod(1 - alpha_k) <= (1 - alpha_min)^t, is <= eps."""
    return int(np.ceil(np.log(eps) / np.log1p(-alpha_min)))


def dump(obj, name):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path
