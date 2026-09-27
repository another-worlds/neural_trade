"""Adversarial check: float32 precision of the two-level matrix recurrence (mat2_proto.scan_mat2, the
form hybrid_mat2.py uses for A2) when it runs the INCREMENT form d_t = (1-a_t)(d_{t-1} - dx_t)
(state = EMA - close, O(1-10) target-scale units) with per-bar alpha, as the recommendation's step 1
specifies. mat2_proto's own precision test ran an EMA of the increments (state ~0.01-0.1), and
check_long.py ran constant alpha with a lag-based (not cumsum-based) inter-chunk matrix.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import tensorflow as tf

from mat2_proto import scan_mat2

CSV = "C:/Users/Step/Documents/neural_trade/binance_btcusdt_1min_ccxt.csv"
S = 257.51813253642973  # pred_scale of run 20260924T182915Z (as in hybrid_proto.py)
close = pd.read_csv(CSV, usecols=["close"])["close"].to_numpy(np.float64)
periods = np.array([2, 5, 14, 30, 60, 240, 1440], np.float64)
for T in (30720, 43008):
    c = close[-(T + 1):]
    dx = np.diff(c) / S                                           # scaled increments
    z = (dx - dx.mean()) / dx.std()
    base = np.log(2.0 / (periods + 1.0)) - np.log(1 - 2.0 / (periods + 1.0))
    alpha = 1.0 / (1.0 + np.exp(-(base[:, None] + 0.5 * np.tanh(z)[None, :])))
    alpha = np.clip(alpha, 1e-6, 1 - 1e-6)                        # [K, T] per-bar alpha
    a64 = 1.0 - alpha
    b64 = -(1.0 - alpha) * dx[None, :]                            # increment form
    d = np.zeros_like(a64)
    for t in range(1, T):                                         # float64 reference, d_0 = 0
        d[:, t] = a64[:, t] * d[:, t - 1] + b64[:, t]
    b64[:, 0] = 0.0
    for C in (64, 128):
        h = scan_mat2(tf.constant(a64, tf.float32), tf.constant(b64, tf.float32), C).numpy()
        err = np.abs(h - d).max(1)
        print(f"T={T} C={C} increment form, per-bar alpha: max|d| by period "
              + ", ".join(f"p{int(p)}={m:.2f}" for p, m in zip(periods, np.abs(d).max(1))))
        print("   max abs err by period " + ", ".join(f"p{int(p)}={e:.2e}" for p, e in zip(periods, err)))
    # stage-2 Bollinger variance EWMA(d^2) for the p=60 / p=240 channels
    for k in (4, 5):
        a2 = alpha[k]
        v = np.zeros(T)
        for t in range(1, T):
            v[t] = (1 - a2[t]) * v[t - 1] + a2[t] * d[k, t] ** 2
        hv = scan_mat2(tf.constant((1 - a2)[None], tf.float32),
                       tf.constant((a2 * d[k] ** 2)[None], tf.float32), 64).numpy()[0]
        print(f"   T={T} stage-2 EWMA(d^2) p{int(periods[k])}: max|v|={np.abs(v).max():.2f}, "
              f"max abs err={np.abs(hv - v).max():.2e}, max rel err (v>1e-3)={np.max(np.abs(hv - v)[v > 1e-3] / v[v > 1e-3]):.2e}")
