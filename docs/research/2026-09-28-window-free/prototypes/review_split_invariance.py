"""Adversarial check of the step-1 criterion '(4) two halves with carried state equal one pass within
1e-6' for the prototyped two-level kernel in increment form with per-bar alpha (float32, CPU)."""
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
S = 257.51813253642973
close = pd.read_csv(CSV, usecols=["close"])["close"].to_numpy(np.float64)
periods = np.array([5, 30, 60, 240, 1440], np.float64)
T = 30720
c = close[-(T + 1):]
dx = np.diff(c) / S
z = (dx - dx.mean()) / dx.std()
base = np.log(2.0 / (periods + 1.0)) - np.log(1 - 2.0 / (periods + 1.0))
alpha = np.clip(1.0 / (1.0 + np.exp(-(base[:, None] + 0.5 * np.tanh(z)[None, :]))), 1e-6, 1 - 1e-6)
a = (1.0 - alpha).astype(np.float32)
b = (-(1.0 - alpha) * dx[None, :]).astype(np.float32)
b[:, 0] = 0.0


def run(a_, b_, C=64):
    Tn = a_.shape[1]
    pad = (-Tn) % C
    if pad:
        a_ = np.concatenate([a_, np.ones((a_.shape[0], pad), np.float32)], 1)
        b_ = np.concatenate([b_, np.zeros((b_.shape[0], pad), np.float32)], 1)
    return scan_mat2(tf.constant(a_), tf.constant(b_), C).numpy()[:, :Tn]


full = run(a, b)
for split in (15360, 15001):
    h1 = run(a[:, :split], b[:, :split])
    b2 = b[:, split:].copy()
    b2[:, 0] = b2[:, 0] + a[:, split] * h1[:, -1]          # carried state into the second half
    h2 = run(a[:, split:], b2)
    two = np.concatenate([h1, h2], 1)
    print(f"split at {split}: max |two halves - one pass| by period "
          + ", ".join(f"p{int(p)}={e:.2e}" for p, e in zip(periods, np.abs(two - full).max(1))))
