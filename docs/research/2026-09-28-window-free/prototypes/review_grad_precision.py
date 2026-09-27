"""Adversarial check of the step-1 gradient criterion ('within 1e-3 relative of float64') for the
prototyped two-level kernel, increment form, per-bar alpha = sigmoid(base_logit + 0.5 tanh(z_t)),
30,720 bars: gradient w.r.t. the 7 base logits (float32 mat2) against float64 tf.scan autodiff."""
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
periods = np.array([2, 5, 14, 30, 60, 240, 1440], np.float64)
T = 30720
c = close[-(T + 1):]
dx = np.diff(c) / S
z = (dx - dx.mean()) / dx.std()
base = np.log(2.0 / (periods + 1.0)) - np.log(1 - 2.0 / (periods + 1.0))
w = np.random.default_rng(0).normal(size=(len(periods), T))
w[:, : T // 2] = 0.0   # loss only on the second half, as anchors far from the series start


def loss(fn, dt):
    lg = tf.Variable(base.astype(dt))
    with tf.GradientTape() as tape:
        al = tf.clip_by_value(tf.sigmoid(lg[:, None] + tf.constant((0.5 * np.tanh(z))[None, :].astype(dt))),
                              1e-6, 1 - 1e-6)
        a = 1.0 - al
        b = -a * tf.constant(dx[None, :].astype(dt))
        b = tf.concat([tf.zeros_like(b[:, :1]), b[:, 1:]], 1)
        h = fn(a, b)
        L = tf.reduce_sum(tf.constant(w.astype(dt)) * h)
    return tape.gradient(L, lg).numpy()


def scan64(a, b):
    out = tf.scan(lambda prev, ab: ab[0] * prev + ab[1], (tf.transpose(a), tf.transpose(b)),
                  initializer=tf.zeros_like(a[:, 0]))
    return tf.transpose(out)


g64 = loss(tf.function(scan64), np.float64)
g32 = loss(lambda a, b: scan_mat2(a, b, 64), np.float32)
rel = np.abs(g32 - g64) / np.abs(g64)
print("grad w.r.t. base logits, rel err by period: " + ", ".join(f"p{int(p)}={r:.2e}" for p, r in zip(periods, rel)))
