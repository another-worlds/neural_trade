"""CPU-only: forward+backward time of the LearnableIndicators layer (matrix EWMA) vs window length T.

Shows how the in-window EWMA cost grows with T (O(T^2) weights). CPU numbers do not transfer to the
GPU (launch-bound there); only the growth ratio is of interest.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import neural_trade  # noqa: F401
import numpy as np
import tensorflow as tf
from neural_trade.core.config import Config
from neural_trade.models.layers.learnable_indicators import LearnableIndicators

tf.config.threading.set_inter_op_parallelism_threads(4)
B = 64
for T in (60, 120, 240):
    cfg = Config(LOOKBACK=T)
    layer = LearnableIndicators(cfg)
    x = tf.constant(np.cumsum(np.random.default_rng(0).normal(0, 0.05, (B, T)), axis=1).astype("float32"))
    meta = tf.zeros([B, 18])

    @tf.function
    def step(x, meta):
        with tf.GradientTape() as tape:
            out = layer([x, meta])
            loss = tf.reduce_mean(tf.square(out))
        return loss, tape.gradient(loss, layer.trainable_variables)

    step(x, meta)  # trace + build
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        loss, g = step(x, meta)
        _ = float(loss)
        ts.append(time.perf_counter() - t0)
    print(f"T={T:4d} B={B}: fwd+bwd median {np.median(ts)*1e3:7.1f} ms (CPU)   output {tuple(layer([x, meta]).shape)}")
