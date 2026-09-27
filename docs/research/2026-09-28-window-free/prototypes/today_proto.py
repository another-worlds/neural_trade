"""Today's model on CPU: graph op counts (a proxy for GPU kernel launches), CPU time, cuDNN
eligibility of the Bi-GRU, memory of the matrix EWMA vs LOOKBACK, and the warm-up bias of
window-restarted EWMAs on real BTC data. Read-only on the repo (imports the package)."""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import neural_trade  # noqa: F401  (puts CUDA DLLs on PATH; must precede tensorflow)
import numpy as np
import pandas as pd
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.registries.models import Models
import neural_trade.utils.math as mh

tf.config.threading.set_inter_op_parallelism_threads(4)
CSV = "C:/Users/Step/Documents/neural_trade/binance_btcusdt_1min_ccxt.csv"
PRED_SCALE = 257.51813253642973   # runs/20260924T182915Z-1aeff1c-dirty-af67ee43/artifacts/meta.json


def count_ops(cf):
    """Ops in a concrete function's graph, including nested function bodies (once each)."""
    g = cf.graph
    n = len(g.get_operations())
    lib = g.as_graph_def().library.function
    return n, n + sum(len(f.node_def) for f in lib)


def main():
    tf.random.set_seed(1)
    cfg = Config()
    model = Models.build(cfg.MODEL_NAME, cfg)
    B, L = cfg.BATCH_SIZE, cfg.LOOKBACK
    x = tf.constant(np.random.default_rng(0).normal(size=(B, L)).astype(np.float32))

    # cuDNN eligibility of the Bi-GRU (keras/layers/rnn/gru.py: _could_use_gpu_kernel)
    for lyr in model.layers:
        if isinstance(lyr, tf.keras.layers.Bidirectional):
            print("Bi-GRU cuDNN-eligible (forward, backward):",
                  lyr.forward_layer._could_use_gpu_kernel, lyr.backward_layer._could_use_gpu_kernel)

    @tf.function
    def fwd_bwd(x):
        with tf.GradientTape() as tape:
            outs = model(x, training=True)
            loss = tf.add_n([tf.reduce_mean(o) for o in outs])
        return tape.gradient(loss, model.trainable_variables)

    cf = fwd_bwd.get_concrete_function(x)
    top, total = count_ops(cf)
    fwd_bwd(x)
    t0 = time.perf_counter()
    for _ in range(5):
        fwd_bwd(x)
    t_model = (time.perf_counter() - t0) / 5
    print(f"base model fwd+bwd, B={B}, L={L}: graph ops {top} top-level / {total} incl. nested; CPU {t_model:.3f} s")

    ind = [lyr for lyr in model.layers if lyr.name.startswith("learnable_indicators")][0]
    meta = tf.zeros((B, 18))

    @tf.function
    def ind_fb(x):
        with tf.GradientTape() as tape:
            y = ind([x, meta])
            loss = tf.reduce_mean(tf.square(y))
        return tape.gradient(loss, ind.trainable_variables)

    cf2 = ind_fb.get_concrete_function(x)
    top2, total2 = count_ops(cf2)
    ind_fb(x)
    t0 = time.perf_counter()
    for _ in range(5):
        ind_fb(x)
    t_ind = (time.perf_counter() - t0) / 5
    print(f"LearnableIndicators fwd+bwd, B={B}, L={L}: graph ops {top2} / {total2}; CPU {t_ind:.3f} s "
          f"({100 * t_ind / t_model:.0f}% of the base model's CPU time)")

    # memory of the stage-1 decay tensor [B, K=18, L, L] (math.py ewma_sequence_matrix_multi)
    for LL in (60, 120, 240, 480, 1440):
        mb = B * 18 * LL * LL * 4 / 2**20
        print(f"  stage-1 EWMA weight tensor at LOOKBACK={LL}: {mb:,.0f} MiB (forward, one copy; x~3 with where/exp/grad)")

    # warm-up bias: window-restarted EWMA (repo: ema[0] = x[0]) vs full-history EWMA, real data,
    # both expressed like the model input: (ema - last_close) / pred_scale.
    close = pd.read_csv(CSV, usecols=["close"])["close"].to_numpy(np.float64)
    rng = np.random.default_rng(3)
    anchors = rng.integers(5000, len(close) - 1, size=512)
    print("\nwarm-up bias of window-restarted EWMAs (LOOKBACK=60), real BTC, scaled units (pred_scale $257.5):")
    print(" period | weight of x[0] at last bar | RMS err at last bar | RMS err mid-window (bar 30) | RMS of the feature itself")
    for p in (5, 10, 14, 26, 30, 60, 99):
        a = 2.0 / (p + 1.0)
        # full-history EWMA in float64
        ema = np.empty_like(close); ema[0] = close[0]
        for t in range(1, len(close)):
            ema[t] = a * close[t] + (1 - a) * ema[t - 1]
        W = np.stack([close[i - 59:i + 1] for i in anchors])            # windows ending at anchor
        win = mh.ewma_sequence_matrix(tf.constant((W - W[:, -1:]) / PRED_SCALE, tf.float32), a).numpy()
        full = np.stack([(ema[i - 59:i + 1] - close[i]) / PRED_SCALE for i in anchors])
        e_last = np.sqrt(np.mean((win[:, -1] - full[:, -1]) ** 2))
        e_mid = np.sqrt(np.mean((win[:, 30] - full[:, 30]) ** 2))
        rms = np.sqrt(np.mean(full[:, -1] ** 2))
        print(f" {p:6d} | {(1 - a) ** 59:26.3f} | {e_last:19.4f} | {e_mid:27.4f} | {rms:.4f}")


if __name__ == "__main__":
    main()
