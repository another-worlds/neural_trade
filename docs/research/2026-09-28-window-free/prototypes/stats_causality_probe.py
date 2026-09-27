"""Look-ahead probes for a chunk (sequence-to-sequence) forward pass, on CPU.

For each op: run it on a chunk, perturb every input AFTER position t, and check the outputs at positions
<= t. Also a NaN / Inf placed in the future, and a Jacobian check d out[t] / d x[t'] for t' > t.
Ops: the repo's batched EWMA (utils/math.py ewma_sequence_matrix_multi), and the Keras layers the current
gru_attention model uses along time (models/gru_attention.py), in their current and causal forms.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import neural_trade  # noqa: F401  (puts the CUDA DLLs on PATH; must precede tensorflow)
import tensorflow as tf
from neural_trade.utils.math import ewma_sequence_matrix_multi

tf.random.set_seed(0)
rng = np.random.default_rng(0)
B, T, t = 2, 200, 120


def probe(name, fn, x):
    """Max |change| at positions <= t after perturbing x[:, t+1:], plus NaN / Inf contamination."""
    base = fn(x).numpy()
    xp = x.copy()
    xp[:, t + 1:] += rng.normal(0, 5.0, xp[:, t + 1:].shape)
    pert = fn(xp).numpy()
    d = float(np.max(np.abs(base[:, : t + 1] - pert[:, : t + 1])))
    res = {"max_abs_change_past": d, "bitwise_equal_past": bool(np.array_equal(base[:, : t + 1], pert[:, : t + 1]))}
    for bad in (np.nan, np.inf):
        xb = x.copy()
        xb[:, -1] = bad
        out = fn(xb).numpy()
        res[f"past_nonfinite_after_future_{'nan' if bad != bad else 'inf'}"] = int(
            np.sum(~np.isfinite(out[:, : T - 1])))
    print(f"{name:52s} " + "  ".join(f"{k}={v}" for k, v in res.items()))


# ---- EWMA (matrix form over the whole chunk), K series, per-sample alphas
K = 3
x_ew = np.cumsum(rng.normal(0, 1, (B, K, T)), axis=2).astype("float32")
alpha = tf.constant(rng.uniform(0.02, 0.5, (B, K)).astype("float32"))
ew = lambda x: tf.transpose(ewma_sequence_matrix_multi(tf.constant(x), alpha), [0, 2, 1])  # [B, T, K]
x_ew_t = np.transpose(x_ew, [0, 2, 1]).copy()           # time on axis 1 for probe()
probe("ewma_sequence_matrix_multi (repo)", lambda x: ew(np.transpose(x, [0, 2, 1]).copy()), x_ew_t)

# ---- Keras layers along time
F = 8
x_seq = np.cumsum(rng.normal(0, 1, (B, T, F)), axis=1).astype("float32")
gru_bi = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16, return_sequences=True))
gru_uni = tf.keras.layers.GRU(16, return_sequences=True)
conv_same = tf.keras.layers.Conv1D(4, 15, padding="same")
conv_causal = tf.keras.layers.Conv1D(4, 15, padding="causal")
mha = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=8)
ln = tf.keras.layers.LayerNormalization()
probe("Bidirectional(GRU) (today, gru_attention.py:62)", lambda x: gru_bi(tf.constant(x)), x_seq)
probe("GRU unidirectional", lambda x: gru_uni(tf.constant(x)), x_seq)
probe("Conv1D k=15 padding='same' (today, :80)", lambda x: conv_same(tf.constant(x)), x_seq)
probe("Conv1D k=15 padding='causal'", lambda x: conv_causal(tf.constant(x)), x_seq)
probe("MultiHeadAttention, no mask (today, :67,:92)", lambda x: mha(tf.constant(x), tf.constant(x)), x_seq)
probe("MultiHeadAttention, use_causal_mask=True", lambda x: mha(tf.constant(x), tf.constant(x), use_causal_mask=True),
      x_seq)
probe("LayerNormalization over features", lambda x: ln(tf.constant(x)), x_seq)
probe("mean over time broadcast (GlobalAvgPool, :49,:102)",
      lambda x: tf.repeat(tf.reduce_mean(tf.constant(x), axis=1, keepdims=True), T, axis=1), x_seq)
probe("causal cumulative mean over time", lambda x: tf.cumsum(tf.constant(x), axis=1)
      / tf.reshape(tf.range(1, T + 1, dtype=tf.float32), (1, T, 1)), x_seq)

# ---- Jacobian check on the EWMA: d out[:, t, :] / d x[:, t', :] for t' > t must be exactly 0
xv = tf.Variable(x_ew)
with tf.GradientTape() as tape:
    out = ewma_sequence_matrix_multi(xv, alpha)[:, :, t]      # [B, K] at position t
jac = tape.jacobian(out, xv).numpy()                           # [B, K, B, K, T]
print(f"EWMA Jacobian: max |d out[t] / d x[t' > t]| = {np.max(np.abs(jac[..., t + 1:])):.3g}, "
      f"max |d out[t] / d x[t' <= t]| = {np.max(np.abs(jac[..., : t + 1])):.3g}")
