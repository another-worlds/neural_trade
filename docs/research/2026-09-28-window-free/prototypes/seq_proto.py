"""Seq2seq building blocks in Keras 2.10 on CPU:
  1. GRU state carry across chunks (TBPTT): 2 chunks with initial_state == 1 long sequence.
  2. Causality: MultiHeadAttention(use_causal_mask=True), Conv1D(padding='causal', dilation).
  3. Selective (per-step alpha_t) EWMA with a Hillis-Steele scan and a hand-written reverse-scan
     gradient (tf.custom_gradient): equals autodiff; op count vs autodiff.
  4. Cost sketch (CPU, FLOP-bound, NOT the launch-bound GPU): today's base model at B=256 x L=60
     vs a causal seq2seq sketch of similar widths on B=16 chunks x (256 burn-in + 1024) bars.
"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import neural_trade  # noqa: F401
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(1)
tf.config.threading.set_inter_op_parallelism_threads(4)
rng = np.random.default_rng(0)


def hs(a, b):
    T = a.shape[-1]
    d = 1
    while d < T:
        a_prev = tf.concat([tf.ones_like(a[..., :d]), a[..., :-d]], axis=-1)
        b_prev = tf.concat([tf.zeros_like(b[..., :d]), b[..., :-d]], axis=-1)
        b = a * b_prev + b
        a = a * a_prev
        d *= 2
    return b


@tf.custom_gradient
def linrec(a, b):
    """h_t = a_t h_{t-1} + b_t (h_{-1} = 0) along the last axis; backward = one reverse scan."""
    h = hs(a, b)

    def grad(g):
        # dL/db_t = G_t with G_t = g_t + a_{t+1} G_{t+1}  (reverse recurrence)
        a_next = tf.concat([a[..., 1:], tf.zeros_like(a[..., :1])], axis=-1)
        G = tf.reverse(hs(tf.reverse(a_next, [-1]), tf.reverse(g, [-1])), [-1])
        h_prev = tf.concat([tf.zeros_like(h[..., :1]), h[..., :-1]], axis=-1)
        return G * h_prev, G

    return h, grad


def count(fn, *args):
    cf = tf.function(fn).get_concrete_function(*args)
    g = cf.graph
    return len(g.get_operations()) + sum(len(f.node_def) for f in g.as_graph_def().library.function)


def main():
    # 1. GRU chunk carry --------------------------------------------------------------------
    T, C, U = 128, 31, 64
    gru = layers.GRU(U, return_sequences=True, return_state=True)
    x = tf.constant(rng.normal(size=(4, 2 * T, C)).astype(np.float32))
    full, _ = gru(x)
    o1, s1 = gru(x[:, :T])
    o2, _ = gru(x[:, T:], initial_state=s1)
    print("1. GRU carry: max |2 chunks - 1 sequence| =", float(tf.reduce_max(tf.abs(tf.concat([o1, o2], 1) - full))),
          "| cuDNN-eligible:", gru._could_use_gpu_kernel)

    # 2. causality ---------------------------------------------------------------------------
    mha = layers.MultiHeadAttention(num_heads=4, key_dim=16)
    conv = layers.Conv1D(16, 3, padding="causal", dilation_rate=8)
    q = tf.constant(rng.normal(size=(2, 64, 32)).astype(np.float32))
    q2 = tf.concat([q[:, :40], q[:, 40:] + 5.0], axis=1)            # change the future after t=39
    d_mha = tf.reduce_max(tf.abs(mha(q, q, use_causal_mask=True)[:, :40] - mha(q2, q2, use_causal_mask=True)[:, :40]))
    d_mha_nc = tf.reduce_max(tf.abs(mha(q, q)[:, :40] - mha(q2, q2)[:, :40]))
    d_conv = tf.reduce_max(tf.abs(conv(q)[:, :40] - conv(q2)[:, :40]))
    print(f"2. past outputs changed by a future edit: causal MHA {float(d_mha):.2g}, "
          f"unmasked MHA {float(d_mha_nc):.2g}, causal dilated Conv1D {float(d_conv):.2g}")

    # 3. selective EWMA with custom gradient ------------------------------------------------
    K, N = 24, 10240
    logit = tf.Variable(rng.normal(-2.5, 1.0, size=(K, N)).astype(np.float32))
    xs = tf.constant(rng.normal(size=(K, N)).astype(np.float32))
    w = tf.constant(rng.normal(size=(K, N)).astype(np.float32))

    def loss_with(fn):
        def f():
            with tf.GradientTape() as tape:
                al = tf.sigmoid(logit)
                L = tf.reduce_sum(w * fn(1.0 - al, al * xs))
            return tape.gradient(L, logit)
        return f

    g_auto = tf.function(loss_with(hs))()
    g_cust = tf.function(loss_with(linrec))()
    rel = float(tf.reduce_max(tf.abs(g_auto - g_cust)) / tf.reduce_max(tf.abs(g_auto)))
    ops_auto, ops_cust = count(loss_with(hs)), count(loss_with(linrec))
    f_auto, f_cust = tf.function(loss_with(hs)), tf.function(loss_with(linrec))
    f_auto(); f_cust()
    t0 = time.perf_counter(); [f_auto() for _ in range(5)]; ta = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter(); [f_cust() for _ in range(5)]; tc = (time.perf_counter() - t0) / 5
    print(f"3. selective EWMA K={K}, N={N}: custom-gradient vs autodiff max rel diff {rel:.2g}; "
          f"graph ops fwd+bwd {ops_auto} (autodiff) vs {ops_cust} (custom); CPU {ta:.4f} s vs {tc:.4f} s")

    # 4. cost sketch ---------------------------------------------------------------------------
    from neural_trade.core.config import Config
    from neural_trade.registries.models import Models
    cfg = Config()
    today = Models.build(cfg.MODEL_NAME, cfg)
    xb = tf.constant(rng.normal(size=(256, 60)).astype(np.float32))

    def fb_today():
        with tf.GradientTape() as tape:
            outs = today(xb, training=True)
            L = tf.add_n([tf.reduce_mean(o) for o in outs])
        return tape.gradient(L, today.trainable_variables)

    # causal sketch: per-step features [B, P+T, 31] -> GRU(64) -> 2 x (causal MHA + FF) -> per-step towers
    Bc, P, Tc = 16, 256, 1024
    inp = layers.Input((P + Tc, 31))
    h = layers.GRU(64, return_sequences=True)(inp)
    h = layers.Dense(32)(h)
    for _ in range(2):
        a = layers.MultiHeadAttention(num_heads=4, key_dim=16)(h, h, use_causal_mask=True)
        h = layers.LayerNormalization()(h + a)
        f = layers.Dense(32, activation="gelu")(h)
        h = layers.LayerNormalization()(h + layers.Dense(32)(f))
    outs = []
    for k in range(3):
        tw = layers.Dense(16, activation="gelu")(h)
        outs += [layers.Dense(1)(tw), layers.Dense(1, activation="sigmoid")(tw), layers.Dense(1, activation="softplus")(tw)]
    seq = tf.keras.Model(inp, outs)
    xc = tf.constant(rng.normal(size=(Bc, P + Tc, 31)).astype(np.float32))
    mask = tf.concat([tf.zeros((Bc, P, 1)), tf.ones((Bc, Tc, 1))], axis=1)   # loss only after the burn-in

    def fb_seq():
        with tf.GradientTape() as tape:
            o = seq(xc, training=True)
            L = tf.add_n([tf.reduce_sum(oi * mask) / tf.reduce_sum(mask) for oi in o])
        return tape.gradient(L, seq.trainable_variables)

    # same sketch with a local (banded, 64-bar) causal attention mask instead of the full causal mask
    for name, fn, n_pred in (("today B=256 x L=60", fb_today, 256), (f"causal sketch B={Bc} x ({P}+{Tc})", fb_seq, Bc * Tc)):
        f = tf.function(fn)
        f()
        t0 = time.perf_counter(); [f() for _ in range(3)]; t = (time.perf_counter() - t0) / 3
        print(f"4. {name}: CPU fwd+bwd {t:.3f} s for {n_pred} predictions -> {1e6 * t / n_pred:.1f} us/prediction; "
              f"graph ops {count(fn)}")


if __name__ == "__main__":
    main()
