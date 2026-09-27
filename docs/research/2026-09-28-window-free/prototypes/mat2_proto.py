"""Two-level matrix form of h_t = a_t h_{t-1} + b_t (no log-depth scan): intra-chunk C x C decay
matrices from LOCAL log-cumsums, inter-chunk carry by an n x n decay matrix over chunk states.
Error vs float64, gradient vs float64 tf.scan, graph ops and CPU time vs Hillis-Steele."""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from scan_proto import make_inputs, ref64, scan_hs, scan_seq

tf.config.threading.set_inter_op_parallelism_threads(4)
NEG = -1e30


def _decay_matrix(cs):
    """cs [..., m] (cumsum of log decays) -> [..., m, m] with exp(cs_i - cs_j) for i >= j, else 0."""
    m = cs.shape[-1]
    diff = cs[..., :, None] - cs[..., None, :]
    lower = tf.range(m)[:, None] >= tf.range(m)[None, :]
    return tf.exp(tf.where(lower, diff, tf.fill(tf.shape(diff), tf.constant(NEG, cs.dtype))))


def scan_mat2(a, b, C=128):
    K, T = a.shape
    n = T // C
    la = tf.math.log(tf.reshape(a, [K, n, C]))
    cs = tf.cumsum(la, axis=-1)                                     # local, |cs| <= C * 13.8
    intra = tf.einsum("knij,knj->kni", _decay_matrix(cs), tf.reshape(b, [K, n, C]))
    # chunk c maps the entering state H_{c-1} to A_c H_{c-1} + s_c
    logA = cs[..., -1]                                              # [K, n]
    s = intra[..., -1]
    CS = tf.cumsum(logA, axis=-1)
    H = tf.einsum("kcd,kd->kc", _decay_matrix(CS), s)               # state at the end of chunk c
    H_prev = tf.concat([tf.zeros_like(H[..., :1]), H[..., :-1]], axis=-1)
    return tf.reshape(intra + tf.exp(cs) * H_prev[..., None], [K, T])


def count(fn):
    cf = tf.function(fn).get_concrete_function()
    g = cf.graph
    return len(g.get_operations()) + sum(len(f.node_def) for f in g.as_graph_def().library.function)


def main():
    periods = [2, 5, 14, 30, 60, 240, 1440, 5, 14, 30, 60, 240]
    for T in (10240, 43008):
        alpha, x, scale, close = make_inputs(T, periods)
        a64, b64 = 1 - alpha, alpha * x
        h_ref = ref64(a64, b64)
        for C in (64, 128):
            h = scan_mat2(tf.constant(a64, tf.float32), tf.constant(b64, tf.float32), C).numpy()
            print(f"T={T} C={C}: mat2 max |h - ref64| = {np.max(np.abs(h - h_ref)):.3g}")
    # gradient vs float64 tf.scan
    T = 2048
    alpha, x, scale, close = make_inputs(T, periods)
    w = np.random.default_rng(0).normal(size=alpha.shape)
    lg64 = np.log(alpha) - np.log1p(-alpha)

    def grad(fn, dt):
        lg = tf.Variable(lg64.astype(dt))
        with tf.GradientTape() as tape:
            al = tf.sigmoid(lg)
            L = tf.reduce_sum(tf.constant(w.astype(dt)) * fn(1 - al, al * tf.constant(x.astype(dt))))
        return tape.gradient(L, lg).numpy()
    g_ref = grad(scan_seq, np.float64)
    g = grad(lambda a, b: scan_mat2(a, b, 128), np.float32)
    print(f"mat2 grad vs tf.scan float64: max rel err {np.max(np.abs(g - g_ref)) / np.max(np.abs(g_ref)):.3g}, finite {np.isfinite(g).all()}")
    # ops and CPU time, K = 24
    rng = np.random.default_rng(2)
    for T in (10240, 30720):
        lg = tf.Variable(rng.normal(-2, 1, size=(24, T)).astype(np.float32))
        xx = tf.constant(rng.normal(size=(24, T)).astype(np.float32))
        for name, fn in (("hs", scan_hs), ("mat2_C128", lambda a, b: scan_mat2(a, b, 128)),
                         ("mat2_C256", lambda a, b: scan_mat2(a, b, 256))):
            def fb():
                with tf.GradientTape() as tape:
                    al = tf.sigmoid(lg)
                    L = tf.reduce_sum(tf.square(fn(1 - al, al * xx)))
                return tape.gradient(L, lg)
            f = tf.function(fb); f()
            t0 = time.perf_counter(); [f() for _ in range(3)]; t = (time.perf_counter() - t0) / 3
            print(f"T={T} {name}: graph ops fwd+bwd {count(fb)}, CPU {t:.4f} s")


if __name__ == "__main__":
    main()
