"""Q1 addendum: one cold-started 60-bar window through the series kernel vs today's
ewma_sequence_matrix_multi (both float32), 500 random windows, absolute and relative error; plus both
against a float64 recursion (which of the two float32 forms carries the error?).
Output: q1_window_repro.json"""
from common import dump, load_close, np, target_scale, tf
import neural_trade.utils.math as mh
from kernel import linrec

S = target_scale()
CLOSE = load_close()
rng = np.random.default_rng(1)
per = np.array([5, 10, 30, 12, 26, 9, 5, 35, 5, 8, 17, 9, 10, 20, 25, 9, 14, 21], float)
a = (2.0 / (per + 1.0)).astype(np.float32)
rows = []
for _ in range(500):
    i = int(rng.integers(100, len(CLOSE) - 1))
    x = ((CLOSE[i - 60:i] - CLOSE[i - 1]) / S).astype(np.float32)
    today = mh.ewma_sequence_matrix_multi(tf.constant(np.broadcast_to(x, (1, len(per), 60)).copy()),
                                          tf.constant(a[None, :])).numpy()[0]
    lam = tf.constant(np.log(a) - np.log1p(-a))[:, None] * tf.ones([1, 60])
    h, _ = linrec(-tf.nn.softplus(lam), tf.sigmoid(lam) * tf.constant(x)[None, :], h0=tf.fill([len(per)], x[0]), C=64)
    h = h.numpy()
    a64 = a.astype(np.float64)[:, None]
    ref = np.empty((len(per), 60))
    ref[:, 0] = x[0]
    for t in range(1, 60):
        ref[:, t] = (1 - a64[:, 0]) * ref[:, t - 1] + a64[:, 0] * x[t]
    rows.append((np.abs(h - today).max(), np.abs(today).max(), np.abs(h - ref).max(), np.abs(today - ref).max()))
r = np.array(rows)
out = {"n_windows": len(r), "kernel_vs_today_max_abs": float(r[:, 0].max()),
       "kernel_vs_today_max_rel_of_window_max": float((r[:, 0] / r[:, 1]).max()),
       "kernel_vs_float64_max_abs": float(r[:, 2].max()), "today_vs_float64_max_abs": float(r[:, 3].max()),
       "max_abs_state": float(r[:, 1].max()), "PASS_abs_1e-6": bool(r[:, 0].max() <= 1e-6),
       "PASS_rel_1e-5_of_state": bool((r[:, 0] / r[:, 1]).max() <= 1e-5)}
print(out)
dump("q1_window_repro.json", out)
