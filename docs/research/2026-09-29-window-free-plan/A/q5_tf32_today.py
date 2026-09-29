"""Q5: today's windowed EWMA (utils/math.py ewma_sequence_matrix_multi, an einsum -> BatchMatMulV2) under
TensorFloat-32, which TF 2.10 enables by default on Ampere/Ada GPUs (tensor_float_32_execution_enabled() is
True in the nt env; the repo never disables it). Emulated on the CPU by rounding both einsum inputs to TF32
(10 mantissa bits; round-to-nearest and truncation), fp32 accumulation. Estimate for the GPU, not measured.
Error of the 60-bar window EWMA outputs against the float64 recursion, 500 random real windows, periods of
today's defaults; relative to each channel's max |value| in the window.
Output: q5_tf32_today.json"""
from common import dump, load_close, np, target_scale, tf
from kernel import round_tf32

S = target_scale()
CLOSE = load_close()
per = np.array([5, 10, 30, 12, 26, 9, 5, 35, 5, 8, 17, 9, 10, 20, 25, 9, 14, 21], float)
a32 = (2.0 / (per + 1.0)).astype(np.float32)


def windowed(x, a, mode):
    """ewma_sequence_matrix_multi with optional TF32 input rounding of the einsum (same formula as utils/math.py)."""
    x = tf.constant(x)
    a = tf.clip_by_value(tf.constant(a), 1e-6, 1 - 1e-6)
    t = tf.range(60)
    lag = t[:, None] - t[None, :]
    lag_f = tf.cast(tf.maximum(lag, 0), tf.float32)
    decay = tf.exp(lag_f[None, None] * tf.math.log1p(-a)[:, :, None, None])
    w = decay * a[:, :, None, None]
    w = tf.where(tf.equal(t, 0)[None, None, None, :], decay, w)
    w = tf.where((lag >= 0)[None, None], w, tf.zeros_like(w))
    if mode:
        w, x = round_tf32(w, mode), round_tf32(x, mode)
    return tf.einsum("bjtk,bjk->bjt", w, x).numpy()


rng = np.random.default_rng(5)
idx = rng.integers(100, len(CLOSE), size=500)
X = np.stack([(CLOSE[i - 60:i] - CLOSE[i - 1]) / S for i in idx]).astype(np.float32)
Xk = np.repeat(X[:, None, :], len(per), 1)
A = np.repeat(a32[None, :], 500, 0)
ref = np.empty_like(Xk, dtype=np.float64)
ref[..., 0] = Xk[..., 0]
for t in range(1, 60):
    ref[..., t] = (1 - A.astype(np.float64)) * ref[..., t - 1] + A * Xk[..., t]
out = {}
for mode in (None, "rn", "rz"):
    y = windowed(Xk, A, mode)
    err = np.abs(y - ref)
    relw = (err.max(-1) / np.abs(ref).max(-1))
    out[str(mode or "fp32")] = {"max_abs_err_scaled_units": float(err.max()), "median_rel_err_per_window_channel": float(np.median(relw)),
                                "p99_rel_err": float(np.percentile(relw, 99)), "max_abs_err_dollars": float(err.max() * S)}
print(out)
dump("q5_tf32_today.json", out)
