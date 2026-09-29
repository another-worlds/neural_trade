"""Q4 (prototype): gap handling on a synthetic minute series with holes, and training-vs-serving state parity.

Policy (iii), hybrid: dt_t = minutes since the previous bar.
  dt <= G_RESET: elapsed-time decay, log decay la_t = dt_t * (-softplus(logit_t)), input with the one-bar
                 alpha (previous-tick semantics: the price held at close_{t-1} through the gap, then moved):
                 increment form d_t = (1-a)^dt d_{t-1} - (1-a) dx_t ; plain EWMA h_t = (1-a)^dt h_{t-1} + a x_t
  dt >  G_RESET: reset: la_t = NEG (decay exactly 0), input 0 (EMA = close, variance and RSI sums 0), and
                 the next M bars are masked (no loss, no early stopping, no evaluation, no serving).
The data start is a reset. M = burn_in(longest period after the -0.5 shift, eps 1e-3).

Checks
  E  elapsed-time decay == the same kernel on the forward-filled regular grid (constant alpha, no reset)
  A1 Predictor recomputes a prefix pass from the data start (same origin, same chunk phase): bitwise?
  A2 Predictor recomputes from the last reset, with the pass start aligned / not aligned to the training
     chunk grid: bitwise?
  A3 Predictor recomputes from a cold start M + L - 1 bars back (no reset in between): relative difference
  B  carry-forward: the bundle stores the state at the end of training; the service steps bar by bar in
     float32 (same dt / reset rule): relative difference to the training pass
Command: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q4_serving.py
Output:  q4_serving.json
"""
from common import dump, np, tf
from kernel import NEG, burn_in, linrec, logit_period

C, G = 64, 32
G_RESET = 60
PHANTOM_VAR = True   # set per run in main()
PER = np.array([5, 30, 60, 240], float)
M = burn_in(PER.max())
L = 60
rng = np.random.default_rng(11)

# ---------------------------------------------------------------- synthetic minute series with holes
T_WALL = 40000
walk = 50000.0 + np.cumsum(rng.normal(0, 20.0, size=T_WALL))
present = np.ones(T_WALL, bool)
for s in rng.choice(np.arange(500, T_WALL - 500), size=60, replace=False):
    present[s:s + int(rng.integers(1, 4))] = False              # 60 short gaps of 1-3 minutes
for s, n in ((12000, 30), (20000, 360), (26000, 2880)):
    present[s:s + n] = False                                     # 30 min, 6 h, 2 days
t_min = np.nonzero(present)[0].astype(np.int64)
close = walk[present]
N = len(t_min)
SCALE = float(np.std(close[20:] - close[:-20]))
SIGMA = float(np.std(np.diff(close)) / SCALE)                    # a bundle constant for the context


def inputs(t_min, close, g_reset=G_RESET, perbar=True):
    """Kernel inputs from raw bars (timestamps in minutes, closes): the SAME function in training and serving."""
    dx = np.diff(close, prepend=close[0]) / SCALE
    dt = np.diff(t_min, prepend=t_min[0] - 1).astype(np.float64)
    reset = dt > g_reset
    reset[0] = True                                              # the data start (first bar given)
    z = dx / SIGMA
    lam = logit_period(PER)[:, None] + (0.5 * np.tanh(z)[None, :] if perbar else 0.0)
    lam32 = lam.astype(np.float32)
    return dx.astype(np.float32), dt.astype(np.float32), reset, lam32


def tf_states(dx, dt, reset, lam32, C=C):
    lam = tf.constant(lam32)
    la1 = -tf.nn.softplus(lam)                                   # one-bar log decay
    la = tf.where(tf.constant(reset)[None, :], tf.constant(NEG, tf.float32), la1 * tf.constant(dt)[None, :])
    keep = tf.constant((~reset).astype(np.float32))[None, :]
    dxt = tf.constant(dx)[None, :]
    d, _ = linrec(la, -tf.exp(la1) * dxt * keep, C=C, G=G)
    al = tf.sigmoid(lam)
    b_var = al * tf.square(d) * keep
    if PHANTOM_VAR:
        # previous-tick phantom bars inside a gap of dt minutes feed a*d_k^2 with d_k = (1-a)^k d_{t-1}:
        # closed form (1-a)^(dt+1) (1 - (1-a)^(dt-1)) d_{t-1}^2 (0 when dt = 1)
        dtt = tf.constant(dt)[None, :]
        d_prev = tf.concat([tf.zeros_like(d[:, :1]), d[:, :-1]], 1)
        b_var += tf.exp(la1 * (dtt + 1.0)) * (1.0 - tf.exp(la1 * (dtt - 1.0))) * tf.square(d_prev) * keep
    var, _ = linrec(la, b_var, C=C, G=G)
    gain, _ = linrec(la, al * tf.nn.relu(dxt) * keep, C=C, G=G)
    return np.concatenate([d.numpy(), var.numpy(), gain.numpy()], 0)


def step_states(state, dx, dt, reset, lam32):
    """Carry-forward serving: one bar at a time, float32, the same rule."""
    s = state.astype(np.float32).copy()
    K = len(PER)
    out = np.empty((3 * K, len(dx)), np.float32)
    for t in range(len(dx)):
        lam = lam32[:, t]
        la1 = -np.logaddexp(np.float32(0), lam).astype(np.float32)
        dec = np.float32(0) if reset[t] else np.exp(la1 * dt[t]).astype(np.float32)
        keep = np.float32(0) if reset[t] else np.float32(1)
        al = (1 / (1 + np.exp(-lam))).astype(np.float32)
        d_prev = s[:K].copy()
        d = dec * s[:K] - np.exp(la1) * dx[t] * keep
        extra = (np.exp(la1 * (dt[t] + 1)) * (1 - np.exp(la1 * (dt[t] - 1))) * d_prev * d_prev * keep).astype(np.float32)
        var = dec * s[K:2 * K] + (al * d * d * keep + extra).astype(np.float32)
        gain = dec * s[2 * K:] + al * np.maximum(dx[t], 0) * keep
        s = np.concatenate([d, var, gain]).astype(np.float32)
        out[:, t] = s
    return out


def rel(a, b, ref):
    return float(np.abs(a.astype(np.float64) - b).max() / np.abs(ref).max())


def main():
    dx, dt, reset, lam32 = inputs(t_min, close)
    H = tf_states(dx, dt, reset, lam32)
    reset_idx = np.nonzero(reset)[0]
    masked = np.zeros(N, bool)
    for r in reset_idx:
        masked[r:r + M] = True
    out = {"N_bars": int(N), "missing_minutes": int(T_WALL - N), "G_reset_minutes": G_RESET, "M_burn_in": M,
           "periods": PER.tolist(), "resets_at_bars": reset_idx.tolist(),
           "reset_gap_minutes": [int(dt[r]) for r in reset_idx[1:]], "masked_anchors": int(masked.sum()),
           "short_gaps_handled_by_elapsed_time": int(((dt > 1) & (dt <= G_RESET)).sum())}

    # E: elapsed-time decay == kernel on the forward-filled minute grid (constant alpha, no reset except the start)
    global PHANTOM_VAR
    K = len(PER)
    grid_t = np.arange(t_min[0], t_min[-1] + 1)
    grid_c = close[np.searchsorted(t_min, grid_t, side="right") - 1]   # previous-tick forward fill
    e_out = {}
    for phantom in (False, True):
        PHANTOM_VAR = phantom
        dxc, dtc, rc, lamc = inputs(t_min, close, g_reset=10 ** 9, perbar=False)
        He = tf_states(dxc, dtc, rc, lamc)
        dxg, dtg, rg, lamg = inputs(grid_t, grid_c, g_reset=10 ** 9, perbar=False)
        Hg = tf_states(dxg, dtg, rg, lamg)[:, t_min - t_min[0]]
        e_out[f"phantom_variance_term_{phantom}"] = {g: rel(He[i * K:(i + 1) * K], Hg[i * K:(i + 1) * K], Hg[i * K:(i + 1) * K])
                                                     for i, g in enumerate(("d_ema_minus_close", "bb_var", "rsi_gain"))}
    PHANTOM_VAR = True
    out["E_elapsed_time_vs_ffill_grid_max_rel"] = e_out

    # the state right after the 2-day outage, with and without a reset
    big = int(reset_idx[-1])
    out["after_2day_outage"] = {"bar": big, "jump_dx_scaled": float(dx[big]),
                                "d_with_elapsed_time_only": He[:len(PER), big].tolist(),
                                "d_with_reset": H[:len(PER), big].tolist(),
                                "typical_abs_d_p5_p30_p60_p240": np.median(np.abs(H[:len(PER)]), 1).tolist()}

    serve = np.sort(rng.choice(np.nonzero(~masked)[0], size=30, replace=False))
    a1_bit, a1_rel, a2a_bit, a2u_bit, a2g_bit, a2_rel, a3_rel = [], [], [], [], [], [], []
    for ts in serve:
        # A1: prefix from the data start (same origin and chunk phase), history up to ts
        d1 = inputs(t_min[:ts + 1], close[:ts + 1])
        h1 = tf_states(*d1)[:, -1]
        a1_bit.append(bool(np.array_equal(h1.view(np.uint32), H[:, ts].view(np.uint32))))
        a1_rel.append(rel(h1, H[:, ts], H))
        # A2: from the last reset; start at the reset bar (unaligned) or at the chunk boundary before it (aligned)
        r = int(reset_idx[reset_idx <= ts].max())
        for start, bucket in ((r, a2u_bit), ((r // C) * C, a2a_bit), ((r // (C * G)) * (C * G), a2g_bit)):
            dd = list(inputs(t_min[start:ts + 1], close[start:ts + 1]))
            if start < r:                                         # keep the reset at bar r of the original series
                dd[2] = np.zeros_like(dd[2])
                dd[2][r - start] = True
                dd[2][0] = True
            h2 = tf_states(*dd)[:, -1]
            bucket.append(bool(np.array_equal(h2.view(np.uint32), H[:, ts].view(np.uint32))))
            a2_rel.append(rel(h2, H[:, ts], H))
        # A3: cold start M + L - 1 bars back
        s3 = max(0, ts - (M + L - 1))
        h3 = tf_states(*inputs(t_min[s3:ts + 1], close[s3:ts + 1]))[:, -1]
        a3_rel.append(rel(h3, H[:, ts], H))
    out["A1_prefix_same_origin"] = {"bitwise_equal": f"{sum(a1_bit)}/{len(a1_bit)}", "max_rel": max(a1_rel)}
    out["A2_from_last_reset"] = {"bitwise_equal_start_at_reset_bar": f"{sum(a2u_bit)}/{len(a2u_bit)}",
                                 "bitwise_equal_start_aligned_to_training_chunk_grid_C": f"{sum(a2a_bit)}/{len(a2a_bit)}",
                                 "bitwise_equal_start_aligned_to_level2_grid_CxG": f"{sum(a2g_bit)}/{len(a2g_bit)}",
                                 "max_rel": max(a2_rel)}
    out["A3_cold_start_M_plus_L_back"] = {"max_rel": max(a3_rel), "eps": 1e-3,
                                          "note": "truncation, not rounding: bounded by eps x the state before the pass start"}
    # B: carry-forward serving from the midpoint
    mid = N // 2
    Hb = step_states(H[:, mid], dx[mid + 1:], dt[mid + 1:], reset[mid + 1:], lam32[:, mid + 1:])
    out["B_carry_forward_float32_steps"] = {"n_steps": int(N - mid - 1), "max_rel": rel(Hb, H[:, mid + 1:], H),
                                            "bitwise_equal_fraction": float(np.mean(Hb.view(np.uint32) == H[:, mid + 1:].view(np.uint32)))}
    print(out)
    dump("q4_serving.json", out)


if __name__ == "__main__":
    main()
