"""Q3: per-step series pass cost (CPU fwd+bwd, graph op counts, fp32 intermediates for the GPU).

Part A  bare kernel pass, K = 31 / 87 channels, constant / per-bar alpha, C = 64 / 128, mulsum / einsum,
        over a 7-day block (10,080 bars + M + L - 1 = 11,501 bars; M = burn_in(240) at eps 1e-3 after the
        -0.5 logit shift = 1,362) and a 30,720-bar block.
Part B  the A2 layer (today's 31 channels over the pass span, per-bar meta shift, D6 window assembly at
        256 anchors) against today's LearnableIndicators layer (with its meta_adjust Dense) at B = 256.
Part C  span-restricted pass: S anchors + M + L - 1 bars, S = 1,024 / 2,048 / 4,096.
Part D  forward-only pass for one validation block (2,866 anchors + M + L - 1 bars), once per validation.

Command: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q3_span.py
Output:  q3_span.json
"""
import time

from common import dump, env_info, graph_census, interleaved_timing, load_close, np, target_scale, tf
from kernel import burn_in, linrec, logit_period
from a2layer import L, SeriesIndicatorsA2, assemble_D6, context_features

tf.config.threading.set_inter_op_parallelism_threads(4)
S_SCALE = target_scale()
CLOSE = load_close()
M240 = burn_in(240)
T7 = 10080 + M240 + L - 1


def bare_pass(K, T, perbar, C, contract):
    rng = np.random.default_rng(K + T)
    per = np.exp(np.linspace(np.log(2), np.log(1440), K))
    beta = tf.Variable(logit_period(per).astype(np.float32))
    c = CLOSE[-(T + 1):]
    dx = tf.constant((np.diff(c) / S_SCALE).astype(np.float32))
    shift = tf.constant((0.5 * np.tanh(rng.normal(size=(K, T)))).astype(np.float32)) if perbar else None
    w = tf.constant(rng.normal(size=(K, T)).astype(np.float32))

    def fwd():
        lam = beta[:, None] + (shift if perbar else tf.zeros([1, T]))
        la = -tf.nn.softplus(lam)
        h, _ = linrec(la, -tf.exp(la) * dx[None, :], C=C, contract=contract)
        return h

    def fb():
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(fwd() * w)
        return tape.gradient(loss, beta)
    return fwd, fb


def today_layer(B=256):
    """Today's LearnableIndicators with the meta_adjust Dense of gru_attention.py:47-56."""
    from neural_trade.core.config import Config
    from neural_trade.models.layers.learnable_indicators import LearnableIndicators
    cfg = Config(CSV_PATH="D:/neural_trade/binance_btcusdt_1min_ccxt.csv")
    rng = np.random.default_rng(0)
    idx = rng.integers(100, len(CLOSE), size=B)
    X = np.stack([(CLOSE[i - 60:i] - CLOSE[i - 1]) / S_SCALE for i in idx]).astype(np.float32)
    x = tf.constant(X)
    layer = LearnableIndicators(cfg)
    dense = tf.keras.layers.Dense(18, activation="tanh")
    meta_in = tf.concat([tf.reduce_mean(x, 1, keepdims=True), tf.reduce_max(x, 1, keepdims=True)], 1)
    layer([x, dense(meta_in)])
    W = tf.constant(rng.normal(size=(B, 60, 31)).astype(np.float32))
    params = layer.trainable_variables + dense.trainable_variables

    def fwd():
        mi = tf.concat([tf.reduce_mean(x, 1, keepdims=True), tf.reduce_max(x, 1, keepdims=True)], 1)
        return layer([x, dense(mi)])

    def fb():
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(fwd() * W)
        return tape.gradient(loss, params)
    return fwd, fb


def a2_layer(N, contract, B=256, C=64):
    rng = np.random.default_rng(N)
    c = CLOSE[-(N + 1):]
    dx = tf.constant((np.diff(c) / S_SCALE).astype(np.float32))
    ctx = tf.constant(context_features(c[1:], S_SCALE))
    cum = tf.constant(((c[1:] - c[1]) / S_SCALE).astype(np.float32))
    anchors = rng.choice(np.arange(M240 + L - 1, N), size=B, replace=False).astype(np.int32)
    inv = np.full(N + L - 1, B, np.int32)
    inv[anchors] = np.arange(B, dtype=np.int32)
    a_t, inv_t = tf.constant(anchors), tf.constant(inv)
    mod = SeriesIndicatorsA2(C=C, contract=contract)
    W = tf.constant(rng.normal(size=(B, L, 31)).astype(np.float32))

    def fwd():
        return assemble_D6(mod.series(dx, ctx), cum, a_t, inv_t)

    def fb():
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(fwd() * W)
        return tape.gradient(loss, mod.trainable)

    def series_only():
        return mod.series(dx, ctx)
    return fwd, fb, series_only


def census_and_time(fns, reps=7):
    res = {}
    for k, (fwd, fb) in fns.items():
        cf, cb = graph_census(fwd), graph_census(fb)
        res[k] = {"fwd_total_ops": cf["total_ops"], "fwd_compute_ops": cf["compute_ops"],
                  "fwdbwd_total_ops": cb["total_ops"], "fwdbwd_compute_ops": cb["compute_ops"],
                  "largest_intermediate_MB": cb["largest_output_bytes"] / 2 ** 20,
                  "static_output_GB_fwdbwd": cb["static_output_bytes"] / 2 ** 30,
                  "raise_on_gpu": cb["raise_on_gpu"], "host_round_trip_on_gpu": cb["host_round_trip_on_gpu"],
                  "top_types": dict(list(cb["types"].items())[:12])}
    tfn = {k: tf.function(fb) for k, (fwd, fb) in fns.items()}
    t0 = time.perf_counter()
    tim = interleaved_timing({k: f for k, f in tfn.items()}, reps=reps, warm=1)
    for k in res:
        res[k]["cpu_fwdbwd"] = tim[k]
    print(f"   timing {time.perf_counter() - t0:.0f}s:", {k: round(v["median_s"] * 1e3, 1) for k, v in tim.items()})
    return res


def main():
    out = {"env": env_info(), "M_burn_in_p240_eps1e-3_shift0.5": M240, "T_7day_pass": T7}
    # ---------------- Part A: bare kernel pass
    partA = {}
    for T in (T7, 30720):
        for K in (31, 87):
            fns = {}
            for perbar in (False, True):
                for C in (64, 128):
                    for contract in ("mulsum", "einsum"):
                        if C == 128 and contract == "einsum":
                            continue
                        fns[f"{'perbar' if perbar else 'const'}_C{C}_{contract}"] = bare_pass(K, T, perbar, C, contract)
            print(f"Part A T={T} K={K}")
            partA[f"T{T}_K{K}"] = census_and_time(fns)
    out["A_bare_pass"] = partA
    # ---------------- Part B: A2 layer vs today's layer
    print("Part B")
    fnsB = {"today_LearnableIndicators_B256": today_layer()}
    series_fns = {}
    for N in (T7, 30720):
        for contract in ("mulsum", "einsum"):
            fwd, fb, so = a2_layer(N, contract)
            fnsB[f"A2_N{N}_{contract}_D6"] = (fwd, fb)
            series_fns[f"N{N}_{contract}"] = so
    out["B_layer"] = census_and_time(fnsB)
    # ---------------- Part C: span-restricted pass (S anchors + M + L - 1 bars)
    print("Part C")
    fnsC = {}
    for Sa in (1024, 2048, 4096):
        fwd, fb, _ = a2_layer(Sa + M240 + L - 1, "mulsum")
        fnsC[f"A2_span{Sa}_N{Sa + M240 + L - 1}_mulsum"] = (fwd, fb)
    out["C_span_restricted"] = census_and_time(fnsC)
    # ---------------- Part D: forward-only series for one validation block, once per validation pass
    print("Part D")
    Nv = 2866 + M240 + L - 1
    _, _, so_val = a2_layer(Nv, "mulsum")
    fD = {"val_block_series_fwd_only": tf.function(so_val)}
    cD = graph_census(so_val)
    tD = interleaved_timing(fD, reps=7)
    out["D_eval_once"] = {"N_bars": Nv, "fwd_total_ops": cD["total_ops"], "fwd_compute_ops": cD["compute_ops"],
                          "largest_intermediate_MB": cD["largest_output_bytes"] / 2 ** 20, "cpu_fwd": tD["val_block_series_fwd_only"],
                          "passes_per_validation_if_per_batch": int(np.ceil(2866 / 256)), "passes_if_once": 1}
    print(out["D_eval_once"])
    dump("q3_span.json", out)


if __name__ == "__main__":
    main()
