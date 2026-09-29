"""Window-free benchmark kit (NT-059): kernel V1, window assembly D6b, the whole A2 indicator layer
and today's ``LearnableIndicators`` layer - forward and backward time, op counts, and the G-A1 gate.

    C:/Users/Step/miniforge3/envs/nt/python scripts/bench/window_free.py --out PATH.json
        [--device {cpu,gpu}] [--reps N] [--smoke]

``--device cpu`` (the default) forces ``CUDA_VISIBLE_DEVICES=-1`` before TensorFlow loads; NT-060 runs
this same script with ``--device gpu`` (stage 1g, gate G-A2). ``--smoke`` uses tiny sizes, a small B
and 2 repeats (a few seconds; ``tests/test_bench_window_free.py`` runs it in the fast suite) instead of
the window-free plan's sizes (README "Stages and dependencies", stage 1).

Sizes (the plan's, at ``--device cpu`` without ``--smoke``): a 7-day training block's pass span
(10,080 bars plus the burn-in ``M(240)`` plus ``L - 1``) and a 30,720-bar block; K = 31 and 87
channels; B = 256; chunk C = 16 (the plan's recommended default).

G-A1 (``docs/research/2026-09-29-window-free-plan/README.md``, "Stages and dependencies"): the op
census of the whole A2 layer (forward and backward, at every benchmarked size) contains no MatMul,
BatchMatMul, BatchMatMulV2 or Einsum and nothing from the determinism RAISE or host-round-trip lists
(``common.py``); the kernel's CPU precision is within 1e-5 x max|state| and 3e-5 x RMS per channel at
43,008 bars, for the plan's period set and constant and per-bar alpha. Exits 1 and prints which check
failed if either fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- the device must be chosen before TensorFlow (or anything importing it) loads ------------------
def _pre_parse_device(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--device="):
            return a.split("=", 1)[1]
    return "cpu"


_DEVICE = _pre_parse_device(sys.argv[1:])
if _DEVICE == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = Path(HERE).resolve().parents[1]
for _p in (str(REPO / "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neural_trade  # noqa: E402,F401  (DLL PATH + TF_DETERMINISTIC_OPS=1 before TF loads)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import a2_layer  # noqa: E402
import assembly_d6b  # noqa: E402
import common  # noqa: E402
import kernel_v1  # noqa: E402

CSV = str(REPO / "binance_btcusdt_1min_ccxt.csv")
L = a2_layer.L
M240 = kernel_v1.burn_in(240)                     # eps=1e-3, shift=0.5 (the plan's default)
T7 = 10_080 + M240 + L - 1                        # the 7-day training block's per-step pass span
T30720 = 30_720
PRECISION_T = 43_008
PERIODS = [2, 5, 14, 30, 60, 240, 1440, 10080, 40000, 1e6]
INF_LOGIT = -40.0                                 # "no ceiling": alpha = 4e-18, exp(la) == 1.0f


# --------------------------------------------------------------------------------- benchmark builders
def kernel_pass(K, T, seed=0):
    """A bare kernel V1 pass: K independent periods spread log-uniformly over [2, 1440], per-bar
    alpha (the real usage; constant and per-bar alpha cost the same, A/FINDINGS.md Q1)."""
    rng = np.random.default_rng(seed)
    per = np.exp(np.linspace(np.log(2), np.log(1440), K))
    beta = tf.Variable(kernel_v1.logit_period(per).astype(np.float32))
    dx = tf.constant(rng.normal(0, 1e-3, size=T).astype(np.float32))
    shift = tf.constant((0.5 * np.tanh(rng.normal(size=(K, T)))).astype(np.float32))
    w = tf.constant(rng.normal(size=(K, T)).astype(np.float32))

    def fwd():
        lam = beta[:, None] + shift
        la = -tf.nn.softplus(lam)
        h, _ = kernel_v1.linrec(la, -tf.exp(la) * dx[None, :])
        return h

    def fb():
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(fwd() * w)
        return tape.gradient(loss, beta)
    return fwd, fb


def assembly_pass(N, C, B, seed=0):
    """A bare D6b assembly pass: [N, C] -> [B, L, C] windows at B distinct anchors."""
    rng = np.random.default_rng(seed)
    anchors = np.sort(rng.choice(np.arange(L - 1, N), size=B, replace=False)).astype(np.int32)
    inv = assembly_d6b.inverse_map(anchors, N, L)
    V = tf.Variable(rng.normal(size=(N, C)).astype(np.float32))
    a_t, inv_t = tf.constant(anchors), tf.constant(inv)
    W = tf.constant(rng.normal(size=(B, L, C)).astype(np.float32))

    def fwd():
        F = V * 1.0                                # a dense producer, like the kernel's output
        return assembly_d6b.assemble(F, a_t, inv_t, L)

    def fb():
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(fwd() * W)
        return tape.gradient(loss, V)
    return fwd, fb


def a2_layer_pass(N, close, scale, B, min_history=None, seed=0):
    """The whole A2 layer over an N-bar pass span, D6b-assembled at B anchors."""
    min_history = M240 if min_history is None else min_history
    rng = np.random.default_rng(seed)
    c = close[-(N + 1):]
    dx = tf.constant((np.diff(c) / scale).astype(np.float32))
    ctx = tf.constant(a2_layer.context_features(c[1:], scale))
    cum = tf.constant(((c[1:] - c[1]) / scale).astype(np.float32))
    anchors = rng.choice(np.arange(min_history + L - 1, N), size=B, replace=False).astype(np.int32)
    inv = assembly_d6b.inverse_map(anchors, N, L)
    a_t, inv_t = tf.constant(anchors), tf.constant(inv)
    mod = a2_layer.SeriesIndicatorsA2()
    W = tf.constant(rng.normal(size=(B, L, 31)).astype(np.float32))

    def fwd():
        F = mod.series(dx, ctx)
        return a2_layer.assemble(F, cum, a_t, inv_t)

    def fb():
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(fwd() * W)
        return tape.gradient(loss, mod.trainable)
    return fwd, fb


def today_layer_pass(close, scale, B, seed=0):
    """Today's ``LearnableIndicators`` layer (imported from its current public path, src-only, for
    comparison), with the ``meta_adjust`` Dense of ``gru_attention.py``."""
    from neural_trade.core.config import Config
    from neural_trade.models.layers.learnable_indicators import LearnableIndicators

    cfg = Config(CSV_PATH=CSV)
    rng = np.random.default_rng(seed)
    idx = rng.integers(100, len(close), size=B)
    X = np.stack([(close[i - 60:i] - close[i - 1]) / scale for i in idx]).astype(np.float32)
    x = tf.constant(X)
    layer = LearnableIndicators(cfg)
    dense = tf.keras.layers.Dense(18, activation="tanh")
    meta_in = tf.concat([tf.reduce_mean(x, 1, keepdims=True), tf.reduce_max(x, 1, keepdims=True)], 1)
    layer([x, dense(meta_in)])                     # build
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


def bench(name, fwd_fn, fb_fn, reps, out):
    """Runs the op census (forward and forward+backward) and interleaved timing for one variant.

    ``fb_fn`` is traced once (as a ``tf.function``) and that same concrete function is reused for both
    its census and its timing, so each variant costs two tracings (fwd, fb), not three."""
    cf = common.graph_census(fwd_fn)
    fb_tf = tf.function(fb_fn)
    cb = common.graph_census(fb_tf)
    timing = common.interleaved_timing({name: fb_tf}, reps=reps, warm=1)[name]
    out[name] = {
        "fwd_total_ops": cf["total_ops"], "fwd_compute_ops": cf["compute_ops"],
        "fwdbwd_total_ops": cb["total_ops"], "fwdbwd_compute_ops": cb["compute_ops"],
        "raise_on_gpu": cb["raise_on_gpu"], "host_round_trip_on_gpu": cb["host_round_trip_on_gpu"],
        "matmul_like_ops": cb["matmul_like_ops"],
        "static_output_GB_fwdbwd": cb["static_output_bytes"] / 2 ** 30,
        "largest_intermediate_MB": cb["largest_output_bytes"] / 2 ** 20,
        "cpu_fwdbwd": timing,
    }
    print(f"  {name}: {cb['compute_ops']} compute ops, {timing['median_s'] * 1e3:.1f} ms "
          f"[{timing['min_s'] * 1e3:.1f}-{timing['max_s'] * 1e3:.1f}] (n={timing['reps']})")


# ---------------------------------------------------------------------------------------------- G-A1
def check_census(a2_results):
    """G-A1's census half: every A2-layer benchmark entry must be free of MatMul-like ops and of
    everything in the determinism RAISE / host-round-trip lists."""
    bad = set()
    for entry in a2_results.values():
        bad |= set(entry["raise_on_gpu"]) | set(entry["host_round_trip_on_gpu"]) | set(entry["matmul_like_ops"])
    return {"PASS": not bad, "offending_ops": sorted(bad)}


MACD_TRIPLES = [(12, 26, 9), (1440, 10080, 1440)]     # (fast, slow, signal); a short and a long cascade


def _rel_errors(h, ref):
    """Per-row (period or setting) relative error of `h` (TF, [K, T]) against `ref` (float64, [K, T]):
    max|error| / max|state| and RMS(error) / RMS(state)."""
    err = np.abs(np.asarray(h, np.float64) - ref)
    max_state, rms_state = np.abs(ref).max(-1), np.sqrt((ref ** 2).mean(-1))
    rel_max = err.max(-1) / np.maximum(max_state, 1e-300)
    rel_rms = np.sqrt((err ** 2).mean(-1)) / np.maximum(rms_state, 1e-300)
    return rel_max, rel_rms


def _verdict(rel_max, rel_rms, names):
    ok = bool(np.all(rel_max <= 1e-5) and np.all(rel_rms <= 3e-5))
    return ok, {"max_rel_err_vs_max_state": float(rel_max.max()), "max_rel_err_vs_rms_state": float(rel_rms.max()),
                "PASS": ok, "per_period_rel_max": dict(zip(names, map(float, rel_max))),
                "per_period_rel_rms": dict(zip(names, map(float, rel_rms)))}


def _macd_precision(dx32, dx64, shift, triples, chunk):
    """MACD line (d_fast - d_slow) and signal (EWMA(macd, g)): a two-stage cascade the single-stage
    increment/RSI/Bollinger channels above do not exercise (the lead's review of this item). Same
    tolerance, checked per (fast, slow, signal) setting."""
    dxv, dxx = tf.constant(dx32)[None, :], dx64[None, :]
    macd_h, macd_ref, sig_h, sig_ref, labels = [], [], [], [], []
    for f, s, g in triples:
        labels.append(f"{f}-{s}-{g}")
        lam_f64 = kernel_v1.logit_period(f) + shift
        lam_s64 = kernel_v1.logit_period(s) + shift
        lam_g64 = kernel_v1.logit_period(g) + shift

        la_f = -tf.nn.softplus(tf.constant(lam_f64.astype(np.float32))[None, :])
        la_s = -tf.nn.softplus(tf.constant(lam_s64.astype(np.float32))[None, :])
        lam_g32 = tf.constant(lam_g64.astype(np.float32))[None, :]
        la_g, al_g = -tf.nn.softplus(lam_g32), tf.sigmoid(lam_g32)
        d_f, _ = kernel_v1.linrec(la_f, -tf.exp(la_f) * dxv, C=chunk)
        d_s, _ = kernel_v1.linrec(la_s, -tf.exp(la_s) * dxv, C=chunk)
        macd = d_f - d_s
        sig, _ = kernel_v1.linrec(la_g, al_g * macd, C=chunk)

        la_f64, la_s64 = -kernel_v1.softplus64(lam_f64)[None, :], -kernel_v1.softplus64(lam_s64)[None, :]
        la_g64r, al_g64 = -kernel_v1.softplus64(lam_g64)[None, :], kernel_v1.sigmoid64(lam_g64)[None, :]
        d_f64 = kernel_v1.ref_linrec(la_f64, -np.exp(la_f64) * dxx)
        d_s64 = kernel_v1.ref_linrec(la_s64, -np.exp(la_s64) * dxx)
        macd64 = d_f64 - d_s64
        sig64 = kernel_v1.ref_linrec(la_g64r, al_g64 * macd64)

        macd_h.append(macd.numpy()[0])
        macd_ref.append(macd64[0])
        sig_h.append(sig.numpy()[0])
        sig_ref.append(sig64[0])
    macd_h, macd_ref = np.stack(macd_h), np.stack(macd_ref)
    sig_h, sig_ref = np.stack(sig_h), np.stack(sig_ref)
    out, all_pass = {}, True
    for cname, h, ref in (("macd_line", macd_h, macd_ref), ("macd_signal", sig_h, sig_ref)):
        rel_max, rel_rms = _rel_errors(h, ref)
        ok, entry = _verdict(rel_max, rel_rms, labels)
        entry["per_setting_rel_max"], entry["per_setting_rel_rms"] = entry.pop("per_period_rel_max"), entry.pop("per_period_rel_rms")
        out[cname] = entry
        all_pass = all_pass and ok
    return out, all_pass


def check_precision(close, scale, T, periods, chunk=16, macd_triples=MACD_TRIPLES):
    """G-A1's precision half: kernel V1 (production chunk C=16) against the float64 sequential
    recursion, at T bars of the bundled close series, for `periods` (plus the "no ceiling" logit -40),
    constant and per-bar alpha, on the increment / RSI-gain / RSI-loss / Bollinger-variance channels
    and on MACD line / signal for `macd_triples` (a two-stage cascade; A/FINDINGS.md Q1 checks the same
    six channels). Tolerance: max|error| <= 1e-5 x max|state|, and RMS(error) <= 3e-5 x RMS(state), per
    channel and per period / setting."""
    base = np.concatenate([kernel_v1.logit_period(periods), [INF_LOGIT]])
    names = [f"p{int(p)}" for p in periods] + ["p_inf(logit-40)"]
    c = close[-(T + 1):]
    dx64 = np.diff(c) / scale
    dx32 = dx64.astype(np.float32)
    results, all_pass = {}, True
    for mode in ("constant", "per_bar"):
        shift = 0.5 * np.tanh((dx64 - dx64.mean()) / dx64.std()) if mode == "per_bar" else np.zeros(T)
        lam32 = (base[:, None] + shift[None, :]).astype(np.float32)
        lam64 = base[:, None] + shift[None, :]

        lam = tf.constant(lam32)
        dxv = tf.constant(dx32)[None, :]
        la = -tf.nn.softplus(lam)
        dec, al = tf.exp(la), tf.sigmoid(lam)
        d, _ = kernel_v1.linrec(la, -dec * dxv, C=chunk)
        g, _ = kernel_v1.linrec(la, al * tf.nn.relu(dxv), C=chunk)
        ls, _ = kernel_v1.linrec(la, al * tf.nn.relu(-dxv), C=chunk)
        var, _ = kernel_v1.linrec(la, al * tf.square(d), C=chunk)

        la64 = -kernel_v1.softplus64(lam64)
        dec64, al64 = np.exp(la64), kernel_v1.sigmoid64(lam64)
        dxx = dx64[None, :]
        d_ref = kernel_v1.ref_linrec(la64, -dec64 * dxx)
        g_ref = kernel_v1.ref_linrec(la64, al64 * np.maximum(dxx, 0))
        ls_ref = kernel_v1.ref_linrec(la64, al64 * np.maximum(-dxx, 0))
        var_ref = kernel_v1.ref_linrec(la64, al64 * d_ref * d_ref)

        channels = {"d_ema_minus_close": (d.numpy(), d_ref), "rsi_gain_ewma": (g.numpy(), g_ref),
                    "rsi_loss_ewma": (ls.numpy(), ls_ref), "bb_var_ewma_d2": (var.numpy(), var_ref)}
        mode_res = {}
        for cname, (h, ref) in channels.items():
            rel_max, rel_rms = _rel_errors(h, ref)
            ok, entry = _verdict(rel_max, rel_rms, names)
            mode_res[cname] = entry
            all_pass = all_pass and ok

        macd_res, macd_pass = _macd_precision(dx32, dx64, shift, macd_triples, chunk)
        mode_res.update(macd_res)
        all_pass = all_pass and macd_pass
        results[mode] = mode_res
    return {"PASS": bool(all_pass), "T": T, "periods": names,
            "macd_settings": [f"{f}-{s}-{g}" for f, s, g in macd_triples], "channels": results}


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--out", required=True, help="JSON output path (not written into the repo by default)")
    ap.add_argument("--reps", type=int, default=5, help="interleaved timing repeats (>= 5)")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, few repeats (a few seconds)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    scale = common.target_scale(CSV)
    close = common.load_close(CSV)

    if args.smoke:
        Ns, Ks, B, reps, prec_T, periods = (300,), (6,), 6, 2, 400, [2, 5, 14, 30]
    else:
        Ns, Ks, B, reps, prec_T, periods = (T7, T30720), (31, 87), 256, max(5, args.reps), PRECISION_T, PERIODS

    out = {"env": common.env_info(args.device),
           "sizes": {"pass_spans": list(Ns), "K": list(Ks), "B": B, "L": L,
                     "M_burn_in_p240_eps1e-3_shift0.5": M240, "T7_7day_pass": T7,
                     "precision_T": prec_T, "reps": reps},
           "kernel_v1": {}, "assembly_d6b": {}, "a2_layer": {}, "today_layer": {}}

    print(f"device={args.device} sizes(pass_spans)={Ns} K={Ks} B={B} reps={reps}")
    for N in Ns:
        print(f"kernel V1, N={N}:")
        for K in Ks:
            fwd, fb = kernel_pass(K, N)
            bench(f"N{N}_K{K}", fwd, fb, reps, out["kernel_v1"])
        print(f"assembly D6b, N={N}:")
        for C in Ks:
            fwd, fb = assembly_pass(N, C, B)
            bench(f"N{N}_C{C}", fwd, fb, reps, out["assembly_d6b"])
        print(f"A2 layer, N={N}:")
        fwd, fb = a2_layer_pass(N, close, scale, B, min_history=M240 if not args.smoke else 0)
        bench(f"N{N}", fwd, fb, reps, out["a2_layer"])

    print("today's LearnableIndicators layer:")
    fwd, fb = today_layer_pass(close, scale, B)
    bench("today_LearnableIndicators", fwd, fb, reps, out["today_layer"])

    census = check_census(out["a2_layer"])
    precision = check_precision(close, scale, T=prec_T, periods=periods)
    out["g_a1"] = {"census": census, "precision": precision, "PASS": bool(census["PASS"] and precision["PASS"])}
    out["seconds"] = time.perf_counter() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {out_path} ({out['seconds']:.1f}s)")
    print(f"G-A1 census PASS={census['PASS']} offending_ops={census['offending_ops']}")
    print(f"G-A1 precision PASS={precision['PASS']}")

    if not out["g_a1"]["PASS"]:
        failed = []
        if not census["PASS"]:
            failed.append(f"census (offending ops: {census['offending_ops']})")
        if not precision["PASS"]:
            failed.append("precision (see the 'precision' block of the output JSON)")
        print("G-A1 FAILED: " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
