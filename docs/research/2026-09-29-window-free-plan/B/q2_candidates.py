"""Q2: series-mode replacements for the per-window adaptive periods, CPU prototypes (TF 2.10).

(a) per-bar alpha from causal context (today's meta_adjust Dense applied per bar), (b) a bank of M fixed
periods per instance with per-bar hat interpolation, (c) no adaptation (global learned period).
Kernel: kernels.py (two-level chunked form, chunk-local cumsums, Hillis-Steele carry over chunk states;
Toeplitz specialisation for constant alpha). Weights: the newest local run's served logits and meta Dense
(runs/20260924T182915Z-1aeff1c-dirty-af67ee43/artifacts/weights.h5).

Sections (all CPU, CUDA_VISIBLE_DEVICES=-1):
 1. precision against a float64 recursion (30,720 and 43,008 bars; also periods x20 for long memory);
 2. cost: graph ops fwd+bwd and CPU time (median of interleaved repeats) of the full 31-channel layer and
    of bare K-channel passes (K = 31, 87) at N = 30,720 bars;
 3. causality: end-to-end perturbation from the raw closes (context features included), Jacobian, NaN;
 4. how the shift changes the longest effective memory: realized worst case vs the tanh bound.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q2_candidates.py [--reps 7]
Writes q2_candidates.json.
"""
import argparse
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import neural_trade  # noqa: F401
import numpy as np
import tensorflow as tf

import common as C
from candidates import SeriesLayer, hat_weights, reference64
from kernels import graph_ops, linrec_chunked, linrec_toeplitz

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)
NEWEST = f"{C.RUNS}/20260924T182915Z-1aeff1c-dirty-af67ee43/"
GROUP_OF = {}
for k, idx in [("ma", range(0, 3)), ("macd", range(3, 15)), ("rsi", range(15, 18)), ("bb", range(18, 30)), ("raw", [30])]:
    for i in idx:
        GROUP_OF[i] = k


def series_inputs(close, scale, n, ctx_kind="rolling"):
    c = close[-(n + 1):]
    dx = np.diff(c) / scale
    ctx_full = C.rolling_context(close, scale) if ctx_kind == "rolling" else C.ew_context(close, scale)
    ctx = ctx_full[-n:]
    return dx, ctx


def per_group_err(out, ref):
    res = {}
    for g in ("ma", "macd", "rsi", "bb"):
        cols = [i for i in range(31) if GROUP_OF[i] == g]
        e = np.abs(out[:, cols] - ref[:, cols]).max(0)
        m = np.abs(ref[:, cols]).max(0)
        res[g] = {"max_abs_err": float(e.max()), "max_err_over_max_state": float((e / np.maximum(m, 1e-12)).max()),
                  "max_state": float(m.max())}
    return res


def precision(info, close):
    out = {}
    for n in (30720, 43008):
        dx, ctx = series_inputs(close, info["scale"], n)
        for tag, lg in (("run_periods", info["logits"]), ("periods_x20", C.logit_of_period(20 * C.period_of_logit(info["logits"])))):
            lg32 = lg.astype(np.float32).astype(np.float64)
            W32, b32 = info["W"].astype(np.float32).astype(np.float64), info["b"].astype(np.float32).astype(np.float64)
            ctx32 = ctx.astype(np.float32).astype(np.float64)
            delta64 = (0.5 * np.tanh(ctx32 @ W32 + b32)).T
            dx32 = dx.astype(np.float32).astype(np.float64)
            for mode, M in (("c", 1), ("a", 1), ("b", 3), ("b", 5)):
                layer = SeriesLayer(mode, lg, info["W"], info["b"], M=M)
                F = layer(dx.astype(np.float32), ctx.astype(np.float32)).numpy().astype(np.float64)
                R = reference64(mode, dx32, lg32, delta64 if mode != "c" else None, M=M)
                key = f"N{n}_{tag}_{mode}{M if mode == 'b' else ''}"
                out[key] = per_group_err(F, R)
                print("precision", key, {g: "%.2e / %.2e" % (v["max_abs_err"], v["max_err_over_max_state"]) for g, v in out[key].items()})
    return out


def timed_rounds(fns, reps):
    """fns: {name: tf.function without args}. Interleaved rounds; returns {name: [seconds]}."""
    for f in fns.values():
        f()                                                    # trace + warm
        f()
    t = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            g = f()
            _ = [x.numpy() for x in g]
            t[k].append(time.perf_counter() - t0)
    return t


def stats(v):
    v = np.asarray(v)
    return {"median_s": float(np.median(v)), "min_s": float(v.min()), "max_s": float(v.max()),
            "p25_s": float(np.percentile(v, 25)), "p75_s": float(np.percentile(v, 75)), "n": int(len(v))}


def cost_full_layer(info, close, reps, n=30720):
    dx, ctx = series_inputs(close, info["scale"], n)
    dx, ctx = tf.constant(dx.astype(np.float32)), tf.constant(ctx.astype(np.float32))
    R = tf.constant(np.random.default_rng(0).normal(size=(n, 31)).astype(np.float32))
    variants = {"c_fixed_toeplitz": ("c", 1, "toeplitz"), "c_fixed_general_kernel": ("c", 1, "general"),
                "a_perbar": ("a", 1, "general"), "b_bank_M3": ("b", 3, "toeplitz"), "b_bank_M5": ("b", 5, "toeplitz")}
    fns, ops = {}, {}
    for name, (mode, M, kern) in variants.items():
        layer = SeriesLayer(mode, info["logits"], info["W"], info["b"], M=M)
        if kern == "general":                                   # run the fixed alphas through the per-bar kernel
            layer._run_fixed = lambda al, b, L=layer: linrec_chunked(1.0 - al[:, None] * tf.ones_like(b), b, L.C)
        vars_ = [layer.lg, layer.W, layer.b] if mode != "c" else [layer.lg]

        def fb(layer=layer, vars_=vars_):
            with tf.GradientTape() as tape:
                F = layer(dx, ctx)
                L = tf.reduce_mean(tf.square(F * R))
            return tape.gradient(L, vars_)
        ops[name] = graph_ops(fb)
        fns[name] = tf.function(fb)
    t = timed_rounds(fns, reps)
    base = np.median(t["c_fixed_toeplitz"])
    return {k: {"graph_ops_fwd_bwd": ops[k], **stats(t[k]), "time_vs_c_fixed_toeplitz": float(np.median(t[k]) / base),
                "ops_vs_c_fixed_toeplitz": ops[k] / ops["c_fixed_toeplitz"]} for k in fns}


def cost_bare(reps, n=30720, Ks=(31, 87), Cc=64):
    rng = np.random.default_rng(1)
    close = C.blocks(-1)["close"]
    scale = 257.51813253642973
    dx = tf.constant((np.diff(close[-(n + 1):]) / scale).astype(np.float32))
    ctx = tf.constant(C.rolling_context(close, scale)[-n:].astype(np.float32))
    res = {}
    for K in Ks:
        periods = np.exp(rng.uniform(np.log(3), np.log(240), K))
        lg = tf.Variable(C.logit_of_period(periods).astype(np.float32))
        W = tf.Variable(rng.uniform(-0.55, 0.55, (2, K)).astype(np.float32))
        b = tf.Variable(np.zeros(K, np.float32))
        R = tf.constant(rng.normal(size=(K, n)).astype(np.float32))

        def d_input(al):                     # d_t = (1 - a)(d_{t-1} - dx_t): b_t = -(1 - a_t) dx_t
            return -(1.0 - al) * dx[None, :]

        def c_toe():
            al = tf.clip_by_value(tf.sigmoid(lg), 1e-6, 1 - 1e-6)
            return linrec_toeplitz(tf.math.log1p(-al), d_input(al[:, None]) * tf.ones([1, n]), Cc)

        def c_gen():
            al = tf.clip_by_value(tf.sigmoid(lg), 1e-6, 1 - 1e-6)[:, None] * tf.ones([1, n])
            return linrec_chunked(1.0 - al, d_input(al), Cc)

        def a_gen(Cx=Cc):
            delta = tf.transpose(0.5 * tf.tanh(tf.matmul(ctx, W) + b))
            al = tf.clip_by_value(tf.sigmoid(lg[:, None] + delta), 1e-6, 1 - 1e-6)
            return linrec_chunked(1.0 - al, d_input(al), Cx)

        def bank(M):
            off = np.linspace(-0.5, 0.5, M)
            delta = tf.transpose(0.5 * tf.tanh(tf.matmul(ctx, W) + b))
            al = tf.clip_by_value(tf.sigmoid(lg[:, None] + tf.constant(off, tf.float32)[None, :]), 1e-6, 1 - 1e-6)
            h = linrec_toeplitz(tf.math.log1p(-tf.reshape(al, [-1])),
                                tf.reshape(-(1.0 - al)[..., None] * dx[None, None, :], [K * M, n]), Cc)
            return tf.reduce_sum(hat_weights(delta, off) * tf.reshape(h, [K, M, n]), axis=1)

        variants = {"c_fixed_toeplitz": (c_toe, [lg]), "c_fixed_general_kernel": (c_gen, [lg]),
                    "a_perbar_C64": (a_gen, [lg, W, b]), "a_perbar_C32": (lambda: a_gen(32), [lg, W, b]),
                    "b_bank_M3": (lambda: bank(3), [lg, W, b]), "b_bank_M5": (lambda: bank(5), [lg, W, b])}
        fns, ops = {}, {}
        for name, (fn, vs) in variants.items():
            def fb(fn=fn, vs=vs):
                with tf.GradientTape() as tape:
                    L = tf.reduce_mean(tf.square(fn() * R))
                return tape.gradient(L, vs)
            ops[name] = graph_ops(fb)
            fns[name] = tf.function(fb)
        t = timed_rounds(fns, reps)
        base = np.median(t["c_fixed_toeplitz"])
        nch = -(-n // Cc)
        mem = {"c_fixed_toeplitz": K * n * 4, "c_fixed_general_kernel": K * nch * Cc * Cc * 4,
               "a_perbar_C64": K * nch * Cc * Cc * 4, "a_perbar_C32": K * (-(-n // 32)) * 32 * 32 * 4,
               "b_bank_M3": 3 * K * n * 4, "b_bank_M5": 5 * K * n * 4}
        res[f"K{K}"] = {k: {"graph_ops_fwd_bwd": ops[k], **stats(t[k]),
                            "time_vs_c_fixed_toeplitz": float(np.median(t[k]) / base),
                            "ops_vs_c_fixed_toeplitz": ops[k] / ops["c_fixed_toeplitz"],
                            "largest_intermediate_MB_calc": mem[k] / 2 ** 20} for k in fns}
        print("bare", K, {k: "%d ops %.1f ms" % (v["graph_ops_fwd_bwd"], 1e3 * v["median_s"]) for k, v in res[f"K{K}"].items()})
    return res


def causality(info, close, n=30720):
    """End to end: raw closes -> context (numpy) -> dx -> layer. Perturb every close after bar t*."""
    rng = np.random.default_rng(7)
    res = {}
    base_close = close[-(n + 1):].copy()
    tstar = n - 1 - 3000 + 17                                   # inside a chunk (not at a boundary)
    pert = base_close.copy()
    pert[tstar + 2:] += np.cumsum(rng.normal(0, 5 * info["scale"], len(pert) - tstar - 2))   # bars after t* (series index tstar+1..)
    for ctx_kind in ("rolling", "ew"):
        for mode, M in (("a", 1), ("b", 3), ("c", 1)):
            if ctx_kind == "ew" and mode == "c":
                continue
            outs, ctxs = [], []
            for cl in (base_close, pert):
                dx = np.diff(cl) / info["scale"]
                ctx = (C.rolling_context(cl, info["scale"]) if ctx_kind == "rolling" else C.ew_context(cl, info["scale"]))[1:]
                layer = SeriesLayer(mode, info["logits"], info["W"], info["b"], M=M)
                outs.append(layer(dx.astype(np.float32), ctx.astype(np.float32)).numpy())
                ctxs.append(ctx.astype(np.float32))
            key = f"{mode}{M if mode == 'b' else ''}_{ctx_kind}"
            res[key] = {"t_star": int(tstar), "outputs_up_to_t_star_bitwise_equal": bool(np.array_equal(outs[0][:tstar + 1], outs[1][:tstar + 1])),
                        "context_up_to_t_star_bitwise_equal": bool(np.array_equal(ctxs[0][:tstar + 1], ctxs[1][:tstar + 1])),
                        "later_outputs_changed": bool(not np.array_equal(outs[0][tstar + 1:], outs[1][tstar + 1:]))}
    # Jacobian d out[t*] / d dx[t'] for t' > t*: exactly zero (context held fixed: it depends on data only)
    dx0, ctx0 = series_inputs(close, info["scale"], n)
    for mode, M in (("a", 1), ("b", 3), ("c", 1)):
        layer = SeriesLayer(mode, info["logits"], info["W"], info["b"], M=M)
        v = tf.Variable(dx0.astype(np.float32))
        r = tf.constant(np.random.default_rng(3).normal(size=31).astype(np.float32))
        with tf.GradientTape() as tape:
            F = layer(v, tf.constant(ctx0.astype(np.float32)))
            y = tf.reduce_sum(F[tstar] * r)
        g = tape.gradient(y, v).numpy()
        res[f"jacobian_{mode}{M if mode == 'b' else ''}"] = {"n_future_nonzero": int(np.count_nonzero(g[tstar + 1:])),
                                                           "n_past_nonzero": int(np.count_nonzero(g[:tstar + 1]))}
    # NaN in a future bar: how many past outputs become non-finite (chunk-local 0 * NaN)
    for mode, M in (("a", 1), ("c", 1)):
        dxn = dx0.astype(np.float32).copy()
        dxn[tstar + 5] = np.nan
        layer = SeriesLayer(mode, info["logits"], info["W"], info["b"], M=M)
        F = layer(dxn, ctx0.astype(np.float32)).numpy()
        bad = ~np.isfinite(F[:tstar + 1]).all(1)
        res[f"nan_future_{mode}"] = {"past_bars_nonfinite": int(bad.sum()),
                                     "first_bad_bar": int(np.argmax(bad)) if bad.any() else None, "chunk": 64}
    print("causality", res)
    return res


def longest_memory(info, close, eps_list=(1e-2, 1e-3, 1e-4, 1e-6)):
    """Mode (a) on the whole 30-day file: realized worst-case bars for prod(1 - alpha_t) <= eps, per
    instance, against the bound (shift -0.5) and against no shift (mode c)."""
    ctx = C.rolling_context(close, info["scale"])[1:]
    delta = (0.5 * np.tanh(ctx @ info["W"] + info["b"])).T                  # [18, N]
    al = np.clip(C.sigmoid(info["logits"][:, None] + delta), 1e-6, 1 - 1e-6)
    out = {}
    for j, name in enumerate(C.NAMES):
        S = np.concatenate([[0.0], np.cumsum(np.log1p(-al[j]))])           # S[t] = sum_{k<t} log(1-a_k)
        row = {"base_period": float(C.period_of_logit(info["logits"][j])),
               "bound_period_shift_-0.5": float(C.period_of_logit(info["logits"][j] - 0.5))}
        for eps in eps_list:
            target = -S - np.log(eps)                                      # need -S[s+W] >= -S[s] - ln eps
            pos = np.searchsorted(-S, target[:-1], side="left")
            W = pos - np.arange(len(S) - 1)
            ok = pos < len(S)
            a_min = C.sigmoid(info["logits"][j] - 0.5)
            row[f"eps{eps:g}"] = {"realized_worst": int(W[ok].max()), "realized_median": float(np.median(W[ok])),
                                  "bound": C.m_eps(a_min, eps), "no_shift": C.m_eps(C.sigmoid(info["logits"][j]), eps)}
        out[name] = row
    worst = max(out, key=lambda k: out[k]["eps0.001"]["bound"])
    print("longest memory, worst instance", worst, out[worst])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    info = C.run_info(NEWEST)
    close = C.blocks(-1)["close"]
    res = {"run": info["run"], "threads": {"intra": 8, "inter": 2}, "tf": tf.__version__,
           "kernel": "kernels.py two-level chunked (chunk-local cumsums, HS carry), C=64; Toeplitz for fixed alpha"}
    if "prec" not in args.skip:
        res["precision"] = precision(info, close)
    if "cost" not in args.skip:
        res["cost_full_layer_N30720"] = cost_full_layer(info, close, args.reps)
        print("full layer", {k: "%d ops %.1f ms" % (v["graph_ops_fwd_bwd"], 1e3 * v["median_s"]) for k, v in res["cost_full_layer_N30720"].items()})
        res["cost_bare_N30720"] = cost_bare(args.reps)
    if "caus" not in args.skip:
        res["causality"] = causality(info, close)
    if "mem" not in args.skip:
        res["longest_memory_mode_a"] = longest_memory(info, close)
    print("wrote", C.dump(res, "q2_candidates.json"))


if __name__ == "__main__":
    main()
