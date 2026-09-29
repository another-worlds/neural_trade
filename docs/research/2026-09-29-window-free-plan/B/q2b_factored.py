"""Q2 follow-up: the per-bar alpha through a FACTORED chunk-local kernel (kernels.linrec_factored: no C x C
decay matrix, memory O(K T)) against the matrix chunked kernel and the fixed-period Toeplitz form.

1. precision of the full 31-channel mode-(a) layer against float64 (30,720 and 43,008 bars; the run's
   periods and periods x20), and finiteness at the alpha ceiling the kernel allows (C = 32: alpha 0.918);
2. cost: graph ops fwd+bwd and CPU time (interleaved medians) at N = 30,720 for the full layer and for
   bare K = 31 / 87 passes.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q2b_factored.py [--reps 7]
Writes q2b_factored.json.
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import neural_trade  # noqa: F401
import numpy as np
import tensorflow as tf

import common as C
from candidates import SeriesLayer, hat_weights, reference64
from kernels import graph_ops, linrec_chunked, linrec_factored, linrec_toeplitz
from q2_candidates import NEWEST, per_group_err, series_inputs, stats, timed_rounds

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)


def layer_a(info, Cf):
    L = SeriesLayer("a", info["logits"], info["W"], info["b"])
    L._run_perbar = lambda al, b, Cf=Cf: linrec_factored(1.0 - al, b, Cf)
    return L


def precision(info, close):
    out = {}
    for n in (30720, 43008):
        dx, ctx = series_inputs(close, info["scale"], n)
        for tag, lg in (("run_periods", info["logits"]),
                        ("periods_x20", C.logit_of_period(20 * C.period_of_logit(info["logits"])))):
            lg32 = lg.astype(np.float32).astype(np.float64)
            W32, b32 = (info["W"].astype(np.float32).astype(np.float64), info["b"].astype(np.float32).astype(np.float64))
            delta64 = (0.5 * np.tanh(ctx.astype(np.float32).astype(np.float64) @ W32 + b32)).T
            R = reference64("a", dx.astype(np.float32).astype(np.float64), lg32, delta64)
            for Cf in (16, 32):
                inf2 = dict(info, logits=lg)
                F = layer_a(inf2, Cf)(dx.astype(np.float32), ctx.astype(np.float32)).numpy().astype(np.float64)
                out[f"N{n}_{tag}_a_factored_C{Cf}"] = per_group_err(F, R)
                print("precision", n, tag, Cf, {g: "%.2e / %.2e" % (v["max_abs_err"], v["max_err_over_max_state"])
                                                for g, v in out[f"N{n}_{tag}_a_factored_C{Cf}"].items()})
    # finiteness and error at the largest alpha the C=32 kernel admits, and at today's applied floor
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (4, 30720))
    for Cf, alpha in ((32, 0.918), (32, 0.767), (16, 0.99), (16, 0.767)):
        a = np.full_like(x, 1 - alpha)
        a[1] = 1 - rng.uniform(0.001, alpha, 30720)              # per-bar mix of long and short periods
        h = linrec_factored(tf.constant(a, tf.float32), tf.constant(x * (1 - a), tf.float32), Cf).numpy()
        ref = C.linrec64(a, x * (1 - a))
        out[f"edge_C{Cf}_alpha{alpha}"] = {"finite": bool(np.isfinite(h).all()),
                                           "max_abs_err": float(np.abs(h - ref).max()),
                                           "max_state": float(np.abs(ref).max())}
    print("edge", {k: v for k, v in out.items() if k.startswith("edge")})
    return out


def cost(info, close, reps, n=30720):
    dx, ctx = series_inputs(close, info["scale"], n)
    dx, ctx = tf.constant(dx.astype(np.float32)), tf.constant(ctx.astype(np.float32))
    R = tf.constant(np.random.default_rng(0).normal(size=(n, 31)).astype(np.float32))
    layers = {"c_fixed_toeplitz": SeriesLayer("c", info["logits"], info["W"], info["b"]),
              "a_perbar_matrix_C64": SeriesLayer("a", info["logits"], info["W"], info["b"]),
              "a_perbar_factored_C32": layer_a(info, 32), "a_perbar_factored_C16": layer_a(info, 16),
              "b_bank_M3": SeriesLayer("b", info["logits"], info["W"], info["b"], M=3)}
    fns, ops = {}, {}
    for name, layer in layers.items():
        vs = [layer.lg] if layer.mode == "c" else [layer.lg, layer.W, layer.b]

        def fb(layer=layer, vs=vs):
            with tf.GradientTape() as tape:
                L = tf.reduce_mean(tf.square(layer(dx, ctx) * R))
            return tape.gradient(L, vs)
        ops[name] = graph_ops(fb)
        fns[name] = tf.function(fb)
    t = timed_rounds(fns, reps)
    base = np.median(t["c_fixed_toeplitz"])
    full = {k: {"graph_ops_fwd_bwd": ops[k], **stats(t[k]), "time_vs_c_fixed_toeplitz": float(np.median(t[k]) / base),
                "ops_vs_c_fixed_toeplitz": ops[k] / ops["c_fixed_toeplitz"]} for k in fns}
    print("full layer", {k: "%d ops %.1f ms (x%.2f)" % (v["graph_ops_fwd_bwd"], 1e3 * v["median_s"], v["time_vs_c_fixed_toeplitz"])
                         for k, v in full.items()})
    # bare passes
    rng = np.random.default_rng(1)
    bare = {}
    for K in (31, 87):
        periods = np.exp(rng.uniform(np.log(3), np.log(240), K))
        lg = tf.Variable(C.logit_of_period(periods).astype(np.float32))
        W = tf.Variable(rng.uniform(-0.55, 0.55, (2, K)).astype(np.float32))
        b = tf.Variable(np.zeros(K, np.float32))
        Rk = tf.constant(rng.normal(size=(K, n)).astype(np.float32))

        def perbar(kern):
            delta = tf.transpose(0.5 * tf.tanh(tf.matmul(ctx, W) + b))
            al = tf.clip_by_value(tf.sigmoid(lg[:, None] + delta), 1e-6, 1 - 1e-6)
            return kern(1.0 - al, -(1.0 - al) * dx[None, :])

        def fixed():
            al = tf.clip_by_value(tf.sigmoid(lg), 1e-6, 1 - 1e-6)
            return linrec_toeplitz(tf.math.log1p(-al), -(1.0 - al)[:, None] * dx[None, :], 64)

        def fixed_factored():
            al = tf.clip_by_value(tf.sigmoid(lg), 1e-6, 1 - 1e-6)[:, None] * tf.ones([1, n])
            return linrec_factored(1.0 - al, -(1.0 - al) * dx[None, :], 32)

        variants = {"c_fixed_toeplitz": (fixed, [lg]), "c_fixed_factored_C32": (fixed_factored, [lg]),
                    "a_factored_C32": (lambda: perbar(lambda a_, b_: linrec_factored(a_, b_, 32)), [lg, W, b]),
                    "a_factored_C16": (lambda: perbar(lambda a_, b_: linrec_factored(a_, b_, 16)), [lg, W, b]),
                    "a_matrix_C64": (lambda: perbar(lambda a_, b_: linrec_chunked(a_, b_, 64)), [lg, W, b])}
        fns, ops = {}, {}
        for name, (fn, vs) in variants.items():
            def fb(fn=fn, vs=vs):
                with tf.GradientTape() as tape:
                    L = tf.reduce_mean(tf.square(fn() * Rk))
                return tape.gradient(L, vs)
            ops[name] = graph_ops(fb)
            fns[name] = tf.function(fb)
        t = timed_rounds(fns, reps)
        base = np.median(t["c_fixed_toeplitz"])
        bare[f"K{K}"] = {k: {"graph_ops_fwd_bwd": ops[k], **stats(t[k]), "time_vs_c_fixed_toeplitz": float(np.median(t[k]) / base),
                             "ops_vs_c_fixed_toeplitz": ops[k] / ops["c_fixed_toeplitz"]} for k in fns}
        print("bare", K, {k: "%d ops %.1f ms (x%.2f)" % (v["graph_ops_fwd_bwd"], 1e3 * v["median_s"], v["time_vs_c_fixed_toeplitz"])
                          for k, v in bare[f"K{K}"].items()})
    return {"full_layer_N30720": full, "bare_N30720": bare}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()
    info = C.run_info(NEWEST)
    close = C.blocks(-1)["close"]
    res = {"precision": precision(info, close), "cost": cost(info, close, args.reps)}
    print("wrote", C.dump(res, "q2b_factored.json"))


if __name__ == "__main__":
    main()
