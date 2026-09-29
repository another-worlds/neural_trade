"""Q3 addendum: the A2 layer with the recommended chunk C = 16 (mulsum) against today's layer and C = 64:
op census, fp32 bytes, CPU fwd+bwd (interleaved medians); plus the Q1 gradient check (vs float64 finite
differences) at C = 16 and C = 32.
Output: q3_layer_c16.json"""
import time

from common import dump, graph_census, interleaved_timing, np, tf
from q3_span import T7, a2_layer, today_layer
import q1_kernel_checks as q1
from kernel import linrec, logit_period

tf.config.threading.set_inter_op_parallelism_threads(4)


def grad_check(C):
    T = 30720
    per = [2, 5, 14, 30, 60, 240, 1440, 10080, 40000]
    base = logit_period(per)
    rng = np.random.default_rng(0)
    w = rng.normal(size=(len(per), T))
    w[:, : T // 2] = 0.0
    res = {}
    for perbar in (False, True):
        dx32, _, shift32 = q1.series(T, perbar, base)
        dxx = np.asarray(dx32, np.float64)[None, :]

        def L64(beta):
            lam = beta[:, None] + np.asarray(shift32, np.float64)[None, :]
            la, dec, al = q1.ref_decay(lam)
            d = q1.ref_linrec(la, -dec * dxx)
            var = q1.ref_linrec(la, al * d * d)
            return (w * (d + var)).sum(1)
        g_fd = (L64(base + 1e-5) - L64(base - 1e-5)) / 2e-5
        beta = tf.Variable(base.astype(np.float32))
        with tf.GradientTape() as tape:
            la, dec, al = q1.tf_decay(beta[:, None] + tf.constant(shift32)[None, :], "logdecay")
            d, _ = linrec(la, -dec * tf.constant(dx32)[None, :], C=C)
            var, _ = linrec(la, al * tf.square(d), C=C)
            Lv = tf.reduce_sum(tf.constant(w.astype(np.float32)) * (d + var))
        g = tape.gradient(Lv, beta).numpy()
        rel = np.abs(g - g_fd) / np.abs(g_fd)
        res["per_bar" if perbar else "constant"] = {"max_rel_err_vs_fd64": float(rel.max()), "PASS_1e-3": bool(rel.max() <= 1e-3)}
    return res


def main():
    out = {"grad": {f"C{C}": grad_check(C) for C in (16, 32)}}
    print(out["grad"])
    fns = {"today_LearnableIndicators_B256": today_layer()}
    for N in (T7, 30720):
        for C in (16, 64):
            fwd, fb, _ = a2_layer(N, "mulsum", C=C)
            fns[f"A2_N{N}_C{C}_mulsum_D6"] = (fwd, fb)
    res = {}
    for k, (fwd, fb) in fns.items():
        cb = graph_census(fb)
        res[k] = {"fwdbwd_total_ops": cb["total_ops"], "fwdbwd_compute_ops": cb["compute_ops"],
                  "largest_intermediate_MB": cb["largest_output_bytes"] / 2 ** 20,
                  "static_output_GB_fwdbwd": cb["static_output_bytes"] / 2 ** 30,
                  "raise_on_gpu": cb["raise_on_gpu"], "host_round_trip_on_gpu": cb["host_round_trip_on_gpu"]}
    t0 = time.perf_counter()
    tim = interleaved_timing({k: tf.function(fb) for k, (fwd, fb) in fns.items()}, reps=9)
    for k in res:
        res[k]["cpu_fwdbwd"] = tim[k]
        print(k, res[k]["fwdbwd_compute_ops"], round(res[k]["largest_intermediate_MB"], 1),
              round(res[k]["static_output_GB_fwdbwd"], 2), round(tim[k]["median_s"] * 1e3, 1),
              f"[{tim[k]['min_s'] * 1e3:.0f}-{tim[k]['max_s'] * 1e3:.0f}]")
    print(f"timing {time.perf_counter() - t0:.0f}s")
    out["layer"] = res
    dump("q3_layer_c16.json", out)


if __name__ == "__main__":
    main()
