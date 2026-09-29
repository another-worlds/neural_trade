"""Q3 addendum: chunk size C (16 / 32 / 64) of the recommended kernel form (V1: segsum, G = 32, log-decay
input, mulsum contraction): precision at 43,008 bars (the Q1 tolerances) and per-step cost (fwd+bwd CPU,
op counts, largest fp32 intermediate, static bytes written by the graph) for K = 31 / 87, per-bar alpha,
over the 7-day pass (11,504 bars) and 30,720 bars. Memory of the [K, n, C, C] tensors scales with T x C.

Command: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q3_chunk_size.py
Output:  q3_chunk_size.json
"""
import time

from common import dump, graph_census, interleaved_timing, np, tf
from kernel import burn_in
import q1_kernel_checks as q1
from q3_span import bare_pass

tf.config.threading.set_inter_op_parallelism_threads(4)
T7 = 10080 + burn_in(240) + 60 - 1


def precision(C):
    res = {}
    v = dict(q1.VARIANTS["V1_segsum_hier_mulsum"])
    for perbar in (False, True):
        dx32, lam32, _ = q1.series(43008, perbar)
        ref = q1.ref_pipeline(dx32, lam32)
        out = q1.run_pipeline(dx32, lam32, v, C)
        worst = 0.0
        ok = True
        for h, r in zip(out, ref):
            e, mx, rel = q1.stats(h, r)
            worst = max(worst, float(rel.max()))
            ok &= bool(np.all((rel <= q1.TOL_REL) | (e <= q1.TOL_ABS)))
        res["per_bar" if perbar else "constant"] = {"worst_rel_err_all_channels": worst, "ALL_PASS": ok}
    return res


def main():
    out = {"T7": T7, "precision_T43008": {}, "cost": {}}
    for C in (16, 32, 64):
        out["precision_T43008"][f"C{C}"] = precision(C)
        print("precision C", C, out["precision_T43008"][f"C{C}"])
    for T in (T7, 30720):
        for K in (31, 87):
            fns = {f"C{C}": bare_pass(K, T, True, C, "mulsum") for C in (16, 32, 64)}
            res = {}
            for k, (fwd, fb) in fns.items():
                cb = graph_census(fb)
                res[k] = {"fwdbwd_total_ops": cb["total_ops"], "fwdbwd_compute_ops": cb["compute_ops"],
                          "largest_intermediate_MB": cb["largest_output_bytes"] / 2 ** 20,
                          "static_output_GB_fwdbwd": cb["static_output_bytes"] / 2 ** 30}
            t0 = time.perf_counter()
            tim = interleaved_timing({k: tf.function(fb) for k, (fwd, fb) in fns.items()}, reps=7)
            for k in res:
                res[k]["cpu_fwdbwd"] = tim[k]
            out["cost"][f"T{T}_K{K}"] = res
            print(f"T={T} K={K} ({time.perf_counter() - t0:.0f}s):",
                  {k: (v["fwdbwd_compute_ops"], round(v["largest_intermediate_MB"]), round(v["static_output_GB_fwdbwd"], 2),
                       round(v["cpu_fwdbwd"]["median_s"] * 1e3)) for k, v in res.items()})
    dump("q3_chunk_size.json", out)


if __name__ == "__main__":
    main()
