"""Q1: today's per-window adaptation, measured on the saved runs (CPU, read-only).

For every local run (served weights): the per-window logit shift delta = 0.5 * tanh([mean, max] @ W + b)
of the window-relative close (gru_attention.py:44-56, learnable_indicators.py:110-115), over the fold -1
train and test windows of the bundled 30-day file; the applied periods; how much of the +-0.5 tanh range
is used; and a check that this numpy re-implementation equals the package's applied_periods() on the
newest run (Predictor.from_artifacts).

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_adapt_today.py
Writes q1_adapt_today.json.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import neural_trade  # noqa: F401
import numpy as np

import common as C

Q = [0, 0.1, 5, 50, 95, 99.9, 100]


def qs(v):
    return {f"p{q:g}": float(np.percentile(v, q)) for q in Q}


def main():
    B = C.blocks(-1)
    close = B["close"]
    res = {"fold": {k: [int(B[k][0]), int(B[k][-1]), int(len(B[k]))] for k in ("train", "val", "cal", "test")},
           "n_bars": int(len(close)), "runs": {}}
    newest = None
    for r in C.run_dirs():
        info = C.run_info(r)
        newest = (r, info)
        run = {"weights": os.path.relpath(info["weights"], C.RUNS), "scale": info["scale"], "blocks": {}}
        base_p = C.period_of_logit(info["logits"])
        run["base_period"] = dict(zip(C.NAMES, np.round(base_p, 3).tolist()))
        for blk in ("train", "test"):
            ctx = C.today_context(close, B[blk], info["scale"])
            pre = ctx @ info["W"] + info["b"]
            delta = C.META_SCALE * np.tanh(pre)                       # [N, 18]
            app = C.period_of_logit(info["logits"][None, :] + delta)
            ratio = app / base_p[None, :]
            d = {"context": {"mean_offset": qs(ctx[:, 0]), "max_offset": qs(ctx[:, 1])},
                 "abs_preactivation_p99": float(np.percentile(np.abs(pre), 99)),
                 "share_|tanh|>0.9": float(np.mean(np.abs(np.tanh(pre)) > 0.9)),
                 "delta_all": qs(delta.ravel()),
                 "per_logit": {}}
            for j, n in enumerate(C.NAMES):
                d["per_logit"][n] = {"delta_min": float(delta[:, j].min()), "delta_max": float(delta[:, j].max()),
                                     "delta_p5": float(np.percentile(delta[:, j], 5)),
                                     "delta_p95": float(np.percentile(delta[:, j], 95)),
                                     "applied_p5": float(np.percentile(app[:, j], 5)),
                                     "applied_p50": float(np.percentile(app[:, j], 50)),
                                     "applied_p95": float(np.percentile(app[:, j], 95)),
                                     "applied_max": float(app[:, j].max()),
                                     "ratio_min": float(ratio[:, j].min()), "ratio_max": float(ratio[:, j].max())}
            run["blocks"][blk] = d
        res["runs"][info["run"]] = run
    # bound of the shift (tanh in (-1, 1)): ratio of the applied period to the base period
    bound = {}
    for p in (2, 5, 9, 12, 26, 35, 60, 240, 1440):
        lg = C.logit_of_period(p)
        bound[str(p)] = {"shift_-0.5": float(C.period_of_logit(lg - 0.5)), "shift_+0.5": float(C.period_of_logit(lg + 0.5))}
    res["shift_bound_periods"] = bound

    # verification against the package on the newest run (served bundle)
    r, info = newest
    try:
        from neural_trade.serving.predictor import Predictor
        from neural_trade.visualization.indicator_evolution import applied_periods
        pred = Predictor.from_artifacts(r + "artifacts")
        W = C.windows(close, B["test"][:4000]).astype("float32")
        app_pkg = applied_periods(pred, W, block="test").to_numpy(np.float64)
        ctx = C.today_context(close, B["test"][:4000], info["scale"])
        app_np = C.period_of_logit(info["logits"][None, :] + C.meta_shift(ctx, info["W"], info["b"]))
        res["check_vs_package_applied_periods"] = {
            "run": info["run"], "n_windows": 4000,
            "max_rel_diff": float(np.max(np.abs(app_pkg - app_np) / app_np)),
            "columns_match": list(applied_periods(pred, W[:2]).columns) == C.NAMES}
    except Exception as e:  # report, do not hide
        res["check_vs_package_applied_periods"] = {"error": repr(e)}
    path = C.dump(res, "q1_adapt_today.json")
    # short console summary
    print("check:", res["check_vs_package_applied_periods"])
    for run, v in res["runs"].items():
        t = v["blocks"]["test"]
        print(run[:24], "delta all p0.1..p99.9 %.3f..%.3f" % (t["delta_all"]["p0.1"], t["delta_all"]["p99.9"]),
              "share|tanh|>0.9 %.3f" % t["share_|tanh|>0.9"],
              "macd_1_slow base %.1f applied p5-p95 %.1f-%.1f max %.1f" % (
                  v["base_period"]["macd_1_slow"], t["per_logit"]["macd_1_slow"]["applied_p5"],
                  t["per_logit"]["macd_1_slow"]["applied_p95"], t["per_logit"]["macd_1_slow"]["applied_max"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
