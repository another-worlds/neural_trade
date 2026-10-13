"""T04 Part A (SPEC.md): retrospective test 2017-09..2018-12. usage: python run_t04.py -> resultsA.json, tablesA.txt."""
import json, os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t04_lib as L  # noqa: E402
T = L.T

try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    s = T.load_bars(T.TRAIN_CSV, L.A_LO, L.A_HI); print("bars", len(s), s.index.min(), s.index.max(), el(), flush=True)
    gaps = s.index.to_series().diff().dropna(); nm = gaps[gaps != pd.Timedelta(minutes=1)]
    print("non-1-minute steps", len(nm), "largest", nm.max() if len(nm) else None, flush=True)
    B = L.build(s); dr = L.drop_reasons(B); print("drops", dr, el(), flush=True)
    rows = []
    for m in pd.PeriodIndex(L.A_MONTHS, freq="M"):
        tr = T.train_idx(B, "A", m.start_time); mod = L.fit(B, tr)
        r = L.score_month(mod, B, m); ta = B["D"].ta[tr]
        r["n_tr"] = int(len(tr)); r["train_days"] = round(float((m.start_time - ta.min()).total_seconds() / 86400), 1); rows.append(r)
        print(m, "ntr", len(tr), "days", r["train_days"], "n_te", r["n_te"], {k: round(v, 4) for k, v in r["skill_ar"].items()}, el(), flush=True)
    v = L.verdict(rows)
    comps = {"S3C-S3": v, "S3-ar": L.stats_for(rows, "S3", "ar", "S3-ar"), "S3-harlvl": L.stats_for(rows, "S3", "harlvl", "S3-harlvl"),
             "C-ar": L.stats_for(rows, "C", "ar", "C-ar"), "S3C-ar": L.stats_for(rows, "S3C", "ar", "S3C-ar"), "C-S3": L.stats_for(rows, "C", "S3", "C-S3")}
    lines = [f"== T04 Part A: {len(rows)} months 2017-09..2018-12; bars {len(s)} ({s.index.min()} .. {s.index.max()}); non-1-minute steps {len(nm)}; drops {dr} =="]
    lines.append("-- mean over months: MSE / skill vs ar [95% t-CI] / months with positive skill")
    for k in L.MODELS:
        sk = [r["skill_ar"][k] for r in rows]
        if k in ("rw", "ar"):
            lines.append(f"   {k:7s} mse {np.mean([r['mse'][k] for r in rows]):.4f}  skill_ar {np.mean(sk):+.4f}"); continue
        mu, lo, hi, _ = T.ci(sk)
        lines.append(f"   {k:7s} mse {np.mean([r['mse'][k] for r in rows]):.4f}  skill_ar {mu:+.4f} [{lo:+.4f},{hi:+.4f}]  positive {int((np.array(sk) > 0).sum())}/{len(sk)}")
    lines.append("-- paired monthly differences of skill vs ar (a - b): mean [95% t-CI], months positive; and 1 - mse_a/mse_b")
    for c in comps.values():
        lines.append(f"   {c['label']:10s} diff {c['mean_diff_skill_ar']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] pos {c['months_positive']}/{c['n_months']}   "
                     f"mse-ratio {c['mse_ratio_skill'][0]:+.4f} [{c['mse_ratio_skill'][1]:+.4f},{c['mse_ratio_skill'][2]:+.4f}] pos {c['months_positive_ratio']}/{c['n_months']}")
    lines.append(f"== CRITERION A (S3C - S3, skill vs ar): {'PASS' if v['pass'] else 'FAIL'}  mean>=0.02 {v['c1_mean_ge_0.02']}, lower>0 {v['c2_lo_gt_0']}, "
                 f"positive>= {v['need_positive']}/16 {v['c3_months']} ({v['months_positive']}/16) ==")
    lines.append("-- per month: month n_tr train_days n_te | skill_ar " + "/".join(L.MODELS))
    for r in rows:
        lines.append(f"{r['month']} {r['n_tr']:6d} {r['train_days']:6.1f}d {r['n_te']:5d} | " + " ".join(f"{r['skill_ar'][k]:+.4f}" for k in L.MODELS))
    json.dump({"months": rows, "comparisons": comps, "criterion_A": v, "drops": dr, "bars": int(len(s)), "non_minute_steps": int(len(nm)),
               "runtime_s": time.time() - t0}, open(f"{HERE}/resultsA.json", "w"), indent=1)
    open(f"{HERE}/tablesA.txt", "w").write("\n".join(lines)); print("\n".join(lines)); print("done", el())


if __name__ == "__main__":
    main()
