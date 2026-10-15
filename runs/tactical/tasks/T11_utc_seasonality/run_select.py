"""T11 stage 1 (SPEC.md "Periods", selection): monthly walk-forward 2019-01..2024-06 on the cut file, V0..V3, target A; picks one of V1, V2, V3 by the
mean monthly skill difference over V0 and writes selection.json. Run it, COMMIT selection.json, and only then run run_report.py.
usage: python run_select.py  -> selection.json, selection_results.json, selection_tables.txt (log to stdout)."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t11_lib as L  # noqa: E402
T = L.T

try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

RULE = ("Among V1, V2, V3 the variant with the highest mean over the 66 months 2019-01..2024-06 of (skill_ar[variant] - skill_ar[V0]) on target A "
        "(skill_ar = 1 - mse / mse_ar, the monthly ridge refit on the 365 days before the month). Target 'R' (|r_15|) is printed, not used.")


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    s = T.load_bars(T.TRAIN_CSV, L.SEL_LO, L.SEL_HI); print("bars", len(s), s.index.min(), s.index.max(), el(), flush=True)
    assert s.index.max() < pd.Timestamp("2025-01-01") and s.index.min() >= pd.Timestamp("2017-12-01")      # nothing of the report-only or forbidden spans
    B = L.build(s, 1441, "bitstamp"); print("drops", B["drops"], el(), flush=True)
    rows = L.walk_forward(B, L.SEL_MONTHS, "selection", log=lambda m, r: print(m, "ntr", r["n_tr"], "n_te", r["n_te"],
                          {k: round(v, 4) for k, v in r["skill_ar"].items() if k.startswith("V")}, el(), flush=True))
    comps = {v: L.paired(rows, v) for v in L.VARIANTS[1:]}
    score = {v: comps[v]["skill_ar_diff"]["mean"] for v in comps}; chosen = max(score, key=score.get)
    lines = [f"== T11 selection: {len(rows)} months 2019-01..2024-06 (cut file, UTC conversion by timefix.py); bars {len(s)}; drops {B['drops']} =="]
    lines.append("-- mean over months: skill vs ar [95% t-CI] / months with positive skill, target A")
    for k in ["rw", "ar"] + L.VARIANTS:
        sk = [r["skill_ar"][k] for r in rows]
        if k in ("rw", "ar"):
            lines.append(f"   {k:3s} mse {np.mean([r['mse'][k] for r in rows]):.5f}  skill_ar {np.mean(sk):+.4f}"); continue
        mu, lo, hi, _ = T.ci(sk)
        lines.append(f"   {k:3s} mse {np.mean([r['mse'][k] for r in rows]):.5f}  skill_ar {mu:+.4f} [{lo:+.4f},{hi:+.4f}]  positive {int((np.array(sk) > 0).sum())}/{len(sk)}")
    lines.append("-- paired monthly differences, variant minus V0 [95% t-CI] / months positive")
    for v, c in comps.items():
        d, q, sp, ql = c["skill_ar_diff"], c["mse_ratio_skill"], c["spearman_diff"], c["qlike_diff"]
        lines.append(f"   {v}-V0 skill_ar diff {d['mean']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] pos {d['positive']}/{len(rows)}   1-mse/mse_V0 {q['mean']:+.4f} "
                     f"[{q['ci'][0]:+.4f},{q['ci'][1]:+.4f}]   | R: spearman diff {sp['mean']:+.4f} [{sp['ci'][0]:+.4f},{sp['ci'][1]:+.4f}] qlike diff {ql['mean']:+.5f}")
    lines.append(f"== CHOSEN: {chosen} (mean skill_ar diff over V0: " + ", ".join(f"{v} {x:+.4f}" for v, x in score.items()) + ") ==")
    lines.append("-- per month: month n_tr n_te | skill_ar " + "/".join(L.VARIANTS))
    for r in rows:
        lines.append(f"{r['month']} {r['n_tr']:6d} {r['n_te']:5d} | " + " ".join(f"{r['skill_ar'][k]:+.4f}" for k in L.VARIANTS))
    sel = {"spec": "T11 SPEC.md (0a55dac)", "rule": RULE, "months": [rows[0]["month"], rows[-1]["month"]], "n_months": len(rows), "chosen": chosen,
           "mean_skill_ar_diff_vs_V0": score, "comparisons": comps, "drops": B["drops"], "code_sha256": {"t11_lib.py": L.sha_file(os.path.join(HERE, "t11_lib.py")),
                                                                                                         "timefix.py": L.sha_file(os.path.join(L.TAC, "common", "timefix.py"))}}
    json.dump(sel, open(f"{HERE}/selection.json", "w"), indent=1)
    json.dump({"rows": rows, "runtime_s": time.time() - t0}, open(f"{HERE}/selection_results.json", "w"), indent=1)
    open(f"{HERE}/selection_tables.txt", "w").write("\n".join(lines)); print("\n".join(lines[:16])); print("done", el())


if __name__ == "__main__":
    main()
