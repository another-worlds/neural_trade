"""M6: whole-SPEC GPU budgets for A/B-1 and A/B-2 under the recommended designs. Every per-run time is C's
ESTIMATE (7-day-block run: arm A 2.6-6.3 min, i.e. 21-60 epochs x 5.6 s + 40 s; A/B-1's series arm taken equal
to A (C: -3 to +3 ms on a 98 ms step); A/B-2's per-bar arm 1.3-2.5 min). Contingencies: re-time share = the
measured share of v1 DEV fold -2 runs slower than 1.10x the fold's median epoch time (C/q2_v1_per_run.csv);
cap extensions = 5% of judged runs re-run at 2x length (Estimate from q3_epoch_cap: 0.5-8% per run).
Run: ...python q5_budget.py. Writes q5_budget.json."""
import json
import numpy as np
import pandas as pd

t = pd.read_csv("D:/nt_research/wfp/C/q2_v1_per_run.csv"); t = t[t.period == "P1"]
retime = float((t.median_epoch_s > 1.10 * t.median_epoch_s.median()).mean())
EXT = 0.05
A = (2.6, 6.3); B2 = (1.3, 2.5)


def total(items):
    lo = sum(n * a[0] for n, a in items) / 60; hi = sum(n * a[1] for n, a in items) / 60
    return round(lo, 2), round(hi, 2)


out = {"retime_share_v1_dev": retime, "extension_share": EXT}
# A/B-1: F = 10 per look (Pocock two-look), dev 3 folds x both arms, 1 lambda calibration
F = 10
look1 = 2 * F; look2 = 2 * F
cont = lambda n: n * (retime + 2 * EXT)       # re-time runs + extension runs (2x length)
ab1 = {
    "calibration": (1, A), "dev (3 folds x 2 arms)": (6, A),
    "judged look 1 (2 x 10)": (look1, A), "contingency look 1": (cont(look1), A),
    "judged look 2 (2 x 10)": (look2, A), "contingency look 2": (cont(look2), A),
}
out["AB1_items_runs"] = {k: round(v[0], 2) for k, v in ab1.items()}
out["AB1_look1_only"] = total([ab1[k] for k in ("calibration", "dev (3 folds x 2 arms)", "judged look 1 (2 x 10)", "contingency look 1")])
out["AB1_worst_two_looks"] = total(list(ab1.values()))
for Ef, lab in ((1.14, "E[folds]=1.14F (edge .03)"), (1.46, "E[folds]=1.46F (edge .02)")):
    out[f"AB1_expected_{lab}"] = total([ab1["calibration"], ab1["dev (3 folds x 2 arms)"], (2 * F * Ef * (1 + retime + 2 * EXT), A)])
# A/B-2: F = 12 per look, variant choice 3 x 2 dev folds (B), dev A 3 folds, 1 calibration, A' 0.6 h after ADOPT
F2 = 12
ab2 = [("calibration", 1, A), ("variant choice B (3 x 2 dev folds)", 6, B2), ("dev A (3 folds)", 3, A),
       ("judged look 1 A", F2 * (1 + retime + 2 * EXT), A), ("judged look 1 B", F2 * (1 + retime + 2 * EXT), B2),
       ("judged look 2 A", F2 * (1 + retime + 2 * EXT), A), ("judged look 2 B", F2 * (1 + retime + 2 * EXT), B2)]
out["AB2_items_runs"] = {k: round(n, 2) for k, n, _ in ab2}
lo1, hi1 = total([(n, a) for k, n, a in ab2 if "look 2" not in k])
lo2, hi2 = total([(n, a) for k, n, a in ab2])
out["AB2_look1_only"] = (lo1, hi1); out["AB2_look1_only_plus_Aprime"] = (round(lo1 + 0.6, 2), round(hi1 + 0.6, 2))
out["AB2_worst_two_looks"] = (lo2, hi2); out["AB2_worst_two_looks_plus_Aprime"] = (round(lo2 + 0.6, 2), round(hi2 + 0.6, 2))
# the draft's counts, for comparison (C: 2F + 3 dev + 1 calibration, F = 21, no second look)
out["draft_AB1_F21_one_look"] = total([(2 * 21 + 4, A)])
out["draft_AB1_F21_with_rerun_2F_new"] = total([(2 * 21 + 4 + 2 * 42, A)])
print(json.dumps(out, indent=1))
open("q5_budget.json", "w").write(json.dumps(out, indent=1))
