"""T13: merge the three selection runs into selection.json (the ONE pair per part: the highest mean monthly gain on 2019-01..2024-06) and selection_table.txt.
Run after run_select.py A, B and C; then COMMIT selection.json, the sel_part_*.json and selection_table.txt BEFORE run_judge.py."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = ("per part, the (bar minutes x lookback bars) pair among 1x60, 1x240, 15x60, 15x240 with the highest mean over the 66 months 2019-01..2024-06 of the monthly gain "
        "(A: 1 - Brier(A1) / Brier(H3), the barrier hit X = 50 bps T = 60 min; B: Spearman(B1) - Spearman(L1), next-60-minute RV; C: skill_ar(C1) - skill_ar(V3), target A)")
LABEL = {"A": "relative Brier gain A1 vs H3", "B": "Spearman gain B1 - L1", "C": "skill_ar gain C1 - V3"}
chosen, lines, sha = {}, ["T13 selection, 2019-01..2024-06 (cut file, monthly walk-forward, 66 months): mean monthly gain per pair (bar minutes x lookback bars)", ""], {}
gains = {}
for part in "ABC":
    path = os.path.join(HERE, f"sel_part_{part}.json"); d = json.load(open(path)); sha[f"sel_part_{part}.json"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    pr = d["pairs"]; best = max(pr, key=lambda k: pr[k]["mean_gain"]); chosen[part] = best; gains[part] = {k: v["mean_gain"] for k, v in pr.items()}
    lines.append(f"Part {part}: {LABEL[part]}")
    for k, v in pr.items():
        pos = sum(1 for g in v["monthly_gain"].values() if g > 0)
        extra = {"A": f" Brier H3 {v.get('brier_H3', float('nan')):.5f} A1 {v.get('brier_A1', float('nan')):.5f}", "B": f" Spearman L1 {v.get('spearman_L1', 0):.4f} B1 {v.get('spearman_B1', 0):.4f}",
                 "C": f" skill V3 {v.get('skill_V3', 0):+.4f} C1 {v.get('skill_C1', 0):+.4f}"}[part]
        lines.append(f"   {k:7s} mean monthly gain {v['mean_gain']:+.5f}  positive months {pos}/{v['n_months']}{extra}{'   <-- CHOSEN' if k == best else ''}")
    lines.append("")
sel = {"spec": "T13 SPEC.md (1869df6)", "rule": RULE, "period": "2019-01..2024-06", "n_months": 66, "chosen": chosen, "mean_gain": gains, "input_sha256": sha}
json.dump(sel, open(os.path.join(HERE, "selection.json"), "w"), indent=1)
open(os.path.join(HERE, "selection_table.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines)); print("chosen", chosen)
