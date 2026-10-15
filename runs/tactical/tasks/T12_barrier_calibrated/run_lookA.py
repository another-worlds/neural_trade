"""T12 Look A (SPEC.md): cut file 2017-09..2018-12, monthly walk-forward in UTC; 2017-09 only calibrates, 2017-10..2018-12 are scored.
usage: python run_lookA.py -> resultsA.json, tablesA.txt, rowsA.npz. Reads no bar after 2018-12-31 23:59."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import t12_lib as T12

L = T12.L


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    st = subprocess.run(["git", "status", "--porcelain", "--", HERE], cwd=HERE, capture_output=True, text=True).stdout.strip()
    bars = T12.load_lookA(); print("bars", len(bars), bars.index.min(), bars.index.max(), "HEAD", head, "dirty:", bool(st), el(), flush=True)
    assert bars.index.max() <= T12.A_END
    P = L.prep(bars, "cut"); months = list(pd.period_range(T12.A_FIRST, T12.A_LAST, freq="M")); R, info, rows = {}, {}, {}
    for T in T12.TS:
        B = L.build(P, T); recs, inf = T12.walk(B, T, months); R[T] = T12.pool(recs); info[str(T)] = inf
        print(f"T={T}: anchors {len(B['a'])} valid {int(B['ok'].sum())}, scored rows {len(R[T]['ta'])}", el(), flush=True)
        rows[f"{T}|month"] = R[T]["month"]; rows[f"{T}|ta"] = R[T]["ta"]; rows[f"{T}|day"] = R[T]["day"]
        for X in T12.XS:
            rows[f"{T}|y|{X}"] = R[T]["y"][X]
            for m in T12.MODELS:
                rows[f"{T}|p|{m}|{X}"] = R[T]["p"][m][X].astype(np.float32)
    res = T12.assess(R); res["head"] = head; res["dirty_at_run"] = bool(st); res["info"] = info; res["runtime_s"] = time.time() - t0
    res["scored_months"] = sorted(set(R[60]["month"].tolist()))
    json.dump(res, open(os.path.join(HERE, "resultsA.json"), "w"), indent=1); np.savez_compressed(os.path.join(HERE, "rowsA.npz"), **rows)
    sm = res["scored_months"]
    lines = [f"T12 Look A: cut file 2017-09..2018-12, H3c = H3 + isotonic on the previous month; scored months {sm[0]}..{sm[-1]} ({len(sm)}); run at HEAD {head}", ""]
    lines.append("Training days and rows per month (H3 refit at each month start; n_train for T = 15 / 60 / 240)")
    for k in range(len(info["60"])):
        a = info["60"][k]
        if a.get("skipped"):
            lines.append(f" {a['month']}: SKIPPED"); continue
        n = " / ".join(str(info[str(T)][k].get("n_train")) for T in T12.TS)
        lines.append(f" {a['month']}: train_days={a['train_days']:.1f} (from {a['train_first']}) n_train={n} scored_anchors(T=60)={a['n']} n_cal={a['n_cal']}")
    lines.append("")
    for T in T12.TS:
        for X in T12.XS:
            c = res["cells"][f"{X}|{T}"]; mark = "   <-- PRIMARY" if (X, T) == T12.PRIMARY else ""
            lines.append(f" X={X} bps T={T} min n={c['B1']['n']} base_rate={c['B1']['base_rate']:.4f}{mark}")
            for m in T12.MODELS:
                d = c[m]
                ci = (f" CI[{100 * d['skill_ci'][0]:+.2f},{100 * d['skill_ci'][1]:+.2f}] p={d['skill_p']:.4f} holm_p={res['holm'][m][f'{X}|{T}']['p_holm']:.4f}") if "skill_ci" in d else ""
                rel = (f" vsH3={100 * d['rel_brier_vs_H3']:+.2f}% CI[{100 * d['rel_brier_vs_H3_ci'][0]:+.2f},{100 * d['rel_brier_vs_H3_ci'][1]:+.2f}]") if "rel_brier_vs_H3" in d else ""
                lines.append(f"    {m:<3} brier={d['brier']:.5f} skill_vs_B1={100 * d['skill_vs_B1']:+.2f}%{ci}{rel} ece={100 * d['ece']:.2f}pp "
                             f"CI[{100 * d['ece_ci'][0]:.2f},{100 * d['ece_ci'][1]:.2f}] auc={d['auc']:.4f} logloss={d['logloss']:.4f}")
    v = res["verdict"]
    lines += ["", "Criteria, primary cell X=50 bps T=60 min:",
              f" 1 skill {100 * v['skill']:+.2f}% CI [{100 * v['skill_ci'][0]:+.2f},{100 * v['skill_ci'][1]:+.2f}] (>=2% and CI>0): {'PASS' if v['c1_skill'] else 'FAIL'}",
              f" 2 ECE {100 * v['ece']:.2f}pp CI [{100 * v['ece_ci'][0]:.2f},{100 * v['ece_ci'][1]:.2f}] (<1pp): {'PASS' if v['c2_ece'] else 'FAIL'}",
              f" 3 Brier H3c {v['brier_h3c']:.5f} vs H3 {v['brier_h3']:.5f} ({100 * v['rel_brier_vs_H3']:+.2f}%, not worse than +1%): {'PASS' if v['c3_not_worse'] else 'FAIL'}",
              f"LOOK A {'PASS' if v['pass'] else 'FAIL'}"]
    open(os.path.join(HERE, "tablesA.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("total", el(), flush=True)


if __name__ == "__main__":
    main()
