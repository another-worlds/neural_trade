"""T10 selection run (SPEC.md "Selection"): 2019-01..2024-06 on the cut file, monthly walk-forward, all models and cells.
Picks the ONE of H1, H2, H3 with the lowest mean monthly Brier in the primary cell (X = 50, T = 60) and writes selection.json.
Rows from 2024-07 on are removed from the data before anything is fitted (the looks are not visible to this script).
usage: python run_select.py   (then COMMIT selection.json before run_looks.py)"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import t10_lib as L

try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    s = L.guard.load_ohlcv("cut"); s = s[s.index < pd.Timestamp("2024-07-01")]                  # the selection never sees a later bar
    print("cut bars", len(s), s.index.min(), s.index.max(), el(), flush=True)
    P = L.prep(s, "cut"); months = list(pd.period_range(L.SEL_FIRST, L.SEL_LAST, freq="M"))
    res = {}; table = {}
    for T in L.TS:
        B = L.build(P, T)
        print(f"T={T}: {len(B['a'])} anchors, {int(B['ok'].sum())} valid", el(), flush=True)
        R = L.walk(B, T, months)
        idx = L.boot_index(R["day"], 500, seed=0)                                               # CIs are not used by the selection
        S = L.summarize(R, idx)
        for X in L.XS:
            res[f"{X}|{T}"] = {m: {k: v for k, v in S[X][m].items() if k != "monthly"} for m in L.MODELS}
            res[f"{X}|{T}"]["n_months"] = len(set(R["month"]))
        res[f"info|{T}"] = R["info"]
        print(f"T={T} done", el(), flush=True)
    prim = res[f"{L.PRIMARY[0]}|{L.PRIMARY[1]}"]
    mmb = {m: prim[m]["mean_monthly_brier"] for m in L.CANDIDATES}
    chosen = min(mmb, key=mmb.get)
    lines = ["T10 selection, 2019-01..2024-06 (cut file, monthly walk-forward): mean monthly Brier (pooled Brier; skill over B1 in %)", ""]
    for T in L.TS:
        for X in L.XS:
            r = res[f"{X}|{T}"]
            lines.append(f"X={X} bps T={T} min  n={r['B1']['n']} base_rate={r['B1']['base_rate']:.4f} months={r['n_months']}" + ("   <-- PRIMARY" if (X, T) == L.PRIMARY else ""))
            for m in L.MODELS:
                lines.append(f"   {m}  mean_monthly_brier={r[m]['mean_monthly_brier']:.5f}  pooled={r[m]['brier']:.5f}  skill_vs_B1={100 * r[m]['skill_vs_B1']:+.2f}%  "
                             f"ece={100 * r[m]['ece']:.2f}pp  auc={r[m]['auc']:.4f}  logloss={r[m]['logloss']:.4f}")
    lines += ["", f"chosen = {chosen} (lowest mean monthly Brier among H1, H2, H3 in the primary cell): " + ", ".join(f"{m} {v:.5f}" for m, v in mmb.items())]
    open(os.path.join(HERE, "selection_table.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(HERE, "selection_results.json"), "w"), indent=1)
    sel = {"chosen": chosen, "primary": {"X_bps": L.PRIMARY[0], "T_min": L.PRIMARY[1]}, "mean_monthly_brier_primary": mmb,
           "period": "2019-01..2024-06", "rule": "lowest mean monthly Brier among H1, H2, H3 in the primary cell",
           "selection_results_sha256": hashlib.sha256(open(os.path.join(HERE, "selection_results.json"), "rb").read()).hexdigest()}
    json.dump(sel, open(os.path.join(HERE, "selection.json"), "w"), indent=1)
    print("\n".join(lines)); print("chosen", chosen, "total", el(), flush=True)


if __name__ == "__main__":
    main()
