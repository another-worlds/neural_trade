"""T13 selection (SPEC.md "Periods"): for each part, the ONE pair (bar size, lookback) among 1x60, 1x240, 15x60, 15x240 with the highest mean monthly gain on
2019-01..2024-06 of the cut file (A: relative Brier gain of A1 over H3; B: Spearman gain of B1 over L1; C: skill gain of C1 over V3). Reads no bar after
2024-07-01 03:59. usage: python run_select.py A|B|C -> sel_part_<X>.json (then make_selection.py, then COMMIT selection.json before run_judge.py)"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import t13_lib as L

try:
    from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def part_A():
    s = L.T01.load_bars(L.guard.CUT_CSV, pd.Timestamp("2017-08-17"), L.SEL_HI_BARS); s = s[s.index <= L.SEL_HI_BARS]
    assert s.index.max() <= L.SEL_HI_BARS; log("A bars", len(s), s.index.min(), s.index.max())
    P = L.L10.prep(s, "cut"); B = L.L10.build(P, L.T_MIN); out = {}
    for pr in L.PAIRS:
        t0 = time.time(); R, info = L.a_walk(B, P, L.T_MIN, L.SEL_MONTHS, pr)
        S = L.a_summary(R, seed=0) if False else None
        mg = L.a_monthly_gain(R); out[L.pair_key(pr)] = {"mean_gain": float(np.mean(list(mg.values()))), "n_months": len(mg), "monthly_gain": mg,
                                                        "n_rows": int(len(R["y"])), "brier_H3": float(np.mean((R["p"]["H3"] - R["y"]) ** 2)), "brier_A1": float(np.mean((R["p"]["A1"] - R["y"]) ** 2))}
        log("A", L.pair_key(pr), f"mean monthly gain {out[L.pair_key(pr)]['mean_gain']:+.5f}", f"{time.time() - t0:.0f}s")
    return out


def part_B():
    os.makedirs(L.SCRATCH, exist_ok=True); p = os.path.join(L.SCRATCH, "frame_sel_B.npz")
    if os.path.exists(p):
        z = np.load(p); F = {k: z[k] for k in z.files}
    else:
        s = L.T01.load_bars(L.T01.TRAIN_CSV, L.O5.LOAD_FROM, L.O5.LOAD_HI); assert s.index.max() <= L.O5.LOAD_HI; log("B bars", len(s), s.index.min(), s.index.max())
        t0 = time.time(); F = L.build_B_frame(s, L.O5.FIRST5); del s; np.savez(p, **F); log("B frame", len(F["ta"]), f"{time.time() - t0:.0f}s")
    assert pd.Timestamp(F["ta"].max()) <= L.O5.LOAD_HI
    out = {}
    for pr in L.PAIRS:
        t0 = time.time(); rows = L.b_walk(F, L.SEL_MONTHS, pr); g = L.b_gain(rows)
        out[L.pair_key(pr)] = {"mean_gain": float(np.mean(list(g.values()))), "n_months": len(g), "monthly_gain": g,
                               "spearman_L1": float(np.mean([r["spearman"]["L1"] for r in rows])), "spearman_B1": float(np.mean([r["spearman"]["B1"] for r in rows]))}
        log("B", L.pair_key(pr), f"mean monthly gain {out[L.pair_key(pr)]['mean_gain']:+.5f}", f"{time.time() - t0:.0f}s")
    return out


def part_C():
    s = L.T01.load_bars(L.T01.TRAIN_CSV, L.L11.SEL_LO, L.L11.SEL_HI); assert s.index.max() < pd.Timestamp("2024-07-02 00:00") and s.index.min() >= pd.Timestamp("2017-12-01")
    log("C bars", len(s), s.index.min(), s.index.max()); t0 = time.time(); B = L.build_C(s, 1441, "bitstamp"); log("C frame", f"{time.time() - t0:.0f}s", B["drops"]); out = {}
    for pr in L.PAIRS:
        t0 = time.time(); rows = L.c_walk(B, L.SEL_MONTHS, pr); g = L.c_gain(rows)
        out[L.pair_key(pr)] = {"mean_gain": float(np.mean(list(g.values()))), "n_months": len(g), "monthly_gain": g,
                               "skill_V3": float(np.mean([r["skill_ar"]["V3"] for r in rows])), "skill_C1": float(np.mean([r["skill_ar"]["C1"] for r in rows]))}
        log("C", L.pair_key(pr), f"mean monthly gain {out[L.pair_key(pr)]['mean_gain']:+.5f}", f"{time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    part = sys.argv[1]; t0 = time.time(); res = {"A": part_A, "B": part_B, "C": part_C}[part]()
    json.dump({"part": part, "pairs": res, "runtime_s": time.time() - t0}, open(os.path.join(HERE, f"sel_part_{part}.json"), "w"), indent=1)
    log("done", part, f"{time.time() - t0:.0f}s")
