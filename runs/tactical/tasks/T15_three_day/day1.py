"""T15 Day 1: harmonic sweep views, harness evaluation, sub-sample checks and the stability rule (NOTE_day1.md). Resumable.
Run: D:/nt/nt_mm_env/python.exe day1.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import t15_lib as T  # noqa: E402
import build_m7 as B7  # noqa: E402
import evaluate as E  # noqa: E402

DAY = "day1"
OUT = os.path.join(HERE, DAY)
KS = (4, 6, 8, 12)
NAMES = {k: f"T15_h{k}" for k in KS}
STORED = ["M7_gbm", "M7c_gbm_resid", "M9_prof_str", "M9h6_prof_str", "M9h3_prof_str"]
L2 = [str(m) for m in pd.period_range("2025-11", "2026-09", freq="M")]
WINDOWS = [(0, 3), (2, 5), (5, 8), (7, 10)]   # inclusive 0-based indices into the 11 Look 2 months L2 (a view lacking a month uses the months it has)


def pick(months, vals, names):
    idx = [i for i, m in enumerate(months) if m in names]
    return np.asarray(vals)[idx]


def build_views():
    D = B7.load_all(); ta = D["ta"]; K = D["keys"]
    ok = D["ok"] & B7.finite_rows(K) & B7.finite_rows(D["dA"]) & B7.finite_rows(D["hit"])
    h, w = T.hour_wk(ta); info = {"n_ok_rows": int(ok.sum()), "views": {}}
    for k in KS:
        Z, names = T.harm_prof_str(k, h, w, K); Z[~ok] = np.nan
        B7.split_write(NAMES[k], D, {c: Z[:, j] for j, c in enumerate(names)}, f"T15 Day 1 harmonic profile K={k} + weekend + 4 stretch terms ({len(names)} columns)", info)
        print(NAMES[k], len(names), flush=True)
    T.dump(info, os.path.join(OUT, "build_views.json"))


def evaluate(name):
    out = os.path.join(OUT, name)
    if os.path.exists(os.path.join(out, "results.json")):
        return
    E.evaluate_view(name, out_dir=out, n_perm=5, n_boot=10000, figure=(name == "T15_h6"))


def rst(res, tg):
    mo, v = T.pooled(res, tg)
    s = T.series(res, tg)
    return mo, v, s


def analyse():
    ref = T.stored("M7_gbm"); gref = {tg: T.pooled(ref, tg)[1].mean() for tg in "AB"}
    res = {nm: T.load_json(os.path.join(OUT, nm, "results.json")) for nm in NAMES.values()}
    for nm in STORED:
        res[nm] = T.stored(nm)
    # reproduction check
    a, b = T.pooled(res["T15_h6"], "A")[1], T.pooled(res["M9h6_prof_str"], "A")[1]
    ab, bb = T.pooled(res["T15_h6"], "B")[1], T.pooled(res["M9h6_prof_str"], "B")[1]
    repro = {"max_abs_diff_A": float(np.max(np.abs(a - b))), "max_abs_diff_B": float(np.max(np.abs(ab - bb)))}
    out = {"reproduction_T15_h6_vs_M9h6_prof_str": repro, "reference_M7_gbm_pooled": {k: float(v) for k, v in gref.items()}, "views": {}}
    names = ["M7_gbm", "M7c_gbm_resid", "M9_prof_str"] + [NAMES[k] for k in KS] + ["M9h3_prof_str"]
    for nm in names:
        r = res[nm]; o = {"n_cols": len(r["columns"])}
        for tg in "AB":
            mo, v, s = rst(r, tg); l2m, l2 = s["look2"]
            o[tg] = {"pooled": T.tci(v), "R": float(v.mean() / gref[tg]), "look1": T.tci(s["look1"][1]), "look2": T.tci(l2), "look2_months": l2m,
                     "pass_both_looks": bool(r["looks"]["look1"][tg]["flags"]["pass"] and r["looks"]["look2"][tg]["flags"]["pass"]),
                     "flags": {lk: r["looks"][lk][tg]["flags"]["pass"] for lk in r["looks"]},
                     "drop_nov_dec": T.tci(pick(l2m, l2, L2[2:])), "drop_first3": T.tci(pick(l2m, l2, L2[3:])),
                     "windows": [{"months": f"{L2[a_]}..{L2[b_]}", **T.tci(pick(l2m, l2, L2[a_:b_ + 1]))} for a_, b_ in WINDOWS]}
            o[tg]["n_windows_lo_gt0"] = int(sum(1 for w in o[tg]["windows"] if w["lo"] is not None and w["lo"] > 0))
            o[tg]["stable"] = bool(o[tg]["n_windows_lo_gt0"] >= 3)
            # Look 1 rolling windows (descriptive; same construction, 13 months, 4 evenly spread starts)
            l1m, l1 = s["look1"]; st = [int(round(x)) for x in np.linspace(0, len(l1) - 4, 4)]
            o[tg]["look1_windows_descriptive"] = [{"months": f"{l1m[i]}..{l1m[i + 3]}", **T.tci(l1[i:i + 4])} for i in st]
        o["min_R"] = float(min(o["A"]["R"], o["B"]["R"]))
        out["views"][nm] = o
    T.dump(out, os.path.join(OUT, "results_day1.json"))
    f = {"A": lambda x: "n/a" if x is None else f"{x:+.4f}", "B": lambda x: "n/a" if x is None else f"{100 * x:+.2f}%"}
    ci = lambda tg, d: f"{f[tg](d['mean'])} [{f[tg](d['lo'])}, {f[tg](d['hi'])}]"
    L = [f"T15 Day 1: harmonic sweep, reproduction check T15_h6 vs stored M9h6_prof_str: max|diff| A {repro['max_abs_diff_A']:.2e}, B {repro['max_abs_diff_B']:.2e}",
         f"M7_gbm pooled gain (24 mo): A {f['A'](gref['A'])}, B {f['B'](gref['B'])}", ""]
    for tg, ttl in (("A", "target A: monthly skill increment over V3"), ("B", "target B: monthly relative Brier improvement over H3")):
        L += [f"=== {ttl} ===", f"{'view':16s} {'cols':>4s} {'look1 mean [t-CI]':>30s} {'look2 mean [t-CI]':>30s} {'pooled':>9s} {'R':>6s} pass(l1,l2) {'drop Nov+Dec [t-CI]':>30s} {'drop first 3 [t-CI]':>30s}  win lo>0  stable"]
        for nm, o in out["views"].items():
            a = o[tg]; fl = a["flags"]
            L.append(f"{nm:16s} {o['n_cols']:4d} {ci(tg, a['look1']):>30s} {ci(tg, a['look2']):>30s} {f[tg](a['pooled']['mean']):>9s} {a['R']:6.2f} {str(fl['look1'])[0]},{str(fl['look2'])[0]}"
                     f"         {ci(tg, a['drop_nov_dec']):>30s} {ci(tg, a['drop_first3']):>30s}  {a['n_windows_lo_gt0']}/4     {a['stable']}")
        L.append("")
    L.append("=== rolling 4-month windows on Binance (Look 2): mean [95% t-CI, 3 df] ===")
    for tg in "AB":
        L.append(f"target {tg}")
        for nm, o in out["views"].items():
            L.append(f"  {nm:16s} " + "  ".join(f"{w['months']}: {ci(tg, w)}" for w in o[tg]["windows"]))
    L += ["", "=== STABILITY RULE (A increment's t-interval lower bound > 0 in >= 3 of 4 rolling windows) ==="]
    for nm, o in out["views"].items():
        L.append(f"  {nm:16s} A windows lo>0: {o['A']['n_windows_lo_gt0']}/4 -> {'STABLE' if o['A']['stable'] else 'not stable'};  B windows lo>0: {o['B']['n_windows_lo_gt0']}/4 -> {'B-stable' if o['B']['stable'] else 'B not stable'} (descriptive)")
    open(os.path.join(OUT, "tables_day1.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    T.below_normal(); os.makedirs(OUT, exist_ok=True); T.wall_hours()
    T.run_job(DAY, "build harmonic views K=4,6,8,12", build_views, "day1/build_views.json") if not os.path.exists(os.path.join(OUT, "build_views.json")) else None
    for k in KS:
        T.run_job(DAY, f"harness evaluate {NAMES[k]}", lambda n=NAMES[k]: evaluate(n), f"day1/{NAMES[k]}/results.json")
    T.run_job(DAY, "analyse: gains, sub-samples, windows, stability rule", analyse, "day1/results_day1.json, day1/tables_day1.txt")


if __name__ == "__main__":
    main()
