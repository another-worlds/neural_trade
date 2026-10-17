"""M9 readout (CPU, seconds): recovery ratios, pass flags and the freeze choice from the stored harness results.json files (SPEC_M9 "Readout", "Freeze rule").
Run: D:/nt/nt_mm_env/python.exe analyse_m9.py -> results_m9.json, tables_m9.txt"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); T14 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import m9_lib as M9  # noqa: E402

REF = "M7_gbm"
SHOW = ["M7_gbm", "M7c_gbm_resid", "M8_lin", "M8x_lin_wk", "M9_prof", "M9_prof_str", "M9_prof_str_vp", "M9_gbm7", "M9x_B", "M9x_A", "M9x_AB"] + M9.POST_HOC_H


def load(name):
    p = os.path.join(T14, name, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def series(res, tg):
    return {lk: np.array(r["A"]["increment_series"]["view"] if tg == "A" else r["B"]["monthly_rel"]["view"], float) for lk, r in res["looks"].items()}


def paired(a, b, tg):
    """M7_gbm minus view as a monthly series paired by month name (a view may lack a month)."""
    d = []
    for lk in a["looks"]:
        ma = a["looks"][lk][tg]["months"]; mb = b["looks"][lk][tg]["months"]; sa = series(a, tg)[lk]; sb = series(b, tg)[lk]
        d += [sa[ma.index(m)] - sb[mb.index(m)] for m in ma if m in mb]
    return tci(d)


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return {"mean": float(m), "lo": float(m - t * se), "hi": float(m + t * se), "n": int(n)}


def harness(res, tg, lk):
    r = res["looks"][lk]; v = r["A"]["increment"]["view"] if tg == "A" else r["B"]["rel_improvement"]["view"]
    ctrl = r["A"]["increment"] if tg == "A" else r["B"]["rel_improvement"]
    return {"mean": v["mean"], "lo": v["ci"][0], "hi": v["ci"][1], "c1": ctrl["c1"]["mean"], "c2a": ctrl["c2a"]["mean"],
            **({"c2b": ctrl["c2b"]["mean"]} if "c2b" in ctrl else {}), "pass": r[tg]["flags"]["pass"], "flags": r[tg]["flags"]}


def main():
    full = load(REF); out = {"views": {}}; L = []
    f = {"A": lambda v: f"{v:+.4f}", "B": lambda v: f"{100 * v:+.2f}%"}
    gref = {}
    for tg in "AB":
        s = series(full, tg); gref[tg] = {"pooled": float(np.concatenate([s["look1"], s["look2"]]).mean()), "look1": float(s["look1"].mean()), "look2": float(s["look2"].mean())}
    out["reference"] = {"view": REF, "gain": gref}
    for nm in SHOW:
        r = load(nm)
        if r is None:
            continue
        o = {"n_cols": len(r["columns"]), "columns_head": r["columns"][:6]}
        for tg in "AB":
            s = series(r, tg); sf = series(full, tg); p = np.concatenate([s["look1"], s["look2"]]); pf = np.concatenate([sf["look1"], sf["look2"]])
            o[tg] = {"pooled": tci(p), "R": float(p.mean() / gref[tg]["pooled"]), "R_look1": float(s["look1"].mean() / gref[tg]["look1"]), "R_look2": float(s["look2"].mean() / gref[tg]["look2"]),
                     "ref_minus_view": paired(full, r, tg), "look1": harness(r, tg, "look1"), "look2": harness(r, tg, "look2"),
                     "pass_both_looks": bool(r["looks"]["look1"][tg]["flags"]["pass"] and r["looks"]["look2"][tg]["flags"]["pass"])}
        o["pass"] = bool(o["A"]["pass_both_looks"] or o["B"]["pass_both_looks"]); o["min_R"] = float(min(o["A"]["R"], o["B"]["R"])); o["label"] = M9.recovery_label(o["min_R"])
        out["views"][nm] = o
    for tg, nm_t in (("A", "monthly skill increment over V3"), ("B", "monthly relative Brier improvement over H3 (day-block CI)")):
        L += [f"=== target {tg}: {nm_t} ===",
              f"{'view':16s} {'cols':>4s} {'look1 [CI]':>34s} {'look2 [CI]':>34s} {'pooled 24 mo':>13s} {'R':>6s} {'R1':>6s} {'R2':>6s}  C1 l1/l2        C2a l1/l2       pass(l1,l2)"]
        for nm, o in out["views"].items():
            a = o[tg]; h1, h2 = a["look1"], a["look2"]; c = lambda h: f"{f[tg](h['mean'])} [{f[tg](h['lo'])}, {f[tg](h['hi'])}]"
            L.append(f"{nm:16s} {o['n_cols']:4d} {c(h1):>34s} {c(h2):>34s} {f[tg](a['pooled']['mean']):>13s} {a['R']:6.2f} {a['R_look1']:6.2f} {a['R_look2']:6.2f}  "
                     f"{f[tg](h1['c1'])}/{f[tg](h2['c1'])}  {f[tg](h1['c2a'])}/{f[tg](h2['c2a'])}  {h1['pass']},{h2['pass']}")
        L.append("")
    L.append("=== per view: paired M7_gbm minus view (24 months), verdict ===")
    for nm, o in out["views"].items():
        d = lambda t: f"{f[t](o[t]['ref_minus_view']['mean'])} [{f[t](o[t]['ref_minus_view']['lo'])}, {f[t](o[t]['ref_minus_view']['hi'])}]"
        L.append(f"  {nm:16s} A R={o['A']['R']:.2f} B R={o['B']['R']:.2f} min R={o['min_R']:.2f} -> {o['label']:9s} passes(any target, both looks)={o['pass']}  "
                 f"(A pass {o['A']['pass_both_looks']}, B pass {o['B']['pass_both_looks']}); M7 minus view A {d('A')}, B {d('B')}")
    elig = {nm: {"pass": out["views"][nm]["pass"], "R_A": out["views"][nm]["A"]["R"], "R_B": out["views"][nm]["B"]["R"], "n_cols": out["views"][nm]["n_cols"]}
            for nm in M9.PRE_REGISTERED if nm in out["views"]}
    ch = M9.choose_freeze(elig); tag = "pre-registered"
    if ch is None:
        el2 = {nm: {"pass": out["views"][nm]["pass"], "R_A": out["views"][nm]["A"]["R"], "R_B": out["views"][nm]["B"]["R"], "n_cols": out["views"][nm]["n_cols"]}
               for nm in M9.POST_HOC + M9.POST_HOC_H if nm in out["views"]}
        ch = M9.choose_freeze(el2); tag = "post hoc"
    out["freeze_choice"] = {"view": ch, "kind": tag if ch else None, "eligible_pre": elig}
    L.append(f"\nFREEZE RULE: eligible {elig}\n  -> {ch} ({tag if ch else 'nothing frozen'})")
    # sensitivity (addendum 1): Look 2 without the months with < 1500 training rows (descriptive)
    L.append("\n=== sensitivity: Look 2 without months with < 1500 training rows (Nov, Dec 2025); descriptive, no flag ===")
    out["sensitivity"] = {}
    for tg in "AB":
        L.append(f"target {tg}: {'view':18s} look2-trimmed mean [t-CI]   R_look2-trimmed   (n months)")
        sfull = series(full, tg)["look2"]; mfull = full["looks"]["look2"][tg]["months"]
        for nm in out["views"]:
            r = load(nm); mo = r["looks"]["look2"][tg]["months"]; nt = {m["month"]: m.get("n_tr", 0) for m in r["looks"]["look2"]["A"]["per_month"]}
            keep = [i for i, m in enumerate(mo) if nt.get(m, 0) >= 1500]; x = series(r, tg)["look2"][keep]
            kf = [i for i, m in enumerate(mfull) if m in [mo[j] for j in keep]]; xf = sfull[kf]
            t = tci(x); out["sensitivity"].setdefault(nm, {})[tg] = {**t, "R": float(x.mean() / xf.mean())}
            L.append(f"          {nm:18s} {f[tg](t['mean'])} [{f[tg](t['lo'])}, {f[tg](t['hi'])}]   {x.mean() / xf.mean():5.2f}   ({t['n']})")
    json.dump(out, open(os.path.join(HERE, "results_m9.json"), "w"), indent=1)
    open(os.path.join(HERE, "tables_m9.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
