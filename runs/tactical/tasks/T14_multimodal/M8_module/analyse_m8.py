"""M8 readout (CPU, seconds): the ablation table (SPEC_M8 (a)) and the cheap-linear recovery (SPEC_M8 (b)) from the stored harness results.json files.
Run: D:/nt/nt_mm_env/python.exe analyse_m8.py -> results_m8.json, tables_m8.txt"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); T14 = os.path.dirname(HERE)
GROUPS = ["stretch", "volpct", "hourwk", "sr", "lrv"]
SHARE_MIN, R_YES, R_PART = 0.25, 0.75, 0.40


def load(name):
    return json.load(open(os.path.join(T14, name, "results.json")))


def series(res, target):
    """{look: monthly series} of the view's gain: A increment over V3, B relative Brier improvement over H3 (monthly)."""
    out = {}
    for lk, r in res["looks"].items():
        out[lk] = np.array(r["A"]["increment_series"]["view"] if target == "A" else r["B"]["monthly_rel"]["view"], float)
    return out


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return {"mean": float(m), "lo": float(m - t * se), "hi": float(m + t * se), "n": int(n), "pos": int((x > 0).sum())}


def harness(res, target, lk):
    r = res["looks"][lk]
    if target == "A":
        v = r["A"]["increment"]["view"]; return {"mean": v["mean"], "lo": v["ci"][0], "hi": v["ci"][1]}
    v = r["B"]["rel_improvement"]["view"]; return {"mean": v["mean"], "lo": v["ci"][0], "hi": v["ci"][1]}


def main():
    full = load("M7_gbm"); abl = {g: load(f"M8_abl_{g}") for g in GROUPS}; lin = load("M8_lin")
    out = {"ablation": {}, "linear": {}}; L = []
    f = lambda v, pct: (f"{100 * v:+.2f}%" if pct else f"{v:+.4f}")
    for tg, pct in (("A", False), ("B", True)):
        sf = series(full, tg); pooled_full = np.concatenate([sf["look1"], sf["look2"]])
        out["ablation"][tg] = {"full": {"pooled": tci(pooled_full), **{lk: harness(full, tg, lk) for lk in sf}}, "groups": {}}
        L += [f"=== target {tg}: " + ("monthly skill increment over V3" if tg == "A" else "monthly relative Brier improvement over H3") + " ===",
              f"{'model':22s} {'look1 (harness CI)':>36s} {'look2 (harness CI)':>36s} {'pooled 24 mo':>14s}   drop vs full (paired, 24 months)   share  carries"]
        def row(nm, res, drop=None, share=None, carries=None):
            h1, h2 = harness(res, tg, "look1"), harness(res, tg, "look2"); s = series(res, tg); p = np.concatenate([s["look1"], s["look2"]])
            c = lambda h: f"{f(h['mean'], pct)} [{f(h['lo'], pct)}, {f(h['hi'], pct)}]"
            tail = "" if drop is None else f"   {f(drop['mean'], pct)} [{f(drop['lo'], pct)}, {f(drop['hi'], pct)}]   {share:5.0%}  {'YES' if carries else 'no'}"
            L.append(f"{nm:22s} {c(h1):>36s} {c(h2):>36s} {f(p.mean(), pct):>14s}{tail}")
        row("M7_gbm (full)", full)
        drops = {}
        for g in GROUPS:
            s = series(abl[g], tg); pg = np.concatenate([s["look1"], s["look2"]]); d = pooled_full - pg; dt = tci(d)
            share = dt["mean"] / pooled_full.mean(); carries = bool(share >= SHARE_MIN and dt["lo"] > 0)
            per_look = {lk: tci(sf[lk] - s[lk]) for lk in sf}
            out["ablation"][tg]["groups"][g] = {"pooled_mean": float(pg.mean()), "drop": dt, "share": float(share), "carries": carries,
                                                "drop_per_look": per_look, **{lk: harness(abl[g], tg, lk) for lk in sf}}
            drops[g] = dt["mean"]; row(f"without {g}", abl[g], dt, share, carries)
        main_g = max(drops, key=drops.get); out["ablation"][tg]["main_carrier"] = main_g
        L.append(f"  main carrier on {tg}: {main_g}; groups that carry (share >= 25% and drop CI above 0): {[g for g in GROUPS if out['ablation'][tg]['groups'][g]['carries']]}")
        L.append("  per-look paired drops (t-interval over the look's months): " + "; ".join(
            f"{g}: " + "/".join(f"{f(out['ablation'][tg]['groups'][g]['drop_per_look'][lk]['mean'], pct)}" for lk in ("look1", "look2")) for g in GROUPS))
        L.append("")
    L += ["=== (b) cheap linear recovery: M8_lin against M7_gbm ==="]
    Rs = {}
    for tg, pct in (("A", False), ("B", True)):
        sf = series(full, tg); sl = series(lin, tg); pf = np.concatenate([sf["look1"], sf["look2"]]); pl = np.concatenate([sl["look1"], sl["look2"]])
        R = float(pl.mean() / pf.mean()); Rs[tg] = R; dd = tci(pf - pl)
        out["linear"][tg] = {"R": R, "lin_pooled": tci(pl), "full_pooled": tci(pf), "diff_full_minus_lin": dd,
                             "R_look1": float(sl["look1"].mean() / sf["look1"].mean()), "R_look2": float(sl["look2"].mean() / sf["look2"].mean()),
                             **{lk: harness(lin, tg, lk) for lk in sf}}
        L.append(f"  {tg}: M8_lin pooled {f(pl.mean(), pct)} (look1 {f(sl['look1'].mean(), pct)}, look2 {f(sl['look2'].mean(), pct)}) vs M7_gbm {f(pf.mean(), pct)}: R = {R:.2f} "
                 f"(look1 {out['linear'][tg]['R_look1']:.2f}, look2 {out['linear'][tg]['R_look2']:.2f}); paired full minus lin {f(dd['mean'], pct)} [{f(dd['lo'], pct)}, {f(dd['hi'], pct)}]")
        for lk in sf:
            h = harness(lin, tg, lk); L.append(f"     {lk}: M8_lin harness {f(h['mean'], pct)} [{f(h['lo'], pct)}, {f(h['hi'], pct)}]")
    m = min(Rs.values()); verdict = "recovers" if m >= R_YES else ("partly" if m >= R_PART else "no")
    out["linear"]["verdict"] = verdict; out["linear"]["min_R"] = m
    out["linear"]["flags"] = {lk: {"A_pass": lin["looks"][lk]["A"]["flags"]["pass"], "B_pass": lin["looks"][lk]["B"]["flags"]["pass"]} for lk in lin["looks"]}
    L.append(f"  VERDICT (min R = {m:.2f}; yes >= 0.75, partly >= 0.40): {verdict}.  M8_lin harness flags: {out['linear']['flags']}")
    # ---- POST HOC views (not in SPEC_M8): hour-only / weekend-only ablations and M8_lin plus weekend terms
    L += ["", "=== POST HOC (labelled; written after the pre-registered readout) ==="]; out["posthoc"] = {}
    for nm in ("M8x_abl_hour", "M8x_abl_wkend", "M8x_lin_wk"):
        if not os.path.exists(os.path.join(T14, nm, "results.json")):
            continue
        r = load(nm); out["posthoc"][nm] = {}
        for tg, pct in (("A", False), ("B", True)):
            sf = series(full, tg); s = series(r, tg); pf = np.concatenate([sf["look1"], sf["look2"]]); p = np.concatenate([s["look1"], s["look2"]]); dd = tci(pf - p)
            h1, h2 = harness(r, tg, "look1"), harness(r, tg, "look2")
            out["posthoc"][nm][tg] = {"pooled": float(p.mean()), "R_vs_full": float(p.mean() / pf.mean()), "full_minus_view": dd, "look1": h1, "look2": h2}
            L.append(f"  {nm:15s} {tg}: look1 {f(h1['mean'], pct)} [{f(h1['lo'], pct)}, {f(h1['hi'], pct)}]  look2 {f(h2['mean'], pct)} [{f(h2['lo'], pct)}, {f(h2['hi'], pct)}]  "
                     f"pooled {f(p.mean(), pct)} ({p.mean() / pf.mean():.0%} of M7_gbm; full minus view {f(dd['mean'], pct)} [{f(dd['lo'], pct)}, {f(dd['hi'], pct)}])")
    json.dump(out, open(os.path.join(HERE, "results_m8.json"), "w"), indent=1)
    open(os.path.join(HERE, "tables_m8.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
