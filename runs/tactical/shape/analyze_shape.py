"""Tables and verdicts against SPEC.md. usage: python analyze_shape.py mcA|mcB|sel|look"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
TS = (30, 60, 120); TARGETS = ("rho", "vr", "z2", "z3"); MODELS = ("ridge", "hgb")


def load(phase):
    return [json.loads(l) for l in open(f"{RES}/{phase}.jsonl", encoding="utf-8")]


def cell_series(rows, tgt, T, model, key=("sp_b", "sp_c")):
    out = {}
    for r in rows:
        for c in r["cells"]:
            if c["target"] == tgt and c["T"] == T and c["model"] == model:
                out[r["month"]] = c
    ms = sorted(out)
    return ms, {k: np.array([out[m][k] for m in ms]) for k in (set(key) | {"n"} | ({"mse_b", "mse_c"} if tgt == "rho" else set()))}


def t_stats(d):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); t = m / se if se > 0 else 0.0
    return {"n_months": n, "mean": float(m), "se": float(se), "t": float(t), "p_one": float(1 - stats.t.cdf(t, n - 1)),
            "lo95": float(m - stats.t.ppf(0.975, n - 1) * se), "hi95": float(m + stats.t.ppf(0.975, n - 1) * se)}


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0; k = len(ps)
    for rank, i in enumerate(order):
        run = max(run, (k - rank) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def table(rows, cells=None):
    out = []
    for tgt in TARGETS:
        for T in TS:
            for model in MODELS:
                if cells is not None and (tgt, model) not in cells:
                    continue
                ms, s = cell_series(rows, tgt, T, model)
                if len(ms) < 3:
                    continue
                d = s["sp_c"] - s["sp_b"]; st = t_stats(d)
                rec = {"target": tgt, "T": T, "model": model, "sp_base": float(s["sp_b"].mean()), "sp_chal": float(s["sp_c"].mean()),
                       "n_eff_month": float(np.mean(s["n"]) / T), **st}
                if tgt == "rho":
                    rec["mse_skill"] = float(np.mean(1 - s["mse_c"] / s["mse_b"]))
                out.append(rec)
    return out


def fmt(tab, extra=""):
    h = f"{'target':6s} {'T':>4s} {'model':5s} {'SpB':>7s} {'SpC':>7s} {'dMean':>8s} {'SE':>7s} {'lo95':>8s} {'hi95':>8s} {'p1':>8s} {'pHolm':>8s} {'neff/mo':>8s} {'mseSk':>8s}{extra}"
    L = [h]
    for r in tab:
        L.append(f"{r['target']:6s} {r['T']:4d} {r['model']:5s} {r['sp_base']:+7.4f} {r['sp_chal']:+7.4f} {r['mean']:+8.4f} {r['se']:7.4f} {r['lo95']:+8.4f} "
                 f"{r['hi95']:+8.4f} {r['p_one']:8.4f} {r.get('p_holm', float('nan')):8.4f} {r['n_eff_month']:8.1f} {r.get('mse_skill', float('nan')):+8.4f}")
    return "\n".join(L)


def decay(rows, tgt, model):
    ms, s = cell_series(rows, tgt, 60, model); d = pd.Series(s["sp_c"] - s["sp_b"], index=ms)
    a = {r["month"]: r["auc_tb7"] for r in rows}
    ax = pd.Series(a).sort_index() - 0.5
    def seg(x, lo, hi):
        return x[(x.index >= lo) & (x.index <= hi)].mean()
    rng = np.random.default_rng(0); boots = []
    idx_late = [m for m in d.index if "2023-01" <= m <= "2024-06"]; idx_early = [m for m in d.index if "2019-01" <= m <= "2020-12"]
    for _ in range(2000):
        bl = rng.choice(idx_late, len(idx_late)); be = rng.choice(idx_early, len(idx_early))
        boots.append(d[bl].mean() / d[be].mean() if d[be].mean() != 0 else np.nan)
    ra = seg(ax, "2023-01", "2024-06") / seg(ax, "2019-01", "2020-12")
    al = [m for m in ax.index if "2023-01" <= m <= "2024-06"]; ae = [m for m in ax.index if "2019-01" <= m <= "2020-12"]
    ab = [ax[rng.choice(al, len(al))].mean() / ax[rng.choice(ae, len(ae))].mean() for _ in range(2000)]
    return {"shape_early": float(d[idx_early].mean()), "shape_late": float(d[idx_late].mean()), "shape_ratio": float(d[idx_late].mean() / d[idx_early].mean()),
            "shape_ratio_ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
            "auc_excess_early": float(seg(ax, "2019-01", "2020-12")), "auc_excess_late": float(seg(ax, "2023-01", "2024-06")), "auc_ratio": float(ra),
            "auc_ratio_ci": [float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))]}


def main(phase):
    rows = load(phase); tab = table(rows)
    adj = holm(np.array([r["p_one"] for r in tab]))
    for r, a in zip(tab, adj):
        r["p_holm"] = float(a)
    months = sorted(r["month"] for r in rows)
    head = f"== {phase}: {len(months)} months {months[0]}..{months[-1]}; {len(tab)} cells, Holm over {len(tab)} one-sided tests"
    txt = [head, fmt(tab)]
    res = {"phase": phase, "months": [months[0], months[-1], len(months)], "table": tab}
    if phase in ("mcA", "mcB"):
        if phase == "mcA":
            ok = all(abs(r["mean"]) < 0.01 for r in tab) and not any(r["p_holm"] < 0.05 and r["mean"] > 0 for r in tab)
            txt.append(f"\nNULL A pass (all |mean Delta| < 0.01 and no Holm-significant positive Delta): {ok}")
        else:
            hit = [r for r in tab if r["T"] == 60 and r["mean"] >= 0.02 and r["p_holm"] < 0.05]
            ok = len(hit) > 0
            txt.append(f"\nPOSITIVE CONTROL B pass (T=60, some target Delta >= 0.02 with Holm p < 0.05): {ok}; hits: {[(h['target'], h['model']) for h in hit]}")
        res["pass"] = bool(ok)
    elif phase == "sel":
        sel = [r for r in tab if r["T"] == 60 and r["mean"] >= 0.02 and r["lo95"] > 0 and r["p_holm"] < 0.05]
        res["selected"] = [{"target": r["target"], "model": r["model"]} for r in sel]
        json.dump({"cells": res["selected"]}, open(f"{RES}/selected.json", "w"), indent=1)
        txt.append(f"\nSELECTED (T=60, mean >= +0.02, lo95 > 0, Holm p < 0.05): {res['selected'] or 'none'}")
        res["decay"] = {}
        for tgt in TARGETS:
            for model in MODELS:
                res["decay"][f"{tgt}:{model}"] = decay(rows, tgt, model)
        txt.append("\ndecay (T=60) shape Delta early(2019-20)/late(2023-24H1) vs tb7 AUC excess:")
        for k, d in res["decay"].items():
            txt.append(f"  {k:10s} shape {d['shape_early']:+.4f} -> {d['shape_late']:+.4f} ratio {d['shape_ratio']:+.2f} CI {d['shape_ratio_ci'][0]:+.2f}..{d['shape_ratio_ci'][1]:+.2f}"
                       f" | AUC excess {d['auc_excess_early']:+.4f} -> {d['auc_excess_late']:+.4f} ratio {d['auc_ratio']:.2f} CI {d['auc_ratio_ci'][0]:.2f}..{d['auc_ratio_ci'][1]:.2f}")
    elif phase == "look":
        sel = json.load(open(f"{RES}/selected.json"))["cells"]; m = len(sel); out = []
        for r in tab:
            if r["T"] == 60 and (r["target"], r["model"]) in {(c["target"], c["model"]) for c in sel}:
                conf = r["mean"] >= 0.01 and min(1.0, r["p_one"] * m) < 0.05
                out.append((r["target"], r["model"], conf, r["mean"], min(1.0, r["p_one"] * m)))
        res["confirmed"] = [{"target": a, "model": b, "confirmed": bool(c), "mean": d, "p_bonf": e} for a, b, c, d, e in out]
        txt.append("\nCONFIRMATION (mean >= +0.01 and one-sided Bonferroni p < 0.05): " + str(res["confirmed"]))
    s = "\n".join(txt); print(s)
    open(f"{RES}/{phase}_table.txt", "w", encoding="utf-8").write(s); json.dump(res, open(f"{RES}/{phase}_result.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
