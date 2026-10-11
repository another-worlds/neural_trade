"""Tables for the hourly long-context study (exploratory, dev only): per-year mean AUC with 95% t-intervals over months, paired
differences against the tb7 logistic of the same months, net minus its own logistic start, and the best-epoch distribution.
usage: python analyze.py -> results_T365.json, results_T1095.json, tables.txt"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["tb7", "ctx", "patch168", "tcn168", "patch336", "tcn336", "ens168", "ens336", "ens_all"]
NETS = ["patch168", "tcn168", "patch336", "tcn336"]


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return (float(m), float(m - h), float(m + h), n)


def load(T):
    p = f"{HERE}/walk_T{T}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def summarize(rows):
    yrs = sorted({r["month"][:4] for r in rows}); out = {"n_months": len(rows), "years": {}, "overall": {}}
    for scope, sel in [("overall", rows)] + [(y, [r for r in rows if r["month"][:4] == y]) for y in yrs]:
        d = {}
        for mdl in MODELS:
            v = [r[f"auc_{mdl}"] for r in sel if f"auc_{mdl}" in r]
            if not v:
                continue
            e = {"auc": ci(v)}
            if mdl != "tb7":
                e["d_vs_tb7"] = ci([r[f"auc_{mdl}"] - r["auc_tb7"] for r in sel if f"auc_{mdl}" in r])
            if mdl in NETS:
                e["d_vs_start"] = ci([r[f"auc_{mdl}"] - r[f"auc_{mdl}_start"] for r in sel])
            d[mdl] = e
        (out["overall"] if scope == "overall" else out["years"]).update(d if scope == "overall" else {scope: d})
    out["best_epoch"] = {}
    for mdl in NETS:
        be = np.array([r[f"{mdl}_info"]["best_epoch"] for r in rows if f"{mdl}_info" in r])
        if len(be):
            out["best_epoch"][mdl] = {"n": int(len(be)), "frac_epoch0": float((be == 0).mean()), "median": float(np.median(be)),
                                      "mean": float(be.mean()), "max": int(be.max()), "hist": np.bincount(np.minimum(be, 10), minlength=11).tolist(),
                                      "note": "hist index = best epoch, last bin = 10 or more"}
    return out


def fmt(c):
    return f"{c[0]:.4f} [{c[1]:+.4f},{c[2]:+.4f}]" if not np.isnan(c[1]) else f"{c[0]:.4f}"


def table(T, s):
    L = [f"=== T = {T} days  (exploratory, dev only; months {s['n_months']}; cells: mean [95% t-interval over months]) ==="]
    for scope in ["overall"] + sorted(s["years"]):
        d = s["overall"] if scope == "overall" else s["years"][scope]
        n = s["n_months"] if scope == "overall" else next(iter(d.values()))["auc"][3]
        L.append(f"-- {scope} ({n} months)")
        L.append(f"{'model':10s} {'AUC':30s} {'paired diff vs tb7':32s} {'net minus own linear start'}")
        for mdl, e in d.items():
            L.append(f"{mdl:10s} {fmt(e['auc']):30s} {fmt(e['d_vs_tb7']) if 'd_vs_tb7' in e else '':32s} {fmt(e['d_vs_start']) if 'd_vs_start' in e else ''}")
    L.append("-- best epoch of the nets (0 = never left the linear start)")
    for mdl, e in s["best_epoch"].items():
        L.append(f"{mdl:10s} n {e['n']}  never left start {e['frac_epoch0']:.0%}  median {e['median']:.0f}  mean {e['mean']:.1f}  max {e['max']}  hist(0..10+) {e['hist']}")
    return "\n".join(L)


if __name__ == "__main__":
    txt = []
    for T in (365, 1095):
        rows = load(T)
        if not rows:
            continue
        s = summarize(rows); json.dump(s, open(f"{HERE}/results_T{T}.json", "w"), indent=1); txt.append(table(T, s))
    open(f"{HERE}/tables.txt", "w", encoding="utf-8").write("\n\n".join(txt) + "\n"); print("\n\n".join(txt))
