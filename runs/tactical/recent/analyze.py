"""Tables for the 'recent' study. usage: python analyze.py sel   -> selection.json (table + picks) and selection_table.txt
                                       python analyze.py look  -> look.json and look_table.txt (needs the look runs)"""
import json, math, os, sys
import numpy as np
from scipy.stats import t as T

HERE = os.path.dirname(os.path.abspath(__file__))
BARS = ("1m", "1h")
ORDER = ["tb7", "rich", "ctx", "gR", "novel", "har", "all"]


def tci(v, level=0.95):
    v = np.asarray(v, float); n = len(v)
    if n < 2:
        return [float(v.mean()), None, None]
    h = T.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def load(phase, bar):
    rows = {}
    for l in open(f"{HERE}/{phase}_{bar}.jsonl", encoding="utf-8"):
        d = json.loads(l); rows[(d["set"], d["model"], d["T"])] = d
    return rows


def key(c):
    return (c[2], ORDER.index(c[0]), c[1])


def summarize(rows, bar):
    out = []
    for c in sorted(rows, key=key):
        d = rows[c]; base = rows.get(("tb7", "logreg", c[2]))
        mo = {m["month"]: m for m in d["months"]}; bm = {m["month"]: m for m in base["months"]} if base else {}
        com = sorted(set(mo) & set(bm))
        auc = [mo[m]["auc"] for m in com]; diff = [mo[m]["auc"] - bm[m]["auc"] for m in com]
        out.append({"bar": bar, "set": c[0], "model": c[1], "T": c[2], "n_features": d["n_features"], "n_months": len(com),
                    "auc": tci(auc), "diff": tci(diff) if c != ("tb7", "logreg", c[2]) else None,
                    "top10_hit": float(np.mean([mo[m]["top10_hit"] for m in com])), "top10_bps": float(np.mean([mo[m]["top10_bps"] for m in com])),
                    "n_top10": float(np.mean([mo[m]["n_top10"] for m in com]))})
    return out


def fmt(s):
    a, d = s["auc"], s["diff"]
    dd = "   baseline          " if d is None else f"{d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]"
    return (f"{s['set']:6s} {s['model']:6s} T={s['T']:<3d} F={s['n_features']:<4d} n={s['n_months']:<2d} AUC {a[0]:.4f} [{a[1]:.4f},{a[2]:.4f}]  "
            f"diff {dd}  top10 hit {s['top10_hit']:.4f} {s['top10_bps']:+.2f} bps")


def sel():
    res = {"table": {}, "picks": {}}; txt = []
    for bar in BARS:
        s = summarize(load("sel", bar), bar); res["table"][bar] = s
        cand = [x for x in s if x["diff"] is not None]
        top = sorted(cand, key=lambda x: -x["diff"][0])[:3]      # ties keep the table order (sorted is stable)
        res["picks"][bar] = [f"{x['set']}:{x['model']}:{x['T']}" for x in top]
        txt += [f"== {bar} SELECTION months 2021-01..2024-06 ({s[0]['n_months']} common months); paired diff vs tb7:logreg of the same T"]
        txt += [fmt(x) for x in s] + ["picks: " + ", ".join(res["picks"][bar]), ""]
    json.dump(res, open(f"{HERE}/selection.json", "w"), indent=1)
    open(f"{HERE}/selection_table.txt", "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt))


def look():
    picks = json.load(open(f"{HERE}/selection.json"))["picks"]; res = {}; txt = []
    for bar in BARS:
        rows = load("look", bar); res[bar] = []
        txt.append(f"== {bar} LOOK months 2024-07..2025-07-20 (recut; one look)")
        for p in picks[bar]:
            s, m, t_ = p.split(":"); t_ = int(t_); d = rows[(s, m, t_)]; b = rows[("tb7", "logreg", t_)]
            mo = {x["month"]: x for x in d["months"]}; bm = {x["month"]: x for x in b["months"]}; com = sorted(set(mo) & set(bm))
            auc = [mo[k]["auc"] for k in com]; diff = [mo[k]["auc"] - bm[k]["auc"] for k in com]; bauc = [bm[k]["auc"] for k in com]
            r = {"pick": p, "n_months": len(com), "auc": tci(auc), "auc_vs_half95": tci(np.array(auc) - 0.5), "base_auc": tci(bauc),
                 "diff95": tci(diff), "diff9833": tci(diff, 1 - 0.05 / 3), "success_95": bool(tci(diff)[1] is not None and tci(diff)[1] > 0),
                 "success_9833": bool(tci(diff, 1 - 0.05 / 3)[1] is not None and tci(diff, 1 - 0.05 / 3)[1] > 0),
                 "top10_hit": float(np.mean([mo[k]["top10_hit"] for k in com])), "top10_bps": float(np.mean([mo[k]["top10_bps"] for k in com])),
                 "monthly_auc": {k: round(mo[k]["auc"], 4) for k in com}}
            res[bar].append(r)
            txt.append(f"{p:16s} n={len(com)} AUC {r['auc'][0]:.4f} [{r['auc'][1]:.4f},{r['auc'][2]:.4f}] (baseline {r['base_auc'][0]:.4f}); "
                       f"diff95 {r['diff95'][0]:+.4f} [{r['diff95'][1]:+.4f},{r['diff95'][2]:+.4f}] success95={r['success_95']}; "
                       f"diff98.3 [{r['diff9833'][1]:+.4f},{r['diff9833'][2]:+.4f}] success={r['success_9833']}; "
                       f"AUC-0.5 95% [{r['auc_vs_half95'][1]:+.4f},{r['auc_vs_half95'][2]:+.4f}]; top10 hit {r['top10_hit']:.4f} {r['top10_bps']:+.2f} bps")
        txt.append("")
    json.dump(res, open(f"{HERE}/look.json", "w"), indent=1)
    open(f"{HERE}/look_table.txt", "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt))


if __name__ == "__main__":
    {"sel": sel, "look": look}[sys.argv[1]]()
