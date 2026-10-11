"""Tables for SPEC.md. usage: python analyze.py sel   -> sel_table.txt, selection.json (adds) ; python analyze.py look -> look_table.txt"""
import os, sys, json, math
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def read(phase, bar):
    p = f"{HERE}/{'look_recut' if phase == 'look' else phase}_{bar}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def monthly(rows, tag, s, model, T):
    for d in rows:
        if d["tag"] == tag and d["set"] == s and d["model"] == model and d["T"] == T:
            return pd.DataFrame(d["months"]).set_index("month")
    raise KeyError((tag, s, model, T))


def tint(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / math.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return m, m - q * se, m + q * se


def paired(a, b):
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner"); dlt = (j["auc_a"] - j["auc_b"]).to_numpy()
    m, lo, hi = tint(dlt); p = stats.ttest_1samp(dlt, 0.0).pvalue
    return dict(n=len(dlt), mean=m, lo=lo, hi=hi, p=float(p), pos_months=int((dlt > 0).sum()))


def holm(ps):
    order = np.argsort(ps); k = len(ps); adj = np.zeros(k); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (k - rank) * ps[i])); adj[i] = run
    return adj


def sel():
    out = []; tests = []
    for bar in ("1m", "5m"):
        rows = read("sel", bar)
        for kind in ("logreg", "hgb"):
            tb, fl, bo = (monthly(rows, "main", s, kind, 365) for s in ("tb7", "flow", "tb7+flow"))
            d = paired(bo, tb); tests.append((f"{bar}:{kind}", d))
            f0 = tint(fl["auc"]); fd = paired(fl, tb)
            out.append(f"{bar} {kind}: months {d['n']}  AUC tb7 {tb['auc'].mean():.4f}  flow {fl['auc'].mean():.4f} "
                       f"[{f0[1]:.4f},{f0[2]:.4f}]  tb7+flow {bo['auc'].mean():.4f}")
            out.append(f"    tb7+flow - tb7: {d['mean']:+.4f} 95% t [{d['lo']:+.4f},{d['hi']:+.4f}] p {d['p']:.4f} positive months {d['pos_months']}/{d['n']};"
                       f"  flow - tb7: {fd['mean']:+.4f} [{fd['lo']:+.4f},{fd['hi']:+.4f}]")
            out.append(f"    top-10% hit / bps (mean over months): tb7 {tb['top10_hit'].mean():.4f} / {tb['top10_bps'].mean():+.2f}; "
                       f"flow {fl['top10_hit'].mean():.4f} / {fl['top10_bps'].mean():+.2f}; tb7+flow {bo['top10_hit'].mean():.4f} / {bo['top10_bps'].mean():+.2f}")
    adj = holm(np.array([t[1]["p"] for t in tests])); adds = []
    out.append("\nHolm over the 4 tests (adds = adjusted p < 0.05 and mean difference > 0):")
    for (k, d), a in zip(tests, adj):
        ok = bool(a < 0.05 and d["mean"] > 0)
        out.append(f"  {k}: mean {d['mean']:+.4f}  p {d['p']:.4f}  Holm p {a:.4f}  -> {'ADDS' if ok else 'does not add'}")
        if ok:
            adds.append(k)
    json.dump({"adds": adds, "tests": {k: {**d, "holm_p": float(a)} for (k, d), a in zip(tests, adj)}}, open(f"{HERE}/selection.json", "w"), indent=1)
    out += ["", stability("sel")]
    open(f"{HERE}/sel_table.txt", "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))


def stability(upto):
    lines = ["Per-year AUC of flow alone (logreg), mean of the year's months, 95% t-interval over the months (0.5 = chance):"]
    for bar in ("1m", "5m"):
        for T in (60, 365):
            frames = []
            for ph in (["sel"] + (["look"] if upto == "look" else [])):
                rows = read(ph, bar)
                if rows:
                    try:
                        frames.append(monthly(rows, "stab", "flow", "logreg", T))
                    except KeyError:
                        pass
            if not frames:
                continue
            df = pd.concat(frames); df.index = pd.PeriodIndex(df.index, freq="M")
            cells = []
            for y, g in df.groupby(df.index.year):
                if len(g) >= 3:
                    m, lo, hi = tint(g["auc"]); cells.append(f"{y}: {m:.4f} [{lo:.4f},{hi:.4f}] n={len(g)}")
                else:
                    cells.append(f"{y}: n={len(g)}")
            lines.append(f"  {bar} T={T}: " + "; ".join(cells))
    lines.append("  (T=365 in 2017-2018 trains on whatever history exists, less than 365 days, until 2018-01.)")
    return "\n".join(lines)


def look():
    sel_ = json.load(open(f"{HERE}/selection.json")); out = ["LOOK 2024-07..2025-07-20, recut (descriptive; period already used by H38, H56, H63)"]
    if not sel_["adds"]:
        out.append("no test added in selection: no tb7 vs tb7+flow look was run.")
    for key in sel_["adds"]:
        bar, kind = key.split(":"); rows = read("look", bar)
        tb, bo = (monthly(rows, "main", s, kind, 365) for s in ("tb7", "tb7+flow")); d = paired(bo, tb)
        out.append(f"{key}: tb7 {tb['auc'].mean():.4f}  tb7+flow {bo['auc'].mean():.4f}  diff {d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] "
                   f"positive months {d['pos_months']}/{d['n']}")
    out += ["", stability("look")]
    open(f"{HERE}/look_table_recut.txt", "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))


if __name__ == "__main__":
    {"sel": sel, "look": look}[sys.argv[1]]()
