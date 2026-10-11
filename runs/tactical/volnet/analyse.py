"""usage: python analyse.py select   -> selection.json + selection tables (selection months only)
         python analyse.py look     -> look.json (the selected model and the baseline only)"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = ["hgb", "mlp", "seq", "ens"]


def load(stage):
    rows = [json.loads(l) for f in sorted(glob.glob(f"{HERE}/results_{stage}_[0-9].jsonl")) for l in open(f)]
    return pd.DataFrame(rows)


def paired(df, model, H):
    b = df[(df.model == "har") & (df.H == H)].set_index("month"); m = df[(df.model == model) & (df.H == H)].set_index("month")
    j = b.index.intersection(m.index)
    return pd.DataFrame({"sp": m.loc[j, "spearman"] - b.loc[j, "spearman"], "ql": b.loc[j, "qlike"] - m.loc[j, "qlike"]}).sort_index()


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(np.mean(x)) if n else float("nan"), float("nan"), float("nan"), float("nan")
    mu, se = x.mean(), x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return float(mu), float(mu - h), float(mu + h), float(mu / se if se > 0 else 0.0)


def table(df, models, label):
    out, lines = {}, [f"== {label} =="]
    Hs = sorted(df.H.unique())
    for H in Hs:
        lines.append(f"-- horizon {H} min: mean level of har: spearman {df[(df.model=='har')&(df.H==H)].spearman.mean():.4f}, qlike {df[(df.model=='har')&(df.H==H)].qlike.mean():.4f}")
        lines.append("model        metric   all months [95% CI]            " + "  ".join(sorted(set(df.month.str[:4]))))
        for mo in models:
            p = paired(df, mo, H)
            if p.empty:
                continue
            out[f"{mo}_{H}"] = {}
            for col, nm in (("sp", "dSpear"), ("ql", "dQLIKE")):
                mu, lo, hi, t = ci(p[col]); yr = p[col].groupby(p.index.str[:4]).mean()
                out[f"{mo}_{H}"][col] = {"mean": mu, "lo": lo, "hi": hi, "t": t, "n": len(p), "by_year": {k: float(v) for k, v in yr.items()}}
                lines.append(f"{mo:11s}  {nm:7s} {mu:+.4f} [{lo:+.4f},{hi:+.4f}] n={len(p):2d}  " + "  ".join(f"{v:+.4f}" for v in yr.values))
    return out, "\n".join(lines)


def main():
    mode = sys.argv[1]
    if mode == "select":
        df = load("sel"); out, txt = table(df, CAND + ["ridge_full"], "selection months (paired vs har; QLIKE diff = har - model, positive = better)")
        lv = []
        for mo in sorted(df.model.unique()):
            for H in sorted(df.H.unique()):
                d = df[(df.model == mo) & (df.H == H)]; lv.append(f"{mo:11s} H{H:<4d} spearman {d.spearman.mean():.4f}  qlike {d.qlike.mean():.4f}  n={len(d)}")
        txt += "\n\n== levels (selection months) ==\n" + "\n".join(lv)
        score = {mo: float(np.mean([out[f"{mo}_{H}"][c]["t"] for H in sorted(df.H.unique()) for c in ("sp", "ql")])) for mo in CAND}
        best = max(score, key=score.get)
        json.dump({"selected": best, "scores_mean_t": score, "detail": out, "n_months": int(df.month.nunique())}, open(f"{HERE}/selection.json", "w"), indent=1)
        open(f"{HERE}/selection_tables.txt", "w").write(txt + f"\n\nselection scores (mean of six t-values): {score}\nSELECTED: {best}\n")
        print(txt); print(score, "SELECTED", best)
    else:
        sel = json.load(open(f"{HERE}/selection.json"))["selected"]; df = load("look")
        out, txt = table(df, [sel], f"THE ONE LOOK, 2024-07..2025-09: {sel} vs har")
        verdict = {}
        for H in sorted(df.H.unique()):
            o = out[f"{sel}_{H}"]; verdict[str(H)] = bool(o["sp"]["lo"] > 0 and o["ql"]["lo"] > 0)
        json.dump({"selected": sel, "success_by_horizon": verdict, "detail": out}, open(f"{HERE}/look.json", "w"), indent=1)
        open(f"{HERE}/look_tables.txt", "w").write(txt + f"\n\nsuccess (both lower bounds > 0) by horizon: {verdict}\n")
        print(txt); print(verdict)


if __name__ == "__main__":
    main()
