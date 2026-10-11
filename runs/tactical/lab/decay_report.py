"""Tables for the decay study: per-year AUC (mean [95% t-interval over months]) for Q1-Q3 and the Q4 selection. usage: python decay_report.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decay_lib as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recut", "decay")
YEARS = [str(y) for y in range(2017, 2026)]
ld = lambda n: json.load(open(os.path.join(OUT, n + ".json")))
f = lambda v: "   n/a   " if not v or v["mean"] is None else (f"{v['mean']:.3f}" + (f"±{(v['hi'] - v['lo']) / 2:.3f}" if v["lo"] is not None else ""))


def table(res, key="auc", cols=YEARS, title=""):
    print(f"\n{title}\n{'cell':28s}" + "".join(f"{y:>12s}" for y in cols))
    for name, v in res.items():
        by = D.by_year(v["rows"], key)
        print(f"{name:28s}" + "".join(f"{f(by.get(y)):>12s}" for y in cols))


def main():
    q1, q2, q3 = ld("q1"), ld("q2"), ld("q3")
    summ = {}
    q1s = {k.replace("|60|logreg|0.0", ""): v for k, v in sorted(q1.items(), key=lambda kv: (D.BAR_MIN[kv[0].split("|")[0]], int(kv[0].split("|")[1])))}
    table(q1s, "auc", title="Q1 AUC per year, cell = bar|horizon in bars (logreg, 60 days, tb7)")
    table(q1s, "uni_auc_ret1", title="Q1 univariate AUC of the 1-bar return feature (>0.5 momentum, <0.5 reversal)")
    table(q1s, "uni_auc_bb", title="Q1 univariate AUC of BB %b (>0.5 momentum / trend, <0.5 mean reversion)")
    table(q1s, "coef_ret1", title="Q1 logistic coefficient on the 1-bar return (standardised features; + momentum)")
    table(q1s, "coef_bb", title="Q1 logistic coefficient on BB %b")
    q2s = {k.replace("1m|15|", "").replace("|0.0", ""): v for k, v in q2.items()}
    table({k: q2s[k] for k in sorted(q2s)}, "auc", cols=YEARS[4:], title="Q2 1m|15 min, AUC per year: train days|model")
    q3s = {k.replace("|60|logreg|", " k="): v for k, v in q3.items()}
    table(q3s, "auc", title="Q3 sign among |r| > k*sigma*sqrt(H), model trained on the same large moves")
    table(q3s, "auc_deadband_model", title="Q3 same test bars, model trained with the deadband target")
    table(q3s, "keep_frac", title="Q3 fraction of the month's bars kept")
    # Q4: pick on months up to 2024-06 (2021-01.. as the 'recent' selection window), show 2024-07..2025-09 once
    sel = ("2021-01", "2024-06"); hold = ("2024-07", "2025-07")
    cand = {**{f"Q1 {k}": v for k, v in q1s.items()}, **{f"Q2 {k}": v for k, v in q2s.items()}}
    sc = {k: D.tint([r["auc"] for r in v["rows"] if sel[0] <= r["month"] <= sel[1]])["mean"] for k, v in cand.items()}
    print(f"\nQ4 selection on {sel[0]}..{sel[1]} (mean monthly AUC), best 6 of {len(sc)} candidates; then the one-time look at {hold[0]}..{hold[1]}")
    top = sorted(sc, key=sc.get, reverse=True)[:6]
    for k in top + ["Q1 1m|15"]:
        h = D.tint([r["auc"] for r in cand[k]["rows"] if hold[0] <= r["month"] <= hold[1]])
        print(f"{k:28s} selection {sc[k]:.3f}   held-out {f(h)}")
        summ[k] = {"selection_mean": sc[k], "heldout": h}
    json.dump(summ, open(os.path.join(OUT, "q4_summary.json"), "w"), indent=1)


main()
