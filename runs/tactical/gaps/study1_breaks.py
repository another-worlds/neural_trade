"""Study 1, step 2 (SPEC.md): breaks vs a linear trend in the monthly AUC and HAR-Spearman series, the same search on the market's monthly
log notional volume and log realised variance, and the event checks from the daily data. usage: python study1_breaks.py"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gaps_lib as G  # noqa: E402

G.nice(); t0 = time.time()
daily = pd.read_csv(f"{HERE}/daily.csv", index_col=0, parse_dates=True)
SERIES = {}
for nm, f, col in (("S_auc", "series_auc.json", "auc"), ("S_har", "series_har.json", "har")):
    d = pd.DataFrame(json.load(open(f"{HERE}/{f}"))).set_index("month")[col]; SERIES[nm] = d
mon = daily.copy(); mon.index = mon.index.to_period("M")
mv = mon.groupby(level=0).agg(notional=("notional", "sum"), rv=("rv", "sum"))
mv = mv[(mv.notional > 0) & (mv.rv > 0)]
SERIES["M_logvol"] = np.log(mv["notional"]); SERIES["M_logrv"] = np.log(mv["rv"])
SERIES = {k: v.rename(k) for k, v in SERIES.items()}
SERIES["M_logvol"] = SERIES["M_logvol"][[str(i) >= "2017-04" for i in SERIES["M_logvol"].index]]
SERIES["M_logrv"] = SERIES["M_logrv"][[str(i) >= "2017-04" for i in SERIES["M_logrv"].index]]


def date_of(months, i):
    return str(months[int(round(i))]) if 0 <= int(round(i)) < len(months) else "?"


def analyse(name, ser, ex_july=False):
    ser = ser.copy(); ser.index = [str(i) for i in ser.index]
    if ex_july:
        ser = ser[ser.index <= "2025-06"]
    y = ser.values.astype(float); months = list(ser.index); T = len(y)
    sel = G.select_k(y); k = sel["kstar"]; br = sel["breaks"]
    supF, tau, p = G.chow_pvalue(y, nboot=2000)
    ci = G.break_intervals(y, k, br, nboot=1000)
    f, rss_t, b = G.trend_fit(y)
    supported = bool(sel["bic_k"][k] + 2 < sel["bic_trend"] and p < 0.05)
    segs = []
    edges = [0] + br + [T]
    for a, c in zip(edges[:-1], edges[1:]):
        segs.append({"from": months[a], "to": months[c - 1], "mean": float(y[a:c].mean()), "n": c - a})
    return {"series": name, "T": T, "ex_july": ex_july, "kstar": k, "breaks": [{"first_month_of_new_regime": months[x],
            "ci95_months": [date_of(months, lo), date_of(months, hi)]} for x, (lo, hi) in zip(br, ci)] if k else [],
            "segments": segs, "bic_k": {str(a): float(v) for a, v in sel["bic_k"].items()}, "bic_trend": float(sel["bic_trend"]),
            "trend_slope_per_month": float(b[1]), "sup_chow_F": supF, "sup_chow_tau": months[tau], "sup_chow_p": p, "breaks_supported": supported}


results = []
for nm, ser in SERIES.items():
    for ex in (False, True):
        r = analyse(nm, ser, ex); results.append(r)
        print(nm, "ex_july" if ex else "all", "k*", r["kstar"], "supF p", round(r["sup_chow_p"], 4), "supported", r["breaks_supported"], f"{time.time() - t0:.0f}s", flush=True)

# ---- events: ratio of 14-day medians after/before, percentile among all dates
dd = daily[daily.index >= "2017-06-01"].copy(); dd["rv"] = dd["rv"].where(dd["rv"] > 0)
def ratios(col):
    x = dd[col]; after = x[::-1].rolling(14, min_periods=14).median()[::-1].shift(-1)   # median of d+1..d+14
    before = x.rolling(14, min_periods=14).median()                                       # median of d-13..d
    return (after / before)
EV = [("zero-fee start (memory)", "2022-07-21"), ("zero-fee end (memory)", "2023-03-31"), ("FTX collapse (memory)", "2022-11-08"),
      ("COVID crash (memory)", "2020-03-12"), ("China mining ban (memory)", "2021-05-19"), ("halving 2020 (memory)", "2020-05-11"),
      ("halving 2024 (memory)", "2024-04-20"), ("spot-ETF launch (memory)", "2024-01-11"), ("Terra/Luna (memory)", "2022-05-09")]
ev_rows = []
for col in ("notional", "rv"):
    R = np.log(ratios(col)).dropna()
    for nm, d in EV:
        d = pd.Timestamp(d)
        if d not in R.index:
            continue
        v = R[d]; pct = float((R.abs() <= abs(v)).mean() * 100)          # percentile of |log ratio|
        ev_rows.append({"event": nm, "date": str(d.date()), "metric": col, "ratio_after_before": float(np.exp(v)), "abs_percentile": pct,
                        "data_verified": bool(pct >= 95), "label": "data-verified" if pct >= 95 else "unverified in the data (memory date)"})
    # the data's own largest jumps near the zero-fee window
    for nm, lo, hi in (("zero-fee start window", "2022-05-22", "2022-09-19"), ("zero-fee end window", "2023-01-30", "2023-05-31")):
        w = R[lo:hi]; dmax = w.abs().idxmax(); ev_rows.append({"event": nm + " (data's own largest |jump|)", "date": str(dmax.date()), "metric": col,
            "ratio_after_before": float(np.exp(w[dmax])), "abs_percentile": float((R.abs() <= abs(w[dmax])).mean() * 100),
            "data_verified": bool((R.abs() <= abs(w[dmax])).mean() >= 0.95), "label": "data date"})
json.dump({"results": results, "events": ev_rows}, open(f"{HERE}/study1_results.json", "w"), indent=1)

# ---- coincidence table (events and data dates inside a break's 95% interval widened by one month)
def to_ts(m): return pd.Period(m, "M").start_time
cand = [(e["event"], pd.Timestamp(e["date"]), e["label"], e["metric"]) for e in ev_rows if e["data_verified"] or "memory" in e["event"] or e["label"] == "data date"]
lines = []
for r in results:
    if r["ex_july"]:
        continue
    lines.append(f"\n== {r['series']} (T={r['T']}): k*={r['kstar']}, BIC k* {r['bic_k'][str(r['kstar'])]:.1f} vs trend {r['bic_trend']:.1f} "
                 f"(delta {r['bic_trend'] - r['bic_k'][str(r['kstar'])]:+.1f}), sup-Chow F={r['sup_chow_F']:.2f} at {r['sup_chow_tau']}, p={r['sup_chow_p']:.4f}, "
                 f"breaks supported: {r['breaks_supported']}; trend slope {r['trend_slope_per_month']:+.5f}/month")
    for s_ in r["segments"]:
        lines.append(f"   regime {s_['from']}..{s_['to']} (n={s_['n']}): mean {s_['mean']:.4f}")
    for b in r["breaks"]:
        lo, hi = to_ts(b["ci95_months"][0]) - pd.Timedelta(days=31), to_ts(b["ci95_months"][1]) + pd.Timedelta(days=62)
        hits = sorted({f"{c[0]} {c[1].date()} [{c[3]}; {c[2]}]" for c in cand if lo <= c[1] <= hi})
        lines.append(f"   break -> new regime from {b['first_month_of_new_regime']}, 95% interval {b['ci95_months'][0]}..{b['ci95_months'][1]}; events inside (+-1 month): "
                     + ("; ".join(hits) if hits else "none"))
lines.append("\n== event checks (14-day median after / before, percentile of |log ratio| over all dates) ==")
for e in ev_rows:
    lines.append(f"  {e['event']:55s} {e['date']} {e['metric']:9s} ratio {e['ratio_after_before']:.2f} pct {e['abs_percentile']:.1f}  {e['label']}")
lines.append(f"\nrun time {time.time() - t0:.0f}s")
open(f"{HERE}/study1_tables.txt", "w").write("\n".join(lines)); print("\n".join(lines))
