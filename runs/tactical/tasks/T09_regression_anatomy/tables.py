"""T09: the result tables as text (tables.txt), from results_minute.json, results_forward.json, results_hourly.json, sanity.json.
usage: python tables.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import lib

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
M = json.load(open(os.path.join(HERE, "results_minute.json"))); Fw = json.load(open(os.path.join(HERE, "results_forward.json")))
Hr = json.load(open(os.path.join(HERE, "results_hourly.json"))); S = json.load(open(os.path.join(HERE, "sanity.json")))
OUT = []


def P(*a):
    OUT.append(" ".join(str(x) for x in a))


def pm(t, d=4):
    """mean with the half-width of the 95% t-interval over months."""
    if t is None or t.get("mean") is None:
        return "n/a"
    hw = (t["hi"] - t["lo"]) / 2 if t.get("lo") is not None else float("nan")
    return f"{t['mean']:.{d}f}±{hw:.{d}f}"


def mean_only(t, d=4):
    return "n/a" if t is None or t.get("mean") is None else f"{t['mean']:+.{d}f}"


def sec(title):
    P(""); P("=" * 110); P(title); P("=" * 110)


def tab(df):
    P(df.to_string())


def feat_year_table(q, key, f=pm, names=None):
    names = names or q["features"]
    return pd.DataFrame({y: [f(q[key][nm][y]) for nm in names] for y in q["years"]}, index=names)


sec("SANITY: monthly tb7 AUC of this study against the stored H51 (close) and H59 (next-open) values (months 2017-04..2025-06)")
for e, ref in (("close", "H51 decay.json"), ("nopen", "H59 decay_nextopen.json")):
    s = S[e]; P(f"{e} vs {ref}: {s['n_months']} common months, max |diff| {s['max_abs_diff']:.2e} (worst {s['worst_month']}), mean diff {s['mean_diff']:.1e}, "
                f"months over 0.001: {s['n_over_0.001']}; mean AUC mine {s['mean_auc_mine']:.4f} stored {s['mean_auc_stored']:.4f}")
P("(differences below 1e-6 come from the stricter purge: training anchors must have their whole label, exit-bar close included, finished before the month starts)")
P(f"anchors {M['n_anchors']}, months {len(M['months'])} ({M['months'][0]}..{M['months'][-1]}; 2025-07 is partial, to 07-20)")

sec("DATA NOTES (coordinator note 2026-10-14): the cut file is Bitstamp BTC/USD, not Binance; the forward file (Q9) is Binance BTCUSDT")
P("File timestamps are UTC+4 during US daylight saving and UTC+5 otherwise until the splice at 2024-09-13 08:00 (file time); after it they are UTC. "
  "Hour of day (Q5) is therefore CONVERTED to UTC here (file hour minus 4 or 5, by US daylight-saving date; minus 0 after the splice); anchors on the DST-change days and around the splice are left out of the hour split.")
P("Seam anchors = windows or labels touching a whole DST-change day (second Sunday of March, first Sunday of November) or +-2 h around the splice, plus 61 minutes of margin: "
  f"{M['seam']['n_seam_anchors']} of {M['n_anchors']} anchors ({M['seam']['share']:.2%}). They are EXCLUDED from Q2 (partial dependence and tails) and Q3 (impulse response); "
  "they are INCLUDED in the monthly regressions (needed to reproduce H51 / H59) and in Q1, Q4-Q7.")
P("Effect of leaving them out: max |change| of any impulse AUC cell", {k: round(v, 4) for k, v in M["seam"]["impulse_auc_max_abs_diff_incl_vs_excl"].items()}, "mean |change|",
  {k: round(v, 5) for k, v in M["seam"]["impulse_auc_mean_abs_diff"].items()})
P("Yearly regression AUC with the seam anchors left out of the test sets (compare Q1 / Q4 tables; identical to the 3rd decimal):")
tab(pd.DataFrame({e: {y: pm(t) for y, t in M["seam"]["yearly_auc_without_seam_anchors"][e].items()} for e in M["seam"]["yearly_auc_without_seam_anchors"]}).T)
P("Other data events: Bitstamp tick $0.01 -> $1 on 2022-08-10 (the Corwin-Schultz estimate is 0 on most bars from 2022; zero-return minutes rise from 3-4% to 12-31%), "
  "the splice 2024-09-13 (about 4 h missing, candle construction changes), a 1,160-minute gap on 2025-03-15. Yearly 2022 and 2024 values contain these changes.")
for e in ("close", "nopen"):
    q = M["q1"][e]
    sec(f"Q1 [{e} entry] yearly mean over months (± 95% t half-width): AUC of the monthly regression")
    P(pd.Series({y: pm(q["auc"][y]) for y in q["years"]}).to_string())
    P("\nstandardised coefficient (feature × year; the 13 features are z-scored with the training month's mean/sd)")
    tab(feat_year_table(q, "coef", mean_only, q["features"]))
    P("\nunivariate AUC of the raw feature against the label on the month's own data (< 0.5 = reversal: a high value precedes a fall; > 0.5 = momentum)")
    tab(feat_year_table(q, "uni"))
    P("\ndrop-one AUC loss (refit without the feature; ~0 for all because the features are collinear)")
    tab(feat_year_table(q, "drop"))
    P("\npermutation AUC loss of one feature (the fitted model, that column shuffled on the month's own test data, mean of 5 shuffles)")
    tab(feat_year_table(q, "perm"))
    P("\nGROUPED permutation loss (returns = ret1..ret30; position = ema20, bb20, stoch14, cci20; oscillator = rsi14, macd; volatility = atr14)")
    tab(pd.DataFrame({y: [pm(q["gperm"][g][y]) for g in q["gperm"]] for y in q["years"]}, index=list(q["gperm"])))
    P("\ngroup drop-one AUC loss (refit without the whole group)")
    tab(pd.DataFrame({y: [pm(q["gdrop"][g][y]) for g in q["gdrop"]] for y in q["years"]}, index=list(q["gdrop"])))
    P("\nAUC of a model that uses ONLY the group")
    tab(pd.DataFrame({y: [pm(q["gonly"][g][y]) for g in q["gonly"]] for y in q["years"]}, index=list(q["gonly"])))
    P("\ntop-3 features by drop-one loss, per year:", json.dumps(q["top3_by_year"]))
    P("rank of the features by drop-one loss over all months:", q.get("rank_all_months"))
    P("Spearman correlation between the years' drop-one rankings:")
    tab(pd.DataFrame(q["rank_spearman"]["matrix"], index=q["rank_spearman"]["years"], columns=q["rank_spearman"]["years"]))

sec("Q1 subsets: monthly walk-forward restricted to the top-1/2/3 features (ranked by drop-one loss over all months, close entry)")
rows = {}
for n, v in M["q1_subsets"]["close"].items():
    rows[f"top{n}: {'+'.join(v['features'])}"] = {y: pm(a) for y, a in v["auc_by_year"].items()}
rows["all 13"] = {y: pm(M["q1"]["close"]["auc"][y]) for y in M["q1"]["close"]["years"]}
tab(pd.DataFrame(rows).T)
P("share of the full model's edge ((AUC-0.5)/(full AUC-0.5)):")
sh = {}
for n, v in M["q1_subsets"]["close"].items():
    sh[f"top{n}"] = {y: (a["mean"] - 0.5) / (M["q1"]["close"]["auc"][y]["mean"] - 0.5) for y, a in v["auc_by_year"].items()}
tab(pd.DataFrame(sh).T.round(2))

sec("Q2 partial dependence (close entry): 20 quantile bins per era. p_up = empirical share of up moves among deadband-masked anchors (95% Wilson interval, "
    "n_eff = n*10/15); pd_model = the monthly models' average P(up) with the feature set to the bin median")
for e in ("close",):
    for era, fs in M["q2"][e].items():
        for f, v in fs.items():
            P(f"\n[{e}] era {era}, feature {f}: n {v['n']}, mean P(up) {v['mean_p_up']:.4f}, Spearman(bin, p_up) {v['spearman']:+.3f}, R2 of a straight line through the bin means "
              f"{v['r2_linear']:.3f}, slope lower/upper half {v['slope_lower_half']:+.4f}/{v['slope_upper_half']:+.4f} per unit, edge of the 3 lowest bins "
              f"{v['edge_low_tail']:+.4f} (P(up) above mean: a fall precedes a rise), of the 3 highest bins {v['edge_high_tail']:+.4f} (P(up) below mean), "
              f"mean |edge| ends {v['edge_ends_vs_middle']['ends']:.4f} vs middle 6 bins {v['edge_ends_vs_middle']['middle']:.4f}")
            tab(pd.DataFrame(v["bins"])[["bin", "x_median", "n", "p_up", "lo", "hi", "pd_model"]].round(4))
            P("fade hit rate (sign(-x) vs the move) by |x| decile:", [round(x["fade_hit"], 4) for x in v["tails"]])

for e in ("close", "nopen"):
    sec(f"Q3 [{e} entry] impulse response by year: AUC of fading the last k-minute return against the sign of the h-minute move (rows k, columns h; "
        "AUC > 0.5 = reversal); ± = 95% t half-width over the year's months")
    ci = M["q3"]["auc_halfwidth"][e]
    for y, v in M["q3"][e]["by_year_auc"].items():
        P(f"\n{y}  (cell half-widths: k = 1 row {np.round(ci[y][0], 3).tolist()}, k = 60 row {np.round(ci[y][-1], 3).tolist()})")
        tab(pd.DataFrame(np.round(np.array(v), 4), index=[f"k={k}" for k in lib.KS], columns=[f"h={h}" for h in lib.HS]))
    sec(f"Q3 [{e} entry] mean signed return of fading the last k minutes (bps per signal, all anchors; rows k, columns h)")
    for y, v in M["q3"][e]["by_year_bps"].items():
        P(f"\n{y}"); tab(pd.DataFrame(np.round(np.array(v), 3), index=[f"k={k}" for k in lib.KS], columns=[f"h={h}" for h in lib.HS]))
    sec(f"Q3 [{e} entry] the best simple rule 'fade the last k minutes' at h = 15 against the regression, per year")
    rs = M["q3"][e]["rule_share"]
    tab(pd.DataFrame({y: {"regression AUC": v["regression_auc"], "best k (in-year)": v["best_k"], "best-k AUC": v["best_auc"], "share of regression edge": v["share_best"],
                          "k=1 AUC": v["k1_auc"], "k=1 share": v["share_k1"], "k chosen by previous year": v.get("prev_year_k"),
                          "causal AUC": v.get("causal_auc"), "causal share": v.get("share_causal")} for y, v in rs.items()}).T.round(4))

sec("Q4 microstructure: yearly regression AUC by entry (close, next open, mid of the next bar's high/low), ± 95% half-width over months")
tab(pd.DataFrame({e: {y: pm(t) for y, t in M["q4"]["auc_by_year"][e].items()} for e in M["q4"]["auc_by_year"]}).T)
P("\nclose-entry model scored against the NEXT-OPEN labels (what survives when you must enter at the next open):")
P(pd.Series({y: pm(t) for y, t in M["q4"]["close_model_on_nopen_labels"].items()}).to_string())
P("\nspread / bounce proxies by year (means of monthly values): cs = Corwin-Schultz two-bar estimate (bps), hl = median 1-minute high-low range (bps), "
  "roll = Roll estimate from the serial covariance of 1-minute returns (bps), ac1 = lag-1 autocorrelation, vr = variance ratio VR(1,15), volume = mean BTC per bar")
tab(pd.DataFrame(M["q4"]["roll_and_cs_by_year"]).T.round(3))

for e in ("close", "nopen"):
    for c in ("hl", "cs", "vol", "volume", "move1bps", "move1sig", "weekend"):
        sec(f"Q4/Q5 [{e} entry] AUC by {c} bin, by period (bins from the training span's quantiles; 95% t-interval over months), with the mean signed return "
            "of trading every anchor on the model's side (bps) and the top-10% trades (bps, pooled)")
        df = pd.DataFrame(M["q5_period"][e]); d = df[df.cond == c]
        rows = []
        for b in dict.fromkeys(d["bin"]):
            r = {"bin": b}
            for g in ("2017-19", "2020-23", "2024-25"):
                x = d[(d["bin"] == b) & (d["grp"] == g)]
                if len(x):
                    x = x.iloc[0]; r[f"AUC {g}"] = f"{x['auc']:.4f}±{(x['auc_hi'] - x['auc_lo']) / 2:.4f}"; r[f"bps_all {g}"] = round(x["bps_all"], 3); r[f"bps_top10 {g}"] = round(x["bps_top"], 3)
            rows.append(r)
        tab(pd.DataFrame(rows).set_index("bin"))
    sec(f"Q5 [{e} entry] AUC by hour of day (UTC), by period")
    df = pd.DataFrame(M["q5_period"][e]); d = df[df.cond == "hour"]
    tab(d.pivot_table(index="grp", columns="bin", values="auc").round(3))
    sec(f"Q5 [{e} entry] AUC by condition and YEAR (vol, volume, hl, move1sig, weekend)")
    d = pd.DataFrame(M["q5_year"][e])
    for c in ("vol", "volume", "hl", "move1sig", "weekend"):
        P(f"\n{c}"); tab(d[d.cond == c].pivot_table(index="bin", columns="grp", values="auc", sort=False).round(3))

sec("Q6 decay decomposition (waterfall vs the base year 2019, monthly AUC averaged per year). total = AUC(year) - AUC(2019), all with monthly refits. "
    "coef = change in the gain of refitting over the stable 2019 coefficients; cond = change from the volatility x volume (or x high-low) mix of the base year; "
    "strength = what remains: the SAME coefficients on the SAME mix predict less. Sum of the three = total.")
for e in ("close", "nopen"):
    for k, title in (("waterfall", "cells = volatility tercile x volume tercile (2019 cuts)"), ("waterfall_vol_x_hl", "cells = volatility tercile x high-low-range tercile (2019 cuts)")):
        P(f"\n[{e}] {title}")
        w = M["q6"][e][k]; tab(pd.DataFrame(w).T[["full", "fixed", "fixed_rew", "total", "strength", "cond", "coef", "cosine_to_base"]].round(4))
P("\nthe stable 2019 coefficients (close):", dict(zip(lib.FEAT_NAMES, np.round(M["q6"]["close"]["fixed_coef"], 3))))
sec("Q6 market-quality proxies against the monthly AUC (100 months): Spearman, and Pearson after removing a linear time trend from both (n = 100; months are autocorrelated)")
for e in ("close", "nopen"):
    P(f"\n[{e}]"); tab(pd.DataFrame(M["q6"][f"correlations_{e}"]).T.round(3))

sec("Q7 economic size: top-10% most confident anchors per month (threshold from the training predictions), gross bps per trade, yearly mean ± 95% half-width "
    "over months; no costs. 'mid' = nopen model traded at the mid of the next bar; 'close_model_at_nopen' = the close model entered at the next open.")
rows = {}
for k in ("bps_close", "bps_nopen", "bps_mid", "bps_close_model_at_nopen", "hit_close", "hit_nopen", "cs_trades", "hl_trades", "cs_all", "hl_all"):
    rows[k] = {y: pm(t, 3) for y, t in M["q7"][k].items()}
tab(pd.DataFrame(rows).T)
P("\nBREAK-EVEN round-trip cost = the mean gross bps per trade. Gross bps divided by the Corwin-Schultz spread of the traded bars is undefined where CS = 0 (from 2022).")
P("trades per month:", {y: round(t["mean"]) for y, t in M["q7"]["n_close"].items()})

sec("Q8 the hourly cousin (H62 rich logistic, 43 features, 4-hour horizon, next-open entry, months 2019-01..2025-06)")
P(f"months {Hr['n_months']}; cross-check against recent/wf.py walk_forward on the same inputs: {Hr['crosscheck_vs_wf']}")
q = Hr["q1"]
P("\nyearly AUC:"); P(pd.Series({y: pm(q["auc"][y]) for y in q["years"]}).to_string())
P("\nfamily permutation loss / group-only AUC (yearly)")
tab(pd.DataFrame({y: [pm(q["gperm"][g][y]) for g in q["groups"]] for y in q["years"]}, index=q["groups"]))
tab(pd.DataFrame({y: [pm(q["gonly"][g][y]) for g in q["groups"]] for y in q["years"]}, index=q["groups"]))
P("\nfamily drop-one loss (yearly)"); tab(pd.DataFrame({y: [pm(q["gdrop"][g][y]) for g in q["groups"]] for y in q["years"]}, index=q["groups"]))
am = q["all_months_mean"]
top = sorted(q["features"], key=lambda f: -am["perm"][f])[:15]
P("\ntop-15 features by permutation loss over all months: perm, univariate AUC (< 0.5 = reversal), mean standardised coefficient, drop-one")
tab(pd.DataFrame({f: {"perm": am["perm"][f], "uni_auc": am["uni"][f], "coef": am["coef"][f], "drop": am["drop"][f]} for f in top}).T.round(4))
P("\nunivariate AUC by year of every feature with |AUC-0.5| above 0.03 in some year")
sel = [f for f in q["features"] if max(abs(q["uni"][f][y]["mean"] - 0.5) for y in q["years"]) > 0.03]
tab(pd.DataFrame({y: [pm(q["uni"][f][y]) for f in sel] for y in q["years"]}, index=sel))
for e in ("nopen", "close"):
    r3 = Hr["q3"][e]
    for y, v in {"ALL YEARS": r3["all"], **r3["by_year"]}.items():
        P(f"\n[{e}] impulse AUC {y} (rows: fade the last k hours, columns: h hours ahead)" + ("" if y == "ALL YEARS" else f"; half-width k=1 row {np.round(v['auc_halfwidth'][0], 3).tolist()}"))
        tab(pd.DataFrame(np.round(np.array(v["auc"]), 4), index=[f"k={k}" for k in r3["ks"]], columns=[f"h={h}" for h in r3["hs"]]))
        if e == "nopen":
            P("mean signed bps"); tab(pd.DataFrame(np.round(np.array(v["bps"]), 2), index=[f"k={k}" for k in r3["ks"]], columns=[f"h={h}" for h in r3["hs"]]))
P("\nbest simple fade rule at the 4-hour horizon vs the regression, per year"); tab(pd.DataFrame(Hr["q3"]["rule_share_nopen_h4"]).T.round(4))

sec("Q9 the present: Binance forward year, scored months 2025-11..2026-09 (training only on 2025-10-01 onward; the first month trains on about 30 days)")
P("monthly AUC:"); tab(pd.DataFrame({e: {r["month"]: round(r["auc"], 4) for r in Fw["monthly"][e]} for e in Fw["monthly"]}).T)
P("\nmean AUC (± 95% half-width over the 11 months):", {e: pm(t) for e, t in Fw["q4"]["auc"].items()}, " close model on next-open labels:", pm(Fw["q4"]["close_model_on_nopen_labels"]))
for e in ("close", "nopen"):
    q = Fw["q1"][e]
    P(f"\n[{e}] Q1 features (mean over months): coef, univariate AUC, drop-one, permutation")
    tab(pd.DataFrame({nm: {"coef": q["coef"][nm]["mean"], "uni_auc": pm(q["uni"][nm]), "drop": pm(q["drop"][nm]), "perm": pm(q["perm"][nm])} for nm in q["features"]}).T)
    tab(pd.DataFrame({g: {"gperm": pm(q["gperm"][g]), "gdrop": pm(q["gdrop"][g]), "gonly_auc": pm(q["gonly"][g])} for g in q["gperm"]}).T)
for e in ("close", "nopen", "mid"):
    r3 = Fw["q3"][e]
    P(f"\n[{e}] Q3 impulse AUC over the forward months (rows k, columns h); half-width of the k=1 row {np.round(r3['auc_halfwidth'][0], 3).tolist()}, k=60 row {np.round(r3['auc_halfwidth'][-1], 3).tolist()}")
    tab(pd.DataFrame(np.round(np.array(r3["auc"]), 4), index=[f"k={k}" for k in lib.KS], columns=[f"h={h}" for h in lib.HS]))
    if e == "nopen":
        P("mean signed bps (nopen)"); tab(pd.DataFrame(np.round(np.array(r3["bps"]), 3), index=[f"k={k}" for k in lib.KS], columns=[f"h={h}" for h in lib.HS]))
P("\nbest simple rule vs regression (calendar halves of the forward year):"); tab(pd.DataFrame({e: {y: {"reg": v["regression_auc"], "best_k": v["best_k"], "best_auc": v["best_auc"], "k1_auc": v["k1_auc"]} for y, v in Fw["q3"]["rule_share"][e].items()} for e in Fw["q3"]["rule_share"]}).T)
P("\nQ4 by condition (all forward months, AUC with half-width; bps_all = every anchor on the model's side)")
for e in ("close", "nopen"):
    df = pd.DataFrame(Fw["q4"]["by_spread_proxy"][e]); df["AUC"] = [f"{a:.4f}±{(h - l) / 2:.4f}" for a, l, h in zip(df["auc"], df["auc_lo"], df["auc_hi"])]
    P(f"\n[{e}]"); tab(df[["cond", "bin", "AUC", "n_months", "bps_all", "bps_top", "n_top"]].round(3).set_index(["cond", "bin"]))
P("\nQ7 forward: top-10% gross bps per trade (95% t-interval over 11 months):", {k: pm(v, 3) for k, v in Fw["q7"].items()})
P("\nmarket-quality proxies by month:"); tab(pd.DataFrame(Fw["q4"]["market_quality"]).T.round(3))
P("\ncorrelation of monthly AUC with the proxies (n = 11):"); tab(pd.DataFrame(Fw["corr"]).T.round(3))

open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
print(len(OUT), "lines")
