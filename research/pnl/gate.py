"""Dev gates and per-track selection (SPEC 'selection_and_preregistration' DEV GATE, SELECTION PER TRACK, 'objective'
INFERENCE, KILL CRITERIA K1). Reads $PNL_WORK/dev_daily.npz, dev_expo.npz and results/dev_runs.json.

G1 p_sel <= 0.05 (circular-shift null, max over the track's K candidates, studentised by HAC t)
G2 one-sided 95% stationary-bootstrap lower bound of the mean daily net at C7-B > 0
G3 point estimates beat every comparator (T1: mean daily net > plain fade, intraday trend;
   T2: Sharpe > B&H, TSMOM unscaled, vol-target long)
G4 net PnL at C7-B > 0 in >= 3 of 5 test years and in both halves
G5 net PnL at C7-S > 0
Deploy: the passing candidate with the highest mean daily net at C7-B, else CASH (information-only: highest HAC t).

  python gate.py   -> results/gate.json, results/gate.md
"""
import json
import os

import numpy as np
from scipy.stats import kurtosis, skew

import folds as FO
import prep
import stats as ST

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CANDIDATES = {1: ["T1-FADEV", "T1-LIN", "T1-NET", "T1-NT"], 2: ["T2-P", "T2-LIN", "T2-NET"]}
COMPARATORS = {1: ["plain_fade", "intraday_trend"], 2: ["B&H", "tsmom_unscaled", "voltarget_long"]}
N_TRIALS = {1: 64, 2: 117}
BLOCK = {1: 10, 2: 20}
B = 10_000
SCEN6 = ["C7B", "C7S", "C4B", "C4S", "C10B", "C10S"]
SEEN_DEV = (FO.ts(2025, 10, 11), FO.SEAL_START)


def year_of(days):
    return (np.asarray(days).astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970)


def ann(x):
    return float(np.mean(x) * 365 * 1e4)


def sharpe(x):
    sd = np.std(x)
    return float(np.mean(x) / sd * np.sqrt(365)) if sd > 0 else 0.0


def main():
    dd = dict(np.load(os.path.join(prep.WORK, "dev_daily.npz")))
    ex = dict(np.load(os.path.join(prep.WORK, "dev_expo.npz")))
    runs = json.load(open(os.path.join(RESULTS, "dev_runs.json")))
    st = prep.Store()
    s, e = st.idx(FO.DEV_OOS[0]), st.n
    r1 = np.r_[0.0, np.diff(st.lc)][s:e]
    f1 = st.b.funding[s:e]
    day = st.day[s:e]
    days = dd["days"]
    yrs = year_of(days)
    tsd = days * 86400
    halves = [(tsd >= a) & (tsd < b) for a, b in FO.HALVES]
    sub22 = tsd >= FO.ts(2022)
    unseen = ~((tsd >= SEEN_DEV[0]) & (tsd < SEEN_DEV[1]))
    reruns = {}
    for line in open(os.path.join(RESULTS, "runs.jsonl")):
        j = json.loads(line)
        if j["scenario"] == "C7B":
            reruns.setdefault(j["name"], set()).add(j["config_sha"])
    out = {"tracks": {}}
    md = ["# Dev gates (dev OOS 2021-01-01..2025-10-31, engine E)", ""]
    for track in (1, 2):
        cands = [c for c in CANDIDATES[track] if f"{c}|C7B" in dd and c in ex]
        missing = [c for c in CANDIDATES[track] if c not in cands]
        X = np.stack([ex[c] for c in cands])
        cost = np.stack([dd[f"fees|{c}"] for c in cands])          # engine-E fees per day (shift-invariant)
        ps = ST.p_sel(X, r1, f1, day, n_shifts=999, seed=track, tstat="hac", lag=BLOCK[track], daily_cost=cost)
        comps = {c: dd[f"{c}|C7B"] for c in COMPARATORS[track]}
        trk = {"candidates": {}, "missing": missing, "p_sel": ps["p_sel"], "t_obs_engineF": dict(zip(cands, ps["t_obs"].tolist())),
               "p_single": dict(zip(cands, ps["p_single"].tolist())), "best_by_t": cands[ps["best"]],
               "comparators": {c: {"mean_daily_bps": float(np.mean(v) * 1e4), "ann_bps": ann(v), "sharpe": sharpe(v)}
                               for c, v in comps.items()}}
        n_trials = N_TRIALS[track] + sum(max(0, len(reruns.get(c, ())) - 1) for c in cands)
        srs = [np.mean(dd[f"{c}|C7B"]) / np.std(dd[f"{c}|C7B"]) for c in cands + COMPARATORS[track]
               if np.std(dd[f"{c}|C7B"]) > 0]
        for c in cands:
            d = dd[f"{c}|C7B"]
            T = len(d)
            dist = ST.stationary_bootstrap(d, B=B, mean_block=BLOCK[track], seed=11)
            lb = ST.lower_bound(dist)
            by_year = {int(y): float(d[yrs == y].sum()) for y in np.unique(yrs)}
            half = [float(d[m].sum()) for m in halves]
            g1 = ps["p_sel"] <= 0.05
            g2 = lb > 0
            if track == 1:
                g3 = all(np.mean(d) > np.mean(v) for v in comps.values())
            else:
                g3 = all(sharpe(d) > sharpe(v) for v in comps.values())
            g4 = sum(v > 0 for v in by_year.values()) >= 3 and all(h > 0 for h in half)
            g5 = float(dd[f"{c}|C7S"].sum()) > 0
            sr_d = float(np.mean(d) / np.std(d)) if np.std(d) > 0 else 0.0
            var_sr = max(1.0 / T, float(np.var(srs)) if len(srs) > 1 else 0.0)
            info = {
                "gates": {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4), "G5": bool(g5)},
                "pass": bool(g1 and g2 and g3 and g4 and g5),
                "mean_daily_bps": float(np.mean(d) * 1e4), "ann_bps": ann(d), "sharpe": sharpe(d),
                "boot_lb95_daily_bps": lb * 1e4, "boot_ci95_daily_bps": [v * 1e4 for v in ST.percentile_ci(dist)],
                "boot_p": ST.boot_pvalue(dist), "hac_t": float(ST.newey_west_t(d, BLOCK[track])),
                "deflated_sharpe": ST.deflated_sharpe(sr_d, n_trials, var_sr, float(skew(d)),
                                                      float(kurtosis(d, fisher=False)), T),
                "n_trials": n_trials, "by_year_bps": {k: v * 1e4 for k, v in by_year.items()},
                "halves_bps": [h * 1e4 for h in half],
                "scenarios_ann_bps": {sc: ann(dd[f"{c}|{sc}"]) for sc in SCEN6 + ["C10N"] if f"{c}|{sc}" in dd},
                "sub_2022_on_ann_bps": ann(d[sub22]), "dev_without_seen_ann_bps": ann(d[unseen]),
                "mean_x": float(np.mean(ex[c])), "mean_abs_x": float(np.mean(np.abs(ex[c]))),
                "p_single_shift": float(ps["p_single"][cands.index(c)]),
            }
            # paired excess over cash (= the series), B&H, static-matched and the comparators
            base = {"B&H": dd.get("B&H|C7B"), f"static[{c}]": dd.get(f"static[{c}]|C7B")}
            base.update(comps)
            info["paired"] = {}
            for bn, bv in base.items():
                if bv is None:
                    continue
                pdist = ST.paired_bootstrap(d, bv, B=B, mean_block=BLOCK[track], seed=12)
                info["paired"][bn] = {"mean_diff_ann_bps": float(np.mean(d - bv) * 365 * 1e4),
                                      "ci95_ann_bps": [v * 365 * 1e4 for v in ST.percentile_ci(pdist)],
                                      "P_candidate_better": ST.prob_positive(pdist)}
            tr = dd.get(f"trades|{c}")
            if tr is not None and len(tr) >= 20:
                tdist = ST.stationary_bootstrap(tr, B=B, mean_block=20, seed=13)
                info["trades"] = {"n": int(len(tr)), "mean_net_bps": float(tr.mean()),
                                  "ci95_net_bps": list(ST.percentile_ci(tdist)), "ub95_net_bps": ST.upper_bound(tdist),
                                  "mean_gross_bps": float(dd[f"trades_gross|{c}"].mean())}
            elif tr is not None:
                info["trades"] = {"n": int(len(tr)), "mean_net_bps": float(tr.mean()) if len(tr) else None}
            trk["candidates"][c] = info
        passing = [c for c in cands if trk["candidates"][c]["pass"]]
        if passing:
            dep = max(passing, key=lambda c: trk["candidates"][c]["mean_daily_bps"])
            trk["deployed"], trk["information_only"] = dep, None
        else:
            trk["deployed"] = "CASH"
            trk["information_only"] = max(cands, key=lambda c: trk["candidates"][c]["hac_t"])
        if track == 1:
            ubs = [trk["candidates"][c].get("trades", {}).get("ub95_net_bps") for c in cands]
            trk["K1_track1_closed"] = all(u is not None and u < 0 for u in ubs)
        out["tracks"][track] = trk
        md += [f"## Track {track}", "", f"p_sel = {ps['p_sel']:.3f} (K = {len(cands)}; best by t: {trk['best_by_t']})"
               + (f"; missing: {missing}" if missing else ""), "",
               "| candidate | ann. net C7-B (bps) | Sharpe | 95% LB daily (bps) | HAC t | G1 | G2 | G3 | G4 | G5 | pass |",
               "|---|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|"]
        for c in cands:
            i = trk["candidates"][c]
            g = i["gates"]
            md.append(f"| {c} | {i['ann_bps']:.0f} | {i['sharpe']:.2f} | {i['boot_lb95_daily_bps']:.2f} | "
                      f"{i['hac_t']:.2f} | " + " | ".join("Y" if g[k] else "n" for k in ("G1", "G2", "G3", "G4", "G5"))
                      + f" | {'PASS' if i['pass'] else '-'} |")
        md += ["", f"Deployed: **{trk['deployed']}**; information-only: {trk['information_only']}", ""]
    json.dump(out, open(os.path.join(RESULTS, "gate.json"), "w"), indent=1, default=float)
    open(os.path.join(RESULTS, "gate.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return out


if __name__ == "__main__":
    main()
