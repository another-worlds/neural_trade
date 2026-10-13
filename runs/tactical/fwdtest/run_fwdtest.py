"""Run the forward test ONCE (AMENDMENT.md): frozen arm (verdict) and diagnostic arm (scaler re-fitted on 2025-10). Writes fwdtest_result.json/.txt."""
import json, time
import numpy as np, pandas as pd
from scipy import stats as sps
import fwdtest_lib as T
import eqlib as E
import os; os.chdir(T.HERE)
t0 = time.time()
mdl, sc, ntr, span = T.train_frozen()
s = T.load_fwd_bars(); X, t, r, anch = T.anchors(s)
ti = pd.DatetimeIndex(t.astype("datetime64[ns]")); mon = np.asarray(ti.to_period("M").astype(str))
months_all = sorted(set(mon)); last_ts = s.index.max()
complete = [m for m in months_all if pd.Period(m, "M").end_time.floor("h") <= last_ts]     # the month's last hourly bar exists
cal = "2025-10"; scored = [m for m in complete if m > cal]
mo = mon == cal
dmu, dsd = X[mo].astype(np.float64).mean(0), X[mo].astype(np.float64).std(0) + 1e-12
ARMS = {"frozen": None, "diag_oct_scaler": (dmu, dsd)}
COSTS = (0, 1, 2, 5)
res = {"train_rows": ntr, "train_span": span, "calibration_month": cal, "scored_months": scored, "bars_last": str(last_ts), "arms": {}}
for arm, ov in ARMS.items():
    p_by = {}; conf = {}
    for m in [cal] + scored:
        ix = np.flatnonzero(mon == m); p = T.probs(mdl, sc, X[ix], ov); p_by[m] = (ix, p); conf[m] = np.abs(p - 0.5)
    thr = {m: T.month_threshold(conf[prev]) for prev, m in zip([cal] + scored[:-1], scored)}
    ent, ex, side, idx, mm = T.signals({m: p_by[m] for m in scored}, t, thr)
    keep, nskip = E.non_overlapping(ent, ex)
    ent, ex, side, idx, mm = ent[keep], ex[keep], side[keep], idx[keep], mm[keep]
    ret = side * r[idx]
    A = {"thresholds": thr, "n_signals": int(len(keep)), "n_trades": int(len(ret)), "levels": {}}
    start_ns = int(pd.Timestamp(scored[0] + "-01").value)
    for c in COSTS:
        net = ret * 1e4 - c
        df = pd.DataFrame({"m": mm, "net": net, "win": net > 0}).groupby("m").agg(sum_bps=("net", "sum"), mean_bps=("net", "mean"), hit=("win", "mean"), n=("net", "size"))
        df = df.reindex(scored)
        ddm = []
        for m in scored:      # in-month max drawdown, equity restarting each month (additive, 1x)
            q = net[mm == m] * 1e-4
            if len(q) == 0:
                ddm.append(float("nan")); continue
            eq = 1 + np.cumsum(q); pk = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]; ddm.append(float(((pk - eq) / pk).max() * 100))
        df["max_dd_pct"] = ddm
        st = E.equity_stats(ret, ent, ex, start_ns, c)
        ms = df["sum_bps"].fillna(0).values; k = len(ms)
        se = ms.std(ddof=1) / np.sqrt(k); tq = sps.t.ppf(0.975, k - 1); ci = [float(ms.mean() - tq * se), float(ms.mean() + tq * se)]
        pl_long, pl_short = r[idx], -r[idx]
        nl = E.null_summary(pl_long, pl_short, c, draws=2000)
        eqn, ddn, _ = E.null_paths(pl_long, pl_short, c, draws=2000)
        nl["share_null_cum_ge_actual"] = float((((eqn[:, -1] - 1) * 100) >= st["cum_return_pct"]).mean())
        lev = T.leverage_frontier(ret - c * 1e-4)
        A["levels"][str(c)] = {"stats": st, "n_months": k, "mean_monthly_net_bps": float(ms.mean()), "monthly_ci95_t": ci,
                               "monthly": df.round(3).reset_index().to_dict("records"), "null": nl, "leverage": lev,
                               "months_hit_gt_60": int((df["hit"] > 0.6).sum()), "months_dd_lt_5": int((df["max_dd_pct"] < 5).sum()),
                               "goal_hit_gt_60": bool(st["hit"] > 0.6), "goal_dd_lt_5": bool(st["max_drawdown_pct"] < 5)}
    L2 = A["levels"]["2"]
    crit = {"months >= 6": L2["n_months"] >= 6, "mean monthly net bps at 2 bps > 0": bool(L2["mean_monthly_net_bps"] > 0),
            "95% t lower bound > 0": bool(L2["monthly_ci95_t"][0] > 0), "max drawdown < 15%": bool(L2["stats"]["max_drawdown_pct"] < 15)}
    A["criteria"] = crit
    if L2["n_months"] < 6:
        A["verdict"] = "inconclusive (fewer than 6 scored months)"
    elif all(crit.values()):
        A["verdict"] = "not refuted (criteria 1-3 met)"
    elif crit["mean monthly net bps at 2 bps > 0"] and not crit["95% t lower bound > 0"] and crit["max drawdown < 15%"]:
        A["verdict"] = "inconclusive (positive mean, interval includes 0)"
    else:
        A["verdict"] = "failed: " + ", ".join(k_ for k_, v in crit.items() if not v)
    res["arms"][arm] = A
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open("fwdtest_result.json", "w"), indent=1)
out = []
for arm, A in res["arms"].items():
    out.append(f"=== arm {arm}" + ("  (VERDICT ARM)" if arm == "frozen" else "  (diagnostic only, not a verdict)") + " ===")
    out.append(f"trades {A['n_trades']} of {A['n_signals']} signals; thresholds " + ", ".join(f"{m}:{v:.4f}" for m, v in A["thresholds"].items()))
    for c, L in A["levels"].items():
        st = L["stats"]
        out.append(f"-- cost {c} bps: hit {st['hit']:.3f}, mean {st['mean_bps']:+.1f} bps/trade, mean monthly {L['mean_monthly_net_bps']:+.0f} bps, 95% CI [{L['monthly_ci95_t'][0]:+.0f}, {L['monthly_ci95_t'][1]:+.0f}], cum {st['cum_return_pct']:+.2f}%, maxDD {st['max_drawdown_pct']:.2f}%, months hit>60% {L['months_hit_gt_60']}/{L['n_months']}, months DD<5% {L['months_dd_lt_5']}/{L['n_months']}")
        if c in ("0", "2"):
            out.append("   month      n   hit  mean_bps  sum_bps  maxDD%")
            for m_ in L["monthly"]:
                out.append(f"   {m_['m']}  {int(m_['n']) if m_['n'] == m_['n'] else 0:3d}  {m_['hit']:.3f}  {m_['mean_bps']:+8.1f} {m_['sum_bps']:+8.0f}  {m_['max_dd_pct']:.2f}")
        n_ = L["null"]
        out.append(f"   null (random side, same trades): cum median {n_['cum_return_pct_median']:+.2f}% p95 {n_['cum_return_pct_p95']:+.2f}%, hit median {n_['hit_median']:.3f} p95 {n_['hit_p95']:.3f}, DD median {n_['max_drawdown_pct_median']:.2f}% ; share of null draws with cum >= actual {n_['share_null_cum_ge_actual']:.3f}")
        if c == "2":
            ml = L["leverage"]["max_leverage_dd_lt_5pct"]
            out.append("   leverage frontier (2 bps): " + "; ".join(f"{k_}x: {v['cum_return_pct']:+.1f}% / DD {v['max_drawdown_pct']:.1f}%" for k_, v in L["leverage"].items() if k_[0].isdigit()) + f"; max leverage with DD<5%: {ml['leverage']:.2f}x -> {ml['cum_return_pct']:+.1f}%")
    out.append("criteria (2 bps): " + str(A["criteria"])); out.append("VERDICT: " + A["verdict"]); out.append("")
out.append(f"run time {res['seconds']} s")
open("fwdtest_result.txt", "w").write("\n".join(out)); print("\n".join(out))
