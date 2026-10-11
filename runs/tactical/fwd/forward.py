"""Forward test of the frozen candidate (FORWARD_SPEC.md). One command:
    python runs/tactical/fwd/forward.py --csv <1-minute OHLCV file extended past 2025-09-29> --first 2025-10 --last <last COMPLETE month>
Frozen: rich features, logistic (C = 0.1), 730-day monthly walk-forward, horizon 4 hours, next-open entry, top 10% of each month's own
confidence, non-overlapping trades, round trip 2 bps. Refuses test months before 2025-10 (spent data). Writes forward_<first>_<last>.json."""
import argparse, json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fwdlib as F  # noqa

CFG = dict(set="rich", model="logreg", T=730, H=4, top=0.10, cost=2.0)
MIN_MONTH = "2025-10"
CRIT = dict(min_months=6, dd_limit_pct=15.0)


def run_forward(csv, first, last, min_month=MIN_MONTH, cfg=CFG, out_dir=None):
    if pd.Period(first, "M") < pd.Period(min_month, "M"):
        raise SystemExit(f"refused: test months before {min_month} are spent data")
    import feats as F0  # noqa  (sys.path for lab)
    import lab
    import eqlib as E
    s = F.load_bars(csv, cutoff=pd.Timestamp("2100-01-01"))
    A = s.values.astype(np.float64); n = len(A); H = cfg["H"]
    anch = np.arange(F.L, n - (H - 1)); t = s.index[anch - 1].values.astype("datetime64[ns]").astype(np.int64)
    W = A[anch[:, None] + np.arange(-F.L, 0)].astype(np.float32)
    X = np.nan_to_num(np.asarray(lab.fs_rich(W), np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r = F.next_open_ret(A[:, 0], A[:, 3], anch, H)
    preds = list(F.month_predictions(X, t, r, cfg["model"], cfg["T"], H, first, last))
    if not preds:
        raise SystemExit("no feasible month")
    ent, ex, side, ret, mon = F.trades_from(preds, t, H, cfg["top"])
    mt = F.month_table(ret, mon, cfg["cost"]); ms = mt["sum_bps"].values; k = len(ms)
    st = E.equity_stats(ret, ent, ex, int(pd.Timestamp(preds[0][0] + "-01").value), cfg["cost"])
    se = ms.std(ddof=1) / np.sqrt(k) if k > 1 else float("nan")
    from scipy import stats as sps
    tq = sps.t.ppf(0.975, k - 1) if k > 1 else float("nan")
    ci = [float(ms.mean() - tq * se), float(ms.mean() + tq * se)]
    res = {"config": cfg, "months": [p[0] for p in preds], "n_months": k, "n_trades": int(len(ret)), "hit_rate": st["hit"],
           "mean_trade_net_bps": st["mean_bps"], "mean_monthly_net_bps": float(ms.mean()), "monthly_ci95_t": ci,
           "max_drawdown_pct": st["max_drawdown_pct"], "cum_return_pct": st["cum_return_pct"],
           "monthly": mt.round(3).reset_index().to_dict("records"), "criteria": {
               "months >= 6": k >= CRIT["min_months"], "mean monthly net bps at 2 bps > 0": bool(ms.mean() > 0),
               "95% t-interval lower bound > 0": bool(ci[0] > 0), f"max drawdown < {CRIT['dd_limit_pct']}%": bool(st["max_drawdown_pct"] < CRIT["dd_limit_pct"])}}
    res["PASS"] = all(res["criteria"].values())
    out_dir = out_dir or HERE
    json.dump(res, open(f"{out_dir}/forward_{preds[0][0]}_{preds[-1][0]}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--first", required=True); ap.add_argument("--last", required=True)
    a = ap.parse_args(); res = run_forward(a.csv, a.first, a.last)
    print(json.dumps({k: v for k, v in res.items() if k != "monthly"}, indent=1))
