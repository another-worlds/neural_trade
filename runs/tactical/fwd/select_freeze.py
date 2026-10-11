"""Reads grid.jsonl, prints the grid table (selection months only), picks the best by the pre-declared score
(tie within 0.5 bps: lower max DD), and writes choice.json + frozen_stats.json (in-sample-of-selection, costs 0/1/2/5)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fwdlib as F  # noqa
import eqlib as E  # noqa

rows = [r for l in open(f"{HERE}/grid.jsonl", encoding="utf-8") for r in json.loads(l)["rows"]]
assert len(rows) == 48, len(rows)
rows.sort(key=lambda r: -r["score_mean_monthly_net_bps"])
best = rows[0]["score_mean_monthly_net_bps"]
tied = [r for r in rows if best - r["score_mean_monthly_net_bps"] <= 0.5]
choice = min(tied, key=lambda r: r["max_dd_pct"])
lines = [f"{'rank':>4s} {'set':5s} {'model':7s} {'T':>4s} {'H':>2s} {'top':>5s} {'score(bps/mo)':>13s} {'trades':>6s} {'mo':>3s} {'hit':>6s} {'bps/trade':>9s} {'maxDD%':>7s}"]
for i, r in enumerate(rows, 1):
    lines.append(f"{i:4d} {r['set']:5s} {r['model']:7s} {r['T']:4d} {r['H']:2d} {r['top']:5.2f} {r['score_mean_monthly_net_bps']:13.1f} {r['n_trades']:6d} {r['n_months']:3d} {r['hit']:6.3f} {r['mean_trade_net_bps']:9.2f} {r['max_dd_pct']:7.1f}")
lines.append(f"\nties within 0.5 bps of the best: {len(tied)}; CHOICE: {choice['set']}:{choice['model']}:{choice['T']}d H={choice['H']}h top {choice['top']:.0%}")
open(f"{HERE}/grid_table.txt", "w").write("\n".join(lines)); print("\n".join(lines))
json.dump(choice, open(f"{HERE}/choice.json", "w"), indent=1)

# frozen stats at costs 0/1/2/5 (recompute the trades)
name, kind, T, H, frac = choice["set"], choice["model"], choice["T"], choice["H"], choice["top"]
X = np.load(f"{F.CACHE}/{name}.npy"); t = np.load(f"{F.CACHE}/t.npy"); anch = np.load(f"{F.CACHE}/anch.npy")
r = F.next_open_ret(np.load(f"{F.CACHE}/open.npy"), np.load(f"{F.CACHE}/close.npy"), anch, H)
preds = list(F.month_predictions(X, t, r, kind, T, H)); ent, ex, side, ret, mon = F.trades_from(preds, t, H, frac)
start = int(F.pd.Timestamp("2019-01-01").value); out = {"label": "IN-SAMPLE-OF-SELECTION 2019-01..2024-06; not evidence", "choice": choice, "costs": {}}
for c in (0, 1, 2, 5):
    st = E.equity_stats(ret, ent, ex, start, c); mt = F.month_table(ret, mon, c)
    st["mean_monthly_net_bps"] = float(mt["sum_bps"].mean()); st["monthly_positive_share"] = float((mt["sum_bps"] > 0).mean())
    st["monthly_hit_months_above_60_traded"] = int((mt["hit"] > 0.6).sum()); st["n_months_traded"] = int(len(mt))
    ms = mt["sum_bps"].values; st["monthly_ci95_mean"] = [float(ms.mean() - 1.96 * ms.std(ddof=1) / np.sqrt(len(ms))), float(ms.mean() + 1.96 * ms.std(ddof=1) / np.sqrt(len(ms)))]
    out["costs"][str(c)] = st
json.dump(out, open(f"{HERE}/frozen_stats.json", "w"), indent=1)
print(json.dumps({c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ("n_trades", "hit", "mean_bps", "max_drawdown_pct", "months_hit_above_60", "n_months", "mean_monthly_net_bps", "monthly_positive_share", "monthly_ci95_mean", "cum_return_pct")} for c, s in out["costs"].items()}, indent=1))
