"""Descriptive per-year breakdown of the FROZEN candidate on the selection months (no new selection)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats as sps
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fwdlib as F, eqlib as E  # noqa
c = json.load(open(f"{HERE}/choice.json")); name, kind, T, H, frac = c["set"], c["model"], c["T"], c["H"], c["top"]
X = np.load(f"{F.CACHE}/{name}.npy"); t = np.load(f"{F.CACHE}/t.npy"); anch = np.load(f"{F.CACHE}/anch.npy")
r = F.next_open_ret(np.load(f"{F.CACHE}/open.npy"), np.load(f"{F.CACHE}/close.npy"), anch, H)
ent, ex, side, ret, mon = F.trades_from(list(F.month_predictions(X, t, r, kind, T, H)), t, H, frac)
yr = np.array([m[:4] for m in mon]); out = {"label": "IN-SAMPLE-OF-SELECTION, descriptive", "config": c, "blocks": {}}
def block(mask, start):
    d = {}
    for cost in (0, 2):
        st = E.equity_stats(ret[mask], ent[mask], ex[mask], start, cost); mt = F.month_table(ret[mask], mon[mask], cost); ms = mt["sum_bps"].values; k = len(ms)
        h = (ms.std(ddof=1) / np.sqrt(k) * sps.t.ppf(.975, k - 1)) if k > 1 else float("nan")
        d[str(cost)] = {"trades": st["n_trades"], "hit": st["hit"], "net_bps_trade": st["mean_bps"], "net_bps_month": float(ms.mean()),
                        "month_ci95": [float(ms.mean() - h), float(ms.mean() + h)], "max_dd_pct": st["max_drawdown_pct"],
                        "months": k, "months_hit_gt60": int((mt["hit"] > .6).sum())}
    return d
for y in sorted(set(yr)): out["blocks"][y] = block(yr == y, int(pd.Timestamp(f"{y}-01-01").value))
out["blocks"]["2022-01..2024-06"] = block(np.array([m >= "2022-01" for m in mon]), int(pd.Timestamp("2022-01-01").value))
json.dump(out, open(f"{HERE}/frozen_by_year.json", "w"), indent=1)
L = [f"{'block':17s} {'cost':>4s} {'trades':>6s} {'hit':>6s} {'bps/trade':>9s} {'bps/month':>9s} {'95% CI month':>18s} {'maxDD%':>6s} {'mo':>3s} {'hit>60':>6s}"]
for b, d in out["blocks"].items():
    for cost, s in d.items():
        L.append(f"{b:17s} {cost:>4s} {s['trades']:6d} {s['hit']:6.3f} {s['net_bps_trade']:9.2f} {s['net_bps_month']:9.1f} [{s['month_ci95'][0]:7.0f},{s['month_ci95'][1]:7.0f}] {s['max_dd_pct']:6.1f} {s['months']:3d} {s['months_hit_gt60']:6d}")
open(f"{HERE}/frozen_by_year.txt", "w").write("\n".join(L)); print("\n".join(L))
