"""T02 selection: 2019-01..2024-06 on the cut main file. Writes selection.json. Never computes Look 1 or Look 2."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t02lib as L  # noqa: E402

t0 = time.time()
df, nmiss = L.load_main(); B = L.Bars(df)
sig = L.monthly_signals(B, "2019-01", "2024-06")
f = L.hour_frame(B, sig)
assert f["time"].max() < L.pd.Timestamp("2024-07-01")
print("hours", len(f), "missing minutes filled", nmiss, f"{time.time()-t0:.0f}s", flush=True)
res = {}
for cfg in L.CONFIGS:
    row = {}
    for bps in (0, 2, 5):
        c, _ = L.compare(f, cfg, bps); row[str(bps)] = c
    s, n = row["2"]["strategy"], row["2"]["null"]
    row["ratio_calmar_2bps"] = s["calmar"] / n["calmar"] if (n["calmar"] and n["calmar"] > 0 and np.isfinite(s["calmar"])) else None
    res[L.cname(cfg)] = row
valid = {k: v["ratio_calmar_2bps"] for k, v in res.items() if v["ratio_calmar_2bps"] is not None}
chosen = max(valid, key=valid.get)
cfg = [c for c in L.CONFIGS if L.cname(c) == chosen][0]
out = {"selected": chosen, "cfg": list(cfg), "ratio_calmar_2bps": valid[chosen], "all_ratios": valid, "period": "2019-01..2024-06",
       "n_hours": int(len(f)), "detail": res}
json.dump(out, open(f"{HERE}/selection.json", "w"), indent=1)
lines = ["config            mean_w turn/yr  strat(ret/DD/Calmar)        null(ret/DD/Calmar)       ratio"]
for k, v in res.items():
    s, n = v["2"]["strategy"], v["2"]["null"]
    lines.append(f"{k:16s} {v['2']['mean_w']:.3f} {v['2']['turnover_per_year']:7.0f}  {s['ann_return']:+.3f}/{s['max_dd']:.3f}/{s['calmar']:+.3f}   {n['ann_return']:+.3f}/{n['max_dd']:.3f}/{n['calmar']:+.3f}  {v['ratio_calmar_2bps']}")
b = res[chosen]["2"]["bh"]; lines.append(f"buy-and-hold ann {b['ann_return']:+.3f} DD {b['max_dd']:.3f} Calmar {b['calmar']:+.3f}")
lines.append(f"SELECTED {chosen}")
open(f"{HERE}/selection_tables.txt", "w").write("\n".join(lines)); print("\n".join(lines)); print(f"{time.time()-t0:.0f}s")
