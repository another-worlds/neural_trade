"""Runs SPEC.md. Selection 2021-01..2024-06 -> strategy -> ONE look at 2024-07..2025-09 -> per-year descriptive."""
import sys as _sys; _sys.path.insert(0, "D:/nt/nt_recut/runs/tactical/common")  # noqa: E402
import cutoff as C  # noqa: E402  (H66: no bar after C.CUTOFF is read)
import json, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import callib as C  # noqa

CSV = sys.argv[1] if len(sys.argv) > 1 else C.CSV
df = C.hourly_returns(CSV)
sel = df.loc[C.SEL_START:C.SEL_END]
assert sel.index.max() < pd.Timestamp("2024-07-01")
b = C.buckets(sel.index)
names = list(b)
eff = {n: C.effect(sel.r.values, b[n]) for n in names}
rej = C.holm([eff[n]["p"] for n in names])
for n, rj in zip(names, rej):
    eff[n]["holm_significant"] = bool(rj)
signs = {n: (1 if eff[n]["mean_bps"] > 0 else -1) for n in names if eff[n]["holm_significant"]}
fa = eff["funding_post"]["mean_bps"] - eff["funding_pre"]["mean_bps"]
out = {"selection": {"period": [C.SEL_START, C.SEL_END], "hours": len(sel), "buckets": eff, "strategy_buckets": signs,
                     "funding_post_minus_pre_bps": fa}}
yr = {}
for y in range(2017, 2026):
    d = df.loc[f"{y}-01-01":f"{y}-12-31"]
    if len(d) < 100:
        continue
    bb = C.buckets(d.index)
    yr[y] = {n: C.effect(d.r.values, bb[n]) for n in names}
out["per_year"] = yr
te = df.loc[C.TEST_START:C.TEST_END]
if signs:
    s = C.signals(te.index, signs)
    m = s != 0
    gross = (s * te.r.values)[m]
    times = te.index[m]
    out["test"] = {"period": [C.TEST_START, C.TEST_END], "hours": len(te),
                   "costs": {str(c): C.trade_stats(gross, times, c) for c in (0, 1, 2)}}
else:
    out["test"] = {"note": "no bucket significant after Holm on the selection period: no strategy, no trades"}
bt = C.buckets(te.index)
out["test_descriptive_buckets"] = {n: C.effect(te.r.values, bt[n]) for n in names}
json.dump(out, open(f"{HERE}/results.json", "w"), indent=1, default=float)

L = [f"SELECTION {C.SEL_START}..{C.SEL_END}, {len(sel)} hours; Holm over {len(names)}; NW lags {C.NW_LAGS}",
     f"{'bucket':16s}{'n':>7s}{'mean bps':>9s}{'95% lo':>8s}{'95% hi':>8s}{'p':>8s} holm"]
for n in names:
    e = eff[n]
    L.append(f"{n:16s}{e['n']:7d}{e['mean_bps']:9.2f}{e['lo']:8.2f}{e['hi']:8.2f}{e['p']:8.4f} {'*' if e['holm_significant'] else ''}")
L += [f"funding post-pre (selection, descriptive): {fa:.2f} bps", f"strategy buckets: {signs}", "",
      "TEST (one look) " + json.dumps(out["test"], default=float), "", "TEST period per-bucket means (descriptive)"]
for n in names:
    e = out["test_descriptive_buckets"][n]
    L.append(f"{n:16s}{e['n']:7d}{e['mean_bps']:9.2f} [{e['lo']:7.2f},{e['hi']:7.2f}]")
L += ["", "PER YEAR mean bps (descriptive); * = NW 95% interval excludes 0"]
ys = sorted(yr)
L.append(f"{'bucket':16s}" + "".join(f"{y:>9d}" for y in ys))
for n in names:
    L.append(f"{n:16s}" + "".join(f"{yr[y][n]['mean_bps']:8.2f}{'*' if yr[y][n]['lo'] > 0 or yr[y][n]['hi'] < 0 else ' '}" for y in ys))
open(f"{HERE}/results.txt", "w").write("\n".join(L))
print("\n".join(L))
