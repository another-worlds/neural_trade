"""Q4 addendum (measured): the long file fills missing minutes with flat zero-volume bars equal to the previous
close. Runs of such bars are the real outage distribution (a timestamp census sees only one gap). Run-length
distribution, per year, and the masked share on 7-day blocks if each run longer than G minutes were treated
as a reset (M-bar burn-in after the run), for M of periods 60 / 240 / 1440 after the shift.
Output: q4_ffill_runs.json"""
import numpy as np
import pandas as pd

from common import LONG_CSV, dump
from kernel import burn_in

lg = pd.read_csv(LONG_CSV)
ts = pd.to_datetime(lg["timestamp"], format="%Y-%m-%d %H:%M:%S")
o, h, l, c, v = (lg[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
prev = np.concatenate([[np.nan], c[:-1]])
fill = (v == 0) & (o == h) & (h == l) & (l == c) & (c == prev)
edges = np.diff(np.concatenate([[0], fill.astype(np.int8), [0]]))
starts = np.nonzero(edges == 1)[0]
ends = np.nonzero(edges == -1)[0]
runs = ends - starts
years = ts.dt.year.to_numpy()
BINS = [(1, 1), (2, 5), (6, 15), (16, 60), (61, 240), (241, 1440), (1441, 10 ** 9)]
out = {"filled_bars": int(fill.sum()), "n_runs": int(len(runs)), "longest_run_minutes": int(runs.max()),
       "run_quantiles": {q: float(np.percentile(runs, q)) for q in (50, 90, 99, 99.9)},
       "run_histogram": {f"{a}-{b if b < 10 ** 9 else 'inf'}": int(((runs >= a) & (runs <= b)).sum()) for a, b in BINS},
       "filled_bars_in_runs_over_60": int(runs[runs > 60].sum()),
       "runs_per_year": {int(y): int((years[starts] == y).sum()) for y in np.unique(years)},
       "runs_over_60min_per_year": {int(y): int(((years[starts] == y) & (runs > 60)).sum()) for y in np.unique(years)},
       "longest_runs": [{"start": str(ts.iloc[s]), "minutes": int(r)} for s, r in sorted(zip(starts, runs), key=lambda x: -x[1])[:8]]}
# masked share on 7-day blocks (bar-index blocks of 10,080 bars, the file is a regular grid) per reset threshold
n = len(c)
blocks = np.arange(0, n - 10080, 10080)
for p in (60, 240, 1440):
    M = burn_in(p)
    for G in (15, 60, 240, 1440):
        st = ends[runs > G]                                    # first real bar after a long filled run
        ev = np.zeros(n + 1, np.int64)
        np.add.at(ev, np.clip(st, 0, n), 1)
        np.add.at(ev, np.clip(st + M, 0, n), -1)
        mask = np.cumsum(ev)[:n] > 0
        cm = np.concatenate([[0], np.cumsum(mask)])
        frac = (cm[blocks + 10080] - cm[blocks]) / 10080
        out[f"reset_runs_over_{G}min_p{p}_M{M}"] = {"mean": float(frac.mean()), "max": float(frac.max()),
                                                   "blocks_with_masked": int((frac > 0).sum()), "n_blocks": int(len(blocks))}
# NT-041-style dropping of anchors whose window/label touches a filled bar (any run)
ev = np.zeros(n + 1, np.int64)
np.add.at(ev, np.clip(starts - 19, 0, n), 1)
np.add.at(ev, np.clip(ends + 59, 0, n), -1)
mask = np.cumsum(ev)[:n] > 0
cm = np.concatenate([[0], np.cumsum(mask)])
frac = (cm[blocks + 10080] - cm[blocks]) / 10080
out["anchors_whose_window_or_label_touches_a_filled_bar"] = {"mean": float(frac.mean()), "p90": float(np.percentile(frac, 90)),
                                                             "max": float(frac.max())}
print(out)
dump("q4_ffill_runs.json", out)
