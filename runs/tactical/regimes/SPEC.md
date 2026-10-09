# Market regimes: SPEC (fixed before any result is computed, 2026-10-09)

Owner, 2026-10-09: "режимы рынка - в очередь". Question: are there market regimes, defined only from past data, in which the
direction signal or the trading edge is reliably stronger, so that trading only in them helps out of sample?
Motivation: H7/H9 (high-AUC slices were regime luck), H27/H28/H35 (logistic regression ~0.570 mean3 AUC on 24 slices),
H33 + runs/tactical/hourly (triple-barrier + meta model). CPU only. Costs 0 (D-044).

## Regime variables (all from the minute bars up to and including the decision bar; nothing later)

Decision bar = the last bar of the window (minute study) or the last minute of the hourly entry bar (hourly study).
Per-minute log returns r. Windows of 1440 min (24 h) and 60 min (1 h) end at the decision bar.
- (a) vol level: v24 = sqrt(mean r^2) over 1440 min. Terciles (low / mid / high).
- (b) vol trend: v60 / v24 (RMS over 60 min over RMS over 1440 min). Terciles.
- (c) trend strength: |log c[i] - log c[i-1440]| / (v24 * sqrt(1440)). Terciles.
- (d) session by the UTC hour of the decision instant (bar open + bar length): Asian 00-08, European 08-16, US 16-24.
- (e) weekend (decision instant on Saturday or Sunday UTC) vs weekday.
Tercile cut points (1/3, 2/3 quantiles) are fitted on the TRAINING block only (part 1: the slice's train windows; part 2:
the fold's training anchors, or the whole dev training set for the held-out fit) and then applied to the evaluated bars.
A decision bar with fewer than 1440 earlier minutes has no regime and is dropped. The one 19-hour hole in the file (2017)
is ignored (positional windows). 14 cells in total (3+3+3+3+2).

## Part 1: minute direction (24 cached slices of runs/tactical/lab/cache, the lab's tb7 logistic regression C=0.1)

Per slice: the lab's per-horizon fit on the train block, P(up) on the val block (as `lab.run`). Val windows are located in
the raw file by exact OHLCV match of the window's last bar (stride 1 verified). Per cell and slice:
- cell AUC = mean over the 3 horizons of the AUC on the cell's val bars (deadband-masked per horizon, as the network's
  direction AUC); valid only with >= 300 masked bars per horizon and both classes. Difference d_s = cell AUC - the slice's overall AUC3.
- inference over slices (paired, the unit of inference): mean d, 95% t-interval, one-sided t-test p (H1: mean d > 0) over
  the valid slices; a cell is tested only with >= 12 valid slices.
- **A cell "helps" iff mean d > 0 and the Holm-adjusted p (over all tested cells, family = every cell tested in part 1,
  alpha 0.05) < 0.05.** Holm is applied to the p-values (equivalent to the rule "CI above 0 after Holm"); the plain 95% CI is shown too.
  Cells that are reliably WEAKER are listed (same test, other side) but carry no verdict.
- honest top-10% tail, descriptive only (no verdict, no correction): thresholds on |mean of 3 horizons' P - 0.5| from the FIRST
  half of the slice's whole val block, applied to the second half after a 20-bar gap, restricted to the cell; hit rate and gross
  bps on h1 (deadband-masked), slices with >= 20 such trades; random-side null95 (200 draws); mean over slices with 95% t-interval;
  and the paired difference to the slice's unrestricted tail on the same slices.

## Part 2: hourly trading (runs/tactical/hourly/hourly.py machinery, unchanged)

Configuration: `top3[0]` of runs/tactical/hourly/ranking.json = **T12_tp1.5_sl1.5_rich_hgb_barrier_mag** (the night program is done;
q 0.2), the trades are the meta model's selection. 8 dev walk-forward folds (entries before 2024-07-01), exactly as the study.
- Candidate filters (26): for each of the vol level, vol trend, trend strength: every non-empty proper subset of its 3 cells (6 each);
  session: every non-empty proper subset (6); weekend: {weekday}, {weekend}. A filter keeps the meta trades whose decision bar falls
  in the kept cells. Each kept cell must hold >= 20% of all pooled dev meta trades.
- Score of a filter: per fold, mean bps of the kept trades minus the random-side null95 of the same trades (200 draws, seed 0);
  a fold with < 10 kept trades is skipped; at least 6 valid folds; score = mean over folds. The filter with the highest score is THE filter
  (at most one). It is applied once, whatever its dev gain; the dev score of "no filter" is reported beside it.
- **Held-out look, once:** the study's final fit (all dev data minus the 2T gap, seed as the study) scores 2024-07-01..2025-09-29; the
  chosen filter is applied to the meta trades. Reported: n, trades/day, hit, mean bps, random-side null95 (200 draws, seed 0), 95%
  t-interval of the monthly mean bps over calendar months, filtered vs unfiltered. The unfiltered row must reproduce the study's final.jsonl
  (meta: n 2013, bps 9.83). Nothing is re-chosen after this look; a failure is recorded as such.
- Success (same rule as the hourly SPEC): filtered held-out bps > its null95 AND its monthly 95% interval above 0.

## Tests (seconds)
Regime variables are unchanged when every bar after the decision bar is altered; tercile cut points are computed from training
values only (altering evaluated values does not move them).
