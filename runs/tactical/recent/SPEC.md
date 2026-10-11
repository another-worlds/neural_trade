# Is there ANY direction signal left in the recent market? SPEC (fixed 2026-10-11 ~05:45, before any result is computed)

Owner, 2026-10-11 (tactical, autonomous until 13:00): the final question of the night. LOG H51-H54: minute direction AUC fell from
~0.60 (2018-20) to ~0.52 (2025) at every scale while volatility became more predictable. Is there anything left with everything
we built? CPU only. Descriptive-exploratory rigour (tactical, D-062/63); nothing here changes a default.

## Protocol (the monthly walk-forward of runs/tactical/lab/decay.py and decay_lib.py)

- Data: `Bitcoin_BTCUSDT.csv` (2017-01-01 .. 2025-09-29, 1-minute). Two cells:
  - **1m**: 1-minute bars, 60-bar window, anchors every 10 bars, label = sign of the 15-bar return beyond the 5-bps deadband
    (`r = close[i+14]/close[i-1]-1`, anchor `i` = window `A[i-60:i]`).
  - **1h**: 1-hour bars (resampled as `decay_lib.load_bars('1h')`), 60-bar window, every bar an anchor, label = sign of the next
    1-bar return beyond the 1e-3 deadband (`5e-4*sqrt(60/15)`, as `decay_lib.deadband`).
- For each calendar month M: train on the `T` days before M, T in {60, 365}; test on the anchors of M. Train and test anchors are
  masked to `|r| > deadband` (as decay.py). Score = AUC on the masked test anchors of M. Features: standardised on the training
  rows, clipped at +-8 (the same pipeline for every set, tb7 included).
- **Purge (stricter than decay_lib, which let one training label end inside M's first bar):** a training anchor needs its label
  bar to CLOSE by the month start: `t_open + (H+1) bars <= M start`. Test anchors are `t_open in [M start, next start)`.
  Nothing from M or later enters a training row, a scaler, a hyper-parameter choice or a threshold.
- **SELECTION months: 2021-01 .. 2024-06 (42 months).** All choices (which configurations go to the look) use only these.
  **The ONE look: 2024-07 .. 2025-09 (15 months)**, for the top 3 selected configurations of each cell and the baseline, run once,
  after the selection table and the picks are committed. Look months are never computed before that commit (`--phase look` refuses
  to run unless `selection.json` with `picks` exists). Nothing is tuned on them; no second look.

## Feature sets (all causal, all computed from the window ending at the anchor's last bar, or from per-bar trailing statistics)

| set | content | code |
|---|---|---|
| tb7 | the textbook 13 (EMA20, RSI14, MACD, BB20, ATR14, Stoch14, CCI20 + 6 trailing returns) | lab.fs_tb7 |
| rich | 43 features of the 14 families at several periods | lab.fs_rich |
| ctx | the 17 context features beyond the window (returns over 60/240/1440/10080 bars over volatility, log realised vols, 3 scale ratios, 1-day/1-week range positions, time sin/cos) | newnet/data.context_features at the last bar |
| gR | the geometry set (slopes, positions, crossings at 3/5/10/20 bars, squeezes of 24 channels) | geom2/geomlib.geom_rich |
| novel | N1-N7 | novel/features.compute |
| har | log realised volatility over the last 15/60/240/1440 bars (sqrt of the mean squared 1-bar log return, +1e-7, log) + sin/cos of the hour and of the day of week | new (feats.har) |
| all | tb7 + rich + ctx + gR + novel + har, concatenated | |

On the 1h cell the same code runs on hourly bars; every window and scale is in bars of that cell (a "1440-bar" scale is 60 days
there). `har` and `ctx` time features use the bar's open time.

## Models

- **logreg**: LogisticRegression(C = 0.1).
- **l1**: L1-logistic (liblinear); C chosen from {0.003, 0.01, 0.03, 0.1, 0.3} by AUC on the LAST 20% (by time) of the training rows
  after fitting on the first 80% with a gap of H+1 bars; then refitted on all training rows at the chosen C.
- **hgb**: HistGradientBoostingClassifier(max_iter 150, learning_rate 0.05, max_depth 4, min_samples_leaf 200,
  l2_regularization 1.0, random_state 0) (as decay_lib.make_model).

7 sets x 3 models x 2 training lengths = 42 configurations per cell.

## Metrics and decision rules (fixed now)

- Per configuration and month: AUC; the top-10% hit rate and mean gross bps of the month's most confident 10% of anchors
  (threshold = the 90th percentile of |P-0.5| among the month's own predictions: label-free, no in-sample thresholds), sign =
  sign(P-0.5), over all of the month's anchors (not masked), a hit = a positive signed return (as decay.py).
- Per configuration over the selection months: mean AUC with a 95% t-interval over months (t quantile of n-1 df); the PAIRED
  difference against the tb7-logreg baseline **of the same training length T and cell**, mean and 95% t-interval over months;
  mean top-10% hit and bps. Months with fewer than 500 training rows or 300 masked test rows are skipped (identically for all).
- **Picks:** per cell, the 3 configurations (excluding the baseline itself) with the largest mean paired difference on the
  SELECTION months. No other criterion. Ties by the table order.
- **Success of a pick on the look (pre-registered):** the 95% t-interval over the 15 look months of (pick AUC - baseline AUC)
  lies entirely above 0. Three picks per cell are looked at, so the interval at 98.33% (Bonferroni over 3) is reported next to
  it. A pick that fails is a negative result and is recorded, not tuned away. "Any signal left" is answered as: AUC itself above
  0.5 with an interval that excludes 0.5 on the look months is reported separately (absolute signal), with the honest top-10%
  hit and bps on the look months. Months are autocorrelated (overlapping 60/365-day training windows); the t-interval over
  months is the pre-registered test and is optimistic by that amount.

## Tests (seconds; `runs/tactical/recent/test_recent.py`)

1. The training mask never includes an anchor whose label closes after the month start (on a synthetic grid, both cells).
2. Corrupting every bar after a month's end leaves that month's walk-forward rows unchanged to 1e-9 (all models fitted on
   synthetic features through the real `walk_forward`).
3. Every feature block (tb7, rich, ctx, gR, novel, har) of every anchor whose last bar is at or before bar b is unchanged when
   all bars after b change.

## Outputs

`runs/tactical/recent/{selection_1m,selection_1h}.json` (monthly rows per configuration), `selection.json` (table, picks),
`look_{1m,1h}.json`, `REPORT.md` (tables), logs. Feature caches stay in `D:/nt/nt_recent_cache/` (not committed).
Resource rule: at most 6 worker processes, below-normal priority, OMP_NUM_THREADS 4.
