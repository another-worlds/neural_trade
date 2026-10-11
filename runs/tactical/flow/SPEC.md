# Order flow from OHLCV bars: does it add direction skill? SPEC (fixed 2026-10-11, before any result is computed)

Owner, 2026-10-11 (tactical, autonomous until 13:00). Hypothesis (H64, not tested before): order-flow imbalance
**estimated from OHLCV bars alone** (no new data source, D-050) adds short-term direction skill in the recent market.
CPU only, up to 4 processes at below-normal priority. Exploratory rigour (D-062/63); nothing here changes a default.

## Data and cells

`Bitcoin_BTCUSDT.csv` (1-minute, 2017-01-01 .. 2025-09-29). Two cells:

| cell | bars | window | anchor stride | horizon H | label |
|---|---|---|---|---|---|
| 1m | 1-minute | 60 bars | 10 bars | 15 bars | sign of `r = close[a+14] / open[a] - 1` beyond the deadband |
| 5m | 5-minute (resample first/max/min/last/sum) | 60 bars | 2 bars | 3 bars | sign of `r = close[a+2] / open[a] - 1` beyond the deadband |

Anchor `a` = window `A[a-60:a]`; its last bar is `a-1` (opens at `t`). **Entry is the NEXT OPEN `open[a]`** (as
`runs/tactical/nextopen/`); the exit is the close of bar `a+H-1`. Deadband 5e-4 in both cells (`5e-4*sqrt(H*bar_min/15)`).
Anchors whose features are NaN (warm-up of the first days) are dropped.

## Flow features (9, all causal: bar `t` reads bars `<= t` only; computed per bar, taken at the anchor's last bar)

Per bar: `dP_t = close_t - close_{t-1}`; `sigma_t` = std of `dP` over the trailing 4 hours (240 bars at 1m, 48 at 5m, current bar
included, zero sigma gives z = 0); bulk volume classification (Easley, Lopez de Prado, O'Hara): buy share `Phi(dP_t / sigma_t)`;
signed volume `sv_t = V_t (2 Phi - 1)`.

1. `imb5`, `imb15`, `imb60`: `sum(sv) / sum(V)` over the last 5 / 15 / 60 bars.
2. `vpin`: volume clock with 50 buckets per day: bar `t` adds `V_t / Vb_t` buckets, `Vb_t` = trailing 24-hour volume ending at
   `t-1`, divided by 50 (no bar before the first full day counts). Bucket id = floor of the cumulative sum (a bar belongs to
   the bucket in which it ends). `vpin_t = sum_j |sum sv in bucket j| / sum_j V_j` over the last 50 COMPLETED buckets
   (ids below the current bar's id).
3. `ad15`, `ad60`: accumulation/distribution: `sum(clv V) / sum(V)` over 15 / 60 bars, `clv = ((c-l)-(h-c))/(h-l)` (0 if `h = l`).
4. `vwd15`, `vwd60`: volume-weighted log return minus the plain return over k bars: `k * sum(V lr)/sum(V) - sum(lr)`.
5. `kyle60`: Kyle-lambda-style impact: OLS slope of the 1-bar log return on `sv / mean(V over the trailing 24 h)` over the last
   60 bars (0 if the variance is 0).

Feature sets: **tb7** (the textbook 13 of `lab.fs_tb7` on the 60-bar window), **flow** (the 9 above), **tb7+flow** (22).

## Walk-forward and models (the monthly walk-forward of `runs/tactical/recent/wf.py`, imported, unchanged)

For each calendar month M: train on the 365 days before M (training anchors whose label bar closes by the month start), test
on the anchors of M; both masked to `|r| > deadband`; standardise on the training rows, clip at 8; AUC on the masked test
anchors. Months with fewer than 500 training or 300 masked test rows are skipped identically for all sets. Models: logreg
(C = 0.1) and HistGradientBoosting (150 iterations, lr 0.05, depth 4, min leaf 200, l2 1, seed 0), as `wf.py`.
Top-10% hit rate and mean bps (month's own 90th percentile of |P-0.5|) are reported descriptively.

## Selection (months 2021-01 .. 2024-06 ONLY; 42 months)

- 4 tests: {1m, 5m} x {logreg, hgb}. Statistic per test: the paired monthly difference `AUC(tb7+flow) - AUC(tb7)`, mean and
  95% t-interval over months (n-1 df).
- **A flow set "adds"** in a test iff the Holm-adjusted two-sided paired-t p-value over the 4 tests is below 0.05 and the mean
  difference is positive (equivalently the t-interval at the Holm-adjusted level excludes 0 on the positive side). Holm: sort
  the 4 p-values ascending, multiply the i-th (1-based) by (4 - i + 1), enforce monotonicity, cap at 1. The unadjusted 95%
  interval is also printed. No minimum practical effect is imposed; the mean difference is reported with it.
- `flow` alone: mean AUC with 95% interval over the selection months and its paired difference to tb7, descriptive.
- Nothing is chosen on any month after 2024-06; a test that does not add is a clear negative and stays one.

## The one look (2024-07 .. 2025-09; 15 months)

Run ONCE, after the selection table and `selection.json` are committed, and refused otherwise: for every test that added,
tb7 and tb7+flow on the look months, reported descriptively (the paired difference with its interval, no decision). This
period was already used by other hypotheses (H38, H56, H63), so it is not independent evidence. The same single pass also
computes the flow-alone logistic AUC of 2024-07 .. 2025-09 for the stability table (descriptive; nothing selected from it).
If no test added, the pass still runs for the stability table only.

## Stability: per-year AUC of flow alone, 2017-2025

Flow alone, logreg, both cells, trained on the 365 days before each month (2018-01 onward) and, because 2017 has no 365-day
history, also on 60 days (2017-03 onward); mean monthly AUC per calendar year with a 95% t-interval over the year's months
and the 0.5 reference. Descriptive.

## Tests (seconds, `test_flow.py`)

- feature rows at bar `t` are unchanged when bars after `t` are altered (all 9 features, both bar sizes);
- the label uses the next open (`close[a+H-1]/open[a]-1`), checked on a synthetic series;
- the selection phase refuses months after 2024-06 and its output is unchanged when every row after 2024-06 is replaced.
