# T10: probability that price moves ±X bps within T minutes (an unsigned barrier hit)

Lead, 2026-10-15. Committed BEFORE any result.

- **Owner:** "запускай". The task is hypothesis 1.2 of
  [research/2026-10-15-accuracy-hypotheses](../../research/2026-10-15-accuracy-hypotheses/README.md).
- **Why it may work:** an hour of BTC is Brownian in its own clock (H67), so the hit probability is mostly set by
  sigma. HAR sigma is the best sigma we have (T03, T05).

## Target
- y = 1 iff max over minutes s in (t, t+T] of |ln(high_s or low_s) - ln(close_t)| ≥ X. Entry reference: close_t. The
  1-minute high/low monitor is near-continuous, so no discreteness correction is applied.
- X ∈ {25, 50, 100} bps and T ∈ {15, 60, 240} min: 9 cells.
- Anchors are non-overlapping: stride T, aligned to the hour start (UTC).

## Models

| ID | Model |
|---|---|
| B0 | the previous calendar month's base rate |
| B1 | Brownian two-sided first passage with sigma from trailing RV over 4T bars |
| B2 | the same with sigma from EWMA 0.94 |
| H1 | the same with sigma from HAR: T03's model fitted on the 365 days before each month; hour terms in UTC via `T09_regression_anatomy/lib.py:utc_hour`; seam anchors excluded |
| H2 | H1 + isotonic calibration on the previous month |
| H3 | logistic on [logit(H1), S3 extras of T01, UTC hour harmonics], fitted on the 365 days before each month |

## Selection: 2019-01..2024-06, cut file, monthly walk-forward
- The ONE model among H1, H2 and H3 with the lowest mean monthly Brier in the primary cell (X = 50, T = 60) is chosen.
- It is written to `selection.json` and committed before the looks.
- All cells and all models are reported.

## Looks

| Look | Period | Kind | Note |
|---|---|---|---|
| Look 1 | 2024-07..2025-07-20 | binding | cut file |
| Look 2 | 2025-11..2026-09 | binding | Binance forward file; models frozen as fitted on the 365 days to 2025-07-20. 2025-10 only calibrates: the H2 isotonic map and the B0 base rate. |

This target has not been evaluated on either period.

## Pre-registered criterion (primary cell X = 50 bps, T = 60 min)
Success iff, on BOTH looks:
1. the Brier skill of the chosen model over B1 is ≥ 2%, with a 95% CI above 0 (block bootstrap by day, 10,000 draws);
2. the reliability error (ECE, 10 bins) is < 1 percentage point.

**Also reported:**
- the other 8 cells, with Holm over the 9 cells;
- AUC;
- log loss;
- reliability diagrams.

## Never
- No bar 2025-07-21..2025-09-30.
- No choice on the looks.
- The main file is not modified.

## Outputs
`runs/tactical/tasks/T10_barrier_hit/` holds:
- code, `selection.json`, `results.json`, `tables.txt`;
- figures in the tactical_report style:
  - reliability diagrams;
  - Brier skill by month;
  - one real BTC window with the ±50 bps band and the predicted P(hit) drawn on price.

The report text goes to the lead.
