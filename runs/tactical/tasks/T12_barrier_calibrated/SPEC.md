# T12: the barrier-hit model with calibration (T10's H3 plus isotonic)

Lead, 2026-10-15. Committed BEFORE any result.

**Why.** T10 (H93) found a large, stable Brier skill: +15.4% over trailing-RV Brownian on both looks. It missed only the
calibration bar (ECE 2.11 / 1.26 pp against 1 pp). This task tests one fixed repair.

**Data.** Both T10 looks are now seen, so the verdict uses data no barrier study has touched:
- the cut file 2017-09..2018-12;
- the months from 2026-10 onward, prospectively.

## Model (fixed here, no choice)
- **H3c** = T10's H3, unchanged (code: `runs/tactical/tasks/T10_barrier_hit/`).
- Its output is mapped by an isotonic regression, fitted on the PREVIOUS calendar month's H3 predictions and their realised labels, per X and T. The labels must end before the month starts.
- **Comparators:** H3 (uncalibrated) and B1, the trailing-RV Brownian baseline.

## Look A, the verdict: cut file 2017-09..2018-12
- Monthly walk-forward in UTC (T09's offset logic). H3 is fitted on the 365 days before each month, or on everything since
  2017-08-17 where less exists; the training days are reported.
- The first month, 2017-09, only calibrates. The scored months are 2017-10..2018-12 (15 months).
- **Success** iff, in the primary cell (X = 50 bps, T = 60 min):
  1. the Brier skill of H3c over B1 is ≥ 2%, with a 95% CI above 0 (block bootstrap by day, 10,000 draws);
  2. ECE (10 equal-width bins) is < 1 pp;
  3. the Brier of H3c is not worse than H3 by more than 1% relative.
- **Also reported:** all 9 cells with Holm, reliability diagrams, ECE with a day-block bootstrap CI.

## Look B, prospective
- H3c's H3 part is frozen on the Binance file 2025-10-02..2026-09-30. The isotonic map uses 2026-09 for October 2026,
  then always the previous month.
- Stored with a sha256 in `frozen_B.json`. `score_B.py` scores the months from 2026-10-02 once at least 3 complete
  months exist, on the same criteria.

## Never
- No bar 2025-07-21..2025-09-30.
- No October-2026 target read in this task.
- Nothing tuned.

## Outputs
`runs/tactical/tasks/T12_barrier_calibrated/` with:
- code, `results.json`, `tables.txt`, `frozen_B.json`, `score_B.py`;
- figures in the tactical_report style (reliability before and after calibration);
- `tests.log`.

The report text goes to the lead.
