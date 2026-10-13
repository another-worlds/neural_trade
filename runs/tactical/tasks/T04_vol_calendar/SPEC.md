# T04: the 15-minute calendar for the next-hour volatility change: a retrospective and a PROSPECTIVE test

Lead, 2026-10-13. Committed BEFORE any result for the blocks named here.

**Origin.** qa-deep of T01 (H83) found that 96 dummies for each 15-minute slot of the day add +0.076 [+0.062, +0.090] skill
over S3 on the forward months, 12 of 12. That number was found ON those months, so they cannot confirm it.

**The two clean tests left:**
- the earliest history, never used for a vol-change study;
- the months that have not happened yet.

## Target, features, models
- **Target:** target A of T01 / gaps study 3: dA, the next-hour change in log RV, stride 7. Drops are as in T01.
- **Models**, each a ridge with alpha 1, standardised:
  - S3 (as T01);
  - S3C = S3 + 96 slot dummies, the 15-minute slot of the anchor's UTC time of day;
  - C = 96 slot dummies + `ar`'s regressor (L_0 - L_1).
- **Baselines:** `ar`, `rw`.
- **Also reported:** HAR levels only, the QA's strongest simple baseline.

## Part A: retrospective confirmation on 2017-09..2018-12 (16 months)
- **Why this block:** it lies before every vol-change selection (H72 selected on 2019-01..2024-06). It was used only descriptively (H53 decay_vol, HAR + hour harmonics, not 15-minute slots) and by H74 for a different target.
- **Fit:** monthly walk-forward, each model fitted on the 365 days before month M. Where less history exists (before
  2018-09), on all bars since 2017-08-17; the number of training days is reported.
- **Criterion A:** the monthly paired skill of S3C over S3 is >= +0.02, with the 95% t-interval lower bound > 0, and
  positive in at least 11 of 16 months.

## Part B: prospective (the forward file after 2026-09-30)
- **Freeze, done in this task:** fit S3, S3C and C plus `ar` and `rw` ONCE on the Binance file 2025-10-02..2026-09-30
  (the refit arm's full span; the first warm-up bars are skipped as in T01). Save the coefficients, scalers and a sha256 of
  the frozen artefact in `frozen_B.json`, and commit it.
- **Scoring, in a later session:** each complete calendar month from 2026-10 that is downloaded later. It is scored ONCE,
  when at least 3 complete months exist. The verdict is the first scoring with at least 3 months; later months are
  reported as follow-up.
- **Criterion B:** the same as A over the scored months: mean monthly skill of S3C over S3 >= +0.02, lower bound > 0, and
  positive in at least 3 of 3 months (or 80% when more).
- **Also reported:** S3 vs `ar` and vs HAR levels, so H83's +0.03 beyond HAR is re-measured prospectively.
- The 12 days of October 2026 already downloaded are NOT scored in this task, and nothing reads their targets.

## Tests (seconds)
- Part A reads only bars <= 2018-12-31 23:59 plus training history.
- No bar 2025-07-21..2025-09-30 is read.
- Part B's freeze reads no bar after 2026-09-30 23:59.
- The slot dummies come from the anchor's timestamp only.
- The frozen artefact reloads to identical predictions.

## Outputs
`runs/tactical/tasks/T04_vol_calendar/`: code, `resultsA.json`, `tablesA.txt`, `frozen_B.json`, `score_B.py` (ready for
the later scoring, with its own test on synthetic data), `tests.log`, `run.log`. CPU only. The report text goes to the
lead.
