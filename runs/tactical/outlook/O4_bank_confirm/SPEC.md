# O4: confirm the EWMA-bank volatility forecast (O3's control) on data that did not choose it

Outlook lead, 2026-10-13, committed BEFORE any number of this study is computed. Tactical, exploratory; CPU only.

## Why
In O3 (REPORT.md) the CONTROL arm BANK beat the S3 base at every span and horizon. BANK is a ridge on S3 plus the logs of 12 EWMAs
(tau 1, 2, 4, ..., 2048 minutes) of r^2, (high - low) / close and the de-meaned volume level. At H = 60 the gain was
+0.0037 on selection (34/42 months), +0.0068 on the look (13/13) and +0.0073 on the forward (12/12), with QLIKE better too. That
is a finding seen AFTER the results, on spans that are now spent. O4 tests it where no choice was made.

## Model (frozen definitions; no tuning)
R0 and BANK exactly as in `runs/tactical/outlook/O3_synth_volind/o3_lib.py` at 5e4ab42. The target is y_H = ln RV(a, a+H), with
H in {15, 60, 240}, stride 7, the ok mask, ridge alpha 1 and the hold-out QLIKE scale. Metrics are Spearman(prediction, realised RV)
and QLIKE per month.

## Part A: retrospective, 2017-09 .. 2018-12 (16 months)
- **Why this block:**
  - It lies before every span that O1/O3 used (2021-01 onward).
  - It was used for the volatility LEVEL only descriptively (H53: HAR on the 15-minute |r|).
  - It was used by T04-A for a different target (the volatility change).
  - No EWMA-bank model was ever scored on it.
- **Data:** the cut file, bars up to 2018-12-31 23:59 plus the training history. Nothing later is read.
- **Fit:**
  - monthly walk-forward on the 365 days before month M, with labels ending before M;
  - where less history exists (before 2018-09), the fit uses all bars from the first warm-up-complete anchor of 2017-08-17. That anchor needs 3 x 2048 bars of bank warm-up and the 1,620 bars of S3 history;
  - the training days are reported per month.
- **Criterion A** (H = 60, BANK against R0), all of:
  1. the mean monthly Spearman gain is >= +0.005 and its 95% t-interval lower bound is > 0;
  2. the gain is positive in at least 11 of 16 months;
  3. the mean QLIKE difference is <= 0.
- H = 15 and 240 are secondary (Holm over the two) and reported.
- **Verdict A:** "confirmed" or "not confirmed".

## Part B: prospective (months from 2026-10 that are downloaded later)
- **Freeze, done now:**
  - fit R0 and BANK ONCE on the Binance file 2025-10-05 .. 2026-09-30, with the warm-up as in O3;
  - save the coefficients, scalers, bank definitions and the sha256 of the artefact in `frozen_B.json`;
  - commit it together with `score_B.py`.
- **Scoring:** in a later session, on each complete month from 2026-10, ONCE. The verdict needs 3 complete months. Criterion B is criterion A at H = 60, with "positive in at least 3 of 3".
- No bar after 2026-09-30 23:59 is read by the freeze.

## Tests (seconds)
- Part A reads no bar after 2018-12-31 23:59 (plus the H-minute label tail), and every training label ends before its month;
- the freeze reads no bar after 2026-09-30 23:59;
- the frozen artefact reloads to identical predictions;
- R0 and BANK on one O3 selection month reproduce O3's `results.jsonl` values to 1e-9 (same code path).

## Outputs
`runs/tactical/outlook/O4_bank_confirm/`: code, `resultsA.jsonl`, `tablesA.txt`, `frozen_B.json`, `score_B.py`, `run.log`,
`tests.log`, `REPORT.md`.
