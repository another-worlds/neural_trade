# O6: the geometry ladder: does hourly-bar geometry help where 1- and 15-minute geometry did not (H = 240)?

Outlook lead, 2026-10-13, committed BEFORE any number of this study is computed. Tactical, exploratory; CPU only.

## Why
O1 and O5 found that geometry adds to the volatility forecast at H = 15 and 60, but not at H = 240:
- O5, L1 - L0 at H = 240: +0.0004, 14/30.
- At H = 60, the 15-minute window (15 hours of history) carried most of the gain, which suggests that geometry works when its scale matches the horizon.

The next rung is G60, the same `geomlib.geom_rich` on the last 60 bars of 60 minutes (2.5 days, 3,600 one-minute bars ending at a-1).

## Data, rows, targets, metrics
As O5 (`o5_lib.py`, 0ab6466, imported unchanged where possible):
- the cut file only, with no bar after 2025-07-20 23:59;
- anchors at stride 7, features from bars < a;
- target y_H = ln RV(a, a+H);
- Spearman and the SPEC-form QLIKE per month;
- monthly walk-forward on the 365 days before each month.

G60 needs 3,600 bars of history, so the first anchor moves to at least 3,601 bars plus the bank warm-up.

## Models (ridge alpha 1, standardised; no tuning)
- **L1:** the O5 candidate, S3 + bank + G1 + G15.
- **L2:** L1 + G60.
- **L2sh:** L2 with the G60 rows permuted (train and test independently), as the control.

## Spans
- **Selection:** none. There is one candidate (L2) and no choice.
- **Judgement:** 2019-01 .. 2020-12 (24 months). These months were already the judgement span of O5 for G1 + G15. G60 was never scored anywhere, and O6 makes no choice on them. This is a SECOND look at the span for a different feature block, stated as a caveat.
- **Prospective:** L1 and L2 are frozen on the Binance file 2025-10-05 .. 2026-09-30. `score_B.py` scores them once, on 3 complete months from 2026-10, with the same criterion at H = 240 and 3/3 months positive.
- **Also reported, with no verdict:** 2021-01..2024-06 and 2024-07..2025-07 (spent spans: descriptive only).

## Pre-registered criterion (H = 240, L2 against L1, 24 judgement months)
**Pass** iff all of:
1. the mean Spearman gain is >= +0.003 and its 95% t lower bound is > 0;
2. the gain is positive in at least 16 of 24 months;
3. the mean QLIKE difference is <= 0.

H = 15 and 60 are secondary (Holm over the two).

**Sanity gate:** L2sh - L1 must be within +-0.003 at H = 240 on the judgement span. Otherwise the result is "broken".

## Tests (seconds)
- G60 at anchor a is unchanged when bars >= a are overwritten. The last hourly group ends at a-1.
- The hourly aggregation equals a pandas resample on an aligned example.
- No bar after 2025-07-20 23:59 is read, and the training labels end before each month.
- L1 reproduces O5's L1 on one judgement month to 1e-9, where the anchor grid allows. Otherwise the reason is stated.

## Outputs
`runs/tactical/outlook/O6_geom_hourly/`: code, `results.jsonl`, `tables.txt`, `frozen_B.json` (+ sha256), `score_B.py`, `run.log`,
`tests.log`, `REPORT.md`, and `interp.json` (the top G60 coefficients at H = 240 of the frozen fit, plus 6 real hourly windows before 2025-07-20 for the top 3).
