# O5: does indicator geometry add over the BEST volatility base, on months that chose nothing?

Outlook lead, 2026-10-13, committed BEFORE any number of this study is computed. Tactical, exploratory; CPU only.

## Why
O1 ("holds") found that the geometry G = G1 + G15 adds about +0.005 Spearman at H = 60 to the LINEAR S3 base. That held on
the look and on the forward. Two stronger bases exist, though:
- **HGB on S3:** it beat the linear S3 by +0.020 on the forward, and geometry did not add to it there (-0.0030).
- **BANK** (O3): S3 plus the logs of 36 EWMAs, a gain of +0.007 on the look and forward. It was not confirmed on 2017-09..2018-12 (O4).

The positive claim worth making is "geometry adds to the best forecast we have". The look and forward months are now spent for
this question, so O5 chooses on the old selection months and judges on a span that no volatility-level choice used.

## Data, rows, targets
As O1 and O3: the cut file (no bar after 2025-07-20 23:59) and the Binance forward file. Anchors use stride 7, features use bars < a,
and the target is y_H = ln RV(a, a+H), H in {15, 60, 240}. Metrics are Spearman(prediction, realised RV) and QLIKE (the SPEC form
v / (c e^mu), with c from the hold rows) per month. The fit is a monthly walk-forward on the 365 days before each month. The code
is O1's `o1_lib.py` (6cb5617) and O3's `o3_lib.py` (5e4ab42), imported unchanged.

## Bases and candidates (no tuning: the O1/O3 settings)
- **L0 = BANK:** ridge on S3 + bank. **L1 = L0 + G**, the same ridge.
- **N0 = HGB on S3 + bank**, with O1's HGB settings and rows strided by 10. **N1 = N0 + G.**
- **Base choice B*:** the better of L0 and N0 by mean Spearman at H = 60 on the selection months 2021-01..2024-06. These months
  were already seen by O1 and O3; using them only for this choice is allowed. The candidate is the matching B* + G.

## Judgement span (fixed now): 2018-07 .. 2020-12 (30 months)
- **Why this span:**
  - it comes before every span O1, O3 and O4 scored;
  - each month has at least 318 days of training history, avoiding O4's short-history failure;
  - the volatility level there was used only descriptively (H53: HAR on |r15|) and by H47's 24 short lab slices, some of which fall in these years. That is a partial touch, stated here.
- **Prospective part:** the candidate and B* are frozen on the Binance file 2025-10-05 .. 2026-09-30. `score_B.py` scores them once, on 3 complete months from 2026-10, under the same criterion.

## Pre-registered criterion (H = 60; candidate against B* on 2018-07..2020-12)
**Pass** iff all of:
1. the mean monthly Spearman gain is >= +0.003 and its 95% t-interval lower bound is > 0;
2. the gain is positive in at least 20 of 30 months;
3. the mean QLIKE difference is <= 0.

The bar is lower than O1's +0.005 because the base is stronger; it is fixed here before any result.
- **Verdict:** "geometry adds over the best base" or "does not".
- **Also reported, without a verdict:**
  - both pairs (L1 - L0, N1 - N0) whichever base is chosen;
  - H = 15 and 240 (Holm over the two);
  - B* against plain S3 (does BANK or HGB itself hold on this span? O4's open question at full training length).

## Tests (seconds)
- No bar after 2025-07-20 23:59 from the cut file and none from 2025-07-21..2025-09-30 are read. Every training label ends before its month.
- On one O1 selection month, L0 and N0 reproduce O3's BANK and O1's HGB H0 predictions to 1e-9. Where seeds or striding make that impossible, it reproduces them to their Spearman to 1e-6, with the reason stated.
- The frozen artefact reloads to identical predictions.

## Outputs
`runs/tactical/outlook/O5_geom_bestbase/`: code, `results.jsonl`, `tables.txt`, `frozen_B.json` (+ sha256), `score_B.py`, `run.log`,
`tests.log`, `REPORT.md`.
