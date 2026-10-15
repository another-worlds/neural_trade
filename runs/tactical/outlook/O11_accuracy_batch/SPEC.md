# O11: accuracy gains from untested promising hypotheses (a batch of four)

Outlook lead, 2026-10-15. Committed BEFORE any number of this study is computed. The owner asked (2026-10-15) to find gains in
accuracy on the promising hypotheses that have not been tested yet. Tactical, exploratory; CPU only.

## Spans (common to all parts)
| span | months | use |
|---|---|---|
| **S** selection | 2021-01 .. 2024-06 | every choice inside a part (a variant, a window length) |
| **J** judgement | B = 2024-07 .. 2025-07-20, plus C = Binance 2025-10 .. 2026-09 | the verdict, about 25 months pooled, the month as the unit |
| **P** prospective | first 3 complete months after 2026-10 | frozen for the winners |

- In S, data come from the cut file, with no bar after 2025-07-20 23:59. In C, the models are frozen on the 365 days to 2025-07-20, except in part b.
- **Caveat:** J was read by earlier studies for other claims:
  - O1 and O3 scored volatility-level models on B and C;
  - T01 scored S3 on the volatility change on C;
  - O8-O10 ran strategies on B and C.

  None of the four claims below was formed on J. A J verdict is therefore a "second-look" verdict, and P is the clean one.
- The parts are independent. The four primary tests are corrected with Holm (family-wise 5 %).

## Part a: geometry for the volatility-CHANGE forecast
- **Target:** T01's dA = ln RV(a, a+60) - ln RV(a-60, a), at stride 7 with T01's ok mask.
- **Models:** ridge (alpha 1, standardised), monthly walk-forward on the 365 days before each month. C is frozen.
  - **base:** S3, which is T01's model;
  - **candidate:** S3 + G1 + G15 + the EWMA bank, which is O5's L1 feature set.
- **Metric:** monthly skill vs AR = 1 - MSE / MSE_AR, as in T01.
- **Primary test:** the gain candidate - base on J. Pass iff the mean is >= +0.010 and the 95 % t lower bound is > 0 (after Holm).
- **Secondary:** Spearman with dA, and the gain at the 4-hour vol-of-vol target B of T01.

## Part b: monthly refit instead of a frozen fit (C only)
- **Setting:** the volatility level, O5's L1, ln RV at H = 15 and 60.
- **frozen:** fitted once on the 365 days to 2025-07-20, as in O1.
- **refit:** for each month M of C from 2025-11, refitted on the most recent 365 days available before M. That is the cut file to
  2025-07-20 plus Binance bars from 2025-10-01 up to M, never 2025-07-21..2025-09-30. Labels end before M.
- **Metric:** monthly Spearman with realised RV.
- **Primary test:** refit - frozen at H = 15 on C (2025-11..2026-09, 11 months). Pass iff the mean is >= +0.003 and the lower bound is > 0 (after Holm).
- **Secondary:** H = 60, QLIKE, and the same for S3.
- J here is C only, because B has no frozen arm.

## Part c: quantile regression with geometry for the tails (VaR 5 %, 95 %)
- **Target:** the next-60-minute log return r_60.
- **base:** O7's interval method. Empirical 5 % and 95 % quantiles of r / sigma_hat (sigma from L1) on the previous month, times sigma_hat.
- **candidate:** a linear quantile regression on [ln sigma_hat, G15, G1 top-15 features, hour sin/cos]. It is fitted separately for the
  5 % and 95 % quantiles, using sklearn QuantileRegressor (alpha 1e-4, highs solver) on a stride-70 subsample of the 365 days before
  the month, and predicts r directly. Monthly walk-forward; C is frozen.
- **Metric:** the monthly pinball loss at 5 % and at 95 %, averaged over the two, in bps.
- **Primary test:** candidate - base on J. Pass iff the mean is < 0 with the upper bound < 0 (after Holm), and the pooled 5 % and 95 % hit rates of the candidate are within [0.04, 0.06] and [0.94, 0.96].

## Part d: direction accuracy (the entry that carries the strategies of O9/O10)
- **Target:** the sign of the next-open 60-minute return, open[t+60] / open[t] - 1, at the 15-minute decision times of O9. Training
  uses a 5 bps dead band, as O9's p_up does.
- **base:** O9's p_up, a logistic (C 0.1) on tb7, with a 365-day walk-forward.
- **Variants:**
  - d1: tb7 + G15;
  - d2: tb7 with a 90-day training window;
  - d3: tb7 + G15 with a 90-day window.
- **Choice:** the variant with the highest mean monthly AUC on S. It is written to `choice_d.json` and committed before J is computed.
- **Metric:** the monthly AUC on all decision rows, with no dead band in scoring.
- **Primary test:** the chosen variant - base on J. Pass iff the mean is >= +0.005 and the lower bound is > 0 (after Holm).
- **Secondary:** AUC of the decisions where |p - 0.5| >= 0.02 (the strategy's gated subset), the hit rate of that subset, and the
  Brier score.

## Order, gates and outputs
- Part d writes `choice_d.json` from S and commits it alone before any J number.
- Parts a, b and c make no choice. Their only selection-span numbers are descriptive.
- **Freeze:** each passing candidate is frozen on Binance data to 2026-09-30 23:59 (as O9). It goes to `frozen_P_<part>.json`, and
  `score_P.py` scores P once later.
- **Tests (seconds):**
  - causality: features use bars < a;
  - no forbidden bar is read;
  - labels end before each month;
  - part b's refit never reads 2025-07-21..2025-09-30;
  - QuantileRegressor reproduces the empirical quantile when given only an intercept;
  - the choice gate (as O8).
- **Outputs:** `runs/tactical/outlook/O11_accuracy_batch/`:
  - `part_a/`, `part_b/`, `part_c/`, `part_d/` (code, `results.jsonl`, `tables.txt`);
  - `summary.txt` with the four verdicts and the Holm adjustment;
  - `tests.log`, `REPORT.md`.
