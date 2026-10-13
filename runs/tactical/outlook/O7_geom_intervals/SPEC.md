# O7: do geometry-enhanced volatility forecasts give sharper return intervals? (outlook prospect 5: judge as a risk model)

Outlook lead, 2026-10-13, committed BEFORE any number of this study is computed. Tactical, exploratory; CPU only.

## Why
O1 and O5 (both QA PASS) found that geometry (G1 + G15) improves the RANKING and QLIKE of the volatility forecast at 15 and 60
minutes. OUTLOOK_positive.md prospect 5 asks for a risk-model judgement instead. The question here is whether that improvement
becomes better 90% return intervals at the same coverage. T03 of the volatility session (H85) set the method: intervals scaled
by HAR beat intervals scaled by trailing RV on unseen months. O7 reuses that interval method, with a different forecaster and
a different span.

## Forecasters of sigma_hat (bars < a; monthly walk-forward ridge on the 365 days before each month, as O5)
- **L1:** O5's candidate (S3 + bank + G1 + G15), ridge on y_T = ln RV(a, a+T). sigma_hat = exp(prediction / 2).
- **R0:** S3 only, the same target and transform.
- **HAR:** T03's form, a ridge of ln(|r_T| + 1e-5) on ln RV 15/60/240/1440 plus hour sin/cos. sigma_hat = exp(prediction).
  T03's `t03_lib.py` is imported read-only, or reimplemented with identical definitions. Secondary.
- Any constant scale cancels, because the quantiles are taken of r_T / sigma_hat.

## Intervals (T03's `cal` version)
- The target is the next-T log return r_T, for T in {15, 60, 240}, with anchors at stride 7.
- For each scored month M:
  - q_lo and q_hi are the 5% and 95% empirical quantiles of r_T / sigma_hat on month M-1 (the previous month's rows);
  - the forecaster is the fit for month M-1, so no row of M enters;
  - the interval is [q_lo, q_hi] x sigma_hat.
- Metrics per month: coverage, mean width (bps) and interval score (Gneiting-Raftery, alpha 0.1, in bps).

## Spans
- **Judgement:** 2019-02 .. 2020-12, 23 scored months (2019-01 serves only for calibration).
  - L1's RANKING was already judged on these months (O5), so this is a further look at the same months. The claim tested here
    is a different, downstream one (interval sharpness).
  - Stated as a caveat. No choice is made here: one candidate, one comparison.
- **Descriptive:** 2021-02..2024-06 and 2024-08..2025-07 (spent spans), with no verdict.
- **Prospective:** O5's frozen artefact (`O5_geom_bestbase/frozen_B.json`, sha 79f7f25d..., L1 and L0) is unchanged.
  - Calibration uses 2026-10, and scoring uses the next 3 complete months, once.
  - The criterion is the same, with L0 as the comparator, 3/3 months better, and the coverage band applied to the pooled 3 months.
  - `score_B.py` is written now and not run.

## Pre-registered criterion (T = 60, L1 against R0, judgement span)
**Pass** iff both hold:
1. The mean monthly interval-score difference L1 - R0 is below 0, with its 95% t-interval entirely below 0, and L1 is better in at least 15 of 23 months.
2. L1's pooled coverage is within [0.88, 0.92].

T = 15 and 240 are secondary (Holm over the two). L1 against HAR is secondary.

## Tests (seconds)
- The calibration month is strictly before the scored month, and the forecaster used for calibration never saw a row of the scored month.
- No bar after 2025-07-20 23:59 is read.
- L1 and R0 predictions reproduce O5's (L1) and O1/O3's (R0) predictions on one shared month to 1e-9, where the anchor grid allows.
- With sigma_hat = 1, the interval reduces to raw empirical quantiles (a unit check).

## Outputs
`runs/tactical/outlook/O7_geom_intervals/`: code, `results.jsonl`, `tables.txt`, `score_B.py`, a coverage-and-width figure per T
(`visualization/theme.py`, horizon colours not used for models), `run.log`, `tests.log`, `REPORT.md`.
