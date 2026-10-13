# T03: forward test of volatility-based return intervals (risk measurement)

Lead, 2026-10-13. Committed BEFORE any result. This continues the volatility track after H83 (the vol-change forecast
holds forward) and H84 (vol as an exposure switch does not help). Volatility forecasts are useful for MEASURING risk:
intervals and VaR, which the MVP's variance head and conformal intervals also do (D-008).

## Question
Are return intervals scaled by a frozen HAR forecast sharper than those scaled by trailing realised volatility, at
the same coverage, on unseen data? Sharper means a lower pinball / interval score.

## Data
- **Training:** the cut main file, last bar 2025-07-20 23:59.
- **Forward:** the Binance file (H82).
- **Never read:** 2025-07-21..2025-09-30.

## Targets and horizons
- The next-T-minute log return r_T for T in {15, 60, 240}, anchors at stride 7.
- The forward warm-up is as in T01: features need their history inside the forward file.

## Forecasters of the scale sigma_hat (all from bars < a)
- **trail:** sqrt(RV over the last T x 4 minutes, scaled to T).
- **HAR:** ridge (alpha 1, standardised) of ln |r_T| (+1e-5) on ln RV 15/60/240/1440 plus hour sin/cos (H53's model).
  It is fitted once on the 365 days ending 2025-07-20 and frozen. sigma_hat = exp(prediction) x a constant fitted on the
  same span.
- **HAR+:** HAR plus T01's frozen S1 vol-change prediction as an extra regressor. Secondary.

## Intervals
- A symmetric central interval at 90%: r in [q_lo, q_hi] x sigma_hat.
- q_lo and q_hi are empirical quantiles of r_T / sigma_hat. Three versions:
  - **frozen:** quantiles from the training span;
  - **cal:** quantiles recalibrated on the previous calendar month of the forward file, a rolling conformal step.
    2025-10 is used only for calibration;
  - VaR at 5% (one-sided, lower tail) from the same quantiles.

## Scored months and metrics
- Scored months: 2025-11..2026-09 (11 months), for every forecaster and version.
- Per month:
  - coverage;
  - mean width in bps;
  - interval score (Gneiting-Raftery, alpha 0.1);
  - VaR hit rate;
  - FZ0 loss of the (VaR, ES) pair at 5%. ES is the mean of r / sigma_hat below q_5% on the calibration data, times sigma_hat.

## Pre-registered criterion (per T; the primary is the `cal` version, HAR vs trail)
**Success at horizon T** iff:
1. HAR's mean monthly interval score is lower than trail's, with the 95% t-interval of the monthly paired difference
   entirely below 0;
2. HAR's pooled coverage is within [0.88, 0.92].

The task passes if T = 60 succeeds; this is the primary. T = 15 and 240 are reported with Holm over the three. HAR+ vs
HAR is reported descriptively.

## Tests (seconds)
- the never-read span;
- features use bars < a only;
- the calibration month is strictly before the scored month;
- the frozen coefficients are identical across months.

## Outputs
`runs/tactical/tasks/T03_vol_intervals/`: code, `results.json`, `tables.txt`, a coverage and width figure per horizon
(`visualization/theme.py`), `tests.log`, `run.log`. The report text is returned to the lead, who writes REPORT.md.
CPU only.
