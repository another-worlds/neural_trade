# T05: better 60-minute return intervals than HAR, checked on untouched history and prospectively

Lead, 2026-10-13. Committed BEFORE any result.

**Why.** T03 (H85) passed: HAR-scaled intervals beat trailing RV(4T). But QA showed two things:
- against the best simple baselines (trail over 1T, EWMA 0.94) the T = 60 gain is only about 2.2%;
- HAR's monthly coverage swings from 0.874 to 0.92.

This study looks for a model that is sharper still and holds its coverage month by month.

## Target
- The next-60-minute log return, stride 7, as in T03.
- T = 15 and T = 240 are descriptive only.

## Candidates (6, fixed here)
All quantiles are empirical quantiles of r / sigma_hat.

| ID | Model |
|---|---|
| M0 | **HAR**, exactly T03's model. The reference. |
| M1 | **EWMA 0.94** on per-minute squared returns, scaled to T. The QA's strongest simple baseline at T = 60. |
| M2 | **HAR + semivariances:** HAR plus ln RV⁺ and ln RV⁻ (positive and negative squared returns) over 60 and 1440 bars, plus the ratio RV⁻ / RV over 60 bars. H70 found the semivariance ratio carried the VaR gain. |
| M3 | **HAR + jump state:** HAR plus ln(1 + max abs 1-minute return over the last 60 bars / sigma60), plus the bipower-variation share of RV over 60 bars. |
| M4 | **Quantile regression:** the 5% and 95% conditional quantiles of r / sigma_HAR, as linear functions of the M2 features (statsmodels QuantReg, or sklearn QuantileRegressor with alpha 0). This gives asymmetric, state-dependent bounds. |
| M5 | **Adaptive conformal:** M0 with ACI (Gibbs-Candès). The quantile level is updated after every scored anchor whose label is complete, gamma 0.005, target 0.90. It reacts to coverage drift inside the month. |

## Selection: 2019-01..2024-06, cut main file
- Monthly walk-forward. Each model is fitted on the 365 days before month M. Quantiles come from the previous month (the `cal` rule of T03). ACI runs through the month.
- The selection score is the mean monthly interval score (alpha 0.1). Only models with pooled coverage in [0.88, 0.92] qualify.
- **One** model, other than M0 and M1, is chosen by the lowest score. The choice is written to `selection.json` and committed before any look.
- Also reported: the monthly coverage standard deviation for each model.

## Look A, the verdict: 2017-09..2018-12, cut main file
- **Why this block:** no interval study used it. It was used by:
  - H53, descriptively, for the HAR Spearman correlation;
  - H74, for the 240/1440-minute variance level;
  - T04 part A, for the vol-change target.

  So it is partly seen, and this is stated.
- **Fit:** walk-forward. Training is the 365 days before each month, or everything since 2017-08-17 where less exists. Report the days used.
- **Success** iff the chosen model, against BOTH M0 (HAR) and M1 (EWMA 0.94):
  1. has a monthly paired interval-score difference with a 95% t-interval entirely below 0;
  2. is better in at least 11 of 16 months;
  3. has pooled coverage in [0.88, 0.92];
  4. has a monthly coverage standard deviation no greater than M0's.
- Also reported: M0 against M1 on this block, the T03 question re-asked on untouched history.

## Look B: prospective
- The chosen model, M0 and M1 are frozen on the Binance file 2025-10-02..2026-09-30, fit plus the quantiles of 2026-09, with a sha256.
- `score_B.py` scores the months from 2026-10 once at least 3 complete months exist, on the same criterion with "at least 80% of months".

## Never
- No bar 2025-07-21..2025-09-30 is read.
- No forward month 2025-10..2026-09 is used for selection.
- Nothing is chosen on Look A.

## Tests (seconds)
- features use bars < a only;
- quantiles come only from months before the scored one;
- ACI updates only from anchors whose label ended;
- the never-read span;
- selection.json is committed before the looks (asserted);
- the frozen artefact reloads to identical predictions.

## Outputs
`runs/tactical/tasks/T05_interval_models/`: code, `selection.json`, `resultsA.json`, `tables.txt`, `frozen_B.json`,
`score_B.py`, `tests.log`, `run.log`. The report text goes to the lead. A report section in the style of tactical_report.html
follows in a separate step.
