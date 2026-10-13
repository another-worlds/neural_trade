# T05 report: can a model beat HAR for 60-minute intervals?

The lead wrote this from the implementer's hand-back (2026-10-13).
- SPEC: [SPEC.md](SPEC.md), commit 7a02d55.
- selection.json: f763b84, committed before the look.
- Final: 09bd009.
- Tests: 9 passed.

## Verdict: NO. HAR stays the reference. HAR's own edge over simple baselines is confirmed on untouched history.

## Selection (2019-01..2024-06, 66 months, T = 60)

| Model | Interval score (bps) | Coverage | Monthly coverage sd | vs M0: diff [95% CI], months lower |
|---|---|---|---|---|
| M0 HAR | 291.73 | 0.897 | 0.018 | |
| M1 EWMA 0.94 | 300.06 | 0.897 | 0.019 | +8.33 [+6.97, +9.69], 1/66 |
| M2 semivariance | 291.56 | 0.897 | 0.018 | -0.17 [-0.44, +0.11], 40/66 |
| M3 jump state | 291.65 | 0.897 | 0.018 | -0.08 [-0.24, +0.08], 40/66 |
| **M4 quantile regression (chosen)** | 290.57 | 0.898 | 0.018 | -1.16 [-1.74, -0.58], 43/66 |
| M5 ACI | 301.96 | 0.900 | **0.0014** | +10.2 [+8.75, +11.7], 1/66 |

## Look A (2017-09..2018-12, 16 months, run once)

| Comparison | Interval score diff (bps) | Months better | Coverage sd (chosen vs reference) | Result |
|---|---|---|---|---|
| M4 vs M0 (HAR) | +3.83 [-2.81, +10.48] | 8/16 | 0.027 vs 0.025 | fail |
| M4 vs M1 (EWMA) | -13.4 [-24.1, -2.7] | 13/16 | 0.027 vs 0.027 | fail (coverage sd only) |
| **M0 vs M1 (the T03 question re-asked)** | **-17.2 [-24.8, -9.7]** | **15/16** | | |

## Reading

- **The extra features add nothing.** Semivariances, jumps and quantile regression add at most 0.4% over HAR, and that does not survive the look.
- **HAR's gain over simple baselines is now confirmed on two independent periods:**
  - the Binance forward year (T03);
  - untouched 2017-18, where it beats EWMA 0.94, the strongest simple baseline, by ~4% of the interval score in 15/16 months.
- **ACI holds coverage almost exactly every month** (sd 0.0014 against 0.018), at a cost of ~3.5% in interval score. Use it where steady coverage matters more than sharpness.

## Look B (prospective)

M0, M1 and M4 were frozen on the Binance file to 2026-09-30.

| Item | Value |
|---|---|
| Artefact | frozen_B.json |
| sha256 | `e9c38d61466687163801d13f798f495c01c83b1dffb62a176bb39ef7702dfeaa` |
| Fit anchors | 74,302 |
| Scoring script | `score_B.py` |

`score_B.py` waits for 3 complete months from 2026-10.

## Deviations

- Only T = 60 was run. T = 15 and T = 240 were descriptive in the SPEC and were not computed.
- M4 was fitted on every 4th training row with sklearn QuantileRegressor (statsmodels is not installed).
- M4's calibration is a conformalised quantile-regression correction, the 90% quantile of max(qlo - z, z - qhi) on the previous month.
- The calibration month lies inside the 365-day fit window, so its residuals are in-sample. This follows the SPEC literally and is the same for every model.
- Look A's first months train on 14-105 days.
- Run times: selection 48 min, Look A 7 min.
