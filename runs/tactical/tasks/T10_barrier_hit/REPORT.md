# T10 report: P(price moves ±X bps within T minutes)

Written by the lead from the implementer's hand-back (2026-10-15).
- SPEC: [SPEC.md](SPEC.md), commit 0a55dac.
- Selection: commit a042f4f, made before the looks.
- Results: commit b371fd7.
- Tests: 17 passed, including the first-passage formula checked against Monte Carlo.

## Verdict: the pre-registered criterion is NOT met

**What the criterion asked.** In the primary cell (X = 50 bps, T = 60 min), on both looks:
1. Brier skill over B1 of at least 2%, with a 95% CI above 0;
2. calibration error (ECE) below 1 percentage point.

**Result for the chosen model H3:**

| Look | Skill over B1 [95% CI] | ECE | AUC |
|---|---|---|---|
| Look 1, Bitstamp 2024-07..2025-07 | **+15.4% [+13.8, +16.9]** | 2.11 pp | 0.809 |
| Look 2, Binance 2025-11..2026-09 | **+15.4% [+13.9, +16.9]** | 1.26 pp | 0.803 |

- H3 is a logistic model on the HAR first-passage probability, T01's volatility extras and UTC hour harmonics.
- Criterion 1 passes on both looks. Criterion 2 fails on both. The criterion stays as written.

## Reading

- **The skill is large and stable, the calibration is not quite there.** The HAR-based hit probability beats the naive Brownian baseline by about 15% Brier on both venues and periods. That is far above the research's 2-5% estimate.
- **Much of that gain is repairing B1's calibration.** B1 (trailing RV) is badly miscalibrated, about 10 pp ECE. Against the EWMA baseline B2 the gain is about 8% (Look 1: 0.1693 against 0.1850).
- **Every cell has positive skill.** H3's skill is positive with the CI above 0 in all 18 look-cells (9 cells × 2 looks), Holm-significant everywhere. It runs from +6.5% (X = 100, T = 15) to +25% (X = 100, T = 240).
- **The miss is reliability: 1.3-2.1 pp against a 1 pp bar.**
  - Isotonic calibration (H2) is better calibrated in some short-horizon cells, but has less skill.
  - A calibrated H3 (H3 plus isotonic on the previous month) is a natural next candidate. It needs its own SPEC.

## Selection, 2019-01..2024-06, primary cell, skill over B1

| Model | Skill |
|---|---|
| B0 base rate | -11% |
| B2 EWMA | +8.0% |
| H1 HAR formula | +16.9% |
| H2 + isotonic | +17.4% |
| **H3 logistic (chosen)** | **+19.3%** |

H3 had the lowest Brier in every cell.

## Deviations

- Everything is on a UTC timeline (T09's offset logic).
- Anchors within 1,500 minutes of a seam, or with non-contiguous history, are dropped (0.1-3%).
- T03's HAR level, a mean absolute return, is converted to a standard deviation with sqrt(pi/2) before the Brownian formula.
- H3 details left open by the SPEC: C = 1, one regression per X, the 10 S3 extras.
- ECE uses 10 equal-width bins with no CI. The reliability-plot error bars assume independent anchors.

## Figures (`figs/`)

- reliability diagrams for both looks;
- Brier skill by month;
- a real BTC window (2025-11-15, picked by the median-hit-count rule) with ±50 bps boxes and P(hit) drawn on price.
