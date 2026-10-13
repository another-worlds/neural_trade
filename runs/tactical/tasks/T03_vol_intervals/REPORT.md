# T03 report: volatility-based return intervals on unseen data

Written by the lead from the implementer's hand-back (2026-10-13). SPEC: [SPEC.md](SPEC.md) (ada88f5, committed before
the results). Code and results: 34b41dd. Whole study about 9 s CPU; tests 5 passed (`tests.log`).

## Verdict: PASS at every horizon (primary T = 60)

Scored on 11 months, 2025-11..2026-09, with the `cal` version: HAR against trail.

| T | Interval score diff, HAR - trail (bps) | Months HAR better | Pooled HAR coverage | Holm p (one-sided) | Result |
|---|---|---|---|---|---|
| 15 | -1.68 [-2.02, -1.33] | 11/11 | 0.899 | 7.5e-7 | pass |
| 60 | -9.42 [-11.25, -7.59] | 11/11 | 0.900 | 6.7e-7 | **pass (primary)** |
| 240 | -26.9 [-38.5, -15.3] | 10/11 | 0.901 | 2.1e-4 | pass |

**Size of the gain.** At T = 60 the interval score drops by about 4.5% (207 → 198 bps). The interval is about 7%
narrower (140.5 → 130.4 bps) at the same 90% coverage. FZ0 for VaR/ES at 5% is lower in 10-11 of 11 months at every
horizon. The frozen quantiles from the training year already cover at 0.90-0.905, so recalibrating on the forward data
changes little.

## Monthly means (cal version)

| T | Forecaster | Coverage | Width (bps) | Interval score (bps) | VaR 5% hit | FZ0 |
|---|---|---|---|---|---|---|
| 15 | trail | 0.899 | 68.9 | 98.4 | 0.050 | -5.419 |
| 15 | HAR | 0.899 | 65.1 | 96.7 | 0.050 | -5.441 |
| 60 | trail | 0.900 | 140.5 | 207.0 | 0.050 | -4.608 |
| 60 | HAR | 0.900 | 130.4 | 197.5 | 0.050 | -4.688 |
| 240 | trail | 0.899 | 309.3 | 447.2 | 0.050 | -3.783 |
| 240 | HAR | 0.901 | 287.7 | 420.4 | 0.049 | -3.880 |

- Frozen-version numbers and the per-month tables are in `tables.txt` and `results.json`.
- Figures: `cover_width_T15/T60/T240.png`.

## Readings and caveats

- **HAR+ is degenerate by construction.** The frozen S1 vol-change forecast is a linear function of the same HAR
  features, so HAR+ = HAR (difference < 0.001 bps). It does not test whether the vol-change forecast helps.
- **HAR's coverage swings more by month** (T = 60: 0.871-0.919), though pooled coverage is exactly on target.
- **The months are not independent.** Anchors overlap within a month and the months share one market regime. The t-CIs over 11 months are the pre-registered inference and nothing more.
- **Different feeds.** Training is on the main file and scoring on Binance (H82).

## Deviations

- The scale constant c cancels in the empirical quantiles.
- Returns are log returns.
- trail = sqrt(RV over the last 4T bars / 4).
- Rows with zero trailing RV, or with log-RV features at or below -20, are dropped and counted.
- Calibration rows for month M end before M starts. 2025-10 is calibration-only and skips T01's 1,620-bar warm-up.

All deviations are listed in the hand-back and the code. None changes a criterion.
