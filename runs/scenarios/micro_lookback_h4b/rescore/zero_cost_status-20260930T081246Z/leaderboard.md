# Strategy study `zero_cost_status` on scenario `micro_lookback_h4b`

7 configurations x 1 stored cells (1 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 925398d, scenario spec hash 091ac910adb0.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 1 | +3.50 | n/a | +5.55% | 3.96% | 298.0 | 100% | 90 |
| 2 | `cq[entry_quantile=0.9]` | 1 | +2.54 | n/a | +5.20% | 4.98% | 1300.0 | 100% | 70 |
| 3 | `cq[entry_quantile=0.95]` | 1 | +2.47 | n/a | +4.51% | 3.80% | 764.0 | 100% | 85 |
| 4 | `nk0` | 1 | +1.14 | n/a | +2.45% | 6.28% | 1031.0 | 100% | 95 |
| 5 | `ec0` | 1 | -0.92 | n/a | -2.89% | 5.74% | 2046.0 | 100% | 35 |
| 6 | `vt` | 1 | -2.44 | n/a | -6.81% | 13.91% | 34.0 | 100% | 50 |
| 7 | `cq[entry_quantile=0.99]` | 1 | -4.60 | n/a | -6.27% | 8.22% | 239.0 | 100% | 10 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `cq[entry_quantile=0.9]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `cq[entry_quantile=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `nk0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `ec0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `vt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `cq[entry_quantile=0.99]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
