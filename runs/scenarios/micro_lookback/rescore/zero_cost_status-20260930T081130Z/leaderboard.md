# Strategy study `zero_cost_status` on scenario `micro_lookback`

7 configurations x 3 stored cells (3 dev, 0 test); 5 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 925398d, scenario spec hash 8db20edec4ae.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 3 | +3.37 | 1.43 | +3.96% | 2.33% | 170.0 | 100% | 80 |
| 2 | `nk0` | 3 | +3.07 | 2.47 | +7.40% | 6.75% | 1313.7 | 100% | 88 |
| 3 | `ec0` | 3 | +2.21 | 2.50 | +5.83% | 8.65% | 2046.0 | 100% | 82 |
| 4 | `cq[entry_quantile=0.95]` | 3 | +1.85 | 1.75 | +3.31% | 4.77% | 805.0 | 100% | 65 |
| 5 | `cq[entry_quantile=0.9]` | 3 | +0.48 | 0.70 | +0.76% | 6.37% | 1437.3 | 100% | 48 |
| 6 | `cq[entry_quantile=0.99]` | 3 | -0.52 | 3.79 | -0.78% | 5.31% | 231.3 | 100% | 50 |
| 7 | `vt` | 3 | -2.35 | 0.13 | -6.13% | 13.15% | 146.0 | 100% | 20 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `nk0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `ec0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `cq[entry_quantile=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `cq[entry_quantile=0.9]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `cq[entry_quantile=0.99]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `vt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
