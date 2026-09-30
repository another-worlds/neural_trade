# Strategy study `zero_cost_status` on scenario `micro_horizons`

7 configurations x 3 stored cells (3 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 925398d, scenario spec hash db231b2fbbcc.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 3 | +1.19 | 1.97 | +1.30% | 3.52% | 102.0 | 100% | 57 |
| 2 | `ec0` | 3 | +1.12 | 3.69 | +2.91% | 8.72% | 2046.0 | 67% | 48 |
| 3 | `cq[entry_quantile=0.95]` | 3 | -0.63 | 1.96 | -1.18% | 7.37% | 923.3 | 100% | 45 |
| 4 | `cq[entry_quantile=0.99]` | 3 | -1.20 | 2.96 | -1.48% | 4.71% | 238.0 | 100% | 40 |
| 5 | `cq[entry_quantile=0.9]` | 3 | -1.49 | 1.30 | -3.07% | 10.14% | 1668.0 | 100% | 35 |
| 6 | `vt` | 3 | -1.70 | 0.27 | -4.42% | 11.52% | 219.0 | 100% | 82 |
| 7 | `nk0` | 3 | -1.86 | 2.91 | -4.68% | 10.92% | 1258.7 | 67% | 38 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `ec0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `cq[entry_quantile=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `cq[entry_quantile=0.99]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `cq[entry_quantile=0.9]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `vt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `nk0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
