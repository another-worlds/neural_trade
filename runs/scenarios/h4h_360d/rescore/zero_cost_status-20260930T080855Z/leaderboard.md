# Strategy study `zero_cost_status` on scenario `h4h_360d`

7 configurations x 1 stored cells (1 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 925398d, scenario spec hash 5bf4b03856ca.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.9]` | 1 | +7.59 | n/a | +14.72% | 4.32% | 1194.0 | 100% | 95 |
| 2 | `cq[entry_quantile=0.95]` | 1 | +6.85 | n/a | +10.93% | 3.62% | 608.0 | 100% | 95 |
| 3 | `gt` | 1 | +6.01 | n/a | +8.95% | 1.99% | 146.0 | 100% | 95 |
| 4 | `nk0` | 1 | +4.60 | n/a | +10.25% | 5.61% | 1896.0 | 100% | 100 |
| 5 | `cq[entry_quantile=0.99]` | 1 | +4.28 | n/a | +4.52% | 2.63% | 123.0 | 100% | 85 |
| 6 | `ec0` | 1 | +2.57 | n/a | +7.09% | 8.06% | 2205.0 | 100% | 95 |
| 7 | `vt` | 1 | -2.46 | n/a | -5.80% | 11.84% | 285.0 | 0% | 0 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.9]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `cq[entry_quantile=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `gt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `nk0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `cq[entry_quantile=0.99]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `ec0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `vt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
