# Strategy study `zero_cost_status` on scenario `long_360d`

7 configurations x 1 stored cells (1 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 925398d, scenario spec hash e23742608cc0.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.9]` | 1 | +8.12 | n/a | +17.27% | 4.48% | 1838.0 | 100% | 100 |
| 2 | `cq[entry_quantile=0.95]` | 1 | +5.81 | n/a | +10.29% | 4.88% | 1011.0 | 100% | 95 |
| 3 | `nk0` | 1 | +5.29 | n/a | +14.20% | 8.26% | 1563.0 | 100% | 100 |
| 4 | `gt` | 1 | +3.81 | n/a | +5.53% | 2.14% | 133.0 | 100% | 90 |
| 5 | `cq[entry_quantile=0.99]` | 1 | +1.44 | n/a | +1.69% | 7.10% | 263.0 | 100% | 70 |
| 6 | `ec0` | 1 | +1.00 | n/a | +2.44% | 7.97% | 2205.0 | 100% | 60 |
| 7 | `vt` | 1 | -2.70 | n/a | -5.95% | 11.52% | 326.0 | 0% | 5 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.9]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `cq[entry_quantile=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `nk0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `gt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `cq[entry_quantile=0.99]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `ec0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `vt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
