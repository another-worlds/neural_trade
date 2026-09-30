# Strategy study `zero_cost_status` on scenario `long_360d_stab`

7 configurations x 6 stored cells (6 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git dce15ed, scenario spec hash 1bd46f3e6edf.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.95]` | 6 | +4.91 | 2.14 | +8.57% | 4.02% | 1010.0 | 67% | 88 |
| 2 | `cq[entry_quantile=0.9]` | 6 | +4.86 | 3.02 | +10.17% | 4.72% | 1738.7 | 67% | 83 |
| 3 | `nk0` | 6 | +2.24 | 2.56 | +5.38% | 7.67% | 1481.7 | 67% | 68 |
| 4 | `cq[entry_quantile=0.99]` | 6 | +1.82 | 1.73 | +1.95% | 3.78% | 254.8 | 50% | 74 |
| 5 | `ec0` | 6 | +1.66 | 2.45 | +3.99% | 7.59% | 2205.0 | 33% | 69 |
| 6 | `gt` | 6 | +1.03 | 3.01 | +1.83% | 3.25% | 111.3 | 50% | 67 |
| 7 | `vt` | 6 | +0.48 | 3.76 | +1.03% | 9.03% | 288.5 | 0% | 26 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `cq[entry_quantile=0.9]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `nk0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `cq[entry_quantile=0.99]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `ec0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `gt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `vt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
