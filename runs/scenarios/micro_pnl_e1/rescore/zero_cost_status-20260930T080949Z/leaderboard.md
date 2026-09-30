# Strategy study `zero_cost_status` on scenario `micro_pnl_e1`

7 configurations x 6 stored cells (6 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 925398d, scenario spec hash 3fecc0829644.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 6 | +3.24 | 2.05 | +3.74% | 3.57% | 169.5 | 100% | 72 |
| 2 | `cq[entry_quantile=0.99]` | 6 | -1.41 | 2.33 | -2.00% | 5.32% | 209.3 | 83% | 32 |
| 3 | `vt` | 6 | -2.02 | 0.15 | -5.49% | 12.97% | 119.5 | 100% | 37 |
| 4 | `ec0` | 6 | -2.19 | 3.98 | -5.72% | 14.45% | 2046.0 | 50% | 28 |
| 5 | `nk0` | 6 | -2.50 | 1.62 | -6.21% | 13.79% | 848.7 | 67% | 38 |
| 6 | `cq[entry_quantile=0.9]` | 6 | -2.92 | 3.20 | -5.90% | 11.16% | 1313.3 | 67% | 28 |
| 7 | `cq[entry_quantile=0.95]` | 6 | -3.04 | 3.47 | -5.37% | 9.98% | 751.8 | 67% | 19 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `cq[entry_quantile=0.99]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `vt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `ec0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `nk0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `cq[entry_quantile=0.9]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `cq[entry_quantile=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
