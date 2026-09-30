# Strategy study `zero_cost_status` on scenario `micro_l2`

7 configurations x 15 stored cells (15 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 925398d, scenario spec hash 4609f7f14e0d.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 15 | +3.14 | 2.14 | +3.81% | 3.97% | 158.7 | 100% | 75 |
| 2 | `ec0` | 15 | -0.21 | 4.65 | -0.28% | 11.99% | 2046.0 | 67% | 43 |
| 3 | `cq[entry_quantile=0.9]` | 15 | -1.10 | 2.39 | -2.37% | 7.86% | 1280.9 | 73% | 41 |
| 4 | `nk0` | 15 | -1.18 | 2.48 | -3.04% | 11.10% | 1236.1 | 80% | 52 |
| 5 | `cq[entry_quantile=0.99]` | 15 | -1.23 | 1.48 | -1.71% | 4.89% | 192.6 | 100% | 30 |
| 6 | `cq[entry_quantile=0.95]` | 15 | -1.34 | 2.86 | -2.48% | 7.35% | 731.3 | 73% | 31 |
| 7 | `vt` | 15 | -1.97 | 0.16 | -5.30% | 12.71% | 129.1 | 100% | 62 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `ec0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `cq[entry_quantile=0.9]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `nk0` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `cq[entry_quantile=0.99]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `cq[entry_quantile=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `vt` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
