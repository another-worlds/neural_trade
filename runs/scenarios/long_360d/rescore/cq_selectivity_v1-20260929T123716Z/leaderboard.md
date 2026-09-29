# Strategy study `cq_selectivity_v1` on scenario `long_360d`

15 configurations x 1 stored cells (1 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 2642fe1, scenario spec hash e23742608cc0.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top.

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.995,max_hold=240]` | 1 | -27.34 | n/a | -28.43% | 28.55% | 125.0 | 0% | 100 |
| 2 | `cq[entry_quantile=0.995,max_hold=60]` | 1 | -27.73 | n/a | -28.76% | 28.87% | 126.0 | 0% | 100 |
| 3 | `cq[entry_quantile=0.995,max_hold=15]` | 1 | -28.65 | n/a | -29.43% | 29.43% | 135.0 | 0% | 100 |
| 4 | `cq[entry_quantile=0.99,max_hold=240]` | 1 | -41.23 | n/a | -46.40% | 46.41% | 245.0 | 0% | 100 |
| 5 | `cq[entry_quantile=0.99,max_hold=60]` | 1 | -41.56 | n/a | -46.64% | 46.65% | 246.0 | 0% | 100 |
| 6 | `cq[entry_quantile=0.99,max_hold=15]` | 1 | -44.40 | n/a | -48.72% | 48.72% | 263.0 | 0% | 100 |
| 7 | `cq[entry_quantile=0.98,max_hold=240]` | 1 | -61.43 | n/a | -67.92% | 67.92% | 454.0 | 0% | 100 |
| 8 | `cq[entry_quantile=0.98,max_hold=60]` | 1 | -61.70 | n/a | -68.07% | 68.07% | 455.0 | 0% | 100 |
| 9 | `cq[entry_quantile=0.98,max_hold=15]` | 1 | -65.79 | n/a | -70.42% | 70.42% | 486.0 | 0% | 100 |
| 10 | `cq[entry_quantile=0.95,max_hold=240]` | 1 | -96.12 | n/a | -90.04% | 90.04% | 930.0 | 0% | 100 |
| 11 | `cq[entry_quantile=0.95,max_hold=60]` | 1 | -96.39 | n/a | -90.10% | 90.10% | 931.0 | 0% | 100 |
| 12 | `cq[entry_quantile=0.95,max_hold=15]` | 1 | -105.04 | n/a | -92.07% | 92.07% | 1011.0 | 0% | 100 |
| 13 | `cq[entry_quantile=0.9,max_hold=240]` | 1 | -138.75 | n/a | -98.38% | 98.38% | 1649.0 | 0% | 100 |
| 14 | `cq[entry_quantile=0.9,max_hold=60]` | 1 | -139.04 | n/a | -98.39% | 98.39% | 1651.0 | 0% | 100 |
| 15 | `cq[entry_quantile=0.9,max_hold=15]` | 1 | -153.60 | n/a | -99.02% | 99.02% | 1838.0 | 0% | 100 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.995,max_hold=240]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `cq[entry_quantile=0.995,max_hold=60]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `cq[entry_quantile=0.995,max_hold=15]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `cq[entry_quantile=0.99,max_hold=240]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `cq[entry_quantile=0.99,max_hold=60]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `cq[entry_quantile=0.99,max_hold=15]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `cq[entry_quantile=0.98,max_hold=240]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 8 | `cq[entry_quantile=0.98,max_hold=60]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 9 | `cq[entry_quantile=0.98,max_hold=15]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 10 | `cq[entry_quantile=0.95,max_hold=240]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 11 | `cq[entry_quantile=0.95,max_hold=60]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 12 | `cq[entry_quantile=0.95,max_hold=15]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 13 | `cq[entry_quantile=0.9,max_hold=240]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 14 | `cq[entry_quantile=0.9,max_hold=60]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 15 | `cq[entry_quantile=0.9,max_hold=15]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
