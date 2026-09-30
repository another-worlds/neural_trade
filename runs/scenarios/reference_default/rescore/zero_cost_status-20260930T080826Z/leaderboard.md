# Strategy study `zero_cost_status` on scenario `reference_default`

7 configurations x 9 stored cells (6 dev, 3 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 925398d, scenario spec hash 4ea988c880ba.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top, 20 random-null seeds (override).

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.95]` | 6 | +6.27 | 7.23 | +2.68% | 3.66% | 268.0 | 100% | 78 |
| 2 | `cq[entry_quantile=0.99]` | 6 | +5.08 | 7.66 | +1.65% | 2.10% | 70.5 | 100% | 79 |
| 3 | `cq[entry_quantile=0.9]` | 6 | +3.94 | 7.56 | +1.83% | 3.80% | 392.8 | 100% | 64 |
| 4 | `gt` | 6 | +3.65 | 4.23 | +1.07% | 2.30% | 24.7 | 100% | 61 |
| 5 | `ec0` | 6 | -0.72 | 10.47 | -0.73% | 7.29% | 334.0 | 100% | 41 |
| 6 | `nk0` | 6 | -3.58 | 10.53 | -2.05% | 6.41% | 195.0 | 67% | 57 |
| 7 | `vt` | 6 | -8.14 | 5.41 | -4.24% | 8.48% | 44.7 | 67% | 18 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq[entry_quantile=0.95]` | 3 | +8.71 | 13.45 | +2.36% | 1.92% | 101.7 | 33% | 70 |
| 2 | `cq[entry_quantile=0.99]` | 3 | +2.40 | 2.04 | +0.22% | 0.54% | 4.0 | 0% | 65 |
| 3 | `cq[entry_quantile=0.9]` | 3 | +5.73 | 12.32 | +2.25% | 2.64% | 224.3 | 33% | 53 |
| 4 | `gt` | 3 | +4.01 | 7.21 | +0.80% | 1.23% | 4.7 | 0% | 65 |
| 5 | `ec0` | 3 | -2.63 | 6.45 | -1.64% | 5.35% | 334.0 | 0% | 40 |
| 6 | `nk0` | 3 | -0.70 | 9.65 | -0.48% | 4.43% | 216.3 | 33% | 62 |
| 7 | `vt` | 3 | +7.65 | 0.46 | +4.37% | 4.44% | 19.3 | 67% | 83 |
