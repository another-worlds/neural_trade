# Strategy study `zero_cost_v1` on scenario `long_360d`

40 configurations x 1 stored cells (1 dev, 0 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git b65f32a, scenario spec hash e23742608cc0.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top.

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq_0c` | 1 | +8.12 | n/a | +17.27% | 4.48% | 1838.0 | 100% | 100 |
| 2 | `gt_0c[primary=ma_cross,q=0.8]` | 1 | +3.81 | n/a | +5.53% | 2.14% | 133.0 | 100% | 86 |
| 3 | `gt_ewma_0c[primary=ma_cross,q=0.8]` | 1 | +3.79 | n/a | +5.20% | 2.67% | 129.0 | 100% | 86 |
| 4 | `gt_ewma_0c[primary=ma_cross,q=0.5]` | 1 | +0.94 | n/a | +1.52% | 6.37% | 330.0 | 100% | 66 |
| 5 | `gt_0c[primary=ma_cross,q=0.5]` | 1 | +0.44 | n/a | +0.62% | 6.31% | 360.0 | 100% | 61 |
| 6 | `ec_0c` | 1 | +0.12 | n/a | +0.04% | 1.70% | 7.0 | 100% | 50 |
| 7 | `vt_0c[band=0.25]` | 1 | -1.68 | n/a | -3.77% | 9.79% | 130.0 | 100% | 49 |
| 8 | `vt_ewma_0c[band=0.1]` | 1 | -2.11 | n/a | -4.82% | 11.41% | 188.0 | 100% | 27 |
| 9 | `vt_ewma_0c[band=0.25]` | 1 | -2.32 | n/a | -5.01% | 10.08% | 56.0 | 0% | 21 |
| 10 | `gt_ewma_0c[primary=bollinger,q=0.5]` | 1 | -2.68 | n/a | -5.59% | 8.48% | 930.0 | 0% | 21 |
| 11 | `vt_0c[band=0.1]` | 1 | -2.70 | n/a | -5.95% | 11.52% | 326.0 | 0% | 6 |
| 12 | `rs_0c[q_out=0.95]` | 1 | -2.75 | n/a | -6.30% | 12.52% | 91.0 | 0% | 19 |
| 13 | `gt_ewma_0c[primary=bollinger,q=0.8]` | 1 | -2.80 | n/a | -4.73% | 7.36% | 382.0 | 100% | 23 |
| 14 | `rs_ewma_0c[q_out=0.95]` | 1 | -3.30 | n/a | -7.85% | 12.45% | 22.0 | 0% | 16 |
| 15 | `nk_0c[f=0.25]` | 1 | -3.43 | n/a | -1.15% | 1.78% | 13.0 | 100% | 7 |
| 16 | `nk_0c[f=0.5]` | 1 | -3.43 | n/a | -1.15% | 1.78% | 13.0 | 100% | 7 |
| 17 | `ec` | 1 | -3.47 | n/a | -1.76% | 2.59% | 7.0 | 100% | 96 |
| 18 | `vt_ewma[band=0.25]` | 1 | -3.53 | n/a | -7.40% | 11.11% | 58.0 | 0% | 15 |
| 19 | `rs_0c[q_out=0.8]` | 1 | -3.70 | n/a | -6.36% | 9.43% | 201.0 | 0% | 13 |
| 20 | `rs_ewma_0c[q_out=0.8]` | 1 | -3.98 | n/a | -7.30% | 10.74% | 44.0 | 0% | 16 |
| 21 | `vt_ewma[band=0.1]` | 1 | -4.21 | n/a | -9.16% | 12.89% | 188.0 | 0% | 20 |
| 22 | `vt[band=0.25]` | 1 | -4.59 | n/a | -9.57% | 12.36% | 130.0 | 0% | 35 |
| 23 | `nk[f=0.25]` | 1 | -4.61 | n/a | -1.59% | 1.99% | 14.0 | 100% | 9 |
| 24 | `nk[f=0.5]` | 1 | -4.61 | n/a | -1.59% | 1.99% | 14.0 | 100% | 9 |
| 25 | `gt_0c[primary=bollinger,q=0.5]` | 1 | -4.64 | n/a | -9.67% | 11.47% | 1097.0 | 0% | 8 |
| 26 | `gt_0c[primary=bollinger,q=0.8]` | 1 | -4.88 | n/a | -8.74% | 9.71% | 481.0 | 0% | 6 |
| 27 | `rs_ewma[q_out=0.95]` | 1 | -5.68 | n/a | -12.99% | 14.94% | 22.0 | 0% | 15 |
| 28 | `vt[band=0.1]` | 1 | -7.33 | n/a | -15.02% | 17.10% | 326.0 | 0% | 1 |
| 29 | `rs_ewma[q_out=0.8]` | 1 | -9.95 | n/a | -17.34% | 19.24% | 44.0 | 0% | 7 |
| 30 | `rs[q_out=0.95]` | 1 | -12.90 | n/a | -26.07% | 27.24% | 91.0 | 0% | 3 |
| 31 | `gt[primary=ma_cross,q=0.8]` | 1 | -18.44 | n/a | -25.35% | 25.51% | 133.0 | 0% | 100 |
| 32 | `gt_ewma[primary=ma_cross,q=0.8]` | 1 | -18.93 | n/a | -24.80% | 24.87% | 129.0 | 0% | 100 |
| 33 | `rs[q_out=0.8]` | 1 | -30.94 | n/a | -44.51% | 45.31% | 201.0 | 0% | 0 |
| 34 | `gt_ewma[primary=ma_cross,q=0.5]` | 1 | -41.34 | n/a | -57.00% | 57.12% | 330.0 | 0% | 99 |
| 35 | `gt[primary=ma_cross,q=0.5]` | 1 | -43.47 | n/a | -60.58% | 60.65% | 360.0 | 0% | 97 |
| 36 | `gt_ewma[primary=bollinger,q=0.8]` | 1 | -52.01 | n/a | -64.77% | 64.77% | 382.0 | 0% | 100 |
| 37 | `gt[primary=bollinger,q=0.8]` | 1 | -61.05 | n/a | -73.92% | 74.01% | 481.0 | 0% | 100 |
| 38 | `gt_ewma[primary=bollinger,q=0.5]` | 1 | -93.94 | n/a | -91.62% | 91.63% | 930.0 | 0% | 100 |
| 39 | `gt[primary=bollinger,q=0.5]` | 1 | -106.35 | n/a | -94.81% | 94.83% | 1097.0 | 0% | 100 |
| 40 | `cq` | 1 | -153.60 | n/a | -99.02% | 99.02% | 1838.0 | 0% | 100 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cq_0c` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | `gt_0c[primary=ma_cross,q=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | `gt_ewma_0c[primary=ma_cross,q=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | `gt_ewma_0c[primary=ma_cross,q=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 5 | `gt_0c[primary=ma_cross,q=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6 | `ec_0c` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | `vt_0c[band=0.25]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 8 | `vt_ewma_0c[band=0.1]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 9 | `vt_ewma_0c[band=0.25]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 10 | `gt_ewma_0c[primary=bollinger,q=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 11 | `vt_0c[band=0.1]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 12 | `rs_0c[q_out=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 13 | `gt_ewma_0c[primary=bollinger,q=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 14 | `rs_ewma_0c[q_out=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 15 | `nk_0c[f=0.25]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 16 | `nk_0c[f=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 17 | `ec` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 18 | `vt_ewma[band=0.25]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 19 | `rs_0c[q_out=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 20 | `rs_ewma_0c[q_out=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 21 | `vt_ewma[band=0.1]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 22 | `vt[band=0.25]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 23 | `nk[f=0.25]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 24 | `nk[f=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 25 | `gt_0c[primary=bollinger,q=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 26 | `gt_0c[primary=bollinger,q=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 27 | `rs_ewma[q_out=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 28 | `vt[band=0.1]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 29 | `rs_ewma[q_out=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 30 | `rs[q_out=0.95]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 31 | `gt[primary=ma_cross,q=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 32 | `gt_ewma[primary=ma_cross,q=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 33 | `rs[q_out=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 34 | `gt_ewma[primary=ma_cross,q=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 35 | `gt[primary=ma_cross,q=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 36 | `gt_ewma[primary=bollinger,q=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 37 | `gt[primary=bollinger,q=0.8]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 38 | `gt_ewma[primary=bollinger,q=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 39 | `gt[primary=bollinger,q=0.5]` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 40 | `cq` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
