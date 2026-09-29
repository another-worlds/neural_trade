# Strategy study `strategy_study_v1` on scenario `reference_default`

20 configurations x 9 stored cells (6 dev, 3 test); 0 run directories skipped, 0 cells without a usable run (meta.json lists them). Git 80c15a4, scenario spec hash 4ea988c880ba.

Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample block: next-open fills, stops on high/low, the scenario's backtest settings (the default cost profile (13 bps per side)) with the entry's on top.

**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).

## Ranking (dev cells)

| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `nk[f=0.25]` | 6 | -5.37 | 7.04 | -0.35% | 0.51% | 4.8 | 100% | 42 |
| 2 | `nk[f=0.5]` | 6 | -5.82 | 6.82 | -0.38% | 0.54% | 4.8 | 100% | 42 |
| 3 | `ec` | 6 | -6.79 | 10.13 | -0.91% | 1.45% | 3.3 | 67% | 66 |
| 4 | `rs_ewma[q_out=0.8]` | 6 | -8.28 | 5.90 | -3.08% | 5.63% | 6.5 | 100% | 36 |
| 5 | `vt_ewma[band=0.25]` | 6 | -8.38 | 6.63 | -4.15% | 7.06% | 9.5 | 100% | 42 |
| 6 | `vt_ewma[band=0.1]` | 6 | -9.45 | 5.06 | -4.43% | 7.51% | 27.0 | 100% | 28 |
| 7 | `vt[band=0.25]` | 6 | -10.35 | 4.73 | -5.35% | 9.11% | 19.3 | 17% | 36 |
| 8 | `vt[band=0.1]` | 6 | -11.35 | 4.92 | -5.81% | 9.40% | 44.7 | 0% | 18 |
| 9 | `rs_ewma[q_out=0.95]` | 6 | -15.93 | 3.77 | -7.41% | 9.81% | 4.0 | 0% | 7 |
| 10 | `gt[primary=ma_cross,q=0.8]` | 6 | -18.22 | 12.98 | -5.17% | 6.45% | 24.7 | 50% | 76 |
| 11 | `rs[q_out=0.95]` | 6 | -28.81 | 4.87 | -12.68% | 14.46% | 33.7 | 0% | 8 |
| 12 | `gt_ewma[primary=ma_cross,q=0.8]` | 6 | -30.29 | 13.70 | -8.73% | 9.22% | 35.0 | 50% | 56 |
| 13 | `gt[primary=ma_cross,q=0.5]` | 6 | -31.84 | 4.43 | -11.35% | 11.98% | 49.2 | 0% | 69 |
| 14 | `gt_ewma[primary=ma_cross,q=0.5]` | 6 | -37.32 | 13.17 | -13.35% | 13.73% | 57.0 | 0% | 64 |
| 15 | `gt[primary=bollinger,q=0.8]` | 6 | -50.35 | 23.81 | -21.12% | 21.48% | 94.3 | 0% | 99 |
| 16 | `gt_ewma[primary=bollinger,q=0.8]` | 6 | -53.15 | 24.35 | -22.07% | 22.31% | 98.5 | 0% | 98 |
| 17 | `rs[q_out=0.8]` | 6 | -64.41 | 29.79 | -22.31% | 22.67% | 90.2 | 0% | 2 |
| 18 | `gt[primary=bollinger,q=0.5]` | 6 | -75.52 | 23.34 | -33.37% | 33.65% | 158.3 | 0% | 99 |
| 19 | `gt_ewma[primary=bollinger,q=0.5]` | 6 | -80.71 | 37.42 | -34.75% | 34.83% | 169.5 | 0% | 93 |
| 20 | `cq` | 6 | -155.97 | 33.26 | -62.37% | 62.37% | 392.8 | 0% | 94 |

## Test cells (shown, never used to rank or choose; D-020)

Same order as the ranking above.

| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | beats buy-and-hold | random-null pctile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `nk[f=0.25]` | 3 | +0.00 | 0.00 | +0.00% | 0.00% | 0.0 | 0% | n/a |
| 2 | `nk[f=0.5]` | 3 | +0.00 | 0.00 | +0.00% | 0.00% | 0.0 | 0% | n/a |
| 3 | `ec` | 3 | +0.67 | 1.16 | +0.05% | 0.15% | 0.3 | 0% | 93 |
| 4 | `rs_ewma[q_out=0.8]` | 3 | +5.92 | 0.00 | +3.42% | 5.35% | 3.0 | 0% | 74 |
| 5 | `vt_ewma[band=0.25]` | 3 | +6.16 | 0.00 | +3.49% | 4.89% | 5.0 | 0% | 52 |
| 6 | `vt_ewma[band=0.1]` | 3 | +6.45 | 0.00 | +3.63% | 4.79% | 8.0 | 0% | 54 |
| 7 | `vt[band=0.25]` | 3 | +6.69 | 0.76 | +3.92% | 4.85% | 6.3 | 33% | 60 |
| 8 | `vt[band=0.1]` | 3 | +6.43 | 0.58 | +3.64% | 4.63% | 19.3 | 0% | 69 |
| 9 | `rs_ewma[q_out=0.95]` | 3 | +7.31 | 0.00 | +4.44% | 4.95% | 1.0 | 100% | 97 |
| 10 | `gt[primary=ma_cross,q=0.8]` | 3 | -3.41 | 7.03 | -0.42% | 2.03% | 4.7 | 0% | 79 |
| 11 | `rs[q_out=0.95]` | 3 | +4.80 | 0.98 | +2.81% | 5.70% | 4.7 | 0% | 74 |
| 12 | `gt_ewma[primary=ma_cross,q=0.8]` | 3 | -4.88 | 0.00 | -0.51% | 1.44% | 2.0 | 0% | 59 |
| 13 | `gt[primary=ma_cross,q=0.5]` | 3 | -33.29 | 3.20 | -9.99% | 10.43% | 28.7 | 0% | 19 |
| 14 | `gt_ewma[primary=ma_cross,q=0.5]` | 3 | -10.10 | 0.00 | -2.58% | 3.24% | 12.0 | 0% | 83 |
| 15 | `gt[primary=bollinger,q=0.8]` | 3 | -31.12 | 5.73 | -6.14% | 6.62% | 18.0 | 0% | 48 |
| 16 | `gt_ewma[primary=bollinger,q=0.8]` | 3 | -29.00 | 0.00 | -2.14% | 2.14% | 4.0 | 0% | 5 |
| 17 | `rs[q_out=0.8]` | 3 | -10.53 | 3.83 | -5.89% | 11.02% | 30.0 | 0% | 66 |
| 18 | `gt[primary=bollinger,q=0.5]` | 3 | -76.54 | 6.63 | -26.19% | 26.19% | 92.3 | 0% | 32 |
| 19 | `gt_ewma[primary=bollinger,q=0.5]` | 3 | -43.31 | 0.00 | -11.52% | 11.52% | 36.0 | 0% | 55 |
| 20 | `cq` | 3 | -115.27 | 14.79 | -42.87% | 42.93% | 224.3 | 0% | 88 |
