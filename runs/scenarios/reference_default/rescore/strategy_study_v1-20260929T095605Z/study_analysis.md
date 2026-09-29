# Strategy study v1: guard-rails and winner (SPEC rules)

Dev bars pooled: 14472; K = 12 candidates (20 with twins). Ranked by mean dev net Sharpe; test columns shown, never used.

```
{
  "config_id": "always_flat",
  "reason": "no candidate passes every guard-rail"
}
```

| config_id | dev_sharpe_mean | dev_sharpe_sd | dev_return_mean | dev_excess_vs_bh_mean | dev_null_pct_min_fold | dev_maxdd_mean | dev_trades_mean | dev_breakeven_bps_mean | g1_beats_flat | g2_beats_bh_paired | g3_beats_null | g4_drawdown | g5_activity | passes | test_sharpe_mean | test_return_mean | test_bh_return_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nk[f=0.25] | -5.372 | 7.044 | -0.003479 | 0.0423 | 34.5 | 0.005102 | 4.833 | -31.73 | False | True | False | True | False | False | 0 | 0 | 0.04097 |
| nk[f=0.5] | -5.818 | 6.817 | -0.003764 | 0.04202 | 34.5 | 0.005386 | 4.833 | -34.89 | False | True | False | True | False | False | 0 | 0 | 0.04097 |
| ec | -6.787 | 10.13 | -0.009144 | 0.03664 | 63 | 0.01445 | 3.333 | 2.432 | False | True | False | True | False | False | 0.6674 | 0.0004597 | 0.04097 |
| rs_ewma[q_out=0.8] | -8.281 | 5.903 | -0.03082 | 0.01496 | 11 | 0.05632 | 6.5 | -25.98 | False | True | False | True | False | False | 5.924 | 0.03424 | 0.04097 |
| vt_ewma[band=0.25] | -8.384 | 6.63 | -0.04152 | 0.004259 | 0 | 0.07061 | 9.5 | -196.7 | False | True | False | True | False | False | 6.16 | 0.03494 | 0.04097 |
| vt_ewma[band=0.1] | -9.446 | 5.063 | -0.04428 | 0.0015 | 2 | 0.0751 | 27 | -152.5 | False | True | False | True | True | False | 6.454 | 0.03633 | 0.04097 |
| vt[band=0.25] | -10.35 | 4.729 | -0.05354 | -0.007762 | 11 | 0.09105 | 19.33 | -117.6 | False | False | False | True | False | False | 6.687 | 0.03925 | 0.04097 |
| vt[band=0.1] | -11.35 | 4.918 | -0.05806 | -0.01228 | 1 | 0.09401 | 44.67 | -74.36 | False | False | False | True | True | False | 6.427 | 0.03644 | 0.04097 |
| rs_ewma[q_out=0.95] | -15.93 | 3.767 | -0.07413 | -0.02835 | 1 | 0.09812 | 4 | -197.5 | False | False | False | False | False | False | 7.306 | 0.04436 | 0.04097 |
| gt[primary=ma_cross,q=0.8] | -18.22 | 12.98 | -0.0517 | -0.005924 | 64 | 0.06453 | 24.67 | 5.987 | False | False | False | True | True | False | -3.406 | -0.004185 | 0.04097 |
| rs[q_out=0.95] | -28.81 | 4.869 | -0.1268 | -0.08103 | 1.333 | 0.1446 | 33.67 | -19.32 | False | False | False | False | True | False | 4.803 | 0.02812 | 0.04097 |
| gt_ewma[primary=ma_cross,q=0.8] | -30.29 | 13.7 | -0.08734 | -0.04156 | 49 | 0.09218 | 35 | -1.086 | False | False | False | True | True | False | -4.878 | -0.005119 | 0.04097 |
| gt[primary=ma_cross,q=0.5] | -31.84 | 4.431 | -0.1135 | -0.06774 | 57.67 | 0.1198 | 49.17 | 1.088 | False | False | False | False | True | False | -33.29 | -0.09986 | 0.04097 |
| gt_ewma[primary=ma_cross,q=0.5] | -37.32 | 13.17 | -0.1335 | -0.08774 | 58 | 0.1373 | 57 | -0.2761 | False | False | False | False | True | False | -10.1 | -0.02576 | 0.04097 |
| gt[primary=bollinger,q=0.8] | -50.35 | 23.81 | -0.2112 | -0.1654 | 98.67 | 0.2148 | 94.33 | 0.3484 | False | False | True | False | True | False | -31.12 | -0.06138 | 0.04097 |
| gt_ewma[primary=bollinger,q=0.8] | -53.15 | 24.35 | -0.2207 | -0.175 | 97 | 0.2231 | 98.5 | -0.6542 | False | False | True | False | True | False | -29 | -0.02136 | 0.04097 |
| rs[q_out=0.8] | -64.41 | 29.79 | -0.2231 | -0.1773 | 1.667 | 0.2267 | 90.17 | -3.725 | False | False | False | False | True | False | -10.53 | -0.05886 | 0.04097 |
| gt[primary=bollinger,q=0.5] | -75.52 | 23.34 | -0.3337 | -0.2879 | 98.33 | 0.3365 | 158.3 | -0.1057 | False | False | True | False | True | False | -76.54 | -0.2619 | 0.04097 |
| gt_ewma[primary=bollinger,q=0.5] | -80.71 | 37.42 | -0.3475 | -0.3017 | 87 | 0.3483 | 169.5 | -0.6802 | False | False | False | False | True | False | -43.31 | -0.1152 | 0.04097 |
| cq | -156 | 33.26 | -0.6237 | -0.5779 | 88 | 0.6237 | 392.8 | 0.3063 | False | False | False | False | True | False | -115.3 | -0.4287 | 0.04097 |
