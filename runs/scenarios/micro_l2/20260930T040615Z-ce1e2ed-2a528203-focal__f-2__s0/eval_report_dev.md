# Evaluation report - dev split - run `20260930T040615Z-ce1e2ed-2a528203-focal__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.7877 | 0.8200 | 0.9278 |
| accuracy | 0.4817 | 0.4800 | 0.4770 |
| balanced accuracy | 0.4918 | 0.4909 | 0.4973 |
| precision (up) | 0.4772 | 0.4774 | 0.4748 |
| recall / sensitivity (up) | 0.7793 | 0.8107 | 0.9249 |
| specificity (down) | 0.2044 | 0.1712 | 0.0696 |
| F1 (up) | 0.5920 | 0.6009 | 0.6275 |
| MCC | -0.0199 | -0.0236 | -0.0106 |
| AUC | 0.4957 | 0.4902 | 0.5069 |
| Brier | 0.2588 | 0.2523 | 0.2597 |
| ECE (positive class) | 0.0820 | 0.0469 | 0.0912 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 13644 / 14945 / 3839 / 3864 | 14801 / 16201 / 3347 / 3457 | 17377 / 19218 / 1438 / 1411 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.7689 | 0.6667 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | 0.0079 | 0.0045 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4963 | 0.5006 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2507 | 0.2519 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0277 | 0.0421 |
| Gaussian readout of the raw heads: calls up | 0.9968 | 0.7689 | 0.6667 |
| Gaussian readout of the raw heads: MCC | -0.0326 | 0.0079 | 0.0045 |
| Gaussian readout of the raw heads: AUC | 0.4696 | 0.4963 | 0.5006 |
| Gaussian readout of the raw heads: Brier | 0.2737 | 0.2599 | 0.2703 |
| Gaussian readout of the raw heads: ECE | 0.1391 | 0.0756 | 0.1066 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.31 | 565.34 | 811.10 |
| RMSE ($), raw heads | 417.47 | 579.60 | 843.31 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.46 | 378.82 | 553.92 |
| MAE ($), raw heads | 289.65 | 391.60 | 585.89 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0052 | -0.0086 |
| skill vs zero, raw heads | -0.0876 | -0.0565 | -0.0903 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0029 | -0.0036 |
| EV, raw heads | -0.0217 | -0.0320 | -0.0511 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0690 | -0.0441 | -0.0108 |
| corr, Spearman, raw heads | -0.0608 | -0.0229 | -0.0102 |
| mean predicted ($), served | 0.00 | 12.63 | 28.68 |
| mean predicted ($), raw heads | 92.36 | 69.00 | 123.07 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9969 | 0.7636 | 0.6642 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1830 | 0.2330 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.16 | 285.52 | 419.59 |
| CRPSS vs constant variance | 0.0213 | 0.0311 | 0.0567 |
| NLL | 7.4217 | 7.7806 | 8.1731 |
| PIT KS | 0.0442 | 0.0656 | 0.0661 |
| var / err^2 Spearman | 0.1403 | 0.0865 | 0.0786 |
| coverage of the 90% interval | 0.9028 | 0.9009 | 0.8624 |
| width of the 90% interval ($) | 1212.02 | 1748.69 | 2393.90 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0029 | [-0.0305, 0.0275] | NOISE |
| h1 | -0.0149 | [-0.0381, 0.0071] | NOISE |
| h2 | 0.0199 | [-0.0195, 0.0609] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.183 / h2 0.233) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.3623 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.8379 | 0.8829 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.3391 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7735 | 0.6797 | 0.6689 | 0.4351 |
| expected if the two signs were independent | 0.7749 | 0.6729 | 0.6399 | 0.3822 |

- P(up) unanimity (all three horizons call the same side): 0.6090

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0199 vs 0.0025 (-0.0224): does not beat, noise (boot z -0.73) | -0.0236 vs 0.0205 (-0.0441): does not beat, noise (boot z -1.77) | -0.0106 vs -0.0138 (+0.0032): beats, noise (boot z +0.12) |
| logreg_lags | direction/auc | 0.4957 vs 0.5320 (-0.0364): does not beat, noise (boot z -1.60) | 0.4902 vs 0.5251 (-0.0348): does not beat, noise (boot z -1.68) | 0.5069 vs 0.5095 (-0.0025): does not beat, noise (boot z -0.29) |
| logreg_lags | direction/brier | 0.2588 vs 0.2536 (-0.0052): does not beat, significantly worse (DM z -2.66) | 0.2523 vs 0.2575 (+0.0051): beats, noise (DM z +1.79) | 0.2597 vs 0.2702 (+0.0105): beats (DM z +3.40) |
| logreg_lags | direction/ece_pos | 0.0820 vs 0.0663 (-0.0157): does not beat, noise (boot z -1.69) | 0.0469 vs 0.0812 (+0.0343): beats (boot z +4.28) | 0.0912 vs 0.1230 (+0.0318): beats (boot z +6.99) |
| logreg_lags | direction/acc | 0.4817 vs 0.4836 (-0.0019): does not beat, noise (DM z -0.17) | 0.4800 vs 0.4911 (-0.0111): does not beat, noise (DM z -1.14) | 0.4770 vs 0.4756 (+0.0014): beats, noise (DM z +0.18) |
| logreg_lags | direction/bal_acc | 0.4918 vs 0.5004 (-0.0085): does not beat, noise (boot z -0.93) | 0.4909 vs 0.5055 (-0.0146): does not beat, noise (boot z -1.87) | 0.4973 vs 0.4971 (+0.0001): beats, noise (boot z +0.02) |
| class_prior | direction/mcc | -0.0199 vs 0.0000 (-0.0199): does not beat, noise (boot z -1.00) | -0.0236 vs 0.0000 (-0.0236): does not beat, noise (boot z -1.65) | -0.0106 vs 0.0000 (-0.0106): does not beat, noise (boot z -0.54) |
| class_prior | direction/auc | 0.4957 vs 0.5000 (-0.0043): does not beat, noise (boot z -0.32) | 0.4902 vs 0.5000 (-0.0098): does not beat, noise (boot z -1.15) | 0.5069 vs 0.5000 (+0.0069): beats, noise (boot z +0.45) |
| class_prior | direction/brier | 0.2588 vs 0.2532 (-0.0056): does not beat, significantly worse (DM z -3.49) | 0.2523 vs 0.2533 (+0.0010): beats, noise (DM z +0.89) | 0.2597 vs 0.2573 (-0.0024): does not beat, noise (DM z -1.24) |
| class_prior | direction/ece_pos | 0.0820 vs 0.0593 (-0.0227): does not beat, significantly worse (boot z -2.47) | 0.0469 vs 0.0603 (+0.0134): beats, noise (boot z +1.70) | 0.0912 vs 0.0886 (-0.0025): does not beat, noise (boot z -0.62) |
| class_prior | direction/acc | 0.4817 vs 0.4824 (-0.0007): does not beat, noise (DM z -0.06) | 0.4800 vs 0.4829 (-0.0029): does not beat, noise (DM z -0.28) | 0.4770 vs 0.4763 (+0.0007): beats, noise (DM z +0.08) |
| class_prior | direction/bal_acc | 0.4918 vs 0.5000 (-0.0082): does not beat, noise (boot z -1.00) | 0.4909 vs 0.5000 (-0.0091): does not beat, noise (boot z -1.64) | 0.4973 vs 0.5000 (-0.0027): does not beat, noise (boot z -0.53) |
| zero_delta | delta/rmse | 400.31 vs 400.31 (+0.00, +0.00%): does not beat | 565.34 vs 563.88 (-1.46, -0.26%): does not beat, significantly worse (DM z -2.12) | 811.10 vs 807.63 (-3.47, -0.43%): does not beat, noise (DM z -1.43) |
| zero_delta | delta/mae | 271.46 vs 271.46 (+0.00, +0.00%): does not beat | 378.82 vs 377.88 (-0.95, -0.25%): does not beat, noise (DM z -1.70) | 553.92 vs 550.98 (-2.94, -0.53%): does not beat, noise (DM z -1.57) |
| mean_delta | delta/rmse | 400.31 vs 405.60 (+5.29, +1.31%): beats (DM z +2.85) | 565.34 vs 578.56 (+13.22, +2.29%): beats (DM z +2.81) | 811.10 vs 847.03 (+35.93, +4.24%): beats (DM z +2.87) |
| mean_delta | delta/mae | 271.46 vs 277.50 (+6.04, +2.18%): beats (DM z +3.94) | 378.82 vs 393.77 (+14.94, +3.80%): beats (DM z +3.90) | 553.92 vs 595.01 (+41.10, +6.91%): beats (DM z +3.88) |
| const_var | variance/crps | 202.16 vs 206.55 (+4.39, +2.13%): beats (DM z +4.64) | 285.52 vs 294.69 (+9.17, +3.11%): beats (DM z +3.74) | 419.59 vs 444.83 (+25.24, +5.67%): beats (DM z +3.72) |
| const_var | variance/nll | 7.4217 vs 7.4448 (+0.0231): beats (DM z +2.58) | 7.7806 vs 7.8022 (+0.0217): beats, noise (DM z +1.36) | 8.1731 vs 8.2104 (+0.0373): beats, noise (DM z +1.46) |
| const_var | variance/pit_ks | 0.0442 vs 0.1064 (+0.0622): beats (boot z +8.35) | 0.0656 vs 0.1404 (+0.0748): beats (boot z +13.47) | 0.0661 vs 0.1810 (+0.1150): beats (boot z +27.77) |
| const_var | variance/corr_var_err2_spearman | 0.1403 vs 0.0000 (+0.1403): beats (boot z +7.80) | 0.0865 vs 0.0000 (+0.0865): beats (boot z +3.59) | 0.0786 vs 0.0000 (+0.0786): beats (boot z +3.44) |

## Backtest (costs included)

- n_trades: 1559
- total_return: -0.9843
- sharpe_net: -144.9717
- sharpe_gross: -6.5079
- sortino: -160.4099
- max_drawdown: 0.9843
- hit_rate: 0.0577
- hit_rate_gross: 0.4400
- profit_factor: 0.0297
- avg_hold_bars: 10.9442
- exposure: 0.3950
- turnover: 723.7497
- fees_paid: 7237.7224
- traded_notional: 7237722.4484
- breakeven_cost_bps: -1.1991
- gross_edge_per_trade_bps: -0.5931
- costs_paid: 9409.0392
- gross_pnl: -433.9278
- net_pnl: -9842.9670

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.798, long_above 0.5884, short_below 0.4986, median 0.5404. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.43% | -144.97 | +98.43% | 1559 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.47% .. -97.96%) | -98.24% | -151.88 | | |

The random null enters at the strategy's rate (0.0596 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 9% of its seeds on net return, 96% on net Sharpe and 1% on gross return.
