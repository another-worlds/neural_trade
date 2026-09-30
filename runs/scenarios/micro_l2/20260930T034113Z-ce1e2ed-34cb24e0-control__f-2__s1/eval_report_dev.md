# Evaluation report - dev split - run `20260930T034113Z-ce1e2ed-34cb24e0-control__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5920 | 0.8390 | 0.8973 |
| accuracy | 0.4794 | 0.4855 | 0.4799 |
| balanced accuracy | 0.4826 | 0.4970 | 0.4988 |
| precision (up) | 0.4677 | 0.4812 | 0.4756 |
| recall / sensitivity (up) | 0.5740 | 0.8359 | 0.8960 |
| specificity (down) | 0.3912 | 0.1581 | 0.1015 |
| F1 (up) | 0.5155 | 0.6108 | 0.6214 |
| MCC | -0.0354 | -0.0081 | -0.0041 |
| AUC | 0.4822 | 0.5027 | 0.5013 |
| Brier | 0.2553 | 0.2560 | 0.2589 |
| ECE (positive class) | 0.0634 | 0.0677 | 0.0854 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 10050 / 11436 / 7348 / 7458 | 15262 / 16457 / 3091 / 2996 | 16834 / 18559 / 2097 / 1954 |
| Gaussian readout: calls up | 0.9507 | 0.9071 | 0.9033 |
| Gaussian readout: MCC | 0.0093 | 0.0049 | 0.0069 |
| Gaussian readout: AUC | 0.4974 | 0.5183 | 0.5122 |
| Gaussian readout: Brier | 0.2511 | 0.2515 | 0.2519 |
| Gaussian readout: ECE | 0.0352 | 0.0430 | 0.0489 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.45 | 566.26 | 811.13 |
| RMSE ($), raw heads | 411.59 | 577.59 | 821.72 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.29 | 379.10 | 553.97 |
| MAE ($), raw heads | 280.27 | 387.59 | 564.54 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0057 | -0.0085 | -0.0087 |
| skill vs zero, raw heads | -0.0572 | -0.0492 | -0.0352 |
| EV, served | -0.0030 | -0.0030 | -0.0010 |
| EV, raw heads | -0.0270 | -0.0205 | -0.0084 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0425 | -0.0130 | 0.0092 |
| corr, Spearman, raw heads | -0.0121 | 0.0171 | 0.0049 |
| mean predicted ($), served | 12.58 | 25.07 | 39.99 |
| mean predicted ($), raw heads | 59.41 | 75.93 | 96.32 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9491 | 0.9030 | 0.9034 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.2118 | 0.3302 | 0.4152 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.68 | 285.65 | 419.67 |
| CRPSS vs constant variance | 0.0187 | 0.0307 | 0.0565 |
| NLL | 7.4328 | 7.8338 | 8.1674 |
| PIT KS | 0.0523 | 0.0595 | 0.0757 |
| var / err^2 Spearman | 0.1006 | -0.0456 | 0.0909 |
| coverage of the 90% interval | 0.9000 | 0.9007 | 0.8646 |
| width of the 90% interval ($) | 1208.59 | 1755.76 | 2415.92 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0075 | [-0.0373, 0.0215] | NOISE |
| h1 | -0.0024 | [-0.0362, 0.0292] | NOISE |
| h2 | -0.0106 | [-0.0370, 0.0136] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.212 / h1 0.330 / h2 0.415) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6239 | 0.8782 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.5664 | 0.8287 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.3065 | 0.7336 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5849 | 0.8127 | 0.8102 | 0.4120 |
| expected if the two signs were independent | 0.5988 | 0.7681 | 0.8210 | 0.4065 |

- P(up) unanimity (all three horizons call the same side): 0.4775

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0354 vs 0.0025 (-0.0379): does not beat, noise (boot z -1.34) | -0.0081 vs 0.0205 (-0.0286): does not beat, noise (boot z -0.91) | -0.0041 vs -0.0138 (+0.0097): beats, noise (boot z +0.44) |
| logreg_lags | direction/auc | 0.4822 vs 0.5320 (-0.0498): does not beat, significantly worse (boot z -2.00) | 0.5027 vs 0.5251 (-0.0224): does not beat, noise (boot z -1.03) | 0.5013 vs 0.5095 (-0.0082): does not beat, noise (boot z -0.45) |
| logreg_lags | direction/brier | 0.2553 vs 0.2536 (-0.0017): does not beat, noise (DM z -0.67) | 0.2560 vs 0.2575 (+0.0014): beats, noise (DM z +0.56) | 0.2589 vs 0.2702 (+0.0113): beats (DM z +2.61) |
| logreg_lags | direction/ece_pos | 0.0634 vs 0.0663 (+0.0029): beats, noise (boot z +0.17) | 0.0677 vs 0.0812 (+0.0135): beats (boot z +2.00) | 0.0854 vs 0.1230 (+0.0376): beats (boot z +7.38) |
| logreg_lags | direction/acc | 0.4794 vs 0.4836 (-0.0042): does not beat, noise (DM z -0.25) | 0.4855 vs 0.4911 (-0.0057): does not beat, noise (DM z -0.46) | 0.4799 vs 0.4756 (+0.0044): beats, noise (DM z +0.61) |
| logreg_lags | direction/bal_acc | 0.4826 vs 0.5004 (-0.0178): does not beat, noise (boot z -1.63) | 0.4970 vs 0.5055 (-0.0085): does not beat, noise (boot z -0.85) | 0.4988 vs 0.4971 (+0.0016): beats, noise (boot z +0.30) |
| class_prior | direction/mcc | -0.0354 vs 0.0000 (-0.0354): does not beat, noise (boot z -1.61) | -0.0081 vs 0.0000 (-0.0081): does not beat, noise (boot z -0.45) | -0.0041 vs 0.0000 (-0.0041): does not beat, noise (boot z -0.26) |
| class_prior | direction/auc | 0.4822 vs 0.5000 (-0.0178): does not beat, noise (boot z -1.27) | 0.5027 vs 0.5000 (+0.0027): beats, noise (boot z +0.21) | 0.5013 vs 0.5000 (+0.0013): beats, noise (boot z +0.13) |
| class_prior | direction/brier | 0.2553 vs 0.2532 (-0.0021): does not beat, noise (DM z -1.04) | 0.2560 vs 0.2533 (-0.0027): does not beat, noise (DM z -1.81) | 0.2589 vs 0.2573 (-0.0016): does not beat, noise (DM z -1.48) |
| class_prior | direction/ece_pos | 0.0634 vs 0.0593 (-0.0041): does not beat, noise (boot z -0.24) | 0.0677 vs 0.0603 (-0.0074): does not beat, noise (boot z -1.19) | 0.0854 vs 0.0886 (+0.0032): beats, noise (boot z +0.72) |
| class_prior | direction/acc | 0.4794 vs 0.4824 (-0.0030): does not beat, noise (DM z -0.18) | 0.4855 vs 0.4829 (+0.0025): beats, noise (DM z +0.23) | 0.4799 vs 0.4763 (+0.0036): beats, noise (DM z +0.45) |
| class_prior | direction/bal_acc | 0.4826 vs 0.5000 (-0.0174): does not beat, noise (boot z -1.61) | 0.4970 vs 0.5000 (-0.0030): does not beat, noise (boot z -0.45) | 0.4988 vs 0.5000 (-0.0012): does not beat, noise (boot z -0.25) |
| zero_delta | delta/rmse | 401.45 vs 400.31 (-1.14, -0.29%): does not beat, noise (DM z -1.76) | 566.26 vs 563.88 (-2.38, -0.42%): does not beat, noise (DM z -1.44) | 811.13 vs 807.63 (-3.50, -0.43%): does not beat, noise (DM z -1.03) |
| zero_delta | delta/mae | 272.29 vs 271.46 (-0.83, -0.31%): does not beat, noise (DM z -1.92) | 379.10 vs 377.88 (-1.23, -0.32%): does not beat, noise (DM z -1.03) | 553.97 vs 550.98 (-2.99, -0.54%): does not beat, noise (DM z -1.23) |
| mean_delta | delta/rmse | 401.45 vs 405.60 (+4.15, +1.02%): beats (DM z +2.65) | 566.26 vs 578.56 (+12.30, +2.13%): beats (DM z +3.15) | 811.13 vs 847.03 (+35.90, +4.24%): beats (DM z +3.10) |
| mean_delta | delta/mae | 272.29 vs 277.50 (+5.21, +1.88%): beats (DM z +4.28) | 379.10 vs 393.77 (+14.67, +3.72%): beats (DM z +4.33) | 553.97 vs 595.01 (+41.04, +6.90%): beats (DM z +4.10) |
| const_var | variance/crps | 202.68 vs 206.55 (+3.87, +1.87%): beats (DM z +5.27) | 285.65 vs 294.69 (+9.04, +3.07%): beats (DM z +4.36) | 419.67 vs 444.83 (+25.15, +5.65%): beats (DM z +3.95) |
| const_var | variance/nll | 7.4328 vs 7.4448 (+0.0120): beats, noise (DM z +1.48) | 7.8338 vs 7.8022 (-0.0316): does not beat, noise (DM z -1.05) | 8.1674 vs 8.2104 (+0.0431): beats, noise (DM z +1.76) |
| const_var | variance/pit_ks | 0.0523 vs 0.1064 (+0.0541): beats (boot z +22.96) | 0.0595 vs 0.1404 (+0.0809): beats (boot z +25.34) | 0.0757 vs 0.1810 (+0.1053): beats (boot z +31.82) |
| const_var | variance/corr_var_err2_spearman | 0.1006 vs 0.0000 (+0.1006): beats (boot z +4.98) | -0.0456 vs 0.0000 (-0.0456): does not beat, noise (boot z -1.74) | 0.0909 vs 0.0000 (+0.0909): beats (boot z +3.76) |

## Backtest (costs included)

- n_trades: 1261
- total_return: -0.9632
- sharpe_net: -119.2099
- sharpe_gross: 1.8168
- sortino: -138.2813
- max_drawdown: 0.9632
- hit_rate: 0.0523
- hit_rate_gross: 0.4837
- profit_factor: 0.0361
- avg_hold_bars: 11.3251
- exposure: 0.3306
- turnover: 751.4395
- fees_paid: 7514.4381
- traded_notional: 7514438.0920
- breakeven_cost_bps: 0.3638
- gross_edge_per_trade_bps: -0.1365
- costs_paid: 9768.7695
- gross_pnl: 136.7023
- net_pnl: -9632.0672

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6955, long_above 0.5665, short_below 0.4842, median 0.5281. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.32% | -119.21 | +96.32% | 1261 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.85% .. -95.82%) | -96.34% | -136.90 | | |

The random null enters at the strategy's rate (0.0436 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 52% of its seeds on net return, 100% on net Sharpe and 73% on gross return.
