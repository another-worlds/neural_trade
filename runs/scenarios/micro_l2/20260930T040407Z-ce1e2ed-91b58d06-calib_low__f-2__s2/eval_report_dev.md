# Evaluation report - dev split - run `20260930T040407Z-ce1e2ed-91b58d06-calib_low__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.7250 | 0.7910 | 0.5185 |
| accuracy | 0.4804 | 0.4962 | 0.4946 |
| balanced accuracy | 0.4883 | 0.5061 | 0.4954 |
| precision (up) | 0.4744 | 0.4868 | 0.4719 |
| recall / sensitivity (up) | 0.7129 | 0.7973 | 0.5137 |
| specificity (down) | 0.2638 | 0.2149 | 0.4771 |
| F1 (up) | 0.5697 | 0.6045 | 0.4919 |
| MCC | -0.0261 | 0.0151 | -0.0091 |
| AUC | 0.4884 | 0.5186 | 0.4886 |
| Brier | 0.2568 | 0.2583 | 0.2545 |
| ECE (positive class) | 0.0705 | 0.0758 | 0.0474 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 12481 / 13829 / 4955 / 5027 | 14558 / 15347 / 4201 / 3700 | 9652 / 10800 / 9856 / 9136 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.8483 | 0.7138 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | 0.0143 | 0.0234 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4968 | 0.5105 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2503 | 0.2518 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0231 | 0.0449 |
| Gaussian readout of the raw heads: calls up | 0.8232 | 0.8483 | 0.7138 |
| Gaussian readout of the raw heads: MCC | -0.0041 | 0.0143 | 0.0234 |
| Gaussian readout of the raw heads: AUC | 0.4752 | 0.4968 | 0.5105 |
| Gaussian readout of the raw heads: Brier | 0.2669 | 0.2620 | 0.2595 |
| Gaussian readout of the raw heads: ECE | 0.1043 | 0.0893 | 0.0805 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.31 | 564.58 | 812.48 |
| RMSE ($), raw heads | 416.57 | 591.20 | 829.59 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.46 | 378.37 | 555.14 |
| MAE ($), raw heads | 287.06 | 403.55 | 572.47 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0025 | -0.0121 |
| skill vs zero, raw heads | -0.0829 | -0.0992 | -0.0551 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0009 | -0.0043 |
| EV, raw heads | -0.0419 | -0.0429 | -0.0271 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0436 | -0.0231 | 0.0020 |
| corr, Spearman, raw heads | -0.0491 | -0.0181 | 0.0014 |
| mean predicted ($), served | 0.00 | 9.24 | 40.22 |
| mean predicted ($), raw heads | 70.92 | 113.65 | 99.05 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.8111 | 0.8393 | 0.7118 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0813 | 0.4061 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.98 | 292.77 | 420.12 |
| CRPSS vs constant variance | 0.0173 | 0.0065 | 0.0555 |
| NLL | 7.3837 | 7.7613 | 8.1576 |
| PIT KS | 0.0651 | 0.1100 | 0.0780 |
| var / err^2 Spearman | 0.2087 | 0.1147 | 0.1270 |
| coverage of the 90% interval | 0.9028 | 0.9018 | 0.8602 |
| width of the 90% interval ($) | 1212.02 | 1753.30 | 2373.26 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0153 | [-0.0420, 0.0115] | NOISE |
| h1 | 0.0018 | [-0.0311, 0.0365] | NOISE |
| h2 | -0.0150 | [-0.0438, 0.0116] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.081 / h2 0.406) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6862 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.5814 | 0.9717 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.3105 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5979 | 0.8185 | 0.4469 | 0.2766 |
| expected if the two signs were independent | 0.6459 | 0.6927 | 0.5079 | 0.2484 |

- P(up) unanimity (all three horizons call the same side): 0.3381

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0261 vs 0.0025 (-0.0286): does not beat, noise (boot z -1.05) | 0.0151 vs 0.0205 (-0.0054): does not beat, noise (boot z -0.19) | -0.0091 vs -0.0138 (+0.0047): beats, noise (boot z +0.15) |
| logreg_lags | direction/auc | 0.4884 vs 0.5320 (-0.0436): does not beat, noise (boot z -1.86) | 0.5186 vs 0.5251 (-0.0065): does not beat, noise (boot z -0.51) | 0.4886 vs 0.5095 (-0.0208): does not beat, noise (boot z -0.76) |
| logreg_lags | direction/brier | 0.2568 vs 0.2536 (-0.0032): does not beat, noise (DM z -1.82) | 0.2583 vs 0.2575 (-0.0009): does not beat, noise (DM z -0.61) | 0.2545 vs 0.2702 (+0.0157): beats (DM z +2.01) |
| logreg_lags | direction/ece_pos | 0.0705 vs 0.0663 (-0.0042): does not beat, noise (boot z -0.36) | 0.0758 vs 0.0812 (+0.0054): beats, noise (boot z +0.97) | 0.0474 vs 0.1230 (+0.0756): beats (boot z +4.62) |
| logreg_lags | direction/acc | 0.4804 vs 0.4836 (-0.0032): does not beat, noise (DM z -0.27) | 0.4962 vs 0.4911 (+0.0051): beats, noise (DM z +0.36) | 0.4946 vs 0.4756 (+0.0190): beats, noise (DM z +0.61) |
| logreg_lags | direction/bal_acc | 0.4883 vs 0.5004 (-0.0120): does not beat, noise (boot z -1.42) | 0.5061 vs 0.5055 (+0.0006): beats, noise (boot z +0.06) | 0.4954 vs 0.4971 (-0.0017): does not beat, noise (boot z -0.14) |
| class_prior | direction/mcc | -0.0261 vs 0.0000 (-0.0261): does not beat, noise (boot z -1.49) | 0.0151 vs 0.0000 (+0.0151): beats, noise (boot z +0.73) | -0.0091 vs 0.0000 (-0.0091): does not beat, noise (boot z -0.41) |
| class_prior | direction/auc | 0.4884 vs 0.5000 (-0.0116): does not beat, noise (boot z -1.12) | 0.5186 vs 0.5000 (+0.0186): beats, noise (boot z +1.35) | 0.4886 vs 0.5000 (-0.0114): does not beat, noise (boot z -0.77) |
| class_prior | direction/brier | 0.2568 vs 0.2532 (-0.0036): does not beat, significantly worse (DM z -2.90) | 0.2583 vs 0.2533 (-0.0050): does not beat, significantly worse (DM z -2.09) | 0.2545 vs 0.2573 (+0.0028): beats, noise (DM z +0.63) |
| class_prior | direction/ece_pos | 0.0705 vs 0.0593 (-0.0112): does not beat, noise (boot z -0.96) | 0.0758 vs 0.0603 (-0.0154): does not beat, significantly worse (boot z -2.54) | 0.0474 vs 0.0886 (+0.0413): beats (boot z +2.47) |
| class_prior | direction/acc | 0.4804 vs 0.4824 (-0.0020): does not beat, noise (DM z -0.17) | 0.4962 vs 0.4829 (+0.0133): beats, noise (DM z +0.92) | 0.4946 vs 0.4763 (+0.0183): beats, noise (DM z +0.56) |
| class_prior | direction/bal_acc | 0.4883 vs 0.5000 (-0.0117): does not beat, noise (boot z -1.49) | 0.5061 vs 0.5000 (+0.0061): beats, noise (boot z +0.73) | 0.4954 vs 0.5000 (-0.0046): does not beat, noise (boot z -0.41) |
| zero_delta | delta/rmse | 400.31 vs 400.31 (+0.00, +0.00%): does not beat | 564.58 vs 563.88 (-0.70, -0.12%): does not beat, noise (DM z -1.25) | 812.48 vs 807.63 (-4.86, -0.60%): does not beat, noise (DM z -1.32) |
| zero_delta | delta/mae | 271.46 vs 271.46 (+0.00, +0.00%): does not beat | 378.37 vs 377.88 (-0.49, -0.13%): does not beat, noise (DM z -1.19) | 555.14 vs 550.98 (-4.16, -0.76%): does not beat, noise (DM z -1.48) |
| mean_delta | delta/rmse | 400.31 vs 405.60 (+5.29, +1.31%): beats (DM z +2.85) | 564.58 vs 578.56 (+13.98, +2.42%): beats (DM z +2.93) | 812.48 vs 847.03 (+34.55, +4.08%): beats (DM z +3.00) |
| mean_delta | delta/mae | 271.46 vs 277.50 (+6.04, +2.18%): beats (DM z +3.94) | 378.37 vs 393.77 (+15.40, +3.91%): beats (DM z +3.92) | 555.14 vs 595.01 (+39.87, +6.70%): beats (DM z +3.93) |
| const_var | variance/crps | 202.98 vs 206.55 (+3.57, +1.73%): beats (DM z +3.50) | 292.77 vs 294.69 (+1.92, +0.65%): beats, noise (DM z +0.66) | 420.12 vs 444.83 (+24.70, +5.55%): beats (DM z +3.79) |
| const_var | variance/nll | 7.3837 vs 7.4448 (+0.0612): beats (DM z +3.43) | 7.7613 vs 7.8022 (+0.0409): beats, noise (DM z +1.11) | 8.1576 vs 8.2104 (+0.0528): beats (DM z +2.34) |
| const_var | variance/pit_ks | 0.0651 vs 0.1064 (+0.0413): beats (boot z +4.87) | 0.1100 vs 0.1404 (+0.0304): beats (boot z +4.03) | 0.0780 vs 0.1810 (+0.1030): beats (boot z +22.10) |
| const_var | variance/corr_var_err2_spearman | 0.2087 vs 0.0000 (+0.2087): beats (boot z +9.96) | 0.1147 vs 0.0000 (+0.1147): beats (boot z +4.67) | 0.1270 vs 0.0000 (+0.1270): beats (boot z +5.15) |

## Backtest (costs included)

- n_trades: 1105
- total_return: -0.9405
- sharpe_net: -105.8716
- sharpe_gross: 3.9728
- sortino: -122.5768
- max_drawdown: 0.9406
- hit_rate: 0.0679
- hit_rate_gross: 0.5195
- profit_factor: 0.0342
- avg_hold_bars: 12.6995
- exposure: 0.3248
- turnover: 748.9367
- fees_paid: 7489.4218
- traded_notional: 7489421.8315
- breakeven_cost_bps: 0.8833
- gross_edge_per_trade_bps: 0.5106
- costs_paid: 9736.2484
- gross_pnl: 330.7733
- net_pnl: -9405.4751

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.206, long_above 0.5738, short_below 0.4799, median 0.5285. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -94.05% | -105.87 | +94.06% | 1105 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -95.02% .. -93.38%) | -94.23% | -122.38 | | |

The random null enters at the strategy's rate (0.0379 per flat bar), holds 13 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 64% of its seeds on net return, 100% on net Sharpe and 96% on gross return.
