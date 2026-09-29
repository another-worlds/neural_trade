# Evaluation report - dev split - run `20260929T175243Z-80fd54c-4610a498-db5__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5549 | 0.8251 | 0.8564 |
| accuracy | 0.4827 | 0.4898 | 0.4754 |
| balanced accuracy | 0.4846 | 0.5009 | 0.4922 |
| precision (up) | 0.4685 | 0.4835 | 0.4718 |
| recall / sensitivity (up) | 0.5390 | 0.8260 | 0.8483 |
| specificity (down) | 0.4302 | 0.1758 | 0.1362 |
| F1 (up) | 0.5013 | 0.6099 | 0.6063 |
| MCC | -0.0310 | 0.0024 | -0.0221 |
| AUC | 0.4820 | 0.5179 | 0.4954 |
| Brier | 0.2553 | 0.2574 | 0.2584 |
| ECE (positive class) | 0.0582 | 0.0720 | 0.0843 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 9436 / 10703 / 8081 / 8072 | 15081 / 16111 / 3437 / 3177 | 15937 / 17842 / 2814 / 2851 |
| Gaussian readout: calls up | 0.8878 | 0.9304 | 0.8115 |
| Gaussian readout: MCC | -0.0163 | -0.0062 | 0.0146 |
| Gaussian readout: AUC | 0.4932 | 0.5201 | 0.5139 |
| Gaussian readout: Brier | 0.2507 | 0.2504 | 0.2517 |
| Gaussian readout: ECE | 0.0313 | 0.0280 | 0.0465 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.91 | 564.79 | 811.42 |
| RMSE ($), raw heads | 420.93 | 574.58 | 827.20 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.99 | 378.43 | 553.94 |
| MAE ($), raw heads | 289.10 | 386.97 | 569.64 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0030 | -0.0032 | -0.0094 |
| skill vs zero, raw heads | -0.1057 | -0.0383 | -0.0491 |
| EV, served | -0.0009 | -0.0005 | -0.0022 |
| EV, raw heads | -0.0435 | -0.0089 | -0.0179 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0088 | -0.0087 | 0.0028 |
| corr, Spearman, raw heads | -0.0161 | 0.0098 | 0.0060 |
| mean predicted ($), served | 10.35 | 14.53 | 38.25 |
| mean predicted ($), raw heads | 89.47 | 77.09 | 106.08 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.8813 | 0.9243 | 0.8096 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1156 | 0.1884 | 0.3606 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 203.02 | 288.64 | 419.09 |
| CRPSS vs constant variance | 0.0171 | 0.0205 | 0.0579 |
| NLL | 7.3868 | 7.7508 | 8.1794 |
| PIT KS | 0.0733 | 0.0949 | 0.0695 |
| var / err^2 Spearman | 0.1997 | 0.1167 | 0.1193 |
| coverage of the 90% interval | 0.9011 | 0.9013 | 0.8638 |
| width of the 90% interval ($) | 1208.53 | 1754.38 | 2404.92 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0078 | [-0.0318, 0.0163] | NOISE |
| h1 | 0.0149 | [-0.0180, 0.0473] | NOISE |
| h2 | -0.0006 | [-0.0311, 0.0312] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.116 / h1 0.188 / h2 0.361) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4536 | 0.7871 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.7770 | 0.9036 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.2819 | 0.6996 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5219 | 0.8298 | 0.7202 | 0.3450 |
| expected if the two signs were independent | 0.5485 | 0.7710 | 0.7204 | 0.3249 |

- P(up) unanimity (all three horizons call the same side): 0.4047

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0310 vs 0.0025 (-0.0335): does not beat, noise (boot z -1.13) | 0.0024 vs 0.0205 (-0.0181): does not beat, noise (boot z -0.72) | -0.0221 vs -0.0138 (-0.0083): does not beat, noise (boot z -0.30) |
| logreg_lags | direction/auc | 0.4820 vs 0.5320 (-0.0500): does not beat, significantly worse (boot z -2.03) | 0.5179 vs 0.5251 (-0.0071): does not beat, noise (boot z -0.67) | 0.4954 vs 0.5095 (-0.0141): does not beat, noise (boot z -0.62) |
| logreg_lags | direction/brier | 0.2553 vs 0.2536 (-0.0017): does not beat, noise (DM z -0.72) | 0.2574 vs 0.2575 (+0.0000): beats, noise (DM z +0.03) | 0.2584 vs 0.2702 (+0.0118): beats (DM z +2.30) |
| logreg_lags | direction/ece_pos | 0.0582 vs 0.0663 (+0.0081): beats, noise (boot z +0.50) | 0.0720 vs 0.0812 (+0.0092): beats, noise (boot z +1.78) | 0.0843 vs 0.1230 (+0.0387): beats (boot z +4.87) |
| logreg_lags | direction/acc | 0.4827 vs 0.4836 (-0.0010): does not beat, noise (DM z -0.06) | 0.4898 vs 0.4911 (-0.0013): does not beat, noise (DM z -0.11) | 0.4754 vs 0.4756 (-0.0002): does not beat, noise (DM z -0.02) |
| logreg_lags | direction/bal_acc | 0.4846 vs 0.5004 (-0.0158): does not beat, noise (boot z -1.56) | 0.5009 vs 0.5055 (-0.0046): does not beat, noise (boot z -0.55) | 0.4922 vs 0.4971 (-0.0049): does not beat, noise (boot z -0.65) |
| class_prior | direction/mcc | -0.0310 vs 0.0000 (-0.0310): does not beat, noise (boot z -1.69) | 0.0024 vs 0.0000 (+0.0024): beats, noise (boot z +0.12) | -0.0221 vs 0.0000 (-0.0221): does not beat, noise (boot z -1.28) |
| class_prior | direction/auc | 0.4820 vs 0.5000 (-0.0180): does not beat, noise (boot z -1.60) | 0.5179 vs 0.5000 (+0.0179): beats, noise (boot z +1.36) | 0.4954 vs 0.5000 (-0.0046): does not beat, noise (boot z -0.39) |
| class_prior | direction/brier | 0.2553 vs 0.2532 (-0.0020): does not beat, noise (DM z -1.15) | 0.2574 vs 0.2533 (-0.0041): does not beat, significantly worse (DM z -1.97) | 0.2584 vs 0.2573 (-0.0011): does not beat, noise (DM z -0.72) |
| class_prior | direction/ece_pos | 0.0582 vs 0.0593 (+0.0011): beats, noise (boot z +0.07) | 0.0720 vs 0.0603 (-0.0117): does not beat, significantly worse (boot z -2.04) | 0.0843 vs 0.0886 (+0.0043): beats, noise (boot z +0.58) |
| class_prior | direction/acc | 0.4827 vs 0.4824 (+0.0002): beats, noise (DM z +0.01) | 0.4898 vs 0.4829 (+0.0069): beats, noise (DM z +0.57) | 0.4754 vs 0.4763 (-0.0009): does not beat, noise (DM z -0.08) |
| class_prior | direction/bal_acc | 0.4846 vs 0.5000 (-0.0154): does not beat, noise (boot z -1.69) | 0.5009 vs 0.5000 (+0.0009): beats, noise (boot z +0.12) | 0.4922 vs 0.5000 (-0.0078): does not beat, noise (boot z -1.27) |
| zero_delta | delta/rmse | 400.91 vs 400.31 (-0.61, -0.15%): does not beat, noise (DM z -1.20) | 564.79 vs 563.88 (-0.91, -0.16%): does not beat, noise (DM z -1.05) | 811.42 vs 807.63 (-3.79, -0.47%): does not beat, noise (DM z -1.12) |
| zero_delta | delta/mae | 271.99 vs 271.46 (-0.53, -0.20%): does not beat, noise (DM z -1.53) | 378.43 vs 377.88 (-0.55, -0.15%): does not beat, noise (DM z -0.88) | 553.94 vs 550.98 (-2.96, -0.54%): does not beat, noise (DM z -1.18) |
| mean_delta | delta/rmse | 400.91 vs 405.60 (+4.69, +1.16%): beats (DM z +3.26) | 564.79 vs 578.56 (+13.77, +2.38%): beats (DM z +3.09) | 811.42 vs 847.03 (+35.61, +4.20%): beats (DM z +3.06) |
| mean_delta | delta/mae | 271.99 vs 277.50 (+5.51, +1.99%): beats (DM z +4.31) | 378.43 vs 393.77 (+15.34, +3.90%): beats (DM z +4.12) | 553.94 vs 595.01 (+41.08, +6.90%): beats (DM z +4.04) |
| const_var | variance/crps | 203.02 vs 206.55 (+3.53, +1.71%): beats (DM z +4.20) | 288.64 vs 294.69 (+6.05, +2.05%): beats (DM z +2.40) | 419.09 vs 444.83 (+25.74, +5.79%): beats (DM z +3.94) |
| const_var | variance/nll | 7.3868 vs 7.4448 (+0.0580): beats (DM z +3.57) | 7.7508 vs 7.8022 (+0.0514): beats (DM z +2.10) | 8.1794 vs 8.2104 (+0.0310): beats, noise (DM z +1.06) |
| const_var | variance/pit_ks | 0.0733 vs 0.1064 (+0.0331): beats (boot z +7.73) | 0.0949 vs 0.1404 (+0.0455): beats (boot z +7.56) | 0.0695 vs 0.1810 (+0.1116): beats (boot z +27.43) |
| const_var | variance/corr_var_err2_spearman | 0.1997 vs 0.0000 (+0.1997): beats (boot z +10.73) | 0.1167 vs 0.0000 (+0.1167): beats (boot z +4.79) | 0.1193 vs 0.0000 (+0.1193): beats (boot z +4.72) |

## Backtest (costs included)

- n_trades: 1196
- total_return: -0.9552
- sharpe_net: -114.8058
- sharpe_gross: 1.6001
- sortino: -131.6433
- max_drawdown: 0.9552
- hit_rate: 0.0602
- hit_rate_gross: 0.5050
- profit_factor: 0.0282
- avg_hold_bars: 11.9490
- exposure: 0.3308
- turnover: 744.4680
- fees_paid: 7444.8074
- traded_notional: 7444807.4091
- breakeven_cost_bps: 0.3394
- gross_edge_per_trade_bps: 0.0882
- costs_paid: 9678.2496
- gross_pnl: 126.3360
- net_pnl: -9551.9136

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.074, long_above 0.5638, short_below 0.4863, median 0.5245. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -95.52% | -114.81 | +95.52% | 1196 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.13% .. -94.91%) | -95.51% | -130.11 | | |

The random null enters at the strategy's rate (0.0414 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 53% of its seeds on net return, 100% on net Sharpe and 77% on gross return.
