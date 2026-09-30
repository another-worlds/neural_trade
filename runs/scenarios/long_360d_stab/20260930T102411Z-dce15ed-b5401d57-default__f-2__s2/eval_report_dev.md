# Evaluation report - dev split - run `20260930T102411Z-dce15ed-b5401d57-default__f-2__s2`

n = 46544 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 28800 | 32081 | 33877 |
| n_eff of the scored moves (n scored // bars ahead) | 2880 | 2138 | 1693 |
| true up-rate | 0.4906 | 0.4906 | 0.4890 |
| calls up (predicted up-rate) | 0.3731 | 0.6378 | 0.3514 |
| accuracy | 0.5110 | 0.5106 | 0.5164 |
| balanced accuracy | 0.5086 | 0.5132 | 0.5132 |
| precision (up) | 0.5021 | 0.5009 | 0.5077 |
| recall / sensitivity (up) | 0.3819 | 0.6512 | 0.3649 |
| specificity (down) | 0.6354 | 0.3751 | 0.6614 |
| F1 (up) | 0.4338 | 0.5663 | 0.4246 |
| MCC | 0.0179 | 0.0274 | 0.0276 |
| AUC | 0.5143 | 0.5190 | 0.5165 |
| Brier | 0.2499 | 0.2501 | 0.2497 |
| ECE (positive class) | 0.0071 | 0.0182 | 0.0054 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0094 | 0.0094 | 0.0110 |
| TP / FP / TN / FN | 5395 / 5349 / 9323 / 8733 | 10250 / 10212 / 6130 / 5489 | 6044 / 5861 / 11451 / 10521 |
| Gaussian readout: calls up | 0.3770 | 0.4024 | 0.4094 |
| Gaussian readout: MCC | 0.0158 | 0.0140 | 0.0232 |
| Gaussian readout: AUC | 0.5077 | 0.5137 | 0.5123 |
| Gaussian readout: Brier | 0.2499 | 0.2499 | 0.2500 |
| Gaussian readout: ECE | 0.0083 | 0.0089 | 0.0124 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 164.15 | 198.31 | 228.32 |
| RMSE ($), raw heads | 164.63 | 200.69 | 230.41 |
| RMSE ($), zero prediction | 164.21 | 198.38 | 228.36 |
| MAE ($), served | 112.61 | 138.00 | 158.98 |
| MAE ($), raw heads | 112.81 | 138.87 | 159.95 |
| MAE ($), zero prediction | 112.64 | 138.04 | 158.99 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0008 | 0.0007 | 0.0004 |
| skill vs zero, raw heads | -0.0051 | -0.0234 | -0.0180 |
| EV, served | 0.0008 | 0.0007 | 0.0004 |
| EV, raw heads | -0.0052 | -0.0234 | -0.0181 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0336 | 0.0389 | 0.0425 |
| corr, Spearman, raw heads | 0.0141 | 0.0192 | 0.0168 |
| mean predicted ($), served | -0.15 | -0.05 | -0.05 |
| mean predicted ($), raw heads | -1.11 | -0.85 | -1.90 |
| mean realised ($) | -1.18 | -1.76 | -2.32 |
| share predicted up, raw heads | 0.3509 | 0.3975 | 0.4034 |
| share realised up | 0.4897 | 0.4876 | 0.4888 |
| shrink beta (served = beta x raw, fit on cal) | 0.1352 | 0.0563 | 0.0238 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 81.77 | 99.90 | 115.21 |
| CRPSS vs constant variance | 0.0493 | 0.0467 | 0.0461 |
| NLL | 6.3931 | 6.5920 | 6.7405 |
| PIT KS | 0.0273 | 0.0263 | 0.0287 |
| var / err^2 Spearman | 0.3426 | 0.3356 | 0.3380 |
| coverage of the 90% interval | 0.9070 | 0.9081 | 0.9078 |
| width of the 90% interval ($) | 518.33 | 634.69 | 728.94 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0172 | [-0.0010, 0.0350] | NOISE |
| h1 | 0.0056 | [-0.0132, 0.0250] | NOISE |
| h2 | 0.0147 | [-0.0041, 0.0340] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.135 / h1 0.056 / h2 0.024) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8267 | 0.3366 | 0.6180 |
| abs(d h1) <= abs(d h2) | 0.8257 | 0.1346 | 0.5923 |
| full chain h0 <= h1 <= h2 | 0.6896 | 0.0270 | 0.3360 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5008 | 0.5529 | 0.5621 | 0.1585 |
| expected if the two signs were independent | 0.5335 | 0.4759 | 0.5287 | 0.1232 |

- P(up) unanimity (all three horizons call the same side): 0.2420

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0179 vs 0.0344 (-0.0165): does not beat, noise (boot z -1.10) | 0.0274 vs 0.0378 (-0.0103): does not beat, noise (boot z -0.86) | 0.0276 vs 0.0411 (-0.0136): does not beat, noise (boot z -0.72) |
| logreg_lags | direction/auc | 0.5143 vs 0.5243 (-0.0100): does not beat, noise (boot z -1.09) | 0.5190 vs 0.5272 (-0.0083): does not beat, noise (boot z -1.15) | 0.5165 vs 0.5314 (-0.0150): does not beat, noise (boot z -1.23) |
| logreg_lags | direction/brier | 0.2499 vs 0.2497 (-0.0002): does not beat, noise (DM z -0.77) | 0.2501 vs 0.2497 (-0.0004): does not beat, noise (DM z -1.52) | 0.2497 vs 0.2497 (-0.0000): does not beat, noise (DM z -0.11) |
| logreg_lags | direction/ece_pos | 0.0071 vs 0.0112 (+0.0041): beats, noise (boot z +0.94) | 0.0182 vs 0.0116 (-0.0066): does not beat, noise (boot z -1.10) | 0.0054 vs 0.0137 (+0.0083): beats, noise (boot z +1.40) |
| logreg_lags | direction/acc | 0.5110 vs 0.5165 (-0.0055): does not beat, noise (DM z -0.72) | 0.5106 vs 0.5178 (-0.0072): does not beat, noise (DM z -1.14) | 0.5164 vs 0.5191 (-0.0027): does not beat, noise (DM z -0.26) |
| logreg_lags | direction/bal_acc | 0.5086 vs 0.5171 (-0.0085): does not beat, noise (boot z -1.16) | 0.5132 vs 0.5188 (-0.0056): does not beat, noise (boot z -0.95) | 0.5132 vs 0.5204 (-0.0073): does not beat, noise (boot z -0.79) |
| class_prior | direction/mcc | 0.0179 vs 0.0000 (+0.0179): beats, noise (boot z +1.45) | 0.0274 vs 0.0000 (+0.0274): beats (boot z +2.18) | 0.0276 vs 0.0000 (+0.0276): beats (boot z +2.18) |
| class_prior | direction/auc | 0.5143 vs 0.5000 (+0.0143): beats, noise (boot z +1.85) | 0.5190 vs 0.5000 (+0.0190): beats (boot z +2.32) | 0.5165 vs 0.5000 (+0.0165): beats, noise (boot z +1.96) |
| class_prior | direction/brier | 0.2499 vs 0.2501 (+0.0002): beats, noise (DM z +1.03) | 0.2501 vs 0.2501 (-0.0001): does not beat, noise (DM z -0.16) | 0.2497 vs 0.2501 (+0.0004): beats, noise (DM z +1.06) |
| class_prior | direction/ece_pos | 0.0071 vs 0.0123 (+0.0051): beats, noise (boot z +1.28) | 0.0182 vs 0.0134 (-0.0048): does not beat, significantly worse (boot z -2.41) | 0.0054 vs 0.0164 (+0.0109): beats, noise (boot z +1.48) |
| class_prior | direction/acc | 0.5110 vs 0.4906 (+0.0205): beats, noise (DM z +1.83) | 0.5106 vs 0.4906 (+0.0200): beats (DM z +2.35) | 0.5164 vs 0.4890 (+0.0275): beats, noise (DM z +1.90) |
| class_prior | direction/bal_acc | 0.5086 vs 0.5000 (+0.0086): beats, noise (boot z +1.45) | 0.5132 vs 0.5000 (+0.0132): beats (boot z +2.18) | 0.5132 vs 0.5000 (+0.0132): beats (boot z +2.18) |
| zero_delta | delta/rmse | 164.15 vs 164.21 (+0.07, +0.04%): beats, noise (DM z +1.24) | 198.31 vs 198.38 (+0.07, +0.04%): beats, noise (DM z +1.17) | 228.32 vs 228.36 (+0.04, +0.02%): beats, noise (DM z +1.47) |
| zero_delta | delta/mae | 112.61 vs 112.64 (+0.03, +0.02%): beats, noise (DM z +1.14) | 138.00 vs 138.04 (+0.04, +0.03%): beats, noise (DM z +1.40) | 158.98 vs 158.99 (+0.01, +0.01%): beats, noise (DM z +1.18) |
| mean_delta | delta/rmse | 164.15 vs 164.22 (+0.07, +0.04%): beats, noise (DM z +1.37) | 198.31 vs 198.40 (+0.09, +0.04%): beats, noise (DM z +1.34) | 228.32 vs 228.38 (+0.06, +0.03%): beats, noise (DM z +1.57) |
| mean_delta | delta/mae | 112.61 vs 112.65 (+0.04, +0.04%): beats, noise (DM z +1.68) | 138.00 vs 138.07 (+0.07, +0.05%): beats (DM z +2.15) | 158.98 vs 159.03 (+0.05, +0.03%): beats, noise (DM z +1.82) |
| const_var | variance/crps | 81.77 vs 86.01 (+4.24, +4.93%): beats (DM z +21.03) | 99.90 vs 104.80 (+4.90, +4.67%): beats (DM z +16.78) | 115.21 vs 120.77 (+5.56, +4.61%): beats (DM z +15.02) |
| const_var | variance/nll | 6.3931 vs 6.5314 (+0.1383): beats (DM z +8.27) | 6.5920 vs 6.7225 (+0.1305): beats (DM z +7.38) | 6.7405 vs 6.8636 (+0.1231): beats (DM z +6.92) |
| const_var | variance/pit_ks | 0.0273 vs 0.0915 (+0.0642): beats (boot z +16.00) | 0.0263 vs 0.0894 (+0.0631): beats (boot z +15.87) | 0.0287 vs 0.0910 (+0.0623): beats (boot z +15.24) |
| const_var | variance/corr_var_err2_spearman | 0.3426 vs 0.0000 (+0.3426): beats (boot z +25.58) | 0.3356 vs 0.0000 (+0.3356): beats (boot z +23.27) | 0.3380 vs 0.0000 (+0.3380): beats (boot z +22.17) |

## Backtest (costs included)

- n_trades: 2026
- total_return: -0.9940
- sharpe_net: -157.7873
- sharpe_gross: 6.3442
- sortino: -171.7912
- max_drawdown: 0.9940
- hit_rate: 0.0350
- hit_rate_gross: 0.5533
- profit_factor: 0.0184
- avg_hold_bars: 8.9176
- exposure: 0.3882
- turnover: 794.5392
- fees_paid: 7945.3685
- traded_notional: 7945368.5381
- breakeven_cost_bps: 0.9794
- gross_edge_per_trade_bps: 0.8019
- costs_paid: 10328.9791
- gross_pnl: 389.0713
- net_pnl: -9939.9078

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 13, 13 usable folds); this report scores the fold's out-of-sample block: 46544 sequences, 2025-07-27T08:10:00 .. 2025-08-28T15:53:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4103, long_above 0.5094, short_below 0.4892, median 0.4980. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.40% | -157.79 | +99.40% | 2026 |
| buy and hold | -5.10% | -1.62 | +12.57% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.54% .. -99.40%) | -99.47% | -176.12 | | |

The random null enters at the strategy's rate (0.0711 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 94% of its seeds on net return, 100% on net Sharpe and 98% on gross return.
