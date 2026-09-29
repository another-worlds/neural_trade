# Evaluation report - dev split - run `20260929T092132Z-82a848f-6ce8c78e-default__f-3__s1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.5191 | 0.4490 | 0.5202 |
| accuracy | 0.5168 | 0.5085 | 0.5209 |
| balanced accuracy | 0.5166 | 0.5089 | 0.5211 |
| precision (up) | 0.5209 | 0.5137 | 0.5147 |
| recall / sensitivity (up) | 0.5355 | 0.4578 | 0.5415 |
| specificity (down) | 0.4977 | 0.5600 | 0.5007 |
| F1 (up) | 0.5281 | 0.4842 | 0.5277 |
| MCC | 0.0333 | 0.0179 | 0.0423 |
| AUC | 0.5272 | 0.5169 | 0.5305 |
| Brier | 0.2504 | 0.2514 | 0.2498 |
| ECE (positive class) | 0.0289 | 0.0334 | 0.0140 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1319 / 1213 / 1202 / 1144 | 1216 / 1151 / 1465 / 1440 | 1493 / 1408 / 1412 / 1264 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.3602 | 0.5907 | 0.6394 |
| Gaussian readout of the raw heads: MCC | 0.0537 | 0.0904 | 0.1114 |
| Gaussian readout of the raw heads: AUC | 0.5495 | 0.5723 | 0.5601 |
| Gaussian readout of the raw heads: Brier | 0.2472 | 0.2448 | 0.2485 |
| Gaussian readout of the raw heads: ECE | 0.0155 | 0.0089 | 0.0420 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 210.22 | 248.15 | 283.67 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 136.80 | 166.92 | 195.91 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | 0.0111 | 0.0164 | 0.0155 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | 0.0110 | 0.0164 | 0.0197 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.1079 | 0.1501 | 0.1601 |
| corr, Spearman, raw heads | 0.0775 | 0.1017 | 0.1051 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -2.47 | 0.53 | 14.36 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.3509 | 0.6020 | 0.6566 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 102.91 | 125.53 | 145.71 |
| CRPSS vs constant variance | 0.0446 | 0.0390 | 0.0369 |
| NLL | 6.7031 | 6.8833 | 7.0331 |
| PIT KS | 0.0588 | 0.0576 | 0.0518 |
| var / err^2 Spearman | 0.2644 | 0.2387 | 0.2336 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0213 | [-0.0238, 0.0556] | NOISE |
| h1 | 0.0247 | [-0.0023, 0.0610] | NOISE |
| h2 | 0.0409 | [-0.0009, 0.0786] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6996 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.8747 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.6108 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6121 | 0.5181 | 0.5913 | 0.2247 |
| expected if the two signs were independent | 0.4900 | 0.4855 | 0.5116 | 0.1506 |

- P(up) unanimity (all three horizons call the same side): 0.3488

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0333 vs 0.0474 (-0.0141): does not beat, noise (boot z -0.33) | 0.0179 vs 0.0487 (-0.0308): does not beat, noise (boot z -0.92) | 0.0423 vs 0.0397 (+0.0026): beats, noise (boot z +0.05) |
| logreg_lags | direction/auc | 0.5272 vs 0.5348 (-0.0076): does not beat, noise (boot z -0.26) | 0.5169 vs 0.5440 (-0.0271): does not beat, noise (boot z -1.29) | 0.5305 vs 0.5205 (+0.0099): beats, noise (boot z +0.28) |
| logreg_lags | direction/brier | 0.2504 vs 0.2494 (-0.0010): does not beat, noise (DM z -0.41) | 0.2514 vs 0.2491 (-0.0023): does not beat, noise (DM z -1.17) | 0.2498 vs 0.2494 (-0.0003): does not beat, noise (DM z -0.14) |
| logreg_lags | direction/ece_pos | 0.0289 vs 0.0274 (-0.0015): does not beat, noise (boot z -0.08) | 0.0334 vs 0.0284 (-0.0050): does not beat, noise (boot z -0.29) | 0.0140 vs 0.0161 (+0.0021): beats, noise (boot z +0.15) |
| logreg_lags | direction/acc | 0.5168 vs 0.5150 (+0.0018): beats, noise (DM z +0.08) | 0.5085 vs 0.5152 (-0.0066): does not beat, noise (DM z -0.35) | 0.5209 vs 0.5184 (+0.0025): beats, noise (DM z +0.08) |
| logreg_lags | direction/bal_acc | 0.5166 vs 0.5181 (-0.0015): does not beat, noise (boot z -0.08) | 0.5089 vs 0.5178 (-0.0088): does not beat, noise (boot z -0.65) | 0.5211 vs 0.5145 (+0.0066): beats, noise (boot z +0.29) |
| class_prior | direction/mcc | 0.0333 vs 0.0000 (+0.0333): beats, noise (boot z +1.10) | 0.0179 vs 0.0000 (+0.0179): beats, noise (boot z +0.73) | 0.0423 vs 0.0000 (+0.0423): beats, noise (boot z +1.57) |
| class_prior | direction/auc | 0.5272 vs 0.5000 (+0.0272): beats, noise (boot z +1.41) | 0.5169 vs 0.5000 (+0.0169): beats, noise (boot z +1.14) | 0.5305 vs 0.5000 (+0.0305): beats, noise (boot z +1.87) |
| class_prior | direction/brier | 0.2504 vs 0.2504 (+0.0001): beats, noise (DM z +0.04) | 0.2514 vs 0.2504 (-0.0010): does not beat, noise (DM z -0.56) | 0.2498 vs 0.2500 (+0.0003): beats, noise (DM z +0.17) |
| class_prior | direction/ece_pos | 0.0289 vs 0.0217 (-0.0072): does not beat, noise (boot z -0.41) | 0.0334 vs 0.0193 (-0.0141): does not beat, noise (boot z -0.90) | 0.0140 vs 0.0080 (-0.0060): does not beat, noise (boot z -0.63) |
| class_prior | direction/acc | 0.5168 vs 0.4951 (+0.0217): beats, noise (DM z +0.94) | 0.5085 vs 0.4962 (+0.0123): beats, noise (DM z +0.53) | 0.5209 vs 0.5056 (+0.0152): beats, noise (DM z +0.49) |
| class_prior | direction/bal_acc | 0.5166 vs 0.5000 (+0.0166): beats, noise (boot z +1.10) | 0.5089 vs 0.5000 (+0.0089): beats, noise (boot z +0.73) | 0.5211 vs 0.5000 (+0.0211): beats, noise (boot z +1.57) |
| zero_delta | delta/rmse | 211.40 vs 211.40 (+0.00, +0.00%): does not beat | 250.20 vs 250.20 (+0.00, +0.00%): does not beat | 285.89 vs 285.89 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 137.51 vs 137.51 (+0.00, +0.00%): does not beat | 168.19 vs 168.19 (+0.00, +0.00%): does not beat | 195.53 vs 195.53 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 211.40 vs 211.39 (-0.01, -0.01%): does not beat, noise (DM z -0.16) | 250.20 vs 250.18 (-0.03, -0.01%): does not beat, noise (DM z -0.19) | 285.89 vs 285.84 (-0.04, -0.01%): does not beat, noise (DM z -0.19) |
| mean_delta | delta/mae | 137.51 vs 137.52 (+0.01, +0.01%): beats, noise (DM z +0.15) | 168.19 vs 168.27 (+0.07, +0.04%): beats, noise (DM z +0.60) | 195.53 vs 195.56 (+0.03, +0.02%): beats, noise (DM z +0.17) |
| const_var | variance/crps | 102.91 vs 107.72 (+4.80, +4.46%): beats (DM z +10.33) | 125.53 vs 130.63 (+5.10, +3.90%): beats (DM z +8.69) | 145.71 vs 151.29 (+5.58, +3.69%): beats (DM z +6.86) |
| const_var | variance/nll | 6.7031 vs 6.7814 (+0.0783): beats (DM z +3.12) | 6.8833 vs 6.9539 (+0.0706): beats (DM z +3.21) | 7.0331 vs 7.0883 (+0.0552): beats, noise (DM z +1.94) |
| const_var | variance/pit_ks | 0.0588 vs 0.1058 (+0.0470): beats (boot z +9.06) | 0.0576 vs 0.1039 (+0.0463): beats (boot z +6.16) | 0.0518 vs 0.1039 (+0.0521): beats (boot z +5.61) |
| const_var | variance/corr_var_err2_spearman | 0.2644 vs 0.0000 (+0.2644): beats (boot z +8.36) | 0.2387 vs 0.0000 (+0.2387): beats (boot z +6.24) | 0.2336 vs 0.0000 (+0.2336): beats (boot z +5.74) |

## Backtest (costs included)

- n_trades: 526
- total_return: -0.7359
- sharpe_net: -193.2386
- sharpe_gross: 4.3788
- sortino: -215.8902
- max_drawdown: 0.7359
- hit_rate: 0.0570
- hit_rate_gross: 0.5418
- profit_factor: 0.0274
- avg_hold_bars: 7.8365
- exposure: 0.5697
- turnover: 575.4496
- fees_paid: 5754.4530
- costs_paid: 7480.7888
- gross_pnl: 122.2305
- net_pnl: -7358.5584

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -3 (TimeSeriesSplit fold 3, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-10-26T05:22:00+00:00 .. 2025-10-31T05:57:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.5098, long_above 0.5349, short_below 0.4782, median 0.5076. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -73.59% | -193.24 | +73.59% | 526 |
| buy and hold | -1.69% | -2.62 | +8.46% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -75.84% .. -72.15%) | -74.06% | -210.02 | | |

The random null enters at the strategy's rate (0.1689 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 69% of its seeds on net return, 98% on net Sharpe and 72% on gross return.
