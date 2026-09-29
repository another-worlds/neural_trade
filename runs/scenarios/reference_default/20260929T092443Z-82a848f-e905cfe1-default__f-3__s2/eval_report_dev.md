# Evaluation report - dev split - run `20260929T092443Z-82a848f-e905cfe1-default__f-3__s2`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.2977 | 0.3773 | 0.6853 |
| accuracy | 0.5127 | 0.5237 | 0.4917 |
| balanced accuracy | 0.5147 | 0.5246 | 0.4938 |
| precision (up) | 0.5296 | 0.5365 | 0.4898 |
| recall / sensitivity (up) | 0.3122 | 0.4017 | 0.6790 |
| specificity (down) | 0.7172 | 0.6476 | 0.3085 |
| F1 (up) | 0.3928 | 0.4594 | 0.5691 |
| MCC | 0.0322 | 0.0508 | -0.0134 |
| AUC | 0.5283 | 0.5368 | 0.5073 |
| Brier | 0.2493 | 0.2488 | 0.2524 |
| ECE (positive class) | 0.0199 | 0.0171 | 0.0480 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 769 / 683 / 1732 / 1694 | 1067 / 922 / 1694 / 1589 | 1872 / 1950 / 870 / 885 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.2856 | 0.3302 | 0.5467 |
| Gaussian readout of the raw heads: MCC | -0.0167 | -0.0041 | 0.0380 |
| Gaussian readout of the raw heads: AUC | 0.5081 | 0.5135 | 0.5274 |
| Gaussian readout of the raw heads: Brier | 0.2527 | 0.2532 | 0.2519 |
| Gaussian readout of the raw heads: ECE | 0.0594 | 0.0640 | 0.0420 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.39 | 252.65 | 288.63 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 139.36 | 171.36 | 199.16 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0094 | -0.0197 | -0.0192 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0095 | -0.0199 | -0.0157 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0606 | 0.0563 | 0.0677 |
| corr, Spearman, raw heads | 0.0191 | 0.0285 | 0.0488 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -2.60 | -3.98 | 12.86 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.2616 | 0.3143 | 0.5448 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 103.59 | 126.34 | 147.02 |
| CRPSS vs constant variance | 0.0383 | 0.0328 | 0.0282 |
| NLL | 6.6948 | 6.8773 | 7.0310 |
| PIT KS | 0.0681 | 0.0713 | 0.0715 |
| var / err^2 Spearman | 0.2687 | 0.2585 | 0.2435 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0222 | [-0.0245, 0.0690] | NOISE |
| h1 | 0.0236 | [-0.0161, 0.0702] | NOISE |
| h2 | 0.0293 | [-0.0158, 0.0694] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7763 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.4686 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3172 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6790 | 0.7044 | 0.5571 | 0.2454 |
| expected if the two signs were independent | 0.6045 | 0.5522 | 0.5178 | 0.1630 |

- P(up) unanimity (all three horizons call the same side): 0.2005

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0322 vs 0.0474 (-0.0152): does not beat, noise (boot z -0.51) | 0.0508 vs 0.0487 (+0.0021): beats, noise (boot z +0.06) | -0.0134 vs 0.0397 (-0.0531): does not beat, noise (boot z -0.97) |
| logreg_lags | direction/auc | 0.5283 vs 0.5348 (-0.0065): does not beat, noise (boot z -0.31) | 0.5368 vs 0.5440 (-0.0072): does not beat, noise (boot z -0.33) | 0.5073 vs 0.5205 (-0.0133): does not beat, noise (boot z -0.37) |
| logreg_lags | direction/brier | 0.2493 vs 0.2494 (+0.0001): beats, noise (DM z +0.05) | 0.2488 vs 0.2491 (+0.0003): beats, noise (DM z +0.18) | 0.2524 vs 0.2494 (-0.0030): does not beat, noise (DM z -1.02) |
| logreg_lags | direction/ece_pos | 0.0199 vs 0.0274 (+0.0075): beats, noise (boot z +0.78) | 0.0171 vs 0.0284 (+0.0113): beats, noise (boot z +0.82) | 0.0480 vs 0.0161 (-0.0319): does not beat, noise (boot z -1.15) |
| logreg_lags | direction/acc | 0.5127 vs 0.5150 (-0.0023): does not beat, noise (DM z -0.16) | 0.5237 vs 0.5152 (+0.0085): beats, noise (DM z +0.48) | 0.4917 vs 0.5184 (-0.0267): does not beat, noise (DM z -0.77) |
| logreg_lags | direction/bal_acc | 0.5147 vs 0.5181 (-0.0034): does not beat, noise (boot z -0.28) | 0.5246 vs 0.5178 (+0.0069): beats, noise (boot z +0.48) | 0.4938 vs 0.5145 (-0.0208): does not beat, noise (boot z -0.94) |
| class_prior | direction/mcc | 0.0322 vs 0.0000 (+0.0322): beats, noise (boot z +1.20) | 0.0508 vs 0.0000 (+0.0508): beats, noise (boot z +1.84) | -0.0134 vs 0.0000 (-0.0134): does not beat, noise (boot z -0.45) |
| class_prior | direction/auc | 0.5283 vs 0.5000 (+0.0283): beats, noise (boot z +1.56) | 0.5368 vs 0.5000 (+0.0368): beats (boot z +2.17) | 0.5073 vs 0.5000 (+0.0073): beats, noise (boot z +0.39) |
| class_prior | direction/brier | 0.2493 vs 0.2504 (+0.0011): beats, noise (DM z +0.91) | 0.2488 vs 0.2504 (+0.0016): beats, noise (DM z +0.93) | 0.2524 vs 0.2500 (-0.0024): does not beat, noise (DM z -1.10) |
| class_prior | direction/ece_pos | 0.0199 vs 0.0217 (+0.0018): beats, noise (boot z +0.19) | 0.0171 vs 0.0193 (+0.0022): beats, noise (boot z +0.19) | 0.0480 vs 0.0080 (-0.0400): does not beat, noise (boot z -1.65) |
| class_prior | direction/acc | 0.5127 vs 0.4951 (+0.0176): beats, noise (DM z +1.06) | 0.5237 vs 0.4962 (+0.0275): beats, noise (DM z +1.28) | 0.4917 vs 0.5056 (-0.0140): does not beat, noise (DM z -0.38) |
| class_prior | direction/bal_acc | 0.5147 vs 0.5000 (+0.0147): beats, noise (boot z +1.20) | 0.5246 vs 0.5000 (+0.0246): beats, noise (boot z +1.84) | 0.4938 vs 0.5000 (-0.0062): does not beat, noise (boot z -0.45) |
| zero_delta | delta/rmse | 211.40 vs 211.40 (+0.00, +0.00%): does not beat | 250.20 vs 250.20 (+0.00, +0.00%): does not beat | 285.89 vs 285.89 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 137.51 vs 137.51 (+0.00, +0.00%): does not beat | 168.19 vs 168.19 (+0.00, +0.00%): does not beat | 195.53 vs 195.53 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 211.40 vs 211.39 (-0.01, -0.01%): does not beat, noise (DM z -0.16) | 250.20 vs 250.18 (-0.03, -0.01%): does not beat, noise (DM z -0.19) | 285.89 vs 285.84 (-0.04, -0.01%): does not beat, noise (DM z -0.19) |
| mean_delta | delta/mae | 137.51 vs 137.52 (+0.01, +0.01%): beats, noise (DM z +0.15) | 168.19 vs 168.27 (+0.07, +0.04%): beats, noise (DM z +0.60) | 195.53 vs 195.56 (+0.03, +0.02%): beats, noise (DM z +0.17) |
| const_var | variance/crps | 103.59 vs 107.72 (+4.13, +3.83%): beats (DM z +9.20) | 126.34 vs 130.63 (+4.29, +3.28%): beats (DM z +7.32) | 147.02 vs 151.29 (+4.27, +2.82%): beats (DM z +5.79) |
| const_var | variance/nll | 6.6948 vs 6.7814 (+0.0866): beats (DM z +3.54) | 6.8773 vs 6.9539 (+0.0766): beats (DM z +3.62) | 7.0310 vs 7.0883 (+0.0573): beats (DM z +2.55) |
| const_var | variance/pit_ks | 0.0681 vs 0.1058 (+0.0377): beats (boot z +7.40) | 0.0713 vs 0.1039 (+0.0326): beats (boot z +4.76) | 0.0715 vs 0.1039 (+0.0324): beats (boot z +4.75) |
| const_var | variance/corr_var_err2_spearman | 0.2687 vs 0.0000 (+0.2687): beats (boot z +8.02) | 0.2585 vs 0.0000 (+0.2585): beats (boot z +6.71) | 0.2435 vs 0.0000 (+0.2435): beats (boot z +6.02) |

## Backtest (costs included)

- n_trades: 541
- total_return: -0.7405
- sharpe_net: -202.9703
- sharpe_gross: 12.9146
- sortino: -219.2251
- max_drawdown: 0.7405
- hit_rate: 0.0573
- hit_rate_gross: 0.4880
- profit_factor: 0.0406
- avg_hold_bars: 7.0795
- exposure: 0.5293
- turnover: 593.0787
- fees_paid: 5930.6450
- costs_paid: 7709.8385
- gross_pnl: 305.0503
- net_pnl: -7404.7883

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -3 (TimeSeriesSplit fold 3, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-10-26T05:22:00+00:00 .. 2025-10-31T05:57:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.5427, long_above 0.5117, short_below 0.4732, median 0.4915. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -74.05% | -202.97 | +74.05% | 541 |
| buy and hold | -1.69% | -2.62 | +8.46% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -77.25% .. -74.23%) | -75.76% | -222.92 | | |

The random null enters at the strategy's rate (0.1588 per flat bar), holds 7 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 95% of its seeds on net return, 100% on net Sharpe and 94% on gross return.
