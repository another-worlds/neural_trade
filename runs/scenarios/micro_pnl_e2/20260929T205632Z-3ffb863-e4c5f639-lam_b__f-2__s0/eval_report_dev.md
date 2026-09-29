# Evaluation report - dev split - run `20260929T205632Z-3ffb863-e4c5f639-lam_b__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.6355 | 0.9117 | 0.8746 |
| accuracy | 0.4839 | 0.4791 | 0.4848 |
| balanced accuracy | 0.4887 | 0.4931 | 0.5025 |
| precision (up) | 0.4735 | 0.4792 | 0.4778 |
| recall / sensitivity (up) | 0.6237 | 0.9045 | 0.8773 |
| specificity (down) | 0.3536 | 0.0816 | 0.1278 |
| F1 (up) | 0.5383 | 0.6265 | 0.6186 |
| MCC | -0.0236 | -0.0243 | 0.0076 |
| AUC | 0.4890 | 0.4836 | 0.5041 |
| Brier | 0.2578 | 0.2537 | 0.2577 |
| ECE (positive class) | 0.0735 | 0.0575 | 0.0783 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 10920 / 12142 / 6642 / 6588 | 16515 / 17952 / 1596 / 1743 | 16482 / 18017 / 2639 / 2306 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.7661 | 0.7641 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0181 | -0.0013 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4819 | 0.4925 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2504 | 0.2525 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0223 | 0.0461 |
| Gaussian readout of the raw heads: calls up | 0.9857 | 0.7661 | 0.7641 |
| Gaussian readout of the raw heads: MCC | -0.0075 | -0.0181 | -0.0013 |
| Gaussian readout of the raw heads: AUC | 0.4655 | 0.4819 | 0.4925 |
| Gaussian readout of the raw heads: Brier | 0.2669 | 0.2584 | 0.2743 |
| Gaussian readout of the raw heads: ECE | 0.1087 | 0.0706 | 0.1197 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.31 | 564.50 | 811.93 |
| RMSE ($), raw heads | 412.50 | 576.14 | 849.29 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.46 | 378.37 | 554.91 |
| MAE ($), raw heads | 284.60 | 390.09 | 593.41 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0022 | -0.0107 |
| skill vs zero, raw heads | -0.0618 | -0.0439 | -0.1058 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0013 | -0.0042 |
| EV, raw heads | -0.0218 | -0.0260 | -0.0505 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0667 | -0.0442 | -0.0218 |
| corr, Spearman, raw heads | -0.0656 | -0.0398 | -0.0244 |
| mean predicted ($), served | 0.00 | 5.80 | 34.96 |
| mean predicted ($), raw heads | 69.94 | 56.72 | 152.17 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9860 | 0.7609 | 0.7636 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1023 | 0.2297 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.21 | 285.96 | 420.13 |
| CRPSS vs constant variance | 0.0210 | 0.0296 | 0.0555 |
| NLL | 7.4174 | 7.7782 | 8.1736 |
| PIT KS | 0.0455 | 0.0644 | 0.0701 |
| var / err^2 Spearman | 0.1430 | 0.0273 | 0.0914 |
| coverage of the 90% interval | 0.9028 | 0.9014 | 0.8622 |
| width of the 90% interval ($) | 1212.02 | 1750.91 | 2399.13 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0093 | [-0.0381, 0.0195] | NOISE |
| h1 | -0.0151 | [-0.0383, 0.0083] | NOISE |
| h2 | -0.0123 | [-0.0474, 0.0267] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.102 / h2 0.230) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4047 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.8766 | 0.9478 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.3859 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6153 | 0.7240 | 0.7577 | 0.4423 |
| expected if the two signs were independent | 0.6196 | 0.7168 | 0.6968 | 0.3648 |

- P(up) unanimity (all three horizons call the same side): 0.5403

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0236 vs 0.0025 (-0.0260): does not beat, noise (boot z -0.80) | -0.0243 vs 0.0205 (-0.0448): does not beat, noise (boot z -1.75) | 0.0076 vs -0.0138 (+0.0214): beats, noise (boot z +0.68) |
| logreg_lags | direction/auc | 0.4890 vs 0.5320 (-0.0430): does not beat, noise (boot z -1.80) | 0.4836 vs 0.5251 (-0.0414): does not beat, noise (boot z -1.86) | 0.5041 vs 0.5095 (-0.0054): does not beat, noise (boot z -0.41) |
| logreg_lags | direction/brier | 0.2578 vs 0.2536 (-0.0042): does not beat, noise (DM z -1.82) | 0.2537 vs 0.2575 (+0.0037): beats, noise (DM z +1.50) | 0.2577 vs 0.2702 (+0.0125): beats (DM z +3.30) |
| logreg_lags | direction/ece_pos | 0.0735 vs 0.0663 (-0.0072): does not beat, noise (boot z -0.51) | 0.0575 vs 0.0812 (+0.0237): beats (boot z +4.66) | 0.0783 vs 0.1230 (+0.0447): beats (boot z +8.87) |
| logreg_lags | direction/acc | 0.4839 vs 0.4836 (+0.0003): beats, noise (DM z +0.02) | 0.4791 vs 0.4911 (-0.0121): does not beat, noise (DM z -1.69) | 0.4848 vs 0.4756 (+0.0092): beats, noise (DM z +0.72) |
| logreg_lags | direction/bal_acc | 0.4887 vs 0.5004 (-0.0117): does not beat, noise (boot z -1.03) | 0.4931 vs 0.5055 (-0.0124): does not beat, noise (boot z -1.75) | 0.5025 vs 0.4971 (+0.0054): beats, noise (boot z +0.59) |
| class_prior | direction/mcc | -0.0236 vs 0.0000 (-0.0236): does not beat, noise (boot z -1.10) | -0.0243 vs 0.0000 (-0.0243): does not beat, noise (boot z -1.72) | 0.0076 vs 0.0000 (+0.0076): beats, noise (boot z +0.31) |
| class_prior | direction/auc | 0.4890 vs 0.5000 (-0.0110): does not beat, noise (boot z -0.78) | 0.4836 vs 0.5000 (-0.0164): does not beat, noise (boot z -1.75) | 0.5041 vs 0.5000 (+0.0041): beats, noise (boot z +0.26) |
| class_prior | direction/brier | 0.2578 vs 0.2532 (-0.0046): does not beat, significantly worse (DM z -2.47) | 0.2537 vs 0.2533 (-0.0004): does not beat, noise (DM z -0.59) | 0.2577 vs 0.2573 (-0.0004): does not beat, noise (DM z -0.19) |
| class_prior | direction/ece_pos | 0.0735 vs 0.0593 (-0.0142): does not beat, noise (boot z -1.01) | 0.0575 vs 0.0603 (+0.0028): beats, noise (boot z +0.59) | 0.0783 vs 0.0886 (+0.0103): beats (boot z +2.32) |
| class_prior | direction/acc | 0.4839 vs 0.4824 (+0.0015): beats, noise (DM z +0.09) | 0.4791 vs 0.4829 (-0.0039): does not beat, noise (DM z -0.65) | 0.4848 vs 0.4763 (+0.0084): beats, noise (DM z +0.62) |
| class_prior | direction/bal_acc | 0.4887 vs 0.5000 (-0.0113): does not beat, noise (boot z -1.10) | 0.4931 vs 0.5000 (-0.0069): does not beat, noise (boot z -1.71) | 0.5025 vs 0.5000 (+0.0025): beats, noise (boot z +0.31) |
| zero_delta | delta/rmse | 400.31 vs 400.31 (+0.00, +0.00%): does not beat | 564.50 vs 563.88 (-0.61, -0.11%): does not beat, significantly worse (DM z -2.03) | 811.93 vs 807.63 (-4.31, -0.53%): does not beat, noise (DM z -1.58) |
| zero_delta | delta/mae | 271.46 vs 271.46 (+0.00, +0.00%): does not beat | 378.37 vs 377.88 (-0.49, -0.13%): does not beat, noise (DM z -1.85) | 554.91 vs 550.98 (-3.93, -0.71%): does not beat, noise (DM z -1.83) |
| mean_delta | delta/rmse | 400.31 vs 405.60 (+5.29, +1.31%): beats (DM z +2.85) | 564.50 vs 578.56 (+14.07, +2.43%): beats (DM z +2.80) | 811.93 vs 847.03 (+35.10, +4.14%): beats (DM z +2.88) |
| mean_delta | delta/mae | 271.46 vs 277.50 (+6.04, +2.18%): beats (DM z +3.94) | 378.37 vs 393.77 (+15.40, +3.91%): beats (DM z +3.80) | 554.91 vs 595.01 (+40.10, +6.74%): beats (DM z +3.92) |
| const_var | variance/crps | 202.21 vs 206.55 (+4.34, +2.10%): beats (DM z +4.56) | 285.96 vs 294.69 (+8.73, +2.96%): beats (DM z +3.33) | 420.13 vs 444.83 (+24.70, +5.55%): beats (DM z +3.77) |
| const_var | variance/nll | 7.4174 vs 7.4448 (+0.0274): beats (DM z +3.72) | 7.7782 vs 7.8022 (+0.0240): beats, noise (DM z +1.81) | 8.1736 vs 8.2104 (+0.0368): beats, noise (DM z +1.46) |
| const_var | variance/pit_ks | 0.0455 vs 0.1064 (+0.0609): beats (boot z +8.06) | 0.0644 vs 0.1404 (+0.0760): beats (boot z +10.08) | 0.0701 vs 0.1810 (+0.1110): beats (boot z +31.69) |
| const_var | variance/corr_var_err2_spearman | 0.1430 vs 0.0000 (+0.1430): beats (boot z +8.07) | 0.0273 vs 0.0000 (+0.0273): beats, noise (boot z +1.10) | 0.0914 vs 0.0000 (+0.0914): beats (boot z +4.20) |

## Backtest (costs included)

- n_trades: 1446
- total_return: -0.9792
- sharpe_net: -137.2135
- sharpe_gross: -5.3027
- sortino: -152.9049
- max_drawdown: 0.9792
- hit_rate: 0.0602
- hit_rate_gross: 0.4371
- profit_factor: 0.0319
- avg_hold_bars: 11.7842
- exposure: 0.3945
- turnover: 725.1618
- fees_paid: 7251.9075
- traded_notional: 7251907.5239
- breakeven_cost_bps: -1.0053
- gross_edge_per_trade_bps: -0.7298
- costs_paid: 9427.4798
- gross_pnl: -364.5172
- net_pnl: -9791.9970

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8246, long_above 0.5850, short_below 0.4848, median 0.5309. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.92% | -137.21 | +97.92% | 1446 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.90% .. -97.19%) | -97.60% | -143.07 | | |

The random null enters at the strategy's rate (0.0553 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 4% of its seeds on net return, 94% on net Sharpe and 8% on gross return.
