# Evaluation report - dev split - run `20260929T204654Z-3ffb863-7bb143c2-lam0__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5541 | 0.8256 | 0.8574 |
| accuracy | 0.4829 | 0.4897 | 0.4754 |
| balanced accuracy | 0.4848 | 0.5008 | 0.4923 |
| precision (up) | 0.4687 | 0.4835 | 0.4719 |
| recall / sensitivity (up) | 0.5383 | 0.8264 | 0.8494 |
| specificity (down) | 0.4312 | 0.1753 | 0.1353 |
| F1 (up) | 0.5011 | 0.6100 | 0.6067 |
| MCC | -0.0306 | 0.0022 | -0.0219 |
| AUC | 0.4818 | 0.5179 | 0.4953 |
| Brier | 0.2553 | 0.2574 | 0.2584 |
| ECE (positive class) | 0.0580 | 0.0719 | 0.0844 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 9425 / 10684 / 8100 / 8083 | 15089 / 16122 / 3426 / 3169 | 15958 / 17861 / 2795 / 2830 |
| Gaussian readout: calls up | 0.8823 | 0.9337 | 0.8159 |
| Gaussian readout: MCC | -0.0153 | -0.0067 | 0.0131 |
| Gaussian readout: AUC | 0.4934 | 0.5205 | 0.5141 |
| Gaussian readout: Brier | 0.2507 | 0.2504 | 0.2516 |
| Gaussian readout: ECE | 0.0308 | 0.0280 | 0.0465 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.91 | 564.78 | 811.40 |
| RMSE ($), raw heads | 420.97 | 574.62 | 827.77 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.99 | 378.42 | 553.92 |
| MAE ($), raw heads | 289.11 | 387.04 | 570.22 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0030 | -0.0032 | -0.0094 |
| skill vs zero, raw heads | -0.1059 | -0.0384 | -0.0505 |
| EV, served | -0.0009 | -0.0005 | -0.0020 |
| EV, raw heads | -0.0442 | -0.0086 | -0.0181 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0086 | -0.0083 | 0.0032 |
| corr, Spearman, raw heads | -0.0158 | 0.0105 | 0.0064 |
| mean predicted ($), served | 10.28 | 14.44 | 38.44 |
| mean predicted ($), raw heads | 89.08 | 77.79 | 108.91 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.8758 | 0.9279 | 0.8141 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1154 | 0.1857 | 0.3529 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 203.02 | 288.66 | 419.09 |
| CRPSS vs constant variance | 0.0171 | 0.0205 | 0.0579 |
| NLL | 7.3865 | 7.7508 | 8.1783 |
| PIT KS | 0.0733 | 0.0948 | 0.0699 |
| var / err^2 Spearman | 0.2005 | 0.1155 | 0.1190 |
| coverage of the 90% interval | 0.9011 | 0.9014 | 0.8641 |
| width of the 90% interval ($) | 1208.31 | 1754.87 | 2407.38 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0099 | [-0.0340, 0.0142] | NOISE |
| h1 | 0.0155 | [-0.0170, 0.0483] | NOISE |
| h2 | -0.0012 | [-0.0311, 0.0303] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.115 / h1 0.186 / h2 0.353) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4636 | 0.7782 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.7771 | 0.9021 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.2889 | 0.6891 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5211 | 0.8309 | 0.7237 | 0.3464 |
| expected if the two signs were independent | 0.5470 | 0.7738 | 0.7243 | 0.3260 |

- P(up) unanimity (all three horizons call the same side): 0.4044

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0306 vs 0.0025 (-0.0331): does not beat, noise (boot z -1.12) | 0.0022 vs 0.0205 (-0.0183): does not beat, noise (boot z -0.72) | -0.0219 vs -0.0138 (-0.0081): does not beat, noise (boot z -0.30) |
| logreg_lags | direction/auc | 0.4818 vs 0.5320 (-0.0502): does not beat, significantly worse (boot z -2.04) | 0.5179 vs 0.5251 (-0.0072): does not beat, noise (boot z -0.68) | 0.4953 vs 0.5095 (-0.0142): does not beat, noise (boot z -0.62) |
| logreg_lags | direction/brier | 0.2553 vs 0.2536 (-0.0017): does not beat, noise (DM z -0.73) | 0.2574 vs 0.2575 (+0.0000): beats, noise (DM z +0.03) | 0.2584 vs 0.2702 (+0.0118): beats (DM z +2.29) |
| logreg_lags | direction/ece_pos | 0.0580 vs 0.0663 (+0.0083): beats, noise (boot z +0.51) | 0.0719 vs 0.0812 (+0.0093): beats, noise (boot z +1.79) | 0.0844 vs 0.1230 (+0.0386): beats (boot z +4.91) |
| logreg_lags | direction/acc | 0.4829 vs 0.4836 (-0.0007): does not beat, noise (DM z -0.04) | 0.4897 vs 0.4911 (-0.0014): does not beat, noise (DM z -0.12) | 0.4754 vs 0.4756 (-0.0002): does not beat, noise (DM z -0.01) |
| logreg_lags | direction/bal_acc | 0.4848 vs 0.5004 (-0.0156): does not beat, noise (boot z -1.54) | 0.5008 vs 0.5055 (-0.0047): does not beat, noise (boot z -0.56) | 0.4923 vs 0.4971 (-0.0048): does not beat, noise (boot z -0.64) |
| class_prior | direction/mcc | -0.0306 vs 0.0000 (-0.0306): does not beat, noise (boot z -1.66) | 0.0022 vs 0.0000 (+0.0022): beats, noise (boot z +0.11) | -0.0219 vs 0.0000 (-0.0219): does not beat, noise (boot z -1.27) |
| class_prior | direction/auc | 0.4818 vs 0.5000 (-0.0182): does not beat, noise (boot z -1.62) | 0.5179 vs 0.5000 (+0.0179): beats, noise (boot z +1.36) | 0.4953 vs 0.5000 (-0.0047): does not beat, noise (boot z -0.40) |
| class_prior | direction/brier | 0.2553 vs 0.2532 (-0.0021): does not beat, noise (DM z -1.16) | 0.2574 vs 0.2533 (-0.0041): does not beat, significantly worse (DM z -1.98) | 0.2584 vs 0.2573 (-0.0011): does not beat, noise (DM z -0.74) |
| class_prior | direction/ece_pos | 0.0580 vs 0.0593 (+0.0013): beats, noise (boot z +0.08) | 0.0719 vs 0.0603 (-0.0116): does not beat, significantly worse (boot z -2.03) | 0.0844 vs 0.0886 (+0.0042): beats, noise (boot z +0.58) |
| class_prior | direction/acc | 0.4829 vs 0.4824 (+0.0005): beats, noise (DM z +0.03) | 0.4897 vs 0.4829 (+0.0068): beats, noise (DM z +0.57) | 0.4754 vs 0.4763 (-0.0009): does not beat, noise (DM z -0.08) |
| class_prior | direction/bal_acc | 0.4848 vs 0.5000 (-0.0152): does not beat, noise (boot z -1.66) | 0.5008 vs 0.5000 (+0.0008): beats, noise (boot z +0.11) | 0.4923 vs 0.5000 (-0.0077): does not beat, noise (boot z -1.26) |
| zero_delta | delta/rmse | 400.91 vs 400.31 (-0.60, -0.15%): does not beat, noise (DM z -1.19) | 564.78 vs 563.88 (-0.89, -0.16%): does not beat, noise (DM z -1.04) | 811.40 vs 807.63 (-3.77, -0.47%): does not beat, noise (DM z -1.12) |
| zero_delta | delta/mae | 271.99 vs 271.46 (-0.53, -0.19%): does not beat, noise (DM z -1.52) | 378.42 vs 377.88 (-0.54, -0.14%): does not beat, noise (DM z -0.87) | 553.92 vs 550.98 (-2.94, -0.53%): does not beat, noise (DM z -1.18) |
| mean_delta | delta/rmse | 400.91 vs 405.60 (+4.69, +1.16%): beats (DM z +3.25) | 564.78 vs 578.56 (+13.79, +2.38%): beats (DM z +3.09) | 811.40 vs 847.03 (+35.64, +4.21%): beats (DM z +3.06) |
| mean_delta | delta/mae | 271.99 vs 277.50 (+5.51, +1.99%): beats (DM z +4.30) | 378.42 vs 393.77 (+15.35, +3.90%): beats (DM z +4.12) | 553.92 vs 595.01 (+41.09, +6.91%): beats (DM z +4.05) |
| const_var | variance/crps | 203.02 vs 206.55 (+3.53, +1.71%): beats (DM z +4.20) | 288.66 vs 294.69 (+6.03, +2.05%): beats (DM z +2.39) | 419.09 vs 444.83 (+25.74, +5.79%): beats (DM z +3.94) |
| const_var | variance/nll | 7.3865 vs 7.4448 (+0.0583): beats (DM z +3.57) | 7.7508 vs 7.8022 (+0.0514): beats (DM z +2.09) | 8.1783 vs 8.2104 (+0.0321): beats, noise (DM z +1.11) |
| const_var | variance/pit_ks | 0.0733 vs 0.1064 (+0.0331): beats (boot z +7.72) | 0.0948 vs 0.1404 (+0.0456): beats (boot z +7.57) | 0.0699 vs 0.1810 (+0.1111): beats (boot z +27.52) |
| const_var | variance/corr_var_err2_spearman | 0.2005 vs 0.0000 (+0.2005): beats (boot z +10.77) | 0.1155 vs 0.0000 (+0.1155): beats (boot z +4.75) | 0.1190 vs 0.0000 (+0.1190): beats (boot z +4.71) |

## Backtest (costs included)

- n_trades: 1203
- total_return: -0.9556
- sharpe_net: -114.8102
- sharpe_gross: 1.5401
- sortino: -131.7173
- max_drawdown: 0.9557
- hit_rate: 0.0615
- hit_rate_gross: 0.5054
- profit_factor: 0.0282
- avg_hold_bars: 11.9800
- exposure: 0.3336
- turnover: 744.4404
- fees_paid: 7444.4809
- traded_notional: 7444480.8869
- breakeven_cost_bps: 0.3262
- gross_edge_per_trade_bps: 0.1554
- costs_paid: 9677.8252
- gross_pnl: 121.4244
- net_pnl: -9556.4008

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.075, long_above 0.5639, short_below 0.4863, median 0.5244. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -95.56% | -114.81 | +95.57% | 1203 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.20% .. -94.98%) | -95.59% | -130.48 | | |

The random null enters at the strategy's rate (0.0418 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 57% of its seeds on net return, 100% on net Sharpe and 74% on gross return.
