# Evaluation report - dev split - run `20260930T035044Z-ce1e2ed-dd27d60a-clip100__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.6099 | 0.7712 | 0.7123 |
| accuracy | 0.4731 | 0.4879 | 0.4856 |
| balanced accuracy | 0.4770 | 0.4971 | 0.4957 |
| precision (up) | 0.4636 | 0.4811 | 0.4733 |
| recall / sensitivity (up) | 0.5861 | 0.7683 | 0.7077 |
| specificity (down) | 0.3679 | 0.2260 | 0.2836 |
| F1 (up) | 0.5177 | 0.5917 | 0.5672 |
| MCC | -0.0472 | -0.0068 | -0.0096 |
| AUC | 0.4724 | 0.5051 | 0.4974 |
| Brier | 0.2563 | 0.2576 | 0.2561 |
| ECE (positive class) | 0.0694 | 0.0702 | 0.0637 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 10261 / 11874 / 6910 / 7247 | 14027 / 15130 / 4418 / 4231 | 13297 / 14798 / 5858 / 5491 |
| Gaussian readout: calls up | 0.6046 | n/a (beta = 0: readout is the constant 0.5) | 0.7680 |
| Gaussian readout: MCC | -0.0186 | n/a (beta = 0: readout is the constant 0.5) | 0.0107 |
| Gaussian readout: AUC | 0.5020 | n/a (beta = 0: readout is the constant 0.5) | 0.5071 |
| Gaussian readout: Brier | 0.2506 | n/a (beta = 0: readout is the constant 0.5) | 0.2515 |
| Gaussian readout: ECE | 0.0317 | n/a (beta = 0: readout is the constant 0.5) | 0.0415 |
| Gaussian readout of the raw heads: calls up | 0.6046 | 0.8157 | 0.7680 |
| Gaussian readout of the raw heads: MCC | -0.0186 | 0.0071 | 0.0107 |
| Gaussian readout of the raw heads: AUC | 0.5020 | 0.4920 | 0.5071 |
| Gaussian readout of the raw heads: Brier | 0.2609 | 0.2614 | 0.2581 |
| Gaussian readout of the raw heads: ECE | 0.0973 | 0.0760 | 0.0810 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.15 | 563.88 | 811.27 |
| RMSE ($), raw heads | 413.28 | 587.25 | 824.29 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.00 | 377.88 | 553.84 |
| MAE ($), raw heads | 282.35 | 400.62 | 567.00 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0042 | n/a (beta = 0: served delta is 0) | -0.0090 |
| skill vs zero, raw heads | -0.0659 | -0.0846 | -0.0417 |
| EV, served | -0.0032 | n/a (beta = 0: served delta is 0) | -0.0033 |
| EV, raw heads | -0.0574 | -0.0502 | -0.0204 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0053 | -0.0360 | -0.0052 |
| corr, Spearman, raw heads | -0.0101 | -0.0268 | -0.0038 |
| mean predicted ($), served | 6.03 | 0.00 | 31.74 |
| mean predicted ($), raw heads | 27.50 | 84.84 | 82.80 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.5906 | 0.8076 | 0.7673 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.2191 | 0.0000 | 0.3833 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 203.30 | 288.40 | 419.22 |
| CRPSS vs constant variance | 0.0157 | 0.0213 | 0.0576 |
| NLL | 7.3906 | 7.7557 | 8.1680 |
| PIT KS | 0.0710 | 0.0849 | 0.0688 |
| var / err^2 Spearman | 0.1857 | 0.1052 | 0.1169 |
| coverage of the 90% interval | 0.9027 | 0.9027 | 0.8625 |
| width of the 90% interval ($) | 1219.29 | 1755.40 | 2398.79 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0213 | [-0.0473, 0.0037] | NOISE |
| h1 | -0.0003 | [-0.0305, 0.0299] | NOISE |
| h2 | -0.0075 | [-0.0355, 0.0207] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.219 / h1 0.000 / h2 0.383) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5120 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.6673 | n/a (beta = 0: served delta is 0) | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.2421 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4910 | 0.7826 | 0.6142 | 0.2658 |
| expected if the two signs were independent | 0.5221 | 0.6626 | 0.6127 | 0.2267 |

- P(up) unanimity (all three horizons call the same side): 0.3359

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0472 vs 0.0025 (-0.0497): does not beat, noise (boot z -1.68) | -0.0068 vs 0.0205 (-0.0273): does not beat, noise (boot z -1.00) | -0.0096 vs -0.0138 (+0.0043): beats, noise (boot z +0.15) |
| logreg_lags | direction/auc | 0.4724 vs 0.5320 (-0.0596): does not beat, significantly worse (boot z -2.31) | 0.5051 vs 0.5251 (-0.0200): does not beat, noise (boot z -1.56) | 0.4974 vs 0.5095 (-0.0121): does not beat, noise (boot z -0.57) |
| logreg_lags | direction/brier | 0.2563 vs 0.2536 (-0.0027): does not beat, noise (DM z -1.21) | 0.2576 vs 0.2575 (-0.0002): does not beat, noise (DM z -0.12) | 0.2561 vs 0.2702 (+0.0141): beats (DM z +2.37) |
| logreg_lags | direction/ece_pos | 0.0694 vs 0.0663 (-0.0031): does not beat, noise (boot z -0.19) | 0.0702 vs 0.0812 (+0.0111): beats, noise (boot z +1.37) | 0.0637 vs 0.1230 (+0.0593): beats (boot z +5.31) |
| logreg_lags | direction/acc | 0.4731 vs 0.4836 (-0.0105): does not beat, noise (DM z -0.65) | 0.4879 vs 0.4911 (-0.0032): does not beat, noise (DM z -0.22) | 0.4856 vs 0.4756 (+0.0100): beats, noise (DM z +0.52) |
| logreg_lags | direction/bal_acc | 0.4770 vs 0.5004 (-0.0234): does not beat, significantly worse (boot z -2.31) | 0.4971 vs 0.5055 (-0.0084): does not beat, noise (boot z -0.87) | 0.4957 vs 0.4971 (-0.0015): does not beat, noise (boot z -0.15) |
| class_prior | direction/mcc | -0.0472 vs 0.0000 (-0.0472): does not beat, significantly worse (boot z -2.50) | -0.0068 vs 0.0000 (-0.0068): does not beat, noise (boot z -0.35) | -0.0096 vs 0.0000 (-0.0096): does not beat, noise (boot z -0.49) |
| class_prior | direction/auc | 0.4724 vs 0.5000 (-0.0276): does not beat, significantly worse (boot z -2.32) | 0.5051 vs 0.5000 (+0.0051): beats, noise (boot z +0.40) | 0.4974 vs 0.5000 (-0.0026): does not beat, noise (boot z -0.20) |
| class_prior | direction/brier | 0.2563 vs 0.2532 (-0.0031): does not beat, noise (DM z -1.85) | 0.2576 vs 0.2533 (-0.0043): does not beat, significantly worse (DM z -2.14) | 0.2561 vs 0.2573 (+0.0012): beats, noise (DM z +0.46) |
| class_prior | direction/ece_pos | 0.0694 vs 0.0593 (-0.0100): does not beat, noise (boot z -0.63) | 0.0702 vs 0.0603 (-0.0098): does not beat, noise (boot z -1.18) | 0.0637 vs 0.0886 (+0.0249): beats (boot z +2.21) |
| class_prior | direction/acc | 0.4731 vs 0.4824 (-0.0093): does not beat, noise (DM z -0.58) | 0.4879 vs 0.4829 (+0.0049): beats, noise (DM z +0.33) | 0.4856 vs 0.4763 (+0.0093): beats, noise (DM z +0.45) |
| class_prior | direction/bal_acc | 0.4770 vs 0.5000 (-0.0230): does not beat, significantly worse (boot z -2.51) | 0.4971 vs 0.5000 (-0.0029): does not beat, noise (boot z -0.35) | 0.4957 vs 0.5000 (-0.0043): does not beat, noise (boot z -0.49) |
| zero_delta | delta/rmse | 401.15 vs 400.31 (-0.85, -0.21%): does not beat, noise (DM z -1.24) | 563.88 vs 563.88 (+0.00, +0.00%): does not beat | 811.27 vs 807.63 (-3.64, -0.45%): does not beat, noise (DM z -1.22) |
| zero_delta | delta/mae | 272.00 vs 271.46 (-0.55, -0.20%): does not beat, noise (DM z -1.07) | 377.88 vs 377.88 (+0.00, +0.00%): does not beat | 553.84 vs 550.98 (-2.86, -0.52%): does not beat, noise (DM z -1.26) |
| mean_delta | delta/rmse | 401.15 vs 405.60 (+4.45, +1.10%): beats (DM z +2.88) | 563.88 vs 578.56 (+14.68, +2.54%): beats (DM z +2.78) | 811.27 vs 847.03 (+35.77, +4.22%): beats (DM z +2.95) |
| mean_delta | delta/mae | 272.00 vs 277.50 (+5.49, +1.98%): beats (DM z +3.77) | 377.88 vs 393.77 (+15.89, +4.04%): beats (DM z +3.72) | 553.84 vs 595.01 (+41.17, +6.92%): beats (DM z +3.90) |
| const_var | variance/crps | 203.30 vs 206.55 (+3.25, +1.57%): beats (DM z +3.56) | 288.40 vs 294.69 (+6.29, +2.13%): beats (DM z +2.15) | 419.22 vs 444.83 (+25.60, +5.76%): beats (DM z +3.77) |
| const_var | variance/nll | 7.3906 vs 7.4448 (+0.0542): beats (DM z +3.38) | 7.7557 vs 7.8022 (+0.0465): beats (DM z +2.09) | 8.1680 vs 8.2104 (+0.0425): beats, noise (DM z +1.53) |
| const_var | variance/pit_ks | 0.0710 vs 0.1064 (+0.0354): beats (boot z +5.97) | 0.0849 vs 0.1404 (+0.0555): beats (boot z +5.79) | 0.0688 vs 0.1810 (+0.1122): beats (boot z +24.32) |
| const_var | variance/corr_var_err2_spearman | 0.1857 vs 0.0000 (+0.1857): beats (boot z +10.18) | 0.1052 vs 0.0000 (+0.1052): beats (boot z +4.36) | 0.1169 vs 0.0000 (+0.1169): beats (boot z +4.62) |

## Backtest (costs included)

- n_trades: 1249
- total_return: -0.9610
- sharpe_net: -118.9217
- sharpe_gross: 1.8143
- sortino: -136.1035
- max_drawdown: 0.9610
- hit_rate: 0.0616
- hit_rate_gross: 0.4860
- profit_factor: 0.0309
- avg_hold_bars: 11.7342
- exposure: 0.3393
- turnover: 749.9306
- fees_paid: 7499.3784
- traded_notional: 7499378.3600
- breakeven_cost_bps: 0.3722
- gross_edge_per_trade_bps: 0.0850
- costs_paid: 9749.1919
- gross_pnl: 139.5815
- net_pnl: -9609.6104

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.054, long_above 0.5679, short_below 0.4838, median 0.5206. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.10% | -118.92 | +96.10% | 1249 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.56% .. -95.35%) | -96.00% | -132.55 | | |

The random null enters at the strategy's rate (0.0438 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 37% of its seeds on net return, 100% on net Sharpe and 78% on gross return.
