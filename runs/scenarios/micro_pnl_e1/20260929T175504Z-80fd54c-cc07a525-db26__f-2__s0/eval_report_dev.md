# Evaluation report - dev split - run `20260929T175504Z-80fd54c-cc07a525-db26__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 26 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 26 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 13621 | 19285 | 24800 |
| n_eff of the scored moves (n scored // bars ahead) | 227 | 160 | 103 |
| true up-rate | 0.4907 | 0.4802 | 0.4752 |
| calls up (predicted up-rate) | 0.7254 | 0.9930 | 0.9712 |
| accuracy | 0.4612 | 0.4787 | 0.4756 |
| balanced accuracy | 0.4654 | 0.4982 | 0.4990 |
| precision (up) | 0.4669 | 0.4793 | 0.4746 |
| recall / sensitivity (up) | 0.6902 | 0.9911 | 0.9701 |
| specificity (down) | 0.2406 | 0.0053 | 0.0278 |
| F1 (up) | 0.5570 | 0.6462 | 0.6374 |
| MCC | -0.0776 | -0.0214 | -0.0061 |
| AUC | 0.4564 | 0.4746 | 0.5040 |
| Brier | 0.2668 | 0.2598 | 0.2612 |
| ECE (positive class) | 0.1179 | 0.0949 | 0.1017 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0093 | 0.0198 | 0.0248 |
| TP / FP / TN / FN | 4613 / 5268 / 1669 / 2071 | 9179 / 9971 / 53 / 82 | 11432 / 12654 / 362 / 352 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.9833 | 0.7635 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0007 | -0.0195 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4628 | 0.4917 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2502 | 0.2512 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0213 | 0.0400 |
| Gaussian readout of the raw heads: calls up | 0.9909 | 0.9833 | 0.7635 |
| Gaussian readout of the raw heads: MCC | -0.0435 | -0.0007 | -0.0195 |
| Gaussian readout of the raw heads: AUC | 0.4393 | 0.4628 | 0.4917 |
| Gaussian readout of the raw heads: Brier | 0.2651 | 0.2634 | 0.2589 |
| Gaussian readout of the raw heads: ECE | 0.0987 | 0.0868 | 0.0790 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.31 | 563.99 | 809.21 |
| RMSE ($), raw heads | 408.45 | 573.40 | 818.94 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.46 | 377.96 | 552.20 |
| MAE ($), raw heads | 279.58 | 387.79 | 563.36 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0004 | -0.0039 |
| skill vs zero, raw heads | -0.0411 | -0.0341 | -0.0282 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0002 | -0.0008 |
| EV, raw heads | -0.0159 | -0.0158 | -0.0132 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0823 | -0.0423 | 0.0084 |
| corr, Spearman, raw heads | -0.0683 | -0.0382 | -0.0127 |
| mean predicted ($), served | 0.00 | 1.29 | 19.65 |
| mean predicted ($), raw heads | 53.53 | 57.34 | 65.14 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9935 | 0.9788 | 0.7527 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0225 | 0.3016 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 206.01 | 285.13 | 419.46 |
| CRPSS vs constant variance | 0.0026 | 0.0324 | 0.0570 |
| NLL | 7.4082 | 7.7840 | 8.1426 |
| PIT KS | 0.0818 | 0.0569 | 0.0771 |
| var / err^2 Spearman | 0.0104 | -0.0190 | -0.0048 |
| coverage of the 90% interval | 0.9028 | 0.9023 | 0.8621 |
| width of the 90% interval ($) | 1212.02 | 1753.36 | 2397.17 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0078 | [-0.0582, 0.0449] | NOISE |
| h1 | -0.0394 | [-0.0780, 0.0022] | NOISE |
| h2 | 0.0079 | [-0.0396, 0.0544] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.022 / h2 0.302) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4207 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.5729 | 0.9688 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.2161 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6989 | 0.9745 | 0.7461 | 0.5085 |
| expected if the two signs were independent | 0.7016 | 0.9744 | 0.7355 | 0.5112 |

- P(up) unanimity (all three horizons call the same side): 0.6855

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0776 vs 0.0249 (-0.1025): does not beat, noise (boot z -1.89) | -0.0214 vs 0.0086 (-0.0300): does not beat, noise (boot z -1.11) | -0.0061 vs -0.0220 (+0.0159): beats, noise (boot z +0.57) |
| logreg_lags | direction/auc | 0.4564 vs 0.4982 (-0.0418): does not beat, noise (boot z -1.00) | 0.4746 vs 0.5132 (-0.0386): does not beat, noise (boot z -1.57) | 0.5040 vs 0.5054 (-0.0015): does not beat, noise (boot z -0.10) |
| logreg_lags | direction/brier | 0.2668 vs 0.2667 (-0.0001): does not beat, noise (DM z -0.01) | 0.2598 vs 0.2794 (+0.0196): beats (DM z +3.49) | 0.2612 vs 0.2971 (+0.0359): beats (DM z +3.83) |
| logreg_lags | direction/ece_pos | 0.1179 vs 0.1201 (+0.0022): beats, noise (boot z +0.11) | 0.0949 vs 0.1682 (+0.0733): beats (boot z +21.39) | 0.1017 vs 0.2020 (+0.1003): beats (boot z +28.04) |
| logreg_lags | direction/acc | 0.4612 vs 0.4967 (-0.0355): does not beat, noise (DM z -1.61) | 0.4787 vs 0.4822 (-0.0035): does not beat, noise (DM z -0.90) | 0.4756 vs 0.4742 (+0.0014): beats, noise (DM z +0.37) |
| logreg_lags | direction/bal_acc | 0.4654 vs 0.5052 (-0.0398): does not beat, significantly worse (boot z -2.33) | 0.4982 vs 0.5012 (-0.0030): does not beat, noise (boot z -0.76) | 0.4990 vs 0.4989 (+0.0001): beats, noise (boot z +0.04) |
| class_prior | direction/mcc | -0.0776 vs 0.0000 (-0.0776): does not beat, significantly worse (boot z -2.46) | -0.0214 vs 0.0000 (-0.0214): does not beat, noise (boot z -1.31) | -0.0061 vs 0.0000 (-0.0061): does not beat, noise (boot z -0.37) |
| class_prior | direction/auc | 0.4564 vs 0.5000 (-0.0436): does not beat, significantly worse (boot z -1.97) | 0.4746 vs 0.5000 (-0.0254): does not beat, noise (boot z -1.93) | 0.5040 vs 0.5000 (+0.0040): beats, noise (boot z +0.23) |
| class_prior | direction/brier | 0.2668 vs 0.2657 (-0.0011): does not beat, noise (DM z -0.21) | 0.2598 vs 0.2738 (+0.0140): beats (DM z +3.40) | 0.2612 vs 0.2756 (+0.0143): beats (DM z +2.75) |
| class_prior | direction/ece_pos | 0.1179 vs 0.1255 (+0.0076): beats, noise (boot z +0.38) | 0.0949 vs 0.1556 (+0.0607): beats (boot z +29.64) | 0.1017 vs 0.1618 (+0.0601): beats (boot z +25.16) |
| class_prior | direction/acc | 0.4612 vs 0.4907 (-0.0295): does not beat, noise (DM z -1.42) | 0.4787 vs 0.4802 (-0.0015): does not beat, noise (DM z -1.01) | 0.4756 vs 0.4752 (+0.0004): beats, noise (DM z +0.11) |
| class_prior | direction/bal_acc | 0.4654 vs 0.5000 (-0.0346): does not beat, significantly worse (boot z -2.47) | 0.4982 vs 0.5000 (-0.0018): does not beat, noise (boot z -1.28) | 0.4990 vs 0.5000 (-0.0010): does not beat, noise (boot z -0.37) |
| zero_delta | delta/rmse | 400.31 vs 400.31 (+0.00, +0.00%): does not beat | 563.99 vs 563.88 (-0.10, -0.02%): does not beat, noise (DM z -1.63) | 809.21 vs 807.63 (-1.59, -0.20%): does not beat, noise (DM z -1.01) |
| zero_delta | delta/mae | 271.46 vs 271.46 (+0.00, +0.00%): does not beat | 377.96 vs 377.88 (-0.08, -0.02%): does not beat, noise (DM z -1.51) | 552.20 vs 550.98 (-1.22, -0.22%): does not beat, noise (DM z -0.98) |
| mean_delta | delta/rmse | 400.31 vs 405.60 (+5.29, +1.31%): beats (DM z +2.85) | 563.99 vs 578.56 (+14.58, +2.52%): beats (DM z +2.79) | 809.21 vs 847.03 (+37.82, +4.46%): beats (DM z +2.83) |
| mean_delta | delta/mae | 271.46 vs 277.50 (+6.04, +2.18%): beats (DM z +3.94) | 377.96 vs 393.77 (+15.81, +4.01%): beats (DM z +3.75) | 552.20 vs 595.01 (+42.81, +7.20%): beats (DM z +3.87) |
| const_var | variance/crps | 206.01 vs 206.55 (+0.54, +0.26%): beats, noise (DM z +0.50) | 285.13 vs 294.69 (+9.56, +3.24%): beats (DM z +3.55) | 419.46 vs 444.83 (+25.36, +5.70%): beats (DM z +3.51) |
| const_var | variance/nll | 7.4082 vs 7.4448 (+0.0366): beats, noise (DM z +1.59) | 7.7840 vs 7.8022 (+0.0182): beats, noise (DM z +1.10) | 8.1426 vs 8.2104 (+0.0679): beats (DM z +2.88) |
| const_var | variance/pit_ks | 0.0818 vs 0.1064 (+0.0246): beats (boot z +2.72) | 0.0569 vs 0.1404 (+0.0835): beats (boot z +9.80) | 0.0771 vs 0.1810 (+0.1039): beats (boot z +20.57) |
| const_var | variance/corr_var_err2_spearman | 0.0104 vs 0.0000 (+0.0104): beats, noise (boot z +0.64) | -0.0190 vs 0.0000 (-0.0190): does not beat, noise (boot z -0.78) | -0.0048 vs 0.0000 (-0.0048): does not beat, noise (boot z -0.23) |

## Backtest (costs included)

- n_trades: 1391
- total_return: -0.9774
- sharpe_net: -133.5944
- sharpe_gross: -7.9555
- sortino: -150.2685
- max_drawdown: 0.9774
- hit_rate: 0.0676
- hit_rate_gross: 0.3954
- profit_factor: 0.0285
- avg_hold_bars: 11.7793
- exposure: 0.3793
- turnover: 708.5654
- fees_paid: 7086.0855
- traded_notional: 7086085.5477
- breakeven_cost_bps: -1.5854
- gross_edge_per_trade_bps: -1.1794
- costs_paid: 9211.9112
- gross_pnl: -561.7113
- net_pnl: -9773.6226

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7967, long_above 0.6068, short_below 0.5179, median 0.5541. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.74% | -133.59 | +97.74% | 1391 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.59% .. -96.84%) | -97.23% | -140.26 | | |

The random null enters at the strategy's rate (0.0519 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 1% of its seeds on net return, 96% on net Sharpe and 1% on gross return.
