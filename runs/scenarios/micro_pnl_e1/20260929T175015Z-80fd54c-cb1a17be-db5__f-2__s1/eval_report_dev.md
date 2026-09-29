# Evaluation report - dev split - run `20260929T175015Z-80fd54c-cb1a17be-db5__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5901 | 0.8448 | 0.8943 |
| accuracy | 0.4791 | 0.4846 | 0.4796 |
| balanced accuracy | 0.4823 | 0.4963 | 0.4982 |
| precision (up) | 0.4674 | 0.4808 | 0.4753 |
| recall / sensitivity (up) | 0.5718 | 0.8410 | 0.8924 |
| specificity (down) | 0.3928 | 0.1516 | 0.1040 |
| F1 (up) | 0.5144 | 0.6118 | 0.6203 |
| MCC | -0.0360 | -0.0102 | -0.0057 |
| AUC | 0.4821 | 0.5025 | 0.5014 |
| Brier | 0.2553 | 0.2561 | 0.2588 |
| ECE (positive class) | 0.0636 | 0.0688 | 0.0852 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 10011 / 11406 / 7378 / 7497 | 15355 / 16584 / 2964 / 2903 | 16767 / 18507 / 2149 / 2021 |
| Gaussian readout: calls up | 0.9506 | 0.9064 | 0.8960 |
| Gaussian readout: MCC | 0.0075 | 0.0011 | 0.0052 |
| Gaussian readout: AUC | 0.4974 | 0.5182 | 0.5117 |
| Gaussian readout: Brier | 0.2512 | 0.2517 | 0.2519 |
| Gaussian readout: ECE | 0.0358 | 0.0438 | 0.0506 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.51 | 566.45 | 811.09 |
| RMSE ($), raw heads | 411.21 | 577.74 | 821.46 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.33 | 379.23 | 553.97 |
| MAE ($), raw heads | 279.93 | 387.63 | 564.27 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0060 | -0.0091 | -0.0086 |
| skill vs zero, raw heads | -0.0552 | -0.0497 | -0.0346 |
| EV, served | -0.0032 | -0.0035 | -0.0007 |
| EV, raw heads | -0.0264 | -0.0217 | -0.0079 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0427 | -0.0159 | 0.0142 |
| corr, Spearman, raw heads | -0.0121 | 0.0164 | 0.0043 |
| mean predicted ($), served | 12.99 | 25.67 | 40.79 |
| mean predicted ($), raw heads | 57.84 | 74.97 | 95.94 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9489 | 0.9021 | 0.8959 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.2246 | 0.3424 | 0.4251 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.69 | 285.69 | 419.68 |
| CRPSS vs constant variance | 0.0187 | 0.0305 | 0.0565 |
| NLL | 7.4348 | 7.8389 | 8.1681 |
| PIT KS | 0.0517 | 0.0587 | 0.0757 |
| var / err^2 Spearman | 0.1030 | -0.0433 | 0.0901 |
| coverage of the 90% interval | 0.9001 | 0.9006 | 0.8649 |
| width of the 90% interval ($) | 1209.47 | 1756.47 | 2418.71 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0087 | [-0.0376, 0.0201] | NOISE |
| h1 | -0.0012 | [-0.0359, 0.0306] | NOISE |
| h2 | -0.0076 | [-0.0337, 0.0162] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.225 / h1 0.342 / h2 0.425) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6344 | 0.8755 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.5768 | 0.8290 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.3232 | 0.7333 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5824 | 0.8158 | 0.8010 | 0.4062 |
| expected if the two signs were independent | 0.5972 | 0.7724 | 0.8128 | 0.4051 |

- P(up) unanimity (all three horizons call the same side): 0.4785

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0360 vs 0.0025 (-0.0385): does not beat, noise (boot z -1.36) | -0.0102 vs 0.0205 (-0.0307): does not beat, noise (boot z -0.98) | -0.0057 vs -0.0138 (+0.0081): beats, noise (boot z +0.37) |
| logreg_lags | direction/auc | 0.4821 vs 0.5320 (-0.0499): does not beat, significantly worse (boot z -2.01) | 0.5025 vs 0.5251 (-0.0226): does not beat, noise (boot z -1.03) | 0.5014 vs 0.5095 (-0.0081): does not beat, noise (boot z -0.44) |
| logreg_lags | direction/brier | 0.2553 vs 0.2536 (-0.0017): does not beat, noise (DM z -0.66) | 0.2561 vs 0.2575 (+0.0014): beats, noise (DM z +0.54) | 0.2588 vs 0.2702 (+0.0114): beats (DM z +2.62) |
| logreg_lags | direction/ece_pos | 0.0636 vs 0.0663 (+0.0026): beats, noise (boot z +0.16) | 0.0688 vs 0.0812 (+0.0124): beats, noise (boot z +1.83) | 0.0852 vs 0.1230 (+0.0378): beats (boot z +7.19) |
| logreg_lags | direction/acc | 0.4791 vs 0.4836 (-0.0045): does not beat, noise (DM z -0.26) | 0.4846 vs 0.4911 (-0.0066): does not beat, noise (DM z -0.55) | 0.4796 vs 0.4756 (+0.0040): beats, noise (DM z +0.55) |
| logreg_lags | direction/bal_acc | 0.4823 vs 0.5004 (-0.0181): does not beat, noise (boot z -1.66) | 0.4963 vs 0.5055 (-0.0092): does not beat, noise (boot z -0.94) | 0.4982 vs 0.4971 (+0.0011): beats, noise (boot z +0.20) |
| class_prior | direction/mcc | -0.0360 vs 0.0000 (-0.0360): does not beat, noise (boot z -1.64) | -0.0102 vs 0.0000 (-0.0102): does not beat, noise (boot z -0.57) | -0.0057 vs 0.0000 (-0.0057): does not beat, noise (boot z -0.36) |
| class_prior | direction/auc | 0.4821 vs 0.5000 (-0.0179): does not beat, noise (boot z -1.28) | 0.5025 vs 0.5000 (+0.0025): beats, noise (boot z +0.19) | 0.5014 vs 0.5000 (+0.0014): beats, noise (boot z +0.15) |
| class_prior | direction/brier | 0.2553 vs 0.2532 (-0.0021): does not beat, noise (DM z -1.03) | 0.2561 vs 0.2533 (-0.0027): does not beat, noise (DM z -1.85) | 0.2588 vs 0.2573 (-0.0015): does not beat, noise (DM z -1.35) |
| class_prior | direction/ece_pos | 0.0636 vs 0.0593 (-0.0043): does not beat, noise (boot z -0.26) | 0.0688 vs 0.0603 (-0.0085): does not beat, noise (boot z -1.36) | 0.0852 vs 0.0886 (+0.0034): beats, noise (boot z +0.71) |
| class_prior | direction/acc | 0.4791 vs 0.4824 (-0.0033): does not beat, noise (DM z -0.19) | 0.4846 vs 0.4829 (+0.0016): beats, noise (DM z +0.16) | 0.4796 vs 0.4763 (+0.0032): beats, noise (DM z +0.40) |
| class_prior | direction/bal_acc | 0.4823 vs 0.5000 (-0.0177): does not beat, noise (boot z -1.64) | 0.4963 vs 0.5000 (-0.0037): does not beat, noise (boot z -0.57) | 0.4982 vs 0.5000 (-0.0018): does not beat, noise (boot z -0.36) |
| zero_delta | delta/rmse | 401.51 vs 400.31 (-1.20, -0.30%): does not beat, noise (DM z -1.78) | 566.45 vs 563.88 (-2.56, -0.45%): does not beat, noise (DM z -1.52) | 811.09 vs 807.63 (-3.47, -0.43%): does not beat, noise (DM z -0.98) |
| zero_delta | delta/mae | 272.33 vs 271.46 (-0.87, -0.32%): does not beat, noise (DM z -1.94) | 379.23 vs 377.88 (-1.35, -0.36%): does not beat, noise (DM z -1.10) | 553.97 vs 550.98 (-2.99, -0.54%): does not beat, noise (DM z -1.19) |
| mean_delta | delta/rmse | 401.51 vs 405.60 (+4.09, +1.01%): beats (DM z +2.62) | 566.45 vs 578.56 (+12.11, +2.09%): beats (DM z +3.11) | 811.09 vs 847.03 (+35.94, +4.24%): beats (DM z +3.13) |
| mean_delta | delta/mae | 272.33 vs 277.50 (+5.17, +1.86%): beats (DM z +4.27) | 379.23 vs 393.77 (+14.54, +3.69%): beats (DM z +4.31) | 553.97 vs 595.01 (+41.05, +6.90%): beats (DM z +4.11) |
| const_var | variance/crps | 202.69 vs 206.55 (+3.86, +1.87%): beats (DM z +5.28) | 285.69 vs 294.69 (+8.99, +3.05%): beats (DM z +4.33) | 419.68 vs 444.83 (+25.14, +5.65%): beats (DM z +3.97) |
| const_var | variance/nll | 7.4348 vs 7.4448 (+0.0100): beats, noise (DM z +1.17) | 7.8389 vs 7.8022 (-0.0367): does not beat, noise (DM z -1.15) | 8.1681 vs 8.2104 (+0.0423): beats, noise (DM z +1.72) |
| const_var | variance/pit_ks | 0.0517 vs 0.1064 (+0.0547): beats (boot z +24.15) | 0.0587 vs 0.1404 (+0.0817): beats (boot z +25.59) | 0.0757 vs 0.1810 (+0.1053): beats (boot z +31.77) |
| const_var | variance/corr_var_err2_spearman | 0.1030 vs 0.0000 (+0.1030): beats (boot z +5.09) | -0.0433 vs 0.0000 (-0.0433): does not beat, noise (boot z -1.66) | 0.0901 vs 0.0000 (+0.0901): beats (boot z +3.72) |

## Backtest (costs included)

- n_trades: 1253
- total_return: -0.9623
- sharpe_net: -118.3309
- sharpe_gross: 1.2907
- sortino: -137.4237
- max_drawdown: 0.9623
- hit_rate: 0.0535
- hit_rate_gross: 0.4820
- profit_factor: 0.0371
- avg_hold_bars: 11.3528
- exposure: 0.3293
- turnover: 747.6159
- fees_paid: 7476.2176
- traded_notional: 7476217.6210
- breakeven_cost_bps: 0.2569
- gross_edge_per_trade_bps: -0.1105
- costs_paid: 9719.0829
- gross_pnl: 96.0177
- net_pnl: -9623.0652

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.683, long_above 0.5663, short_below 0.4840, median 0.5280. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.23% | -118.33 | +96.23% | 1253 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.81% .. -95.77%) | -96.27% | -136.55 | | |

The random null enters at the strategy's rate (0.0432 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 56% of its seeds on net return, 100% on net Sharpe and 67% on gross return.
