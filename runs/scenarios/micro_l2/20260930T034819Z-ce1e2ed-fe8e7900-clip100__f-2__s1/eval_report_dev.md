# Evaluation report - dev split - run `20260930T034819Z-ce1e2ed-fe8e7900-clip100__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.4049 | 0.5872 | 0.9170 |
| accuracy | 0.4877 | 0.5042 | 0.4778 |
| balanced accuracy | 0.4844 | 0.5072 | 0.4975 |
| precision (up) | 0.4632 | 0.4891 | 0.4750 |
| recall / sensitivity (up) | 0.3887 | 0.5946 | 0.9144 |
| specificity (down) | 0.5800 | 0.4198 | 0.0807 |
| F1 (up) | 0.4227 | 0.5367 | 0.6252 |
| MCC | -0.0318 | 0.0146 | -0.0089 |
| AUC | 0.4746 | 0.5139 | 0.5099 |
| Brier | 0.2550 | 0.2526 | 0.2589 |
| ECE (positive class) | 0.0541 | 0.0379 | 0.0889 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 6806 / 7889 / 10895 / 10702 | 10856 / 11342 / 8206 / 7402 | 17180 / 18990 / 1666 / 1608 |
| Gaussian readout: calls up | 0.9878 | 0.9778 | 1.0000 |
| Gaussian readout: MCC | 0.0014 | 0.0029 | 0.0000 |
| Gaussian readout: AUC | 0.5143 | 0.5195 | 0.5079 |
| Gaussian readout: Brier | 0.2504 | 0.2509 | 0.2515 |
| Gaussian readout: ECE | 0.0279 | 0.0360 | 0.0462 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.70 | 565.25 | 810.09 |
| RMSE ($), raw heads | 416.42 | 579.74 | 824.41 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.80 | 378.78 | 553.30 |
| MAE ($), raw heads | 286.68 | 391.69 | 568.87 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0020 | -0.0048 | -0.0061 |
| skill vs zero, raw heads | -0.0821 | -0.0570 | -0.0420 |
| EV, served | -0.0006 | -0.0008 | 0.0006 |
| EV, raw heads | -0.0155 | -0.0116 | -0.0017 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0326 | -0.0104 | 0.0245 |
| corr, Spearman, raw heads | 0.0127 | 0.0205 | 0.0024 |
| mean predicted ($), served | 7.56 | 20.07 | 35.89 |
| mean predicted ($), raw heads | 92.93 | 100.19 | 124.90 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9888 | 0.9778 | 1.0000 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0813 | 0.2004 | 0.2873 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.64 | 285.65 | 419.20 |
| CRPSS vs constant variance | 0.0189 | 0.0307 | 0.0576 |
| NLL | 7.4333 | 7.7982 | 8.1575 |
| PIT KS | 0.0489 | 0.0665 | 0.0779 |
| var / err^2 Spearman | 0.0070 | -0.0486 | 0.0919 |
| coverage of the 90% interval | 0.9014 | 0.9005 | 0.8640 |
| width of the 90% interval ($) | 1207.93 | 1749.38 | 2422.05 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0267 | [-0.0540, -0.0010] | INVERTED |
| h1 | 0.0058 | [-0.0218, 0.0334] | NOISE |
| h2 | 0.0094 | [-0.0160, 0.0331] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.081 / h1 0.200 / h2 0.287) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5865 | 0.9689 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.5392 | 0.9300 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.1869 | 0.8989 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4125 | 0.5696 | 0.9178 | 0.2322 |
| expected if the two signs were independent | 0.4167 | 0.5716 | 0.9178 | 0.2317 |

- P(up) unanimity (all three horizons call the same side): 0.2608

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0318 vs 0.0025 (-0.0343): does not beat, noise (boot z -1.13) | 0.0146 vs 0.0205 (-0.0059): does not beat, noise (boot z -0.19) | -0.0089 vs -0.0138 (+0.0049): beats, noise (boot z +0.22) |
| logreg_lags | direction/auc | 0.4746 vs 0.5320 (-0.0574): does not beat, significantly worse (boot z -2.24) | 0.5139 vs 0.5251 (-0.0112): does not beat, noise (boot z -0.59) | 0.5099 vs 0.5095 (+0.0004): beats, noise (boot z +0.02) |
| logreg_lags | direction/brier | 0.2550 vs 0.2536 (-0.0014): does not beat, noise (DM z -0.49) | 0.2526 vs 0.2575 (+0.0049): beats, noise (DM z +1.57) | 0.2589 vs 0.2702 (+0.0113): beats (DM z +2.82) |
| logreg_lags | direction/ece_pos | 0.0541 vs 0.0663 (+0.0122): beats, noise (boot z +0.58) | 0.0379 vs 0.0812 (+0.0433): beats (boot z +4.05) | 0.0889 vs 0.1230 (+0.0341): beats (boot z +7.38) |
| logreg_lags | direction/acc | 0.4877 vs 0.4836 (+0.0041): beats, noise (DM z +0.18) | 0.5042 vs 0.4911 (+0.0131): beats, noise (DM z +0.61) | 0.4778 vs 0.4756 (+0.0022): beats, noise (DM z +0.36) |
| logreg_lags | direction/bal_acc | 0.4844 vs 0.5004 (-0.0160): does not beat, noise (boot z -1.48) | 0.5072 vs 0.5055 (+0.0017): beats, noise (boot z +0.14) | 0.4975 vs 0.4971 (+0.0004): beats, noise (boot z +0.08) |
| class_prior | direction/mcc | -0.0318 vs 0.0000 (-0.0318): does not beat, noise (boot z -1.57) | 0.0146 vs 0.0000 (+0.0146): beats, noise (boot z +0.77) | -0.0089 vs 0.0000 (-0.0089): does not beat, noise (boot z -0.65) |
| class_prior | direction/auc | 0.4746 vs 0.5000 (-0.0254): does not beat, noise (boot z -1.91) | 0.5139 vs 0.5000 (+0.0139): beats, noise (boot z +1.11) | 0.5099 vs 0.5000 (+0.0099): beats, noise (boot z +1.29) |
| class_prior | direction/brier | 0.2550 vs 0.2532 (-0.0018): does not beat, noise (DM z -0.78) | 0.2526 vs 0.2533 (+0.0008): beats, noise (DM z +0.40) | 0.2589 vs 0.2573 (-0.0016): does not beat, noise (DM z -1.60) |
| class_prior | direction/ece_pos | 0.0541 vs 0.0593 (+0.0053): beats, noise (boot z +0.25) | 0.0379 vs 0.0603 (+0.0224): beats (boot z +2.12) | 0.0889 vs 0.0886 (-0.0003): does not beat, noise (boot z -0.07) |
| class_prior | direction/acc | 0.4877 vs 0.4824 (+0.0053): beats, noise (DM z +0.24) | 0.5042 vs 0.4829 (+0.0213): beats, noise (DM z +0.96) | 0.4778 vs 0.4763 (+0.0015): beats, noise (DM z +0.23) |
| class_prior | direction/bal_acc | 0.4844 vs 0.5000 (-0.0156): does not beat, noise (boot z -1.57) | 0.5072 vs 0.5000 (+0.0072): beats, noise (boot z +0.77) | 0.4975 vs 0.5000 (-0.0025): does not beat, noise (boot z -0.65) |
| zero_delta | delta/rmse | 400.70 vs 400.31 (-0.39, -0.10%): does not beat, noise (DM z -1.38) | 565.25 vs 563.88 (-1.36, -0.24%): does not beat, noise (DM z -1.23) | 810.09 vs 807.63 (-2.46, -0.30%): does not beat, noise (DM z -0.91) |
| zero_delta | delta/mae | 271.80 vs 271.46 (-0.34, -0.13%): does not beat, noise (DM z -1.54) | 378.78 vs 377.88 (-0.90, -0.24%): does not beat, noise (DM z -1.06) | 553.30 vs 550.98 (-2.32, -0.42%): does not beat, noise (DM z -1.12) |
| mean_delta | delta/rmse | 400.70 vs 405.60 (+4.90, +1.21%): beats (DM z +3.00) | 565.25 vs 578.56 (+13.32, +2.30%): beats (DM z +3.15) | 810.09 vs 847.03 (+36.95, +4.36%): beats (DM z +3.06) |
| mean_delta | delta/mae | 271.80 vs 277.50 (+5.70, +2.05%): beats (DM z +4.30) | 378.78 vs 393.77 (+14.99, +3.81%): beats (DM z +4.29) | 553.30 vs 595.01 (+41.71, +7.01%): beats (DM z +4.13) |
| const_var | variance/crps | 202.64 vs 206.55 (+3.91, +1.89%): beats (DM z +4.82) | 285.65 vs 294.69 (+9.03, +3.07%): beats (DM z +4.14) | 419.20 vs 444.83 (+25.62, +5.76%): beats (DM z +3.90) |
| const_var | variance/nll | 7.4333 vs 7.4448 (+0.0115): beats, noise (DM z +1.50) | 7.7982 vs 7.8022 (+0.0040): beats, noise (DM z +0.24) | 8.1575 vs 8.2104 (+0.0529): beats (DM z +2.28) |
| const_var | variance/pit_ks | 0.0489 vs 0.1064 (+0.0575): beats (boot z +15.37) | 0.0665 vs 0.1404 (+0.0739): beats (boot z +19.57) | 0.0779 vs 0.1810 (+0.1032): beats (boot z +29.48) |
| const_var | variance/corr_var_err2_spearman | 0.0070 vs 0.0000 (+0.0070): beats, noise (boot z +0.34) | -0.0486 vs 0.0000 (-0.0486): does not beat, noise (boot z -1.95) | 0.0919 vs 0.0000 (+0.0919): beats (boot z +3.72) |

## Backtest (costs included)

- n_trades: 1358
- total_return: -0.9713
- sharpe_net: -125.3887
- sharpe_gross: 0.1952
- sortino: -145.0774
- max_drawdown: 0.9713
- hit_rate: 0.0574
- hit_rate_gross: 0.4882
- profit_factor: 0.0277
- avg_hold_bars: 11.0839
- exposure: 0.3484
- turnover: 748.0020
- fees_paid: 7480.1934
- traded_notional: 7480193.4358
- breakeven_cost_bps: 0.0301
- gross_edge_per_trade_bps: -0.0959
- costs_paid: 9724.2515
- gross_pnl: 11.2540
- net_pnl: -9712.9974

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7729, long_above 0.5526, short_below 0.4680, median 0.5073. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.13% | -125.39 | +97.13% | 1358 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.50% .. -96.69%) | -97.09% | -141.72 | | |

The random null enters at the strategy's rate (0.0482 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 44% of its seeds on net return, 100% on net Sharpe and 48% on gross return.
