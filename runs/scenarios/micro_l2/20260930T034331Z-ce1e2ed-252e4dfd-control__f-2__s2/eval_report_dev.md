# Evaluation report - dev split - run `20260930T034331Z-ce1e2ed-252e4dfd-control__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5644 | 0.8325 | 0.8597 |
| accuracy | 0.4819 | 0.4889 | 0.4753 |
| balanced accuracy | 0.4841 | 0.5002 | 0.4923 |
| precision (up) | 0.4684 | 0.4831 | 0.4719 |
| recall / sensitivity (up) | 0.5480 | 0.8328 | 0.8517 |
| specificity (down) | 0.4203 | 0.1677 | 0.1330 |
| F1 (up) | 0.5051 | 0.6115 | 0.6073 |
| MCC | -0.0320 | 0.0006 | -0.0220 |
| AUC | 0.4819 | 0.5175 | 0.4958 |
| Brier | 0.2553 | 0.2575 | 0.2584 |
| ECE (positive class) | 0.0591 | 0.0728 | 0.0849 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 9594 / 10890 / 7894 / 7914 | 15205 / 16270 / 3278 / 3053 | 16001 / 17908 / 2748 / 2787 |
| Gaussian readout: calls up | 0.8806 | 0.9340 | 0.8170 |
| Gaussian readout: MCC | -0.0201 | -0.0072 | 0.0137 |
| Gaussian readout: AUC | 0.4936 | 0.5207 | 0.5146 |
| Gaussian readout: Brier | 0.2507 | 0.2504 | 0.2516 |
| Gaussian readout: ECE | 0.0322 | 0.0281 | 0.0465 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.89 | 564.77 | 811.37 |
| RMSE ($), raw heads | 421.17 | 574.69 | 827.63 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.97 | 378.42 | 553.89 |
| MAE ($), raw heads | 289.26 | 387.11 | 570.05 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0029 | -0.0032 | -0.0093 |
| skill vs zero, raw heads | -0.1069 | -0.0387 | -0.0501 |
| EV, served | -0.0009 | -0.0005 | -0.0020 |
| EV, raw heads | -0.0445 | -0.0086 | -0.0178 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0086 | -0.0082 | 0.0034 |
| corr, Spearman, raw heads | -0.0155 | 0.0108 | 0.0067 |
| mean predicted ($), served | 10.12 | 14.41 | 38.35 |
| mean predicted ($), raw heads | 89.67 | 78.27 | 108.74 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.8739 | 0.9283 | 0.8154 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1128 | 0.1842 | 0.3527 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.97 | 288.70 | 419.05 |
| CRPSS vs constant variance | 0.0173 | 0.0203 | 0.0579 |
| NLL | 7.3866 | 7.7507 | 8.1770 |
| PIT KS | 0.0727 | 0.0951 | 0.0701 |
| var / err^2 Spearman | 0.2008 | 0.1159 | 0.1199 |
| coverage of the 90% interval | 0.9012 | 0.9014 | 0.8640 |
| width of the 90% interval ($) | 1208.84 | 1755.24 | 2406.89 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0082 | [-0.0328, 0.0157] | NOISE |
| h1 | 0.0160 | [-0.0166, 0.0492] | NOISE |
| h2 | 0.0006 | [-0.0301, 0.0316] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.113 / h1 0.184 / h2 0.353) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4586 | 0.7896 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.7808 | 0.9053 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.2862 | 0.7035 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5297 | 0.8378 | 0.7267 | 0.3573 |
| expected if the two signs were independent | 0.5548 | 0.7801 | 0.7266 | 0.3364 |

- P(up) unanimity (all three horizons call the same side): 0.4165

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0320 vs 0.0025 (-0.0345): does not beat, noise (boot z -1.17) | 0.0006 vs 0.0205 (-0.0199): does not beat, noise (boot z -0.78) | -0.0220 vs -0.0138 (-0.0082): does not beat, noise (boot z -0.30) |
| logreg_lags | direction/auc | 0.4819 vs 0.5320 (-0.0502): does not beat, significantly worse (boot z -2.03) | 0.5175 vs 0.5251 (-0.0075): does not beat, noise (boot z -0.71) | 0.4958 vs 0.5095 (-0.0137): does not beat, noise (boot z -0.61) |
| logreg_lags | direction/brier | 0.2553 vs 0.2536 (-0.0017): does not beat, noise (DM z -0.75) | 0.2575 vs 0.2575 (-0.0001): does not beat, noise (DM z -0.06) | 0.2584 vs 0.2702 (+0.0118): beats (DM z +2.30) |
| logreg_lags | direction/ece_pos | 0.0591 vs 0.0663 (+0.0071): beats, noise (boot z +0.44) | 0.0728 vs 0.0812 (+0.0084): beats, noise (boot z +1.61) | 0.0849 vs 0.1230 (+0.0381): beats (boot z +4.89) |
| logreg_lags | direction/acc | 0.4819 vs 0.4836 (-0.0018): does not beat, noise (DM z -0.10) | 0.4889 vs 0.4911 (-0.0022): does not beat, noise (DM z -0.20) | 0.4753 vs 0.4756 (-0.0003): does not beat, noise (DM z -0.02) |
| logreg_lags | direction/bal_acc | 0.4841 vs 0.5004 (-0.0163): does not beat, noise (boot z -1.60) | 0.5002 vs 0.5055 (-0.0053): does not beat, noise (boot z -0.63) | 0.4923 vs 0.4971 (-0.0048): does not beat, noise (boot z -0.64) |
| class_prior | direction/mcc | -0.0320 vs 0.0000 (-0.0320): does not beat, noise (boot z -1.74) | 0.0006 vs 0.0000 (+0.0006): beats, noise (boot z +0.03) | -0.0220 vs 0.0000 (-0.0220): does not beat, noise (boot z -1.29) |
| class_prior | direction/auc | 0.4819 vs 0.5000 (-0.0181): does not beat, noise (boot z -1.61) | 0.5175 vs 0.5000 (+0.0175): beats, noise (boot z +1.33) | 0.4958 vs 0.5000 (-0.0042): does not beat, noise (boot z -0.36) |
| class_prior | direction/brier | 0.2553 vs 0.2532 (-0.0021): does not beat, noise (DM z -1.19) | 0.2575 vs 0.2533 (-0.0042): does not beat, significantly worse (DM z -2.03) | 0.2584 vs 0.2573 (-0.0011): does not beat, noise (DM z -0.78) |
| class_prior | direction/ece_pos | 0.0591 vs 0.0593 (+0.0002): beats, noise (boot z +0.01) | 0.0728 vs 0.0603 (-0.0125): does not beat, significantly worse (boot z -2.17) | 0.0849 vs 0.0886 (+0.0037): beats, noise (boot z +0.51) |
| class_prior | direction/acc | 0.4819 vs 0.4824 (-0.0006): does not beat, noise (DM z -0.03) | 0.4889 vs 0.4829 (+0.0060): beats, noise (DM z +0.51) | 0.4753 vs 0.4763 (-0.0010): does not beat, noise (DM z -0.09) |
| class_prior | direction/bal_acc | 0.4841 vs 0.5000 (-0.0159): does not beat, noise (boot z -1.74) | 0.5002 vs 0.5000 (+0.0002): beats, noise (boot z +0.03) | 0.4923 vs 0.5000 (-0.0077): does not beat, noise (boot z -1.27) |
| zero_delta | delta/rmse | 400.89 vs 400.31 (-0.59, -0.15%): does not beat, noise (DM z -1.18) | 564.77 vs 563.88 (-0.89, -0.16%): does not beat, noise (DM z -1.04) | 811.37 vs 807.63 (-3.74, -0.46%): does not beat, noise (DM z -1.11) |
| zero_delta | delta/mae | 271.97 vs 271.46 (-0.51, -0.19%): does not beat, noise (DM z -1.50) | 378.42 vs 377.88 (-0.54, -0.14%): does not beat, noise (DM z -0.87) | 553.89 vs 550.98 (-2.91, -0.53%): does not beat, noise (DM z -1.17) |
| mean_delta | delta/rmse | 400.89 vs 405.60 (+4.71, +1.16%): beats (DM z +3.25) | 564.77 vs 578.56 (+13.79, +2.38%): beats (DM z +3.09) | 811.37 vs 847.03 (+35.67, +4.21%): beats (DM z +3.06) |
| mean_delta | delta/mae | 271.97 vs 277.50 (+5.53, +1.99%): beats (DM z +4.30) | 378.42 vs 393.77 (+15.35, +3.90%): beats (DM z +4.12) | 553.89 vs 595.01 (+41.13, +6.91%): beats (DM z +4.05) |
| const_var | variance/crps | 202.97 vs 206.55 (+3.58, +1.73%): beats (DM z +4.25) | 288.70 vs 294.69 (+5.99, +2.03%): beats (DM z +2.37) | 419.05 vs 444.83 (+25.77, +5.79%): beats (DM z +3.95) |
| const_var | variance/nll | 7.3866 vs 7.4448 (+0.0583): beats (DM z +3.60) | 7.7507 vs 7.8022 (+0.0515): beats (DM z +2.08) | 8.1770 vs 8.2104 (+0.0334): beats, noise (DM z +1.17) |
| const_var | variance/pit_ks | 0.0727 vs 0.1064 (+0.0337): beats (boot z +7.80) | 0.0951 vs 0.1404 (+0.0452): beats (boot z +7.49) | 0.0701 vs 0.1810 (+0.1110): beats (boot z +27.64) |
| const_var | variance/corr_var_err2_spearman | 0.2008 vs 0.0000 (+0.2008): beats (boot z +10.76) | 0.1159 vs 0.0000 (+0.1159): beats (boot z +4.76) | 0.1199 vs 0.0000 (+0.1199): beats (boot z +4.73) |

## Backtest (costs included)

- n_trades: 1203
- total_return: -0.9557
- sharpe_net: -114.7879
- sharpe_gross: 1.6772
- sortino: -131.7378
- max_drawdown: 0.9557
- hit_rate: 0.0615
- hit_rate_gross: 0.5121
- profit_factor: 0.0270
- avg_hold_bars: 11.9842
- exposure: 0.3337
- turnover: 745.3224
- fees_paid: 7453.3477
- traded_notional: 7453347.6889
- breakeven_cost_bps: 0.3559
- gross_edge_per_trade_bps: 0.1495
- costs_paid: 9689.3520
- gross_pnl: 132.6154
- net_pnl: -9556.7366

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.078, long_above 0.5646, short_below 0.4872, median 0.5253. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -95.57% | -114.79 | +95.57% | 1203 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.20% .. -95.00%) | -95.60% | -130.49 | | |

The random null enters at the strategy's rate (0.0418 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 58% of its seeds on net return, 100% on net Sharpe and 78% on gross return.
