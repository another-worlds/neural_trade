# Evaluation report - dev split - run `20261001T013328Z-fb840fd-8d0838b5-ece0_vol0__f-37__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 17330 | 18405 | 19141 |
| n_eff of the scored moves (n scored // bars ahead) | 1733 | 1227 | 957 |
| true up-rate | 0.4990 | 0.4981 | 0.4981 |
| calls up (predicted up-rate) | 0.5614 | 0.6936 | 0.6075 |
| accuracy | 0.5069 | 0.5108 | 0.5039 |
| balanced accuracy | 0.5070 | 0.5116 | 0.5043 |
| precision (up) | 0.5053 | 0.5064 | 0.5017 |
| recall / sensitivity (up) | 0.5685 | 0.7052 | 0.6119 |
| specificity (down) | 0.4456 | 0.3179 | 0.3968 |
| F1 (up) | 0.5350 | 0.5895 | 0.5513 |
| MCC | 0.0142 | 0.0251 | 0.0089 |
| AUC | 0.5057 | 0.5197 | 0.5098 |
| Brier | 0.2594 | 0.2525 | 0.2553 |
| ECE (positive class) | 0.0629 | 0.0352 | 0.0520 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0010 | 0.0019 | 0.0019 |
| TP / FP / TN / FN | 4916 / 4813 / 3869 / 3732 | 6465 / 6301 / 2937 / 2702 | 5834 / 5794 / 3812 / 3701 |
| Gaussian readout: calls up | 0.8447 | 0.7713 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0148 | 0.0159 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5119 | 0.5192 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2503 | 0.2499 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0177 | 0.0059 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.8447 | 0.7713 | 0.5820 |
| Gaussian readout of the raw heads: MCC | 0.0148 | 0.0159 | -0.0002 |
| Gaussian readout of the raw heads: AUC | 0.5119 | 0.5192 | 0.5084 |
| Gaussian readout of the raw heads: Brier | 0.2521 | 0.2521 | 0.2540 |
| Gaussian readout of the raw heads: ECE | 0.0391 | 0.0390 | 0.0516 |

beta = 0 for h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 90.26 | 109.65 | 126.22 |
| RMSE ($), raw heads | 90.65 | 111.21 | 129.38 |
| RMSE ($), zero prediction | 90.11 | 109.57 | 126.22 |
| MAE ($), served | 59.37 | 71.97 | 82.66 |
| MAE ($), raw heads | 59.54 | 72.44 | 83.62 |
| MAE ($), zero prediction | 59.37 | 71.99 | 82.66 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0033 | -0.0016 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0122 | -0.0302 | -0.0506 |
| EV, served | -0.0020 | -0.0013 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0075 | -0.0260 | -0.0480 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0153 | -0.0239 | -0.0378 |
| corr, Spearman, raw heads | 0.0214 | 0.0154 | 0.0059 |
| mean predicted ($), served | 2.18 | 0.76 | 0.00 |
| mean predicted ($), raw heads | 5.00 | 5.44 | 4.34 |
| mean realised ($) | -1.30 | -1.98 | -2.67 |
| share predicted up, raw heads | 0.8616 | 0.7854 | 0.5742 |
| share realised up | 0.5005 | 0.4945 | 0.4976 |
| shrink beta (served = beta x raw, fit on cal) | 0.4368 | 0.1390 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 43.40 | 52.66 | 60.55 |
| CRPSS vs constant variance | 0.0371 | 0.0389 | 0.0378 |
| NLL | 5.7068 | 5.8951 | 6.0395 |
| PIT KS | 0.0388 | 0.0307 | 0.0276 |
| var / err^2 Spearman | 0.3993 | 0.3950 | 0.3950 |
| coverage of the 90% interval | 0.9066 | 0.9061 | 0.9070 |
| width of the 90% interval ($) | 258.14 | 318.52 | 362.88 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0030 | [-0.0243, 0.0225] | NOISE |
| h1 | 0.0046 | [-0.0208, 0.0302] | NOISE |
| h2 | 0.0162 | [-0.0067, 0.0383] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.437 / h1 0.139 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5763 | 0.1369 | 0.6066 |
| abs(d h1) <= abs(d h2) | 0.6114 | n/a (beta = 0: served delta is 0) | 0.5887 |
| full chain h0 <= h1 <= h2 | 0.3228 | n/a (beta = 0: served delta is 0) | 0.3212 |

beta = 0 for h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6244 | 0.7345 | 0.6649 | 0.3378 |
| expected if the two signs were independent | 0.5479 | 0.6110 | 0.5164 | 0.2107 |

- P(up) unanimity (all three horizons call the same side): 0.4584

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0142 vs 0.0245 (-0.0103): does not beat, noise (boot z -0.48) | 0.0251 vs 0.0175 (+0.0076): beats, noise (boot z +0.55) | 0.0089 vs 0.0080 (+0.0009): beats, noise (boot z +0.04) |
| logreg_lags | direction/auc | 0.5057 vs 0.5197 (-0.0140): does not beat, noise (boot z -1.01) | 0.5197 vs 0.5155 (+0.0042): beats, noise (boot z +0.52) | 0.5098 vs 0.5134 (-0.0036): does not beat, noise (boot z -0.30) |
| logreg_lags | direction/brier | 0.2594 vs 0.2505 (-0.0089): does not beat, significantly worse (DM z -4.43) | 0.2525 vs 0.2515 (-0.0010): does not beat, noise (DM z -1.48) | 0.2553 vs 0.2515 (-0.0038): does not beat, significantly worse (DM z -3.17) |
| logreg_lags | direction/ece_pos | 0.0629 vs 0.0173 (-0.0457): does not beat, significantly worse (boot z -4.15) | 0.0352 vs 0.0253 (-0.0099): does not beat, significantly worse (boot z -2.15) | 0.0520 vs 0.0290 (-0.0230): does not beat, significantly worse (boot z -2.44) |
| logreg_lags | direction/acc | 0.5069 vs 0.5106 (-0.0036): does not beat, noise (DM z -0.34) | 0.5108 vs 0.5070 (+0.0038): beats, noise (DM z +0.60) | 0.5039 vs 0.5028 (+0.0011): beats, noise (DM z +0.11) |
| logreg_lags | direction/bal_acc | 0.5070 vs 0.5110 (-0.0039): does not beat, noise (boot z -0.39) | 0.5116 vs 0.5079 (+0.0037): beats, noise (boot z +0.59) | 0.5043 vs 0.5036 (+0.0007): beats, noise (boot z +0.08) |
| class_prior | direction/mcc | 0.0142 vs 0.0000 (+0.0142): beats, noise (boot z +0.95) | 0.0251 vs 0.0000 (+0.0251): beats, noise (boot z +1.58) | 0.0089 vs 0.0000 (+0.0089): beats, noise (boot z +0.61) |
| class_prior | direction/auc | 0.5057 vs 0.5000 (+0.0057): beats, noise (boot z +0.59) | 0.5197 vs 0.5000 (+0.0197): beats, noise (boot z +1.87) | 0.5098 vs 0.5000 (+0.0098): beats, noise (boot z +1.00) |
| class_prior | direction/brier | 0.2594 vs 0.2502 (-0.0092): does not beat, significantly worse (DM z -5.07) | 0.2525 vs 0.2503 (-0.0022): does not beat, noise (DM z -1.87) | 0.2553 vs 0.2502 (-0.0051): does not beat, significantly worse (DM z -3.66) |
| class_prior | direction/ece_pos | 0.0629 vs 0.0133 (-0.0497): does not beat, significantly worse (boot z -4.30) | 0.0352 vs 0.0162 (-0.0191): does not beat, significantly worse (boot z -3.09) | 0.0520 vs 0.0157 (-0.0363): does not beat, significantly worse (boot z -3.30) |
| class_prior | direction/acc | 0.5069 vs 0.4990 (+0.0079): beats, noise (DM z +0.70) | 0.5108 vs 0.4981 (+0.0128): beats, noise (DM z +1.20) | 0.5039 vs 0.4981 (+0.0058): beats, noise (DM z +0.45) |
| class_prior | direction/bal_acc | 0.5070 vs 0.5000 (+0.0070): beats, noise (boot z +0.95) | 0.5116 vs 0.5000 (+0.0116): beats, noise (boot z +1.58) | 0.5043 vs 0.5000 (+0.0043): beats, noise (boot z +0.61) |
| zero_delta | delta/rmse | 90.26 vs 90.11 (-0.15, -0.16%): does not beat, noise (DM z -1.42) | 109.65 vs 109.57 (-0.09, -0.08%): does not beat, noise (DM z -0.69) | 126.22 vs 126.22 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 59.37 vs 59.37 (+0.00, +0.00%): beats, noise (DM z +0.04) | 71.97 vs 71.99 (+0.02, +0.03%): beats, noise (DM z +0.44) | 82.66 vs 82.66 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 90.26 vs 90.14 (-0.11, -0.13%): does not beat, noise (DM z -1.26) | 109.65 vs 109.62 (-0.03, -0.03%): does not beat, noise (DM z -0.26) | 126.22 vs 126.32 (+0.09, +0.07%): beats, noise (DM z +1.21) |
| mean_delta | delta/mae | 59.37 vs 59.38 (+0.01, +0.01%): beats, noise (DM z +0.17) | 71.97 vs 72.03 (+0.06, +0.08%): beats, noise (DM z +1.21) | 82.66 vs 82.70 (+0.04, +0.05%): beats, noise (DM z +0.64) |
| const_var | variance/crps | 43.40 vs 45.08 (+1.67, +3.71%): beats (DM z +11.33) | 52.66 vs 54.79 (+2.13, +3.89%): beats (DM z +9.50) | 60.55 vs 62.93 (+2.38, +3.78%): beats (DM z +8.68) |
| const_var | variance/nll | 5.7068 vs 5.9949 (+0.2881): beats (DM z +7.13) | 5.8951 vs 6.1909 (+0.2957): beats (DM z +5.63) | 6.0395 vs 6.3349 (+0.2954): beats (DM z +4.89) |
| const_var | variance/pit_ks | 0.0388 vs 0.0468 (+0.0079): beats, noise (boot z +1.90) | 0.0307 vs 0.0515 (+0.0208): beats (boot z +3.47) | 0.0276 vs 0.0525 (+0.0249): beats (boot z +2.88) |
| const_var | variance/corr_var_err2_spearman | 0.3993 vs 0.0000 (+0.3993): beats (boot z +20.23) | 0.3950 vs 0.0000 (+0.3950): beats (boot z +18.57) | 0.3950 vs 0.0000 (+0.3950): beats (boot z +17.25) |

## Backtest (costs included)

- n_trades: 798
- total_return: -0.0016
- sharpe_net: 0.0841
- sharpe_gross: 0.0841
- sortino: 0.1211
- max_drawdown: 0.0693
- hit_rate: 0.5175
- hit_rate_gross: 0.5175
- profit_factor: 0.9977
- avg_hold_bars: 8.8759
- exposure: 0.2904
- turnover: 1605.6275
- fees_paid: 0.0000
- traded_notional: 16056721.7881
- breakeven_cost_bps: -0.0198
- gross_edge_per_trade_bps: 0.0110
- costs_paid: 0.0000
- gross_pnl: -15.9110
- net_pnl: -15.9110

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -37 (TimeSeriesSplit fold 4, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-01-11T11:48:00 .. 2024-01-28T10:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.145, long_above 0.5861, short_below 0.4443, median 0.5122. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -0.16% | +0.08 | +6.93% | 798 |
| buy and hold | -7.01% | -2.90 | +21.43% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -7.94% .. +8.47%) | -0.46% | -0.36 | | |

The random null enters at the strategy's rate (0.0461 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 53% of its seeds on net return, 54% on net Sharpe and 53% on gross return.
