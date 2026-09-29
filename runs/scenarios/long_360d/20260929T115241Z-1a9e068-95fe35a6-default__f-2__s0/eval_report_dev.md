# Evaluation report - dev split - run `20260929T115241Z-1a9e068-95fe35a6-default__f-2__s0`

n = 46544 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 28800 | 32081 | 33877 |
| n_eff of the scored moves (n scored // bars ahead) | 2880 | 2138 | 1693 |
| true up-rate | 0.4906 | 0.4906 | 0.4890 |
| calls up (predicted up-rate) | 0.5189 | 0.3795 | 0.6876 |
| accuracy | 0.5172 | 0.5092 | 0.5124 |
| balanced accuracy | 0.5176 | 0.5070 | 0.5165 |
| precision (up) | 0.5075 | 0.4998 | 0.5010 |
| recall / sensitivity (up) | 0.5368 | 0.3866 | 0.7045 |
| specificity (down) | 0.4984 | 0.6273 | 0.3286 |
| F1 (up) | 0.5217 | 0.4360 | 0.5856 |
| MCC | 0.0352 | 0.0144 | 0.0357 |
| AUC | 0.5220 | 0.5054 | 0.5279 |
| Brier | 0.2499 | 0.2497 | 0.2499 |
| ECE (positive class) | 0.0131 | 0.0070 | 0.0229 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0094 | 0.0094 | 0.0110 |
| TP / FP / TN / FN | 7584 / 7360 / 7312 / 6544 | 6085 / 6090 / 10252 / 9654 | 11670 / 11624 / 5688 / 4895 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.5521 | 0.4904 | 0.5734 |
| Gaussian readout of the raw heads: MCC | 0.0286 | 0.0116 | -0.0057 |
| Gaussian readout of the raw heads: AUC | 0.5149 | 0.5083 | 0.5015 |
| Gaussian readout of the raw heads: Brier | 0.2498 | 0.2497 | 0.2504 |
| Gaussian readout of the raw heads: ECE | 0.0111 | 0.0101 | 0.0240 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 164.21 | 198.38 | 228.36 |
| RMSE ($), raw heads | 164.22 | 210.14 | 228.35 |
| RMSE ($), zero prediction | 164.21 | 198.38 | 228.36 |
| MAE ($), served | 112.64 | 138.04 | 158.99 |
| MAE ($), raw heads | 112.61 | 139.71 | 159.09 |
| MAE ($), zero prediction | 112.64 | 138.04 | 158.99 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0001 | -0.1221 | 0.0001 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0000 | -0.1218 | 0.0003 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0259 | 0.0549 | 0.0382 |
| corr, Spearman, raw heads | 0.0197 | 0.0118 | -0.0023 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 0.54 | 2.20 | 1.30 |
| mean realised ($) | -1.18 | -1.76 | -2.32 |
| share predicted up, raw heads | 0.5401 | 0.4883 | 0.5786 |
| share realised up | 0.4897 | 0.4876 | 0.4888 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 81.99 | 100.63 | 115.29 |
| CRPSS vs constant variance | 0.0467 | 0.0398 | 0.0454 |
| NLL | 6.3950 | 6.5960 | 6.7360 |
| PIT KS | 0.0375 | 0.0419 | 0.0353 |
| var / err^2 Spearman | 0.3404 | 0.3353 | 0.3370 |
| coverage of the 90% interval | 0.9070 | 0.9077 | 0.9079 |
| width of the 90% interval ($) | 518.40 | 634.62 | 729.08 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0069 | [-0.0099, 0.0242] | NOISE |
| h1 | -0.0016 | [-0.0185, 0.0152] | NOISE |
| h2 | 0.0203 | [0.0012, 0.0406] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6530 | n/a (beta = 0: served delta is 0) | 0.6180 |
| abs(d h1) <= abs(d h2) | 0.6631 | n/a (beta = 0: served delta is 0) | 0.5923 |
| full chain h0 <= h1 <= h2 | 0.3784 | n/a (beta = 0: served delta is 0) | 0.3360 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5658 | 0.6377 | 0.4709 | 0.1552 |
| expected if the two signs were independent | 0.5013 | 0.5026 | 0.5285 | 0.1253 |

- P(up) unanimity (all three horizons call the same side): 0.3246

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0352 vs 0.0344 (+0.0008): beats, noise (boot z +0.05) | 0.0144 vs 0.0378 (-0.0234): does not beat, noise (boot z -1.60) | 0.0357 vs 0.0411 (-0.0055): does not beat, noise (boot z -0.43) |
| logreg_lags | direction/auc | 0.5220 vs 0.5243 (-0.0023): does not beat, noise (boot z -0.24) | 0.5054 vs 0.5272 (-0.0218): does not beat, noise (boot z -1.93) | 0.5279 vs 0.5314 (-0.0035): does not beat, noise (boot z -0.46) |
| logreg_lags | direction/brier | 0.2499 vs 0.2497 (-0.0002): does not beat, noise (DM z -0.44) | 0.2497 vs 0.2497 (-0.0000): does not beat, noise (DM z -0.00) | 0.2499 vs 0.2497 (-0.0003): does not beat, noise (DM z -0.70) |
| logreg_lags | direction/ece_pos | 0.0131 vs 0.0112 (-0.0019): does not beat, noise (boot z -0.40) | 0.0070 vs 0.0116 (+0.0046): beats, noise (boot z +0.95) | 0.0229 vs 0.0137 (-0.0092): does not beat, noise (boot z -1.35) |
| logreg_lags | direction/acc | 0.5172 vs 0.5165 (+0.0007): beats, noise (DM z +0.10) | 0.5092 vs 0.5178 (-0.0085): does not beat, noise (DM z -1.08) | 0.5124 vs 0.5191 (-0.0067): does not beat, noise (DM z -1.01) |
| logreg_lags | direction/bal_acc | 0.5176 vs 0.5171 (+0.0004): beats, noise (boot z +0.06) | 0.5070 vs 0.5188 (-0.0118): does not beat, noise (boot z -1.63) | 0.5165 vs 0.5204 (-0.0039): does not beat, noise (boot z -0.62) |
| class_prior | direction/mcc | 0.0352 vs 0.0000 (+0.0352): beats (boot z +3.04) | 0.0144 vs 0.0000 (+0.0144): beats, noise (boot z +1.24) | 0.0357 vs 0.0000 (+0.0357): beats (boot z +2.74) |
| class_prior | direction/auc | 0.5220 vs 0.5000 (+0.0220): beats (boot z +2.95) | 0.5054 vs 0.5000 (+0.0054): beats, noise (boot z +0.68) | 0.5279 vs 0.5000 (+0.0279): beats (boot z +3.20) |
| class_prior | direction/brier | 0.2499 vs 0.2501 (+0.0002): beats, noise (DM z +0.40) | 0.2497 vs 0.2501 (+0.0004): beats, noise (DM z +1.48) | 0.2499 vs 0.2501 (+0.0002): beats, noise (DM z +0.49) |
| class_prior | direction/ece_pos | 0.0131 vs 0.0123 (-0.0009): does not beat, noise (boot z -0.29) | 0.0070 vs 0.0134 (+0.0064): beats, noise (boot z +1.32) | 0.0229 vs 0.0164 (-0.0066): does not beat, significantly worse (boot z -3.49) |
| class_prior | direction/acc | 0.5172 vs 0.4906 (+0.0267): beats (DM z +2.89) | 0.5092 vs 0.4906 (+0.0186): beats, noise (DM z +1.48) | 0.5124 vs 0.4890 (+0.0234): beats (DM z +2.73) |
| class_prior | direction/bal_acc | 0.5176 vs 0.5000 (+0.0176): beats (boot z +3.04) | 0.5070 vs 0.5000 (+0.0070): beats, noise (boot z +1.24) | 0.5165 vs 0.5000 (+0.0165): beats (boot z +2.75) |
| zero_delta | delta/rmse | 164.21 vs 164.21 (+0.00, +0.00%): does not beat | 198.38 vs 198.38 (+0.00, +0.00%): does not beat | 228.36 vs 228.36 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 112.64 vs 112.64 (+0.00, +0.00%): does not beat | 138.04 vs 138.04 (+0.00, +0.00%): does not beat | 158.99 vs 158.99 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 164.21 vs 164.22 (+0.01, +0.00%): beats, noise (DM z +0.73) | 198.38 vs 198.40 (+0.01, +0.01%): beats, noise (DM z +0.73) | 228.36 vs 228.38 (+0.02, +0.01%): beats, noise (DM z +0.72) |
| mean_delta | delta/mae | 112.64 vs 112.65 (+0.02, +0.01%): beats, noise (DM z +1.78) | 138.04 vs 138.07 (+0.03, +0.02%): beats, noise (DM z +1.83) | 158.99 vs 159.03 (+0.04, +0.02%): beats, noise (DM z +1.51) |
| const_var | variance/crps | 81.99 vs 86.01 (+4.02, +4.67%): beats (DM z +21.00) | 100.63 vs 104.80 (+4.17, +3.98%): beats (DM z +13.93) | 115.29 vs 120.77 (+5.48, +4.54%): beats (DM z +15.52) |
| const_var | variance/nll | 6.3950 vs 6.5314 (+0.1363): beats (DM z +8.98) | 6.5960 vs 6.7225 (+0.1265): beats (DM z +7.99) | 6.7360 vs 6.8636 (+0.1276): beats (DM z +8.05) |
| const_var | variance/pit_ks | 0.0375 vs 0.0915 (+0.0540): beats (boot z +14.76) | 0.0419 vs 0.0894 (+0.0475): beats (boot z +11.87) | 0.0353 vs 0.0910 (+0.0557): beats (boot z +14.21) |
| const_var | variance/corr_var_err2_spearman | 0.3404 vs 0.0000 (+0.3404): beats (boot z +25.23) | 0.3353 vs 0.0000 (+0.3353): beats (boot z +22.99) | 0.3370 vs 0.0000 (+0.3370): beats (boot z +22.13) |

## Backtest (costs included)

- n_trades: 1838
- total_return: -0.9902
- sharpe_net: -153.5950
- sharpe_gross: 2.7188
- sortino: -166.7578
- max_drawdown: 0.9902
- hit_rate: 0.0321
- hit_rate_gross: 0.5751
- profit_factor: 0.0116
- avg_hold_bars: 8.3232
- exposure: 0.3287
- turnover: 773.3212
- fees_paid: 7733.1231
- traded_notional: 7733123.0503
- breakeven_cost_bps: 0.3907
- gross_edge_per_trade_bps: 0.8788
- costs_paid: 10053.0600
- gross_pnl: 151.0625
- net_pnl: -9901.9975

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 13, 13 usable folds); this report scores the fold's out-of-sample block: 46544 sequences, 2025-07-27T08:10:00 .. 2025-08-28T15:53:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4478, long_above 0.5261, short_below 0.4754, median 0.5027. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.02% | -153.60 | +99.02% | 1838 |
| buy and hold | -5.10% | -1.62 | +12.57% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.31% .. -99.10%) | -99.20% | -172.31 | | |

The random null enters at the strategy's rate (0.0588 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 100% of its seeds on net return, 100% on net Sharpe and 79% on gross return.
