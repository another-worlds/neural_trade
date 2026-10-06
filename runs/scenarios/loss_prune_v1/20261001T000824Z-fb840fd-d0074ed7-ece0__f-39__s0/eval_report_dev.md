# Evaluation report - dev split - run `20261001T000824Z-fb840fd-d0074ed7-ece0__f-39__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 16776 | 18240 | 18977 |
| n_eff of the scored moves (n scored // bars ahead) | 1677 | 1216 | 948 |
| true up-rate | 0.5013 | 0.5054 | 0.5090 |
| calls up (predicted up-rate) | 0.3938 | 0.3832 | 0.6258 |
| accuracy | 0.4784 | 0.5171 | 0.4900 |
| balanced accuracy | 0.4786 | 0.5183 | 0.4878 |
| precision (up) | 0.4742 | 0.5293 | 0.4992 |
| recall / sensitivity (up) | 0.3725 | 0.4013 | 0.6137 |
| specificity (down) | 0.5847 | 0.6353 | 0.3618 |
| F1 (up) | 0.4173 | 0.4565 | 0.5506 |
| MCC | -0.0437 | 0.0377 | -0.0253 |
| AUC | 0.4633 | 0.5272 | 0.4881 |
| Brier | 0.2867 | 0.2529 | 0.2608 |
| ECE (positive class) | 0.1424 | 0.0495 | 0.0817 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0013 | 0.0054 | 0.0090 |
| TP / FP / TN / FN | 3133 / 3474 / 4892 / 5277 | 3699 / 3290 / 5732 / 5519 | 5928 / 5947 / 3371 / 3731 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6543 | 0.6046 | 0.7412 |
| Gaussian readout of the raw heads: MCC | -0.0170 | -0.0151 | -0.0066 |
| Gaussian readout of the raw heads: AUC | 0.4984 | 0.4877 | 0.4961 |
| Gaussian readout of the raw heads: Brier | 0.2562 | 0.2669 | 0.2618 |
| Gaussian readout of the raw heads: ECE | 0.0666 | 0.0953 | 0.0866 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 79.22 | 96.91 | 110.98 |
| RMSE ($), raw heads | 80.64 | 100.41 | 115.28 |
| RMSE ($), zero prediction | 79.22 | 96.91 | 110.98 |
| MAE ($), served | 52.64 | 64.84 | 74.66 |
| MAE ($), raw heads | 53.55 | 66.99 | 77.27 |
| MAE ($), zero prediction | 52.64 | 64.84 | 74.66 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0362 | -0.0735 | -0.0790 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0357 | -0.0733 | -0.0756 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0441 | -0.0587 | -0.0757 |
| corr, Spearman, raw heads | -0.0074 | -0.0210 | -0.0209 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 1.85 | -1.22 | 6.61 |
| mean realised ($) | 0.03 | 0.05 | 0.07 |
| share predicted up, raw heads | 0.6763 | 0.6162 | 0.7508 |
| share realised up | 0.4976 | 0.5011 | 0.5040 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 39.34 | 48.10 | 55.30 |
| CRPSS vs constant variance | 0.0052 | 0.0116 | 0.0146 |
| NLL | 5.7539 | 6.0556 | 6.1294 |
| PIT KS | 0.0544 | 0.0240 | 0.0277 |
| var / err^2 Spearman | 0.1212 | 0.2013 | 0.1986 |
| coverage of the 90% interval | 0.8932 | 0.8948 | 0.8889 |
| width of the 90% interval ($) | 226.36 | 280.01 | 321.07 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0277 | [-0.0590, 0.0001] | NOISE |
| h1 | 0.0320 | [0.0044, 0.0580] | WORKS |
| h2 | 0.0043 | [-0.0159, 0.0233] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6634 | n/a (beta = 0: served delta is 0) | 0.6180 |
| abs(d h1) <= abs(d h2) | 0.6702 | n/a (beta = 0: served delta is 0) | 0.5907 |
| full chain h0 <= h1 <= h2 | 0.3739 | n/a (beta = 0: served delta is 0) | 0.3343 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.3613 | 0.5486 | 0.6561 | 0.0908 |
| expected if the two signs were independent | 0.4601 | 0.4707 | 0.5673 | 0.0961 |

- P(up) unanimity (all three horizons call the same side): 0.1691

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0437 vs 0.0215 (-0.0652): does not beat, significantly worse (boot z -2.21) | 0.0377 vs 0.0397 (-0.0021): does not beat, noise (boot z -0.10) | -0.0253 vs 0.0430 (-0.0684): does not beat, significantly worse (boot z -3.41) |
| logreg_lags | direction/auc | 0.4633 vs 0.5268 (-0.0635): does not beat, significantly worse (boot z -3.33) | 0.5272 vs 0.5395 (-0.0124): does not beat, noise (boot z -1.23) | 0.4881 vs 0.5375 (-0.0495): does not beat, significantly worse (boot z -3.66) |
| logreg_lags | direction/brier | 0.2867 vs 0.2502 (-0.0365): does not beat, significantly worse (DM z -10.25) | 0.2529 vs 0.2495 (-0.0034): does not beat, significantly worse (DM z -2.04) | 0.2608 vs 0.2496 (-0.0112): does not beat, significantly worse (DM z -6.24) |
| logreg_lags | direction/ece_pos | 0.1424 vs 0.0285 (-0.1138): does not beat, significantly worse (boot z -7.42) | 0.0495 vs 0.0266 (-0.0229): does not beat, noise (boot z -1.36) | 0.0817 vs 0.0226 (-0.0591): does not beat, significantly worse (boot z -5.54) |
| logreg_lags | direction/acc | 0.4784 vs 0.5090 (-0.0306): does not beat, noise (DM z -1.94) | 0.5171 vs 0.5195 (-0.0025): does not beat, noise (DM z -0.18) | 0.4900 vs 0.5229 (-0.0329): does not beat, significantly worse (DM z -3.26) |
| logreg_lags | direction/bal_acc | 0.4786 vs 0.5081 (-0.0295): does not beat, significantly worse (boot z -2.30) | 0.5183 vs 0.5165 (+0.0018): beats, noise (boot z +0.18) | 0.4878 vs 0.5181 (-0.0303): does not beat, significantly worse (boot z -3.45) |
| class_prior | direction/mcc | -0.0437 vs 0.0000 (-0.0437): does not beat, significantly worse (boot z -2.53) | 0.0377 vs 0.0000 (+0.0377): beats (boot z +2.00) | -0.0253 vs 0.0000 (-0.0253): does not beat, noise (boot z -1.63) |
| class_prior | direction/auc | 0.4633 vs 0.5000 (-0.0367): does not beat, significantly worse (boot z -3.63) | 0.5272 vs 0.5000 (+0.0272): beats (boot z +2.23) | 0.4881 vs 0.5000 (-0.0119): does not beat, noise (boot z -1.24) |
| class_prior | direction/brier | 0.2867 vs 0.2505 (-0.0362): does not beat, significantly worse (DM z -11.70) | 0.2529 vs 0.2503 (-0.0026): does not beat, noise (DM z -1.39) | 0.2608 vs 0.2501 (-0.0107): does not beat, significantly worse (DM z -6.38) |
| class_prior | direction/ece_pos | 0.1424 vs 0.0224 (-0.1200): does not beat, significantly worse (boot z -8.18) | 0.0495 vs 0.0173 (-0.0321): does not beat, noise (boot z -1.87) | 0.0817 vs 0.0149 (-0.0668): does not beat, significantly worse (boot z -5.82) |
| class_prior | direction/acc | 0.4784 vs 0.5013 (-0.0229): does not beat, noise (DM z -1.55) | 0.5171 vs 0.5054 (+0.0117): beats, noise (DM z +0.68) | 0.4900 vs 0.5090 (-0.0190): does not beat, noise (DM z -1.54) |
| class_prior | direction/bal_acc | 0.4786 vs 0.5000 (-0.0214): does not beat, significantly worse (boot z -2.53) | 0.5183 vs 0.5000 (+0.0183): beats (boot z +2.00) | 0.4878 vs 0.5000 (-0.0122): does not beat, noise (boot z -1.63) |
| zero_delta | delta/rmse | 79.22 vs 79.22 (+0.00, +0.00%): does not beat | 96.91 vs 96.91 (+0.00, +0.00%): does not beat | 110.98 vs 110.98 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 52.64 vs 52.64 (+0.00, +0.00%): does not beat | 64.84 vs 64.84 (+0.00, +0.00%): does not beat | 74.66 vs 74.66 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 79.22 vs 79.26 (+0.04, +0.05%): beats, noise (DM z +0.86) | 96.91 vs 96.99 (+0.07, +0.08%): beats, noise (DM z +0.87) | 110.98 vs 111.09 (+0.11, +0.10%): beats, noise (DM z +0.87) |
| mean_delta | delta/mae | 52.64 vs 52.68 (+0.05, +0.09%): beats, noise (DM z +1.12) | 64.84 vs 64.90 (+0.06, +0.09%): beats, noise (DM z +0.84) | 74.66 vs 74.72 (+0.06, +0.08%): beats, noise (DM z +0.57) |
| const_var | variance/crps | 39.34 vs 39.54 (+0.21, +0.52%): beats (DM z +4.47) | 48.10 vs 48.66 (+0.56, +1.16%): beats (DM z +6.49) | 55.30 vs 56.12 (+0.82, +1.46%): beats (DM z +5.58) |
| const_var | variance/nll | 5.7539 vs 5.8104 (+0.0565): beats (DM z +3.27) | 6.0556 vs 6.0150 (-0.0405): does not beat, noise (DM z -1.31) | 6.1294 vs 6.1490 (+0.0196): beats, noise (DM z +1.03) |
| const_var | variance/pit_ks | 0.0544 vs 0.0591 (+0.0047): beats, noise (boot z +0.66) | 0.0240 vs 0.0617 (+0.0377): beats (boot z +3.53) | 0.0277 vs 0.0671 (+0.0394): beats (boot z +3.48) |
| const_var | variance/corr_var_err2_spearman | 0.1212 vs 0.0000 (+0.1212): beats (boot z +6.02) | 0.2013 vs 0.0000 (+0.2013): beats (boot z +9.22) | 0.1986 vs 0.0000 (+0.1986): beats (boot z +9.05) |

## Backtest (costs included)

- n_trades: 1065
- total_return: 0.0861
- sharpe_net: 5.6911
- sharpe_gross: 5.6911
- sortino: 8.7492
- max_drawdown: 0.0303
- hit_rate: 0.4056
- hit_rate_gross: 0.4056
- profit_factor: 1.1188
- avg_hold_bars: 8.9568
- exposure: 0.3911
- turnover: 2234.8967
- fees_paid: 0.0000
- traded_notional: 22349339.3034
- breakeven_cost_bps: 0.7705
- gross_edge_per_trade_bps: 0.7980
- costs_paid: 0.0000
- gross_pnl: 861.0513
- net_pnl: 861.0513

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -39 (TimeSeriesSplit fold 2, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2023-12-08T14:48:00 .. 2023-12-25T13:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7093, long_above 0.5582, short_below 0.4339, median 0.4832. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +8.61% | +5.69 | +3.03% | 1065 |
| buy and hold | +0.10% | +0.26 | +10.11% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -9.41% .. +10.71%) | +0.12% | +0.12 | | |

The random null enters at the strategy's rate (0.0717 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 91% of its seeds on net return, 87% on net Sharpe and 91% on gross return.
