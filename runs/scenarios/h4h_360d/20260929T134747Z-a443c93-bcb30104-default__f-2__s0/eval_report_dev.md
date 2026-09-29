# Evaluation report - dev split - run `20260929T134747Z-a443c93-bcb30104-default__f-2__s0`

n = 46544 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 41827 | 42471 | 43236 |
| n_eff of the scored moves (n scored // bars ahead) | 261 | 176 | 135 |
| true up-rate | 0.4888 | 0.4851 | 0.4833 |
| calls up (predicted up-rate) | 0.9350 | 0.9974 | 0.6633 |
| accuracy | 0.4884 | 0.4849 | 0.4956 |
| balanced accuracy | 0.4982 | 0.4997 | 0.5011 |
| precision (up) | 0.4878 | 0.4850 | 0.4841 |
| recall / sensitivity (up) | 0.9331 | 0.9971 | 0.6644 |
| specificity (down) | 0.0633 | 0.0023 | 0.3378 |
| F1 (up) | 0.6407 | 0.6526 | 0.5601 |
| MCC | -0.0073 | -0.0052 | 0.0023 |
| AUC | 0.5191 | 0.5041 | 0.5031 |
| Brier | 0.2519 | 0.2514 | 0.2508 |
| ECE (positive class) | 0.0463 | 0.0404 | 0.0259 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0112 | 0.0149 | 0.0167 |
| TP / FP / TN / FN | 19077 / 20030 / 1353 / 1367 | 20545 / 21816 / 51 / 59 | 13882 / 14795 / 7546 / 7013 |
| Gaussian readout: calls up | 0.9668 | 0.9819 | 0.9855 |
| Gaussian readout: MCC | -0.0195 | -0.0134 | -0.0326 |
| Gaussian readout: AUC | 0.4843 | 0.4955 | 0.4936 |
| Gaussian readout: Brier | 0.2526 | 0.2528 | 0.2543 |
| Gaussian readout: ECE | 0.0475 | 0.0493 | 0.0642 |

## Price heads (dollars)

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| RMSE ($), served | 636.55 | 802.75 | 938.28 |
| RMSE ($), raw heads | 636.55 | 808.57 | 938.94 |
| RMSE ($), zero prediction | 633.50 | 788.78 | 927.30 |
| MAE ($), served | 433.75 | 546.97 | 660.37 |
| MAE ($), raw heads | 433.75 | 549.53 | 660.92 |
| MAE ($), zero prediction | 431.10 | 540.23 | 651.35 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0096 | -0.0357 | -0.0238 |
| skill vs zero, raw heads | -0.0096 | -0.0508 | -0.0253 |
| EV, served | -0.0014 | -0.0270 | -0.0077 |
| EV, raw heads | -0.0014 | -0.0391 | -0.0083 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0093 | -0.0071 | -0.0036 |
| corr, Spearman, raw heads | 0.0115 | -0.0008 | 0.0012 |
| mean predicted ($), served | 42.22 | 51.46 | 86.77 |
| mean predicted ($), raw heads | 42.22 | 62.40 | 89.90 |
| mean realised ($) | -17.93 | -27.24 | -36.49 |
| share predicted up, raw heads | 0.9672 | 0.9825 | 0.9859 |
| share realised up | 0.4876 | 0.4866 | 0.4841 |
| shrink beta (served = beta x raw, fit on cal) | 1.0000 | 0.8247 | 0.9651 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| CRPS ($) | 325.12 | 414.07 | 493.27 |
| CRPSS vs constant variance | 0.0270 | 0.0125 | 0.0094 |
| NLL | 7.8809 | 8.1045 | 8.2702 |
| PIT KS | 0.0804 | 0.0941 | 0.0913 |
| var / err^2 Spearman | 0.2415 | 0.1836 | 0.1902 |
| coverage of the 90% interval | 0.9205 | 0.9135 | 0.9056 |
| width of the 90% interval ($) | 2280.73 | 2958.73 | 3461.81 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0306 | [-0.0038, 0.0626] | NOISE |
| h1 | 0.0091 | [-0.0339, 0.0463] | NOISE |
| h2 | -0.0016 | [-0.0267, 0.0279] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 1.000 / h1 0.825 / h2 0.965) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8916 | 0.7399 | 0.6040 |
| abs(d h1) <= abs(d h2) | 0.8449 | 0.9001 | 0.6013 |
| full chain h0 <= h1 <= h2 | 0.7515 | 0.6648 | 0.3234 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.9159 | 0.9818 | 0.6644 | 0.6339 |
| expected if the two signs were independent | 0.9065 | 0.9801 | 0.6591 | 0.6294 |

- P(up) unanimity (all three horizons call the same side): 0.6580

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0073 vs -0.0006 (-0.0067): does not beat, noise (boot z -0.30) | -0.0052 vs -0.0245 (+0.0192): beats, noise (boot z +0.89) | 0.0023 vs -0.0208 (+0.0231): beats, noise (boot z +0.88) |
| logreg_lags | direction/auc | 0.5191 vs 0.5272 (-0.0081): does not beat, noise (boot z -0.45) | 0.5041 vs 0.5067 (-0.0026): does not beat, noise (boot z -0.23) | 0.5031 vs 0.5138 (-0.0107): does not beat, noise (boot z -0.67) |
| logreg_lags | direction/brier | 0.2519 vs 0.2502 (-0.0017): does not beat, noise (DM z -1.43) | 0.2514 vs 0.2507 (-0.0007): does not beat, noise (DM z -0.98) | 0.2508 vs 0.2508 (-0.0000): does not beat, noise (DM z -0.01) |
| logreg_lags | direction/ece_pos | 0.0463 vs 0.0240 (-0.0222): does not beat, significantly worse (boot z -4.15) | 0.0404 vs 0.0315 (-0.0089): does not beat, significantly worse (boot z -2.90) | 0.0259 vs 0.0342 (+0.0083): beats, noise (boot z +1.26) |
| logreg_lags | direction/acc | 0.4884 vs 0.4895 (-0.0011): does not beat, noise (DM z -0.19) | 0.4849 vs 0.4829 (+0.0020): beats, noise (DM z +0.91) | 0.4956 vs 0.4816 (+0.0140): beats, noise (DM z +0.56) |
| logreg_lags | direction/bal_acc | 0.4982 vs 0.4999 (-0.0017): does not beat, noise (boot z -0.36) | 0.4997 vs 0.4975 (+0.0023): beats, noise (boot z +1.08) | 0.5011 vs 0.4980 (+0.0031): beats, noise (boot z +0.34) |
| class_prior | direction/mcc | -0.0073 vs 0.0000 (-0.0073): does not beat, noise (boot z -0.45) | -0.0052 vs 0.0000 (-0.0052): does not beat, noise (boot z -0.40) | 0.0023 vs 0.0000 (+0.0023): beats, noise (boot z +0.12) |
| class_prior | direction/auc | 0.5191 vs 0.5000 (+0.0191): beats, noise (boot z +1.53) | 0.5041 vs 0.5000 (+0.0041): beats, noise (boot z +0.29) | 0.5031 vs 0.5000 (+0.0031): beats, noise (boot z +0.25) |
| class_prior | direction/brier | 0.2519 vs 0.2507 (-0.0012): does not beat, noise (DM z -1.25) | 0.2514 vs 0.2509 (-0.0005): does not beat, noise (DM z -1.04) | 0.2508 vs 0.2510 (+0.0003): beats, noise (DM z +0.28) |
| class_prior | direction/ece_pos | 0.0463 vs 0.0283 (-0.0179): does not beat, significantly worse (boot z -4.18) | 0.0404 vs 0.0337 (-0.0067): does not beat, significantly worse (boot z -4.30) | 0.0259 vs 0.0361 (+0.0102): beats, noise (boot z +1.46) |
| class_prior | direction/acc | 0.4884 vs 0.4888 (-0.0003): does not beat, noise (DM z -0.06) | 0.4849 vs 0.4851 (-0.0002): does not beat, noise (DM z -0.25) | 0.4956 vs 0.4833 (+0.0123): beats, noise (DM z +0.49) |
| class_prior | direction/bal_acc | 0.4982 vs 0.5000 (-0.0018): does not beat, noise (boot z -0.45) | 0.4997 vs 0.5000 (-0.0003): does not beat, noise (boot z -0.39) | 0.5011 vs 0.5000 (+0.0011): beats, noise (boot z +0.12) |
| zero_delta | delta/rmse | 636.55 vs 633.50 (-3.05, -0.48%): does not beat, noise (DM z -1.00) | 802.75 vs 788.78 (-13.97, -1.77%): does not beat, noise (DM z -1.28) | 938.28 vs 927.30 (-10.98, -1.18%): does not beat, noise (DM z -1.37) |
| zero_delta | delta/mae | 433.75 vs 431.10 (-2.65, -0.61%): does not beat, noise (DM z -1.23) | 546.97 vs 540.23 (-6.75, -1.25%): does not beat, noise (DM z -1.48) | 660.37 vs 651.35 (-9.02, -1.39%): does not beat, noise (DM z -1.50) |
| mean_delta | delta/rmse | 636.55 vs 634.00 (-2.54, -0.40%): does not beat, noise (DM z -1.05) | 802.75 vs 789.69 (-13.06, -1.65%): does not beat, noise (DM z -1.25) | 938.28 vs 928.68 (-9.59, -1.03%): does not beat, noise (DM z -1.55) |
| mean_delta | delta/mae | 433.75 vs 431.55 (-2.19, -0.51%): does not beat, noise (DM z -1.34) | 546.97 vs 541.02 (-5.95, -1.10%): does not beat, noise (DM z -1.55) | 660.37 vs 652.57 (-7.80, -1.20%): does not beat, noise (DM z -1.72) |
| const_var | variance/crps | 325.12 vs 334.14 (+9.02, +2.70%): beats (DM z +3.19) | 414.07 vs 419.29 (+5.23, +1.25%): beats, noise (DM z +1.19) | 493.27 vs 497.97 (+4.70, +0.94%): beats, noise (DM z +0.89) |
| const_var | variance/nll | 7.8809 vs 7.8869 (+0.0059): beats, noise (DM z +0.11) | 8.1045 vs 8.1017 (-0.0028): does not beat, noise (DM z -0.06) | 8.2702 vs 8.2600 (-0.0101): does not beat, noise (DM z -0.22) |
| const_var | variance/pit_ks | 0.0804 vs 0.1120 (+0.0317): beats (boot z +4.06) | 0.0941 vs 0.1181 (+0.0240): beats (boot z +3.63) | 0.0913 vs 0.1033 (+0.0121): beats, noise (boot z +1.86) |
| const_var | variance/corr_var_err2_spearman | 0.2415 vs 0.0000 (+0.2415): beats (boot z +8.93) | 0.1836 vs 0.0000 (+0.1836): beats (boot z +6.21) | 0.1902 vs 0.0000 (+0.1902): beats (boot z +5.79) |

## Backtest (costs included)

- n_trades: 1194
- total_return: -0.9487
- sharpe_net: -113.6040
- sharpe_gross: 1.4842
- sortino: -127.1614
- max_drawdown: 0.9487
- hit_rate: 0.0519
- hit_rate_gross: 0.5452
- profit_factor: 0.0147
- avg_hold_bars: 11.7915
- exposure: 0.3025
- turnover: 736.3132
- fees_paid: 7362.9725
- traded_notional: 7362972.4738
- breakeven_cost_bps: 0.2296
- gross_edge_per_trade_bps: 1.1650
- costs_paid: 9571.8642
- gross_pnl: 84.5225
- net_pnl: -9487.3417

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 13, 13 usable folds); this report scores the fold's out-of-sample block: 46544 sequences, 2025-07-27T03:10:00 .. 2025-08-28T10:53:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.5183, long_above 0.5469, short_below 0.5062, median 0.5271. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -94.87% | -113.60 | +94.87% | 1194 |
| buy and hold | -4.63% | -1.46 | +12.57% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.01% .. -94.83%) | -95.44% | -126.41 | | |

The random null enters at the strategy's rate (0.0368 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 94% of its seeds on net return, 100% on net Sharpe and 67% on gross return.
