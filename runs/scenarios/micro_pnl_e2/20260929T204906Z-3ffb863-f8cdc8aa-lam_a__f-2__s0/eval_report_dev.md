# Evaluation report - dev split - run `20260929T204906Z-3ffb863-f8cdc8aa-lam_a__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.7276 | 0.8887 | 0.8480 |
| accuracy | 0.4807 | 0.4805 | 0.4836 |
| balanced accuracy | 0.4886 | 0.4937 | 0.5001 |
| precision (up) | 0.4746 | 0.4794 | 0.4764 |
| recall / sensitivity (up) | 0.7158 | 0.8822 | 0.8481 |
| specificity (down) | 0.2614 | 0.1052 | 0.1521 |
| F1 (up) | 0.5708 | 0.6212 | 0.6101 |
| MCC | -0.0255 | -0.0200 | 0.0004 |
| AUC | 0.4958 | 0.4857 | 0.5035 |
| Brier | 0.2571 | 0.2537 | 0.2571 |
| ECE (positive class) | 0.0769 | 0.0551 | 0.0736 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 12533 / 13873 / 4911 / 4975 | 16108 / 17492 / 2056 / 2150 | 15935 / 17514 / 3142 / 2853 |
| Gaussian readout: calls up | 0.9931 | 0.7552 | 0.9026 |
| Gaussian readout: MCC | -0.0142 | -0.0004 | 0.0016 |
| Gaussian readout: AUC | 0.4695 | 0.4922 | 0.4945 |
| Gaussian readout: Brier | 0.2500 | 0.2509 | 0.2527 |
| Gaussian readout: ECE | 0.0188 | 0.0271 | 0.0496 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.32 | 565.44 | 812.26 |
| RMSE ($), raw heads | 415.60 | 578.71 | 862.30 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.47 | 379.08 | 555.15 |
| MAE ($), raw heads | 287.16 | 391.88 | 605.57 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | -0.0055 | -0.0115 |
| skill vs zero, raw heads | -0.0779 | -0.0533 | -0.1400 |
| EV, served | -0.0000 | -0.0034 | -0.0039 |
| EV, raw heads | -0.0284 | -0.0340 | -0.0567 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0692 | -0.0406 | -0.0222 |
| corr, Spearman, raw heads | -0.0620 | -0.0266 | -0.0204 |
| mean predicted ($), served | 0.23 | 12.15 | 39.58 |
| mean predicted ($), raw heads | 78.79 | 59.37 | 194.33 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9936 | 0.7506 | 0.9025 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0029 | 0.2047 | 0.2036 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 201.85 | 285.49 | 420.26 |
| CRPSS vs constant variance | 0.0228 | 0.0312 | 0.0552 |
| NLL | 7.4341 | 7.7844 | 8.1881 |
| PIT KS | 0.0358 | 0.0621 | 0.0685 |
| var / err^2 Spearman | 0.1376 | 0.0942 | 0.0898 |
| coverage of the 90% interval | 0.9027 | 0.9010 | 0.8619 |
| width of the 90% interval ($) | 1211.64 | 1750.10 | 2396.21 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0043 | [-0.0222, 0.0316] | NOISE |
| h1 | -0.0202 | [-0.0458, 0.0023] | NOISE |
| h2 | -0.0058 | [-0.0397, 0.0317] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.003 / h1 0.205 / h2 0.204) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.3793 | 0.9860 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.9112 | 0.9109 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.3581 | 0.8970 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7099 | 0.7305 | 0.8134 | 0.5049 |
| expected if the two signs were independent | 0.7126 | 0.6971 | 0.7790 | 0.4428 |

- P(up) unanimity (all three horizons call the same side): 0.5854

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0255 vs 0.0025 (-0.0280): does not beat, noise (boot z -0.93) | -0.0200 vs 0.0205 (-0.0405): does not beat, noise (boot z -1.55) | 0.0004 vs -0.0138 (+0.0142): beats, noise (boot z +0.47) |
| logreg_lags | direction/auc | 0.4958 vs 0.5320 (-0.0362): does not beat, noise (boot z -1.59) | 0.4857 vs 0.5251 (-0.0393): does not beat, noise (boot z -1.86) | 0.5035 vs 0.5095 (-0.0060): does not beat, noise (boot z -0.46) |
| logreg_lags | direction/brier | 0.2571 vs 0.2536 (-0.0035): does not beat, noise (DM z -1.78) | 0.2537 vs 0.2575 (+0.0038): beats, noise (DM z +1.52) | 0.2571 vs 0.2702 (+0.0131): beats (DM z +3.26) |
| logreg_lags | direction/ece_pos | 0.0769 vs 0.0663 (-0.0106): does not beat, noise (boot z -0.97) | 0.0551 vs 0.0812 (+0.0261): beats (boot z +4.63) | 0.0736 vs 0.1230 (+0.0494): beats (boot z +8.33) |
| logreg_lags | direction/acc | 0.4807 vs 0.4836 (-0.0030): does not beat, noise (DM z -0.23) | 0.4805 vs 0.4911 (-0.0107): does not beat, noise (DM z -1.37) | 0.4836 vs 0.4756 (+0.0081): beats, noise (DM z +0.59) |
| logreg_lags | direction/bal_acc | 0.4886 vs 0.5004 (-0.0117): does not beat, noise (boot z -1.21) | 0.4937 vs 0.5055 (-0.0118): does not beat, noise (boot z -1.58) | 0.5001 vs 0.4971 (+0.0030): beats, noise (boot z +0.33) |
| class_prior | direction/mcc | -0.0255 vs 0.0000 (-0.0255): does not beat, noise (boot z -1.28) | -0.0200 vs 0.0000 (-0.0200): does not beat, noise (boot z -1.45) | 0.0004 vs 0.0000 (+0.0004): beats, noise (boot z +0.02) |
| class_prior | direction/auc | 0.4958 vs 0.5000 (-0.0042): does not beat, noise (boot z -0.32) | 0.4857 vs 0.5000 (-0.0143): does not beat, noise (boot z -1.54) | 0.5035 vs 0.5000 (+0.0035): beats, noise (boot z +0.22) |
| class_prior | direction/brier | 0.2571 vs 0.2532 (-0.0039): does not beat, significantly worse (DM z -2.49) | 0.2537 vs 0.2533 (-0.0003): does not beat, noise (DM z -0.41) | 0.2571 vs 0.2573 (+0.0002): beats, noise (DM z +0.09) |
| class_prior | direction/ece_pos | 0.0769 vs 0.0593 (-0.0176): does not beat, noise (boot z -1.62) | 0.0551 vs 0.0603 (+0.0052): beats, noise (boot z +0.98) | 0.0736 vs 0.0886 (+0.0150): beats (boot z +2.68) |
| class_prior | direction/acc | 0.4807 vs 0.4824 (-0.0018): does not beat, noise (DM z -0.14) | 0.4805 vs 0.4829 (-0.0025): does not beat, noise (DM z -0.35) | 0.4836 vs 0.4763 (+0.0073): beats, noise (DM z +0.49) |
| class_prior | direction/bal_acc | 0.4886 vs 0.5000 (-0.0114): does not beat, noise (boot z -1.28) | 0.4937 vs 0.5000 (-0.0063): does not beat, noise (boot z -1.45) | 0.5001 vs 0.5000 (+0.0001): beats, noise (boot z +0.02) |
| zero_delta | delta/rmse | 400.32 vs 400.31 (-0.02, -0.00%): does not beat, significantly worse (DM z -1.97) | 565.44 vs 563.88 (-1.56, -0.28%): does not beat, significantly worse (DM z -2.38) | 812.26 vs 807.63 (-4.63, -0.57%): does not beat, noise (DM z -1.54) |
| zero_delta | delta/mae | 271.47 vs 271.46 (-0.02, -0.01%): does not beat, significantly worse (DM z -2.38) | 379.08 vs 377.88 (-1.21, -0.32%): does not beat, significantly worse (DM z -2.12) | 555.15 vs 550.98 (-4.17, -0.76%): does not beat, noise (DM z -1.75) |
| mean_delta | delta/rmse | 400.32 vs 405.60 (+5.28, +1.30%): beats (DM z +2.85) | 565.44 vs 578.56 (+13.12, +2.27%): beats (DM z +2.76) | 812.26 vs 847.03 (+34.78, +4.11%): beats (DM z +2.93) |
| mean_delta | delta/mae | 271.47 vs 277.50 (+6.03, +2.17%): beats (DM z +3.95) | 379.08 vs 393.77 (+14.69, +3.73%): beats (DM z +3.81) | 555.15 vs 595.01 (+39.86, +6.70%): beats (DM z +4.00) |
| const_var | variance/crps | 201.85 vs 206.55 (+4.70, +2.28%): beats (DM z +4.87) | 285.49 vs 294.69 (+9.20, +3.12%): beats (DM z +3.71) | 420.26 vs 444.83 (+24.57, +5.52%): beats (DM z +3.84) |
| const_var | variance/nll | 7.4341 vs 7.4448 (+0.0107): beats, noise (DM z +0.84) | 7.7844 vs 7.8022 (+0.0178): beats, noise (DM z +1.04) | 8.1881 vs 8.2104 (+0.0223): beats, noise (DM z +0.76) |
| const_var | variance/pit_ks | 0.0358 vs 0.1064 (+0.0706): beats (boot z +9.80) | 0.0621 vs 0.1404 (+0.0782): beats (boot z +13.49) | 0.0685 vs 0.1810 (+0.1126): beats (boot z +32.94) |
| const_var | variance/corr_var_err2_spearman | 0.1376 vs 0.0000 (+0.1376): beats (boot z +7.13) | 0.0942 vs 0.0000 (+0.0942): beats (boot z +4.00) | 0.0898 vs 0.0000 (+0.0898): beats (boot z +4.17) |

## Backtest (costs included)

- n_trades: 1519
- total_return: -0.9832
- sharpe_net: -142.7436
- sharpe_gross: -6.6721
- sortino: -158.1844
- max_drawdown: 0.9832
- hit_rate: 0.0579
- hit_rate_gross: 0.4226
- profit_factor: 0.0315
- avg_hold_bars: 11.1185
- exposure: 0.3910
- turnover: 721.8967
- fees_paid: 7219.3314
- traded_notional: 7219331.4463
- breakeven_cost_bps: -1.2372
- gross_edge_per_trade_bps: -0.8385
- costs_paid: 9385.1309
- gross_pnl: -446.5853
- net_pnl: -9831.7162

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7784, long_above 0.5861, short_below 0.4923, median 0.5337. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.32% | -142.74 | +98.32% | 1519 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.35% .. -97.80%) | -98.10% | -150.36 | | |

The random null enters at the strategy's rate (0.0577 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 10% of its seeds on net return, 99% on net Sharpe and 2% on gross return.
