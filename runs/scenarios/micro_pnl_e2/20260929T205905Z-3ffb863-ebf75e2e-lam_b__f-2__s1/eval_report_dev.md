# Evaluation report - dev split - run `20260929T205905Z-3ffb863-ebf75e2e-lam_b__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5485 | 0.8690 | 0.8684 |
| accuracy | 0.4894 | 0.4817 | 0.4819 |
| balanced accuracy | 0.4911 | 0.4943 | 0.4994 |
| precision (up) | 0.4743 | 0.4797 | 0.4760 |
| recall / sensitivity (up) | 0.5392 | 0.8631 | 0.8677 |
| specificity (down) | 0.4429 | 0.1255 | 0.1311 |
| F1 (up) | 0.5047 | 0.6166 | 0.6147 |
| MCC | -0.0179 | -0.0169 | -0.0018 |
| AUC | 0.4869 | 0.5012 | 0.5105 |
| Brier | 0.2538 | 0.2556 | 0.2573 |
| ECE (positive class) | 0.0483 | 0.0689 | 0.0768 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 9441 / 10464 / 8320 / 8067 | 15758 / 17094 / 2454 / 2500 | 16303 / 17949 / 2707 / 2485 |
| Gaussian readout: calls up | 0.9495 | 0.9363 | 0.8923 |
| Gaussian readout: MCC | -0.0009 | -0.0075 | 0.0093 |
| Gaussian readout: AUC | 0.4928 | 0.5158 | 0.5050 |
| Gaussian readout: Brier | 0.2508 | 0.2510 | 0.2529 |
| Gaussian readout: ECE | 0.0308 | 0.0383 | 0.0551 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.06 | 565.38 | 813.10 |
| RMSE ($), raw heads | 414.61 | 584.71 | 834.24 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.07 | 378.82 | 555.91 |
| MAE ($), raw heads | 283.81 | 395.52 | 577.48 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0038 | -0.0053 | -0.0136 |
| skill vs zero, raw heads | -0.0727 | -0.0752 | -0.0670 |
| EV, served | -0.0018 | -0.0012 | -0.0031 |
| EV, raw heads | -0.0310 | -0.0220 | -0.0208 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0372 | -0.0108 | -0.0038 |
| corr, Spearman, raw heads | -0.0185 | 0.0102 | -0.0054 |
| mean predicted ($), served | 9.80 | 20.34 | 50.40 |
| mean predicted ($), raw heads | 71.52 | 109.93 | 136.03 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9487 | 0.9323 | 0.8917 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1371 | 0.1850 | 0.3705 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.25 | 285.33 | 420.37 |
| CRPSS vs constant variance | 0.0208 | 0.0318 | 0.0550 |
| NLL | 7.4143 | 7.8395 | 8.1558 |
| PIT KS | 0.0539 | 0.0543 | 0.0814 |
| var / err^2 Spearman | 0.2101 | -0.0372 | 0.1383 |
| coverage of the 90% interval | 0.9008 | 0.9006 | 0.8616 |
| width of the 90% interval ($) | 1207.83 | 1754.09 | 2394.74 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0203 | [-0.0472, 0.0042] | NOISE |
| h1 | 0.0059 | [-0.0250, 0.0381] | NOISE |
| h2 | 0.0029 | [-0.0263, 0.0313] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.137 / h1 0.185 / h2 0.371) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8703 | 0.9345 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.5977 | 0.8516 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.5125 | 0.7949 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5688 | 0.8256 | 0.7877 | 0.4291 |
| expected if the two signs were independent | 0.5479 | 0.8181 | 0.7891 | 0.3989 |

- P(up) unanimity (all three horizons call the same side): 0.4744

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0179 vs 0.0025 (-0.0204): does not beat, noise (boot z -0.76) | -0.0169 vs 0.0205 (-0.0374): does not beat, noise (boot z -1.20) | -0.0018 vs -0.0138 (+0.0120): beats, noise (boot z +0.55) |
| logreg_lags | direction/auc | 0.4869 vs 0.5320 (-0.0451): does not beat, noise (boot z -1.90) | 0.5012 vs 0.5251 (-0.0239): does not beat, noise (boot z -1.06) | 0.5105 vs 0.5095 (+0.0010): beats, noise (boot z +0.07) |
| logreg_lags | direction/brier | 0.2538 vs 0.2536 (-0.0002): does not beat, noise (DM z -0.07) | 0.2556 vs 0.2575 (+0.0019): beats, noise (DM z +0.67) | 0.2573 vs 0.2702 (+0.0129): beats (DM z +2.97) |
| logreg_lags | direction/ece_pos | 0.0483 vs 0.0663 (+0.0180): beats, noise (boot z +1.18) | 0.0689 vs 0.0812 (+0.0123): beats, noise (boot z +1.68) | 0.0768 vs 0.1230 (+0.0462): beats (boot z +9.81) |
| logreg_lags | direction/acc | 0.4894 vs 0.4836 (+0.0058): beats, noise (DM z +0.34) | 0.4817 vs 0.4911 (-0.0094): does not beat, noise (DM z -0.90) | 0.4819 vs 0.4756 (+0.0064): beats, noise (DM z +0.80) |
| logreg_lags | direction/bal_acc | 0.4911 vs 0.5004 (-0.0093): does not beat, noise (boot z -1.00) | 0.4943 vs 0.5055 (-0.0112): does not beat, noise (boot z -1.20) | 0.4994 vs 0.4971 (+0.0022): beats, noise (boot z +0.43) |
| class_prior | direction/mcc | -0.0179 vs 0.0000 (-0.0179): does not beat, noise (boot z -1.01) | -0.0169 vs 0.0000 (-0.0169): does not beat, noise (boot z -0.93) | -0.0018 vs 0.0000 (-0.0018): does not beat, noise (boot z -0.16) |
| class_prior | direction/auc | 0.4869 vs 0.5000 (-0.0131): does not beat, noise (boot z -1.09) | 0.5012 vs 0.5000 (+0.0012): beats, noise (boot z +0.09) | 0.5105 vs 0.5000 (+0.0105): beats, noise (boot z +1.22) |
| class_prior | direction/brier | 0.2538 vs 0.2532 (-0.0005): does not beat, noise (DM z -0.31) | 0.2556 vs 0.2533 (-0.0022): does not beat, noise (DM z -1.65) | 0.2573 vs 0.2573 (-0.0000): does not beat, noise (DM z -0.03) |
| class_prior | direction/ece_pos | 0.0483 vs 0.0593 (+0.0111): beats, noise (boot z +0.72) | 0.0689 vs 0.0603 (-0.0085): does not beat, noise (boot z -1.24) | 0.0768 vs 0.0886 (+0.0118): beats (boot z +2.83) |
| class_prior | direction/acc | 0.4894 vs 0.4824 (+0.0070): beats, noise (DM z +0.40) | 0.4817 vs 0.4829 (-0.0012): does not beat, noise (DM z -0.14) | 0.4819 vs 0.4763 (+0.0056): beats, noise (DM z +0.62) |
| class_prior | direction/bal_acc | 0.4911 vs 0.5000 (-0.0089): does not beat, noise (boot z -1.01) | 0.4943 vs 0.5000 (-0.0057): does not beat, noise (boot z -0.93) | 0.4994 vs 0.5000 (-0.0006): does not beat, noise (boot z -0.16) |
| zero_delta | delta/rmse | 401.06 vs 400.31 (-0.76, -0.19%): does not beat, noise (DM z -1.70) | 565.38 vs 563.88 (-1.50, -0.27%): does not beat, noise (DM z -1.23) | 813.10 vs 807.63 (-5.48, -0.68%): does not beat, noise (DM z -1.34) |
| zero_delta | delta/mae | 272.07 vs 271.46 (-0.61, -0.23%): does not beat, noise (DM z -1.89) | 378.82 vs 377.88 (-0.95, -0.25%): does not beat, noise (DM z -1.07) | 555.91 vs 550.98 (-4.93, -0.89%): does not beat, noise (DM z -1.61) |
| mean_delta | delta/rmse | 401.06 vs 405.60 (+4.54, +1.12%): beats (DM z +2.90) | 565.38 vs 578.56 (+13.18, +2.28%): beats (DM z +3.18) | 813.10 vs 847.03 (+33.93, +4.01%): beats (DM z +3.10) |
| mean_delta | delta/mae | 272.07 vs 277.50 (+5.43, +1.96%): beats (DM z +4.26) | 378.82 vs 393.77 (+14.95, +3.80%): beats (DM z +4.27) | 555.91 vs 595.01 (+39.11, +6.57%): beats (DM z +4.12) |
| const_var | variance/crps | 202.25 vs 206.55 (+4.30, +2.08%): beats (DM z +5.54) | 285.33 vs 294.69 (+9.36, +3.18%): beats (DM z +4.28) | 420.37 vs 444.83 (+24.46, +5.50%): beats (DM z +4.06) |
| const_var | variance/nll | 7.4143 vs 7.4448 (+0.0305): beats (DM z +4.42) | 7.8395 vs 7.8022 (-0.0373): does not beat, noise (DM z -1.12) | 8.1558 vs 8.2104 (+0.0546): beats (DM z +2.66) |
| const_var | variance/pit_ks | 0.0539 vs 0.1064 (+0.0525): beats (boot z +17.25) | 0.0543 vs 0.1404 (+0.0861): beats (boot z +22.09) | 0.0814 vs 0.1810 (+0.0996): beats (boot z +30.40) |
| const_var | variance/corr_var_err2_spearman | 0.2101 vs 0.0000 (+0.2101): beats (boot z +11.12) | -0.0372 vs 0.0000 (-0.0372): does not beat, noise (boot z -1.43) | 0.1383 vs 0.0000 (+0.1383): beats (boot z +5.40) |

## Backtest (costs included)

- n_trades: 1386
- total_return: -0.9752
- sharpe_net: -129.8472
- sharpe_gross: -0.7774
- sortino: -148.4916
- max_drawdown: 0.9752
- hit_rate: 0.0541
- hit_rate_gross: 0.4646
- profit_factor: 0.0266
- avg_hold_bars: 10.7641
- exposure: 0.3453
- turnover: 745.4244
- fees_paid: 7454.4676
- traded_notional: 7454467.6049
- breakeven_cost_bps: -0.1639
- gross_edge_per_trade_bps: -0.6171
- costs_paid: 9690.8079
- gross_pnl: -61.0996
- net_pnl: -9751.9075

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6775, long_above 0.5633, short_below 0.4831, median 0.5190. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.52% | -129.85 | +97.52% | 1386 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.54% .. -96.81%) | -97.20% | -142.60 | | |

The random null enters at the strategy's rate (0.0490 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 8% of its seeds on net return, 100% on net Sharpe and 38% on gross return.
