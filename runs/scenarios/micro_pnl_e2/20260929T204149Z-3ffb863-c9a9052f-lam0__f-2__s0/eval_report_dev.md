# Evaluation report - dev split - run `20260929T204149Z-3ffb863-c9a9052f-lam0__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.6093 | 0.9095 | 0.7508 |
| accuracy | 0.4899 | 0.4790 | 0.4933 |
| balanced accuracy | 0.4937 | 0.4929 | 0.5052 |
| precision (up) | 0.4773 | 0.4791 | 0.4798 |
| recall / sensitivity (up) | 0.6028 | 0.9022 | 0.7563 |
| specificity (down) | 0.3846 | 0.0837 | 0.2541 |
| F1 (up) | 0.5327 | 0.6258 | 0.5871 |
| MCC | -0.0129 | -0.0246 | 0.0120 |
| AUC | 0.4931 | 0.4938 | 0.5076 |
| Brier | 0.2574 | 0.2535 | 0.2566 |
| ECE (positive class) | 0.0667 | 0.0586 | 0.0650 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 10553 / 11559 / 7225 / 6955 | 16472 / 17912 / 1636 / 1786 | 14209 / 15407 / 5249 / 4579 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.9902 | 0.9506 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0069 | -0.0023 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4920 | 0.4971 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2506 | 0.2520 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0274 | 0.0463 |
| Gaussian readout of the raw heads: calls up | 0.9506 | 0.9902 | 0.9506 |
| Gaussian readout of the raw heads: MCC | -0.0284 | -0.0069 | -0.0023 |
| Gaussian readout of the raw heads: AUC | 0.4620 | 0.4920 | 0.4971 |
| Gaussian readout of the raw heads: Brier | 0.2795 | 0.2638 | 0.2744 |
| Gaussian readout of the raw heads: ECE | 0.1433 | 0.1065 | 0.1323 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.31 | 564.78 | 810.84 |
| RMSE ($), raw heads | 421.15 | 583.75 | 845.79 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.46 | 378.56 | 553.91 |
| MAE ($), raw heads | 293.24 | 397.61 | 590.06 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0032 | -0.0080 |
| skill vs zero, raw heads | -0.1069 | -0.0717 | -0.0968 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0011 | -0.0019 |
| EV, raw heads | -0.0439 | -0.0207 | -0.0320 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0710 | -0.0398 | -0.0110 |
| corr, Spearman, raw heads | -0.0721 | -0.0276 | -0.0136 |
| mean predicted ($), served | 0.00 | 11.81 | 33.42 |
| mean predicted ($), raw heads | 90.12 | 107.24 | 167.26 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9527 | 0.9899 | 0.9508 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1101 | 0.1998 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.09 | 286.12 | 419.46 |
| CRPSS vs constant variance | 0.0216 | 0.0291 | 0.0570 |
| NLL | 7.4326 | 7.7697 | 8.1979 |
| PIT KS | 0.0386 | 0.0720 | 0.0625 |
| var / err^2 Spearman | 0.1179 | 0.0436 | 0.0773 |
| coverage of the 90% interval | 0.9028 | 0.9007 | 0.8637 |
| width of the 90% interval ($) | 1212.02 | 1749.60 | 2412.24 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0121 | [-0.0389, 0.0154] | NOISE |
| h1 | -0.0052 | [-0.0303, 0.0190] | NOISE |
| h2 | -0.0103 | [-0.0421, 0.0251] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.110 / h2 0.200) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5628 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.6881 | 0.8381 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.4698 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5841 | 0.9128 | 0.7351 | 0.4326 |
| expected if the two signs were independent | 0.5868 | 0.9048 | 0.7241 | 0.4228 |

- P(up) unanimity (all three horizons call the same side): 0.4603

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0129 vs 0.0025 (-0.0154): does not beat, noise (boot z -0.48) | -0.0246 vs 0.0205 (-0.0451): does not beat, noise (boot z -1.94) | 0.0120 vs -0.0138 (+0.0258): beats, noise (boot z +0.86) |
| logreg_lags | direction/auc | 0.4931 vs 0.5320 (-0.0389): does not beat, noise (boot z -1.66) | 0.4938 vs 0.5251 (-0.0313): does not beat, noise (boot z -1.63) | 0.5076 vs 0.5095 (-0.0019): does not beat, noise (boot z -0.18) |
| logreg_lags | direction/brier | 0.2574 vs 0.2536 (-0.0038): does not beat, noise (DM z -1.59) | 0.2535 vs 0.2575 (+0.0040): beats, noise (DM z +1.66) | 0.2566 vs 0.2702 (+0.0136): beats (DM z +3.12) |
| logreg_lags | direction/ece_pos | 0.0667 vs 0.0663 (-0.0004): does not beat, noise (boot z -0.03) | 0.0586 vs 0.0812 (+0.0226): beats (boot z +4.42) | 0.0650 vs 0.1230 (+0.0580): beats (boot z +8.34) |
| logreg_lags | direction/acc | 0.4899 vs 0.4836 (+0.0062): beats, noise (DM z +0.37) | 0.4790 vs 0.4911 (-0.0121): does not beat, noise (DM z -1.82) | 0.4933 vs 0.4756 (+0.0177): beats, noise (DM z +0.87) |
| logreg_lags | direction/bal_acc | 0.4937 vs 0.5004 (-0.0067): does not beat, noise (boot z -0.59) | 0.4929 vs 0.5055 (-0.0126): does not beat, noise (boot z -1.95) | 0.5052 vs 0.4971 (+0.0081): beats, noise (boot z +0.75) |
| class_prior | direction/mcc | -0.0129 vs 0.0000 (-0.0129): does not beat, noise (boot z -0.62) | -0.0246 vs 0.0000 (-0.0246): does not beat, noise (boot z -1.63) | 0.0120 vs 0.0000 (+0.0120): beats, noise (boot z +0.52) |
| class_prior | direction/auc | 0.4931 vs 0.5000 (-0.0069): does not beat, noise (boot z -0.50) | 0.4938 vs 0.5000 (-0.0062): does not beat, noise (boot z -0.67) | 0.5076 vs 0.5000 (+0.0076): beats, noise (boot z +0.47) |
| class_prior | direction/brier | 0.2574 vs 0.2532 (-0.0041): does not beat, significantly worse (DM z -2.15) | 0.2535 vs 0.2533 (-0.0001): does not beat, noise (DM z -0.17) | 0.2566 vs 0.2573 (+0.0007): beats, noise (DM z +0.26) |
| class_prior | direction/ece_pos | 0.0667 vs 0.0593 (-0.0073): does not beat, noise (boot z -0.53) | 0.0586 vs 0.0603 (+0.0018): beats, noise (boot z +0.35) | 0.0650 vs 0.0886 (+0.0236): beats (boot z +3.35) |
| class_prior | direction/acc | 0.4899 vs 0.4824 (+0.0074): beats, noise (DM z +0.45) | 0.4790 vs 0.4829 (-0.0040): does not beat, noise (DM z -0.62) | 0.4933 vs 0.4763 (+0.0170): beats, noise (DM z +0.78) |
| class_prior | direction/bal_acc | 0.4937 vs 0.5000 (-0.0063): does not beat, noise (boot z -0.62) | 0.4929 vs 0.5000 (-0.0071): does not beat, noise (boot z -1.61) | 0.5052 vs 0.5000 (+0.0052): beats, noise (boot z +0.52) |
| zero_delta | delta/rmse | 400.31 vs 400.31 (+0.00, +0.00%): does not beat | 564.78 vs 563.88 (-0.90, -0.16%): does not beat, noise (DM z -1.54) | 810.84 vs 807.63 (-3.21, -0.40%): does not beat, noise (DM z -1.25) |
| zero_delta | delta/mae | 271.46 vs 271.46 (+0.00, +0.00%): does not beat | 378.56 vs 377.88 (-0.69, -0.18%): does not beat, noise (DM z -1.44) | 553.91 vs 550.98 (-2.93, -0.53%): does not beat, noise (DM z -1.48) |
| mean_delta | delta/rmse | 400.31 vs 405.60 (+5.29, +1.31%): beats (DM z +2.85) | 564.78 vs 578.56 (+13.78, +2.38%): beats (DM z +2.92) | 810.84 vs 847.03 (+36.20, +4.27%): beats (DM z +2.96) |
| mean_delta | delta/mae | 271.46 vs 277.50 (+6.04, +2.18%): beats (DM z +3.94) | 378.56 vs 393.77 (+15.21, +3.86%): beats (DM z +3.98) | 553.91 vs 595.01 (+41.10, +6.91%): beats (DM z +4.01) |
| const_var | variance/crps | 202.09 vs 206.55 (+4.46, +2.16%): beats (DM z +4.71) | 286.12 vs 294.69 (+8.57, +2.91%): beats (DM z +3.48) | 419.46 vs 444.83 (+25.37, +5.70%): beats (DM z +3.83) |
| const_var | variance/nll | 7.4326 vs 7.4448 (+0.0122): beats, noise (DM z +1.15) | 7.7697 vs 7.8022 (+0.0325): beats (DM z +2.72) | 8.1979 vs 8.2104 (+0.0126): beats, noise (DM z +0.37) |
| const_var | variance/pit_ks | 0.0386 vs 0.1064 (+0.0679): beats (boot z +9.50) | 0.0720 vs 0.1404 (+0.0683): beats (boot z +11.72) | 0.0625 vs 0.1810 (+0.1186): beats (boot z +26.43) |
| const_var | variance/corr_var_err2_spearman | 0.1179 vs 0.0000 (+0.1179): beats (boot z +6.87) | 0.0436 vs 0.0000 (+0.0436): beats, noise (boot z +1.87) | 0.0773 vs 0.0000 (+0.0773): beats (boot z +4.05) |

## Backtest (costs included)

- n_trades: 1570
- total_return: -0.9850
- sharpe_net: -145.8787
- sharpe_gross: -5.6620
- sortino: -161.2473
- max_drawdown: 0.9850
- hit_rate: 0.0580
- hit_rate_gross: 0.4357
- profit_factor: 0.0296
- avg_hold_bars: 11.1357
- exposure: 0.4047
- turnover: 728.5114
- fees_paid: 7285.3002
- traded_notional: 7285300.1656
- breakeven_cost_bps: -1.0417
- gross_edge_per_trade_bps: -0.7122
- costs_paid: 9470.8902
- gross_pnl: -379.4426
- net_pnl: -9850.3328

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8785, long_above 0.5803, short_below 0.4831, median 0.5275. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.50% | -145.88 | +98.50% | 1570 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.57% .. -98.09%) | -98.34% | -153.09 | | |

The random null enters at the strategy's rate (0.0611 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 12% of its seeds on net return, 98% on net Sharpe and 5% on gross return.
