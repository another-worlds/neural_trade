# Evaluation report - dev split - run `20260930T040844Z-ce1e2ed-ad958ab6-focal__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.3354 | 0.7550 | 0.8243 |
| accuracy | 0.4954 | 0.4865 | 0.4888 |
| balanced accuracy | 0.4896 | 0.4952 | 0.5042 |
| precision (up) | 0.4670 | 0.4798 | 0.4789 |
| recall / sensitivity (up) | 0.3247 | 0.7501 | 0.8287 |
| specificity (down) | 0.6546 | 0.2403 | 0.1798 |
| F1 (up) | 0.3830 | 0.5852 | 0.6070 |
| MCC | -0.0220 | -0.0111 | 0.0111 |
| AUC | 0.4808 | 0.4997 | 0.5125 |
| Brier | 0.2541 | 0.2545 | 0.2575 |
| ECE (positive class) | 0.0484 | 0.0592 | 0.0735 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 5684 / 6488 / 12296 / 11824 | 13695 / 14850 / 4698 / 4563 | 15569 / 16943 / 3713 / 3219 |
| Gaussian readout: calls up | 0.9478 | 0.8166 | 0.7336 |
| Gaussian readout: MCC | -0.0125 | 0.0099 | 0.0284 |
| Gaussian readout: AUC | 0.4775 | 0.5079 | 0.5032 |
| Gaussian readout: Brier | 0.2507 | 0.2511 | 0.2532 |
| Gaussian readout: ECE | 0.0276 | 0.0352 | 0.0546 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.81 | 565.86 | 813.55 |
| RMSE ($), raw heads | 418.19 | 589.29 | 852.58 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.93 | 379.16 | 556.72 |
| MAE ($), raw heads | 288.32 | 399.11 | 596.51 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0025 | -0.0070 | -0.0147 |
| skill vs zero, raw heads | -0.0913 | -0.0921 | -0.1144 |
| EV, served | -0.0014 | -0.0028 | -0.0040 |
| EV, raw heads | -0.0414 | -0.0454 | -0.0480 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0492 | -0.0133 | 0.0037 |
| corr, Spearman, raw heads | -0.0417 | -0.0009 | -0.0035 |
| mean predicted ($), served | 6.27 | 20.85 | 51.12 |
| mean predicted ($), raw heads | 79.21 | 101.95 | 169.86 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9499 | 0.8085 | 0.7308 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0791 | 0.2045 | 0.3009 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 201.47 | 284.24 | 420.43 |
| CRPSS vs constant variance | 0.0246 | 0.0355 | 0.0548 |
| NLL | 7.4361 | 7.7813 | 8.1882 |
| PIT KS | 0.0333 | 0.0622 | 0.0713 |
| var / err^2 Spearman | 0.1801 | 0.1697 | 0.1330 |
| coverage of the 90% interval | 0.9020 | 0.9010 | 0.8605 |
| width of the 90% interval ($) | 1210.14 | 1751.70 | 2383.98 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0074 | [-0.0371, 0.0206] | NOISE |
| h1 | -0.0059 | [-0.0367, 0.0246] | NOISE |
| h2 | -0.0001 | [-0.0292, 0.0282] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.079 / h1 0.204 / h2 0.301) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6012 | 0.7809 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.9002 | 0.9373 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.5748 | 0.7341 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.3820 | 0.6812 | 0.7271 | 0.2457 |
| expected if the two signs were independent | 0.3589 | 0.6546 | 0.6496 | 0.1902 |

- P(up) unanimity (all three horizons call the same side): 0.2920

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0220 vs 0.0025 (-0.0245): does not beat, noise (boot z -0.83) | -0.0111 vs 0.0205 (-0.0317): does not beat, noise (boot z -1.11) | 0.0111 vs -0.0138 (+0.0249): beats, noise (boot z +1.07) |
| logreg_lags | direction/auc | 0.4808 vs 0.5320 (-0.0512): does not beat, significantly worse (boot z -2.13) | 0.4997 vs 0.5251 (-0.0254): does not beat, noise (boot z -1.26) | 0.5125 vs 0.5095 (+0.0030): beats, noise (boot z +0.22) |
| logreg_lags | direction/brier | 0.2541 vs 0.2536 (-0.0006): does not beat, noise (DM z -0.20) | 0.2545 vs 0.2575 (+0.0030): beats, noise (DM z +1.17) | 0.2575 vs 0.2702 (+0.0127): beats (DM z +3.05) |
| logreg_lags | direction/ece_pos | 0.0484 vs 0.0663 (+0.0179): beats, noise (boot z +0.80) | 0.0592 vs 0.0812 (+0.0220): beats (boot z +2.38) | 0.0735 vs 0.1230 (+0.0495): beats (boot z +11.78) |
| logreg_lags | direction/acc | 0.4954 vs 0.4836 (+0.0118): beats, noise (DM z +0.50) | 0.4865 vs 0.4911 (-0.0046): does not beat, noise (DM z -0.37) | 0.4888 vs 0.4756 (+0.0133): beats, noise (DM z +1.26) |
| logreg_lags | direction/bal_acc | 0.4896 vs 0.5004 (-0.0108): does not beat, noise (boot z -1.06) | 0.4952 vs 0.5055 (-0.0103): does not beat, noise (boot z -1.07) | 0.5042 vs 0.4971 (+0.0071): beats, noise (boot z +1.18) |
| class_prior | direction/mcc | -0.0220 vs 0.0000 (-0.0220): does not beat, noise (boot z -1.10) | -0.0111 vs 0.0000 (-0.0111): does not beat, noise (boot z -0.70) | 0.0111 vs 0.0000 (+0.0111): beats, noise (boot z +0.96) |
| class_prior | direction/auc | 0.4808 vs 0.5000 (-0.0192): does not beat, noise (boot z -1.47) | 0.4997 vs 0.5000 (-0.0003): does not beat, noise (boot z -0.03) | 0.5125 vs 0.5000 (+0.0125): beats, noise (boot z +1.14) |
| class_prior | direction/brier | 0.2541 vs 0.2532 (-0.0009): does not beat, noise (DM z -0.39) | 0.2545 vs 0.2533 (-0.0011): does not beat, noise (DM z -0.93) | 0.2575 vs 0.2573 (-0.0002): does not beat, noise (DM z -0.11) |
| class_prior | direction/ece_pos | 0.0484 vs 0.0593 (+0.0109): beats, noise (boot z +0.49) | 0.0592 vs 0.0603 (+0.0012): beats, noise (boot z +0.13) | 0.0735 vs 0.0886 (+0.0151): beats (boot z +4.25) |
| class_prior | direction/acc | 0.4954 vs 0.4824 (+0.0130): beats, noise (DM z +0.54) | 0.4865 vs 0.4829 (+0.0036): beats, noise (DM z +0.28) | 0.4888 vs 0.4763 (+0.0125): beats, noise (DM z +1.05) |
| class_prior | direction/bal_acc | 0.4896 vs 0.5000 (-0.0104): does not beat, noise (boot z -1.10) | 0.4952 vs 0.5000 (-0.0048): does not beat, noise (boot z -0.70) | 0.5042 vs 0.5000 (+0.0042): beats, noise (boot z +0.96) |
| zero_delta | delta/rmse | 400.81 vs 400.31 (-0.50, -0.13%): does not beat, noise (DM z -1.90) | 565.86 vs 563.88 (-1.98, -0.35%): does not beat, noise (DM z -1.45) | 813.55 vs 807.63 (-5.92, -0.73%): does not beat, noise (DM z -1.37) |
| zero_delta | delta/mae | 271.93 vs 271.46 (-0.47, -0.17%): does not beat, significantly worse (DM z -2.36) | 379.16 vs 377.88 (-1.28, -0.34%): does not beat, noise (DM z -1.30) | 556.72 vs 550.98 (-5.74, -1.04%): does not beat, noise (DM z -1.75) |
| mean_delta | delta/rmse | 400.81 vs 405.60 (+4.79, +1.18%): beats (DM z +2.88) | 565.86 vs 578.56 (+12.70, +2.20%): beats (DM z +3.09) | 813.55 vs 847.03 (+33.49, +3.95%): beats (DM z +3.09) |
| mean_delta | delta/mae | 271.93 vs 277.50 (+5.57, +2.01%): beats (DM z +4.06) | 379.16 vs 393.77 (+14.61, +3.71%): beats (DM z +4.12) | 556.72 vs 595.01 (+38.30, +6.44%): beats (DM z +4.01) |
| const_var | variance/crps | 201.47 vs 206.55 (+5.08, +2.46%): beats (DM z +5.58) | 284.24 vs 294.69 (+10.45, +3.55%): beats (DM z +4.66) | 420.43 vs 444.83 (+24.39, +5.48%): beats (DM z +4.03) |
| const_var | variance/nll | 7.4361 vs 7.4448 (+0.0087): beats, noise (DM z +0.57) | 7.7813 vs 7.8022 (+0.0209): beats, noise (DM z +1.29) | 8.1882 vs 8.2104 (+0.0223): beats, noise (DM z +0.79) |
| const_var | variance/pit_ks | 0.0333 vs 0.1064 (+0.0731): beats (boot z +15.85) | 0.0622 vs 0.1404 (+0.0782): beats (boot z +19.48) | 0.0713 vs 0.1810 (+0.1097): beats (boot z +27.69) |
| const_var | variance/corr_var_err2_spearman | 0.1801 vs 0.0000 (+0.1801): beats (boot z +8.08) | 0.1697 vs 0.0000 (+0.1697): beats (boot z +7.07) | 0.1330 vs 0.0000 (+0.1330): beats (boot z +5.26) |

## Backtest (costs included)

- n_trades: 1373
- total_return: -0.9723
- sharpe_net: -126.7776
- sharpe_gross: 2.8953
- sortino: -145.8769
- max_drawdown: 0.9723
- hit_rate: 0.0532
- hit_rate_gross: 0.5047
- profit_factor: 0.0283
- avg_hold_bars: 11.1391
- exposure: 0.3540
- turnover: 764.2543
- fees_paid: 7642.6842
- traded_notional: 7642684.2286
- breakeven_cost_bps: 0.5565
- gross_edge_per_trade_bps: -0.0632
- costs_paid: 9935.4895
- gross_pnl: 212.6603
- net_pnl: -9722.8292

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7453, long_above 0.5527, short_below 0.4638, median 0.5068. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.23% | -126.78 | +97.23% | 1373 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.56% .. -96.85%) | -97.23% | -142.83 | | |

The random null enters at the strategy's rate (0.0492 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 50% of its seeds on net return, 100% on net Sharpe and 86% on gross return.
