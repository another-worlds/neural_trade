# Evaluation report - dev split - run `20260930T040154Z-ce1e2ed-90a1dc6e-calib_low__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5255 | 0.6935 | 0.8250 |
| accuracy | 0.4696 | 0.4825 | 0.4855 |
| balanced accuracy | 0.4704 | 0.4891 | 0.5009 |
| precision (up) | 0.4543 | 0.4751 | 0.4769 |
| recall / sensitivity (up) | 0.4949 | 0.6823 | 0.8260 |
| specificity (down) | 0.4459 | 0.2960 | 0.1759 |
| F1 (up) | 0.4737 | 0.5602 | 0.6047 |
| MCC | -0.0592 | -0.0236 | 0.0025 |
| AUC | 0.4593 | 0.4815 | 0.5052 |
| Brier | 0.2609 | 0.2579 | 0.2583 |
| ECE (positive class) | 0.0916 | 0.0710 | 0.0759 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 8665 / 10408 / 8376 / 8843 | 12457 / 13762 / 5786 / 5801 | 15519 / 17023 / 3633 / 3269 |
| Gaussian readout: calls up | 0.9876 | 0.9872 | 0.9809 |
| Gaussian readout: MCC | -0.0002 | -0.0124 | -0.0285 |
| Gaussian readout: AUC | 0.4753 | 0.4833 | 0.4750 |
| Gaussian readout: Brier | 0.2502 | 0.2501 | 0.2508 |
| Gaussian readout: ECE | 0.0219 | 0.0206 | 0.0363 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.49 | 564.04 | 808.97 |
| RMSE ($), raw heads | 409.33 | 571.67 | 810.35 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.63 | 378.00 | 552.25 |
| MAE ($), raw heads | 280.61 | 386.03 | 553.65 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0009 | -0.0005 | -0.0033 |
| skill vs zero, raw heads | -0.0456 | -0.0278 | -0.0068 |
| EV, served | -0.0004 | -0.0001 | -0.0008 |
| EV, raw heads | -0.0104 | -0.0039 | -0.0016 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0677 | -0.0274 | -0.0414 |
| corr, Spearman, raw heads | -0.0442 | -0.0204 | -0.0371 |
| mean predicted ($), served | 3.27 | 3.00 | 16.20 |
| mean predicted ($), raw heads | 64.95 | 67.86 | 29.40 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9889 | 0.9871 | 0.9800 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0503 | 0.0442 | 0.5509 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.71 | 285.07 | 419.76 |
| CRPSS vs constant variance | 0.0186 | 0.0327 | 0.0564 |
| NLL | 7.4257 | 7.7537 | 8.1256 |
| PIT KS | 0.0479 | 0.0679 | 0.0840 |
| var / err^2 Spearman | 0.1169 | 0.2017 | 0.1187 |
| coverage of the 90% interval | 0.9022 | 0.9015 | 0.8644 |
| width of the 90% interval ($) | 1210.76 | 1750.38 | 2423.91 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0369 | [-0.0701, -0.0032] | INVERTED |
| h1 | -0.0248 | [-0.0567, 0.0073] | NOISE |
| h2 | -0.0122 | [-0.0404, 0.0151] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.050 / h1 0.044 / h2 0.551) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6721 | 0.2303 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.0053 | 0.9944 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.0016 | 0.2292 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5341 | 0.6826 | 0.8305 | 0.3638 |
| expected if the two signs were independent | 0.5418 | 0.6904 | 0.8114 | 0.3594 |

- P(up) unanimity (all three horizons call the same side): 0.3847

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0592 vs 0.0025 (-0.0617): does not beat, noise (boot z -1.91) | -0.0236 vs 0.0205 (-0.0441): does not beat, noise (boot z -1.24) | 0.0025 vs -0.0138 (+0.0163): beats, noise (boot z +0.78) |
| logreg_lags | direction/auc | 0.4593 vs 0.5320 (-0.0727): does not beat, significantly worse (boot z -2.62) | 0.4815 vs 0.5251 (-0.0436): does not beat, noise (boot z -1.55) | 0.5052 vs 0.5095 (-0.0043): does not beat, noise (boot z -0.30) |
| logreg_lags | direction/brier | 0.2609 vs 0.2536 (-0.0073): does not beat, significantly worse (DM z -2.28) | 0.2579 vs 0.2575 (-0.0004): does not beat, noise (DM z -0.11) | 0.2583 vs 0.2702 (+0.0119): beats (DM z +2.72) |
| logreg_lags | direction/ece_pos | 0.0916 vs 0.0663 (-0.0253): does not beat, noise (boot z -1.30) | 0.0710 vs 0.0812 (+0.0102): beats, noise (boot z +0.70) | 0.0759 vs 0.1230 (+0.0471): beats (boot z +8.32) |
| logreg_lags | direction/acc | 0.4696 vs 0.4836 (-0.0141): does not beat, noise (DM z -0.70) | 0.4825 vs 0.4911 (-0.0086): does not beat, noise (DM z -0.50) | 0.4855 vs 0.4756 (+0.0100): beats, noise (DM z +0.98) |
| logreg_lags | direction/bal_acc | 0.4704 vs 0.5004 (-0.0300): does not beat, significantly worse (boot z -2.43) | 0.4891 vs 0.5055 (-0.0164): does not beat, noise (boot z -1.23) | 0.5009 vs 0.4971 (+0.0038): beats, noise (boot z +0.62) |
| class_prior | direction/mcc | -0.0592 vs 0.0000 (-0.0592): does not beat, significantly worse (boot z -2.54) | -0.0236 vs 0.0000 (-0.0236): does not beat, noise (boot z -1.07) | 0.0025 vs 0.0000 (+0.0025): beats, noise (boot z +0.15) |
| class_prior | direction/auc | 0.4593 vs 0.5000 (-0.0407): does not beat, significantly worse (boot z -2.71) | 0.4815 vs 0.5000 (-0.0185): does not beat, noise (boot z -1.29) | 0.5052 vs 0.5000 (+0.0052): beats, noise (boot z +0.47) |
| class_prior | direction/brier | 0.2609 vs 0.2532 (-0.0077): does not beat, significantly worse (DM z -2.93) | 0.2579 vs 0.2533 (-0.0045): does not beat, significantly worse (DM z -2.30) | 0.2583 vs 0.2573 (-0.0010): does not beat, noise (DM z -0.57) |
| class_prior | direction/ece_pos | 0.0916 vs 0.0593 (-0.0322): does not beat, noise (boot z -1.65) | 0.0710 vs 0.0603 (-0.0107): does not beat, noise (boot z -0.75) | 0.0759 vs 0.0886 (+0.0127): beats (boot z +2.17) |
| class_prior | direction/acc | 0.4696 vs 0.4824 (-0.0129): does not beat, noise (DM z -0.64) | 0.4825 vs 0.4829 (-0.0004): does not beat, noise (DM z -0.02) | 0.4855 vs 0.4763 (+0.0092): beats, noise (DM z +0.76) |
| class_prior | direction/bal_acc | 0.4704 vs 0.5000 (-0.0296): does not beat, significantly worse (boot z -2.55) | 0.4891 vs 0.5000 (-0.0109): does not beat, noise (boot z -1.07) | 0.5009 vs 0.5000 (+0.0009): beats, noise (boot z +0.15) |
| zero_delta | delta/rmse | 400.49 vs 400.31 (-0.18, -0.04%): does not beat, noise (DM z -1.62) | 564.04 vs 563.88 (-0.15, -0.03%): does not beat, noise (DM z -1.09) | 808.97 vs 807.63 (-1.34, -0.17%): does not beat, noise (DM z -1.26) |
| zero_delta | delta/mae | 271.63 vs 271.46 (-0.17, -0.06%): does not beat, noise (DM z -1.81) | 378.00 vs 377.88 (-0.13, -0.03%): does not beat, noise (DM z -1.03) | 552.25 vs 550.98 (-1.27, -0.23%): does not beat, noise (DM z -1.31) |
| mean_delta | delta/rmse | 400.49 vs 405.60 (+5.12, +1.26%): beats (DM z +2.89) | 564.04 vs 578.56 (+14.53, +2.51%): beats (DM z +2.83) | 808.97 vs 847.03 (+38.06, +4.49%): beats (DM z +2.79) |
| mean_delta | delta/mae | 271.63 vs 277.50 (+5.87, +2.12%): beats (DM z +4.07) | 378.00 vs 393.77 (+15.77, +4.00%): beats (DM z +3.80) | 552.25 vs 595.01 (+42.76, +7.19%): beats (DM z +3.85) |
| const_var | variance/crps | 202.71 vs 206.55 (+3.84, +1.86%): beats (DM z +4.37) | 285.07 vs 294.69 (+9.62, +3.27%): beats (DM z +3.63) | 419.76 vs 444.83 (+25.07, +5.64%): beats (DM z +3.38) |
| const_var | variance/nll | 7.4257 vs 7.4448 (+0.0191): beats (DM z +3.37) | 7.7537 vs 7.8022 (+0.0485): beats (DM z +3.87) | 8.1256 vs 8.2104 (+0.0849): beats (DM z +3.35) |
| const_var | variance/pit_ks | 0.0479 vs 0.1064 (+0.0586): beats (boot z +10.45) | 0.0679 vs 0.1404 (+0.0724): beats (boot z +9.38) | 0.0840 vs 0.1810 (+0.0971): beats (boot z +17.32) |
| const_var | variance/corr_var_err2_spearman | 0.1169 vs 0.0000 (+0.1169): beats (boot z +7.12) | 0.2017 vs 0.0000 (+0.2017): beats (boot z +9.01) | 0.1187 vs 0.0000 (+0.1187): beats (boot z +4.85) |

## Backtest (costs included)

- n_trades: 899
- total_return: -0.9067
- sharpe_net: -92.9851
- sharpe_gross: -1.6870
- sortino: -111.0258
- max_drawdown: 0.9067
- hit_rate: 0.0801
- hit_rate_gross: 0.4527
- profit_factor: 0.0539
- avg_hold_bars: 13.9755
- exposure: 0.2908
- turnover: 685.8325
- fees_paid: 6858.4939
- traded_notional: 6858493.8799
- breakeven_cost_bps: -0.4400
- gross_edge_per_trade_bps: -0.3239
- costs_paid: 8916.0420
- gross_pnl: -150.8806
- net_pnl: -9066.9226

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8926, long_above 0.5772, short_below 0.4413, median 0.5234. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -90.67% | -92.99 | +90.67% | 899 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -91.53% .. -89.06%) | -90.33% | -108.50 | | |

The random null enters at the strategy's rate (0.0293 per flat bar), holds 14 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 35% of its seeds on net return, 100% on net Sharpe and 23% on gross return.
