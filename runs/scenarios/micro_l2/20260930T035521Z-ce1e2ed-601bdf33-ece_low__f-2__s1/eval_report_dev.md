# Evaluation report - dev split - run `20260930T035521Z-ce1e2ed-601bdf33-ece_low__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.7219 | 0.6907 | 0.7877 |
| accuracy | 0.4724 | 0.4817 | 0.4883 |
| balanced accuracy | 0.4802 | 0.4882 | 0.5020 |
| precision (up) | 0.4687 | 0.4744 | 0.4776 |
| recall / sensitivity (up) | 0.7013 | 0.6785 | 0.7898 |
| specificity (down) | 0.2590 | 0.2980 | 0.2141 |
| F1 (up) | 0.5619 | 0.5584 | 0.5952 |
| MCC | -0.0442 | -0.0254 | 0.0048 |
| AUC | 0.4605 | 0.4807 | 0.5042 |
| Brier | 0.2634 | 0.2586 | 0.2588 |
| ECE (positive class) | 0.0992 | 0.0742 | 0.0744 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 12279 / 13919 / 4865 / 5229 | 12388 / 13723 / 5825 / 5870 | 14839 / 16233 / 4423 / 3949 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 1.0000 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0000 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.4919 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.2503 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0296 |
| Gaussian readout of the raw heads: calls up | 0.9100 | 0.9573 | 1.0000 |
| Gaussian readout of the raw heads: MCC | 0.0054 | -0.0088 | 0.0000 |
| Gaussian readout of the raw heads: AUC | 0.4920 | 0.4932 | 0.4919 |
| Gaussian readout of the raw heads: Brier | 0.2529 | 0.2520 | 0.2670 |
| Gaussian readout of the raw heads: ECE | 0.0505 | 0.0446 | 0.1319 |

beta = 0 for h0, h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.31 | 563.88 | 808.26 |
| RMSE ($), raw heads | 402.74 | 565.92 | 840.87 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.46 | 377.88 | 551.52 |
| MAE ($), raw heads | 274.45 | 380.39 | 588.01 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | -0.0016 |
| skill vs zero, raw heads | -0.0122 | -0.0072 | -0.0840 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | -0.0000 |
| EV, raw heads | -0.0033 | -0.0002 | -0.0004 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0116 | 0.0107 | -0.0101 |
| corr, Spearman, raw heads | -0.0026 | -0.0016 | -0.0069 |
| mean predicted ($), served | 0.00 | 0.00 | 10.50 |
| mean predicted ($), raw heads | 28.34 | 30.11 | 194.67 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9156 | 0.9588 | 1.0000 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0539 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 204.32 | 284.80 | 420.39 |
| CRPSS vs constant variance | 0.0108 | 0.0336 | 0.0549 |
| NLL | 7.4030 | 7.7695 | 8.1179 |
| PIT KS | 0.0693 | 0.0591 | 0.0895 |
| var / err^2 Spearman | 0.2106 | 0.1584 | 0.1137 |
| coverage of the 90% interval | 0.9028 | 0.9027 | 0.8647 |
| width of the 90% interval ($) | 1212.02 | 1755.40 | 2421.04 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0501 | [-0.0846, -0.0140] | INVERTED |
| h1 | -0.0260 | [-0.0594, 0.0068] | NOISE |
| h2 | -0.0144 | [-0.0434, 0.0126] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.054) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4954 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 1.0000 | n/a (beta = 0: served delta is 0) | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.4954 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h0, h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7271 | 0.6636 | 0.7875 | 0.4239 |
| expected if the two signs were independent | 0.6973 | 0.6750 | 0.7875 | 0.4044 |

- P(up) unanimity (all three horizons call the same side): 0.4429

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0442 vs 0.0025 (-0.0467): does not beat, noise (boot z -1.45) | -0.0254 vs 0.0205 (-0.0459): does not beat, noise (boot z -1.27) | 0.0048 vs -0.0138 (+0.0186): beats, noise (boot z +0.92) |
| logreg_lags | direction/auc | 0.4605 vs 0.5320 (-0.0716): does not beat, significantly worse (boot z -2.61) | 0.4807 vs 0.5251 (-0.0443): does not beat, noise (boot z -1.55) | 0.5042 vs 0.5095 (-0.0053): does not beat, noise (boot z -0.39) |
| logreg_lags | direction/brier | 0.2634 vs 0.2536 (-0.0098): does not beat, significantly worse (DM z -3.53) | 0.2586 vs 0.2575 (-0.0012): does not beat, noise (DM z -0.31) | 0.2588 vs 0.2702 (+0.0114): beats (DM z +2.56) |
| logreg_lags | direction/ece_pos | 0.0992 vs 0.0663 (-0.0329): does not beat, significantly worse (boot z -2.32) | 0.0742 vs 0.0812 (+0.0070): beats, noise (boot z +0.47) | 0.0744 vs 0.1230 (+0.0486): beats (boot z +7.84) |
| logreg_lags | direction/acc | 0.4724 vs 0.4836 (-0.0112): does not beat, noise (DM z -0.78) | 0.4817 vs 0.4911 (-0.0094): does not beat, noise (DM z -0.54) | 0.4883 vs 0.4756 (+0.0128): beats, noise (DM z +1.04) |
| logreg_lags | direction/bal_acc | 0.4802 vs 0.5004 (-0.0202): does not beat, noise (boot z -1.78) | 0.4882 vs 0.5055 (-0.0173): does not beat, noise (boot z -1.26) | 0.5020 vs 0.4971 (+0.0048): beats, noise (boot z +0.76) |
| class_prior | direction/mcc | -0.0442 vs 0.0000 (-0.0442): does not beat, noise (boot z -1.84) | -0.0254 vs 0.0000 (-0.0254): does not beat, noise (boot z -1.13) | 0.0048 vs 0.0000 (+0.0048): beats, noise (boot z +0.28) |
| class_prior | direction/auc | 0.4605 vs 0.5000 (-0.0395): does not beat, significantly worse (boot z -2.65) | 0.4807 vs 0.5000 (-0.0193): does not beat, noise (boot z -1.30) | 0.5042 vs 0.5000 (+0.0042): beats, noise (boot z +0.36) |
| class_prior | direction/brier | 0.2634 vs 0.2532 (-0.0102): does not beat, significantly worse (DM z -4.64) | 0.2586 vs 0.2533 (-0.0053): does not beat, significantly worse (DM z -2.59) | 0.2588 vs 0.2573 (-0.0015): does not beat, noise (DM z -0.72) |
| class_prior | direction/ece_pos | 0.0992 vs 0.0593 (-0.0399): does not beat, significantly worse (boot z -2.80) | 0.0742 vs 0.0603 (-0.0139): does not beat, noise (boot z -0.95) | 0.0744 vs 0.0886 (+0.0142): beats (boot z +2.13) |
| class_prior | direction/acc | 0.4724 vs 0.4824 (-0.0100): does not beat, noise (DM z -0.70) | 0.4817 vs 0.4829 (-0.0012): does not beat, noise (DM z -0.07) | 0.4883 vs 0.4763 (+0.0120): beats, noise (DM z +0.82) |
| class_prior | direction/bal_acc | 0.4802 vs 0.5000 (-0.0198): does not beat, noise (boot z -1.85) | 0.4882 vs 0.5000 (-0.0118): does not beat, noise (boot z -1.13) | 0.5020 vs 0.5000 (+0.0020): beats, noise (boot z +0.28) |
| zero_delta | delta/rmse | 400.31 vs 400.31 (+0.00, +0.00%): does not beat | 563.88 vs 563.88 (+0.00, +0.00%): does not beat | 808.26 vs 807.63 (-0.63, -0.08%): does not beat, noise (DM z -0.87) |
| zero_delta | delta/mae | 271.46 vs 271.46 (+0.00, +0.00%): does not beat | 377.88 vs 377.88 (+0.00, +0.00%): does not beat | 551.52 vs 550.98 (-0.54, -0.10%): does not beat, noise (DM z -0.88) |
| mean_delta | delta/rmse | 400.31 vs 405.60 (+5.29, +1.31%): beats (DM z +2.85) | 563.88 vs 578.56 (+14.68, +2.54%): beats (DM z +2.78) | 808.26 vs 847.03 (+38.78, +4.58%): beats (DM z +2.78) |
| mean_delta | delta/mae | 271.46 vs 277.50 (+6.04, +2.18%): beats (DM z +3.94) | 377.88 vs 393.77 (+15.89, +4.04%): beats (DM z +3.72) | 551.52 vs 595.01 (+43.50, +7.31%): beats (DM z +3.80) |
| const_var | variance/crps | 204.32 vs 206.55 (+2.23, +1.08%): beats (DM z +2.24) | 284.80 vs 294.69 (+9.89, +3.36%): beats (DM z +3.67) | 420.39 vs 444.83 (+24.44, +5.49%): beats (DM z +3.18) |
| const_var | variance/nll | 7.4030 vs 7.4448 (+0.0418): beats (DM z +2.61) | 7.7695 vs 7.8022 (+0.0327): beats (DM z +2.56) | 8.1179 vs 8.2104 (+0.0925): beats (DM z +3.13) |
| const_var | variance/pit_ks | 0.0693 vs 0.1064 (+0.0371): beats (boot z +4.33) | 0.0591 vs 0.1404 (+0.0813): beats (boot z +9.25) | 0.0895 vs 0.1810 (+0.0915): beats (boot z +14.21) |
| const_var | variance/corr_var_err2_spearman | 0.2106 vs 0.0000 (+0.2106): beats (boot z +11.55) | 0.1584 vs 0.0000 (+0.1584): beats (boot z +7.63) | 0.1137 vs 0.0000 (+0.1137): beats (boot z +4.95) |

## Backtest (costs included)

- n_trades: 907
- total_return: -0.9082
- sharpe_net: -92.2559
- sharpe_gross: -0.9423
- sortino: -110.1086
- max_drawdown: 0.9083
- hit_rate: 0.0849
- hit_rate_gross: 0.4576
- profit_factor: 0.0592
- avg_hold_bars: 13.8897
- exposure: 0.2916
- turnover: 691.8695
- fees_paid: 6918.7714
- traded_notional: 6918771.4202
- breakeven_cost_bps: -0.2545
- gross_edge_per_trade_bps: -0.2759
- costs_paid: 8994.4028
- gross_pnl: -88.0451
- net_pnl: -9082.4479

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8447, long_above 0.5991, short_below 0.4625, median 0.5450. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -90.82% | -92.26 | +90.83% | 907 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -91.73% .. -89.23%) | -90.49% | -108.92 | | |

The random null enters at the strategy's rate (0.0296 per flat bar), holds 14 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 32% of its seeds on net return, 100% on net Sharpe and 36% on gross return.
