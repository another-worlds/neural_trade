# Evaluation report - dev split - run `20260929T092747Z-82a848f-40c92d2b-default__f-2__s0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4965 | 5381 | 5650 |
| n_eff of the scored moves (n scored // bars ahead) | 496 | 358 | 282 |
| true up-rate | 0.4634 | 0.4620 | 0.4664 |
| calls up (predicted up-rate) | 0.7780 | 0.5679 | 0.8736 |
| accuracy | 0.4634 | 0.4923 | 0.4784 |
| balanced accuracy | 0.4837 | 0.4974 | 0.5036 |
| precision (up) | 0.4530 | 0.4598 | 0.4684 |
| recall / sensitivity (up) | 0.7605 | 0.5652 | 0.8774 |
| specificity (down) | 0.2068 | 0.4297 | 0.1297 |
| F1 (up) | 0.5678 | 0.5070 | 0.6108 |
| MCC | -0.0392 | -0.0052 | 0.0107 |
| AUC | 0.4812 | 0.5035 | 0.4993 |
| Brier | 0.2567 | 0.2505 | 0.2560 |
| ECE (positive class) | 0.0847 | 0.0385 | 0.0755 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0366 | 0.0380 | 0.0336 |
| TP / FP / TN / FN | 1750 / 2113 / 551 / 551 | 1405 / 1651 / 1244 / 1081 | 2312 / 2624 / 391 / 323 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6983 | 0.6817 | 0.6480 |
| Gaussian readout of the raw heads: MCC | 0.0424 | 0.0803 | 0.1037 |
| Gaussian readout of the raw heads: AUC | 0.5348 | 0.5493 | 0.5543 |
| Gaussian readout of the raw heads: Brier | 0.2554 | 0.2559 | 0.2573 |
| Gaussian readout of the raw heads: ECE | 0.0788 | 0.0832 | 0.0870 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 219.46 | 270.05 | 317.32 |
| RMSE ($), raw heads | 220.27 | 274.68 | 320.40 |
| RMSE ($), zero prediction | 219.46 | 270.05 | 317.32 |
| MAE ($), served | 149.54 | 182.77 | 213.96 |
| MAE ($), raw heads | 150.79 | 186.59 | 217.68 |
| MAE ($), zero prediction | 149.54 | 182.77 | 213.96 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0073 | -0.0345 | -0.0195 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | 0.0033 | -0.0215 | 0.0010 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0787 | 0.0790 | 0.1115 |
| corr, Spearman, raw heads | 0.0525 | 0.0737 | 0.0858 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 14.14 | 18.67 | 28.67 |
| mean realised ($) | -10.99 | -16.34 | -21.63 |
| share predicted up, raw heads | 0.7135 | 0.6978 | 0.6488 |
| share realised up | 0.4823 | 0.4779 | 0.4780 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 109.80 | 135.20 | 158.43 |
| CRPSS vs constant variance | 0.0409 | 0.0350 | 0.0347 |
| NLL | 6.7191 | 6.9606 | 7.0983 |
| PIT KS | 0.0350 | 0.0420 | 0.0392 |
| var / err^2 Spearman | 0.4116 | 0.4057 | 0.4055 |
| coverage of the 90% interval | 0.9232 | 0.9279 | 0.9250 |
| width of the 90% interval ($) | 733.91 | 910.25 | 1047.26 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0266 | [-0.0788, 0.0263] | NOISE |
| h1 | 0.0313 | [-0.0145, 0.0765] | NOISE |
| h2 | -0.0370 | [-0.0887, 0.0108] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8314 | n/a (beta = 0: served delta is 0) | 0.6074 |
| abs(d h1) <= abs(d h2) | 0.7078 | n/a (beta = 0: served delta is 0) | 0.5966 |
| full chain h0 <= h1 <= h2 | 0.5710 | n/a (beta = 0: served delta is 0) | 0.3308 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6385 | 0.5894 | 0.6356 | 0.3187 |
| expected if the two signs were independent | 0.6075 | 0.5441 | 0.6141 | 0.2776 |

- P(up) unanimity (all three horizons call the same side): 0.4666

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0392 vs -0.0196 (-0.0195): does not beat, noise (boot z -0.47) | -0.0052 vs -0.0258 (+0.0206): beats, noise (boot z +0.38) | 0.0107 vs -0.0067 (+0.0174): beats, noise (boot z +0.37) |
| logreg_lags | direction/auc | 0.4812 vs 0.4973 (-0.0161): does not beat, noise (boot z -0.69) | 0.5035 vs 0.4916 (+0.0119): beats, noise (boot z +0.36) | 0.4993 vs 0.4990 (+0.0003): beats, noise (boot z +0.01) |
| logreg_lags | direction/brier | 0.2567 vs 0.2518 (-0.0049): does not beat, significantly worse (DM z -1.97) | 0.2505 vs 0.2515 (+0.0010): beats, noise (DM z +0.60) | 0.2560 vs 0.2514 (-0.0046): does not beat, noise (DM z -1.84) |
| logreg_lags | direction/ece_pos | 0.0847 vs 0.0398 (-0.0449): does not beat, significantly worse (boot z -2.92) | 0.0385 vs 0.0413 (+0.0027): beats, noise (boot z +0.21) | 0.0755 vs 0.0389 (-0.0366): does not beat, significantly worse (boot z -3.67) |
| logreg_lags | direction/acc | 0.4634 vs 0.4959 (-0.0324): does not beat, noise (DM z -1.44) | 0.4923 vs 0.4912 (+0.0011): beats, noise (DM z +0.04) | 0.4784 vs 0.4966 (-0.0182): does not beat, noise (DM z -0.70) |
| logreg_lags | direction/bal_acc | 0.4837 vs 0.4903 (-0.0066): does not beat, noise (boot z -0.34) | 0.4974 vs 0.4871 (+0.0103): beats, noise (boot z +0.39) | 0.5036 vs 0.4966 (+0.0069): beats, noise (boot z +0.34) |
| class_prior | direction/mcc | -0.0392 vs 0.0000 (-0.0392): does not beat, noise (boot z -1.34) | -0.0052 vs 0.0000 (-0.0052): does not beat, noise (boot z -0.17) | 0.0107 vs 0.0000 (+0.0107): beats, noise (boot z +0.33) |
| class_prior | direction/auc | 0.4812 vs 0.5000 (-0.0188): does not beat, noise (boot z -1.03) | 0.5035 vs 0.5000 (+0.0035): beats, noise (boot z +0.17) | 0.4993 vs 0.5000 (-0.0007): does not beat, noise (boot z -0.03) |
| class_prior | direction/brier | 0.2567 vs 0.2496 (-0.0070): does not beat, significantly worse (DM z -3.15) | 0.2505 vs 0.2498 (-0.0007): does not beat, noise (DM z -0.74) | 0.2560 vs 0.2501 (-0.0059): does not beat, significantly worse (DM z -2.45) |
| class_prior | direction/ece_pos | 0.0847 vs 0.0309 (-0.0538): does not beat, significantly worse (boot z -4.36) | 0.0385 vs 0.0359 (-0.0026): does not beat, noise (boot z -0.27) | 0.0755 vs 0.0346 (-0.0409): does not beat, significantly worse (boot z -5.85) |
| class_prior | direction/acc | 0.4634 vs 0.5366 (-0.0731): does not beat, significantly worse (DM z -2.26) | 0.4923 vs 0.5380 (-0.0457): does not beat, noise (DM z -1.63) | 0.4784 vs 0.4664 (+0.0120): beats, noise (DM z +1.00) |
| class_prior | direction/bal_acc | 0.4837 vs 0.5000 (-0.0163): does not beat, noise (boot z -1.33) | 0.4974 vs 0.5000 (-0.0026): does not beat, noise (boot z -0.17) | 0.5036 vs 0.5000 (+0.0036): beats, noise (boot z +0.32) |
| zero_delta | delta/rmse | 219.46 vs 219.46 (+0.00, +0.00%): does not beat | 270.05 vs 270.05 (+0.00, +0.00%): does not beat | 317.32 vs 317.32 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 149.54 vs 149.54 (+0.00, +0.00%): does not beat | 182.77 vs 182.77 (+0.00, +0.00%): does not beat | 213.96 vs 213.96 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 219.46 vs 219.53 (+0.07, +0.03%): beats, noise (DM z +1.51) | 270.05 vs 270.18 (+0.13, +0.05%): beats, noise (DM z +1.50) | 317.32 vs 317.51 (+0.19, +0.06%): beats, noise (DM z +1.50) |
| mean_delta | delta/mae | 149.54 vs 149.59 (+0.05, +0.03%): beats, noise (DM z +1.28) | 182.77 vs 182.86 (+0.10, +0.05%): beats, noise (DM z +1.35) | 213.96 vs 214.09 (+0.13, +0.06%): beats, noise (DM z +1.20) |
| const_var | variance/crps | 109.80 vs 114.49 (+4.69, +4.09%): beats (DM z +8.41) | 135.20 vs 140.10 (+4.90, +3.50%): beats (DM z +5.65) | 158.43 vs 164.11 (+5.69, +3.47%): beats (DM z +5.70) |
| const_var | variance/nll | 6.7191 vs 6.8110 (+0.0919): beats (DM z +3.26) | 6.9606 vs 7.0201 (+0.0595): beats, noise (DM z +1.53) | 7.0983 vs 7.1849 (+0.0867): beats (DM z +2.06) |
| const_var | variance/pit_ks | 0.0350 vs 0.0957 (+0.0606): beats (boot z +4.79) | 0.0420 vs 0.1002 (+0.0582): beats (boot z +3.29) | 0.0392 vs 0.0955 (+0.0563): beats (boot z +4.03) |
| const_var | variance/corr_var_err2_spearman | 0.4116 vs 0.0000 (+0.4116): beats (boot z +12.30) | 0.4057 vs 0.0000 (+0.4057): beats (boot z +11.17) | 0.4055 vs 0.0000 (+0.4055): beats (boot z +10.15) |

## Backtest (costs included)

- n_trades: 299
- total_return: -0.5535
- sharpe_net: -130.3700
- sharpe_gross: -6.4276
- sortino: -149.1700
- max_drawdown: 0.5536
- hit_rate: 0.0836
- hit_rate_gross: 0.5050
- profit_factor: 0.0327
- avg_hold_bars: 8.6823
- exposure: 0.3588
- turnover: 409.7578
- fees_paid: 4097.4805
- costs_paid: 5326.7246
- gross_pnl: -207.7906
- net_pnl: -5534.5152

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 4, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-10-31T05:58:00+00:00 .. 2025-11-05T06:33:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.5526, long_above 0.5594, short_below 0.4910, median 0.5263. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -55.35% | -130.37 | +55.36% | 299 |
| buy and hold | -7.47% | -11.21 | +11.07% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -57.43% .. -50.27%) | -53.54% | -141.27 | | |

The random null enters at the strategy's rate (0.0644 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 20% of its seeds on net return, 95% on net Sharpe and 8% on gross return.
