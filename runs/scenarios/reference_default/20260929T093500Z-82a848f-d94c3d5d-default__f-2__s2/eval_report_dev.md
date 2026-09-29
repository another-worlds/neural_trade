# Evaluation report - dev split - run `20260929T093500Z-82a848f-d94c3d5d-default__f-2__s2`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4965 | 5381 | 5650 |
| n_eff of the scored moves (n scored // bars ahead) | 496 | 358 | 282 |
| true up-rate | 0.4634 | 0.4620 | 0.4664 |
| calls up (predicted up-rate) | 0.6788 | 0.7764 | 0.6851 |
| accuracy | 0.4721 | 0.4797 | 0.4952 |
| balanced accuracy | 0.4851 | 0.5007 | 0.5077 |
| precision (up) | 0.4525 | 0.4624 | 0.4720 |
| recall / sensitivity (up) | 0.6628 | 0.7772 | 0.6934 |
| specificity (down) | 0.3074 | 0.2242 | 0.3221 |
| F1 (up) | 0.5378 | 0.5798 | 0.5616 |
| MCC | -0.0318 | 0.0016 | 0.0166 |
| AUC | 0.4795 | 0.4995 | 0.5151 |
| Brier | 0.2545 | 0.2562 | 0.2535 |
| ECE (positive class) | 0.0625 | 0.0744 | 0.0662 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0366 | 0.0380 | 0.0336 |
| TP / FP / TN / FN | 1525 / 1845 / 819 / 776 | 1932 / 2246 / 649 / 554 | 1827 / 2044 / 971 / 808 |
| Gaussian readout: calls up | 0.6596 | 0.8008 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0385 | 0.0161 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5318 | 0.5360 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2501 | 0.2501 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0436 | 0.0404 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6596 | 0.8008 | 0.6942 |
| Gaussian readout of the raw heads: MCC | 0.0385 | 0.0161 | 0.0400 |
| Gaussian readout of the raw heads: AUC | 0.5318 | 0.5360 | 0.5260 |
| Gaussian readout of the raw heads: Brier | 0.2522 | 0.2579 | 0.2580 |
| Gaussian readout of the raw heads: ECE | 0.0588 | 0.0974 | 0.0894 |

beta = 0 for h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 219.27 | 270.04 | 317.32 |
| RMSE ($), raw heads | 219.60 | 273.78 | 321.50 |
| RMSE ($), zero prediction | 219.46 | 270.05 | 317.32 |
| MAE ($), served | 149.44 | 182.77 | 213.96 |
| MAE ($), raw heads | 149.87 | 185.69 | 218.01 |
| MAE ($), zero prediction | 149.54 | 182.77 | 213.96 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0017 | 0.0001 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0013 | -0.0278 | -0.0265 |
| EV, served | 0.0028 | 0.0005 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | 0.0033 | -0.0185 | -0.0200 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0649 | 0.0288 | 0.0417 |
| corr, Spearman, raw heads | 0.0720 | 0.0719 | 0.0822 |
| mean predicted ($), served | 2.21 | 0.88 | 0.00 |
| mean predicted ($), raw heads | 7.48 | 14.47 | 12.00 |
| mean realised ($) | -10.99 | -16.34 | -21.63 |
| share predicted up, raw heads | 0.6925 | 0.8277 | 0.7253 |
| share realised up | 0.4823 | 0.4779 | 0.4780 |
| shrink beta (served = beta x raw, fit on cal) | 0.2949 | 0.0610 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 109.23 | 134.24 | 157.49 |
| CRPSS vs constant variance | 0.0460 | 0.0418 | 0.0404 |
| NLL | 6.6589 | 6.8827 | 7.0436 |
| PIT KS | 0.0460 | 0.0459 | 0.0414 |
| var / err^2 Spearman | 0.4328 | 0.4065 | 0.4055 |
| coverage of the 90% interval | 0.9221 | 0.9281 | 0.9250 |
| width of the 90% interval ($) | 732.95 | 910.12 | 1047.26 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0152 | [-0.0635, 0.0296] | NOISE |
| h1 | -0.0246 | [-0.0664, 0.0232] | NOISE |
| h2 | -0.0207 | [-0.0673, 0.0250] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.295 / h1 0.061 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.9095 | 0.1765 | 0.6074 |
| abs(d h1) <= abs(d h2) | 0.6779 | n/a (beta = 0: served delta is 0) | 0.5966 |
| full chain h0 <= h1 <= h2 | 0.5967 | n/a (beta = 0: served delta is 0) | 0.3308 |

beta = 0 for h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7043 | 0.7863 | 0.7389 | 0.4465 |
| expected if the two signs were independent | 0.5868 | 0.6955 | 0.5967 | 0.3297 |

- P(up) unanimity (all three horizons call the same side): 0.5188

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0318 vs -0.0196 (-0.0122): does not beat, noise (boot z -0.28) | 0.0016 vs -0.0258 (+0.0274): beats, noise (boot z +0.61) | 0.0166 vs -0.0067 (+0.0233): beats, noise (boot z +0.48) |
| logreg_lags | direction/auc | 0.4795 vs 0.4973 (-0.0178): does not beat, noise (boot z -0.73) | 0.4995 vs 0.4916 (+0.0079): beats, noise (boot z +0.30) | 0.5151 vs 0.4990 (+0.0161): beats, noise (boot z +0.52) |
| logreg_lags | direction/brier | 0.2545 vs 0.2518 (-0.0027): does not beat, noise (DM z -1.59) | 0.2562 vs 0.2515 (-0.0048): does not beat, significantly worse (DM z -2.20) | 0.2535 vs 0.2514 (-0.0021): does not beat, noise (DM z -1.00) |
| logreg_lags | direction/ece_pos | 0.0625 vs 0.0398 (-0.0227): does not beat, noise (boot z -1.57) | 0.0744 vs 0.0413 (-0.0331): does not beat, significantly worse (boot z -2.44) | 0.0662 vs 0.0389 (-0.0273): does not beat, significantly worse (boot z -2.61) |
| logreg_lags | direction/acc | 0.4721 vs 0.4959 (-0.0238): does not beat, noise (DM z -1.15) | 0.4797 vs 0.4912 (-0.0115): does not beat, noise (DM z -0.47) | 0.4952 vs 0.4966 (-0.0014): does not beat, noise (DM z -0.06) |
| logreg_lags | direction/bal_acc | 0.4851 vs 0.4903 (-0.0052): does not beat, noise (boot z -0.25) | 0.5007 vs 0.4871 (+0.0135): beats, noise (boot z +0.64) | 0.5077 vs 0.4966 (+0.0111): beats, noise (boot z +0.47) |
| class_prior | direction/mcc | -0.0318 vs 0.0000 (-0.0318): does not beat, noise (boot z -0.99) | 0.0016 vs 0.0000 (+0.0016): beats, noise (boot z +0.06) | 0.0166 vs 0.0000 (+0.0166): beats, noise (boot z +0.51) |
| class_prior | direction/auc | 0.4795 vs 0.5000 (-0.0205): does not beat, noise (boot z -0.97) | 0.4995 vs 0.5000 (-0.0005): does not beat, noise (boot z -0.03) | 0.5151 vs 0.5000 (+0.0151): beats, noise (boot z +0.68) |
| class_prior | direction/brier | 0.2545 vs 0.2496 (-0.0049): does not beat, significantly worse (DM z -3.06) | 0.2562 vs 0.2498 (-0.0064): does not beat, significantly worse (DM z -3.35) | 0.2535 vs 0.2501 (-0.0034): does not beat, noise (DM z -1.73) |
| class_prior | direction/ece_pos | 0.0625 vs 0.0309 (-0.0316): does not beat, significantly worse (boot z -2.70) | 0.0744 vs 0.0359 (-0.0385): does not beat, significantly worse (boot z -6.20) | 0.0662 vs 0.0346 (-0.0316): does not beat, significantly worse (boot z -3.49) |
| class_prior | direction/acc | 0.4721 vs 0.5366 (-0.0645): does not beat, significantly worse (DM z -2.20) | 0.4797 vs 0.5380 (-0.0584): does not beat, noise (DM z -1.64) | 0.4952 vs 0.4664 (+0.0288): beats, noise (DM z +1.34) |
| class_prior | direction/bal_acc | 0.4851 vs 0.5000 (-0.0149): does not beat, noise (boot z -0.99) | 0.5007 vs 0.5000 (+0.0007): beats, noise (boot z +0.06) | 0.5077 vs 0.5000 (+0.0077): beats, noise (boot z +0.51) |
| zero_delta | delta/rmse | 219.27 vs 219.46 (+0.19, +0.09%): beats, noise (DM z +0.85) | 270.04 vs 270.05 (+0.01, +0.00%): beats, noise (DM z +0.09) | 317.32 vs 317.32 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 149.44 vs 149.54 (+0.10, +0.07%): beats, noise (DM z +0.55) | 182.77 vs 182.77 (-0.01, -0.00%): does not beat, noise (DM z -0.06) | 213.96 vs 213.96 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 219.27 vs 219.53 (+0.26, +0.12%): beats, noise (DM z +1.18) | 270.04 vs 270.18 (+0.14, +0.05%): beats, noise (DM z +0.98) | 317.32 vs 317.51 (+0.19, +0.06%): beats, noise (DM z +1.50) |
| mean_delta | delta/mae | 149.44 vs 149.59 (+0.15, +0.10%): beats, noise (DM z +0.87) | 182.77 vs 182.86 (+0.09, +0.05%): beats, noise (DM z +0.90) | 213.96 vs 214.09 (+0.13, +0.06%): beats, noise (DM z +1.20) |
| const_var | variance/crps | 109.23 vs 114.49 (+5.27, +4.60%): beats (DM z +9.57) | 134.24 vs 140.10 (+5.85, +4.18%): beats (DM z +7.52) | 157.49 vs 164.11 (+6.63, +4.04%): beats (DM z +6.51) |
| const_var | variance/nll | 6.6589 vs 6.8110 (+0.1521): beats (DM z +6.64) | 6.8827 vs 7.0201 (+0.1374): beats (DM z +4.24) | 7.0436 vs 7.1849 (+0.1413): beats (DM z +3.64) |
| const_var | variance/pit_ks | 0.0460 vs 0.0957 (+0.0497): beats (boot z +6.29) | 0.0459 vs 0.1002 (+0.0543): beats (boot z +6.03) | 0.0414 vs 0.0955 (+0.0541): beats (boot z +4.34) |
| const_var | variance/corr_var_err2_spearman | 0.4328 vs 0.0000 (+0.4328): beats (boot z +12.61) | 0.4065 vs 0.0000 (+0.4065): beats (boot z +10.68) | 0.4055 vs 0.0000 (+0.4055): beats (boot z +9.72) |

## Backtest (costs included)

- n_trades: 341
- total_return: -0.5987
- sharpe_net: -144.0285
- sharpe_gross: -3.4410
- sortino: -162.8180
- max_drawdown: 0.5987
- hit_rate: 0.0850
- hit_rate_gross: 0.5044
- profit_factor: 0.0308
- avg_hold_bars: 10.3431
- exposure: 0.4874
- turnover: 452.6590
- fees_paid: 4526.5287
- costs_paid: 5884.4873
- gross_pnl: -102.1471
- net_pnl: -5986.6344

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 4, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-10-31T05:58:00+00:00 .. 2025-11-05T06:33:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.629, long_above 0.5427, short_below 0.4825, median 0.5188. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -59.87% | -144.03 | +59.87% | 341 |
| buy and hold | -7.47% | -11.21 | +11.07% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -62.41% .. -56.51%) | -59.24% | -148.93 | | |

The random null enters at the strategy's rate (0.0919 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 33% of its seeds on net return, 71% on net Sharpe and 28% on gross return.
