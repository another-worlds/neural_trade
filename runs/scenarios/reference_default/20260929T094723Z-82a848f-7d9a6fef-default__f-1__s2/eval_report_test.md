# Evaluation report - test split - run `20260929T094723Z-82a848f-7d9a6fef-default__f-1__s2`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.6391 | 0.4738 | 0.1600 |
| accuracy | 0.5197 | 0.4887 | 0.4848 |
| balanced accuracy | 0.5159 | 0.4894 | 0.4977 |
| precision (up) | 0.5263 | 0.5035 | 0.5117 |
| recall / sensitivity (up) | 0.6545 | 0.4635 | 0.1578 |
| specificity (down) | 0.3772 | 0.5154 | 0.8376 |
| F1 (up) | 0.5835 | 0.4827 | 0.2412 |
| MCC | 0.0330 | -0.0211 | -0.0063 |
| AUC | 0.5172 | 0.4801 | 0.5102 |
| Brier | 0.2496 | 0.2518 | 0.2527 |
| ECE (positive class) | 0.0094 | 0.0333 | 0.0510 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1798 / 1618 / 980 / 949 | 1359 / 1340 / 1425 / 1573 | 481 / 459 / 2367 / 2567 |
| Gaussian readout: calls up | 0.4109 | 0.3653 | 0.3716 |
| Gaussian readout: MCC | -0.0347 | -0.0292 | -0.0182 |
| Gaussian readout: AUC | 0.4712 | 0.4769 | 0.4871 |
| Gaussian readout: Brier | 0.2520 | 0.2512 | 0.2518 |
| Gaussian readout: ECE | 0.0391 | 0.0311 | 0.0356 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 197.03 | 236.95 | 270.48 |
| RMSE ($), raw heads | 199.41 | 248.45 | 285.30 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 146.23 | 176.20 | 200.82 |
| MAE ($), raw heads | 148.08 | 184.49 | 213.10 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0083 | -0.0072 | -0.0102 |
| skill vs zero, raw heads | -0.0329 | -0.1074 | -0.1240 |
| EV, served | -0.0073 | -0.0059 | -0.0080 |
| EV, raw heads | -0.0296 | -0.0913 | -0.1049 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0486 | -0.0571 | -0.0450 |
| corr, Spearman, raw heads | -0.0501 | -0.0518 | -0.0369 |
| mean predicted ($), served | -2.55 | -3.39 | -5.30 |
| mean predicted ($), raw heads | -6.66 | -22.07 | -27.02 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.4223 | 0.3780 | 0.3796 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.3831 | 0.1536 | 0.1963 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.91 | 127.33 | 145.37 |
| CRPSS vs constant variance | 0.0181 | 0.0167 | 0.0168 |
| NLL | 6.6566 | 6.8512 | 6.9780 |
| PIT KS | 0.0288 | 0.0234 | 0.0345 |
| var / err^2 Spearman | 0.2440 | 0.2420 | 0.2379 |
| coverage of the 90% interval | 0.9008 | 0.9070 | 0.9128 |
| width of the 90% interval ($) | 639.70 | 791.23 | 919.27 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0088 | [-0.0332, 0.0516] | NOISE |
| h1 | -0.0277 | [-0.0657, 0.0106] | NOISE |
| h2 | 0.0220 | [-0.0284, 0.0711] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.383 / h1 0.154 / h2 0.196) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8180 | 0.3072 | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.8353 | 0.9157 | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.6791 | 0.2789 | 0.3159 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5116 | 0.5666 | 0.6399 | 0.1436 |
| expected if the two signs were independent | 0.4740 | 0.5063 | 0.5795 | 0.1050 |

- P(up) unanimity (all three horizons call the same side): 0.1584

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0330 vs 0.0390 (-0.0060): does not beat, noise (boot z -0.16) | -0.0211 vs 0.0397 (-0.0608): does not beat, noise (boot z -1.68) | -0.0063 vs 0.0411 (-0.0474): does not beat, noise (boot z -1.00) |
| logreg_lags | direction/auc | 0.5172 vs 0.5241 (-0.0069): does not beat, noise (boot z -0.31) | 0.4801 vs 0.5228 (-0.0428): does not beat, significantly worse (boot z -1.99) | 0.5102 vs 0.5319 (-0.0217): does not beat, noise (boot z -0.79) |
| logreg_lags | direction/brier | 0.2496 vs 0.2495 (-0.0001): does not beat, noise (DM z -0.09) | 0.2518 vs 0.2493 (-0.0025): does not beat, significantly worse (DM z -2.14) | 0.2527 vs 0.2492 (-0.0035): does not beat, significantly worse (DM z -2.15) |
| logreg_lags | direction/ece_pos | 0.0094 vs 0.0211 (+0.0116): beats, noise (boot z +1.30) | 0.0333 vs 0.0197 (-0.0136): does not beat, noise (boot z -1.09) | 0.0510 vs 0.0231 (-0.0279): does not beat, significantly worse (boot z -2.49) |
| logreg_lags | direction/acc | 0.5197 vs 0.5160 (+0.0037): beats, noise (DM z +0.18) | 0.4887 vs 0.5166 (-0.0279): does not beat, noise (DM z -1.52) | 0.4848 vs 0.5157 (-0.0308): does not beat, noise (DM z -1.50) |
| logreg_lags | direction/bal_acc | 0.5159 vs 0.5190 (-0.0032): does not beat, noise (boot z -0.17) | 0.4894 vs 0.5195 (-0.0300): does not beat, noise (boot z -1.68) | 0.4977 vs 0.5200 (-0.0223): does not beat, noise (boot z -1.09) |
| class_prior | direction/mcc | 0.0330 vs 0.0000 (+0.0330): beats, noise (boot z +1.36) | -0.0211 vs 0.0000 (-0.0211): does not beat, noise (boot z -0.97) | -0.0063 vs 0.0000 (-0.0063): does not beat, noise (boot z -0.19) |
| class_prior | direction/auc | 0.5172 vs 0.5000 (+0.0172): beats, noise (boot z +1.04) | 0.4801 vs 0.5000 (-0.0199): does not beat, noise (boot z -1.38) | 0.5102 vs 0.5000 (+0.0102): beats, noise (boot z +0.52) |
| class_prior | direction/brier | 0.2496 vs 0.2502 (+0.0006): beats, noise (DM z +0.92) | 0.2518 vs 0.2502 (-0.0016): does not beat, significantly worse (DM z -2.28) | 0.2527 vs 0.2502 (-0.0025): does not beat, noise (DM z -1.66) |
| class_prior | direction/ece_pos | 0.0094 vs 0.0208 (+0.0114): beats, noise (boot z +1.28) | 0.0333 vs 0.0196 (-0.0137): does not beat, noise (boot z -1.00) | 0.0510 vs 0.0234 (-0.0276): does not beat, significantly worse (boot z -2.89) |
| class_prior | direction/acc | 0.5197 vs 0.4861 (+0.0337): beats, noise (DM z +1.29) | 0.4887 vs 0.4853 (+0.0033): beats, noise (DM z +0.15) | 0.4848 vs 0.4811 (+0.0037): beats, noise (DM z +0.27) |
| class_prior | direction/bal_acc | 0.5159 vs 0.5000 (+0.0159): beats, noise (boot z +1.36) | 0.4894 vs 0.5000 (-0.0106): does not beat, noise (boot z -0.97) | 0.4977 vs 0.5000 (-0.0023): does not beat, noise (boot z -0.19) |
| zero_delta | delta/rmse | 197.03 vs 196.21 (-0.82, -0.42%): does not beat, significantly worse (DM z -2.54) | 236.95 vs 236.10 (-0.85, -0.36%): does not beat, significantly worse (DM z -2.32) | 270.48 vs 269.11 (-1.37, -0.51%): does not beat, significantly worse (DM z -2.32) |
| zero_delta | delta/mae | 146.23 vs 145.66 (-0.57, -0.39%): does not beat, significantly worse (DM z -2.39) | 176.20 vs 175.66 (-0.55, -0.31%): does not beat, noise (DM z -1.96) | 200.82 vs 199.93 (-0.89, -0.44%): does not beat, noise (DM z -1.76) |
| mean_delta | delta/rmse | 197.03 vs 196.24 (-0.79, -0.40%): does not beat, significantly worse (DM z -2.53) | 236.95 vs 236.14 (-0.81, -0.34%): does not beat, significantly worse (DM z -2.29) | 270.48 vs 269.17 (-1.31, -0.49%): does not beat, significantly worse (DM z -2.28) |
| mean_delta | delta/mae | 146.23 vs 145.68 (-0.55, -0.38%): does not beat, significantly worse (DM z -2.36) | 176.20 vs 175.69 (-0.51, -0.29%): does not beat, noise (DM z -1.91) | 200.82 vs 199.97 (-0.84, -0.42%): does not beat, noise (DM z -1.73) |
| const_var | variance/crps | 105.91 vs 107.87 (+1.96, +1.81%): beats (DM z +4.18) | 127.33 vs 129.50 (+2.17, +1.67%): beats (DM z +3.31) | 145.37 vs 147.86 (+2.49, +1.68%): beats (DM z +2.95) |
| const_var | variance/nll | 6.6566 vs 6.7057 (+0.0492): beats (DM z +2.56) | 6.8512 vs 6.8904 (+0.0392): beats, noise (DM z +1.62) | 6.9780 vs 7.0219 (+0.0439): beats, noise (DM z +1.73) |
| const_var | variance/pit_ks | 0.0288 vs 0.0663 (+0.0375): beats (boot z +4.78) | 0.0234 vs 0.0614 (+0.0381): beats (boot z +3.48) | 0.0345 vs 0.0644 (+0.0299): beats (boot z +2.80) |
| const_var | variance/corr_var_err2_spearman | 0.2440 vs 0.0000 (+0.2440): beats (boot z +6.44) | 0.2420 vs 0.0000 (+0.2420): beats (boot z +6.03) | 0.2379 vs 0.0000 (+0.2379): beats (boot z +5.84) |

## Backtest (costs included)

- n_trades: 253
- total_return: -0.4768
- sharpe_net: -131.2862
- sharpe_gross: 3.2956
- sortino: -147.3994
- max_drawdown: 0.4780
- hit_rate: 0.0791
- hit_rate_gross: 0.4941
- profit_factor: 0.0409
- avg_hold_bars: 9.3241
- exposure: 0.3261
- turnover: 373.6122
- fees_paid: 3736.0545
- costs_paid: 4856.8708
- gross_pnl: 88.3992
- net_pnl: -4768.4716

## Experiment engine: out-of-sample block and backtest

Role: **test** (shown, never used to rank or choose (D-020)). Fold -1 (TimeSeriesSplit fold 5, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-11-05T06:34:00+00:00 .. 2025-11-10T07:09:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.002, long_above 0.5065, short_below 0.4645, median 0.4898. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -47.68% | -131.29 | +47.80% | 253 |
| buy and hold | +4.10% | +6.65 | +4.95% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -51.81% .. -45.05%) | -48.39% | -136.82 | | |

The random null enters at the strategy's rate (0.0519 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 62% of its seeds on net return, 77% on net Sharpe and 67% on gross return.
