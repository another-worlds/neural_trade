# Evaluation report - dev split - run `20261001T003329Z-fb840fd-1858c1d2-ece0__f-36__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 16483 | 17728 | 18448 |
| n_eff of the scored moves (n scored // bars ahead) | 1648 | 1181 | 922 |
| true up-rate | 0.5096 | 0.5110 | 0.5149 |
| calls up (predicted up-rate) | 0.4465 | 0.6332 | 0.4818 |
| accuracy | 0.5176 | 0.5119 | 0.5246 |
| balanced accuracy | 0.5187 | 0.5090 | 0.5251 |
| precision (up) | 0.5304 | 0.5181 | 0.5409 |
| recall / sensitivity (up) | 0.4648 | 0.6420 | 0.5062 |
| specificity (down) | 0.5725 | 0.3759 | 0.5440 |
| F1 (up) | 0.4955 | 0.5734 | 0.5230 |
| MCC | 0.0375 | 0.0186 | 0.0502 |
| AUC | 0.5210 | 0.5275 | 0.5273 |
| Brier | 0.2563 | 0.2500 | 0.2528 |
| ECE (positive class) | 0.0571 | 0.0282 | 0.0342 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0096 | 0.0110 | 0.0149 |
| TP / FP / TN / FN | 3904 / 3456 / 4628 / 4495 | 5816 / 5410 / 3259 / 3243 | 4808 / 4081 / 4869 / 4690 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.5867 | 0.6637 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | 0.0154 | 0.0471 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.5055 | 0.5185 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2503 | 0.2497 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0066 | 0.0123 |
| Gaussian readout of the raw heads: calls up | 0.6189 | 0.5867 | 0.6637 |
| Gaussian readout of the raw heads: MCC | 0.0150 | 0.0154 | 0.0471 |
| Gaussian readout of the raw heads: AUC | 0.5020 | 0.5055 | 0.5185 |
| Gaussian readout of the raw heads: Brier | 0.2523 | 0.2558 | 0.2531 |
| Gaussian readout of the raw heads: ECE | 0.0255 | 0.0480 | 0.0302 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 76.07 | 92.81 | 106.11 |
| RMSE ($), raw heads | 76.30 | 96.12 | 108.62 |
| RMSE ($), zero prediction | 76.07 | 92.59 | 105.96 |
| MAE ($), served | 52.53 | 64.21 | 73.32 |
| MAE ($), raw heads | 52.87 | 66.02 | 74.61 |
| MAE ($), zero prediction | 52.53 | 64.07 | 73.33 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0048 | -0.0029 |
| skill vs zero, raw heads | -0.0060 | -0.0777 | -0.0508 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0062 | -0.0045 |
| EV, raw heads | -0.0074 | -0.0790 | -0.0534 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0040 | -0.0030 | -0.0121 |
| corr, Spearman, raw heads | 0.0101 | 0.0014 | 0.0194 |
| mean predicted ($), served | 0.00 | 1.95 | 1.96 |
| mean predicted ($), raw heads | 2.87 | 7.12 | 7.62 |
| mean realised ($) | 2.79 | 4.20 | 5.63 |
| share predicted up, raw heads | 0.6054 | 0.5755 | 0.6574 |
| share realised up | 0.5013 | 0.5012 | 0.5102 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.2734 | 0.2568 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 38.49 | 47.26 | 54.15 |
| CRPSS vs constant variance | 0.0258 | 0.0190 | 0.0192 |
| NLL | 5.6154 | 5.8273 | 5.9668 |
| PIT KS | 0.0620 | 0.0607 | 0.0656 |
| var / err^2 Spearman | 0.3284 | 0.3178 | 0.3102 |
| coverage of the 90% interval | 0.8871 | 0.8869 | 0.8786 |
| width of the 90% interval ($) | 219.01 | 269.70 | 303.85 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0080 | [-0.0173, 0.0343] | NOISE |
| h1 | 0.0437 | [0.0204, 0.0708] | WORKS |
| h2 | 0.0117 | [-0.0113, 0.0343] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.273 / h2 0.257) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7404 | n/a (beta = 0: served delta is 0) | 0.6105 |
| abs(d h1) <= abs(d h2) | 0.6545 | 0.6169 | 0.5875 |
| full chain h0 <= h1 <= h2 | 0.4316 | n/a (beta = 0: served delta is 0) | 0.3272 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5902 | 0.6612 | 0.6448 | 0.3182 |
| expected if the two signs were independent | 0.4856 | 0.5211 | 0.4923 | 0.1987 |

- P(up) unanimity (all three horizons call the same side): 0.4837

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0375 vs 0.0600 (-0.0225): does not beat, noise (boot z -1.03) | 0.0186 vs 0.0619 (-0.0432): does not beat, significantly worse (boot z -2.52) | 0.0502 vs 0.0325 (+0.0177): beats, noise (boot z +0.85) |
| logreg_lags | direction/auc | 0.5210 vs 0.5395 (-0.0185): does not beat, noise (boot z -1.48) | 0.5275 vs 0.5444 (-0.0170): does not beat, noise (boot z -1.84) | 0.5273 vs 0.5410 (-0.0137): does not beat, noise (boot z -1.21) |
| logreg_lags | direction/brier | 0.2563 vs 0.2493 (-0.0069): does not beat, significantly worse (DM z -3.75) | 0.2500 vs 0.2493 (-0.0007): does not beat, noise (DM z -0.84) | 0.2528 vs 0.2493 (-0.0035): does not beat, significantly worse (DM z -2.83) |
| logreg_lags | direction/ece_pos | 0.0571 vs 0.0118 (-0.0453): does not beat, significantly worse (boot z -3.94) | 0.0282 vs 0.0108 (-0.0174): does not beat, noise (boot z -1.36) | 0.0342 vs 0.0073 (-0.0269): does not beat, significantly worse (boot z -2.90) |
| logreg_lags | direction/acc | 0.5176 vs 0.5315 (-0.0138): does not beat, noise (DM z -1.25) | 0.5119 vs 0.5327 (-0.0208): does not beat, significantly worse (DM z -2.58) | 0.5246 vs 0.5198 (+0.0047): beats, noise (DM z +0.42) |
| logreg_lags | direction/bal_acc | 0.5187 vs 0.5289 (-0.0102): does not beat, noise (boot z -0.97) | 0.5090 vs 0.5299 (-0.0209): does not beat, significantly worse (boot z -2.52) | 0.5251 vs 0.5156 (+0.0095): beats, noise (boot z +0.95) |
| class_prior | direction/mcc | 0.0375 vs 0.0000 (+0.0375): beats (boot z +2.35) | 0.0186 vs 0.0000 (+0.0186): beats, noise (boot z +1.11) | 0.0502 vs 0.0000 (+0.0502): beats (boot z +3.16) |
| class_prior | direction/auc | 0.5210 vs 0.5000 (+0.0210): beats (boot z +2.03) | 0.5275 vs 0.5000 (+0.0275): beats (boot z +2.42) | 0.5273 vs 0.5000 (+0.0273): beats (boot z +2.63) |
| class_prior | direction/brier | 0.2563 vs 0.2499 (-0.0063): does not beat, significantly worse (DM z -3.41) | 0.2500 vs 0.2499 (-0.0001): does not beat, noise (DM z -0.12) | 0.2528 vs 0.2498 (-0.0030): does not beat, noise (DM z -1.89) |
| class_prior | direction/ece_pos | 0.0571 vs 0.0001 (-0.0570): does not beat, significantly worse (boot z -5.75) | 0.0282 vs 0.0002 (-0.0280): does not beat, significantly worse (boot z -2.66) | 0.0342 vs 0.0038 (-0.0304): does not beat, significantly worse (boot z -3.49) |
| class_prior | direction/acc | 0.5176 vs 0.5096 (+0.0081): beats, noise (DM z +0.58) | 0.5119 vs 0.5110 (+0.0009): beats, noise (DM z +0.08) | 0.5246 vs 0.5149 (+0.0097): beats, noise (DM z +0.59) |
| class_prior | direction/bal_acc | 0.5187 vs 0.5000 (+0.0187): beats (boot z +2.35) | 0.5090 vs 0.5000 (+0.0090): beats, noise (boot z +1.11) | 0.5251 vs 0.5000 (+0.0251): beats (boot z +3.16) |
| zero_delta | delta/rmse | 76.07 vs 76.07 (+0.00, +0.00%): does not beat | 92.81 vs 92.59 (-0.22, -0.24%): does not beat, noise (DM z -0.88) | 106.11 vs 105.96 (-0.15, -0.14%): does not beat, noise (DM z -0.73) |
| zero_delta | delta/mae | 52.53 vs 52.53 (+0.00, +0.00%): does not beat | 64.21 vs 64.07 (-0.14, -0.22%): does not beat, noise (DM z -1.20) | 73.32 vs 73.33 (+0.00, +0.00%): beats, noise (DM z +0.02) |
| mean_delta | delta/rmse | 76.07 vs 76.05 (-0.03, -0.04%): does not beat, noise (DM z -1.68) | 92.81 vs 92.53 (-0.27, -0.29%): does not beat, noise (DM z -1.13) | 106.11 vs 105.88 (-0.24, -0.22%): does not beat, noise (DM z -1.18) |
| mean_delta | delta/mae | 52.53 vs 52.53 (-0.00, -0.00%): does not beat, noise (DM z -0.16) | 64.21 vs 64.07 (-0.14, -0.22%): does not beat, noise (DM z -1.23) | 73.32 vs 73.30 (-0.03, -0.04%): does not beat, noise (DM z -0.24) |
| const_var | variance/crps | 38.49 vs 39.51 (+1.02, +2.58%): beats (DM z +8.76) | 47.26 vs 48.18 (+0.91, +1.90%): beats (DM z +5.50) | 54.15 vs 55.21 (+1.06, +1.92%): beats (DM z +5.37) |
| const_var | variance/nll | 5.6154 vs 5.7555 (+0.1401): beats (DM z +6.72) | 5.8273 vs 5.9518 (+0.1245): beats (DM z +5.66) | 5.9668 vs 6.0862 (+0.1194): beats (DM z +5.08) |
| const_var | variance/pit_ks | 0.0620 vs 0.0612 (-0.0008): does not beat, noise (boot z -0.16) | 0.0607 vs 0.0602 (-0.0006): does not beat, noise (boot z -0.12) | 0.0656 vs 0.0622 (-0.0033): does not beat, noise (boot z -0.77) |
| const_var | variance/corr_var_err2_spearman | 0.3284 vs 0.0000 (+0.3284): beats (boot z +15.74) | 0.3178 vs 0.0000 (+0.3178): beats (boot z +14.11) | 0.3102 vs 0.0000 (+0.3102): beats (boot z +12.59) |

## Backtest (costs included)

- n_trades: 784
- total_return: -0.0355
- sharpe_net: -2.8097
- sharpe_gross: -2.8097
- sortino: -3.8614
- max_drawdown: 0.1069
- hit_rate: 0.5574
- hit_rate_gross: 0.5574
- profit_factor: 0.9402
- avg_hold_bars: 9.6390
- exposure: 0.3098
- turnover: 1551.3285
- fees_paid: 0.0000
- traded_notional: 15514090.0335
- breakeven_cost_bps: -0.4576
- gross_edge_per_trade_bps: -0.4386
- costs_paid: 0.0000
- gross_pnl: -354.9804
- net_pnl: -354.9804

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -36 (TimeSeriesSplit fold 5, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-01-28T10:18:00 .. 2024-02-14T08:47:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.9897, long_above 0.5972, short_below 0.4280, median 0.4992. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -3.55% | -2.81 | +10.69% | 784 |
| buy and hold | +15.72% | +8.30 | +4.49% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -6.80% .. +8.21%) | +0.24% | +0.25 | | |

The random null enters at the strategy's rate (0.0466 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 28% of its seeds on net return, 32% on net Sharpe and 28% on gross return.
