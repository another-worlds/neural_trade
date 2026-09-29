# Evaluation report - dev split - run `20260929T093123Z-82a848f-8fab5ba1-default__f-2__s1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4965 | 5381 | 5650 |
| n_eff of the scored moves (n scored // bars ahead) | 496 | 358 | 282 |
| true up-rate | 0.4634 | 0.4620 | 0.4664 |
| calls up (predicted up-rate) | 0.4463 | 0.8422 | 0.8497 |
| accuracy | 0.4979 | 0.4711 | 0.4750 |
| balanced accuracy | 0.4939 | 0.4971 | 0.4986 |
| precision (up) | 0.4567 | 0.4603 | 0.4655 |
| recall / sensitivity (up) | 0.4398 | 0.8391 | 0.8482 |
| specificity (down) | 0.5480 | 0.1551 | 0.1489 |
| F1 (up) | 0.4481 | 0.5945 | 0.6011 |
| MCC | -0.0122 | -0.0079 | -0.0040 |
| AUC | 0.4901 | 0.4954 | 0.4921 |
| Brier | 0.2533 | 0.2554 | 0.2520 |
| ECE (positive class) | 0.0447 | 0.0773 | 0.0504 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0366 | 0.0380 | 0.0336 |
| TP / FP / TN / FN | 1012 / 1204 / 1460 / 1289 | 2086 / 2446 / 449 / 400 | 2235 / 2566 / 449 / 400 |
| Gaussian readout: calls up | 0.6922 | 0.6051 | 0.4184 |
| Gaussian readout: MCC | 0.0036 | 0.0364 | 0.0399 |
| Gaussian readout: AUC | 0.4985 | 0.5229 | 0.5189 |
| Gaussian readout: Brier | 0.2551 | 0.2510 | 0.2501 |
| Gaussian readout: ECE | 0.0664 | 0.0506 | 0.0374 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 219.77 | 270.46 | 317.13 |
| RMSE ($), raw heads | 219.99 | 274.17 | 321.30 |
| RMSE ($), zero prediction | 219.46 | 270.05 | 317.32 |
| MAE ($), served | 150.38 | 183.14 | 213.68 |
| MAE ($), raw heads | 150.60 | 185.23 | 215.84 |
| MAE ($), zero prediction | 149.54 | 182.77 | 213.96 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0028 | -0.0030 | 0.0012 |
| skill vs zero, raw heads | -0.0048 | -0.0307 | -0.0253 |
| EV, served | -0.0004 | -0.0045 | -0.0012 |
| EV, raw heads | -0.0021 | -0.0334 | -0.0297 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0564 | 0.0302 | 0.0374 |
| corr, Spearman, raw heads | 0.0318 | 0.0403 | 0.0460 |
| mean predicted ($), served | 4.38 | -3.61 | -6.32 |
| mean predicted ($), raw heads | 4.86 | -7.50 | -15.35 |
| mean realised ($) | -10.99 | -16.34 | -21.63 |
| share predicted up, raw heads | 0.7414 | 0.6263 | 0.4273 |
| share realised up | 0.4823 | 0.4779 | 0.4780 |
| shrink beta (served = beta x raw, fit on cal) | 0.9010 | 0.4813 | 0.4117 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 110.41 | 135.36 | 158.13 |
| CRPSS vs constant variance | 0.0356 | 0.0338 | 0.0364 |
| NLL | 6.7272 | 6.9537 | 7.1163 |
| PIT KS | 0.0535 | 0.0397 | 0.0480 |
| var / err^2 Spearman | 0.4182 | 0.4239 | 0.4067 |
| coverage of the 90% interval | 0.9194 | 0.9239 | 0.9265 |
| width of the 90% interval ($) | 727.69 | 901.45 | 1056.40 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0131 | [-0.0401, 0.0633] | NOISE |
| h1 | -0.0094 | [-0.0549, 0.0344] | NOISE |
| h2 | -0.0236 | [-0.0736, 0.0209] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.901 / h1 0.481 / h2 0.412) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4714 | 0.2908 | 0.6074 |
| abs(d h1) <= abs(d h2) | 0.6708 | 0.5973 | 0.5966 |
| full chain h0 <= h1 <= h2 | 0.2957 | 0.1625 | 0.3308 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6508 | 0.6891 | 0.4876 | 0.2881 |
| expected if the two signs were independent | 0.5140 | 0.5920 | 0.4471 | 0.2159 |

- P(up) unanimity (all three horizons call the same side): 0.5553

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0122 vs -0.0196 (+0.0074): beats, noise (boot z +0.15) | -0.0079 vs -0.0258 (+0.0179): beats, noise (boot z +0.39) | -0.0040 vs -0.0067 (+0.0027): beats, noise (boot z +0.05) |
| logreg_lags | direction/auc | 0.4901 vs 0.4973 (-0.0072): does not beat, noise (boot z -0.25) | 0.4954 vs 0.4916 (+0.0039): beats, noise (boot z +0.16) | 0.4921 vs 0.4990 (-0.0068): does not beat, noise (boot z -0.22) |
| logreg_lags | direction/brier | 0.2533 vs 0.2518 (-0.0015): does not beat, noise (DM z -0.61) | 0.2554 vs 0.2515 (-0.0039): does not beat, noise (DM z -1.77) | 0.2520 vs 0.2514 (-0.0006): does not beat, noise (DM z -0.39) |
| logreg_lags | direction/ece_pos | 0.0447 vs 0.0398 (-0.0049): does not beat, noise (boot z -0.29) | 0.0773 vs 0.0413 (-0.0360): does not beat, significantly worse (boot z -2.59) | 0.0504 vs 0.0389 (-0.0114): does not beat, noise (boot z -1.05) |
| logreg_lags | direction/acc | 0.4979 vs 0.4959 (+0.0020): beats, noise (DM z +0.09) | 0.4711 vs 0.4912 (-0.0201): does not beat, noise (DM z -0.78) | 0.4750 vs 0.4966 (-0.0216): does not beat, noise (DM z -0.81) |
| logreg_lags | direction/bal_acc | 0.4939 vs 0.4903 (+0.0037): beats, noise (boot z +0.15) | 0.4971 vs 0.4871 (+0.0100): beats, noise (boot z +0.49) | 0.4986 vs 0.4966 (+0.0019): beats, noise (boot z +0.09) |
| class_prior | direction/mcc | -0.0122 vs 0.0000 (-0.0122): does not beat, noise (boot z -0.39) | -0.0079 vs 0.0000 (-0.0079): does not beat, noise (boot z -0.30) | -0.0040 vs 0.0000 (-0.0040): does not beat, noise (boot z -0.14) |
| class_prior | direction/auc | 0.4901 vs 0.5000 (-0.0099): does not beat, noise (boot z -0.50) | 0.4954 vs 0.5000 (-0.0046): does not beat, noise (boot z -0.25) | 0.4921 vs 0.5000 (-0.0079): does not beat, noise (boot z -0.42) |
| class_prior | direction/brier | 0.2533 vs 0.2496 (-0.0036): does not beat, noise (DM z -1.95) | 0.2554 vs 0.2498 (-0.0055): does not beat, significantly worse (DM z -2.63) | 0.2520 vs 0.2501 (-0.0020): does not beat, significantly worse (DM z -2.04) |
| class_prior | direction/ece_pos | 0.0447 vs 0.0309 (-0.0137): does not beat, noise (boot z -0.83) | 0.0773 vs 0.0359 (-0.0414): does not beat, significantly worse (boot z -5.07) | 0.0504 vs 0.0346 (-0.0158): does not beat, significantly worse (boot z -2.24) |
| class_prior | direction/acc | 0.4979 vs 0.5366 (-0.0387): does not beat, noise (DM z -1.73) | 0.4711 vs 0.5380 (-0.0669): does not beat, noise (DM z -1.76) | 0.4750 vs 0.4664 (+0.0087): beats, noise (DM z +0.66) |
| class_prior | direction/bal_acc | 0.4939 vs 0.5000 (-0.0061): does not beat, noise (boot z -0.39) | 0.4971 vs 0.5000 (-0.0029): does not beat, noise (boot z -0.30) | 0.4986 vs 0.5000 (-0.0014): does not beat, noise (boot z -0.14) |
| zero_delta | delta/rmse | 219.77 vs 219.46 (-0.31, -0.14%): does not beat, noise (DM z -0.32) | 270.46 vs 270.05 (-0.41, -0.15%): does not beat, noise (DM z -0.31) | 317.13 vs 317.32 (+0.19, +0.06%): beats, noise (DM z +0.12) |
| zero_delta | delta/mae | 150.38 vs 149.54 (-0.84, -0.56%): does not beat, noise (DM z -1.40) | 183.14 vs 182.77 (-0.37, -0.20%): does not beat, noise (DM z -0.44) | 213.68 vs 213.96 (+0.28, +0.13%): beats, noise (DM z +0.30) |
| mean_delta | delta/rmse | 219.77 vs 219.53 (-0.24, -0.11%): does not beat, noise (DM z -0.25) | 270.46 vs 270.18 (-0.28, -0.10%): does not beat, noise (DM z -0.21) | 317.13 vs 317.51 (+0.38, +0.12%): beats, noise (DM z +0.23) |
| mean_delta | delta/mae | 150.38 vs 149.59 (-0.79, -0.53%): does not beat, noise (DM z -1.33) | 183.14 vs 182.86 (-0.27, -0.15%): does not beat, noise (DM z -0.32) | 213.68 vs 214.09 (+0.41, +0.19%): beats, noise (DM z +0.42) |
| const_var | variance/crps | 110.41 vs 114.49 (+4.08, +3.56%): beats (DM z +5.59) | 135.36 vs 140.10 (+4.74, +3.38%): beats (DM z +4.56) | 158.13 vs 164.11 (+5.98, +3.64%): beats (DM z +4.50) |
| const_var | variance/nll | 6.7272 vs 6.8110 (+0.0838): beats (DM z +3.04) | 6.9537 vs 7.0201 (+0.0664): beats, noise (DM z +1.89) | 7.1163 vs 7.1849 (+0.0686): beats, noise (DM z +1.57) |
| const_var | variance/pit_ks | 0.0535 vs 0.0957 (+0.0421): beats (boot z +2.76) | 0.0397 vs 0.1002 (+0.0605): beats (boot z +3.72) | 0.0480 vs 0.0955 (+0.0475): beats (boot z +2.35) |
| const_var | variance/corr_var_err2_spearman | 0.4182 vs 0.0000 (+0.4182): beats (boot z +11.76) | 0.4239 vs 0.0000 (+0.4239): beats (boot z +11.26) | 0.4067 vs 0.0000 (+0.4067): beats (boot z +10.23) |

## Backtest (costs included)

- n_trades: 342
- total_return: -0.5807
- sharpe_net: -137.3214
- sharpe_gross: 3.9268
- sortino: -157.2638
- max_drawdown: 0.5807
- hit_rate: 0.0936
- hit_rate_gross: 0.4942
- profit_factor: 0.0432
- avg_hold_bars: 10.3918
- exposure: 0.4912
- turnover: 455.1317
- fees_paid: 4550.8752
- costs_paid: 5916.1377
- gross_pnl: 108.7819
- net_pnl: -5807.3558

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 4, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-10-31T05:58:00+00:00 .. 2025-11-05T06:33:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4787, long_above 0.5421, short_below 0.4688, median 0.5138. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -58.07% | -137.32 | +58.07% | 342 |
| buy and hold | -7.47% | -11.21 | +11.07% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -62.62% .. -56.81%) | -59.46% | -149.43 | | |

The random null enters at the strategy's rate (0.0929 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 79% of its seeds on net return, 98% on net Sharpe and 65% on gross return.
