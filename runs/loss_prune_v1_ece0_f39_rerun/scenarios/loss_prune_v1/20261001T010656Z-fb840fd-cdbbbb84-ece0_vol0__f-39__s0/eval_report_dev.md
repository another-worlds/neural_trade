# Evaluation report - dev split - run `20261001T010656Z-fb840fd-cdbbbb84-ece0_vol0__f-39__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 16776 | 18240 | 18977 |
| n_eff of the scored moves (n scored // bars ahead) | 1677 | 1216 | 948 |
| true up-rate | 0.5013 | 0.5054 | 0.5090 |
| calls up (predicted up-rate) | 0.4335 | 0.6020 | 0.7144 |
| accuracy | 0.4759 | 0.5247 | 0.4989 |
| balanced accuracy | 0.4760 | 0.5236 | 0.4951 |
| precision (up) | 0.4737 | 0.5250 | 0.5055 |
| recall / sensitivity (up) | 0.4096 | 0.6253 | 0.7096 |
| specificity (down) | 0.5424 | 0.4219 | 0.2805 |
| F1 (up) | 0.4393 | 0.5707 | 0.5904 |
| MCC | -0.0484 | 0.0482 | -0.0109 |
| AUC | 0.4641 | 0.5349 | 0.4978 |
| Brier | 0.2847 | 0.2525 | 0.2613 |
| ECE (positive class) | 0.1378 | 0.0459 | 0.0864 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0013 | 0.0054 | 0.0090 |
| TP / FP / TN / FN | 3445 / 3828 / 4538 / 4965 | 5764 / 5216 / 3806 / 3454 | 6854 / 6704 / 2614 / 2805 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.4891 | 0.5346 | 0.6723 |
| Gaussian readout of the raw heads: MCC | 0.0045 | 0.0043 | -0.0033 |
| Gaussian readout of the raw heads: AUC | 0.5051 | 0.5030 | 0.5009 |
| Gaussian readout of the raw heads: Brier | 0.2564 | 0.2620 | 0.2662 |
| Gaussian readout of the raw heads: ECE | 0.0613 | 0.0796 | 0.1036 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 79.22 | 96.91 | 110.98 |
| RMSE ($), raw heads | 80.98 | 99.32 | 115.31 |
| RMSE ($), zero prediction | 79.22 | 96.91 | 110.98 |
| MAE ($), served | 52.64 | 64.84 | 74.66 |
| MAE ($), raw heads | 53.64 | 66.31 | 77.45 |
| MAE ($), zero prediction | 52.64 | 64.84 | 74.66 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0449 | -0.0503 | -0.0796 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0445 | -0.0502 | -0.0750 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0463 | -0.0470 | -0.0501 |
| corr, Spearman, raw heads | 0.0025 | -0.0049 | -0.0123 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -1.54 | -0.90 | 7.60 |
| mean realised ($) | 0.03 | 0.05 | 0.07 |
| share predicted up, raw heads | 0.4975 | 0.5405 | 0.6773 |
| share realised up | 0.4976 | 0.5011 | 0.5040 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 39.24 | 48.15 | 55.35 |
| CRPSS vs constant variance | 0.0076 | 0.0105 | 0.0138 |
| NLL | 5.7394 | 6.0913 | 6.1375 |
| PIT KS | 0.0542 | 0.0276 | 0.0213 |
| var / err^2 Spearman | 0.1751 | 0.1809 | 0.1780 |
| coverage of the 90% interval | 0.8932 | 0.8948 | 0.8889 |
| width of the 90% interval ($) | 226.36 | 280.01 | 321.07 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0267 | [-0.0547, 0.0023] | NOISE |
| h1 | 0.0281 | [0.0009, 0.0554] | WORKS |
| h2 | 0.0024 | [-0.0225, 0.0266] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6292 | n/a (beta = 0: served delta is 0) | 0.6180 |
| abs(d h1) <= abs(d h2) | 0.7665 | n/a (beta = 0: served delta is 0) | 0.5907 |
| full chain h0 <= h1 <= h2 | 0.4752 | n/a (beta = 0: served delta is 0) | 0.3343 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4333 | 0.6154 | 0.7200 | 0.1514 |
| expected if the two signs were independent | 0.5003 | 0.5085 | 0.5794 | 0.1209 |

- P(up) unanimity (all three horizons call the same side): 0.1850

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0484 vs 0.0215 (-0.0698): does not beat, significantly worse (boot z -2.44) | 0.0482 vs 0.0397 (+0.0084): beats, noise (boot z +0.46) | -0.0109 vs 0.0430 (-0.0540): does not beat, significantly worse (boot z -2.85) |
| logreg_lags | direction/auc | 0.4641 vs 0.5268 (-0.0627): does not beat, significantly worse (boot z -3.33) | 0.5349 vs 0.5395 (-0.0046): does not beat, noise (boot z -0.58) | 0.4978 vs 0.5375 (-0.0397): does not beat, significantly worse (boot z -3.31) |
| logreg_lags | direction/brier | 0.2847 vs 0.2502 (-0.0345): does not beat, significantly worse (DM z -10.06) | 0.2525 vs 0.2495 (-0.0030): does not beat, significantly worse (DM z -2.47) | 0.2613 vs 0.2496 (-0.0117): does not beat, significantly worse (DM z -6.46) |
| logreg_lags | direction/ece_pos | 0.1378 vs 0.0285 (-0.1093): does not beat, significantly worse (boot z -7.46) | 0.0459 vs 0.0266 (-0.0193): does not beat, noise (boot z -1.83) | 0.0864 vs 0.0226 (-0.0638): does not beat, significantly worse (boot z -6.68) |
| logreg_lags | direction/acc | 0.4759 vs 0.5090 (-0.0331): does not beat, significantly worse (DM z -2.17) | 0.5247 vs 0.5195 (+0.0052): beats, noise (DM z +0.54) | 0.4989 vs 0.5229 (-0.0240): does not beat, significantly worse (DM z -2.77) |
| logreg_lags | direction/bal_acc | 0.4760 vs 0.5081 (-0.0321): does not beat, significantly worse (boot z -2.56) | 0.5236 vs 0.5165 (+0.0070): beats, noise (boot z +0.83) | 0.4951 vs 0.5181 (-0.0230): does not beat, significantly worse (boot z -2.79) |
| class_prior | direction/mcc | -0.0484 vs 0.0000 (-0.0484): does not beat, significantly worse (boot z -2.91) | 0.0482 vs 0.0000 (+0.0482): beats (boot z +2.34) | -0.0109 vs 0.0000 (-0.0109): does not beat, noise (boot z -0.54) |
| class_prior | direction/auc | 0.4641 vs 0.5000 (-0.0359): does not beat, significantly worse (boot z -3.62) | 0.5349 vs 0.5000 (+0.0349): beats (boot z +2.79) | 0.4978 vs 0.5000 (-0.0022): does not beat, noise (boot z -0.19) |
| class_prior | direction/brier | 0.2847 vs 0.2505 (-0.0342): does not beat, significantly worse (DM z -11.53) | 0.2525 vs 0.2503 (-0.0022): does not beat, noise (DM z -1.29) | 0.2613 vs 0.2501 (-0.0112): does not beat, significantly worse (DM z -5.49) |
| class_prior | direction/ece_pos | 0.1378 vs 0.0224 (-0.1154): does not beat, significantly worse (boot z -8.28) | 0.0459 vs 0.0173 (-0.0285): does not beat, significantly worse (boot z -2.43) | 0.0864 vs 0.0149 (-0.0714): does not beat, significantly worse (boot z -6.34) |
| class_prior | direction/acc | 0.4759 vs 0.5013 (-0.0255): does not beat, noise (DM z -1.79) | 0.5247 vs 0.5054 (+0.0193): beats, noise (DM z +1.45) | 0.4989 vs 0.5090 (-0.0101): does not beat, noise (DM z -0.89) |
| class_prior | direction/bal_acc | 0.4760 vs 0.5000 (-0.0240): does not beat, significantly worse (boot z -2.91) | 0.5236 vs 0.5000 (+0.0236): beats (boot z +2.34) | 0.4951 vs 0.5000 (-0.0049): does not beat, noise (boot z -0.54) |
| zero_delta | delta/rmse | 79.22 vs 79.22 (+0.00, +0.00%): does not beat | 96.91 vs 96.91 (+0.00, +0.00%): does not beat | 110.98 vs 110.98 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 52.64 vs 52.64 (+0.00, +0.00%): does not beat | 64.84 vs 64.84 (+0.00, +0.00%): does not beat | 74.66 vs 74.66 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 79.22 vs 79.26 (+0.04, +0.05%): beats, noise (DM z +0.86) | 96.91 vs 96.99 (+0.07, +0.08%): beats, noise (DM z +0.87) | 110.98 vs 111.09 (+0.11, +0.10%): beats, noise (DM z +0.87) |
| mean_delta | delta/mae | 52.64 vs 52.68 (+0.05, +0.09%): beats, noise (DM z +1.12) | 64.84 vs 64.90 (+0.06, +0.09%): beats, noise (DM z +0.84) | 74.66 vs 74.72 (+0.06, +0.08%): beats, noise (DM z +0.57) |
| const_var | variance/crps | 39.24 vs 39.54 (+0.30, +0.76%): beats (DM z +5.74) | 48.15 vs 48.66 (+0.51, +1.05%): beats (DM z +5.39) | 55.35 vs 56.12 (+0.78, +1.38%): beats (DM z +5.71) |
| const_var | variance/nll | 5.7394 vs 5.8104 (+0.0710): beats (DM z +3.61) | 6.0913 vs 6.0150 (-0.0763): does not beat, noise (DM z -1.71) | 6.1375 vs 6.1490 (+0.0115): beats, noise (DM z +0.56) |
| const_var | variance/pit_ks | 0.0542 vs 0.0591 (+0.0049): beats, noise (boot z +0.73) | 0.0276 vs 0.0617 (+0.0341): beats (boot z +3.04) | 0.0213 vs 0.0671 (+0.0457): beats (boot z +4.11) |
| const_var | variance/corr_var_err2_spearman | 0.1751 vs 0.0000 (+0.1751): beats (boot z +8.72) | 0.1809 vs 0.0000 (+0.1809): beats (boot z +8.34) | 0.1780 vs 0.0000 (+0.1780): beats (boot z +8.34) |

## Backtest (costs included)

- n_trades: 1203
- total_return: 0.0117
- sharpe_net: 0.9434
- sharpe_gross: 0.9434
- sortino: 1.4257
- max_drawdown: 0.0506
- hit_rate: 0.4032
- hit_rate_gross: 0.4032
- profit_factor: 1.0148
- avg_hold_bars: 8.4106
- exposure: 0.4149
- turnover: 2433.9320
- fees_paid: 0.0000
- traded_notional: 24340210.6642
- breakeven_cost_bps: 0.0961
- gross_edge_per_trade_bps: 0.1161
- costs_paid: 0.0000
- gross_pnl: 116.9321
- net_pnl: 116.9321

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -39 (TimeSeriesSplit fold 2, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2023-12-08T14:48:00 .. 2023-12-25T13:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6991, long_above 0.5815, short_below 0.4662, median 0.5197. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +1.17% | +0.94 | +5.06% | 1203 |
| buy and hold | +0.10% | +0.26 | +10.11% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -8.84% .. +10.10%) | +0.20% | +0.21 | | |

The random null enters at the strategy's rate (0.0843 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 59% of its seeds on net return, 56% on net Sharpe and 59% on gross return.
