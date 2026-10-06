# Evaluation report - dev split - run `20261003T093120Z-2573116-d0074ed7-ece0__f-39__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 16776 | 18240 | 18977 |
| n_eff of the scored moves (n scored // bars ahead) | 1677 | 1216 | 948 |
| true up-rate | 0.5013 | 0.5054 | 0.5090 |
| calls up (predicted up-rate) | 0.4072 | 0.4522 | 0.6920 |
| accuracy | 0.4764 | 0.5182 | 0.4936 |
| balanced accuracy | 0.4766 | 0.5187 | 0.4901 |
| precision (up) | 0.4726 | 0.5261 | 0.5019 |
| recall / sensitivity (up) | 0.3839 | 0.4707 | 0.6824 |
| specificity (down) | 0.5693 | 0.5667 | 0.2979 |
| F1 (up) | 0.4237 | 0.4969 | 0.5784 |
| MCC | -0.0475 | 0.0376 | -0.0213 |
| AUC | 0.4632 | 0.5254 | 0.4881 |
| Brier | 0.2868 | 0.2529 | 0.2610 |
| ECE (positive class) | 0.1437 | 0.0467 | 0.0806 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0013 | 0.0054 | 0.0090 |
| TP / FP / TN / FN | 3229 / 3603 / 4763 / 5181 | 4339 / 3909 / 5113 / 4879 | 6591 / 6542 / 2776 / 3068 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.7594 | 0.5825 | 0.8163 |
| Gaussian readout of the raw heads: MCC | -0.0073 | -0.0199 | -0.0195 |
| Gaussian readout of the raw heads: AUC | 0.5009 | 0.4893 | 0.4967 |
| Gaussian readout of the raw heads: Brier | 0.2567 | 0.2661 | 0.2637 |
| Gaussian readout of the raw heads: ECE | 0.0653 | 0.0894 | 0.1013 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 79.22 | 96.91 | 110.98 |
| RMSE ($), raw heads | 80.78 | 100.54 | 116.10 |
| RMSE ($), zero prediction | 79.22 | 96.91 | 110.98 |
| MAE ($), served | 52.64 | 64.84 | 74.66 |
| MAE ($), raw heads | 53.58 | 66.87 | 77.72 |
| MAE ($), zero prediction | 52.64 | 64.84 | 74.66 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0396 | -0.0763 | -0.0944 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0373 | -0.0756 | -0.0879 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0559 | -0.0677 | -0.0852 |
| corr, Spearman, raw heads | -0.0050 | -0.0194 | -0.0253 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 3.88 | -2.42 | 9.00 |
| mean realised ($) | 0.03 | 0.05 | 0.07 |
| share predicted up, raw heads | 0.7790 | 0.5905 | 0.8288 |
| share realised up | 0.4976 | 0.5011 | 0.5040 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 39.35 | 48.07 | 55.20 |
| CRPSS vs constant variance | 0.0049 | 0.0121 | 0.0164 |
| NLL | 5.7651 | 6.0371 | 6.1004 |
| PIT KS | 0.0531 | 0.0228 | 0.0276 |
| var / err^2 Spearman | 0.0902 | 0.2133 | 0.2182 |
| coverage of the 90% interval | 0.8932 | 0.8948 | 0.8889 |
| width of the 90% interval ($) | 226.36 | 280.01 | 321.07 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0279 | [-0.0588, -0.0011] | INVERTED |
| h1 | 0.0210 | [-0.0053, 0.0475] | NOISE |
| h2 | -0.0009 | [-0.0224, 0.0197] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5765 | n/a (beta = 0: served delta is 0) | 0.6180 |
| abs(d h1) <= abs(d h2) | 0.7924 | n/a (beta = 0: served delta is 0) | 0.5907 |
| full chain h0 <= h1 <= h2 | 0.3891 | n/a (beta = 0: served delta is 0) | 0.3343 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.3578 | 0.5592 | 0.7260 | 0.0995 |
| expected if the two signs were independent | 0.4459 | 0.4902 | 0.6337 | 0.1056 |

- P(up) unanimity (all three horizons call the same side): 0.1442

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0475 vs 0.0215 (-0.0690): does not beat, significantly worse (boot z -2.34) | 0.0376 vs 0.0397 (-0.0021): does not beat, noise (boot z -0.10) | -0.0213 vs 0.0430 (-0.0644): does not beat, significantly worse (boot z -3.35) |
| logreg_lags | direction/auc | 0.4632 vs 0.5268 (-0.0637): does not beat, significantly worse (boot z -3.34) | 0.5254 vs 0.5395 (-0.0141): does not beat, noise (boot z -1.59) | 0.4881 vs 0.5375 (-0.0494): does not beat, significantly worse (boot z -3.66) |
| logreg_lags | direction/brier | 0.2868 vs 0.2502 (-0.0366): does not beat, significantly worse (DM z -10.28) | 0.2529 vs 0.2495 (-0.0034): does not beat, significantly worse (DM z -2.31) | 0.2610 vs 0.2496 (-0.0114): does not beat, significantly worse (DM z -6.56) |
| logreg_lags | direction/ece_pos | 0.1437 vs 0.0285 (-0.1152): does not beat, significantly worse (boot z -7.60) | 0.0467 vs 0.0266 (-0.0202): does not beat, noise (boot z -1.33) | 0.0806 vs 0.0226 (-0.0581): does not beat, significantly worse (boot z -6.34) |
| logreg_lags | direction/acc | 0.4764 vs 0.5090 (-0.0326): does not beat, significantly worse (DM z -2.08) | 0.5182 vs 0.5195 (-0.0013): does not beat, noise (DM z -0.11) | 0.4936 vs 0.5229 (-0.0294): does not beat, significantly worse (DM z -3.27) |
| logreg_lags | direction/bal_acc | 0.4766 vs 0.5081 (-0.0315): does not beat, significantly worse (boot z -2.45) | 0.5187 vs 0.5165 (+0.0022): beats, noise (boot z +0.23) | 0.4901 vs 0.5181 (-0.0279): does not beat, significantly worse (boot z -3.36) |
| class_prior | direction/mcc | -0.0475 vs 0.0000 (-0.0475): does not beat, significantly worse (boot z -2.74) | 0.0376 vs 0.0000 (+0.0376): beats (boot z +1.97) | -0.0213 vs 0.0000 (-0.0213): does not beat, noise (boot z -1.32) |
| class_prior | direction/auc | 0.4632 vs 0.5000 (-0.0368): does not beat, significantly worse (boot z -3.65) | 0.5254 vs 0.5000 (+0.0254): beats (boot z +2.08) | 0.4881 vs 0.5000 (-0.0119): does not beat, noise (boot z -1.28) |
| class_prior | direction/brier | 0.2868 vs 0.2505 (-0.0363): does not beat, significantly worse (DM z -11.73) | 0.2529 vs 0.2503 (-0.0026): does not beat, noise (DM z -1.46) | 0.2610 vs 0.2501 (-0.0109): does not beat, significantly worse (DM z -6.69) |
| class_prior | direction/ece_pos | 0.1437 vs 0.0224 (-0.1213): does not beat, significantly worse (boot z -8.39) | 0.0467 vs 0.0173 (-0.0294): does not beat, noise (boot z -1.88) | 0.0806 vs 0.0149 (-0.0657): does not beat, significantly worse (boot z -6.39) |
| class_prior | direction/acc | 0.4764 vs 0.5013 (-0.0249): does not beat, noise (DM z -1.70) | 0.5182 vs 0.5054 (+0.0128): beats, noise (DM z +0.81) | 0.4936 vs 0.5090 (-0.0154): does not beat, noise (DM z -1.43) |
| class_prior | direction/bal_acc | 0.4766 vs 0.5000 (-0.0234): does not beat, significantly worse (boot z -2.74) | 0.5187 vs 0.5000 (+0.0187): beats (boot z +1.97) | 0.4901 vs 0.5000 (-0.0099): does not beat, noise (boot z -1.32) |
| zero_delta | delta/rmse | 79.22 vs 79.22 (+0.00, +0.00%): does not beat | 96.91 vs 96.91 (+0.00, +0.00%): does not beat | 110.98 vs 110.98 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 52.64 vs 52.64 (+0.00, +0.00%): does not beat | 64.84 vs 64.84 (+0.00, +0.00%): does not beat | 74.66 vs 74.66 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 79.22 vs 79.26 (+0.04, +0.05%): beats, noise (DM z +0.86) | 96.91 vs 96.99 (+0.07, +0.08%): beats, noise (DM z +0.87) | 110.98 vs 111.09 (+0.11, +0.10%): beats, noise (DM z +0.87) |
| mean_delta | delta/mae | 52.64 vs 52.68 (+0.05, +0.09%): beats, noise (DM z +1.12) | 64.84 vs 64.90 (+0.06, +0.09%): beats, noise (DM z +0.84) | 74.66 vs 74.72 (+0.06, +0.08%): beats, noise (DM z +0.57) |
| const_var | variance/crps | 39.35 vs 39.54 (+0.19, +0.49%): beats (DM z +4.39) | 48.07 vs 48.66 (+0.59, +1.21%): beats (DM z +6.98) | 55.20 vs 56.12 (+0.92, +1.64%): beats (DM z +6.36) |
| const_var | variance/nll | 5.7651 vs 5.8104 (+0.0453): beats (DM z +3.45) | 6.0371 vs 6.0150 (-0.0221): does not beat, noise (DM z -0.91) | 6.1004 vs 6.1490 (+0.0486): beats (DM z +2.32) |
| const_var | variance/pit_ks | 0.0531 vs 0.0591 (+0.0060): beats, noise (boot z +0.82) | 0.0228 vs 0.0617 (+0.0389): beats (boot z +3.71) | 0.0276 vs 0.0671 (+0.0395): beats (boot z +3.62) |
| const_var | variance/corr_var_err2_spearman | 0.0902 vs 0.0000 (+0.0902): beats (boot z +4.59) | 0.2133 vs 0.0000 (+0.2133): beats (boot z +10.04) | 0.2182 vs 0.0000 (+0.2182): beats (boot z +9.86) |

## Backtest (costs included)

- n_trades: 1098
- total_return: 0.0815
- sharpe_net: 5.1657
- sharpe_gross: 5.1657
- sortino: 8.1267
- max_drawdown: 0.0350
- hit_rate: 0.4062
- hit_rate_gross: 0.4062
- profit_factor: 1.1089
- avg_hold_bars: 8.7641
- exposure: 0.3945
- turnover: 2318.2425
- fees_paid: 0.0000
- traded_notional: 23182670.6476
- breakeven_cost_bps: 0.7028
- gross_edge_per_trade_bps: 0.7367
- costs_paid: 0.0000
- gross_pnl: 814.6516
- net_pnl: 814.6516

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -39 (TimeSeriesSplit fold 2, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2023-12-08T14:48:00 .. 2023-12-25T13:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7089, long_above 0.5633, short_below 0.4438, median 0.4924. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +8.15% | +5.17 | +3.50% | 1098 |
| buy and hold | +0.10% | +0.26 | +10.11% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -9.53% .. +10.37%) | -0.03% | +0.01 | | |

The random null enters at the strategy's rate (0.0744 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 89% of its seeds on net return, 88% on net Sharpe and 89% on gross return.
