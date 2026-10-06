# Evaluation report - dev split - run `20260930T225637Z-fb840fd-6fc821bd-control__f-39__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 16776 | 18240 | 18977 |
| n_eff of the scored moves (n scored // bars ahead) | 1677 | 1216 | 948 |
| true up-rate | 0.5013 | 0.5054 | 0.5090 |
| calls up (predicted up-rate) | 0.3918 | 0.5929 | 0.4611 |
| accuracy | 0.4821 | 0.5156 | 0.4993 |
| balanced accuracy | 0.4824 | 0.5146 | 0.5000 |
| precision (up) | 0.4788 | 0.5177 | 0.5090 |
| recall / sensitivity (up) | 0.3742 | 0.6073 | 0.4611 |
| specificity (down) | 0.5906 | 0.4219 | 0.5388 |
| F1 (up) | 0.4201 | 0.5589 | 0.4839 |
| MCC | -0.0361 | 0.0297 | -0.0000 |
| AUC | 0.4678 | 0.5201 | 0.5017 |
| Brier | 0.2746 | 0.2532 | 0.2581 |
| ECE (positive class) | 0.1044 | 0.0406 | 0.0677 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0013 | 0.0054 | 0.0090 |
| TP / FP / TN / FN | 3147 / 3425 / 4941 / 5263 | 5598 / 5216 / 3806 / 3620 | 4454 / 4297 / 5021 / 5205 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.5510 | 0.3325 | 0.3317 |
| Gaussian readout of the raw heads: MCC | -0.0238 | 0.0086 | 0.0044 |
| Gaussian readout of the raw heads: AUC | 0.4862 | 0.5010 | 0.5008 |
| Gaussian readout of the raw heads: Brier | 0.2652 | 0.2851 | 0.2772 |
| Gaussian readout of the raw heads: ECE | 0.0986 | 0.1408 | 0.1283 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 79.22 | 96.91 | 110.98 |
| RMSE ($), raw heads | 82.22 | 104.18 | 117.47 |
| RMSE ($), zero prediction | 79.22 | 96.91 | 110.98 |
| MAE ($), served | 52.64 | 64.84 | 74.66 |
| MAE ($), raw heads | 54.81 | 69.77 | 78.87 |
| MAE ($), zero prediction | 52.64 | 64.84 | 74.66 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0770 | -0.1556 | -0.1204 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0770 | -0.1320 | -0.1090 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0649 | -0.0746 | -0.0718 |
| corr, Spearman, raw heads | -0.0300 | -0.0179 | -0.0121 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -0.30 | -14.82 | -11.80 |
| mean realised ($) | 0.03 | 0.05 | 0.07 |
| share predicted up, raw heads | 0.5587 | 0.3302 | 0.3332 |
| share realised up | 0.4976 | 0.5011 | 0.5040 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 39.32 | 48.26 | 55.46 |
| CRPSS vs constant variance | 0.0057 | 0.0082 | 0.0119 |
| NLL | 5.8061 | 6.1027 | 6.2202 |
| PIT KS | 0.0468 | 0.0269 | 0.0321 |
| var / err^2 Spearman | 0.0937 | 0.1141 | 0.1698 |
| coverage of the 90% interval | 0.8932 | 0.8948 | 0.8889 |
| width of the 90% interval ($) | 226.36 | 280.01 | 321.07 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0308 | [-0.0544, -0.0078] | INVERTED |
| h1 | 0.0212 | [-0.0022, 0.0427] | NOISE |
| h2 | -0.0010 | [-0.0223, 0.0209] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6910 | n/a (beta = 0: served delta is 0) | 0.6180 |
| abs(d h1) <= abs(d h2) | 0.5014 | n/a (beta = 0: served delta is 0) | 0.5907 |
| full chain h0 <= h1 <= h2 | 0.2776 | n/a (beta = 0: served delta is 0) | 0.3343 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5838 | 0.5563 | 0.6513 | 0.2404 |
| expected if the two signs were independent | 0.4869 | 0.4661 | 0.5135 | 0.1517 |

- P(up) unanimity (all three horizons call the same side): 0.3747

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0361 vs 0.0215 (-0.0575): does not beat, significantly worse (boot z -2.40) | 0.0297 vs 0.0397 (-0.0101): does not beat, noise (boot z -0.63) | -0.0000 vs 0.0430 (-0.0431): does not beat, significantly worse (boot z -1.97) |
| logreg_lags | direction/auc | 0.4678 vs 0.5268 (-0.0590): does not beat, significantly worse (boot z -3.76) | 0.5201 vs 0.5395 (-0.0194): does not beat, significantly worse (boot z -2.70) | 0.5017 vs 0.5375 (-0.0358): does not beat, significantly worse (boot z -3.09) |
| logreg_lags | direction/brier | 0.2746 vs 0.2502 (-0.0244): does not beat, significantly worse (DM z -8.42) | 0.2532 vs 0.2495 (-0.0037): does not beat, significantly worse (DM z -3.99) | 0.2581 vs 0.2496 (-0.0085): does not beat, significantly worse (DM z -4.71) |
| logreg_lags | direction/ece_pos | 0.1044 vs 0.0285 (-0.0759): does not beat, significantly worse (boot z -5.30) | 0.0406 vs 0.0266 (-0.0140): does not beat, noise (boot z -1.47) | 0.0677 vs 0.0226 (-0.0452): does not beat, significantly worse (boot z -3.04) |
| logreg_lags | direction/acc | 0.4821 vs 0.5090 (-0.0269): does not beat, noise (DM z -1.91) | 0.5156 vs 0.5195 (-0.0039): does not beat, noise (DM z -0.48) | 0.4993 vs 0.5229 (-0.0237): does not beat, noise (DM z -1.84) |
| logreg_lags | direction/bal_acc | 0.4824 vs 0.5081 (-0.0257): does not beat, significantly worse (boot z -2.50) | 0.5146 vs 0.5165 (-0.0020): does not beat, noise (boot z -0.27) | 0.5000 vs 0.5181 (-0.0181): does not beat, noise (boot z -1.83) |
| class_prior | direction/mcc | -0.0361 vs 0.0000 (-0.0361): does not beat, significantly worse (boot z -2.49) | 0.0297 vs 0.0000 (+0.0297): beats, noise (boot z +1.68) | -0.0000 vs 0.0000 (-0.0000): does not beat, noise (boot z -0.00) |
| class_prior | direction/auc | 0.4678 vs 0.5000 (-0.0322): does not beat, significantly worse (boot z -3.55) | 0.5201 vs 0.5000 (+0.0201): beats, noise (boot z +1.78) | 0.5017 vs 0.5000 (+0.0017): beats, noise (boot z +0.16) |
| class_prior | direction/brier | 0.2746 vs 0.2505 (-0.0241): does not beat, significantly worse (DM z -9.34) | 0.2532 vs 0.2503 (-0.0029): does not beat, significantly worse (DM z -2.18) | 0.2581 vs 0.2501 (-0.0079): does not beat, significantly worse (DM z -4.23) |
| class_prior | direction/ece_pos | 0.1044 vs 0.0224 (-0.0820): does not beat, significantly worse (boot z -5.83) | 0.0406 vs 0.0173 (-0.0232): does not beat, significantly worse (boot z -2.16) | 0.0677 vs 0.0149 (-0.0528): does not beat, significantly worse (boot z -3.44) |
| class_prior | direction/acc | 0.4821 vs 0.5013 (-0.0192): does not beat, noise (DM z -1.35) | 0.5156 vs 0.5054 (+0.0102): beats, noise (DM z +0.81) | 0.4993 vs 0.5090 (-0.0097): does not beat, noise (DM z -0.59) |
| class_prior | direction/bal_acc | 0.4824 vs 0.5000 (-0.0176): does not beat, significantly worse (boot z -2.49) | 0.5146 vs 0.5000 (+0.0146): beats, noise (boot z +1.68) | 0.5000 vs 0.5000 (-0.0000): does not beat, noise (boot z -0.00) |
| zero_delta | delta/rmse | 79.22 vs 79.22 (+0.00, +0.00%): does not beat | 96.91 vs 96.91 (+0.00, +0.00%): does not beat | 110.98 vs 110.98 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 52.64 vs 52.64 (+0.00, +0.00%): does not beat | 64.84 vs 64.84 (+0.00, +0.00%): does not beat | 74.66 vs 74.66 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 79.22 vs 79.26 (+0.04, +0.05%): beats, noise (DM z +0.86) | 96.91 vs 96.99 (+0.07, +0.08%): beats, noise (DM z +0.87) | 110.98 vs 111.09 (+0.11, +0.10%): beats, noise (DM z +0.87) |
| mean_delta | delta/mae | 52.64 vs 52.68 (+0.05, +0.09%): beats, noise (DM z +1.12) | 64.84 vs 64.90 (+0.06, +0.09%): beats, noise (DM z +0.84) | 74.66 vs 74.72 (+0.06, +0.08%): beats, noise (DM z +0.57) |
| const_var | variance/crps | 39.32 vs 39.54 (+0.23, +0.57%): beats (DM z +3.56) | 48.26 vs 48.66 (+0.40, +0.82%): beats (DM z +3.82) | 55.46 vs 56.12 (+0.67, +1.19%): beats (DM z +3.89) |
| const_var | variance/nll | 5.8061 vs 5.8104 (+0.0043): beats, noise (DM z +0.24) | 6.1027 vs 6.0150 (-0.0876): does not beat, significantly worse (DM z -2.43) | 6.2202 vs 6.1490 (-0.0712): does not beat, significantly worse (DM z -2.50) |
| const_var | variance/pit_ks | 0.0468 vs 0.0591 (+0.0124): beats, noise (boot z +1.24) | 0.0269 vs 0.0617 (+0.0348): beats (boot z +3.16) | 0.0321 vs 0.0671 (+0.0350): beats (boot z +3.02) |
| const_var | variance/corr_var_err2_spearman | 0.0937 vs 0.0000 (+0.0937): beats (boot z +5.05) | 0.1141 vs 0.0000 (+0.1141): beats (boot z +5.58) | 0.1698 vs 0.0000 (+0.1698): beats (boot z +7.97) |

## Backtest (costs included)

- n_trades: 1211
- total_return: -0.0619
- sharpe_net: -3.9816
- sharpe_gross: -3.9816
- sortino: -6.0470
- max_drawdown: 0.1084
- hit_rate: 0.4723
- hit_rate_gross: 0.4723
- profit_factor: 0.9176
- avg_hold_bars: 8.7019
- exposure: 0.4321
- turnover: 2313.3249
- fees_paid: 0.0000
- traded_notional: 23133250.5184
- breakeven_cost_bps: -0.5354
- gross_edge_per_trade_bps: -0.5109
- costs_paid: 0.0000
- gross_pnl: -619.2994
- net_pnl: -619.2994

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -39 (TimeSeriesSplit fold 2, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2023-12-08T14:48:00 .. 2023-12-25T13:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6108, long_above 0.5548, short_below 0.4170, median 0.4899. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -6.19% | -3.98 | +10.84% | 1211 |
| buy and hold | +0.10% | +0.26 | +10.11% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -9.02% .. +11.86%) | +0.06% | +0.06 | | |

The random null enters at the strategy's rate (0.0874 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 19% of its seeds on net return, 22% on net Sharpe and 19% on gross return.
