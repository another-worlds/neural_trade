# Evaluation report - dev split - run `20260930T095943Z-dce15ed-72410bdc-default__f-2__s1`

n = 46544 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 28800 | 32081 | 33877 |
| n_eff of the scored moves (n scored // bars ahead) | 2880 | 2138 | 1693 |
| true up-rate | 0.4906 | 0.4906 | 0.4890 |
| calls up (predicted up-rate) | 0.6050 | 0.6059 | 0.5444 |
| accuracy | 0.5056 | 0.5204 | 0.5146 |
| balanced accuracy | 0.5076 | 0.5224 | 0.5156 |
| precision (up) | 0.4968 | 0.5091 | 0.5033 |
| recall / sensitivity (up) | 0.6128 | 0.6287 | 0.5603 |
| specificity (down) | 0.4025 | 0.4160 | 0.4709 |
| F1 (up) | 0.5487 | 0.5626 | 0.5303 |
| MCC | 0.0156 | 0.0458 | 0.0313 |
| AUC | 0.5141 | 0.5284 | 0.5189 |
| Brier | 0.2500 | 0.2496 | 0.2499 |
| ECE (positive class) | 0.0157 | 0.0124 | 0.0126 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0094 | 0.0094 | 0.0110 |
| TP / FP / TN / FN | 8657 / 8767 / 5905 / 5471 | 9895 / 9543 / 6799 / 5844 | 9282 / 9160 / 8152 / 7283 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.1566 | 0.2331 | 0.3148 |
| Gaussian readout of the raw heads: MCC | 0.0082 | 0.0138 | 0.0156 |
| Gaussian readout of the raw heads: AUC | 0.5066 | 0.5102 | 0.5108 |
| Gaussian readout of the raw heads: Brier | 0.2503 | 0.2511 | 0.2509 |
| Gaussian readout of the raw heads: ECE | 0.0131 | 0.0258 | 0.0250 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 164.21 | 198.38 | 228.36 |
| RMSE ($), raw heads | 164.20 | 237.17 | 228.62 |
| RMSE ($), zero prediction | 164.21 | 198.38 | 228.36 |
| MAE ($), served | 112.64 | 138.04 | 158.99 |
| MAE ($), raw heads | 112.65 | 141.04 | 159.27 |
| MAE ($), zero prediction | 112.64 | 138.04 | 158.99 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | 0.0002 | -0.4293 | -0.0023 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | 0.0004 | -0.4277 | -0.0022 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0257 | -0.0207 | 0.0336 |
| corr, Spearman, raw heads | 0.0090 | 0.0146 | 0.0092 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -4.02 | -9.81 | -5.31 |
| mean realised ($) | -1.18 | -1.76 | -2.32 |
| share predicted up, raw heads | 0.1301 | 0.2093 | 0.3029 |
| share realised up | 0.4897 | 0.4876 | 0.4888 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 81.74 | 100.51 | 115.07 |
| CRPSS vs constant variance | 0.0496 | 0.0409 | 0.0472 |
| NLL | 6.3863 | 6.5947 | 6.7454 |
| PIT KS | 0.0316 | 0.0342 | 0.0201 |
| var / err^2 Spearman | 0.3443 | 0.3360 | 0.3360 |
| coverage of the 90% interval | 0.9070 | 0.9077 | 0.9079 |
| width of the 90% interval ($) | 518.40 | 634.62 | 729.08 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0048 | [-0.0140, 0.0227] | NOISE |
| h1 | 0.0113 | [-0.0100, 0.0342] | NOISE |
| h2 | 0.0085 | [-0.0133, 0.0291] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8822 | n/a (beta = 0: served delta is 0) | 0.6180 |
| abs(d h1) <= abs(d h2) | 0.5829 | n/a (beta = 0: served delta is 0) | 0.5923 |
| full chain h0 <= h1 <= h2 | 0.5016 | n/a (beta = 0: served delta is 0) | 0.3360 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.3974 | 0.4827 | 0.5682 | 0.1502 |
| expected if the two signs were independent | 0.4045 | 0.4399 | 0.4931 | 0.1209 |

- P(up) unanimity (all three horizons call the same side): 0.3686

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0156 vs 0.0344 (-0.0188): does not beat, noise (boot z -1.39) | 0.0458 vs 0.0378 (+0.0080): beats, noise (boot z +0.64) | 0.0313 vs 0.0411 (-0.0098): does not beat, noise (boot z -0.64) |
| logreg_lags | direction/auc | 0.5141 vs 0.5243 (-0.0102): does not beat, noise (boot z -1.21) | 0.5284 vs 0.5272 (+0.0012): beats, noise (boot z +0.17) | 0.5189 vs 0.5314 (-0.0125): does not beat, noise (boot z -1.46) |
| logreg_lags | direction/brier | 0.2500 vs 0.2497 (-0.0003): does not beat, noise (DM z -1.03) | 0.2496 vs 0.2497 (+0.0001): beats, noise (DM z +0.64) | 0.2499 vs 0.2497 (-0.0003): does not beat, noise (DM z -1.16) |
| logreg_lags | direction/ece_pos | 0.0157 vs 0.0112 (-0.0045): does not beat, noise (boot z -0.85) | 0.0124 vs 0.0116 (-0.0008): does not beat, noise (boot z -0.22) | 0.0126 vs 0.0137 (+0.0011): beats, noise (boot z +0.27) |
| logreg_lags | direction/acc | 0.5056 vs 0.5165 (-0.0109): does not beat, noise (DM z -1.58) | 0.5204 vs 0.5178 (+0.0026): beats, noise (DM z +0.42) | 0.5146 vs 0.5191 (-0.0045): does not beat, noise (DM z -0.63) |
| logreg_lags | direction/bal_acc | 0.5076 vs 0.5171 (-0.0095): does not beat, noise (boot z -1.42) | 0.5224 vs 0.5188 (+0.0036): beats, noise (boot z +0.58) | 0.5156 vs 0.5204 (-0.0048): does not beat, noise (boot z -0.63) |
| class_prior | direction/mcc | 0.0156 vs 0.0000 (+0.0156): beats, noise (boot z +1.30) | 0.0458 vs 0.0000 (+0.0458): beats (boot z +3.36) | 0.0313 vs 0.0000 (+0.0313): beats (boot z +2.09) |
| class_prior | direction/auc | 0.5141 vs 0.5000 (+0.0141): beats, noise (boot z +1.87) | 0.5284 vs 0.5000 (+0.0284): beats (boot z +3.17) | 0.5189 vs 0.5000 (+0.0189): beats, noise (boot z +1.95) |
| class_prior | direction/brier | 0.2500 vs 0.2501 (+0.0001): beats, noise (DM z +0.26) | 0.2496 vs 0.2501 (+0.0005): beats (DM z +2.27) | 0.2499 vs 0.2501 (+0.0002): beats, noise (DM z +1.13) |
| class_prior | direction/ece_pos | 0.0157 vs 0.0123 (-0.0034): does not beat, noise (boot z -1.09) | 0.0124 vs 0.0134 (+0.0010): beats, noise (boot z +0.18) | 0.0126 vs 0.0164 (+0.0038): beats, noise (boot z +0.67) |
| class_prior | direction/acc | 0.5056 vs 0.4906 (+0.0151): beats, noise (DM z +1.72) | 0.5204 vs 0.4906 (+0.0298): beats (DM z +3.12) | 0.5146 vs 0.4890 (+0.0257): beats (DM z +2.19) |
| class_prior | direction/bal_acc | 0.5076 vs 0.5000 (+0.0076): beats, noise (boot z +1.30) | 0.5224 vs 0.5000 (+0.0224): beats (boot z +3.36) | 0.5156 vs 0.5000 (+0.0156): beats (boot z +2.09) |
| zero_delta | delta/rmse | 164.21 vs 164.21 (+0.00, +0.00%): does not beat | 198.38 vs 198.38 (+0.00, +0.00%): does not beat | 228.36 vs 228.36 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 112.64 vs 112.64 (+0.00, +0.00%): does not beat | 138.04 vs 138.04 (+0.00, +0.00%): does not beat | 158.99 vs 158.99 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 164.21 vs 164.22 (+0.01, +0.00%): beats, noise (DM z +0.73) | 198.38 vs 198.40 (+0.01, +0.01%): beats, noise (DM z +0.73) | 228.36 vs 228.38 (+0.02, +0.01%): beats, noise (DM z +0.72) |
| mean_delta | delta/mae | 112.64 vs 112.65 (+0.02, +0.01%): beats, noise (DM z +1.78) | 138.04 vs 138.07 (+0.03, +0.02%): beats, noise (DM z +1.83) | 158.99 vs 159.03 (+0.04, +0.02%): beats, noise (DM z +1.51) |
| const_var | variance/crps | 81.74 vs 86.01 (+4.26, +4.96%): beats (DM z +23.01) | 100.51 vs 104.80 (+4.29, +4.09%): beats (DM z +12.06) | 115.07 vs 120.77 (+5.70, +4.72%): beats (DM z +15.13) |
| const_var | variance/nll | 6.3863 vs 6.5314 (+0.1450): beats (DM z +9.36) | 6.5947 vs 6.7225 (+0.1278): beats (DM z +7.60) | 6.7454 vs 6.8636 (+0.1182): beats (DM z +5.93) |
| const_var | variance/pit_ks | 0.0316 vs 0.0915 (+0.0599): beats (boot z +17.59) | 0.0342 vs 0.0894 (+0.0552): beats (boot z +14.02) | 0.0201 vs 0.0910 (+0.0709): beats (boot z +16.93) |
| const_var | variance/corr_var_err2_spearman | 0.3443 vs 0.0000 (+0.3443): beats (boot z +26.20) | 0.3360 vs 0.0000 (+0.3360): beats (boot z +23.31) | 0.3360 vs 0.0000 (+0.3360): beats (boot z +22.02) |

## Backtest (costs included)

- n_trades: 1559
- total_return: -0.9816
- sharpe_net: -136.0386
- sharpe_gross: 7.2409
- sortino: -149.8917
- max_drawdown: 0.9816
- hit_rate: 0.0257
- hit_rate_gross: 0.5670
- profit_factor: 0.0079
- avg_hold_bars: 9.9250
- exposure: 0.3325
- turnover: 791.0268
- fees_paid: 7910.3692
- traded_notional: 7910369.2113
- breakeven_cost_bps: 1.1808
- gross_edge_per_trade_bps: 0.4029
- costs_paid: 10283.4800
- gross_pnl: 467.0318
- net_pnl: -9816.4481

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 13, 13 usable folds); this report scores the fold's out-of-sample block: 46544 sequences, 2025-07-27T08:10:00 .. 2025-08-28T15:53:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4003, long_above 0.5214, short_below 0.4897, median 0.5037. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.16% | -136.04 | +98.16% | 1559 |
| buy and hold | -5.10% | -1.62 | +12.57% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.46% .. -97.99%) | -98.23% | -150.40 | | |

The random null enters at the strategy's rate (0.0502 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 65% of its seeds on net return, 100% on net Sharpe and 100% on gross return.
