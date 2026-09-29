# Evaluation report - dev split - run `20260929T130846Z-f5aee70-dirty-46600bfc-h_15m_L240__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.4106 | 0.4931 | 0.5605 |
| accuracy | 0.4970 | 0.5023 | 0.5023 |
| balanced accuracy | 0.4952 | 0.5022 | 0.5038 |
| precision (up) | 0.4838 | 0.4918 | 0.4912 |
| recall / sensitivity (up) | 0.4057 | 0.4953 | 0.5643 |
| specificity (down) | 0.5847 | 0.5090 | 0.4432 |
| F1 (up) | 0.4414 | 0.4935 | 0.5253 |
| MCC | -0.0098 | 0.0043 | 0.0076 |
| AUC | 0.4922 | 0.5035 | 0.5084 |
| Brier | 0.2639 | 0.2535 | 0.2579 |
| ECE (positive class) | 0.0828 | 0.0422 | 0.0672 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 5296 / 5650 / 7953 / 7757 | 7214 / 7456 / 7730 / 7351 | 8648 / 8957 / 7131 / 6676 |
| Gaussian readout: calls up | 0.6805 | 0.5926 | 0.5868 |
| Gaussian readout: MCC | -0.0087 | -0.0056 | -0.0070 |
| Gaussian readout: AUC | 0.4984 | 0.4981 | 0.5008 |
| Gaussian readout: Brier | 0.2501 | 0.2502 | 0.2501 |
| Gaussian readout: ECE | 0.0127 | 0.0134 | 0.0151 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.10 | 199.84 | 230.21 |
| RMSE ($), raw heads | 168.98 | 216.75 | 244.01 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.33 | 137.95 | 158.90 |
| MAE ($), raw heads | 115.54 | 150.56 | 170.18 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | -0.0008 | -0.0004 |
| skill vs zero, raw heads | -0.0476 | -0.1773 | -0.1240 |
| EV, served | -0.0000 | -0.0006 | -0.0002 |
| EV, raw heads | -0.0395 | -0.1633 | -0.1047 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0033 | -0.0044 | -0.0006 |
| corr, Spearman, raw heads | -0.0086 | -0.0169 | -0.0153 |
| mean predicted ($), served | 0.47 | 1.13 | 1.28 |
| mean predicted ($), raw heads | 13.36 | 21.37 | 28.98 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.6596 | 0.5603 | 0.5572 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.0354 | 0.0527 | 0.0441 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 83.40 | 101.63 | 117.48 |
| CRPSS vs constant variance | 0.0091 | 0.0127 | 0.0122 |
| NLL | 6.6482 | 6.7981 | 6.9561 |
| PIT KS | 0.0298 | 0.0262 | 0.0263 |
| var / err^2 Spearman | 0.2201 | 0.2431 | 0.2257 |
| coverage of the 90% interval | 0.9092 | 0.9112 | 0.9059 |
| width of the 90% interval ($) | 518.48 | 637.69 | 722.94 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0149 | [-0.0358, 0.0064] | NOISE |
| h1 | 0.0027 | [-0.0166, 0.0200] | NOISE |
| h2 | 0.0098 | [-0.0098, 0.0304] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.035 / h1 0.053 / h2 0.044) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8695 | 0.9173 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.7498 | 0.4842 | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.6505 | 0.4253 | 0.3339 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6098 | 0.5215 | 0.6673 | 0.2649 |
| expected if the two signs were independent | 0.4683 | 0.4996 | 0.5053 | 0.1637 |

- P(up) unanimity (all three horizons call the same side): 0.3825

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0098 vs -0.0066 (-0.0032): does not beat, noise (boot z -0.14) | 0.0043 vs -0.0065 (+0.0108): beats, noise (boot z +0.53) | 0.0076 vs -0.0070 (+0.0146): beats, noise (boot z +0.66) |
| logreg_lags | direction/auc | 0.4922 vs 0.5043 (-0.0120): does not beat, noise (boot z -0.84) | 0.5035 vs 0.5122 (-0.0087): does not beat, noise (boot z -0.71) | 0.5084 vs 0.5173 (-0.0089): does not beat, noise (boot z -0.70) |
| logreg_lags | direction/brier | 0.2639 vs 0.2519 (-0.0120): does not beat, significantly worse (DM z -5.83) | 0.2535 vs 0.2521 (-0.0014): does not beat, noise (DM z -1.21) | 0.2579 vs 0.2525 (-0.0054): does not beat, significantly worse (DM z -3.46) |
| logreg_lags | direction/ece_pos | 0.0828 vs 0.0414 (-0.0415): does not beat, significantly worse (boot z -3.68) | 0.0422 vs 0.0454 (+0.0033): beats, noise (boot z +0.30) | 0.0672 vs 0.0510 (-0.0162): does not beat, noise (boot z -1.52) |
| logreg_lags | direction/acc | 0.4970 vs 0.4894 (+0.0077): beats, noise (DM z +0.68) | 0.5023 vs 0.4892 (+0.0131): beats, noise (DM z +1.24) | 0.5023 vs 0.4875 (+0.0148): beats, noise (DM z +1.38) |
| logreg_lags | direction/bal_acc | 0.4952 vs 0.4984 (-0.0032): does not beat, noise (boot z -0.37) | 0.5022 vs 0.4986 (+0.0036): beats, noise (boot z +0.54) | 0.5038 vs 0.4985 (+0.0053): beats, noise (boot z +0.71) |
| class_prior | direction/mcc | -0.0098 vs 0.0000 (-0.0098): does not beat, noise (boot z -0.66) | 0.0043 vs 0.0000 (+0.0043): beats, noise (boot z +0.37) | 0.0076 vs 0.0000 (+0.0076): beats, noise (boot z +0.54) |
| class_prior | direction/auc | 0.4922 vs 0.5000 (-0.0078): does not beat, noise (boot z -0.83) | 0.5035 vs 0.5000 (+0.0035): beats, noise (boot z +0.45) | 0.5084 vs 0.5000 (+0.0084): beats, noise (boot z +0.88) |
| class_prior | direction/brier | 0.2639 vs 0.2509 (-0.0129): does not beat, significantly worse (DM z -6.72) | 0.2535 vs 0.2512 (-0.0023): does not beat, significantly worse (DM z -2.01) | 0.2579 vs 0.2514 (-0.0065): does not beat, significantly worse (DM z -4.03) |
| class_prior | direction/ece_pos | 0.0828 vs 0.0324 (-0.0504): does not beat, significantly worse (boot z -4.32) | 0.0422 vs 0.0367 (-0.0055): does not beat, noise (boot z -0.50) | 0.0672 vs 0.0394 (-0.0278): does not beat, significantly worse (boot z -2.47) |
| class_prior | direction/acc | 0.4970 vs 0.4897 (+0.0074): beats, noise (DM z +0.65) | 0.5023 vs 0.4896 (+0.0127): beats, noise (DM z +1.15) | 0.5023 vs 0.4878 (+0.0145): beats, noise (DM z +1.28) |
| class_prior | direction/bal_acc | 0.4952 vs 0.5000 (-0.0048): does not beat, noise (boot z -0.66) | 0.5022 vs 0.5000 (+0.0022): beats, noise (boot z +0.37) | 0.5038 vs 0.5000 (+0.0038): beats, noise (boot z +0.54) |
| zero_delta | delta/rmse | 165.10 vs 165.09 (-0.01, -0.00%): does not beat, noise (DM z -0.27) | 199.84 vs 199.76 (-0.08, -0.04%): does not beat, noise (DM z -0.85) | 230.21 vs 230.16 (-0.05, -0.02%): does not beat, noise (DM z -0.54) |
| zero_delta | delta/mae | 112.33 vs 112.32 (-0.01, -0.01%): does not beat, noise (DM z -0.57) | 137.95 vs 137.87 (-0.08, -0.06%): does not beat, noise (DM z -1.34) | 158.90 vs 158.85 (-0.05, -0.03%): does not beat, noise (DM z -0.84) |
| mean_delta | delta/rmse | 165.10 vs 165.43 (+0.33, +0.20%): beats (DM z +2.85) | 199.84 vs 200.38 (+0.54, +0.27%): beats (DM z +2.52) | 230.21 vs 231.10 (+0.89, +0.39%): beats (DM z +2.83) |
| mean_delta | delta/mae | 112.33 vs 112.78 (+0.45, +0.40%): beats (DM z +4.38) | 137.95 vs 138.64 (+0.69, +0.50%): beats (DM z +3.65) | 158.90 vs 160.00 (+1.10, +0.69%): beats (DM z +3.90) |
| const_var | variance/crps | 83.40 vs 84.16 (+0.76, +0.91%): beats (DM z +5.58) | 101.63 vs 102.94 (+1.31, +1.27%): beats (DM z +5.62) | 117.48 vs 118.93 (+1.45, +1.22%): beats (DM z +4.43) |
| const_var | variance/nll | 6.6482 vs 6.5675 (-0.0806): does not beat, noise (DM z -1.96) | 6.7981 vs 6.7502 (-0.0479): does not beat, noise (DM z -1.20) | 6.9561 vs 6.8870 (-0.0690): does not beat, noise (DM z -1.44) |
| const_var | variance/pit_ks | 0.0298 vs 0.0615 (+0.0317): beats (boot z +4.39) | 0.0262 vs 0.0661 (+0.0399): beats (boot z +5.11) | 0.0263 vs 0.0738 (+0.0475): beats (boot z +5.82) |
| const_var | variance/corr_var_err2_spearman | 0.2201 vs 0.0000 (+0.2201): beats (boot z +15.19) | 0.2431 vs 0.0000 (+0.2431): beats (boot z +15.83) | 0.2257 vs 0.0000 (+0.2257): beats (boot z +13.52) |

## Backtest (costs included)

- n_trades: 1437
- total_return: -0.9765
- sharpe_net: -133.2508
- sharpe_gross: -4.0550
- sortino: -149.6540
- max_drawdown: 0.9765
- hit_rate: 0.0654
- hit_rate_gross: 0.4823
- profit_factor: 0.0304
- avg_hold_bars: 10.8232
- exposure: 0.3600
- turnover: 729.5019
- fees_paid: 7295.1293
- traded_notional: 7295129.2877
- breakeven_cost_bps: -0.7709
- gross_edge_per_trade_bps: -0.0477
- costs_paid: 9483.6681
- gross_pnl: -281.1915
- net_pnl: -9764.8596

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6788, long_above 0.5642, short_below 0.4354, median 0.4945. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.65% | -133.25 | +97.65% | 1437 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.94% .. -97.17%) | -97.56% | -145.76 | | |

The random null enters at the strategy's rate (0.0520 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 37% of its seeds on net return, 100% on net Sharpe and 6% on gross return.
