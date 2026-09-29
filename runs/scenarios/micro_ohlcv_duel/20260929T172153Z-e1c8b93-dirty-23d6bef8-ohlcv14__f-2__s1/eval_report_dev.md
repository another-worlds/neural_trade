# Evaluation report - dev split - run `20260929T172153Z-e1c8b93-dirty-23d6bef8-ohlcv14__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.5592 | 0.6530 | 0.4880 |
| accuracy | 0.4975 | 0.4911 | 0.5119 |
| balanced accuracy | 0.4987 | 0.4943 | 0.5116 |
| precision (up) | 0.4885 | 0.4852 | 0.4997 |
| recall / sensitivity (up) | 0.5579 | 0.6471 | 0.4999 |
| specificity (down) | 0.4395 | 0.3414 | 0.5234 |
| F1 (up) | 0.5209 | 0.5546 | 0.4998 |
| MCC | -0.0026 | -0.0120 | 0.0232 |
| AUC | 0.5024 | 0.4955 | 0.5164 |
| Brier | 0.2588 | 0.2588 | 0.2564 |
| ECE (positive class) | 0.0754 | 0.0795 | 0.0522 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 7282 / 7624 / 5979 / 5771 | 9425 / 10001 / 5185 / 5140 | 7660 / 7668 / 8420 / 7664 |
| Gaussian readout: calls up | 0.6291 | 0.6167 | 0.6145 |
| Gaussian readout: MCC | 0.0239 | 0.0071 | 0.0263 |
| Gaussian readout: AUC | 0.5188 | 0.5073 | 0.5205 |
| Gaussian readout: Brier | 0.2500 | 0.2502 | 0.2498 |
| Gaussian readout: ECE | 0.0111 | 0.0154 | 0.0162 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.09 | 199.83 | 230.19 |
| RMSE ($), raw heads | 166.53 | 203.21 | 236.32 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.31 | 137.91 | 158.79 |
| MAE ($), raw heads | 113.51 | 140.87 | 163.92 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0001 | -0.0007 | -0.0002 |
| skill vs zero, raw heads | -0.0175 | -0.0348 | -0.0543 |
| EV, served | 0.0001 | -0.0005 | 0.0001 |
| EV, raw heads | -0.0156 | -0.0311 | -0.0460 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0137 | -0.0001 | 0.0171 |
| corr, Spearman, raw heads | 0.0230 | 0.0069 | 0.0242 |
| mean predicted ($), served | 0.14 | 1.24 | 2.29 |
| mean predicted ($), raw heads | 5.79 | 10.08 | 17.97 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.6096 | 0.6042 | 0.6011 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.0242 | 0.1235 | 0.1277 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 82.71 | 101.08 | 116.52 |
| CRPSS vs constant variance | 0.0172 | 0.0178 | 0.0198 |
| NLL | 6.5245 | 6.6875 | 6.8393 |
| PIT KS | 0.0293 | 0.0242 | 0.0249 |
| var / err^2 Spearman | 0.2864 | 0.2679 | 0.2652 |
| coverage of the 90% interval | 0.9062 | 0.9039 | 0.8976 |
| width of the 90% interval ($) | 513.81 | 622.55 | 697.97 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0102 | [-0.0090, 0.0292] | NOISE |
| h1 | 0.0006 | [-0.0186, 0.0180] | NOISE |
| h2 | 0.0186 | [0.0006, 0.0362] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.024 / h1 0.124 / h2 0.128) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7638 | 0.9666 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.6655 | 0.6787 | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.4828 | 0.6461 | 0.3339 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6277 | 0.5692 | 0.5916 | 0.2738 |
| expected if the two signs were independent | 0.5174 | 0.5355 | 0.4979 | 0.1948 |

- P(up) unanimity (all three horizons call the same side): 0.4118

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0026 vs 0.0011 (-0.0037): does not beat, noise (boot z -0.18) | -0.0120 vs 0.0027 (-0.0147): does not beat, noise (boot z -0.70) | 0.0232 vs -0.0018 (+0.0251): beats, noise (boot z +1.32) |
| logreg_lags | direction/auc | 0.5024 vs 0.5085 (-0.0062): does not beat, noise (boot z -0.47) | 0.4955 vs 0.5137 (-0.0182): does not beat, noise (boot z -1.28) | 0.5164 vs 0.5174 (-0.0010): does not beat, noise (boot z -0.10) |
| logreg_lags | direction/brier | 0.2588 vs 0.2513 (-0.0075): does not beat, significantly worse (DM z -4.99) | 0.2588 vs 0.2517 (-0.0071): does not beat, significantly worse (DM z -4.73) | 0.2564 vs 0.2518 (-0.0046): does not beat, significantly worse (DM z -3.36) |
| logreg_lags | direction/ece_pos | 0.0754 vs 0.0347 (-0.0407): does not beat, significantly worse (boot z -4.56) | 0.0795 vs 0.0387 (-0.0408): does not beat, significantly worse (boot z -5.12) | 0.0522 vs 0.0432 (-0.0090): does not beat, noise (boot z -0.84) |
| logreg_lags | direction/acc | 0.4975 vs 0.4921 (+0.0054): beats, noise (DM z +0.60) | 0.4911 vs 0.4923 (-0.0012): does not beat, noise (DM z -0.14) | 0.5119 vs 0.4895 (+0.0224): beats (DM z +2.06) |
| logreg_lags | direction/bal_acc | 0.4987 vs 0.5003 (-0.0016): does not beat, noise (boot z -0.20) | 0.4943 vs 0.5008 (-0.0065): does not beat, noise (boot z -0.85) | 0.5116 vs 0.4995 (+0.0121): beats, noise (boot z +1.73) |
| class_prior | direction/mcc | -0.0026 vs 0.0000 (-0.0026): does not beat, noise (boot z -0.20) | -0.0120 vs 0.0000 (-0.0120): does not beat, noise (boot z -1.04) | 0.0232 vs 0.0000 (+0.0232): beats, noise (boot z +1.96) |
| class_prior | direction/auc | 0.5024 vs 0.5000 (+0.0024): beats, noise (boot z +0.29) | 0.4955 vs 0.5000 (-0.0045): does not beat, noise (boot z -0.58) | 0.5164 vs 0.5000 (+0.0164): beats (boot z +2.16) |
| class_prior | direction/brier | 0.2588 vs 0.2508 (-0.0080): does not beat, significantly worse (DM z -5.81) | 0.2588 vs 0.2510 (-0.0078): does not beat, significantly worse (DM z -6.08) | 0.2564 vs 0.2512 (-0.0052): does not beat, significantly worse (DM z -3.31) |
| class_prior | direction/ece_pos | 0.0754 vs 0.0297 (-0.0457): does not beat, significantly worse (boot z -5.08) | 0.0795 vs 0.0335 (-0.0460): does not beat, significantly worse (boot z -5.85) | 0.0522 vs 0.0371 (-0.0151): does not beat, noise (boot z -1.35) |
| class_prior | direction/acc | 0.4975 vs 0.4897 (+0.0078): beats, noise (DM z +0.84) | 0.4911 vs 0.4896 (+0.0015): beats, noise (DM z +0.18) | 0.5119 vs 0.4878 (+0.0241): beats (DM z +2.02) |
| class_prior | direction/bal_acc | 0.4987 vs 0.5000 (-0.0013): does not beat, noise (boot z -0.20) | 0.4943 vs 0.5000 (-0.0057): does not beat, noise (boot z -1.04) | 0.5116 vs 0.5000 (+0.0116): beats, noise (boot z +1.96) |
| zero_delta | delta/rmse | 165.09 vs 165.09 (+0.01, +0.00%): beats, noise (DM z +0.67) | 199.83 vs 199.76 (-0.07, -0.03%): does not beat, noise (DM z -0.94) | 230.19 vs 230.16 (-0.03, -0.01%): does not beat, noise (DM z -0.21) |
| zero_delta | delta/mae | 112.31 vs 112.32 (+0.01, +0.01%): beats, noise (DM z +1.73) | 137.91 vs 137.87 (-0.04, -0.03%): does not beat, noise (DM z -0.68) | 158.79 vs 158.85 (+0.06, +0.04%): beats, noise (DM z +0.60) |
| mean_delta | delta/rmse | 165.09 vs 165.39 (+0.30, +0.18%): beats (DM z +2.67) | 199.83 vs 200.31 (+0.48, +0.24%): beats (DM z +2.44) | 230.19 vs 231.00 (+0.81, +0.35%): beats (DM z +2.93) |
| mean_delta | delta/mae | 112.31 vs 112.73 (+0.42, +0.37%): beats (DM z +4.25) | 137.91 vs 138.56 (+0.65, +0.47%): beats (DM z +3.80) | 158.79 vs 159.88 (+1.09, +0.68%): beats (DM z +4.23) |
| const_var | variance/crps | 82.71 vs 84.15 (+1.45, +1.72%): beats (DM z +10.87) | 101.08 vs 102.91 (+1.83, +1.78%): beats (DM z +9.41) | 116.52 vs 118.88 (+2.35, +1.98%): beats (DM z +9.67) |
| const_var | variance/nll | 6.5245 vs 6.5630 (+0.0385): beats (DM z +2.00) | 6.6875 vs 6.7470 (+0.0595): beats (DM z +3.33) | 6.8393 vs 6.8845 (+0.0452): beats (DM z +3.67) |
| const_var | variance/pit_ks | 0.0293 vs 0.0612 (+0.0319): beats (boot z +4.72) | 0.0242 vs 0.0652 (+0.0411): beats (boot z +5.78) | 0.0249 vs 0.0725 (+0.0477): beats (boot z +6.34) |
| const_var | variance/corr_var_err2_spearman | 0.2864 vs 0.0000 (+0.2864): beats (boot z +21.21) | 0.2679 vs 0.0000 (+0.2679): beats (boot z +17.78) | 0.2652 vs 0.0000 (+0.2652): beats (boot z +17.32) |

## Backtest (costs included)

- n_trades: 1901
- total_return: -0.9925
- sharpe_net: -160.4063
- sharpe_gross: -1.3101
- sortino: -176.2148
- max_drawdown: 0.9925
- hit_rate: 0.0284
- hit_rate_gross: 0.5476
- profit_factor: 0.0187
- avg_hold_bars: 8.7175
- exposure: 0.3836
- turnover: 756.7642
- fees_paid: 7567.8888
- traded_notional: 7567888.8233
- breakeven_cost_bps: -0.2302
- gross_edge_per_trade_bps: 0.2817
- costs_paid: 9838.2555
- gross_pnl: -87.1092
- net_pnl: -9925.3647

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6713, long_above 0.5858, short_below 0.4291, median 0.5163. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.25% | -160.41 | +99.25% | 1901 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.36% .. -99.13%) | -99.24% | -175.09 | | |

The random null enters at the strategy's rate (0.0714 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 46% of its seeds on net return, 100% on net Sharpe and 33% on gross return.
