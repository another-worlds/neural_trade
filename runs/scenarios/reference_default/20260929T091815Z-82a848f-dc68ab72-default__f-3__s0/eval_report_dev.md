# Evaluation report - dev split - run `20260929T091815Z-82a848f-dc68ab72-default__f-3__s0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.4211 | 0.4727 | 0.5517 |
| accuracy | 0.5197 | 0.5076 | 0.5044 |
| balanced accuracy | 0.5205 | 0.5078 | 0.5050 |
| precision (up) | 0.5292 | 0.5120 | 0.4989 |
| recall / sensitivity (up) | 0.4413 | 0.4804 | 0.5568 |
| specificity (down) | 0.5996 | 0.5352 | 0.4532 |
| F1 (up) | 0.4813 | 0.4957 | 0.5262 |
| MCC | 0.0414 | 0.0156 | 0.0100 |
| AUC | 0.5252 | 0.5139 | 0.5123 |
| Brier | 0.2504 | 0.2500 | 0.2512 |
| ECE (positive class) | 0.0244 | 0.0150 | 0.0341 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1087 / 967 / 1448 / 1376 | 1276 / 1216 / 1400 / 1380 | 1535 / 1542 / 1278 / 1222 |
| Gaussian readout: calls up | 0.4326 | 0.3765 | 0.3540 |
| Gaussian readout: MCC | 0.0725 | 0.0869 | 0.0849 |
| Gaussian readout: AUC | 0.5635 | 0.5639 | 0.5675 |
| Gaussian readout: Brier | 0.2474 | 0.2471 | 0.2482 |
| Gaussian readout: ECE | 0.0092 | 0.0207 | 0.0290 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 209.96 | 248.46 | 284.59 |
| RMSE ($), raw heads | 209.67 | 249.67 | 284.06 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 136.75 | 166.80 | 194.55 |
| MAE ($), raw heads | 136.72 | 167.81 | 194.85 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0135 | 0.0139 | 0.0091 |
| skill vs zero, raw heads | 0.0163 | 0.0043 | 0.0128 |
| EV, served | 0.0134 | 0.0138 | 0.0088 |
| EV, raw heads | 0.0162 | 0.0051 | 0.0131 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.1344 | 0.1176 | 0.1343 |
| corr, Spearman, raw heads | 0.0921 | 0.0947 | 0.1119 |
| mean predicted ($), served | -1.73 | -6.29 | -2.29 |
| mean predicted ($), raw heads | -2.38 | -11.63 | -12.19 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.4389 | 0.3758 | 0.3622 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.7272 | 0.5410 | 0.1879 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 102.97 | 125.22 | 145.50 |
| CRPSS vs constant variance | 0.0441 | 0.0414 | 0.0383 |
| NLL | 6.7094 | 6.8736 | 7.0267 |
| PIT KS | 0.0644 | 0.0639 | 0.0553 |
| var / err^2 Spearman | 0.2282 | 0.1995 | 0.2079 |
| coverage of the 90% interval | 0.8773 | 0.8704 | 0.8525 |
| width of the 90% interval ($) | 576.94 | 696.01 | 768.41 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0198 | [-0.0230, 0.0829] | NOISE |
| h1 | 0.0149 | [-0.0250, 0.0517] | NOISE |
| h2 | 0.0226 | [-0.0172, 0.0697] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.727 / h1 0.541 / h2 0.188) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7483 | 0.6545 | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.7078 | 0.1317 | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.4982 | 0.0321 | 0.3062 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5633 | 0.4344 | 0.5493 | 0.1434 |
| expected if the two signs were independent | 0.5133 | 0.5010 | 0.4894 | 0.1267 |

- P(up) unanimity (all three horizons call the same side): 0.2521

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0414 vs 0.0474 (-0.0060): does not beat, noise (boot z -0.18) | 0.0156 vs 0.0487 (-0.0331): does not beat, noise (boot z -0.64) | 0.0100 vs 0.0397 (-0.0297): does not beat, noise (boot z -0.71) |
| logreg_lags | direction/auc | 0.5252 vs 0.5348 (-0.0096): does not beat, noise (boot z -0.50) | 0.5139 vs 0.5440 (-0.0301): does not beat, noise (boot z -0.86) | 0.5123 vs 0.5205 (-0.0082): does not beat, noise (boot z -0.33) |
| logreg_lags | direction/brier | 0.2504 vs 0.2494 (-0.0010): does not beat, noise (DM z -0.58) | 0.2500 vs 0.2491 (-0.0009): does not beat, noise (DM z -0.46) | 0.2512 vs 0.2494 (-0.0018): does not beat, noise (DM z -0.75) |
| logreg_lags | direction/ece_pos | 0.0244 vs 0.0274 (+0.0030): beats, noise (boot z +0.22) | 0.0150 vs 0.0284 (+0.0134): beats, noise (boot z +0.94) | 0.0341 vs 0.0161 (-0.0180): does not beat, noise (boot z -0.82) |
| logreg_lags | direction/acc | 0.5197 vs 0.5150 (+0.0047): beats, noise (DM z +0.28) | 0.5076 vs 0.5152 (-0.0076): does not beat, noise (DM z -0.29) | 0.5044 vs 0.5184 (-0.0140): does not beat, noise (DM z -0.50) |
| logreg_lags | direction/bal_acc | 0.5205 vs 0.5181 (+0.0023): beats, noise (boot z +0.16) | 0.5078 vs 0.5178 (-0.0100): does not beat, noise (boot z -0.46) | 0.5050 vs 0.5145 (-0.0096): does not beat, noise (boot z -0.53) |
| class_prior | direction/mcc | 0.0414 vs 0.0000 (+0.0414): beats, noise (boot z +1.40) | 0.0156 vs 0.0000 (+0.0156): beats, noise (boot z +0.57) | 0.0100 vs 0.0000 (+0.0100): beats, noise (boot z +0.32) |
| class_prior | direction/auc | 0.5252 vs 0.5000 (+0.0252): beats, noise (boot z +1.39) | 0.5139 vs 0.5000 (+0.0139): beats, noise (boot z +0.77) | 0.5123 vs 0.5000 (+0.0123): beats, noise (boot z +0.57) |
| class_prior | direction/brier | 0.2504 vs 0.2504 (+0.0000): beats, noise (DM z +0.02) | 0.2500 vs 0.2504 (+0.0003): beats, noise (DM z +0.32) | 0.2512 vs 0.2500 (-0.0011): does not beat, noise (DM z -0.56) |
| class_prior | direction/ece_pos | 0.0244 vs 0.0217 (-0.0027): does not beat, noise (boot z -0.20) | 0.0150 vs 0.0193 (+0.0043): beats, noise (boot z +0.39) | 0.0341 vs 0.0080 (-0.0261): does not beat, noise (boot z -1.31) |
| class_prior | direction/acc | 0.5197 vs 0.4951 (+0.0246): beats, noise (DM z +1.20) | 0.5076 vs 0.4962 (+0.0114): beats, noise (DM z +0.46) | 0.5044 vs 0.5056 (-0.0013): does not beat, noise (DM z -0.04) |
| class_prior | direction/bal_acc | 0.5205 vs 0.5000 (+0.0205): beats, noise (boot z +1.40) | 0.5078 vs 0.5000 (+0.0078): beats, noise (boot z +0.57) | 0.5050 vs 0.5000 (+0.0050): beats, noise (boot z +0.31) |
| zero_delta | delta/rmse | 209.96 vs 211.40 (+1.44, +0.68%): beats (DM z +2.51) | 248.46 vs 250.20 (+1.75, +0.70%): beats, noise (DM z +1.06) | 284.59 vs 285.89 (+1.30, +0.45%): beats (DM z +2.00) |
| zero_delta | delta/mae | 136.75 vs 137.51 (+0.76, +0.56%): beats (DM z +2.15) | 166.80 vs 168.19 (+1.40, +0.83%): beats, noise (DM z +1.60) | 194.55 vs 195.53 (+0.98, +0.50%): beats (DM z +2.59) |
| mean_delta | delta/rmse | 209.96 vs 211.39 (+1.42, +0.67%): beats (DM z +2.53) | 248.46 vs 250.18 (+1.72, +0.69%): beats, noise (DM z +1.05) | 284.59 vs 285.84 (+1.26, +0.44%): beats, noise (DM z +1.91) |
| mean_delta | delta/mae | 136.75 vs 137.52 (+0.77, +0.56%): beats (DM z +2.20) | 166.80 vs 168.27 (+1.47, +0.87%): beats, noise (DM z +1.73) | 194.55 vs 195.56 (+1.01, +0.52%): beats (DM z +2.75) |
| const_var | variance/crps | 102.97 vs 107.72 (+4.75, +4.41%): beats (DM z +10.08) | 125.22 vs 130.63 (+5.41, +4.14%): beats (DM z +6.73) | 145.50 vs 151.29 (+5.79, +3.83%): beats (DM z +7.12) |
| const_var | variance/nll | 6.7094 vs 6.7814 (+0.0720): beats (DM z +2.33) | 6.8736 vs 6.9539 (+0.0804): beats (DM z +3.35) | 7.0267 vs 7.0883 (+0.0616): beats (DM z +2.49) |
| const_var | variance/pit_ks | 0.0644 vs 0.1058 (+0.0413): beats (boot z +6.95) | 0.0639 vs 0.1039 (+0.0400): beats (boot z +5.98) | 0.0553 vs 0.1039 (+0.0486): beats (boot z +6.69) |
| const_var | variance/corr_var_err2_spearman | 0.2282 vs 0.0000 (+0.2282): beats (boot z +6.62) | 0.1995 vs 0.0000 (+0.1995): beats (boot z +5.21) | 0.2079 vs 0.0000 (+0.2079): beats (boot z +5.28) |

## Backtest (costs included)

- n_trades: 308
- total_return: -0.5330
- sharpe_net: -127.9207
- sharpe_gross: 9.1020
- sortino: -151.9772
- max_drawdown: 0.5330
- hit_rate: 0.0714
- hit_rate_gross: 0.5455
- profit_factor: 0.0590
- avg_hold_bars: 11.0714
- exposure: 0.4713
- turnover: 432.8480
- fees_paid: 4328.6918
- costs_paid: 5627.2993
- gross_pnl: 296.9074
- net_pnl: -5330.3919

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -3 (TimeSeriesSplit fold 3, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-10-26T05:22:00+00:00 .. 2025-10-31T05:57:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.5168, long_above 0.5171, short_below 0.4495, median 0.4764. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -53.30% | -127.92 | +53.30% | 308 |
| buy and hold | -1.69% | -2.62 | +8.46% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -58.68% .. -52.53%) | -55.47% | -147.40 | | |

The random null enters at the strategy's rate (0.0805 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 89% of its seeds on net return, 99% on net Sharpe and 90% on gross return.
