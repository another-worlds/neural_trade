# Evaluation report - test split - run `20260929T093835Z-82a848f-fe691920-default__f-1__s0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.2382 | 0.2961 | 0.2559 |
| accuracy | 0.5113 | 0.5020 | 0.5061 |
| balanced accuracy | 0.5186 | 0.5080 | 0.5154 |
| precision (up) | 0.5530 | 0.5282 | 0.5489 |
| recall / sensitivity (up) | 0.2563 | 0.3039 | 0.2707 |
| specificity (down) | 0.7810 | 0.7121 | 0.7601 |
| F1 (up) | 0.3502 | 0.3858 | 0.3626 |
| MCC | 0.0437 | 0.0175 | 0.0352 |
| AUC | 0.5201 | 0.5161 | 0.5262 |
| Brier | 0.2518 | 0.2500 | 0.2503 |
| ECE (positive class) | 0.0417 | 0.0204 | 0.0361 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 704 / 569 / 2029 / 2043 | 891 / 796 / 1969 / 2041 | 825 / 678 / 2148 / 2223 |
| Gaussian readout: calls up | 0.4804 | 0.4573 | 0.4227 |
| Gaussian readout: MCC | 0.0054 | 0.0291 | 0.0280 |
| Gaussian readout: AUC | 0.5032 | 0.5153 | 0.5121 |
| Gaussian readout: Brier | 0.2510 | 0.2509 | 0.2528 |
| Gaussian readout: ECE | 0.0300 | 0.0208 | 0.0325 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.76 | 237.14 | 271.12 |
| RMSE ($), raw heads | 196.76 | 240.61 | 274.68 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.92 | 176.31 | 201.33 |
| MAE ($), raw heads | 145.92 | 178.96 | 204.15 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0055 | -0.0088 | -0.0150 |
| skill vs zero, raw heads | -0.0055 | -0.0386 | -0.0419 |
| EV, served | -0.0052 | -0.0076 | -0.0119 |
| EV, raw heads | -0.0052 | -0.0358 | -0.0357 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0146 | 0.0091 | 0.0050 |
| corr, Spearman, raw heads | -0.0065 | 0.0060 | 0.0011 |
| mean predicted ($), served | -0.89 | -3.08 | -7.17 |
| mean predicted ($), raw heads | -0.89 | -6.30 | -12.19 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.4910 | 0.4544 | 0.4276 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 1.0000 | 0.4885 | 0.5883 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.66 | 127.63 | 145.99 |
| CRPSS vs constant variance | 0.0204 | 0.0144 | 0.0126 |
| NLL | 6.6482 | 6.8378 | 6.9689 |
| PIT KS | 0.0358 | 0.0401 | 0.0488 |
| var / err^2 Spearman | 0.2575 | 0.2496 | 0.2352 |
| coverage of the 90% interval | 0.9060 | 0.9098 | 0.9073 |
| width of the 90% interval ($) | 652.44 | 806.20 | 916.83 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0005 | [-0.0378, 0.0381] | NOISE |
| h1 | 0.0218 | [-0.0209, 0.0609] | NOISE |
| h2 | 0.0298 | [-0.0116, 0.0743] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 1.000 / h1 0.489 / h2 0.588) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7447 | 0.4880 | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.7139 | 0.7894 | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5048 | 0.3694 | 0.3159 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5430 | 0.5853 | 0.6329 | 0.2859 |
| expected if the two signs were independent | 0.5050 | 0.5171 | 0.5374 | 0.2167 |

- P(up) unanimity (all three horizons call the same side): 0.5048

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0437 vs 0.0390 (+0.0047): beats, noise (boot z +0.16) | 0.0175 vs 0.0397 (-0.0221): does not beat, noise (boot z -0.54) | 0.0352 vs 0.0411 (-0.0059): does not beat, noise (boot z -0.18) |
| logreg_lags | direction/auc | 0.5201 vs 0.5241 (-0.0040): does not beat, noise (boot z -0.22) | 0.5161 vs 0.5228 (-0.0067): does not beat, noise (boot z -0.27) | 0.5262 vs 0.5319 (-0.0057): does not beat, noise (boot z -0.28) |
| logreg_lags | direction/brier | 0.2518 vs 0.2495 (-0.0023): does not beat, noise (DM z -1.66) | 0.2500 vs 0.2493 (-0.0006): does not beat, noise (DM z -0.65) | 0.2503 vs 0.2492 (-0.0011): does not beat, noise (DM z -0.99) |
| logreg_lags | direction/ece_pos | 0.0417 vs 0.0211 (-0.0207): does not beat, significantly worse (boot z -2.48) | 0.0204 vs 0.0197 (-0.0007): does not beat, noise (boot z -0.09) | 0.0361 vs 0.0231 (-0.0129): does not beat, noise (boot z -1.41) |
| logreg_lags | direction/acc | 0.5113 vs 0.5160 (-0.0047): does not beat, noise (DM z -0.31) | 0.5020 vs 0.5166 (-0.0146): does not beat, noise (DM z -0.74) | 0.5061 vs 0.5157 (-0.0095): does not beat, noise (DM z -0.56) |
| logreg_lags | direction/bal_acc | 0.5186 vs 0.5190 (-0.0004): does not beat, noise (boot z -0.03) | 0.5080 vs 0.5195 (-0.0115): does not beat, noise (boot z -0.59) | 0.5154 vs 0.5200 (-0.0047): does not beat, noise (boot z -0.30) |
| class_prior | direction/mcc | 0.0437 vs 0.0000 (+0.0437): beats, noise (boot z +1.71) | 0.0175 vs 0.0000 (+0.0175): beats, noise (boot z +0.73) | 0.0352 vs 0.0000 (+0.0352): beats, noise (boot z +1.27) |
| class_prior | direction/auc | 0.5201 vs 0.5000 (+0.0201): beats, noise (boot z +1.14) | 0.5161 vs 0.5000 (+0.0161): beats, noise (boot z +1.04) | 0.5262 vs 0.5000 (+0.0262): beats, noise (boot z +1.38) |
| class_prior | direction/brier | 0.2518 vs 0.2502 (-0.0016): does not beat, noise (DM z -1.10) | 0.2500 vs 0.2502 (+0.0002): beats, noise (DM z +0.60) | 0.2503 vs 0.2502 (-0.0001): does not beat, noise (DM z -0.10) |
| class_prior | direction/ece_pos | 0.0417 vs 0.0208 (-0.0209): does not beat, significantly worse (boot z -4.20) | 0.0204 vs 0.0196 (-0.0008): does not beat, noise (boot z -0.19) | 0.0361 vs 0.0234 (-0.0127): does not beat, significantly worse (boot z -1.99) |
| class_prior | direction/acc | 0.5113 vs 0.4861 (+0.0253): beats, noise (DM z +1.92) | 0.5020 vs 0.4853 (+0.0167): beats, noise (DM z +0.99) | 0.5061 vs 0.4811 (+0.0250): beats, noise (DM z +1.60) |
| class_prior | direction/bal_acc | 0.5186 vs 0.5000 (+0.0186): beats, noise (boot z +1.70) | 0.5080 vs 0.5000 (+0.0080): beats, noise (boot z +0.73) | 0.5154 vs 0.5000 (+0.0154): beats, noise (boot z +1.26) |
| zero_delta | delta/rmse | 196.76 vs 196.21 (-0.54, -0.28%): does not beat, noise (DM z -1.09) | 237.14 vs 236.10 (-1.04, -0.44%): does not beat, noise (DM z -1.26) | 271.12 vs 269.11 (-2.01, -0.75%): does not beat, noise (DM z -1.66) |
| zero_delta | delta/mae | 145.92 vs 145.66 (-0.27, -0.18%): does not beat, noise (DM z -0.63) | 176.31 vs 175.66 (-0.65, -0.37%): does not beat, noise (DM z -1.05) | 201.33 vs 199.93 (-1.40, -0.70%): does not beat, noise (DM z -1.47) |
| mean_delta | delta/rmse | 196.76 vs 196.24 (-0.52, -0.26%): does not beat, noise (DM z -1.05) | 237.14 vs 236.14 (-1.00, -0.42%): does not beat, noise (DM z -1.22) | 271.12 vs 269.17 (-1.95, -0.72%): does not beat, noise (DM z -1.63) |
| mean_delta | delta/mae | 145.92 vs 145.68 (-0.25, -0.17%): does not beat, noise (DM z -0.59) | 176.31 vs 175.69 (-0.62, -0.35%): does not beat, noise (DM z -1.01) | 201.33 vs 199.97 (-1.36, -0.68%): does not beat, noise (DM z -1.44) |
| const_var | variance/crps | 105.66 vs 107.87 (+2.21, +2.04%): beats (DM z +4.42) | 127.63 vs 129.50 (+1.87, +1.44%): beats (DM z +2.85) | 145.99 vs 147.86 (+1.86, +1.26%): beats (DM z +2.01) |
| const_var | variance/nll | 6.6482 vs 6.7057 (+0.0575): beats (DM z +3.53) | 6.8378 vs 6.8904 (+0.0526): beats (DM z +2.92) | 6.9689 vs 7.0219 (+0.0530): beats (DM z +2.72) |
| const_var | variance/pit_ks | 0.0358 vs 0.0663 (+0.0305): beats (boot z +5.20) | 0.0401 vs 0.0614 (+0.0214): beats (boot z +3.59) | 0.0488 vs 0.0644 (+0.0156): beats, noise (boot z +1.86) |
| const_var | variance/corr_var_err2_spearman | 0.2575 vs 0.0000 (+0.2575): beats (boot z +7.41) | 0.2496 vs 0.0000 (+0.2496): beats (boot z +6.40) | 0.2352 vs 0.0000 (+0.2352): beats (boot z +5.70) |

## Backtest (costs included)

- n_trades: 246
- total_return: -0.4332
- sharpe_net: -112.4245
- sharpe_gross: 18.6606
- sortino: -130.4312
- max_drawdown: 0.4337
- hit_rate: 0.1057
- hit_rate_gross: 0.5854
- profit_factor: 0.0444
- avg_hold_bars: 9.3211
- exposure: 0.3169
- turnover: 376.5363
- fees_paid: 3765.1922
- costs_paid: 4894.7499
- gross_pnl: 562.6118
- net_pnl: -4332.1381

## Experiment engine: out-of-sample block and backtest

Role: **test** (shown, never used to rank or choose (D-020)). Fold -1 (TimeSeriesSplit fold 5, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-11-05T06:34:00+00:00 .. 2025-11-10T07:09:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.148, long_above 0.5284, short_below 0.4569, median 0.4917. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -43.32% | -112.42 | +43.37% | 246 |
| buy and hold | +4.10% | +6.65 | +4.95% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -50.79% .. -43.90%) | -47.33% | -134.49 | | |

The random null enters at the strategy's rate (0.0498 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 97% of its seeds on net return, 100% on net Sharpe and 98% on gross return.
