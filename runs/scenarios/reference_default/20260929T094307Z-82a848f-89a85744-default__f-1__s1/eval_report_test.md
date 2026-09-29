# Evaluation report - test split - run `20260929T094307Z-82a848f-89a85744-default__f-1__s1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.2462 | 0.3009 | 0.3689 |
| accuracy | 0.4793 | 0.4804 | 0.5146 |
| balanced accuracy | 0.4864 | 0.4863 | 0.5196 |
| precision (up) | 0.4863 | 0.4918 | 0.5455 |
| recall / sensitivity (up) | 0.2330 | 0.2875 | 0.3878 |
| specificity (down) | 0.7398 | 0.6850 | 0.6515 |
| F1 (up) | 0.3150 | 0.3629 | 0.4533 |
| MCC | -0.0316 | -0.0300 | 0.0406 |
| AUC | 0.4934 | 0.4689 | 0.5156 |
| Brier | 0.2532 | 0.2553 | 0.2499 |
| ECE (positive class) | 0.0580 | 0.0587 | 0.0217 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 640 / 676 / 1922 / 2107 | 843 / 871 / 1894 / 2089 | 1182 / 985 / 1841 / 1866 |
| Gaussian readout: calls up | 0.0488 | 0.1548 | 0.1161 |
| Gaussian readout: MCC | 0.0276 | -0.0417 | -0.0137 |
| Gaussian readout: AUC | 0.4956 | 0.4662 | 0.4632 |
| Gaussian readout: Brier | 0.2564 | 0.2557 | 0.2539 |
| Gaussian readout: ECE | 0.0735 | 0.0737 | 0.0489 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 198.32 | 238.88 | 271.84 |
| RMSE ($), raw heads | 198.32 | 239.93 | 275.46 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 147.73 | 177.71 | 201.99 |
| MAE ($), raw heads | 147.73 | 178.52 | 204.81 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0216 | -0.0237 | -0.0204 |
| skill vs zero, raw heads | -0.0216 | -0.0327 | -0.0478 |
| EV, served | -0.0038 | -0.0159 | -0.0134 |
| EV, raw heads | -0.0038 | -0.0220 | -0.0307 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0028 | -0.0451 | -0.0831 |
| corr, Spearman, raw heads | -0.0352 | -0.0775 | -0.0857 |
| mean predicted ($), served | -20.59 | -13.59 | -13.43 |
| mean predicted ($), raw heads | -20.59 | -16.83 | -25.04 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.0424 | 0.1483 | 0.1126 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 1.0000 | 0.8073 | 0.5365 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 106.84 | 128.77 | 146.49 |
| CRPSS vs constant variance | 0.0096 | 0.0056 | 0.0093 |
| NLL | 6.6638 | 6.8885 | 6.9988 |
| PIT KS | 0.0749 | 0.0464 | 0.0425 |
| var / err^2 Spearman | 0.2398 | 0.2246 | 0.2199 |
| coverage of the 90% interval | 0.8951 | 0.9062 | 0.9128 |
| width of the 90% interval ($) | 636.04 | 792.02 | 923.62 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0177 | [-0.0369, 0.0724] | NOISE |
| h1 | -0.0439 | [-0.0871, 0.0039] | NOISE |
| h2 | -0.0100 | [-0.0547, 0.0344] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 1.000 / h1 0.807 / h2 0.536) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4256 | 0.2985 | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.5216 | 0.2385 | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.2452 | 0.0800 | 0.3159 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7497 | 0.8032 | 0.6100 | 0.4164 |
| expected if the two signs were independent | 0.7218 | 0.6508 | 0.6059 | 0.3589 |

- P(up) unanimity (all three horizons call the same side): 0.4938

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0316 vs 0.0390 (-0.0706): does not beat, noise (boot z -1.86) | -0.0300 vs 0.0397 (-0.0696): does not beat, significantly worse (boot z -2.05) | 0.0406 vs 0.0411 (-0.0005): does not beat, noise (boot z -0.01) |
| logreg_lags | direction/auc | 0.4934 vs 0.5241 (-0.0307): does not beat, noise (boot z -1.35) | 0.4689 vs 0.5228 (-0.0539): does not beat, significantly worse (boot z -2.49) | 0.5156 vs 0.5319 (-0.0163): does not beat, noise (boot z -0.70) |
| logreg_lags | direction/brier | 0.2532 vs 0.2495 (-0.0037): does not beat, significantly worse (DM z -2.47) | 0.2553 vs 0.2493 (-0.0059): does not beat, significantly worse (DM z -3.78) | 0.2499 vs 0.2492 (-0.0007): does not beat, noise (DM z -0.96) |
| logreg_lags | direction/ece_pos | 0.0580 vs 0.0211 (-0.0370): does not beat, significantly worse (boot z -2.85) | 0.0587 vs 0.0197 (-0.0390): does not beat, significantly worse (boot z -2.71) | 0.0217 vs 0.0231 (+0.0014): beats, noise (boot z +0.20) |
| logreg_lags | direction/acc | 0.4793 vs 0.5160 (-0.0367): does not beat, significantly worse (DM z -2.17) | 0.4804 vs 0.5166 (-0.0362): does not beat, significantly worse (DM z -2.17) | 0.5146 vs 0.5157 (-0.0010): does not beat, noise (DM z -0.07) |
| logreg_lags | direction/bal_acc | 0.4864 vs 0.5190 (-0.0326): does not beat, noise (boot z -1.86) | 0.4863 vs 0.5195 (-0.0332): does not beat, significantly worse (boot z -2.05) | 0.5196 vs 0.5200 (-0.0004): does not beat, noise (boot z -0.03) |
| class_prior | direction/mcc | -0.0316 vs 0.0000 (-0.0316): does not beat, noise (boot z -1.13) | -0.0300 vs 0.0000 (-0.0300): does not beat, noise (boot z -1.02) | 0.0406 vs 0.0000 (+0.0406): beats, noise (boot z +1.43) |
| class_prior | direction/auc | 0.4934 vs 0.5000 (-0.0066): does not beat, noise (boot z -0.35) | 0.4689 vs 0.5000 (-0.0311): does not beat, noise (boot z -1.73) | 0.5156 vs 0.5000 (+0.0156): beats, noise (boot z +0.88) |
| class_prior | direction/brier | 0.2532 vs 0.2502 (-0.0029): does not beat, significantly worse (DM z -2.14) | 0.2553 vs 0.2502 (-0.0051): does not beat, significantly worse (DM z -3.31) | 0.2499 vs 0.2502 (+0.0003): beats, noise (DM z +0.59) |
| class_prior | direction/ece_pos | 0.0580 vs 0.0208 (-0.0372): does not beat, significantly worse (boot z -2.95) | 0.0587 vs 0.0196 (-0.0391): does not beat, significantly worse (boot z -2.62) | 0.0217 vs 0.0234 (+0.0017): beats, noise (boot z +0.24) |
| class_prior | direction/acc | 0.4793 vs 0.4861 (-0.0067): does not beat, noise (DM z -0.46) | 0.4804 vs 0.4853 (-0.0049): does not beat, noise (DM z -0.27) | 0.5146 vs 0.4811 (+0.0335): beats, noise (DM z +1.67) |
| class_prior | direction/bal_acc | 0.4864 vs 0.5000 (-0.0136): does not beat, noise (boot z -1.12) | 0.4863 vs 0.5000 (-0.0137): does not beat, noise (boot z -1.03) | 0.5196 vs 0.5000 (+0.0196): beats, noise (boot z +1.43) |
| zero_delta | delta/rmse | 198.32 vs 196.21 (-2.10, -1.07%): does not beat, significantly worse (DM z -3.01) | 238.88 vs 236.10 (-2.78, -1.18%): does not beat, significantly worse (DM z -3.43) | 271.84 vs 269.11 (-2.74, -1.02%): does not beat, significantly worse (DM z -2.94) |
| zero_delta | delta/mae | 147.73 vs 145.66 (-2.07, -1.42%): does not beat, significantly worse (DM z -3.18) | 177.71 vs 175.66 (-2.05, -1.17%): does not beat, significantly worse (DM z -2.82) | 201.99 vs 199.93 (-2.06, -1.03%): does not beat, significantly worse (DM z -3.04) |
| mean_delta | delta/rmse | 198.32 vs 196.24 (-2.08, -1.06%): does not beat, significantly worse (DM z -3.06) | 238.88 vs 236.14 (-2.74, -1.16%): does not beat, significantly worse (DM z -3.47) | 271.84 vs 269.17 (-2.67, -0.99%): does not beat, significantly worse (DM z -2.99) |
| mean_delta | delta/mae | 147.73 vs 145.68 (-2.05, -1.41%): does not beat, significantly worse (DM z -3.23) | 177.71 vs 175.69 (-2.02, -1.15%): does not beat, significantly worse (DM z -2.84) | 201.99 vs 199.97 (-2.01, -1.01%): does not beat, significantly worse (DM z -3.12) |
| const_var | variance/crps | 106.84 vs 107.87 (+1.03, +0.96%): beats (DM z +1.96) | 128.77 vs 129.50 (+0.73, +0.56%): beats, noise (DM z +0.92) | 146.49 vs 147.86 (+1.37, +0.93%): beats, noise (DM z +1.51) |
| const_var | variance/nll | 6.6638 vs 6.7057 (+0.0419): beats (DM z +3.01) | 6.8885 vs 6.8904 (+0.0019): beats, noise (DM z +0.07) | 6.9988 vs 7.0219 (+0.0231): beats, noise (DM z +0.92) |
| const_var | variance/pit_ks | 0.0749 vs 0.0663 (-0.0086): does not beat, noise (boot z -1.21) | 0.0464 vs 0.0614 (+0.0151): beats, noise (boot z +1.10) | 0.0425 vs 0.0644 (+0.0219): beats, noise (boot z +1.57) |
| const_var | variance/corr_var_err2_spearman | 0.2398 vs 0.0000 (+0.2398): beats (boot z +7.43) | 0.2246 vs 0.0000 (+0.2246): beats (boot z +5.80) | 0.2199 vs 0.0000 (+0.2199): beats (boot z +5.16) |

## Backtest (costs included)

- n_trades: 174
- total_return: -0.3761
- sharpe_net: -102.1120
- sharpe_gross: -4.3994
- sortino: -118.0135
- max_drawdown: 0.3761
- hit_rate: 0.0862
- hit_rate_gross: 0.4828
- profit_factor: 0.0213
- avg_hold_bars: 8.9713
- exposure: 0.2157
- turnover: 279.2653
- fees_paid: 2792.7810
- costs_paid: 3630.6154
- gross_pnl: -130.1269
- net_pnl: -3760.7422

## Experiment engine: out-of-sample block and backtest

Role: **test** (shown, never used to rank or choose (D-020)). Fold -1 (TimeSeriesSplit fold 5, 5 usable folds); this report scores the fold's out-of-sample block: 7236 sequences, 2025-11-05T06:34:00+00:00 .. 2025-11-10T07:09:00+00:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8239, long_above 0.5254, short_below 0.4416, median 0.4804. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -37.61% | -102.11 | +37.61% | 174 |
| buy and hold | +4.10% | +6.65 | +4.95% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -40.08% .. -32.41%) | -36.26% | -112.25 | | |

The random null enters at the strategy's rate (0.0307 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 29% of its seeds on net return, 88% on net Sharpe and 25% on gross return.
