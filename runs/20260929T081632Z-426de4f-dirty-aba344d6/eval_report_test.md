# Evaluation report - test split - run `20260929T081632Z-426de4f-dirty-aba344d6`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.5523 | 0.4808 | 0.3973 |
| accuracy | 0.5036 | 0.4904 | 0.5036 |
| balanced accuracy | 0.5022 | 0.4910 | 0.5075 |
| precision (up) | 0.5159 | 0.5053 | 0.5283 |
| recall / sensitivity (up) | 0.5544 | 0.4720 | 0.4045 |
| specificity (down) | 0.4500 | 0.5099 | 0.6104 |
| F1 (up) | 0.5345 | 0.4881 | 0.4582 |
| MCC | 0.0044 | -0.0180 | 0.0152 |
| AUC | 0.5058 | 0.4997 | 0.5067 |
| Brier | 0.2510 | 0.2522 | 0.2511 |
| ECE (positive class) | 0.0211 | 0.0423 | 0.0259 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1523 / 1429 / 1169 / 1224 | 1384 / 1355 / 1410 / 1548 | 1233 / 1101 / 1725 / 1815 |
| Gaussian readout: calls up | 0.3899 | 0.3244 | 0.3138 |
| Gaussian readout: MCC | -0.0046 | 0.0007 | 0.0130 |
| Gaussian readout: AUC | 0.4920 | 0.4833 | 0.4902 |
| Gaussian readout: Brier | 0.2508 | 0.2504 | 0.2536 |
| Gaussian readout: ECE | 0.0203 | 0.0182 | 0.0372 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.71 | 236.57 | 272.46 |
| RMSE ($), raw heads | 200.38 | 263.09 | 289.67 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.99 | 175.93 | 202.71 |
| MAE ($), raw heads | 148.63 | 190.78 | 215.89 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0051 | -0.0040 | -0.0251 |
| skill vs zero, raw heads | -0.0429 | -0.2417 | -0.1587 |
| EV, served | -0.0040 | -0.0032 | -0.0198 |
| EV, raw heads | -0.0379 | -0.2209 | -0.1327 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0065 | -0.0333 | -0.0378 |
| corr, Spearman, raw heads | -0.0190 | -0.0363 | -0.0390 |
| mean predicted ($), served | -2.73 | -1.95 | -10.84 |
| mean predicted ($), raw heads | -9.02 | -26.25 | -32.95 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3816 | 0.3152 | 0.3116 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.3032 | 0.0744 | 0.3289 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.72 | 127.70 | 146.54 |
| CRPSS vs constant variance | 0.0199 | 0.0139 | 0.0089 |
| NLL | 6.6521 | 6.8613 | 6.9809 |
| PIT KS | 0.0325 | 0.0258 | 0.0450 |
| var / err^2 Spearman | 0.2533 | 0.2259 | 0.2405 |
| coverage of the 90% interval | 0.9044 | 0.9085 | 0.9120 |
| width of the 90% interval ($) | 649.45 | 790.91 | 926.14 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0213 | [-0.0169, 0.0617] | NOISE |
| h1 | 0.0359 | [-0.0071, 0.0782] | NOISE |
| h2 | -0.0061 | [-0.0484, 0.0330] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.303 / h1 0.074 / h2 0.329) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8881 | 0.2957 | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.6815 | 0.9428 | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5851 | 0.2696 | 0.3159 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5985 | 0.6092 | 0.6530 | 0.2551 |
| expected if the two signs were independent | 0.4874 | 0.5066 | 0.5413 | 0.1550 |

- P(up) unanimity (all three horizons call the same side): 0.3119

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0044 vs 0.0390 (-0.0346): does not beat, noise (boot z -1.09) | -0.0180 vs 0.0397 (-0.0577): does not beat, noise (boot z -1.71) | 0.0152 vs 0.0411 (-0.0259): does not beat, noise (boot z -0.77) |
| logreg_lags | direction/auc | 0.5058 vs 0.5241 (-0.0183): does not beat, noise (boot z -0.98) | 0.4997 vs 0.5228 (-0.0231): does not beat, noise (boot z -1.08) | 0.5067 vs 0.5319 (-0.0252): does not beat, noise (boot z -1.31) |
| logreg_lags | direction/brier | 0.2510 vs 0.2495 (-0.0014): does not beat, noise (DM z -1.08) | 0.2522 vs 0.2493 (-0.0029): does not beat, noise (DM z -1.85) | 0.2511 vs 0.2492 (-0.0019): does not beat, noise (DM z -1.94) |
| logreg_lags | direction/ece_pos | 0.0211 vs 0.0211 (-0.0001): does not beat, noise (boot z -0.01) | 0.0423 vs 0.0197 (-0.0226): does not beat, noise (boot z -1.52) | 0.0259 vs 0.0231 (-0.0028): does not beat, noise (boot z -0.29) |
| logreg_lags | direction/acc | 0.5036 vs 0.5160 (-0.0123): does not beat, noise (DM z -0.70) | 0.4904 vs 0.5166 (-0.0262): does not beat, noise (DM z -1.59) | 0.5036 vs 0.5157 (-0.0121): does not beat, noise (DM z -0.75) |
| logreg_lags | direction/bal_acc | 0.5022 vs 0.5190 (-0.0168): does not beat, noise (boot z -1.08) | 0.4910 vs 0.5195 (-0.0285): does not beat, noise (boot z -1.71) | 0.5075 vs 0.5200 (-0.0126): does not beat, noise (boot z -0.77) |
| class_prior | direction/mcc | 0.0044 vs 0.0000 (+0.0044): beats, noise (boot z +0.19) | -0.0180 vs 0.0000 (-0.0180): does not beat, noise (boot z -0.68) | 0.0152 vs 0.0000 (+0.0152): beats, noise (boot z +0.56) |
| class_prior | direction/auc | 0.5058 vs 0.5000 (+0.0058): beats, noise (boot z +0.40) | 0.4997 vs 0.5000 (-0.0003): does not beat, noise (boot z -0.02) | 0.5067 vs 0.5000 (+0.0067): beats, noise (boot z +0.39) |
| class_prior | direction/brier | 0.2510 vs 0.2502 (-0.0007): does not beat, noise (DM z -0.74) | 0.2522 vs 0.2502 (-0.0021): does not beat, noise (DM z -1.45) | 0.2511 vs 0.2502 (-0.0009): does not beat, noise (DM z -0.89) |
| class_prior | direction/ece_pos | 0.0211 vs 0.0208 (-0.0003): does not beat, noise (boot z -0.02) | 0.0423 vs 0.0196 (-0.0227): does not beat, noise (boot z -1.40) | 0.0259 vs 0.0234 (-0.0026): does not beat, noise (boot z -0.27) |
| class_prior | direction/acc | 0.5036 vs 0.4861 (+0.0176): beats, noise (DM z +0.76) | 0.4904 vs 0.4853 (+0.0051): beats, noise (DM z +0.22) | 0.5036 vs 0.4811 (+0.0225): beats, noise (DM z +1.04) |
| class_prior | direction/bal_acc | 0.5022 vs 0.5000 (+0.0022): beats, noise (boot z +0.19) | 0.4910 vs 0.5000 (-0.0090): does not beat, noise (boot z -0.68) | 0.5075 vs 0.5000 (+0.0075): beats, noise (boot z +0.56) |
| zero_delta | delta/rmse | 196.71 vs 196.21 (-0.50, -0.25%): does not beat, noise (DM z -1.23) | 236.57 vs 236.10 (-0.47, -0.20%): does not beat, noise (DM z -1.57) | 272.46 vs 269.11 (-3.36, -1.25%): does not beat, significantly worse (DM z -2.58) |
| zero_delta | delta/mae | 145.99 vs 145.66 (-0.34, -0.23%): does not beat, noise (DM z -1.21) | 175.93 vs 175.66 (-0.28, -0.16%): does not beat, noise (DM z -1.38) | 202.71 vs 199.93 (-2.78, -1.39%): does not beat, significantly worse (DM z -3.11) |
| mean_delta | delta/rmse | 196.71 vs 196.24 (-0.48, -0.24%): does not beat, noise (DM z -1.18) | 236.57 vs 236.14 (-0.42, -0.18%): does not beat, noise (DM z -1.47) | 272.46 vs 269.17 (-3.29, -1.22%): does not beat, significantly worse (DM z -2.57) |
| mean_delta | delta/mae | 145.99 vs 145.68 (-0.32, -0.22%): does not beat, noise (DM z -1.16) | 175.93 vs 175.69 (-0.24, -0.14%): does not beat, noise (DM z -1.26) | 202.71 vs 199.97 (-2.74, -1.37%): does not beat, significantly worse (DM z -3.12) |
| const_var | variance/crps | 105.72 vs 107.87 (+2.15, +1.99%): beats (DM z +4.87) | 127.70 vs 129.50 (+1.80, +1.39%): beats (DM z +2.90) | 146.54 vs 147.86 (+1.32, +0.89%): beats, noise (DM z +1.34) |
| const_var | variance/nll | 6.6521 vs 6.7057 (+0.0537): beats (DM z +3.23) | 6.8613 vs 6.8904 (+0.0290): beats, noise (DM z +1.29) | 6.9809 vs 7.0219 (+0.0410): beats, noise (DM z +1.75) |
| const_var | variance/pit_ks | 0.0325 vs 0.0663 (+0.0338): beats (boot z +4.80) | 0.0258 vs 0.0614 (+0.0356): beats (boot z +4.35) | 0.0450 vs 0.0644 (+0.0194): beats, noise (boot z +1.69) |
| const_var | variance/corr_var_err2_spearman | 0.2533 vs 0.0000 (+0.2533): beats (boot z +7.11) | 0.2259 vs 0.0000 (+0.2259): beats (boot z +5.74) | 0.2405 vs 0.0000 (+0.2405): beats (boot z +5.89) |
