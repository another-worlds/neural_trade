# Evaluation report - test split - run `20261003T225052Z-91fa363-11993eec`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.6563 | 0.5880 | 0.4837 |
| accuracy | 0.5085 | 0.5071 | 0.5034 |
| balanced accuracy | 0.5042 | 0.5045 | 0.5040 |
| precision (up) | 0.5171 | 0.5185 | 0.5231 |
| recall / sensitivity (up) | 0.6604 | 0.5924 | 0.4875 |
| specificity (down) | 0.3480 | 0.4166 | 0.5205 |
| F1 (up) | 0.5800 | 0.5530 | 0.5047 |
| MCC | 0.0088 | 0.0092 | 0.0081 |
| AUC | 0.5099 | 0.5189 | 0.4987 |
| Brier | 0.2501 | 0.2508 | 0.2531 |
| ECE (positive class) | 0.0309 | 0.0338 | 0.0352 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1814 / 1694 / 904 / 933 | 1737 / 1613 / 1152 / 1195 | 1486 / 1355 / 1471 / 1562 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.3456 | 0.7941 | 0.7521 |
| Gaussian readout of the raw heads: MCC | -0.0663 | -0.0680 | -0.0043 |
| Gaussian readout of the raw heads: AUC | 0.4552 | 0.4688 | 0.4961 |
| Gaussian readout of the raw heads: Brier | 0.2530 | 0.2563 | 0.2545 |
| Gaussian readout of the raw heads: ECE | 0.0582 | 0.0758 | 0.0560 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.21 | 236.10 | 269.11 |
| RMSE ($), raw heads | 197.58 | 239.11 | 272.24 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.66 | 175.66 | 199.93 |
| MAE ($), raw heads | 146.69 | 177.97 | 202.10 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0140 | -0.0256 | -0.0234 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0120 | -0.0269 | -0.0248 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0654 | -0.0767 | -0.0412 |
| corr, Spearman, raw heads | -0.0719 | -0.0792 | -0.0125 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -4.64 | 13.95 | 20.03 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3658 | 0.8027 | 0.7533 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.59 | 126.97 | 144.99 |
| CRPSS vs constant variance | 0.0211 | 0.0195 | 0.0194 |
| NLL | 6.6493 | 6.8386 | 6.9674 |
| PIT KS | 0.0379 | 0.0271 | 0.0382 |
| var / err^2 Spearman | 0.2555 | 0.2430 | 0.2350 |
| coverage of the 90% interval | 0.9027 | 0.9059 | 0.9124 |
| width of the 90% interval ($) | 643.12 | 781.37 | 911.88 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0231 | [-0.0115, 0.0680] | NOISE |
| h1 | 0.0351 | [-0.0094, 0.0754] | NOISE |
| h2 | -0.0097 | [-0.0547, 0.0335] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8433 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.7468 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.6238 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5915 | 0.7011 | 0.6363 | 0.2700 |
| expected if the two signs were independent | 0.4565 | 0.5628 | 0.4914 | 0.1373 |

- P(up) unanimity (all three horizons call the same side): 0.3659

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0088 vs 0.0390 (-0.0302): does not beat, noise (boot z -0.99) | 0.0092 vs 0.0397 (-0.0305): does not beat, noise (boot z -0.76) | 0.0081 vs 0.0411 (-0.0331): does not beat, noise (boot z -0.89) |
| logreg_lags | direction/auc | 0.5099 vs 0.5241 (-0.0142): does not beat, noise (boot z -0.81) | 0.5189 vs 0.5228 (-0.0039): does not beat, noise (boot z -0.17) | 0.4987 vs 0.5319 (-0.0332): does not beat, noise (boot z -1.57) |
| logreg_lags | direction/brier | 0.2501 vs 0.2495 (-0.0006): does not beat, noise (DM z -0.41) | 0.2508 vs 0.2493 (-0.0015): does not beat, noise (DM z -0.82) | 0.2531 vs 0.2492 (-0.0039): does not beat, significantly worse (DM z -2.54) |
| logreg_lags | direction/ece_pos | 0.0309 vs 0.0211 (-0.0098): does not beat, noise (boot z -0.57) | 0.0338 vs 0.0197 (-0.0142): does not beat, noise (boot z -0.81) | 0.0352 vs 0.0231 (-0.0121): does not beat, noise (boot z -0.85) |
| logreg_lags | direction/acc | 0.5085 vs 0.5160 (-0.0075): does not beat, noise (DM z -0.42) | 0.5071 vs 0.5166 (-0.0095): does not beat, noise (DM z -0.44) | 0.5034 vs 0.5157 (-0.0123): does not beat, noise (DM z -0.64) |
| logreg_lags | direction/bal_acc | 0.5042 vs 0.5190 (-0.0149): does not beat, noise (boot z -1.01) | 0.5045 vs 0.5195 (-0.0149): does not beat, noise (boot z -0.76) | 0.5040 vs 0.5200 (-0.0160): does not beat, noise (boot z -0.88) |
| class_prior | direction/mcc | 0.0088 vs 0.0000 (+0.0088): beats, noise (boot z +0.34) | 0.0092 vs 0.0000 (+0.0092): beats, noise (boot z +0.32) | 0.0081 vs 0.0000 (+0.0081): beats, noise (boot z +0.28) |
| class_prior | direction/auc | 0.5099 vs 0.5000 (+0.0099): beats, noise (boot z +0.61) | 0.5189 vs 0.5000 (+0.0189): beats, noise (boot z +1.07) | 0.4987 vs 0.5000 (-0.0013): does not beat, noise (boot z -0.07) |
| class_prior | direction/brier | 0.2501 vs 0.2502 (+0.0001): beats, noise (DM z +0.08) | 0.2508 vs 0.2502 (-0.0006): does not beat, noise (DM z -0.37) | 0.2531 vs 0.2502 (-0.0029): does not beat, noise (DM z -1.86) |
| class_prior | direction/ece_pos | 0.0309 vs 0.0208 (-0.0100): does not beat, noise (boot z -0.51) | 0.0338 vs 0.0196 (-0.0142): does not beat, noise (boot z -0.73) | 0.0352 vs 0.0234 (-0.0118): does not beat, noise (boot z -0.78) |
| class_prior | direction/acc | 0.5085 vs 0.4861 (+0.0225): beats, noise (DM z +0.87) | 0.5071 vs 0.4853 (+0.0218): beats, noise (DM z +0.78) | 0.5034 vs 0.4811 (+0.0223): beats, noise (DM z +0.85) |
| class_prior | direction/bal_acc | 0.5042 vs 0.5000 (+0.0042): beats, noise (boot z +0.34) | 0.5045 vs 0.5000 (+0.0045): beats, noise (boot z +0.32) | 0.5040 vs 0.5000 (+0.0040): beats, noise (boot z +0.28) |
| zero_delta | delta/rmse | 196.21 vs 196.21 (+0.00, +0.00%): does not beat | 236.10 vs 236.10 (+0.00, +0.00%): does not beat | 269.11 vs 269.11 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 145.66 vs 145.66 (+0.00, +0.00%): does not beat | 175.66 vs 175.66 (+0.00, +0.00%): does not beat | 199.93 vs 199.93 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 196.21 vs 196.24 (+0.02, +0.01%): beats, noise (DM z +1.05) | 236.10 vs 236.14 (+0.04, +0.02%): beats, noise (DM z +1.09) | 269.11 vs 269.17 (+0.06, +0.02%): beats, noise (DM z +1.12) |
| mean_delta | delta/mae | 145.66 vs 145.68 (+0.02, +0.01%): beats, noise (DM z +1.01) | 175.66 vs 175.69 (+0.03, +0.02%): beats, noise (DM z +0.94) | 199.93 vs 199.97 (+0.05, +0.02%): beats, noise (DM z +0.91) |
| const_var | variance/crps | 105.59 vs 107.87 (+2.28, +2.11%): beats (DM z +6.27) | 126.97 vs 129.50 (+2.53, +1.95%): beats (DM z +4.49) | 144.99 vs 147.86 (+2.86, +1.94%): beats (DM z +4.01) |
| const_var | variance/nll | 6.6493 vs 6.7057 (+0.0565): beats (DM z +4.30) | 6.8386 vs 6.8904 (+0.0517): beats (DM z +2.80) | 6.9674 vs 7.0219 (+0.0545): beats (DM z +2.71) |
| const_var | variance/pit_ks | 0.0379 vs 0.0663 (+0.0284): beats (boot z +5.76) | 0.0271 vs 0.0614 (+0.0344): beats (boot z +5.16) | 0.0382 vs 0.0644 (+0.0262): beats (boot z +4.19) |
| const_var | variance/corr_var_err2_spearman | 0.2555 vs 0.0000 (+0.2555): beats (boot z +7.45) | 0.2430 vs 0.0000 (+0.2430): beats (boot z +6.42) | 0.2350 vs 0.0000 (+0.2350): beats (boot z +5.49) |
