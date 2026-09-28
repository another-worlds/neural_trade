# Evaluation report - test split - run `20260924T182915Z-1aeff1c-dirty-af67ee43`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.4958 | 0.4308 | 0.4848 |
| accuracy | 0.4849 | 0.4811 | 0.5026 |
| balanced accuracy | 0.4850 | 0.4831 | 0.5031 |
| precision (up) | 0.4989 | 0.4951 | 0.5221 |
| recall / sensitivity (up) | 0.4813 | 0.4144 | 0.4879 |
| specificity (down) | 0.4888 | 0.5519 | 0.5184 |
| F1 (up) | 0.4899 | 0.4512 | 0.5044 |
| MCC | -0.0299 | -0.0340 | 0.0063 |
| AUC | 0.4855 | 0.4781 | 0.5044 |
| Brier | 0.2519 | 0.2529 | 0.2507 |
| ECE (positive class) | 0.0391 | 0.0477 | 0.0226 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1322 / 1328 / 1270 / 1425 | 1215 / 1239 / 1526 / 1717 | 1487 / 1361 / 1465 / 1561 |
| Gaussian readout: calls up | 0.3886 | n/a (beta = 0: readout is the constant 0.5) | 0.5410 |
| Gaussian readout: MCC | -0.0034 | n/a (beta = 0: readout is the constant 0.5) | -0.0185 |
| Gaussian readout: AUC | 0.5020 | n/a (beta = 0: readout is the constant 0.5) | 0.4826 |
| Gaussian readout: Brier | 0.2501 | n/a (beta = 0: readout is the constant 0.5) | 0.2502 |
| Gaussian readout: ECE | 0.0150 | n/a (beta = 0: readout is the constant 0.5) | 0.0187 |
| Gaussian readout of the raw heads: calls up | 0.3886 | 0.4867 | 0.5410 |
| Gaussian readout of the raw heads: MCC | -0.0034 | -0.0099 | -0.0185 |
| Gaussian readout of the raw heads: AUC | 0.5020 | 0.4805 | 0.4826 |
| Gaussian readout of the raw heads: Brier | 0.2518 | 0.2587 | 0.2587 |
| Gaussian readout of the raw heads: ECE | 0.0325 | 0.0552 | 0.0693 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.23 | 236.10 | 269.23 |
| RMSE ($), raw heads | 197.25 | 243.27 | 275.70 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.68 | 175.66 | 200.03 |
| MAE ($), raw heads | 146.68 | 181.51 | 205.62 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | n/a (beta = 0: served delta is 0) | -0.0010 |
| skill vs zero, raw heads | -0.0106 | -0.0616 | -0.0496 |
| EV, served | 0.0000 | n/a (beta = 0: served delta is 0) | -0.0010 |
| EV, raw heads | -0.0092 | -0.0608 | -0.0497 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0064 | -0.0453 | -0.0283 |
| corr, Spearman, raw heads | -0.0037 | -0.0301 | -0.0236 |
| mean predicted ($), served | -0.41 | 0.00 | -0.01 |
| mean predicted ($), raw heads | -3.36 | -2.39 | -0.16 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3910 | 0.4977 | 0.5482 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.1234 | 0.0000 | 0.0695 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.50 | 127.33 | 145.01 |
| CRPSS vs constant variance | 0.0220 | 0.0167 | 0.0193 |
| NLL | 6.6473 | 6.8441 | 6.9669 |
| PIT KS | 0.0348 | 0.0328 | 0.0340 |
| var / err^2 Spearman | 0.2498 | 0.2243 | 0.2275 |
| coverage of the 90% interval | 0.9035 | 0.9059 | 0.9116 |
| width of the 90% interval ($) | 645.00 | 781.37 | 911.32 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0124 | [-0.0281, 0.0505] | NOISE |
| h1 | -0.0188 | [-0.0542, 0.0202] | NOISE |
| h2 | 0.0020 | [-0.0418, 0.0479] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.123 / h1 0.000 / h2 0.069) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7919 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.7974 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.6176 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5703 | 0.5239 | 0.5813 | 0.1625 |
| expected if the two signs were independent | 0.4997 | 0.5003 | 0.4975 | 0.1127 |

- P(up) unanimity (all three horizons call the same side): 0.2258

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0299 vs 0.0390 (-0.0689): does not beat, noise (boot z -1.72) | -0.0340 vs 0.0397 (-0.0737): does not beat, significantly worse (boot z -2.41) | 0.0063 vs 0.0411 (-0.0349): does not beat, noise (boot z -0.81) |
| logreg_lags | direction/auc | 0.4855 vs 0.5241 (-0.0386): does not beat, noise (boot z -1.70) | 0.4781 vs 0.5228 (-0.0447): does not beat, significantly worse (boot z -2.31) | 0.5044 vs 0.5319 (-0.0275): does not beat, noise (boot z -1.07) |
| logreg_lags | direction/brier | 0.2519 vs 0.2495 (-0.0024): does not beat, noise (DM z -1.71) | 0.2529 vs 0.2493 (-0.0036): does not beat, significantly worse (DM z -3.16) | 0.2507 vs 0.2492 (-0.0015): does not beat, noise (DM z -1.12) |
| logreg_lags | direction/ece_pos | 0.0391 vs 0.0211 (-0.0180): does not beat, noise (boot z -1.10) | 0.0477 vs 0.0197 (-0.0280): does not beat, noise (boot z -1.75) | 0.0226 vs 0.0231 (+0.0005): beats, noise (boot z +0.04) |
| logreg_lags | direction/acc | 0.4849 vs 0.5160 (-0.0311): does not beat, noise (DM z -1.55) | 0.4811 vs 0.5166 (-0.0355): does not beat, significantly worse (DM z -2.34) | 0.5026 vs 0.5157 (-0.0131): does not beat, noise (DM z -0.62) |
| logreg_lags | direction/bal_acc | 0.4850 vs 0.5190 (-0.0340): does not beat, noise (boot z -1.72) | 0.4831 vs 0.5195 (-0.0363): does not beat, significantly worse (boot z -2.41) | 0.5031 vs 0.5200 (-0.0169): does not beat, noise (boot z -0.80) |
| class_prior | direction/mcc | -0.0299 vs 0.0000 (-0.0299): does not beat, noise (boot z -1.17) | -0.0340 vs 0.0000 (-0.0340): does not beat, noise (boot z -1.08) | 0.0063 vs 0.0000 (+0.0063): beats, noise (boot z +0.20) |
| class_prior | direction/auc | 0.4855 vs 0.5000 (-0.0145): does not beat, noise (boot z -0.92) | 0.4781 vs 0.5000 (-0.0219): does not beat, noise (boot z -1.13) | 0.5044 vs 0.5000 (+0.0044): beats, noise (boot z +0.22) |
| class_prior | direction/brier | 0.2519 vs 0.2502 (-0.0017): does not beat, noise (DM z -1.91) | 0.2529 vs 0.2502 (-0.0028): does not beat, significantly worse (DM z -2.29) | 0.2507 vs 0.2502 (-0.0005): does not beat, noise (DM z -0.39) |
| class_prior | direction/ece_pos | 0.0391 vs 0.0208 (-0.0182): does not beat, noise (boot z -1.07) | 0.0477 vs 0.0196 (-0.0281): does not beat, noise (boot z -1.57) | 0.0226 vs 0.0234 (+0.0008): beats, noise (boot z +0.06) |
| class_prior | direction/acc | 0.4849 vs 0.4861 (-0.0011): does not beat, noise (DM z -0.05) | 0.4811 vs 0.4853 (-0.0042): does not beat, noise (DM z -0.19) | 0.5026 vs 0.4811 (+0.0215): beats, noise (DM z +0.80) |
| class_prior | direction/bal_acc | 0.4850 vs 0.5000 (-0.0150): does not beat, noise (boot z -1.17) | 0.4831 vs 0.5000 (-0.0169): does not beat, noise (boot z -1.08) | 0.5031 vs 0.5000 (+0.0031): beats, noise (boot z +0.20) |
| zero_delta | delta/rmse | 196.23 vs 196.21 (-0.01, -0.01%): does not beat, noise (DM z -0.15) | 236.10 vs 236.10 (+0.00, +0.00%): does not beat | 269.23 vs 269.11 (-0.13, -0.05%): does not beat, noise (DM z -0.91) |
| zero_delta | delta/mae | 145.68 vs 145.66 (-0.02, -0.02%): does not beat, noise (DM z -0.38) | 175.66 vs 175.66 (+0.00, +0.00%): does not beat | 200.03 vs 199.93 (-0.10, -0.05%): does not beat, noise (DM z -0.94) |
| mean_delta | delta/rmse | 196.23 vs 196.24 (+0.01, +0.00%): beats, noise (DM z +0.11) | 236.10 vs 236.14 (+0.04, +0.02%): beats, noise (DM z +1.09) | 269.23 vs 269.17 (-0.07, -0.02%): does not beat, noise (DM z -0.45) |
| mean_delta | delta/mae | 145.68 vs 145.68 (-0.00, -0.00%): does not beat, noise (DM z -0.07) | 175.66 vs 175.69 (+0.03, +0.02%): beats, noise (DM z +0.94) | 200.03 vs 199.97 (-0.06, -0.03%): does not beat, noise (DM z -0.49) |
| const_var | variance/crps | 105.50 vs 107.87 (+2.37, +2.20%): beats (DM z +6.21) | 127.33 vs 129.50 (+2.16, +1.67%): beats (DM z +4.00) | 145.01 vs 147.86 (+2.85, +1.93%): beats (DM z +4.02) |
| const_var | variance/nll | 6.6473 vs 6.7057 (+0.0584): beats (DM z +3.95) | 6.8441 vs 6.8904 (+0.0463): beats (DM z +2.55) | 6.9669 vs 7.0219 (+0.0550): beats (DM z +2.70) |
| const_var | variance/pit_ks | 0.0348 vs 0.0663 (+0.0315): beats (boot z +5.66) | 0.0328 vs 0.0614 (+0.0286): beats (boot z +4.68) | 0.0340 vs 0.0644 (+0.0304): beats (boot z +4.89) |
| const_var | variance/corr_var_err2_spearman | 0.2498 vs 0.0000 (+0.2498): beats (boot z +6.96) | 0.2243 vs 0.0000 (+0.2243): beats (boot z +5.65) | 0.2275 vs 0.0000 (+0.2275): beats (boot z +5.23) |
