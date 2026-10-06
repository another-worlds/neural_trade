# Evaluation report - test split - run `20261004T073704Z-d1f6fa9-11993eec`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.5139 | 0.1861 | 0.5213 |
| accuracy | 0.4919 | 0.5036 | 0.4743 |
| balanced accuracy | 0.4915 | 0.5128 | 0.4735 |
| precision (up) | 0.5056 | 0.5491 | 0.4935 |
| recall / sensitivity (up) | 0.5056 | 0.1985 | 0.4957 |
| specificity (down) | 0.4773 | 0.8271 | 0.4512 |
| F1 (up) | 0.5056 | 0.2916 | 0.4946 |
| MCC | -0.0171 | 0.0329 | -0.0531 |
| AUC | 0.5001 | 0.5194 | 0.4703 |
| Brier | 0.2520 | 0.2527 | 0.2554 |
| ECE (positive class) | 0.0458 | 0.0533 | 0.0688 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1389 / 1358 / 1240 / 1358 | 582 / 478 / 2287 / 2350 | 1511 / 1551 / 1275 / 1537 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6376 | 0.5292 | 0.6120 |
| Gaussian readout of the raw heads: MCC | -0.0596 | -0.0343 | -0.0276 |
| Gaussian readout of the raw heads: AUC | 0.4722 | 0.4720 | 0.4846 |
| Gaussian readout of the raw heads: Brier | 0.2534 | 0.2567 | 0.2553 |
| Gaussian readout of the raw heads: ECE | 0.0551 | 0.0608 | 0.0624 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.21 | 236.10 | 269.11 |
| RMSE ($), raw heads | 197.62 | 239.49 | 272.25 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.66 | 175.66 | 199.93 |
| MAE ($), raw heads | 146.74 | 178.17 | 202.32 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0144 | -0.0289 | -0.0235 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0147 | -0.0269 | -0.0245 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0618 | -0.0506 | -0.0209 |
| corr, Spearman, raw heads | -0.0562 | -0.0567 | -0.0346 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 1.05 | -4.82 | 3.36 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.6625 | 0.5478 | 0.6229 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.48 | 127.07 | 145.15 |
| CRPSS vs constant variance | 0.0221 | 0.0188 | 0.0183 |
| NLL | 6.6679 | 6.8733 | 7.0091 |
| PIT KS | 0.0209 | 0.0286 | 0.0335 |
| var / err^2 Spearman | 0.2619 | 0.2426 | 0.2387 |
| coverage of the 90% interval | 0.9027 | 0.9059 | 0.9124 |
| width of the 90% interval ($) | 643.12 | 781.37 | 911.88 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0224 | [-0.0195, 0.0674] | NOISE |
| h1 | 0.0171 | [-0.0349, 0.0675] | NOISE |
| h2 | -0.0048 | [-0.0605, 0.0490] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6838 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.7359 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.4453 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7063 | 0.5253 | 0.7557 | 0.3089 |
| expected if the two signs were independent | 0.5070 | 0.4693 | 0.5059 | 0.1551 |

- P(up) unanimity (all three horizons call the same side): 0.3705

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0171 vs 0.0390 (-0.0561): does not beat, noise (boot z -1.77) | 0.0329 vs 0.0397 (-0.0067): does not beat, noise (boot z -0.19) | -0.0531 vs 0.0411 (-0.0942): does not beat, significantly worse (boot z -2.30) |
| logreg_lags | direction/auc | 0.5001 vs 0.5241 (-0.0241): does not beat, noise (boot z -1.31) | 0.5194 vs 0.5228 (-0.0035): does not beat, noise (boot z -0.15) | 0.4703 vs 0.5319 (-0.0616): does not beat, significantly worse (boot z -2.65) |
| logreg_lags | direction/brier | 0.2520 vs 0.2495 (-0.0025): does not beat, noise (DM z -1.73) | 0.2527 vs 0.2493 (-0.0034): does not beat, noise (DM z -1.71) | 0.2554 vs 0.2492 (-0.0062): does not beat, significantly worse (DM z -3.21) |
| logreg_lags | direction/ece_pos | 0.0458 vs 0.0211 (-0.0247): does not beat, noise (boot z -1.46) | 0.0533 vs 0.0197 (-0.0337): does not beat, significantly worse (boot z -3.44) | 0.0688 vs 0.0231 (-0.0456): does not beat, significantly worse (boot z -1.98) |
| logreg_lags | direction/acc | 0.4919 vs 0.5160 (-0.0241): does not beat, noise (DM z -1.41) | 0.5036 vs 0.5166 (-0.0130): does not beat, noise (DM z -0.74) | 0.4743 vs 0.5157 (-0.0414): does not beat, noise (DM z -1.96) |
| logreg_lags | direction/bal_acc | 0.4915 vs 0.5190 (-0.0276): does not beat, noise (boot z -1.77) | 0.5128 vs 0.5195 (-0.0066): does not beat, noise (boot z -0.40) | 0.4735 vs 0.5200 (-0.0466): does not beat, significantly worse (boot z -2.31) |
| class_prior | direction/mcc | -0.0171 vs 0.0000 (-0.0171): does not beat, noise (boot z -0.64) | 0.0329 vs 0.0000 (+0.0329): beats, noise (boot z +1.40) | -0.0531 vs 0.0000 (-0.0531): does not beat, noise (boot z -1.51) |
| class_prior | direction/auc | 0.5001 vs 0.5000 (+0.0001): beats, noise (boot z +0.00) | 0.5194 vs 0.5000 (+0.0194): beats, noise (boot z +1.09) | 0.4703 vs 0.5000 (-0.0297): does not beat, noise (boot z -1.33) |
| class_prior | direction/brier | 0.2520 vs 0.2502 (-0.0018): does not beat, noise (DM z -1.31) | 0.2527 vs 0.2502 (-0.0026): does not beat, noise (DM z -1.34) | 0.2554 vs 0.2502 (-0.0052): does not beat, significantly worse (DM z -2.61) |
| class_prior | direction/ece_pos | 0.0458 vs 0.0208 (-0.0249): does not beat, noise (boot z -1.33) | 0.0533 vs 0.0196 (-0.0337): does not beat, significantly worse (boot z -5.07) | 0.0688 vs 0.0234 (-0.0454): does not beat, noise (boot z -1.87) |
| class_prior | direction/acc | 0.4919 vs 0.4861 (+0.0058): beats, noise (DM z +0.26) | 0.5036 vs 0.4853 (+0.0183): beats, noise (DM z +1.48) | 0.4743 vs 0.4811 (-0.0068): does not beat, noise (DM z -0.24) |
| class_prior | direction/bal_acc | 0.4915 vs 0.5000 (-0.0085): does not beat, noise (boot z -0.64) | 0.5128 vs 0.5000 (+0.0128): beats, noise (boot z +1.39) | 0.4735 vs 0.5000 (-0.0265): does not beat, noise (boot z -1.51) |
| zero_delta | delta/rmse | 196.21 vs 196.21 (+0.00, +0.00%): does not beat | 236.10 vs 236.10 (+0.00, +0.00%): does not beat | 269.11 vs 269.11 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 145.66 vs 145.66 (+0.00, +0.00%): does not beat | 175.66 vs 175.66 (+0.00, +0.00%): does not beat | 199.93 vs 199.93 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 196.21 vs 196.24 (+0.02, +0.01%): beats, noise (DM z +1.05) | 236.10 vs 236.14 (+0.04, +0.02%): beats, noise (DM z +1.09) | 269.11 vs 269.17 (+0.06, +0.02%): beats, noise (DM z +1.12) |
| mean_delta | delta/mae | 145.66 vs 145.68 (+0.02, +0.01%): beats, noise (DM z +1.01) | 175.66 vs 175.69 (+0.03, +0.02%): beats, noise (DM z +0.94) | 199.93 vs 199.97 (+0.05, +0.02%): beats, noise (DM z +0.91) |
| const_var | variance/crps | 105.48 vs 107.87 (+2.38, +2.21%): beats (DM z +5.17) | 127.07 vs 129.50 (+2.43, +1.88%): beats (DM z +3.72) | 145.15 vs 147.86 (+2.70, +1.83%): beats (DM z +3.05) |
| const_var | variance/nll | 6.6679 vs 6.7057 (+0.0378): beats, noise (DM z +1.85) | 6.8733 vs 6.8904 (+0.0170): beats, noise (DM z +0.63) | 7.0091 vs 7.0219 (+0.0128): beats, noise (DM z +0.41) |
| const_var | variance/pit_ks | 0.0209 vs 0.0663 (+0.0454): beats (boot z +3.69) | 0.0286 vs 0.0614 (+0.0328): beats (boot z +2.08) | 0.0335 vs 0.0644 (+0.0309): beats, noise (boot z +1.86) |
| const_var | variance/corr_var_err2_spearman | 0.2619 vs 0.0000 (+0.2619): beats (boot z +7.63) | 0.2426 vs 0.0000 (+0.2426): beats (boot z +6.82) | 0.2387 vs 0.0000 (+0.2387): beats (boot z +5.79) |
