# Evaluation report - test split - run `20260929T184249Z-e0bf5f2-0d8e137f`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.4395 | 0.3867 | 0.4043 |
| accuracy | 0.5166 | 0.5164 | 0.4946 |
| balanced accuracy | 0.5183 | 0.5198 | 0.4982 |
| precision (up) | 0.5347 | 0.5402 | 0.5166 |
| recall / sensitivity (up) | 0.4572 | 0.4059 | 0.4026 |
| specificity (down) | 0.5793 | 0.6336 | 0.5938 |
| F1 (up) | 0.4929 | 0.4635 | 0.4525 |
| MCC | 0.0368 | 0.0405 | -0.0037 |
| AUC | 0.5174 | 0.5284 | 0.4853 |
| Brier | 0.2508 | 0.2496 | 0.2525 |
| ECE (positive class) | 0.0209 | 0.0247 | 0.0283 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1256 / 1093 / 1505 / 1491 | 1190 / 1013 / 1752 / 1742 | 1227 / 1148 / 1678 / 1821 |
| Gaussian readout: calls up | 0.3209 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0093 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5104 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2503 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0191 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.3209 | 0.4532 | 0.3769 |
| Gaussian readout of the raw heads: MCC | 0.0093 | 0.0121 | 0.0001 |
| Gaussian readout of the raw heads: AUC | 0.5104 | 0.5023 | 0.4941 |
| Gaussian readout of the raw heads: Brier | 0.2522 | 0.2574 | 0.2612 |
| Gaussian readout of the raw heads: ECE | 0.0305 | 0.0567 | 0.0730 |

beta = 0 for h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.51 | 236.10 | 269.11 |
| RMSE ($), raw heads | 197.82 | 244.56 | 278.22 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.83 | 175.66 | 199.93 |
| MAE ($), raw heads | 146.68 | 180.65 | 206.28 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0030 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0164 | -0.0729 | -0.0689 |
| EV, served | -0.0022 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0136 | -0.0694 | -0.0612 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0138 | -0.0141 | -0.0312 |
| corr, Spearman, raw heads | 0.0055 | -0.0136 | -0.0199 |
| mean predicted ($), served | -2.04 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -5.96 | -7.68 | -14.43 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3061 | 0.4338 | 0.3612 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.3428 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.44 | 126.96 | 144.44 |
| CRPSS vs constant variance | 0.0225 | 0.0196 | 0.0231 |
| NLL | 6.6414 | 6.8404 | 6.9588 |
| PIT KS | 0.0346 | 0.0281 | 0.0315 |
| var / err^2 Spearman | 0.2642 | 0.2441 | 0.2508 |
| coverage of the 90% interval | 0.9020 | 0.9059 | 0.9124 |
| width of the 90% interval ($) | 643.48 | 781.37 | 911.88 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0094 | [-0.0255, 0.0460] | NOISE |
| h1 | 0.0111 | [-0.0281, 0.0503] | NOISE |
| h2 | -0.0280 | [-0.0704, 0.0136] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.343 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8676 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.6874 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5681 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5404 | 0.6524 | 0.6227 | 0.2407 |
| expected if the two signs were independent | 0.5263 | 0.5151 | 0.5309 | 0.1615 |

- P(up) unanimity (all three horizons call the same side): 0.3009

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0368 vs 0.0390 (-0.0022): does not beat, noise (boot z -0.06) | 0.0405 vs 0.0397 (+0.0009): beats, noise (boot z +0.03) | -0.0037 vs 0.0411 (-0.0449): does not beat, noise (boot z -1.27) |
| logreg_lags | direction/auc | 0.5174 vs 0.5241 (-0.0068): does not beat, noise (boot z -0.34) | 0.5284 vs 0.5228 (+0.0056): beats, noise (boot z +0.28) | 0.4853 vs 0.5319 (-0.0465): does not beat, significantly worse (boot z -2.10) |
| logreg_lags | direction/brier | 0.2508 vs 0.2495 (-0.0013): does not beat, noise (DM z -1.01) | 0.2496 vs 0.2493 (-0.0003): does not beat, noise (DM z -0.22) | 0.2525 vs 0.2492 (-0.0033): does not beat, significantly worse (DM z -3.00) |
| logreg_lags | direction/ece_pos | 0.0209 vs 0.0211 (+0.0001): beats, noise (boot z +0.02) | 0.0247 vs 0.0197 (-0.0050): does not beat, noise (boot z -0.68) | 0.0283 vs 0.0231 (-0.0052): does not beat, noise (boot z -0.45) |
| logreg_lags | direction/acc | 0.5166 vs 0.5160 (+0.0006): beats, noise (DM z +0.03) | 0.5164 vs 0.5166 (-0.0002): does not beat, noise (DM z -0.01) | 0.4946 vs 0.5157 (-0.0211): does not beat, noise (DM z -1.22) |
| logreg_lags | direction/bal_acc | 0.5183 vs 0.5190 (-0.0008): does not beat, noise (boot z -0.04) | 0.5198 vs 0.5195 (+0.0003): beats, noise (boot z +0.02) | 0.4982 vs 0.5200 (-0.0219): does not beat, noise (boot z -1.27) |
| class_prior | direction/mcc | 0.0368 vs 0.0000 (+0.0368): beats, noise (boot z +1.46) | 0.0405 vs 0.0000 (+0.0405): beats, noise (boot z +1.72) | -0.0037 vs 0.0000 (-0.0037): does not beat, noise (boot z -0.15) |
| class_prior | direction/auc | 0.5174 vs 0.5000 (+0.0174): beats, noise (boot z +1.12) | 0.5284 vs 0.5000 (+0.0284): beats, noise (boot z +1.86) | 0.4853 vs 0.5000 (-0.0147): does not beat, noise (boot z -0.86) |
| class_prior | direction/brier | 0.2508 vs 0.2502 (-0.0006): does not beat, noise (DM z -0.60) | 0.2496 vs 0.2502 (+0.0005): beats, noise (DM z +0.40) | 0.2525 vs 0.2502 (-0.0023): does not beat, significantly worse (DM z -2.41) |
| class_prior | direction/ece_pos | 0.0209 vs 0.0208 (-0.0001): does not beat, noise (boot z -0.01) | 0.0247 vs 0.0196 (-0.0051): does not beat, noise (boot z -0.93) | 0.0283 vs 0.0234 (-0.0049): does not beat, noise (boot z -0.43) |
| class_prior | direction/acc | 0.5166 vs 0.4861 (+0.0305): beats, noise (DM z +1.55) | 0.5164 vs 0.4853 (+0.0311): beats, noise (DM z +1.58) | 0.4946 vs 0.4811 (+0.0134): beats, noise (DM z +0.60) |
| class_prior | direction/bal_acc | 0.5183 vs 0.5000 (+0.0183): beats, noise (boot z +1.46) | 0.5198 vs 0.5000 (+0.0198): beats, noise (boot z +1.72) | 0.4982 vs 0.5000 (-0.0018): does not beat, noise (boot z -0.15) |
| zero_delta | delta/rmse | 196.51 vs 196.21 (-0.30, -0.15%): does not beat, noise (DM z -1.00) | 236.10 vs 236.10 (+0.00, +0.00%): does not beat | 269.11 vs 269.11 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 145.83 vs 145.66 (-0.17, -0.12%): does not beat, noise (DM z -0.86) | 175.66 vs 175.66 (+0.00, +0.00%): does not beat | 199.93 vs 199.93 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 196.51 vs 196.24 (-0.27, -0.14%): does not beat, noise (DM z -0.94) | 236.10 vs 236.14 (+0.04, +0.02%): beats, noise (DM z +1.09) | 269.11 vs 269.17 (+0.06, +0.02%): beats, noise (DM z +1.12) |
| mean_delta | delta/mae | 145.83 vs 145.68 (-0.15, -0.10%): does not beat, noise (DM z -0.78) | 175.66 vs 175.69 (+0.03, +0.02%): beats, noise (DM z +0.94) | 199.93 vs 199.97 (+0.05, +0.02%): beats, noise (DM z +0.91) |
| const_var | variance/crps | 105.44 vs 107.87 (+2.43, +2.25%): beats (DM z +5.71) | 126.96 vs 129.50 (+2.54, +1.96%): beats (DM z +4.61) | 144.44 vs 147.86 (+3.41, +2.31%): beats (DM z +4.83) |
| const_var | variance/nll | 6.6414 vs 6.7057 (+0.0644): beats (DM z +4.06) | 6.8404 vs 6.8904 (+0.0500): beats (DM z +2.42) | 6.9588 vs 7.0219 (+0.0631): beats (DM z +3.01) |
| const_var | variance/pit_ks | 0.0346 vs 0.0663 (+0.0317): beats (boot z +5.32) | 0.0281 vs 0.0614 (+0.0334): beats (boot z +5.25) | 0.0315 vs 0.0644 (+0.0329): beats (boot z +5.35) |
| const_var | variance/corr_var_err2_spearman | 0.2642 vs 0.0000 (+0.2642): beats (boot z +7.46) | 0.2441 vs 0.0000 (+0.2441): beats (boot z +6.16) | 0.2508 vs 0.0000 (+0.2508): beats (boot z +5.85) |
