# Evaluation report - test split - run `20260924T165923Z-fa75177-dirty-af67ee43`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.3908 | 0.5431 | 0.4025 |
| accuracy | 0.5158 | 0.4941 | 0.5230 |
| balanced accuracy | 0.5189 | 0.4929 | 0.5267 |
| precision (up) | 0.5381 | 0.5081 | 0.5520 |
| recall / sensitivity (up) | 0.4092 | 0.5362 | 0.4281 |
| specificity (down) | 0.6286 | 0.4495 | 0.6253 |
| F1 (up) | 0.4648 | 0.5217 | 0.4823 |
| MCC | 0.0387 | -0.0143 | 0.0544 |
| AUC | 0.5245 | 0.4948 | 0.5288 |
| Brier | 0.2502 | 0.2520 | 0.2501 |
| ECE (positive class) | 0.0239 | 0.0349 | 0.0277 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1124 / 965 / 1633 / 1623 | 1572 / 1522 / 1243 / 1360 | 1305 / 1059 / 1767 / 1743 |
| Gaussian readout: calls up | 0.0000 | 0.0000 | 0.0000 |
| Gaussian readout: MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian readout: AUC | 0.5000 | 0.5000 | 0.5000 |
| Gaussian readout: Brier | 0.2500 | 0.2500 | 0.2500 |
| Gaussian readout: ECE | 0.0139 | 0.0147 | 0.0189 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.21 | 236.10 | 269.11 |
| RMSE ($), raw heads | 197.73 | 247.04 | 279.07 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.66 | 175.66 | 199.93 |
| MAE ($), raw heads | 146.68 | 183.63 | 208.19 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero, raw heads | -0.0155 | -0.0948 | -0.0754 |
| EV, served | 0.0000 | 0.0000 | 0.0000 |
| EV, raw heads | -0.0126 | -0.0880 | -0.0696 |
| corr, Pearson (the same raw and served) | 0.0000 | 0.0000 | 0.0000 |
| corr, Spearman | 0.0000 | 0.0000 | 0.0000 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -6.08 | -12.44 | -11.80 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3572 | 0.3393 | 0.4288 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.62 | 127.34 | 145.09 |
| CRPSS vs constant variance | 0.0209 | 0.0166 | 0.0187 |
| NLL | 6.6535 | 6.8456 | 6.9704 |
| PIT KS | 0.0337 | 0.0284 | 0.0364 |
| var / err^2 Spearman | 0.2489 | 0.2332 | 0.2213 |
| coverage of the 90% interval | 0.9027 | 0.9059 | 0.9124 |
| width of the 90% interval ($) | 643.12 | 781.37 | 911.88 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0115 | [-0.0253, 0.0511] | NOISE |
| h1 | 0.0060 | [-0.0336, 0.0439] | NOISE |
| h2 | 0.0131 | [-0.0306, 0.0510] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8347 | 1.0000 | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.5969 | 1.0000 | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.4608 | 1.0000 | 0.3159 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so every sign and magnitude check on the served deltas is empty; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(delta) against calibrated P(up) > 0.5 (the same for raw and served deltas while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6054 | 0.5738 | 0.5949 | 0.2030 |
| expected if the two signs were independent | 0.5324 | 0.4849 | 0.5144 | 0.1324 |

- P(up) unanimity (all three horizons call the same side): 0.2692

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0387 vs 0.0390 (-0.0003): does not beat, noise (boot z -0.01) | -0.0143 vs 0.0397 (-0.0540): does not beat, noise (boot z -1.58) | 0.0544 vs 0.0411 (+0.0133): beats, noise (boot z +0.33) |
| logreg_lags | direction/auc | 0.5245 vs 0.5241 (+0.0004): beats, noise (boot z +0.02) | 0.4948 vs 0.5228 (-0.0280): does not beat, noise (boot z -1.34) | 0.5288 vs 0.5319 (-0.0031): does not beat, noise (boot z -0.12) |
| logreg_lags | direction/brier | 0.2502 vs 0.2495 (-0.0007): does not beat, noise (DM z -0.54) | 0.2520 vs 0.2493 (-0.0027): does not beat, significantly worse (DM z -1.98) | 0.2501 vs 0.2492 (-0.0009): does not beat, noise (DM z -0.66) |
| logreg_lags | direction/ece_pos | 0.0239 vs 0.0211 (-0.0028): does not beat, noise (boot z -0.36) | 0.0349 vs 0.0197 (-0.0152): does not beat, noise (boot z -1.02) | 0.0277 vs 0.0231 (-0.0046): does not beat, noise (boot z -0.59) |
| logreg_lags | direction/acc | 0.5158 vs 0.5160 (-0.0002): does not beat, noise (DM z -0.01) | 0.4941 vs 0.5166 (-0.0225): does not beat, noise (DM z -1.28) | 0.5230 vs 0.5157 (+0.0073): beats, noise (DM z +0.38) |
| logreg_lags | direction/bal_acc | 0.5189 vs 0.5190 (-0.0002): does not beat, noise (boot z -0.01) | 0.4929 vs 0.5195 (-0.0266): does not beat, noise (boot z -1.58) | 0.5267 vs 0.5200 (+0.0067): beats, noise (boot z +0.34) |
| class_prior | direction/mcc | 0.0387 vs 0.0000 (+0.0387): beats, noise (boot z +1.60) | -0.0143 vs 0.0000 (-0.0143): does not beat, noise (boot z -0.59) | 0.0544 vs 0.0000 (+0.0544): beats (boot z +2.08) |
| class_prior | direction/auc | 0.5245 vs 0.5000 (+0.0245): beats, noise (boot z +1.54) | 0.4948 vs 0.5000 (-0.0052): does not beat, noise (boot z -0.33) | 0.5288 vs 0.5000 (+0.0288): beats, noise (boot z +1.66) |
| class_prior | direction/brier | 0.2502 vs 0.2502 (+0.0000): beats, noise (DM z +0.01) | 0.2520 vs 0.2502 (-0.0018): does not beat, noise (DM z -1.53) | 0.2501 vs 0.2502 (+0.0001): beats, noise (DM z +0.08) |
| class_prior | direction/ece_pos | 0.0239 vs 0.0208 (-0.0031): does not beat, noise (boot z -0.49) | 0.0349 vs 0.0196 (-0.0153): does not beat, noise (boot z -0.90) | 0.0277 vs 0.0234 (-0.0043): does not beat, noise (boot z -0.85) |
| class_prior | direction/acc | 0.5158 vs 0.4861 (+0.0297): beats, noise (DM z +1.62) | 0.4941 vs 0.4853 (+0.0088): beats, noise (DM z +0.34) | 0.5230 vs 0.4811 (+0.0419): beats, noise (DM z +1.85) |
| class_prior | direction/bal_acc | 0.5189 vs 0.5000 (+0.0189): beats, noise (boot z +1.59) | 0.4929 vs 0.5000 (-0.0071): does not beat, noise (boot z -0.59) | 0.5267 vs 0.5000 (+0.0267): beats (boot z +2.08) |
| zero_delta | delta/rmse | 196.21 vs 196.21 (+0.00, +0.00%): does not beat | 236.10 vs 236.10 (+0.00, +0.00%): does not beat | 269.11 vs 269.11 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 145.66 vs 145.66 (+0.00, +0.00%): does not beat | 175.66 vs 175.66 (+0.00, +0.00%): does not beat | 199.93 vs 199.93 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 196.21 vs 196.24 (+0.02, +0.01%): beats, noise (DM z +1.05) | 236.10 vs 236.14 (+0.04, +0.02%): beats, noise (DM z +1.09) | 269.11 vs 269.17 (+0.06, +0.02%): beats, noise (DM z +1.12) |
| mean_delta | delta/mae | 145.66 vs 145.68 (+0.02, +0.01%): beats, noise (DM z +1.01) | 175.66 vs 175.69 (+0.03, +0.02%): beats, noise (DM z +0.94) | 199.93 vs 199.97 (+0.05, +0.02%): beats, noise (DM z +0.91) |
| const_var | variance/crps | 105.62 vs 107.87 (+2.25, +2.09%): beats (DM z +5.80) | 127.34 vs 129.50 (+2.16, +1.66%): beats (DM z +3.59) | 145.09 vs 147.86 (+2.77, +1.87%): beats (DM z +4.08) |
| const_var | variance/nll | 6.6535 vs 6.7057 (+0.0522): beats (DM z +3.55) | 6.8456 vs 6.8904 (+0.0447): beats (DM z +2.13) | 6.9704 vs 7.0219 (+0.0515): beats (DM z +2.70) |
| const_var | variance/pit_ks | 0.0337 vs 0.0663 (+0.0326): beats (boot z +5.99) | 0.0284 vs 0.0614 (+0.0330): beats (boot z +4.82) | 0.0364 vs 0.0644 (+0.0280): beats (boot z +4.58) |
| const_var | variance/corr_var_err2_spearman | 0.2489 vs 0.0000 (+0.2489): beats (boot z +6.99) | 0.2332 vs 0.0000 (+0.2332): beats (boot z +5.75) | 0.2213 vs 0.0000 (+0.2213): beats (boot z +5.29) |
