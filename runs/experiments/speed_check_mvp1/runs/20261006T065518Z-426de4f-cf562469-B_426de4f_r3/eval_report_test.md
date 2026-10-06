# Evaluation report - test split - run `20261006T065518Z-426de4f-cf562469-B_426de4f_r3`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.4382 | 0.6180 | 0.4597 |
| accuracy | 0.5025 | 0.5210 | 0.4886 |
| balanced accuracy | 0.5043 | 0.5175 | 0.4901 |
| precision (up) | 0.5188 | 0.5288 | 0.5081 |
| recall / sensitivity (up) | 0.4423 | 0.6351 | 0.4501 |
| specificity (down) | 0.5662 | 0.4000 | 0.5301 |
| F1 (up) | 0.4775 | 0.5771 | 0.4774 |
| MCC | 0.0086 | 0.0361 | -0.0198 |
| AUC | 0.4894 | 0.5295 | 0.4784 |
| Brier | 0.2532 | 0.2497 | 0.2537 |
| ECE (positive class) | 0.0260 | 0.0109 | 0.0391 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1215 / 1127 / 1471 / 1532 | 1862 / 1659 / 1106 / 1070 | 1372 / 1328 / 1498 / 1676 |
| Gaussian readout: calls up | 0.4473 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0212 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5166 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2500 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0141 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.4473 | 0.4039 | 0.4855 |
| Gaussian readout of the raw heads: MCC | 0.0212 | 0.0349 | 0.0246 |
| Gaussian readout of the raw heads: AUC | 0.5166 | 0.5092 | 0.5109 |
| Gaussian readout of the raw heads: Brier | 0.2539 | 0.2601 | 0.2601 |
| Gaussian readout of the raw heads: ECE | 0.0336 | 0.0575 | 0.0634 |

beta = 0 for h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.21 | 236.10 | 269.11 |
| RMSE ($), raw heads | 197.82 | 250.78 | 279.54 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.65 | 175.66 | 199.93 |
| MAE ($), raw heads | 146.86 | 185.34 | 207.94 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0001 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0165 | -0.1282 | -0.0790 |
| EV, served | 0.0001 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0138 | -0.1193 | -0.0737 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0316 | -0.0032 | -0.0105 |
| corr, Spearman, raw heads | 0.0216 | 0.0089 | 0.0107 |
| mean predicted ($), served | -0.05 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -5.64 | -15.08 | -11.17 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.4610 | 0.4147 | 0.4902 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.0085 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.55 | 127.43 | 145.00 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.6504 | 6.8482 | 6.9726 |
| PIT KS | 0.0304 | 0.0329 | 0.0332 |
| var / err^2 Spearman | 0.2419 | 0.2310 | 0.2188 |
| coverage of the 90% interval | 0.9030 | 0.9059 | 0.9124 |
| width of the 90% interval ($) | 644.50 | 781.37 | 911.88 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0448 | [-0.0855, -0.0105] | INVERTED |
| h1 | 0.0271 | [-0.0126, 0.0686] | NOISE |
| h2 | -0.0487 | [-0.0989, -0.0007] | INVERTED |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.009 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8329 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.6538 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5162 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5999 | 0.5568 | 0.6803 | 0.2333 |
| expected if the two signs were independent | 0.5046 | 0.4789 | 0.5009 | 0.1338 |

- P(up) unanimity (all three horizons call the same side): 0.3003
