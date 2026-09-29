# Evaluation report - test split - run `20260929T040814Z-8f35053-fd969f30-n2_r2_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.1521 | 0.1210 | 0.4122 |
| accuracy | 0.5254 | 0.5027 | 0.5385 |
| balanced accuracy | 0.5288 | 0.5055 | 0.5375 |
| precision (up) | 0.5997 | 0.5266 | 0.5398 |
| recall / sensitivity (up) | 0.1807 | 0.1265 | 0.4501 |
| specificity (down) | 0.8770 | 0.8846 | 0.6248 |
| F1 (up) | 0.2777 | 0.2040 | 0.4909 |
| MCC | 0.0803 | 0.0170 | 0.0761 |
| AUC | 0.5508 | 0.5024 | 0.5532 |
| Brier | 0.2527 | 0.2601 | 0.2495 |
| ECE (positive class) | 0.0658 | 0.0731 | 0.0141 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 445 / 297 / 2118 / 2018 | 336 / 302 / 2314 / 2320 | 1241 / 1058 / 1762 / 1516 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.3674 | 0.1003 | 0.0000 |
| Gaussian readout of the raw heads: MCC | 0.0452 | 0.0347 | 0.0000 |
| Gaussian readout of the raw heads: AUC | 0.5271 | 0.5429 | 0.5680 |
| Gaussian readout of the raw heads: Brier | 0.2499 | 0.2507 | 0.2496 |
| Gaussian readout of the raw heads: ECE | 0.0217 | 0.0437 | 0.0379 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 211.28 | 250.17 | 285.51 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 137.81 | 169.33 | 196.08 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | 0.0011 | 0.0002 | 0.0026 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | 0.0021 | 0.0094 | 0.0112 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0524 | 0.0970 | 0.1343 |
| corr, Spearman, raw heads | 0.0477 | 0.0828 | 0.1171 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -9.24 | -27.83 | -31.94 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.3817 | 0.0992 | 0.0000 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 107.57 | 131.85 | 151.19 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7509 | 6.9399 | 7.0730 |
| PIT KS | 0.1157 | 0.1231 | 0.1078 |
| var / err^2 Spearman | 0.2539 | 0.2449 | 0.2371 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0113 | [-0.0319, 0.0526] | NOISE |
| h1 | -0.0363 | [-0.0953, 0.0266] | NOISE |
| h2 | 0.0566 | [0.0164, 0.1037] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7634 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.7240 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.4896 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5868 | 0.8154 | 0.5667 | 0.3585 |
| expected if the two signs were independent | 0.5845 | 0.8119 | 0.5667 | 0.2958 |

- P(up) unanimity (all three horizons call the same side): 0.4603
