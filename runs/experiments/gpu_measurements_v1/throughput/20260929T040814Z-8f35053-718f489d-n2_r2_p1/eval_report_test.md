# Evaluation report - test split - run `20260929T040814Z-8f35053-718f489d-n2_r2_p1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.7528 | 0.3522 | 0.4870 |
| accuracy | 0.5082 | 0.5085 | 0.5049 |
| balanced accuracy | 0.5057 | 0.5097 | 0.5048 |
| precision (up) | 0.5087 | 0.5175 | 0.4993 |
| recall / sensitivity (up) | 0.7584 | 0.3618 | 0.4918 |
| specificity (down) | 0.2530 | 0.6575 | 0.5177 |
| F1 (up) | 0.6090 | 0.4259 | 0.4955 |
| MCC | 0.0132 | 0.0202 | 0.0096 |
| AUC | 0.5047 | 0.5110 | 0.5097 |
| Brier | 0.2579 | 0.2539 | 0.2568 |
| ECE (positive class) | 0.0550 | 0.0544 | 0.0514 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1868 / 1804 / 611 / 595 | 961 / 896 / 1720 / 1695 | 1356 / 1360 / 1460 / 1401 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.5545 | 0.1677 | 0.1194 |
| Gaussian readout of the raw heads: MCC | -0.0073 | -0.0095 | 0.0517 |
| Gaussian readout of the raw heads: AUC | 0.5040 | 0.4864 | 0.5235 |
| Gaussian readout of the raw heads: Brier | 0.2510 | 0.2540 | 0.2543 |
| Gaussian readout of the raw heads: ECE | 0.0378 | 0.0493 | 0.0656 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.09 | 252.19 | 290.31 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 138.58 | 171.73 | 201.84 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0066 | -0.0159 | -0.0312 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0050 | -0.0098 | -0.0098 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0385 | 0.0045 | 0.0524 |
| corr, Spearman, raw heads | 0.0152 | -0.0100 | 0.0602 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 6.57 | -23.58 | -47.08 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.5345 | 0.1549 | 0.1142 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.82 | 131.03 | 152.16 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7253 | 6.9343 | 7.0834 |
| PIT KS | 0.0995 | 0.1195 | 0.1154 |
| var / err^2 Spearman | 0.2524 | 0.2604 | 0.2309 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0180 | [-0.0626, 0.0238] | NOISE |
| h1 | 0.0175 | [-0.0322, 0.0583] | NOISE |
| h2 | 0.0191 | [-0.0165, 0.0588] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7555 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.8907 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.6833 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5094 | 0.5695 | 0.5669 | 0.1310 |
| expected if the two signs were independent | 0.5186 | 0.5851 | 0.5160 | 0.1326 |

- P(up) unanimity (all three horizons call the same side): 0.1540
