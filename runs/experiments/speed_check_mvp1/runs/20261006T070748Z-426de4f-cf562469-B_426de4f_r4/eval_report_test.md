# Evaluation report - test split - run `20261006T070748Z-426de4f-cf562469-B_426de4f_r4`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.3978 | 0.4197 | 0.3252 |
| accuracy | 0.4969 | 0.4989 | 0.4971 |
| balanced accuracy | 0.4998 | 0.5012 | 0.5037 |
| precision (up) | 0.5136 | 0.5161 | 0.5246 |
| recall / sensitivity (up) | 0.3975 | 0.4209 | 0.3287 |
| specificity (down) | 0.6020 | 0.5816 | 0.6787 |
| F1 (up) | 0.4482 | 0.4636 | 0.4042 |
| MCC | -0.0005 | 0.0025 | 0.0079 |
| AUC | 0.4993 | 0.5040 | 0.4919 |
| Brier | 0.2513 | 0.2514 | 0.2519 |
| ECE (positive class) | 0.0267 | 0.0303 | 0.0305 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1092 / 1034 / 1564 / 1655 | 1234 / 1157 / 1608 / 1698 | 1002 / 908 / 1918 / 2046 |
| Gaussian readout: calls up | 0.3951 | n/a (beta = 0: readout is the constant 0.5) | 0.3596 |
| Gaussian readout: MCC | -0.0088 | n/a (beta = 0: readout is the constant 0.5) | 0.0228 |
| Gaussian readout: AUC | 0.4973 | n/a (beta = 0: readout is the constant 0.5) | 0.4951 |
| Gaussian readout: Brier | 0.2500 | n/a (beta = 0: readout is the constant 0.5) | 0.2502 |
| Gaussian readout: ECE | 0.0142 | n/a (beta = 0: readout is the constant 0.5) | 0.0212 |
| Gaussian readout of the raw heads: calls up | 0.3951 | 0.3519 | 0.3596 |
| Gaussian readout of the raw heads: MCC | -0.0088 | 0.0244 | 0.0228 |
| Gaussian readout of the raw heads: AUC | 0.4973 | 0.4975 | 0.4951 |
| Gaussian readout of the raw heads: Brier | 0.2536 | 0.2580 | 0.2597 |
| Gaussian readout of the raw heads: ECE | 0.0431 | 0.0533 | 0.0611 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.21 | 236.10 | 269.24 |
| RMSE ($), raw heads | 198.41 | 242.38 | 277.49 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.67 | 175.66 | 200.11 |
| MAE ($), raw heads | 147.69 | 181.23 | 208.20 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0000 | n/a (beta = 0: served delta is 0) | -0.0010 |
| skill vs zero, raw heads | -0.0225 | -0.0539 | -0.0633 |
| EV, served | 0.0000 | n/a (beta = 0: served delta is 0) | -0.0005 |
| EV, raw heads | -0.0197 | -0.0478 | -0.0552 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0061 | 0.0059 | -0.0006 |
| corr, Spearman, raw heads | -0.0102 | -0.0132 | -0.0210 |
| mean predicted ($), served | -0.10 | 0.00 | -1.38 |
| mean predicted ($), raw heads | -6.00 | -11.34 | -14.94 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3800 | 0.3465 | 0.3530 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.0165 | 0.0000 | 0.0921 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.66 | 127.06 | 145.09 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.6454 | 6.8413 | 6.9685 |
| PIT KS | 0.0382 | 0.0298 | 0.0401 |
| var / err^2 Spearman | 0.2479 | 0.2342 | 0.2254 |
| coverage of the 90% interval | 0.9042 | 0.9059 | 0.9121 |
| width of the 90% interval ($) | 646.73 | 781.37 | 911.41 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0037 | [-0.0426, 0.0473] | NOISE |
| h1 | 0.0102 | [-0.0280, 0.0475] | NOISE |
| h2 | -0.0377 | [-0.0810, 0.0009] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.016 / h1 0.000 / h2 0.092) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8129 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.7392 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5857 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6300 | 0.6610 | 0.6824 | 0.3173 |
| expected if the two signs were independent | 0.5284 | 0.5233 | 0.5550 | 0.1907 |

- P(up) unanimity (all three horizons call the same side): 0.3642
