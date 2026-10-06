# Evaluation report - test split - run `20261006T070153Z-1aeff1c-9d77f3bd-A_1aeff1c_r4`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.4007 | 0.4488 | 0.3887 |
| accuracy | 0.5018 | 0.4978 | 0.5082 |
| balanced accuracy | 0.5045 | 0.4993 | 0.5124 |
| precision (up) | 0.5196 | 0.5139 | 0.5348 |
| recall / sensitivity (up) | 0.4052 | 0.4482 | 0.4006 |
| specificity (down) | 0.6039 | 0.5505 | 0.6242 |
| F1 (up) | 0.4553 | 0.4788 | 0.4581 |
| MCC | 0.0093 | -0.0014 | 0.0254 |
| AUC | 0.5048 | 0.5058 | 0.5083 |
| Brier | 0.2513 | 0.2513 | 0.2522 |
| ECE (positive class) | 0.0269 | 0.0339 | 0.0298 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1113 / 1029 / 1569 / 1634 | 1314 / 1243 / 1522 / 1618 | 1221 / 1062 / 1764 / 1827 |
| Gaussian readout: calls up | 0.3757 | n/a (beta = 0: readout is the constant 0.5) | 0.4498 |
| Gaussian readout: MCC | 0.0077 | n/a (beta = 0: readout is the constant 0.5) | -0.0061 |
| Gaussian readout: AUC | 0.4996 | n/a (beta = 0: readout is the constant 0.5) | 0.4993 |
| Gaussian readout: Brier | 0.2502 | n/a (beta = 0: readout is the constant 0.5) | 0.2501 |
| Gaussian readout: ECE | 0.0157 | n/a (beta = 0: readout is the constant 0.5) | 0.0200 |
| Gaussian readout of the raw heads: calls up | 0.3757 | 0.4622 | 0.4498 |
| Gaussian readout of the raw heads: MCC | 0.0077 | 0.0126 | -0.0061 |
| Gaussian readout of the raw heads: AUC | 0.4996 | 0.5011 | 0.4993 |
| Gaussian readout of the raw heads: Brier | 0.2546 | 0.2618 | 0.2617 |
| Gaussian readout of the raw heads: ECE | 0.0440 | 0.0753 | 0.0850 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.27 | 236.10 | 269.21 |
| RMSE ($), raw heads | 198.63 | 244.49 | 278.84 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.70 | 175.66 | 200.02 |
| MAE ($), raw heads | 147.33 | 182.42 | 208.60 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0005 | n/a (beta = 0: served delta is 0) | -0.0007 |
| skill vs zero, raw heads | -0.0248 | -0.0723 | -0.0736 |
| EV, served | -0.0004 | n/a (beta = 0: served delta is 0) | -0.0005 |
| EV, raw heads | -0.0234 | -0.0699 | -0.0695 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0019 | -0.0081 | -0.0073 |
| corr, Spearman, raw heads | -0.0025 | 0.0022 | -0.0061 |
| mean predicted ($), served | -0.47 | 0.00 | -0.60 |
| mean predicted ($), raw heads | -3.42 | -5.72 | -9.18 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3633 | 0.4674 | 0.4505 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.1377 | 0.0000 | 0.0657 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.63 | 127.11 | 145.10 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.6516 | 6.8453 | 6.9689 |
| PIT KS | 0.0360 | 0.0283 | 0.0390 |
| var / err^2 Spearman | 0.2483 | 0.2382 | 0.2272 |
| coverage of the 90% interval | 0.9027 | 0.9059 | 0.9129 |
| width of the 90% interval ($) | 644.27 | 781.37 | 915.85 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0157 | [-0.0221, 0.0543] | NOISE |
| h1 | 0.0132 | [-0.0231, 0.0485] | NOISE |
| h2 | -0.0320 | [-0.0764, 0.0062] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.138 / h1 0.000 / h2 0.066) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8402 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.6675 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5387 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5773 | 0.6327 | 0.6446 | 0.2360 |
| expected if the two signs were independent | 0.5266 | 0.5029 | 0.5122 | 0.1405 |

- P(up) unanimity (all three horizons call the same side): 0.2771
