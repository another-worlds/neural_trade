# Evaluation report - test split - run `20261006T062439Z-1aeff1c-9d77f3bd-A_1aeff1c_r1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.5029 | 0.5485 | 0.2497 |
| accuracy | 0.4853 | 0.5101 | 0.5007 |
| balanced accuracy | 0.4852 | 0.5087 | 0.5102 |
| precision (up) | 0.4993 | 0.5226 | 0.5392 |
| recall / sensitivity (up) | 0.4885 | 0.5570 | 0.2595 |
| specificity (down) | 0.4819 | 0.4604 | 0.7608 |
| F1 (up) | 0.4938 | 0.5392 | 0.3504 |
| MCC | -0.0295 | 0.0174 | 0.0234 |
| AUC | 0.4865 | 0.5110 | 0.5044 |
| Brier | 0.2518 | 0.2502 | 0.2519 |
| ECE (positive class) | 0.0412 | 0.0152 | 0.0376 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1342 / 1346 / 1252 / 1405 | 1633 / 1492 / 1273 / 1299 | 791 / 676 / 2150 / 2257 |
| Gaussian readout: calls up | 0.3130 | n/a (beta = 0: readout is the constant 0.5) | 0.2242 |
| Gaussian readout: MCC | -0.0281 | n/a (beta = 0: readout is the constant 0.5) | 0.0495 |
| Gaussian readout: AUC | 0.4607 | n/a (beta = 0: readout is the constant 0.5) | 0.4976 |
| Gaussian readout: Brier | 0.2511 | n/a (beta = 0: readout is the constant 0.5) | 0.2510 |
| Gaussian readout: ECE | 0.0269 | n/a (beta = 0: readout is the constant 0.5) | 0.0305 |
| Gaussian readout of the raw heads: calls up | 0.3130 | 0.4165 | 0.2242 |
| Gaussian readout of the raw heads: MCC | -0.0281 | -0.0273 | 0.0495 |
| Gaussian readout of the raw heads: AUC | 0.4607 | 0.4761 | 0.4976 |
| Gaussian readout of the raw heads: Brier | 0.2555 | 0.2601 | 0.2601 |
| Gaussian readout of the raw heads: ECE | 0.0492 | 0.0777 | 0.0675 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.57 | 236.10 | 269.86 |
| RMSE ($), raw heads | 198.22 | 241.87 | 276.88 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 146.03 | 175.66 | 200.52 |
| MAE ($), raw heads | 147.48 | 180.12 | 206.48 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0036 | n/a (beta = 0: served delta is 0) | -0.0056 |
| skill vs zero, raw heads | -0.0206 | -0.0495 | -0.0586 |
| EV, served | -0.0030 | n/a (beta = 0: served delta is 0) | -0.0027 |
| EV, raw heads | -0.0174 | -0.0432 | -0.0368 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0421 | -0.0173 | -0.0102 |
| corr, Spearman, raw heads | -0.0690 | -0.0455 | -0.0117 |
| mean predicted ($), served | -1.79 | 0.00 | -6.85 |
| mean predicted ($), raw heads | -6.45 | -11.63 | -29.35 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3042 | 0.4259 | 0.2282 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.2779 | 0.0000 | 0.2335 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.87 | 126.86 | 145.38 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.6512 | 6.8406 | 6.9695 |
| PIT KS | 0.0418 | 0.0259 | 0.0474 |
| var / err^2 Spearman | 0.2481 | 0.2347 | 0.2202 |
| coverage of the 90% interval | 0.9035 | 0.9059 | 0.9133 |
| width of the 90% interval ($) | 646.52 | 781.37 | 918.53 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0018 | [-0.0400, 0.0484] | NOISE |
| h1 | -0.0018 | [-0.0448, 0.0412] | NOISE |
| h2 | 0.0070 | [-0.0501, 0.0535] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.278 / h1 0.000 / h2 0.233) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8148 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.7063 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5539 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6129 | 0.5462 | 0.6821 | 0.2489 |
| expected if the two signs were independent | 0.5032 | 0.4929 | 0.6421 | 0.1722 |

- P(up) unanimity (all three horizons call the same side): 0.2913
