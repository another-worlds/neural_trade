# Evaluation report - test split - run `20261006T063657Z-1aeff1c-9d77f3bd-A_1aeff1c_r2`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.4137 | 0.6001 | 0.4132 |
| accuracy | 0.5098 | 0.5094 | 0.5194 |
| balanced accuracy | 0.5122 | 0.5065 | 0.5227 |
| precision (up) | 0.5287 | 0.5200 | 0.5464 |
| recall / sensitivity (up) | 0.4256 | 0.6064 | 0.4350 |
| specificity (down) | 0.5989 | 0.4065 | 0.6104 |
| F1 (up) | 0.4716 | 0.5599 | 0.4844 |
| MCC | 0.0248 | 0.0132 | 0.0461 |
| AUC | 0.5161 | 0.5142 | 0.5268 |
| Brier | 0.2500 | 0.2497 | 0.2499 |
| ECE (positive class) | 0.0203 | 0.0167 | 0.0239 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1169 / 1042 / 1556 / 1578 | 1778 / 1641 / 1124 / 1154 | 1326 / 1101 / 1725 / 1722 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.4118 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0476 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.5133 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.2501 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0207 |
| Gaussian readout of the raw heads: calls up | 0.4110 | 0.4293 | 0.4118 |
| Gaussian readout of the raw heads: MCC | 0.0144 | 0.0306 | 0.0476 |
| Gaussian readout of the raw heads: AUC | 0.5054 | 0.5121 | 0.5133 |
| Gaussian readout of the raw heads: Brier | 0.2519 | 0.2571 | 0.2563 |
| Gaussian readout of the raw heads: ECE | 0.0314 | 0.0527 | 0.0441 |

beta = 0 for h0, h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.21 | 236.10 | 269.25 |
| RMSE ($), raw heads | 197.32 | 245.12 | 276.11 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.66 | 175.66 | 200.09 |
| MAE ($), raw heads | 146.69 | 182.42 | 206.22 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | -0.0011 |
| skill vs zero, raw heads | -0.0113 | -0.0779 | -0.0527 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | -0.0007 |
| EV, raw heads | -0.0106 | -0.0771 | -0.0501 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0308 | 0.0156 | 0.0143 |
| corr, Spearman, raw heads | 0.0036 | 0.0146 | 0.0156 |
| mean predicted ($), served | 0.00 | 0.00 | -1.17 |
| mean predicted ($), raw heads | -1.83 | -2.43 | -6.31 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.4197 | 0.4254 | 0.4060 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.1852 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.56 | 127.16 | 145.07 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.6431 | 6.8344 | 6.9621 |
| PIT KS | 0.0414 | 0.0379 | 0.0458 |
| var / err^2 Spearman | 0.2501 | 0.2428 | 0.2300 |
| coverage of the 90% interval | 0.9027 | 0.9059 | 0.9132 |
| width of the 90% interval ($) | 643.12 | 781.37 | 915.51 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0011 | [-0.0329, 0.0396] | NOISE |
| h1 | 0.0335 | [-0.0037, 0.0724] | NOISE |
| h2 | 0.0109 | [-0.0391, 0.0469] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.185) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8101 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.6386 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.4974 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h0, h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5784 | 0.6092 | 0.6285 | 0.2255 |
| expected if the two signs were independent | 0.5147 | 0.4850 | 0.5175 | 0.1274 |

- P(up) unanimity (all three horizons call the same side): 0.2680
