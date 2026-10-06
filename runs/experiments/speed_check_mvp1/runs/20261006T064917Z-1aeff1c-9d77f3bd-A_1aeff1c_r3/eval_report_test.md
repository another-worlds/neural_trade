# Evaluation report - test split - run `20261006T064917Z-1aeff1c-9d77f3bd-A_1aeff1c_r3`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.4468 | 0.5459 | 0.4859 |
| accuracy | 0.5096 | 0.4910 | 0.5087 |
| balanced accuracy | 0.5111 | 0.4896 | 0.5092 |
| precision (up) | 0.5264 | 0.5051 | 0.5284 |
| recall / sensitivity (up) | 0.4576 | 0.5358 | 0.4948 |
| specificity (down) | 0.5647 | 0.4434 | 0.5237 |
| F1 (up) | 0.4896 | 0.5200 | 0.5110 |
| MCC | 0.0224 | -0.0209 | 0.0185 |
| AUC | 0.5116 | 0.4930 | 0.5080 |
| Brier | 0.2505 | 0.2533 | 0.2513 |
| ECE (positive class) | 0.0186 | 0.0394 | 0.0214 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1257 / 1131 / 1467 / 1490 | 1571 / 1539 / 1226 / 1361 | 1508 / 1346 / 1480 / 1540 |
| Gaussian readout: calls up | 0.3860 | n/a (beta = 0: readout is the constant 0.5) | 0.3975 |
| Gaussian readout: MCC | 0.0444 | n/a (beta = 0: readout is the constant 0.5) | 0.0218 |
| Gaussian readout: AUC | 0.5095 | n/a (beta = 0: readout is the constant 0.5) | 0.4986 |
| Gaussian readout: Brier | 0.2506 | n/a (beta = 0: readout is the constant 0.5) | 0.2504 |
| Gaussian readout: ECE | 0.0171 | n/a (beta = 0: readout is the constant 0.5) | 0.0216 |
| Gaussian readout of the raw heads: calls up | 0.3860 | 0.3837 | 0.3975 |
| Gaussian readout of the raw heads: MCC | 0.0444 | 0.0144 | 0.0218 |
| Gaussian readout of the raw heads: AUC | 0.5095 | 0.4953 | 0.4986 |
| Gaussian readout of the raw heads: Brier | 0.2584 | 0.2668 | 0.2681 |
| Gaussian readout of the raw heads: ECE | 0.0373 | 0.0811 | 0.0834 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.49 | 236.10 | 269.44 |
| RMSE ($), raw heads | 202.11 | 256.27 | 289.63 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.90 | 175.66 | 200.16 |
| MAE ($), raw heads | 149.90 | 189.33 | 216.32 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0028 | n/a (beta = 0: served delta is 0) | -0.0025 |
| skill vs zero, raw heads | -0.0610 | -0.1782 | -0.1583 |
| EV, served | -0.0024 | n/a (beta = 0: served delta is 0) | -0.0018 |
| EV, raw heads | -0.0578 | -0.1678 | -0.1446 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0162 | -0.0096 | -0.0174 |
| corr, Spearman, raw heads | 0.0134 | -0.0155 | -0.0098 |
| mean predicted ($), served | -1.06 | 0.00 | -1.73 |
| mean predicted ($), raw heads | -6.64 | -16.74 | -21.88 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.3778 | 0.3843 | 0.3937 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.1595 | 0.0000 | 0.0791 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.60 | 127.09 | 145.30 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.6499 | 6.8464 | 6.9791 |
| PIT KS | 0.0309 | 0.0237 | 0.0320 |
| var / err^2 Spearman | 0.2564 | 0.2426 | 0.2228 |
| coverage of the 90% interval | 0.9034 | 0.9059 | 0.9122 |
| width of the 90% interval ($) | 646.04 | 781.37 | 914.71 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0017 | [-0.0417, 0.0376] | NOISE |
| h1 | 0.0176 | [-0.0253, 0.0604] | NOISE |
| h2 | -0.0009 | [-0.0420, 0.0410] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.160 / h1 0.000 / h2 0.079) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8394 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.7301 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5974 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5618 | 0.6032 | 0.6157 | 0.2159 |
| expected if the two signs were independent | 0.5150 | 0.4889 | 0.5056 | 0.1288 |

- P(up) unanimity (all three horizons call the same side): 0.2609
