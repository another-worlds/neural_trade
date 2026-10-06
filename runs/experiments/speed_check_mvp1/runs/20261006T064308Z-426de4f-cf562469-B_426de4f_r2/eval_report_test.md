# Evaluation report - test split - run `20261006T064308Z-426de4f-cf562469-B_426de4f_r2`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.5136 | 0.5124 | 0.5907 |
| accuracy | 0.5020 | 0.4996 | 0.5107 |
| balanced accuracy | 0.5016 | 0.4992 | 0.5073 |
| precision (up) | 0.5155 | 0.5139 | 0.5251 |
| recall / sensitivity (up) | 0.5151 | 0.5116 | 0.5978 |
| specificity (down) | 0.4881 | 0.4868 | 0.4168 |
| F1 (up) | 0.5153 | 0.5127 | 0.5591 |
| MCC | 0.0032 | -0.0016 | 0.0148 |
| AUC | 0.5014 | 0.5021 | 0.5099 |
| Brier | 0.2508 | 0.2498 | 0.2507 |
| ECE (positive class) | 0.0249 | 0.0264 | 0.0192 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1415 / 1330 / 1268 / 1332 | 1500 / 1419 / 1346 / 1432 | 1822 / 1648 / 1178 / 1226 |
| Gaussian readout: calls up | 0.4829 | n/a (beta = 0: readout is the constant 0.5) | 0.5390 |
| Gaussian readout: MCC | -0.0288 | n/a (beta = 0: readout is the constant 0.5) | -0.0081 |
| Gaussian readout: AUC | 0.4881 | n/a (beta = 0: readout is the constant 0.5) | 0.4931 |
| Gaussian readout: Brier | 0.2500 | n/a (beta = 0: readout is the constant 0.5) | 0.2500 |
| Gaussian readout: ECE | 0.0185 | n/a (beta = 0: readout is the constant 0.5) | 0.0187 |
| Gaussian readout of the raw heads: calls up | 0.4829 | 0.5273 | 0.5390 |
| Gaussian readout of the raw heads: MCC | -0.0288 | -0.0148 | -0.0081 |
| Gaussian readout of the raw heads: AUC | 0.4881 | 0.4838 | 0.4931 |
| Gaussian readout of the raw heads: Brier | 0.2524 | 0.2585 | 0.2584 |
| Gaussian readout of the raw heads: ECE | 0.0498 | 0.0692 | 0.0732 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.21 | 236.10 | 269.12 |
| RMSE ($), raw heads | 197.20 | 241.91 | 275.81 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.64 | 175.66 | 199.91 |
| MAE ($), raw heads | 146.28 | 180.09 | 205.37 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0001 | n/a (beta = 0: served delta is 0) | -0.0001 |
| skill vs zero, raw heads | -0.0100 | -0.0498 | -0.0505 |
| EV, served | 0.0002 | n/a (beta = 0: served delta is 0) | -0.0001 |
| EV, raw heads | -0.0094 | -0.0506 | -0.0510 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0125 | 0.0025 | -0.0001 |
| corr, Spearman, raw heads | -0.0239 | -0.0286 | -0.0133 |
| mean predicted ($), served | -0.20 | 0.00 | 0.05 |
| mean predicted ($), raw heads | -1.85 | 2.42 | 1.20 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.4974 | 0.5276 | 0.5485 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.1092 | 0.0000 | 0.0456 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.57 | 126.97 | 144.93 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.6633 | 6.8589 | 6.9803 |
| PIT KS | 0.0210 | 0.0182 | 0.0260 |
| var / err^2 Spearman | 0.2438 | 0.2348 | 0.2274 |
| coverage of the 90% interval | 0.9030 | 0.9059 | 0.9120 |
| width of the 90% interval ($) | 644.16 | 781.37 | 910.87 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0002 | [-0.0378, 0.0358] | NOISE |
| h1 | 0.0209 | [-0.0135, 0.0541] | NOISE |
| h2 | 0.0105 | [-0.0374, 0.0544] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.109 / h1 0.000 / h2 0.046) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8397 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.7229 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5896 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5767 | 0.6169 | 0.6461 | 0.2295 |
| expected if the two signs were independent | 0.4999 | 0.5010 | 0.5088 | 0.1421 |

- P(up) unanimity (all three horizons call the same side): 0.3050
