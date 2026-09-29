# Evaluation report - test split - run `20260929T040349Z-8f35053-48cbf95d-n1_r2_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.4315 | 0.3926 | 0.4380 |
| accuracy | 0.4863 | 0.4951 | 0.4997 |
| balanced accuracy | 0.4869 | 0.4959 | 0.4990 |
| precision (up) | 0.4898 | 0.4986 | 0.4932 |
| recall / sensitivity (up) | 0.4186 | 0.3886 | 0.4371 |
| specificity (down) | 0.5553 | 0.6032 | 0.5610 |
| F1 (up) | 0.4514 | 0.4367 | 0.4635 |
| MCC | -0.0264 | -0.0084 | -0.0020 |
| AUC | 0.4750 | 0.4992 | 0.4911 |
| Brier | 0.2558 | 0.2528 | 0.2598 |
| ECE (positive class) | 0.0585 | 0.0462 | 0.0505 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1031 / 1074 / 1341 / 1432 | 1032 / 1038 / 1578 / 1624 | 1205 / 1238 / 1582 / 1552 |
| Gaussian readout: calls up | 0.5361 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0055 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.4898 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2504 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0057 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.5361 | 0.1481 | 0.2399 |
| Gaussian readout of the raw heads: MCC | 0.0055 | -0.0165 | 0.0105 |
| Gaussian readout of the raw heads: AUC | 0.4898 | 0.4643 | 0.5215 |
| Gaussian readout of the raw heads: Brier | 0.2517 | 0.2522 | 0.2500 |
| Gaussian readout of the raw heads: ECE | 0.0213 | 0.0319 | 0.0197 |

beta = 0 for h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.58 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.21 | 250.64 | 285.77 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.66 | 168.19 | 195.53 |
| MAE ($), raw heads | 138.28 | 169.44 | 195.86 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0017 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0077 | -0.0035 | 0.0008 |
| EV, served | -0.0017 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0075 | -0.0025 | 0.0020 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0236 | -0.0019 | 0.0511 |
| corr, Spearman, raw heads | -0.0143 | -0.0351 | 0.0375 |
| mean predicted ($), served | 0.51 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 1.42 | -12.19 | -15.89 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.5449 | 0.1502 | 0.2483 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.3580 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.39 | 128.44 | 149.58 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7250 | 6.9118 | 7.0524 |
| PIT KS | 0.0940 | 0.0934 | 0.0967 |
| var / err^2 Spearman | 0.2305 | 0.2184 | 0.1914 |
| coverage of the 90% interval | 0.8766 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 580.20 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0374 | [-0.0714, 0.0092] | NOISE |
| h1 | 0.0146 | [-0.0159, 0.0472] | NOISE |
| h2 | -0.0123 | [-0.0504, 0.0243] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.358 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5567 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.5902 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3259 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.3777 | 0.5847 | 0.5540 | 0.1125 |
| expected if the two signs were independent | 0.4913 | 0.5851 | 0.5257 | 0.1506 |

- P(up) unanimity (all three horizons call the same side): 0.2587
