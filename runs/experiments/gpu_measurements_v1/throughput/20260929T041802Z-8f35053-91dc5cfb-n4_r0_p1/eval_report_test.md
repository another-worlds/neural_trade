# Evaluation report - test split - run `20260929T041802Z-8f35053-91dc5cfb-n4_r0_p1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.5670 | 0.1493 | 0.4068 |
| accuracy | 0.5176 | 0.5089 | 0.5076 |
| balanced accuracy | 0.5170 | 0.5116 | 0.5066 |
| precision (up) | 0.5199 | 0.5426 | 0.5024 |
| recall / sensitivity (up) | 0.5838 | 0.1608 | 0.4135 |
| specificity (down) | 0.4501 | 0.8624 | 0.5996 |
| F1 (up) | 0.5500 | 0.2480 | 0.4536 |
| MCC | 0.0343 | 0.0325 | 0.0134 |
| AUC | 0.5118 | 0.5079 | 0.4997 |
| Brier | 0.2519 | 0.2540 | 0.2641 |
| ECE (positive class) | 0.0259 | 0.0534 | 0.0774 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1438 / 1328 / 1087 / 1025 | 427 / 360 / 2256 / 2229 | 1140 / 1129 / 1691 / 1617 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.0998 | 0.4947 | 0.5519 |
| Gaussian readout of the raw heads: MCC | -0.0190 | 0.0532 | 0.0926 |
| Gaussian readout of the raw heads: AUC | 0.4812 | 0.5249 | 0.5512 |
| Gaussian readout of the raw heads: Brier | 0.2527 | 0.2503 | 0.2483 |
| Gaussian readout of the raw heads: ECE | 0.0446 | 0.0070 | 0.0208 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 213.12 | 250.56 | 286.03 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 139.23 | 168.61 | 195.76 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0163 | -0.0028 | -0.0010 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0115 | -0.0030 | -0.0013 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0540 | 0.0347 | 0.0850 |
| corr, Spearman, raw heads | -0.0356 | 0.0355 | 0.0876 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -17.18 | -2.82 | -4.98 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.0930 | 0.5285 | 0.5757 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 107.34 | 130.52 | 150.40 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7435 | 6.9248 | 7.0600 |
| PIT KS | 0.1172 | 0.1153 | 0.1058 |
| var / err^2 Spearman | 0.2650 | 0.2381 | 0.2515 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0230 | [-0.0707, 0.0143] | NOISE |
| h1 | -0.0078 | [-0.0580, 0.0516] | NOISE |
| h2 | -0.0041 | [-0.0478, 0.0327] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4156 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.6909 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3470 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4084 | 0.4743 | 0.5011 | 0.1108 |
| expected if the two signs were independent | 0.4235 | 0.4796 | 0.4871 | 0.1096 |

- P(up) unanimity (all three horizons call the same side): 0.2307
