# Evaluation report - test split - run `20260929T041215Z-8f35053-3298cb6a-n3_r1_p2`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.2905 | 0.4763 | 0.6274 |
| accuracy | 0.5199 | 0.5116 | 0.4987 |
| balanced accuracy | 0.5219 | 0.5118 | 0.5001 |
| precision (up) | 0.5427 | 0.5161 | 0.4944 |
| recall / sensitivity (up) | 0.3122 | 0.4880 | 0.6275 |
| specificity (down) | 0.7317 | 0.5356 | 0.3727 |
| F1 (up) | 0.3964 | 0.5016 | 0.5531 |
| MCC | 0.0483 | 0.0235 | 0.0002 |
| AUC | 0.5407 | 0.5270 | 0.4929 |
| Brier | 0.2506 | 0.2527 | 0.2637 |
| ECE (positive class) | 0.0346 | 0.0498 | 0.0749 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 769 / 648 / 1767 / 1694 | 1296 / 1215 / 1401 / 1360 | 1730 / 1769 / 1051 / 1027 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.2718 | 0.3655 | 0.2971 |
| Gaussian readout of the raw heads: MCC | 0.0133 | 0.0395 | 0.0650 |
| Gaussian readout of the raw heads: AUC | 0.5073 | 0.5237 | 0.5626 |
| Gaussian readout of the raw heads: Brier | 0.2525 | 0.2518 | 0.2499 |
| Gaussian readout of the raw heads: ECE | 0.0440 | 0.0251 | 0.0449 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.84 | 252.63 | 290.81 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 139.45 | 170.37 | 200.62 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0137 | -0.0195 | -0.0348 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0132 | -0.0197 | -0.0342 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0248 | 0.0231 | 0.0792 |
| corr, Spearman, raw heads | 0.0195 | 0.0481 | 0.1067 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -7.58 | -3.88 | -13.49 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.2351 | 0.3748 | 0.3003 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 106.29 | 131.12 | 149.72 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7468 | 6.9426 | 7.0602 |
| PIT KS | 0.1017 | 0.1180 | 0.0984 |
| var / err^2 Spearman | 0.2489 | 0.2304 | 0.2271 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0219 | [-0.0202, 0.0668] | NOISE |
| h1 | 0.0457 | [-0.0009, 0.0843] | NOISE |
| h2 | -0.0003 | [-0.0422, 0.0409] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4422 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.8364 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3368 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6611 | 0.5650 | 0.3610 | 0.1213 |
| expected if the two signs were independent | 0.6177 | 0.4987 | 0.4447 | 0.1365 |

- P(up) unanimity (all three horizons call the same side): 0.1783
