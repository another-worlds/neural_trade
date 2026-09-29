# Evaluation report - test split - run `20260929T040953Z-8f35053-1ce9c801-n3_r0_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.4512 | 0.4080 | 0.4538 |
| accuracy | 0.5170 | 0.5332 | 0.5252 |
| balanced accuracy | 0.5175 | 0.5339 | 0.5247 |
| precision (up) | 0.5243 | 0.5453 | 0.5215 |
| recall / sensitivity (up) | 0.4685 | 0.4416 | 0.4788 |
| specificity (down) | 0.5665 | 0.6261 | 0.5706 |
| F1 (up) | 0.4949 | 0.4880 | 0.4992 |
| MCC | 0.0352 | 0.0690 | 0.0496 |
| AUC | 0.5252 | 0.5488 | 0.5310 |
| Brier | 0.2520 | 0.2494 | 0.2518 |
| ECE (positive class) | 0.0332 | 0.0224 | 0.0386 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1154 / 1047 / 1368 / 1309 | 1173 / 978 / 1638 / 1483 | 1320 / 1211 / 1609 / 1437 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.7197 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | 0.0917 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.5313 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2500 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0428 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.3475 | 0.7197 | 0.6148 |
| Gaussian readout of the raw heads: MCC | 0.0277 | 0.0917 | 0.0544 |
| Gaussian readout of the raw heads: AUC | 0.5234 | 0.5313 | 0.5369 |
| Gaussian readout of the raw heads: Brier | 0.2503 | 0.2527 | 0.2506 |
| Gaussian readout of the raw heads: ECE | 0.0180 | 0.0482 | 0.0247 |

beta = 0 for h0, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.30 | 253.54 | 286.49 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 137.90 | 171.95 | 196.03 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | 0.0000 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0086 | -0.0268 | -0.0042 |
| EV, served | n/a (beta = 0: served delta is 0) | 0.0000 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0086 | -0.0251 | -0.0045 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0079 | 0.0612 | 0.0525 |
| corr, Spearman, raw heads | 0.0439 | 0.0627 | 0.0648 |
| mean predicted ($), served | 0.00 | 0.01 | 0.00 |
| mean predicted ($), raw heads | -4.86 | 7.35 | -2.32 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.3219 | 0.7464 | 0.6144 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0013 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 107.46 | 131.23 | 148.97 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7491 | 6.9360 | 7.0535 |
| PIT KS | 0.1157 | 0.1195 | 0.0897 |
| var / err^2 Spearman | 0.2597 | 0.2591 | 0.2302 |
| coverage of the 90% interval | 0.8802 | 0.8701 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.01 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0380 | [0.0018, 0.0824] | WORKS |
| h1 | 0.0273 | [-0.0085, 0.0746] | NOISE |
| h2 | 0.0040 | [-0.0484, 0.0471] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.001 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8531 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.1809 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.0925 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6031 | 0.5831 | 0.5046 | 0.1570 |
| expected if the two signs were independent | 0.5259 | 0.4514 | 0.4885 | 0.1112 |

- P(up) unanimity (all three horizons call the same side): 0.4165
