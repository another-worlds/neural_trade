# Evaluation report - test split - run `20260929T040453Z-8f35053-3f0ccc90-n2_r0_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.4420 | 0.4539 | 0.1278 |
| accuracy | 0.5205 | 0.5154 | 0.4915 |
| balanced accuracy | 0.5211 | 0.5157 | 0.4873 |
| precision (up) | 0.5288 | 0.5211 | 0.4446 |
| recall / sensitivity (up) | 0.4629 | 0.4695 | 0.1150 |
| specificity (down) | 0.5793 | 0.5619 | 0.8596 |
| F1 (up) | 0.4936 | 0.4940 | 0.1827 |
| MCC | 0.0424 | 0.0316 | -0.0381 |
| AUC | 0.5249 | 0.5168 | 0.4605 |
| Brier | 0.2527 | 0.2560 | 0.2587 |
| ECE (positive class) | 0.0341 | 0.0427 | 0.0706 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1140 / 1016 / 1399 / 1323 | 1247 / 1146 / 1470 / 1409 | 317 / 396 / 2424 / 2440 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0070 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0332 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.5192 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.2499 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0029 |
| Gaussian readout of the raw heads: calls up | 0.1611 | 0.1480 | 0.0070 |
| Gaussian readout of the raw heads: MCC | 0.0269 | 0.0107 | 0.0332 |
| Gaussian readout of the raw heads: AUC | 0.5016 | 0.4901 | 0.5192 |
| Gaussian readout of the raw heads: Brier | 0.2518 | 0.2530 | 0.2537 |
| Gaussian readout of the raw heads: ECE | 0.0312 | 0.0441 | 0.0618 |

beta = 0 for h0, h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.63 |
| RMSE ($), raw heads | 211.51 | 251.35 | 288.27 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.43 |
| MAE ($), raw heads | 138.73 | 170.49 | 199.11 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | 0.0018 |
| skill vs zero, raw heads | -0.0011 | -0.0092 | -0.0168 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | 0.0015 |
| EV, raw heads | 0.0003 | -0.0059 | 0.0051 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0408 | 0.0139 | 0.0733 |
| corr, Spearman, raw heads | 0.0226 | 0.0063 | 0.0695 |
| mean predicted ($), served | 0.00 | 0.00 | -5.95 |
| mean predicted ($), raw heads | -10.52 | -18.44 | -47.49 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.1410 | 0.1375 | 0.0079 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.1252 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.97 | 128.26 | 149.20 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7227 | 6.8998 | 7.0478 |
| PIT KS | 0.1031 | 0.0965 | 0.1039 |
| var / err^2 Spearman | 0.2616 | 0.2651 | 0.2353 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8542 |
| width of the 90% interval ($) | 584.96 | 687.16 | 771.64 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0012 | [-0.0381, 0.0432] | NOISE |
| h1 | 0.0087 | [-0.0270, 0.0388] | NOISE |
| h2 | -0.0222 | [-0.0621, 0.0287] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.125) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7823 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.8618 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.6624 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5321 | 0.4880 | 0.8836 | 0.1857 |
| expected if the two signs were independent | 0.5431 | 0.5186 | 0.8720 | 0.2123 |

- P(up) unanimity (all three horizons call the same side): 0.2186
