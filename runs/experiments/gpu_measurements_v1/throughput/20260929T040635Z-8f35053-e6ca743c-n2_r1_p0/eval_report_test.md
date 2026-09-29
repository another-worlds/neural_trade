# Evaluation report - test split - run `20260929T040635Z-8f35053-e6ca743c-n2_r1_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.4772 | 0.3638 | 0.5117 |
| accuracy | 0.5205 | 0.5190 | 0.5189 |
| balanced accuracy | 0.5207 | 0.5200 | 0.5191 |
| precision (up) | 0.5266 | 0.5313 | 0.5130 |
| recall / sensitivity (up) | 0.4978 | 0.3837 | 0.5310 |
| specificity (down) | 0.5437 | 0.6563 | 0.5071 |
| F1 (up) | 0.5118 | 0.4456 | 0.5218 |
| MCC | 0.0415 | 0.0416 | 0.0381 |
| AUC | 0.5316 | 0.5171 | 0.5337 |
| Brier | 0.2550 | 0.2585 | 0.2545 |
| ECE (positive class) | 0.0571 | 0.0556 | 0.0442 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1226 / 1102 / 1313 / 1237 | 1019 / 899 / 1717 / 1637 | 1464 / 1390 / 1430 / 1293 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.2786 | 0.3517 | 0.1083 |
| Gaussian readout of the raw heads: MCC | 0.0245 | 0.0675 | 0.0570 |
| Gaussian readout of the raw heads: AUC | 0.4933 | 0.5566 | 0.5254 |
| Gaussian readout of the raw heads: Brier | 0.2524 | 0.2485 | 0.2512 |
| Gaussian readout of the raw heads: ECE | 0.0288 | 0.0269 | 0.0419 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 211.94 | 249.95 | 286.55 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 138.69 | 169.09 | 197.33 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0052 | 0.0020 | -0.0046 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0035 | 0.0043 | 0.0046 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0108 | 0.0859 | 0.0694 |
| corr, Spearman, raw heads | 0.0015 | 0.0861 | 0.0508 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -11.18 | -16.20 | -32.80 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.2593 | 0.3520 | 0.1028 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 106.32 | 131.22 | 150.08 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7420 | 6.9476 | 7.0708 |
| PIT KS | 0.1017 | 0.1159 | 0.0949 |
| var / err^2 Spearman | 0.2296 | 0.2059 | 0.2005 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0179 | [-0.0292, 0.0564] | NOISE |
| h1 | -0.0111 | [-0.0634, 0.0362] | NOISE |
| h2 | 0.0353 | [-0.0088, 0.0864] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8118 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.5699 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3881 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4031 | 0.4574 | 0.5574 | 0.1450 |
| expected if the two signs were independent | 0.4997 | 0.5345 | 0.4959 | 0.1573 |

- P(up) unanimity (all three horizons call the same side): 0.3167
