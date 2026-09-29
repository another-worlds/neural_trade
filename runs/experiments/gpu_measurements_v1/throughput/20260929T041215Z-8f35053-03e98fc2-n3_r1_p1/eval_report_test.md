# Evaluation report - test split - run `20260929T041215Z-8f35053-03e98fc2-n3_r1_p1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.2632 | 0.3060 | 0.3466 |
| accuracy | 0.5262 | 0.5055 | 0.4972 |
| balanced accuracy | 0.5286 | 0.5070 | 0.4955 |
| precision (up) | 0.5592 | 0.5152 | 0.4878 |
| recall / sensitivity (up) | 0.2915 | 0.3129 | 0.3420 |
| specificity (down) | 0.7656 | 0.7011 | 0.6489 |
| F1 (up) | 0.3832 | 0.3893 | 0.4021 |
| MCC | 0.0649 | 0.0151 | -0.0095 |
| AUC | 0.5348 | 0.5061 | 0.4886 |
| Brier | 0.2507 | 0.2559 | 0.2538 |
| ECE (positive class) | 0.0385 | 0.0506 | 0.0493 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 718 / 566 / 1849 / 1745 | 831 / 782 / 1834 / 1825 | 943 / 990 / 1830 / 1814 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6125 | 0.1070 | 0.6392 |
| Gaussian readout of the raw heads: MCC | 0.0878 | 0.0637 | 0.0617 |
| Gaussian readout of the raw heads: AUC | 0.5445 | 0.5083 | 0.5300 |
| Gaussian readout of the raw heads: Brier | 0.2494 | 0.2556 | 0.2504 |
| Gaussian readout of the raw heads: ECE | 0.0146 | 0.0684 | 0.0183 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 211.64 | 253.96 | 287.52 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 138.57 | 173.33 | 196.36 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0023 | -0.0303 | -0.0114 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | 0.0005 | -0.0209 | -0.0099 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0576 | 0.0129 | 0.0071 |
| corr, Spearman, raw heads | 0.0805 | 0.0410 | 0.0467 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 9.10 | -28.10 | 7.46 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.6227 | 0.0998 | 0.6499 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 104.85 | 129.11 | 150.16 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7214 | 6.9182 | 7.0685 |
| PIT KS | 0.0863 | 0.1015 | 0.1008 |
| var / err^2 Spearman | 0.2391 | 0.2433 | 0.2118 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0044 | [-0.0366, 0.0585] | NOISE |
| h1 | -0.0055 | [-0.0341, 0.0289] | NOISE |
| h2 | -0.0118 | [-0.0574, 0.0342] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8324 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.2424 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.1316 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4421 | 0.6642 | 0.4320 | 0.1564 |
| expected if the two signs were independent | 0.4364 | 0.6633 | 0.4471 | 0.1585 |

- P(up) unanimity (all three horizons call the same side): 0.4588
