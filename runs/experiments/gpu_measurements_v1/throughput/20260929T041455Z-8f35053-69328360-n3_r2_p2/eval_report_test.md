# Evaluation report - test split - run `20260929T041455Z-8f35053-69328360-n3_r2_p2`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.4467 | 0.3272 | 0.5334 |
| accuracy | 0.4994 | 0.5028 | 0.4929 |
| balanced accuracy | 0.4999 | 0.5042 | 0.4933 |
| precision (up) | 0.5048 | 0.5101 | 0.4881 |
| recall / sensitivity (up) | 0.4466 | 0.3313 | 0.5267 |
| specificity (down) | 0.5532 | 0.6770 | 0.4599 |
| F1 (up) | 0.4739 | 0.4017 | 0.5066 |
| MCC | -0.0002 | 0.0089 | -0.0134 |
| AUC | 0.5020 | 0.5053 | 0.4957 |
| Brier | 0.2652 | 0.2522 | 0.2628 |
| ECE (positive class) | 0.0917 | 0.0462 | 0.0942 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1100 / 1079 / 1336 / 1363 | 880 / 845 / 1771 / 1776 | 1452 / 1523 / 1297 / 1305 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.1445 | 0.0842 | 0.0502 |
| Gaussian readout of the raw heads: MCC | -0.0419 | -0.0378 | 0.0092 |
| Gaussian readout of the raw heads: AUC | 0.4688 | 0.4682 | 0.5475 |
| Gaussian readout of the raw heads: Brier | 0.2628 | 0.2570 | 0.2537 |
| Gaussian readout of the raw heads: ECE | 0.0973 | 0.0728 | 0.0691 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 216.00 | 253.58 | 288.86 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 144.33 | 173.12 | 199.81 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0440 | -0.0272 | -0.0209 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0266 | -0.0138 | 0.0019 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0480 | -0.0422 | 0.0822 |
| corr, Spearman, raw heads | -0.0418 | -0.0360 | 0.0830 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -30.25 | -32.80 | -48.42 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.1258 | 0.0752 | 0.0478 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 104.86 | 128.65 | 147.92 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7081 | 6.9181 | 7.0385 |
| PIT KS | 0.0887 | 0.0925 | 0.0831 |
| var / err^2 Spearman | 0.2608 | 0.2415 | 0.2429 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0084 | [-0.0552, 0.0356] | NOISE |
| h1 | -0.0025 | [-0.0459, 0.0395] | NOISE |
| h2 | 0.0116 | [-0.0401, 0.0548] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5054 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.5209 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.2507 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5911 | 0.6444 | 0.4341 | 0.1985 |
| expected if the two signs were independent | 0.5475 | 0.6546 | 0.4596 | 0.1977 |

- P(up) unanimity (all three horizons call the same side): 0.3290
