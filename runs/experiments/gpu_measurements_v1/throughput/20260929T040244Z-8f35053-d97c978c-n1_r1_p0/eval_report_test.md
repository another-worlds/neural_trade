# Evaluation report - test split - run `20260929T040244Z-8f35053-d97c978c-n1_r1_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.3110 | 0.4977 | 0.5806 |
| accuracy | 0.4879 | 0.4905 | 0.4823 |
| balanced accuracy | 0.4898 | 0.4905 | 0.4832 |
| precision (up) | 0.4885 | 0.4943 | 0.4799 |
| recall / sensitivity (up) | 0.3009 | 0.4883 | 0.5637 |
| specificity (down) | 0.6787 | 0.4927 | 0.4028 |
| F1 (up) | 0.3724 | 0.4913 | 0.5184 |
| MCC | -0.0221 | -0.0189 | -0.0339 |
| AUC | 0.4929 | 0.4909 | 0.4766 |
| Brier | 0.2558 | 0.2620 | 0.2601 |
| ECE (positive class) | 0.0752 | 0.0828 | 0.0750 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 741 / 776 / 1639 / 1722 | 1297 / 1327 / 1289 / 1359 | 1554 / 1684 / 1136 / 1203 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.2681 | 0.3234 | 0.2485 |
| Gaussian readout of the raw heads: MCC | 0.0440 | 0.0438 | 0.0347 |
| Gaussian readout of the raw heads: AUC | 0.5250 | 0.5146 | 0.5306 |
| Gaussian readout of the raw heads: Brier | 0.2499 | 0.2510 | 0.2538 |
| Gaussian readout of the raw heads: ECE | 0.0238 | 0.0155 | 0.0218 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.81 | 253.77 | 292.27 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 139.03 | 171.02 | 200.79 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0134 | -0.0287 | -0.0452 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0135 | -0.0285 | -0.0446 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0328 | 0.0000 | 0.0190 |
| corr, Spearman, raw heads | 0.0439 | 0.0224 | 0.0507 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -1.78 | 1.35 | 3.50 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.2427 | 0.3349 | 0.2436 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 107.72 | 130.53 | 150.54 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7457 | 6.9355 | 7.0702 |
| PIT KS | 0.1119 | 0.1061 | 0.1017 |
| var / err^2 Spearman | 0.2689 | 0.2607 | 0.2419 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0181 | [-0.0232, 0.0694] | NOISE |
| h1 | 0.0077 | [-0.0333, 0.0456] | NOISE |
| h2 | -0.0121 | [-0.0521, 0.0221] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.1763 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.7953 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.0761 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6956 | 0.5444 | 0.3744 | 0.1417 |
| expected if the two signs were independent | 0.6161 | 0.5022 | 0.4555 | 0.1277 |

- P(up) unanimity (all three horizons call the same side): 0.2062
