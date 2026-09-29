# Evaluation report - test split - run `20260929T040635Z-8f35053-b5f7b84d-n2_r1_p1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.3071 | 0.1963 | 0.5272 |
| accuracy | 0.4955 | 0.5104 | 0.4956 |
| balanced accuracy | 0.4974 | 0.5127 | 0.4959 |
| precision (up) | 0.5007 | 0.5362 | 0.4905 |
| recall / sensitivity (up) | 0.3045 | 0.2090 | 0.5230 |
| specificity (down) | 0.6903 | 0.8165 | 0.4688 |
| F1 (up) | 0.3787 | 0.3007 | 0.5062 |
| MCC | -0.0057 | 0.0321 | -0.0082 |
| AUC | 0.4949 | 0.5102 | 0.5075 |
| Brier | 0.2591 | 0.2550 | 0.2521 |
| ECE (positive class) | 0.0668 | 0.0553 | 0.0689 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 750 / 748 / 1667 / 1713 | 555 / 480 / 2136 / 2101 | 1442 / 1498 / 1322 / 1315 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.0724 | 0.2428 | 0.2718 |
| Gaussian readout of the raw heads: MCC | -0.0099 | 0.0178 | 0.0488 |
| Gaussian readout of the raw heads: AUC | 0.4634 | 0.4886 | 0.5301 |
| Gaussian readout of the raw heads: Brier | 0.2545 | 0.2543 | 0.2494 |
| Gaussian readout of the raw heads: ECE | 0.0433 | 0.0388 | 0.0142 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.78 | 251.87 | 284.49 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 139.56 | 171.41 | 195.43 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0131 | -0.0134 | 0.0098 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0083 | -0.0078 | 0.0106 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0359 | 0.0231 | 0.1099 |
| corr, Spearman, raw heads | -0.0479 | 0.0029 | 0.0669 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -17.13 | -22.56 | -14.51 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.0636 | 0.2334 | 0.2805 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 107.40 | 131.04 | 151.60 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7413 | 6.9319 | 7.0819 |
| PIT KS | 0.1140 | 0.1190 | 0.1126 |
| var / err^2 Spearman | 0.2500 | 0.2562 | 0.2301 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0095 | [-0.0302, 0.0605] | NOISE |
| h1 | -0.0003 | [-0.0468, 0.0501] | NOISE |
| h2 | 0.0135 | [-0.0328, 0.0599] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7450 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.2810 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.1625 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6909 | 0.7289 | 0.6636 | 0.3367 |
| expected if the two signs were independent | 0.6866 | 0.6710 | 0.4807 | 0.2197 |

- P(up) unanimity (all three horizons call the same side): 0.2908
