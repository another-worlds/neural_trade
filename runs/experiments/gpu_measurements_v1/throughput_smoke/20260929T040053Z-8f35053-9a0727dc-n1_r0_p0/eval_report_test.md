# Evaluation report - test split - run `20260929T040053Z-8f35053-9a0727dc-n1_r0_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.6214 | 0.7701 | 0.5024 |
| accuracy | 0.5215 | 0.4962 | 0.5225 |
| balanced accuracy | 0.5203 | 0.4942 | 0.5225 |
| precision (up) | 0.5213 | 0.5000 | 0.5168 |
| recall / sensitivity (up) | 0.6415 | 0.7643 | 0.5252 |
| specificity (down) | 0.3992 | 0.2240 | 0.5199 |
| F1 (up) | 0.5752 | 0.6045 | 0.5210 |
| MCC | 0.0419 | -0.0139 | 0.0451 |
| AUC | 0.5189 | 0.4829 | 0.5382 |
| Brier | 0.2625 | 0.2640 | 0.2506 |
| ECE (positive class) | 0.0588 | 0.0905 | 0.0329 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1580 / 1451 / 964 / 883 | 2030 / 2030 / 586 / 626 | 1448 / 1354 / 1466 / 1309 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.0699 | 0.2160 | 0.1931 |
| Gaussian readout of the raw heads: MCC | -0.0099 | 0.0573 | 0.0332 |
| Gaussian readout of the raw heads: AUC | 0.4520 | 0.5202 | 0.4869 |
| Gaussian readout of the raw heads: Brier | 0.2537 | 0.2512 | 0.2567 |
| Gaussian readout of the raw heads: ECE | 0.0389 | 0.0331 | 0.0517 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.96 | 250.54 | 290.60 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 139.17 | 169.43 | 201.00 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0149 | -0.0027 | -0.0333 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0119 | -0.0002 | -0.0276 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0762 | 0.0582 | -0.0193 |
| corr, Spearman, raw heads | -0.0600 | 0.0457 | -0.0099 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -13.97 | -16.71 | -26.94 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.0625 | 0.2185 | 0.1837 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.21 | 129.30 | 148.54 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7058 | 6.9039 | 7.0415 |
| PIT KS | 0.0947 | 0.1051 | 0.0883 |
| var / err^2 Spearman | 0.2679 | 0.2545 | 0.2403 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0229 | [-0.0234, 0.0611] | NOISE |
| h1 | -0.0188 | [-0.0692, 0.0296] | NOISE |
| h2 | 0.0383 | [-0.0069, 0.0863] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6580 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.7211 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.4106 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.3841 | 0.2662 | 0.6112 | 0.0558 |
| expected if the two signs were independent | 0.3966 | 0.3431 | 0.4916 | 0.0654 |

- P(up) unanimity (all three horizons call the same side): 0.2772
