# Evaluation report - test split - run `20260929T041455Z-8f35053-011873c7-n3_r2_p1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.1991 | 0.4099 | 0.4268 |
| accuracy | 0.5092 | 0.4903 | 0.4996 |
| balanced accuracy | 0.5122 | 0.4910 | 0.4987 |
| precision (up) | 0.5355 | 0.4928 | 0.4929 |
| recall / sensitivity (up) | 0.2111 | 0.4010 | 0.4255 |
| specificity (down) | 0.8133 | 0.5810 | 0.5720 |
| F1 (up) | 0.3029 | 0.4422 | 0.4567 |
| MCC | 0.0305 | -0.0183 | -0.0026 |
| AUC | 0.5117 | 0.4854 | 0.4949 |
| Brier | 0.2556 | 0.2609 | 0.2589 |
| ECE (positive class) | 0.0579 | 0.0704 | 0.0683 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 520 / 451 / 1964 / 1943 | 1065 / 1096 / 1520 / 1591 | 1173 / 1207 / 1613 / 1584 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.0742 | 0.1567 | 0.1013 |
| Gaussian readout of the raw heads: MCC | -0.0137 | 0.0207 | -0.0158 |
| Gaussian readout of the raw heads: AUC | 0.4895 | 0.5156 | 0.5184 |
| Gaussian readout of the raw heads: Brier | 0.2534 | 0.2520 | 0.2520 |
| Gaussian readout of the raw heads: ECE | 0.0512 | 0.0402 | 0.0457 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 212.41 | 251.80 | 287.64 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 139.48 | 170.42 | 198.61 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0096 | -0.0128 | -0.0123 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0023 | -0.0085 | -0.0036 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0287 | 0.0121 | 0.0443 |
| corr, Spearman, raw heads | -0.0116 | 0.0211 | 0.0368 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -20.46 | -20.47 | -32.14 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.0626 | 0.1553 | 0.0981 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 106.23 | 129.14 | 151.61 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7233 | 6.9122 | 7.0678 |
| PIT KS | 0.1012 | 0.1047 | 0.1145 |
| var / err^2 Spearman | 0.2272 | 0.2422 | 0.2329 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0097 | [-0.0532, 0.0385] | NOISE |
| h1 | -0.0204 | [-0.0603, 0.0200] | NOISE |
| h2 | 0.0020 | [-0.0436, 0.0431] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6382 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.6668 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3922 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7823 | 0.5872 | 0.5337 | 0.3018 |
| expected if the two signs were independent | 0.7771 | 0.5693 | 0.5478 | 0.2944 |

- P(up) unanimity (all three horizons call the same side): 0.4169
