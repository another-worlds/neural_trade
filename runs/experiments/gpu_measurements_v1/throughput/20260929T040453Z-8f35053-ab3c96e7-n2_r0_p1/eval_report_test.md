# Evaluation report - test split - run `20260929T040453Z-8f35053-ab3c96e7-n2_r0_p1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.2282 | 0.5649 | 0.6948 |
| accuracy | 0.5018 | 0.5008 | 0.5148 |
| balanced accuracy | 0.5045 | 0.5003 | 0.5170 |
| precision (up) | 0.5148 | 0.5040 | 0.5066 |
| recall / sensitivity (up) | 0.2326 | 0.5651 | 0.7120 |
| specificity (down) | 0.7764 | 0.4354 | 0.3220 |
| F1 (up) | 0.3205 | 0.5328 | 0.5920 |
| MCC | 0.0108 | 0.0005 | 0.0369 |
| AUC | 0.5065 | 0.5082 | 0.5200 |
| Brier | 0.2572 | 0.2578 | 0.2581 |
| ECE (positive class) | 0.0588 | 0.0577 | 0.0626 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 573 / 540 / 1875 / 1890 | 1501 / 1477 / 1139 / 1155 | 1963 / 1912 / 908 / 794 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.0822 | 0.0998 | 0.0906 |
| Gaussian readout of the raw heads: MCC | -0.0141 | 0.0380 | -0.0058 |
| Gaussian readout of the raw heads: AUC | 0.5164 | 0.5214 | 0.5315 |
| Gaussian readout of the raw heads: Brier | 0.2566 | 0.2531 | 0.2551 |
| Gaussian readout of the raw heads: ECE | 0.0789 | 0.0562 | 0.0752 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 214.15 | 252.44 | 291.64 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 141.37 | 170.75 | 201.99 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0262 | -0.0180 | -0.0407 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0076 | -0.0107 | -0.0178 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0208 | 0.0104 | 0.0209 |
| corr, Spearman, raw heads | 0.0279 | 0.0269 | 0.0479 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -31.21 | -25.27 | -48.42 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.0857 | 0.0920 | 0.0864 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 106.65 | 128.01 | 148.73 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7357 | 6.9046 | 7.0435 |
| PIT KS | 0.1075 | 0.0880 | 0.0897 |
| var / err^2 Spearman | 0.2368 | 0.2187 | 0.2350 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0095 | [-0.0483, 0.0455] | NOISE |
| h1 | 0.0349 | [0.0020, 0.0816] | WORKS |
| h2 | -0.0032 | [-0.0414, 0.0365] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4858 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.8217 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3827 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7442 | 0.4833 | 0.3080 | 0.1136 |
| expected if the two signs were independent | 0.7429 | 0.4591 | 0.3377 | 0.1239 |

- P(up) unanimity (all three horizons call the same side): 0.2106
