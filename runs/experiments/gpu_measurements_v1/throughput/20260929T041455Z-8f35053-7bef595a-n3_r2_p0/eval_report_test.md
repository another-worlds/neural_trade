# Evaluation report - test split - run `20260929T041455Z-8f35053-7bef595a-n3_r2_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.5146 | 0.1296 | 0.2146 |
| accuracy | 0.5135 | 0.5013 | 0.5022 |
| balanced accuracy | 0.5134 | 0.5041 | 0.4990 |
| precision (up) | 0.5179 | 0.5198 | 0.4921 |
| recall / sensitivity (up) | 0.5278 | 0.1337 | 0.2136 |
| specificity (down) | 0.4990 | 0.8746 | 0.7844 |
| F1 (up) | 0.5228 | 0.2126 | 0.2979 |
| MCC | 0.0268 | 0.0123 | -0.0024 |
| AUC | 0.5197 | 0.5054 | 0.4960 |
| Brier | 0.2538 | 0.2552 | 0.2550 |
| ECE (positive class) | 0.0543 | 0.0520 | 0.0549 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1300 / 1210 / 1205 / 1163 | 355 / 328 / 2288 / 2301 | 589 / 608 / 2212 / 2168 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.5904 | 0.0461 | 0.2731 |
| Gaussian readout of the raw heads: MCC | 0.0332 | 0.0119 | 0.0307 |
| Gaussian readout of the raw heads: AUC | 0.5146 | 0.5389 | 0.5238 |
| Gaussian readout of the raw heads: Brier | 0.2510 | 0.2529 | 0.2497 |
| Gaussian readout of the raw heads: ECE | 0.0247 | 0.0618 | 0.0011 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 211.49 | 251.57 | 285.50 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 138.68 | 171.26 | 195.49 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0008 | -0.0109 | 0.0027 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0004 | 0.0056 | 0.0025 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0569 | 0.0752 | 0.0502 |
| corr, Spearman, raw heads | 0.0305 | 0.0730 | 0.0534 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 2.81 | -36.01 | -3.68 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.6151 | 0.0439 | 0.2695 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 106.18 | 130.32 | 149.11 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7280 | 6.9281 | 7.0526 |
| PIT KS | 0.1054 | 0.1130 | 0.0919 |
| var / err^2 Spearman | 0.2771 | 0.2643 | 0.2554 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0171 | [-0.0237, 0.0543] | NOISE |
| h1 | -0.0136 | [-0.0632, 0.0432] | NOISE |
| h2 | -0.0005 | [-0.0429, 0.0500] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8673 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.0831 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.0105 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.3599 | 0.8686 | 0.6486 | 0.2066 |
| expected if the two signs were independent | 0.5004 | 0.8526 | 0.6383 | 0.2867 |

- P(up) unanimity (all three horizons call the same side): 0.3871
