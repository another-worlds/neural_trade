# Evaluation report - test split - run `20260929T040953Z-8f35053-8557bd4d-n3_r0_p1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.5554 | 0.3818 | 0.5331 |
| accuracy | 0.5137 | 0.4991 | 0.5413 |
| balanced accuracy | 0.5132 | 0.4999 | 0.5417 |
| precision (up) | 0.5168 | 0.5037 | 0.5335 |
| recall / sensitivity (up) | 0.5684 | 0.3818 | 0.5753 |
| specificity (down) | 0.4580 | 0.6181 | 0.5082 |
| F1 (up) | 0.5414 | 0.4344 | 0.5536 |
| MCC | 0.0265 | -0.0001 | 0.0836 |
| AUC | 0.5081 | 0.4998 | 0.5433 |
| Brier | 0.2565 | 0.2577 | 0.2531 |
| ECE (positive class) | 0.0482 | 0.0589 | 0.0319 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 1400 / 1309 / 1106 / 1063 | 1014 / 999 / 1617 / 1642 | 1586 / 1387 / 1433 / 1171 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.3729 | 0.4666 | 0.1081 |
| Gaussian readout of the raw heads: MCC | 0.0624 | 0.0583 | -0.0024 |
| Gaussian readout of the raw heads: AUC | 0.5333 | 0.5360 | 0.4761 |
| Gaussian readout of the raw heads: Brier | 0.2490 | 0.2502 | 0.2538 |
| Gaussian readout of the raw heads: ECE | 0.0128 | 0.0249 | 0.0396 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 211.14 | 250.56 | 289.44 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 137.56 | 169.17 | 199.31 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | 0.0024 | -0.0028 | -0.0250 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | 0.0023 | -0.0027 | -0.0186 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0580 | 0.0715 | -0.0524 |
| corr, Spearman, raw heads | 0.0552 | 0.0535 | -0.0328 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -1.89 | -8.10 | -28.28 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.3745 | 0.4957 | 0.1031 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.75 | 128.34 | 152.34 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7314 | 6.9021 | 7.0859 |
| PIT KS | 0.0917 | 0.0968 | 0.1175 |
| var / err^2 Spearman | 0.2233 | 0.2603 | 0.2290 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0234 | [-0.0625, 0.0200] | NOISE |
| h1 | 0.0016 | [-0.0320, 0.0402] | NOISE |
| h2 | 0.0119 | [-0.0241, 0.0573] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7582 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.5912 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3858 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4854 | 0.4999 | 0.4930 | 0.1172 |
| expected if the two signs were independent | 0.4821 | 0.5011 | 0.4630 | 0.0981 |

- P(up) unanimity (all three horizons call the same side): 0.2595
