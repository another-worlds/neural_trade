# Evaluation report - test split - run `20260929T041215Z-8f35053-1d574af8-n3_r1_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.3243 | 0.7018 | 0.6416 |
| accuracy | 0.4971 | 0.5368 | 0.5225 |
| balanced accuracy | 0.4989 | 0.5353 | 0.5241 |
| precision (up) | 0.5032 | 0.5289 | 0.5131 |
| recall / sensitivity (up) | 0.3232 | 0.7368 | 0.6659 |
| specificity (down) | 0.6745 | 0.3337 | 0.3823 |
| F1 (up) | 0.3936 | 0.6158 | 0.5796 |
| MCC | -0.0024 | 0.0771 | 0.0503 |
| AUC | 0.4998 | 0.5451 | 0.5423 |
| Brier | 0.2563 | 0.2504 | 0.2496 |
| ECE (positive class) | 0.0724 | 0.0433 | 0.0312 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 796 / 786 / 1629 / 1667 | 1957 / 1743 / 873 / 699 | 1836 / 1742 / 1078 / 921 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.2866 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0958 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4470 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2503 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0486 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.2606 | 0.2866 | 0.4796 |
| Gaussian readout of the raw heads: MCC | 0.0357 | -0.0958 | -0.0024 |
| Gaussian readout of the raw heads: AUC | 0.5377 | 0.4470 | 0.4942 |
| Gaussian readout of the raw heads: Brier | 0.2497 | 0.2519 | 0.2516 |
| Gaussian readout of the raw heads: ECE | 0.0189 | 0.0615 | 0.0287 |

beta = 0 for h0, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.48 | 285.89 |
| RMSE ($), raw heads | 211.84 | 251.87 | 286.82 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.43 | 195.53 |
| MAE ($), raw heads | 137.76 | 169.67 | 197.21 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0022 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0042 | -0.0133 | -0.0065 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0024 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0039 | -0.0133 | -0.0067 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0163 | -0.0847 | 0.0119 |
| corr, Spearman, raw heads | 0.0552 | -0.0893 | -0.0035 |
| mean predicted ($), served | 0.00 | -1.72 | 0.00 |
| mean predicted ($), raw heads | -6.96 | -7.69 | -1.74 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.2413 | 0.2653 | 0.4645 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.2241 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 108.60 | 134.29 | 151.82 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7697 | 6.9851 | 7.0893 |
| PIT KS | 0.1236 | 0.1337 | 0.1070 |
| var / err^2 Spearman | 0.2481 | 0.2239 | 0.2315 |
| coverage of the 90% interval | 0.8802 | 0.8690 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 685.36 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0221 | [-0.0248, 0.0739] | NOISE |
| h1 | 0.0070 | [-0.0470, 0.0535] | NOISE |
| h2 | 0.0304 | [-0.0075, 0.0684] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.224 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4656 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.7160 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3167 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7196 | 0.4005 | 0.4746 | 0.1327 |
| expected if the two signs were independent | 0.6069 | 0.3982 | 0.4883 | 0.1269 |

- P(up) unanimity (all three horizons call the same side): 0.2374
