# Evaluation report - test split - run `20260929T040953Z-8f35053-b29124dd-n3_r0_p2`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.3372 | 0.5211 | 0.5987 |
| accuracy | 0.5113 | 0.4991 | 0.4847 |
| balanced accuracy | 0.5129 | 0.4989 | 0.4858 |
| precision (up) | 0.5240 | 0.5027 | 0.4825 |
| recall / sensitivity (up) | 0.3500 | 0.5200 | 0.5843 |
| specificity (down) | 0.6758 | 0.4778 | 0.3872 |
| F1 (up) | 0.4197 | 0.5112 | 0.5285 |
| MCC | 0.0272 | -0.0022 | -0.0290 |
| AUC | 0.5199 | 0.5059 | 0.4778 |
| Brier | 0.2533 | 0.2577 | 0.2571 |
| ECE (positive class) | 0.0398 | 0.0776 | 0.0665 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 862 / 783 / 1632 / 1601 | 1381 / 1366 / 1250 / 1275 | 1611 / 1728 / 1092 / 1146 |
| Gaussian readout: calls up | 0.6349 | 0.3759 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0471 | -0.0216 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5323 | 0.4840 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2499 | 0.2511 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0220 | 0.0248 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6349 | 0.3759 | 0.1872 |
| Gaussian readout of the raw heads: MCC | 0.0471 | -0.0216 | 0.0551 |
| Gaussian readout of the raw heads: AUC | 0.5323 | 0.4840 | 0.5476 |
| Gaussian readout of the raw heads: Brier | 0.2501 | 0.2538 | 0.2517 |
| Gaussian readout of the raw heads: ECE | 0.0132 | 0.0447 | 0.0550 |

beta = 0 for h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.32 | 250.82 | 285.89 |
| RMSE ($), raw heads | 211.88 | 252.52 | 290.05 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.53 | 168.83 | 195.53 |
| MAE ($), raw heads | 138.98 | 170.57 | 201.04 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0007 | -0.0050 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0046 | -0.0186 | -0.0293 |
| EV, served | 0.0008 | -0.0051 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0019 | -0.0187 | -0.0131 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0481 | -0.0449 | 0.0814 |
| corr, Spearman, raw heads | 0.0512 | -0.0247 | 0.0904 |
| mean predicted ($), served | 0.70 | -2.35 | 0.00 |
| mean predicted ($), raw heads | 8.85 | -5.88 | -41.64 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.6540 | 0.3568 | 0.1906 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0792 | 0.4003 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 108.52 | 128.69 | 149.92 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7660 | 6.9217 | 7.0638 |
| PIT KS | 0.1216 | 0.0898 | 0.0966 |
| var / err^2 Spearman | 0.2180 | 0.1750 | 0.1880 |
| coverage of the 90% interval | 0.8803 | 0.8687 | 0.8523 |
| width of the 90% interval ($) | 584.82 | 684.66 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0309 | [-0.0148, 0.0892] | NOISE |
| h1 | 0.0162 | [-0.0310, 0.0587] | NOISE |
| h2 | -0.0291 | [-0.0782, 0.0209] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.079 / h1 0.400 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6086 | 0.9073 | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.7588 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.3810 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4641 | 0.5561 | 0.4288 | 0.1421 |
| expected if the two signs were independent | 0.4449 | 0.4898 | 0.4359 | 0.1097 |

- P(up) unanimity (all three horizons call the same side): 0.0706
