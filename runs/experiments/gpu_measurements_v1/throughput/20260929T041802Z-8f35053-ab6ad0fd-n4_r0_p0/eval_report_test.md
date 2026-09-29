# Evaluation report - test split - run `20260929T041802Z-8f35053-ab6ad0fd-n4_r0_p0`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.2665 | 0.3974 | 0.5110 |
| accuracy | 0.5139 | 0.5211 | 0.4949 |
| balanced accuracy | 0.5162 | 0.5218 | 0.4950 |
| precision (up) | 0.5354 | 0.5313 | 0.4895 |
| recall / sensitivity (up) | 0.2826 | 0.4191 | 0.5060 |
| specificity (down) | 0.7499 | 0.6246 | 0.4840 |
| F1 (up) | 0.3699 | 0.4685 | 0.4976 |
| MCC | 0.0367 | 0.0446 | -0.0100 |
| AUC | 0.5245 | 0.5326 | 0.5021 |
| Brier | 0.2518 | 0.2502 | 0.2534 |
| ECE (positive class) | 0.0469 | 0.0298 | 0.0589 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 696 / 604 / 1811 / 1767 | 1113 / 982 / 1634 / 1543 | 1395 / 1455 / 1365 / 1362 |
| Gaussian readout: calls up | 0.0152 | n/a (beta = 0: readout is the constant 0.5) | 0.5202 |
| Gaussian readout: MCC | -0.0247 | n/a (beta = 0: readout is the constant 0.5) | 0.0509 |
| Gaussian readout: AUC | 0.4923 | n/a (beta = 0: readout is the constant 0.5) | 0.5265 |
| Gaussian readout: Brier | 0.2501 | n/a (beta = 0: readout is the constant 0.5) | 0.2500 |
| Gaussian readout: ECE | 0.0108 | n/a (beta = 0: readout is the constant 0.5) | 0.0128 |
| Gaussian readout of the raw heads: calls up | 0.0152 | 0.2240 | 0.5202 |
| Gaussian readout of the raw heads: MCC | -0.0247 | 0.0701 | 0.0509 |
| Gaussian readout of the raw heads: AUC | 0.4923 | 0.5153 | 0.5265 |
| Gaussian readout of the raw heads: Brier | 0.2567 | 0.2510 | 0.2509 |
| Gaussian readout of the raw heads: ECE | 0.0742 | 0.0361 | 0.0102 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.41 | 250.20 | 285.83 |
| RMSE ($), raw heads | 213.90 | 250.39 | 286.78 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.52 | 168.19 | 195.61 |
| MAE ($), raw heads | 140.30 | 169.23 | 196.53 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0002 | n/a (beta = 0: served delta is 0) | 0.0004 |
| skill vs zero, raw heads | -0.0239 | -0.0015 | -0.0063 |
| EV, served | -0.0002 | n/a (beta = 0: served delta is 0) | 0.0003 |
| EV, raw heads | -0.0094 | -0.0006 | -0.0065 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0415 | 0.0721 | 0.0257 |
| corr, Spearman, raw heads | -0.0070 | 0.0388 | 0.0488 |
| mean predicted ($), served | -1.23 | 0.00 | -0.96 |
| mean predicted ($), raw heads | -27.80 | -12.20 | -2.33 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.0122 | 0.2132 | 0.5299 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0442 | 0.0000 | 0.4106 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 104.96 | 127.60 | 147.30 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7126 | 6.8885 | 7.0318 |
| PIT KS | 0.0851 | 0.0896 | 0.0762 |
| var / err^2 Spearman | 0.2231 | 0.2335 | 0.2289 |
| coverage of the 90% interval | 0.8813 | 0.8702 | 0.8517 |
| width of the 90% interval ($) | 586.05 | 687.16 | 763.92 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0337 | [-0.0051, 0.0918] | NOISE |
| h1 | 0.0455 | [0.0107, 0.0880] | WORKS |
| h2 | 0.0160 | [-0.0298, 0.0672] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.044 / h1 0.000 / h2 0.411) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4353 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.2957 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.0453 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7613 | 0.5873 | 0.6038 | 0.2725 |
| expected if the two signs were independent | 0.7569 | 0.5681 | 0.4996 | 0.2281 |

- P(up) unanimity (all three horizons call the same side): 0.3988
