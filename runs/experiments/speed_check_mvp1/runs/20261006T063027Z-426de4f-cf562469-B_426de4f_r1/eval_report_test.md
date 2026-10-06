# Evaluation report - test split - run `20261006T063027Z-426de4f-cf562469-B_426de4f_r1`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 5345 | 5697 | 5874 |
| n_eff of the scored moves (n scored // bars ahead) | 534 | 379 | 293 |
| true up-rate | 0.5139 | 0.5147 | 0.5189 |
| calls up (predicted up-rate) | 0.5358 | 0.5563 | 0.6095 |
| accuracy | 0.5209 | 0.5048 | 0.5043 |
| balanced accuracy | 0.5199 | 0.5032 | 0.5001 |
| precision (up) | 0.5325 | 0.5175 | 0.5190 |
| recall / sensitivity (up) | 0.5552 | 0.5593 | 0.6096 |
| specificity (down) | 0.4846 | 0.4470 | 0.3907 |
| F1 (up) | 0.5436 | 0.5376 | 0.5607 |
| MCC | 0.0398 | 0.0064 | 0.0002 |
| AUC | 0.5277 | 0.5093 | 0.5012 |
| Brier | 0.2496 | 0.2500 | 0.2509 |
| ECE (positive class) | 0.0120 | 0.0184 | 0.0183 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0139 | 0.0147 | 0.0189 |
| TP / FP / TN / FN | 1525 / 1339 / 1259 / 1222 | 1640 / 1529 / 1236 / 1292 | 1858 / 1722 / 1104 / 1190 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.4270 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0300 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.5152 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.2499 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | 0.0192 |
| Gaussian readout of the raw heads: calls up | 0.4442 | 0.4025 | 0.4270 |
| Gaussian readout of the raw heads: MCC | -0.0053 | 0.0372 | 0.0300 |
| Gaussian readout of the raw heads: AUC | 0.5118 | 0.5141 | 0.5152 |
| Gaussian readout of the raw heads: Brier | 0.2516 | 0.2584 | 0.2578 |
| Gaussian readout of the raw heads: ECE | 0.0432 | 0.0617 | 0.0636 |

beta = 0 for h0, h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 196.21 | 236.10 | 269.03 |
| RMSE ($), raw heads | 197.38 | 244.81 | 276.37 |
| RMSE ($), zero prediction | 196.21 | 236.10 | 269.11 |
| MAE ($), served | 145.66 | 175.66 | 199.90 |
| MAE ($), raw heads | 146.71 | 182.22 | 206.16 |
| MAE ($), zero prediction | 145.66 | 175.66 | 199.93 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | 0.0006 |
| skill vs zero, raw heads | -0.0120 | -0.0751 | -0.0547 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | 0.0006 |
| EV, raw heads | -0.0123 | -0.0746 | -0.0539 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0507 | 0.0320 | 0.0385 |
| corr, Spearman, raw heads | 0.0281 | 0.0312 | 0.0325 |
| mean predicted ($), served | 0.00 | 0.00 | -0.08 |
| mean predicted ($), raw heads | 1.22 | -1.79 | -2.48 |
| mean realised ($) | 6.30 | 9.37 | 12.34 |
| share predicted up, raw heads | 0.4333 | 0.3948 | 0.4182 |
| share realised up | 0.5135 | 0.5146 | 0.5156 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0333 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.51 | 126.98 | 144.60 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.6473 | 6.8428 | 6.9651 |
| PIT KS | 0.0319 | 0.0297 | 0.0333 |
| var / err^2 Spearman | 0.2535 | 0.2400 | 0.2424 |
| coverage of the 90% interval | 0.9027 | 0.9059 | 0.9125 |
| width of the 90% interval ($) | 643.12 | 781.37 | 912.36 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0269 | [-0.0069, 0.0612] | NOISE |
| h1 | 0.0345 | [-0.0045, 0.0737] | NOISE |
| h2 | -0.0027 | [-0.0524, 0.0360] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.033) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8604 | n/a (beta = 0: served delta is 0) | 0.6039 |
| abs(d h1) <= abs(d h2) | 0.6248 | n/a (beta = 0: served delta is 0) | 0.5712 |
| full chain h0 <= h1 <= h2 | 0.5198 | n/a (beta = 0: served delta is 0) | 0.3159 |

beta = 0 for h0, h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6224 | 0.5536 | 0.6501 | 0.2224 |
| expected if the two signs were independent | 0.4964 | 0.4855 | 0.4826 | 0.1259 |

- P(up) unanimity (all three horizons call the same side): 0.2927
