# Evaluation report - test split - run `20260929T041802Z-8f35053-8eefad38-n4_r0_p3`

n = 7236 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 4878 | 5272 | 5577 |
| n_eff of the scored moves (n scored // bars ahead) | 487 | 351 | 278 |
| true up-rate | 0.5049 | 0.5038 | 0.4944 |
| calls up (predicted up-rate) | 0.3083 | 0.2453 | 0.5186 |
| accuracy | 0.4881 | 0.4888 | 0.4952 |
| balanced accuracy | 0.4900 | 0.4907 | 0.4955 |
| precision (up) | 0.4887 | 0.4849 | 0.4900 |
| recall / sensitivity (up) | 0.2984 | 0.2361 | 0.5140 |
| specificity (down) | 0.6816 | 0.7454 | 0.4770 |
| F1 (up) | 0.3706 | 0.3175 | 0.5017 |
| MCC | -0.0217 | -0.0215 | -0.0091 |
| AUC | 0.4874 | 0.4766 | 0.5075 |
| Brier | 0.2566 | 0.2630 | 0.2578 |
| ECE (positive class) | 0.0654 | 0.0932 | 0.0805 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0049 | 0.0038 | 0.0056 |
| TP / FP / TN / FN | 735 / 769 / 1646 / 1728 | 627 / 666 / 1950 / 2029 | 1417 / 1475 / 1345 / 1340 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.2786 | 0.1142 | 0.0922 |
| Gaussian readout of the raw heads: MCC | 0.0218 | 0.0044 | 0.0656 |
| Gaussian readout of the raw heads: AUC | 0.5099 | 0.5216 | 0.5430 |
| Gaussian readout of the raw heads: Brier | 0.2515 | 0.2571 | 0.2524 |
| Gaussian readout of the raw heads: ECE | 0.0293 | 0.0772 | 0.0540 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 211.40 | 250.20 | 285.89 |
| RMSE ($), raw heads | 211.64 | 253.72 | 288.28 |
| RMSE ($), zero prediction | 211.40 | 250.20 | 285.89 |
| MAE ($), served | 137.51 | 168.19 | 195.53 |
| MAE ($), raw heads | 138.51 | 173.25 | 198.58 |
| MAE ($), zero prediction | 137.51 | 168.19 | 195.53 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0023 | -0.0283 | -0.0168 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0016 | -0.0175 | -0.0089 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0334 | 0.0011 | 0.0446 |
| corr, Spearman, raw heads | 0.0290 | 0.0467 | 0.0788 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -8.33 | -29.85 | -30.87 |
| mean realised ($) | -2.28 | -3.60 | -4.94 |
| share predicted up, raw heads | 0.2923 | 0.1020 | 0.0911 |
| share realised up | 0.4974 | 0.5057 | 0.4977 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 105.40 | 127.34 | 147.93 |
| CRPSS vs constant variance | n/a | n/a | n/a |
| NLL | 6.7146 | 6.8963 | 7.0399 |
| PIT KS | 0.0942 | 0.0809 | 0.0814 |
| var / err^2 Spearman | 0.2585 | 0.2497 | 0.2358 |
| coverage of the 90% interval | 0.8802 | 0.8702 | 0.8523 |
| width of the 90% interval ($) | 584.96 | 687.16 | 767.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0021 | [-0.0354, 0.0491] | NOISE |
| h1 | -0.0335 | [-0.0844, 0.0177] | NOISE |
| h2 | 0.0312 | [-0.0050, 0.0651] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.9028 | n/a (beta = 0: served delta is 0) | 0.5949 |
| abs(d h1) <= abs(d h2) | 0.3654 | n/a (beta = 0: served delta is 0) | 0.5760 |
| full chain h0 <= h1 <= h2 | 0.2836 | n/a (beta = 0: served delta is 0) | 0.3062 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5612 | 0.7588 | 0.4863 | 0.2105 |
| expected if the two signs were independent | 0.5913 | 0.7173 | 0.4914 | 0.2355 |

- P(up) unanimity (all three horizons call the same side): 0.3765
