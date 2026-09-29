# Evaluation report - dev split - run `20260929T132543Z-f5aee70-dirty-3ee332f2-h_4h_L240__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 38831 | 39457 | 40128 |
| n_eff of the scored moves (n scored // bars ahead) | 242 | 164 | 125 |
| true up-rate | 0.4792 | 0.4762 | 0.4740 |
| calls up (predicted up-rate) | 0.5129 | 0.9068 | 0.5791 |
| accuracy | 0.5226 | 0.4860 | 0.5008 |
| balanced accuracy | 0.5232 | 0.5054 | 0.5049 |
| precision (up) | 0.5018 | 0.4791 | 0.4782 |
| recall / sensitivity (up) | 0.5371 | 0.9124 | 0.5843 |
| specificity (down) | 0.5093 | 0.0984 | 0.4256 |
| F1 (up) | 0.5188 | 0.6283 | 0.5259 |
| MCC | 0.0463 | 0.0186 | 0.0100 |
| AUC | 0.5332 | 0.5035 | 0.5061 |
| Brier | 0.2540 | 0.2540 | 0.2538 |
| ECE (positive class) | 0.0496 | 0.0648 | 0.0474 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0208 | 0.0238 | 0.0260 |
| TP / FP / TN / FN | 9994 / 9923 / 10299 / 8615 | 17143 / 18636 / 2033 / 1645 | 11112 / 12125 / 8984 / 7907 |
| Gaussian readout: calls up | 0.9868 | n/a (beta = 0: readout is the constant 0.5) | 0.9136 |
| Gaussian readout: MCC | -0.0498 | n/a (beta = 0: readout is the constant 0.5) | 0.0047 |
| Gaussian readout: AUC | 0.4885 | n/a (beta = 0: readout is the constant 0.5) | 0.5164 |
| Gaussian readout: Brier | 0.2501 | n/a (beta = 0: readout is the constant 0.5) | 0.2500 |
| Gaussian readout: ECE | 0.0279 | n/a (beta = 0: readout is the constant 0.5) | 0.0268 |
| Gaussian readout of the raw heads: calls up | 0.9868 | 0.9484 | 0.9136 |
| Gaussian readout of the raw heads: MCC | -0.0498 | -0.0140 | 0.0047 |
| Gaussian readout of the raw heads: AUC | 0.4885 | 0.5179 | 0.5164 |
| Gaussian readout of the raw heads: Brier | 0.2613 | 0.2598 | 0.2623 |
| Gaussian readout of the raw heads: ECE | 0.1038 | 0.1005 | 0.1051 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| RMSE ($), served | 652.32 | 809.64 | 949.42 |
| RMSE ($), raw heads | 671.09 | 827.11 | 973.48 |
| RMSE ($), zero prediction | 652.16 | 809.64 | 949.34 |
| MAE ($), served | 441.84 | 553.16 | 663.91 |
| MAE ($), raw heads | 462.60 | 571.72 | 688.01 |
| MAE ($), zero prediction | 441.66 | 553.16 | 663.87 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0005 | n/a (beta = 0: served delta is 0) | -0.0002 |
| skill vs zero, raw heads | -0.0589 | -0.0436 | -0.0515 |
| EV, served | -0.0001 | n/a (beta = 0: served delta is 0) | -0.0000 |
| EV, raw heads | -0.0078 | -0.0075 | -0.0130 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0124 | 0.0101 | -0.0003 |
| corr, Spearman, raw heads | -0.0183 | 0.0189 | 0.0246 |
| mean predicted ($), served | 3.02 | 0.00 | 1.31 |
| mean predicted ($), raw heads | 120.05 | 115.26 | 136.79 |
| mean realised ($) | -30.45 | -45.03 | -58.66 |
| share predicted up, raw heads | 0.9871 | 0.9490 | 0.9135 |
| share realised up | 0.4790 | 0.4781 | 0.4760 |
| shrink beta (served = beta x raw, fit on cal) | 0.0251 | 0.0000 | 0.0096 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| CRPS ($) | 338.31 | 419.58 | 500.36 |
| CRPSS vs constant variance | 0.0048 | 0.0373 | 0.0499 |
| NLL | 7.9051 | 8.2048 | 8.3827 |
| PIT KS | 0.0945 | 0.0480 | 0.0535 |
| var / err^2 Spearman | 0.0123 | -0.0386 | 0.0161 |
| coverage of the 90% interval | 0.8837 | 0.8582 | 0.8412 |
| width of the 90% interval ($) | 1903.66 | 2305.81 | 2629.64 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0287 | [-0.0043, 0.0609] | NOISE |
| h1 | -0.0172 | [-0.0496, 0.0194] | NOISE |
| h2 | -0.0044 | [-0.0362, 0.0297] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.025 / h1 0.000 / h2 0.010) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4596 | n/a (beta = 0: served delta is 0) | 0.6030 |
| abs(d h1) <= abs(d h2) | 0.6110 | n/a (beta = 0: served delta is 0) | 0.6010 |
| full chain h0 <= h1 <= h2 | 0.2759 | n/a (beta = 0: served delta is 0) | 0.3235 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4974 | 0.8677 | 0.5022 | 0.2772 |
| expected if the two signs were independent | 0.5099 | 0.8648 | 0.5635 | 0.2771 |

- P(up) unanimity (all three horizons call the same side): 0.3141

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0463 vs 0.0012 (+0.0452): beats, noise (boot z +1.27) | 0.0186 vs -0.0152 (+0.0338): beats, noise (boot z +0.89) | 0.0100 vs -0.0174 (+0.0274): beats, noise (boot z +0.83) |
| logreg_lags | direction/auc | 0.5332 vs 0.5233 (+0.0099): beats, noise (boot z +0.40) | 0.5035 vs 0.5143 (-0.0108): does not beat, noise (boot z -0.36) | 0.5061 vs 0.5109 (-0.0049): does not beat, noise (boot z -0.33) |
| logreg_lags | direction/brier | 0.2540 vs 0.2594 (+0.0054): beats, noise (DM z +1.04) | 0.2540 vs 0.2664 (+0.0124): beats (DM z +2.14) | 0.2538 vs 0.2710 (+0.0173): beats (DM z +2.44) |
| logreg_lags | direction/ece_pos | 0.0496 vs 0.0882 (+0.0386): beats (boot z +2.79) | 0.0648 vs 0.1143 (+0.0495): beats (boot z +6.66) | 0.0474 vs 0.1381 (+0.0907): beats (boot z +6.89) |
| logreg_lags | direction/acc | 0.5226 vs 0.4849 (+0.0377): beats, noise (DM z +1.47) | 0.4860 vs 0.4763 (+0.0097): beats, noise (DM z +0.64) | 0.5008 vs 0.4727 (+0.0281): beats, noise (DM z +0.82) |
| logreg_lags | direction/bal_acc | 0.5232 vs 0.5004 (+0.0228): beats, noise (boot z +1.53) | 0.5054 vs 0.4955 (+0.0099): beats, noise (boot z +0.88) | 0.5049 vs 0.4982 (+0.0068): beats, noise (boot z +0.51) |
| class_prior | direction/mcc | 0.0463 vs 0.0000 (+0.0463): beats, noise (boot z +1.87) | 0.0186 vs 0.0000 (+0.0186): beats, noise (boot z +0.95) | 0.0100 vs 0.0000 (+0.0100): beats, noise (boot z +0.38) |
| class_prior | direction/auc | 0.5332 vs 0.5000 (+0.0332): beats (boot z +2.08) | 0.5035 vs 0.5000 (+0.0035): beats, noise (boot z +0.25) | 0.5061 vs 0.5000 (+0.0061): beats, noise (boot z +0.34) |
| class_prior | direction/brier | 0.2540 vs 0.2520 (-0.0020): does not beat, noise (DM z -0.64) | 0.2540 vs 0.2536 (-0.0005): does not beat, noise (DM z -0.50) | 0.2538 vs 0.2586 (+0.0048): beats, noise (DM z +1.06) |
| class_prior | direction/ece_pos | 0.0496 vs 0.0489 (-0.0007): does not beat, noise (boot z -0.05) | 0.0648 vs 0.0642 (-0.0006): does not beat, noise (boot z -0.27) | 0.0474 vs 0.0964 (+0.0490): beats (boot z +3.71) |
| class_prior | direction/acc | 0.5226 vs 0.4792 (+0.0434): beats, noise (DM z +1.48) | 0.4860 vs 0.4762 (+0.0098): beats, noise (DM z +1.11) | 0.5008 vs 0.4740 (+0.0268): beats, noise (DM z +0.77) |
| class_prior | direction/bal_acc | 0.5232 vs 0.5000 (+0.0232): beats, noise (boot z +1.87) | 0.5054 vs 0.5000 (+0.0054): beats, noise (boot z +0.95) | 0.5049 vs 0.5000 (+0.0049): beats, noise (boot z +0.38) |
| zero_delta | delta/rmse | 652.32 vs 652.16 (-0.16, -0.03%): does not beat, noise (DM z -0.91) | 809.64 vs 809.64 (+0.00, +0.00%): does not beat | 949.42 vs 949.34 (-0.08, -0.01%): does not beat, noise (DM z -0.74) |
| zero_delta | delta/mae | 441.84 vs 441.66 (-0.18, -0.04%): does not beat, noise (DM z -1.20) | 553.16 vs 553.16 (+0.00, +0.00%): does not beat | 663.91 vs 663.87 (-0.04, -0.01%): does not beat, noise (DM z -0.45) |
| mean_delta | delta/rmse | 652.32 vs 663.66 (+11.34, +1.71%): beats (DM z +2.14) | 809.64 vs 833.81 (+24.17, +2.90%): beats (DM z +2.21) | 949.42 vs 988.50 (+39.08, +3.95%): beats (DM z +2.29) |
| mean_delta | delta/mae | 441.84 vs 453.49 (+11.65, +2.57%): beats (DM z +2.67) | 553.16 vs 579.15 (+25.99, +4.49%): beats (DM z +2.87) | 663.91 vs 705.48 (+41.57, +5.89%): beats (DM z +3.00) |
| const_var | variance/crps | 338.31 vs 339.95 (+1.64, +0.48%): beats, noise (DM z +0.43) | 419.58 vs 435.82 (+16.24, +3.73%): beats (DM z +2.66) | 500.36 vs 526.67 (+26.30, +4.99%): beats (DM z +2.75) |
| const_var | variance/nll | 7.9051 vs 8.0277 (+0.1226): beats, noise (DM z +1.95) | 8.2048 vs 8.2354 (+0.0306): beats, noise (DM z +0.97) | 8.3827 vs 8.3932 (+0.0105): beats, noise (DM z +0.25) |
| const_var | variance/pit_ks | 0.0945 vs 0.1050 (+0.0104): beats, noise (boot z +0.98) | 0.0480 vs 0.1434 (+0.0954): beats (boot z +11.22) | 0.0535 vs 0.1616 (+0.1081): beats (boot z +7.33) |
| const_var | variance/corr_var_err2_spearman | 0.0123 vs 0.0000 (+0.0123): beats, noise (boot z +0.45) | -0.0386 vs 0.0000 (-0.0386): does not beat, noise (boot z -1.18) | 0.0161 vs 0.0000 (+0.0161): beats, noise (boot z +0.64) |

## Backtest (costs included)

- n_trades: 1376
- total_return: -0.9718
- sharpe_net: -125.9619
- sharpe_gross: -0.4083
- sortino: -142.8447
- max_drawdown: 0.9718
- hit_rate: 0.0799
- hit_rate_gross: 0.4469
- profit_factor: 0.0390
- avg_hold_bars: 11.8844
- exposure: 0.3785
- turnover: 744.8222
- fees_paid: 7448.3441
- traded_notional: 7448344.0673
- breakeven_cost_bps: -0.0931
- gross_edge_per_trade_bps: 0.1314
- costs_paid: 9682.8473
- gross_pnl: -34.6573
- net_pnl: -9717.5046

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T18:38:00 .. 2025-08-30T18:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.096, long_above 0.5722, short_below 0.4814, median 0.5211. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.18% | -125.96 | +97.18% | 1376 |
| buy and hold | -8.03% | -2.84 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.43% .. -96.73%) | -97.13% | -139.17 | | |

The random null enters at the strategy's rate (0.0513 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 46% of its seeds on net return, 100% on net Sharpe and 47% on gross return.
