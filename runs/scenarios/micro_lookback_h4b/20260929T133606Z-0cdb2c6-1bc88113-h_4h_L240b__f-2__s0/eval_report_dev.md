# Evaluation report - dev split - run `20260929T133606Z-0cdb2c6-1bc88113-h_4h_L240b__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 38831 | 39457 | 40128 |
| n_eff of the scored moves (n scored // bars ahead) | 242 | 164 | 125 |
| true up-rate | 0.4792 | 0.4762 | 0.4740 |
| calls up (predicted up-rate) | 0.5790 | 0.8536 | 0.8448 |
| accuracy | 0.5144 | 0.4860 | 0.4796 |
| balanced accuracy | 0.5177 | 0.5029 | 0.4975 |
| precision (up) | 0.4945 | 0.4778 | 0.4725 |
| recall / sensitivity (up) | 0.5975 | 0.8566 | 0.8422 |
| specificity (down) | 0.4380 | 0.1492 | 0.1528 |
| F1 (up) | 0.5411 | 0.6135 | 0.6054 |
| MCC | 0.0359 | 0.0081 | -0.0068 |
| AUC | 0.5327 | 0.5056 | 0.5056 |
| Brier | 0.2548 | 0.2529 | 0.2582 |
| ECE (positive class) | 0.0585 | 0.0555 | 0.0835 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0208 | 0.0238 | 0.0260 |
| TP / FP / TN / FN | 11118 / 11364 / 8858 / 7491 | 16093 / 17586 / 3083 / 2695 | 16018 / 17883 / 3226 / 3001 |
| Gaussian readout: calls up | 0.9862 | n/a (beta = 0: readout is the constant 0.5) | 0.8254 |
| Gaussian readout: MCC | -0.0476 | n/a (beta = 0: readout is the constant 0.5) | 0.0109 |
| Gaussian readout: AUC | 0.4866 | n/a (beta = 0: readout is the constant 0.5) | 0.5154 |
| Gaussian readout: Brier | 0.2501 | n/a (beta = 0: readout is the constant 0.5) | 0.2500 |
| Gaussian readout: ECE | 0.0281 | n/a (beta = 0: readout is the constant 0.5) | 0.0270 |
| Gaussian readout of the raw heads: calls up | 0.9862 | 0.9182 | 0.8254 |
| Gaussian readout of the raw heads: MCC | -0.0476 | -0.0074 | 0.0109 |
| Gaussian readout of the raw heads: AUC | 0.4866 | 0.5213 | 0.5154 |
| Gaussian readout of the raw heads: Brier | 0.2618 | 0.2598 | 0.2572 |
| Gaussian readout of the raw heads: ECE | 0.1046 | 0.0990 | 0.0791 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| RMSE ($), served | 652.36 | 809.64 | 949.45 |
| RMSE ($), raw heads | 671.75 | 828.24 | 964.92 |
| RMSE ($), zero prediction | 652.16 | 809.64 | 949.34 |
| MAE ($), served | 441.88 | 553.16 | 663.92 |
| MAE ($), raw heads | 463.40 | 573.12 | 678.73 |
| MAE ($), zero prediction | 441.66 | 553.16 | 663.87 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0006 | n/a (beta = 0: served delta is 0) | -0.0002 |
| skill vs zero, raw heads | -0.0610 | -0.0465 | -0.0331 |
| EV, served | -0.0001 | n/a (beta = 0: served delta is 0) | -0.0000 |
| EV, raw heads | -0.0093 | -0.0096 | -0.0110 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0127 | 0.0155 | -0.0021 |
| corr, Spearman, raw heads | -0.0225 | 0.0243 | 0.0233 |
| mean predicted ($), served | 3.60 | 0.00 | 1.76 |
| mean predicted ($), raw heads | 120.94 | 116.91 | 94.42 |
| mean realised ($) | -30.45 | -45.03 | -58.66 |
| share predicted up, raw heads | 0.9866 | 0.9196 | 0.8258 |
| share realised up | 0.4790 | 0.4781 | 0.4760 |
| shrink beta (served = beta x raw, fit on cal) | 0.0298 | 0.0000 | 0.0186 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| CRPS ($) | 338.30 | 419.64 | 500.08 |
| CRPSS vs constant variance | 0.0049 | 0.0371 | 0.0505 |
| NLL | 7.9060 | 8.1992 | 8.3599 |
| PIT KS | 0.0942 | 0.0499 | 0.0497 |
| var / err^2 Spearman | 0.0061 | -0.0430 | 0.0151 |
| coverage of the 90% interval | 0.8838 | 0.8582 | 0.8416 |
| width of the 90% interval ($) | 1903.87 | 2305.81 | 2632.12 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0321 | [-0.0015, 0.0653] | NOISE |
| h1 | -0.0089 | [-0.0410, 0.0260] | NOISE |
| h2 | -0.0016 | [-0.0410, 0.0355] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.030 / h1 0.000 / h2 0.019) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4654 | n/a (beta = 0: served delta is 0) | 0.6030 |
| abs(d h1) <= abs(d h2) | 0.4161 | n/a (beta = 0: served delta is 0) | 0.6010 |
| full chain h0 <= h1 <= h2 | 0.1556 | n/a (beta = 0: served delta is 0) | 0.3235 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5627 | 0.8117 | 0.6730 | 0.3964 |
| expected if the two signs were independent | 0.5740 | 0.7967 | 0.7233 | 0.3608 |

- P(up) unanimity (all three horizons call the same side): 0.4480

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0359 vs 0.0012 (+0.0347): beats, noise (boot z +0.97) | 0.0081 vs -0.0152 (+0.0233): beats, noise (boot z +0.58) | -0.0068 vs -0.0174 (+0.0106): beats, noise (boot z +0.34) |
| logreg_lags | direction/auc | 0.5327 vs 0.5233 (+0.0094): beats, noise (boot z +0.38) | 0.5056 vs 0.5143 (-0.0087): does not beat, noise (boot z -0.29) | 0.5056 vs 0.5109 (-0.0054): does not beat, noise (boot z -0.36) |
| logreg_lags | direction/brier | 0.2548 vs 0.2594 (+0.0046): beats, noise (DM z +0.93) | 0.2529 vs 0.2664 (+0.0135): beats (DM z +2.21) | 0.2582 vs 0.2710 (+0.0128): beats (DM z +2.78) |
| logreg_lags | direction/ece_pos | 0.0585 vs 0.0882 (+0.0297): beats (boot z +2.35) | 0.0555 vs 0.1143 (+0.0588): beats (boot z +6.97) | 0.0835 vs 0.1381 (+0.0546): beats (boot z +7.94) |
| logreg_lags | direction/acc | 0.5144 vs 0.4849 (+0.0295): beats, noise (DM z +1.27) | 0.4860 vs 0.4763 (+0.0097): beats, noise (DM z +0.56) | 0.4796 vs 0.4727 (+0.0069): beats, noise (DM z +0.47) |
| logreg_lags | direction/bal_acc | 0.5177 vs 0.5004 (+0.0174): beats, noise (boot z +1.17) | 0.5029 vs 0.4955 (+0.0074): beats, noise (boot z +0.58) | 0.4975 vs 0.4982 (-0.0007): does not beat, noise (boot z -0.07) |
| class_prior | direction/mcc | 0.0359 vs 0.0000 (+0.0359): beats, noise (boot z +1.47) | 0.0081 vs 0.0000 (+0.0081): beats, noise (boot z +0.38) | -0.0068 vs 0.0000 (-0.0068): does not beat, noise (boot z -0.28) |
| class_prior | direction/auc | 0.5327 vs 0.5000 (+0.0327): beats (boot z +2.04) | 0.5056 vs 0.5000 (+0.0056): beats, noise (boot z +0.39) | 0.5056 vs 0.5000 (+0.0056): beats, noise (boot z +0.32) |
| class_prior | direction/brier | 0.2548 vs 0.2520 (-0.0028): does not beat, noise (DM z -0.93) | 0.2529 vs 0.2536 (+0.0007): beats, noise (DM z +0.60) | 0.2582 vs 0.2586 (+0.0004): beats, noise (DM z +0.16) |
| class_prior | direction/ece_pos | 0.0585 vs 0.0489 (-0.0096): does not beat, noise (boot z -0.73) | 0.0555 vs 0.0642 (+0.0087): beats, noise (boot z +1.78) | 0.0835 vs 0.0964 (+0.0129): beats, noise (boot z +1.92) |
| class_prior | direction/acc | 0.5144 vs 0.4792 (+0.0352): beats, noise (DM z +1.35) | 0.4860 vs 0.4762 (+0.0098): beats, noise (DM z +0.77) | 0.4796 vs 0.4740 (+0.0056): beats, noise (DM z +0.38) |
| class_prior | direction/bal_acc | 0.5177 vs 0.5000 (+0.0177): beats, noise (boot z +1.47) | 0.5029 vs 0.5000 (+0.0029): beats, noise (boot z +0.38) | 0.4975 vs 0.5000 (-0.0025): does not beat, noise (boot z -0.28) |
| zero_delta | delta/rmse | 652.36 vs 652.16 (-0.20, -0.03%): does not beat, noise (DM z -0.91) | 809.64 vs 809.64 (+0.00, +0.00%): does not beat | 949.45 vs 949.34 (-0.12, -0.01%): does not beat, noise (DM z -0.73) |
| zero_delta | delta/mae | 441.88 vs 441.66 (-0.22, -0.05%): does not beat, noise (DM z -1.23) | 553.16 vs 553.16 (+0.00, +0.00%): does not beat | 663.92 vs 663.87 (-0.05, -0.01%): does not beat, noise (DM z -0.40) |
| mean_delta | delta/rmse | 652.36 vs 663.66 (+11.30, +1.70%): beats (DM z +2.15) | 809.64 vs 833.81 (+24.17, +2.90%): beats (DM z +2.21) | 949.45 vs 988.50 (+39.05, +3.95%): beats (DM z +2.29) |
| mean_delta | delta/mae | 441.88 vs 453.49 (+11.61, +2.56%): beats (DM z +2.68) | 553.16 vs 579.15 (+25.99, +4.49%): beats (DM z +2.87) | 663.92 vs 705.48 (+41.56, +5.89%): beats (DM z +3.01) |
| const_var | variance/crps | 338.30 vs 339.95 (+1.65, +0.49%): beats, noise (DM z +0.44) | 419.64 vs 435.82 (+16.18, +3.71%): beats (DM z +2.64) | 500.08 vs 526.67 (+26.59, +5.05%): beats (DM z +2.78) |
| const_var | variance/nll | 7.9060 vs 8.0277 (+0.1216): beats (DM z +1.96) | 8.1992 vs 8.2354 (+0.0362): beats, noise (DM z +1.21) | 8.3599 vs 8.3932 (+0.0333): beats, noise (DM z +0.95) |
| const_var | variance/pit_ks | 0.0942 vs 0.1050 (+0.0107): beats, noise (boot z +1.02) | 0.0499 vs 0.1434 (+0.0935): beats (boot z +11.52) | 0.0497 vs 0.1616 (+0.1119): beats (boot z +8.98) |
| const_var | variance/corr_var_err2_spearman | 0.0061 vs 0.0000 (+0.0061): beats, noise (boot z +0.22) | -0.0430 vs 0.0000 (-0.0430): does not beat, noise (boot z -1.32) | 0.0151 vs 0.0000 (+0.0151): beats, noise (boot z +0.64) |

## Backtest (costs included)

- n_trades: 1300
- total_return: -0.9643
- sharpe_net: -119.4148
- sharpe_gross: 1.0733
- sortino: -136.5270
- max_drawdown: 0.9643
- hit_rate: 0.0846
- hit_rate_gross: 0.4569
- profit_factor: 0.0425
- avg_hold_bars: 12.1962
- exposure: 0.3670
- turnover: 747.9876
- fees_paid: 7479.8865
- traded_notional: 7479886.4843
- breakeven_cost_bps: 0.2153
- gross_edge_per_trade_bps: 0.4088
- costs_paid: 9723.8524
- gross_pnl: 80.5060
- net_pnl: -9643.3464

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T18:38:00 .. 2025-08-30T18:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.118, long_above 0.5831, short_below 0.4941, median 0.5333. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.43% | -119.41 | +96.43% | 1300 |
| buy and hold | -8.03% | -2.84 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.03% .. -96.12%) | -96.62% | -135.83 | | |

The random null enters at the strategy's rate (0.0475 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 77% of its seeds on net return, 100% on net Sharpe and 64% on gross return.
