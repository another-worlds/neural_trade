# Evaluation report - dev split - run `20260929T124512Z-9446405-d8cdb9ce-h_1h__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (40 bars) | h1 (60 bars) | h2 (80 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 34654 | 36286 | 36934 |
| n_eff of the scored moves (n scored // bars ahead) | 866 | 604 | 461 |
| true up-rate | 0.4870 | 0.4827 | 0.4851 |
| calls up (predicted up-rate) | 0.4073 | 0.8797 | 0.6554 |
| accuracy | 0.5160 | 0.4756 | 0.5065 |
| balanced accuracy | 0.5136 | 0.4887 | 0.5111 |
| precision (up) | 0.5036 | 0.4763 | 0.4936 |
| recall / sensitivity (up) | 0.4212 | 0.8680 | 0.6668 |
| specificity (down) | 0.6059 | 0.1093 | 0.3555 |
| F1 (up) | 0.4588 | 0.6151 | 0.5673 |
| MCC | 0.0276 | -0.0348 | 0.0234 |
| AUC | 0.5105 | 0.4691 | 0.5254 |
| Brier | 0.2542 | 0.2553 | 0.2531 |
| ECE (positive class) | 0.0383 | 0.0636 | 0.0448 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0130 | 0.0173 | 0.0149 |
| TP / FP / TN / FN | 7109 / 7006 / 10771 / 9768 | 15204 / 16718 / 2052 / 2312 | 11947 / 12258 / 6760 / 5969 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.5167 | 0.6229 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0255 | -0.0317 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4795 | 0.4775 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2507 | 0.2509 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0248 | 0.0315 |
| Gaussian readout of the raw heads: calls up | 0.7047 | 0.5167 | 0.6229 |
| Gaussian readout of the raw heads: MCC | -0.0423 | -0.0255 | -0.0317 |
| Gaussian readout of the raw heads: AUC | 0.4803 | 0.4795 | 0.4775 |
| Gaussian readout of the raw heads: Brier | 0.2599 | 0.2632 | 0.2723 |
| Gaussian readout of the raw heads: ECE | 0.0872 | 0.0945 | 0.1247 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (40 bars) | h1 (60 bars) | h2 (80 bars) |
|---|---|---|---|
| RMSE ($), served | 326.58 | 400.52 | 463.04 |
| RMSE ($), raw heads | 332.53 | 411.27 | 483.49 |
| RMSE ($), zero prediction | 326.58 | 399.93 | 462.20 |
| MAE ($), served | 224.08 | 271.47 | 311.63 |
| MAE ($), raw heads | 229.61 | 281.43 | 330.37 |
| MAE ($), zero prediction | 224.08 | 270.91 | 310.92 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0030 | -0.0036 |
| skill vs zero, raw heads | -0.0368 | -0.0575 | -0.0942 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0025 | -0.0027 |
| EV, raw heads | -0.0250 | -0.0516 | -0.0751 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0283 | -0.0324 | -0.0355 |
| corr, Spearman, raw heads | -0.0461 | -0.0384 | -0.0426 |
| mean predicted ($), served | 0.00 | 3.04 | 5.89 |
| mean predicted ($), raw heads | 29.23 | 22.03 | 51.76 |
| mean realised ($) | -7.01 | -10.43 | -13.64 |
| share predicted up, raw heads | 0.7011 | 0.5094 | 0.6150 |
| share realised up | 0.4863 | 0.4823 | 0.4847 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1381 | 0.1139 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (40 bars) | h1 (60 bars) | h2 (80 bars) |
|---|---|---|---|
| CRPS ($) | 166.20 | 202.43 | 232.68 |
| CRPSS vs constant variance | 0.0153 | 0.0172 | 0.0241 |
| NLL | 7.2154 | 7.4302 | 7.5752 |
| PIT KS | 0.0331 | 0.0418 | 0.0431 |
| var / err^2 Spearman | 0.1284 | 0.0918 | 0.1258 |
| coverage of the 90% interval | 0.8970 | 0.9054 | 0.8987 |
| width of the 90% interval ($) | 977.12 | 1224.68 | 1413.59 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0018 | [-0.0234, 0.0196] | NOISE |
| h1 | -0.0327 | [-0.0582, -0.0052] | INVERTED |
| h2 | 0.0231 | [-0.0054, 0.0529] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.138 / h2 0.114) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7779 | n/a (beta = 0: served delta is 0) | 0.5993 |
| abs(d h1) <= abs(d h2) | 0.7412 | 0.6382 | 0.5794 |
| full chain h0 <= h1 <= h2 | 0.5428 | n/a (beta = 0: served delta is 0) | 0.3136 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5577 | 0.5619 | 0.6544 | 0.2275 |
| expected if the two signs were independent | 0.4553 | 0.5072 | 0.5335 | 0.1536 |

- P(up) unanimity (all three horizons call the same side): 0.3090

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (40 bars) | h1 (60 bars) | h2 (80 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0276 vs -0.0059 (+0.0335): beats, noise (boot z +1.08) | -0.0348 vs 0.0249 (-0.0598): does not beat, significantly worse (boot z -2.59) | 0.0234 vs 0.0345 (-0.0110): does not beat, noise (boot z -0.46) |
| logreg_lags | direction/auc | 0.5105 vs 0.5307 (-0.0202): does not beat, noise (boot z -1.13) | 0.4691 vs 0.5208 (-0.0517): does not beat, significantly worse (boot z -2.64) | 0.5254 vs 0.5290 (-0.0036): does not beat, noise (boot z -0.30) |
| logreg_lags | direction/brier | 0.2542 vs 0.2526 (-0.0016): does not beat, noise (DM z -0.69) | 0.2553 vs 0.2535 (-0.0018): does not beat, noise (DM z -1.82) | 0.2531 vs 0.2556 (+0.0025): beats, noise (DM z +1.62) |
| logreg_lags | direction/ece_pos | 0.0383 vs 0.0557 (+0.0174): beats, noise (boot z +1.11) | 0.0636 vs 0.0630 (-0.0006): does not beat, noise (boot z -0.10) | 0.0448 vs 0.0678 (+0.0230): beats (boot z +3.06) |
| logreg_lags | direction/acc | 0.5160 vs 0.4867 (+0.0293): beats, noise (DM z +1.58) | 0.4756 vs 0.4891 (-0.0136): does not beat, significantly worse (DM z -2.12) | 0.5065 vs 0.5023 (+0.0042): beats, noise (DM z +0.33) |
| logreg_lags | direction/bal_acc | 0.5136 vs 0.4992 (+0.0143): beats, noise (boot z +1.46) | 0.4887 vs 0.5050 (-0.0163): does not beat, significantly worse (boot z -2.93) | 0.5111 vs 0.5125 (-0.0014): does not beat, noise (boot z -0.14) |
| class_prior | direction/mcc | 0.0276 vs 0.0000 (+0.0276): beats, noise (boot z +1.56) | -0.0348 vs 0.0000 (-0.0348): does not beat, significantly worse (boot z -2.79) | 0.0234 vs 0.0000 (+0.0234): beats, noise (boot z +1.38) |
| class_prior | direction/auc | 0.5105 vs 0.5000 (+0.0105): beats, noise (boot z +0.89) | 0.4691 vs 0.5000 (-0.0309): does not beat, significantly worse (boot z -3.31) | 0.5254 vs 0.5000 (+0.0254): beats (boot z +2.08) |
| class_prior | direction/brier | 0.2542 vs 0.2523 (-0.0019): does not beat, noise (DM z -0.96) | 0.2553 vs 0.2532 (-0.0021): does not beat, significantly worse (DM z -3.37) | 0.2531 vs 0.2527 (-0.0004): does not beat, noise (DM z -0.23) |
| class_prior | direction/ece_pos | 0.0383 vs 0.0496 (+0.0114): beats, noise (boot z +0.72) | 0.0636 vs 0.0590 (-0.0046): does not beat, noise (boot z -0.84) | 0.0448 vs 0.0540 (+0.0092): beats, noise (boot z +1.11) |
| class_prior | direction/acc | 0.5160 vs 0.4870 (+0.0289): beats, noise (DM z +1.57) | 0.4756 vs 0.4827 (-0.0072): does not beat, noise (DM z -1.22) | 0.5065 vs 0.4851 (+0.0214): beats, noise (DM z +1.41) |
| class_prior | direction/bal_acc | 0.5136 vs 0.5000 (+0.0136): beats, noise (boot z +1.56) | 0.4887 vs 0.5000 (-0.0113): does not beat, significantly worse (boot z -2.78) | 0.5111 vs 0.5000 (+0.0111): beats, noise (boot z +1.38) |
| zero_delta | delta/rmse | 326.58 vs 326.58 (+0.00, +0.00%): does not beat | 400.52 vs 399.93 (-0.59, -0.15%): does not beat, noise (DM z -1.94) | 463.04 vs 462.20 (-0.83, -0.18%): does not beat, significantly worse (DM z -1.98) |
| zero_delta | delta/mae | 224.08 vs 224.08 (+0.00, +0.00%): does not beat | 271.47 vs 270.91 (-0.56, -0.21%): does not beat, significantly worse (DM z -2.36) | 311.63 vs 310.92 (-0.72, -0.23%): does not beat, significantly worse (DM z -2.18) |
| mean_delta | delta/rmse | 326.58 vs 329.16 (+2.58, +0.78%): beats (DM z +2.69) | 400.52 vs 404.68 (+4.15, +1.03%): beats (DM z +2.47) | 463.04 vs 469.44 (+6.40, +1.36%): beats (DM z +2.60) |
| mean_delta | delta/mae | 224.08 vs 227.03 (+2.95, +1.30%): beats (DM z +3.71) | 271.47 vs 276.40 (+4.93, +1.78%): beats (DM z +3.52) | 311.63 vs 318.91 (+7.27, +2.28%): beats (DM z +3.58) |
| const_var | variance/crps | 166.20 vs 168.79 (+2.59, +1.53%): beats (DM z +4.84) | 202.43 vs 205.98 (+3.54, +1.72%): beats (DM z +3.83) | 232.68 vs 238.42 (+5.74, +2.41%): beats (DM z +4.21) |
| const_var | variance/nll | 7.2154 vs 7.2339 (+0.0185): beats (DM z +2.22) | 7.4302 vs 7.4393 (+0.0091): beats, noise (DM z +0.86) | 7.5752 vs 7.5903 (+0.0151): beats, noise (DM z +1.08) |
| const_var | variance/pit_ks | 0.0331 vs 0.0873 (+0.0543): beats (boot z +7.68) | 0.0418 vs 0.1044 (+0.0626): beats (boot z +9.35) | 0.0431 vs 0.1174 (+0.0743): beats (boot z +14.08) |
| const_var | variance/corr_var_err2_spearman | 0.1284 vs 0.0000 (+0.1284): beats (boot z +6.74) | 0.0918 vs 0.0000 (+0.0918): beats (boot z +3.98) | 0.1258 vs 0.0000 (+0.1258): beats (boot z +5.38) |

## Backtest (costs included)

- n_trades: 1550
- total_return: -0.9828
- sharpe_net: -145.1078
- sharpe_gross: 0.4922
- sortino: -159.9464
- max_drawdown: 0.9828
- hit_rate: 0.0535
- hit_rate_gross: 0.4626
- profit_factor: 0.0285
- avg_hold_bars: 10.6200
- exposure: 0.3811
- turnover: 758.2876
- fees_paid: 7583.1236
- traded_notional: 7583123.6181
- breakeven_cost_bps: 0.0799
- gross_edge_per_trade_bps: -0.1539
- costs_paid: 9858.0607
- gross_pnl: 30.2768
- net_pnl: -9827.7839

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T22:38:00 .. 2025-08-30T22:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8502, long_above 0.5678, short_below 0.4680, median 0.5107. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.28% | -145.11 | +98.28% | 1550 |
| buy and hold | -6.74% | -2.35 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.37% .. -97.82%) | -98.11% | -150.69 | | |

The random null enters at the strategy's rate (0.0580 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 14% of its seeds on net return, 97% on net Sharpe and 56% on gross return.
