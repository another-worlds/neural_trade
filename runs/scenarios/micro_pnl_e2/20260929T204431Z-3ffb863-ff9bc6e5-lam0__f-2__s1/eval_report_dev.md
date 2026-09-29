# Evaluation report - dev split - run `20260929T204431Z-3ffb863-ff9bc6e5-lam0__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5851 | 0.8396 | 0.8956 |
| accuracy | 0.4798 | 0.4848 | 0.4799 |
| balanced accuracy | 0.4828 | 0.4964 | 0.4986 |
| precision (up) | 0.4677 | 0.4808 | 0.4755 |
| recall / sensitivity (up) | 0.5673 | 0.8359 | 0.8942 |
| specificity (down) | 0.3983 | 0.1569 | 0.1030 |
| F1 (up) | 0.5128 | 0.6105 | 0.6209 |
| MCC | -0.0349 | -0.0098 | -0.0046 |
| AUC | 0.4820 | 0.5025 | 0.5013 |
| Brier | 0.2553 | 0.2560 | 0.2589 |
| ECE (positive class) | 0.0629 | 0.0683 | 0.0853 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 9933 / 11303 / 7481 / 7575 | 15262 / 16481 / 3067 / 2996 | 16800 / 18528 / 2128 / 1988 |
| Gaussian readout: calls up | 0.9491 | 0.9066 | 0.9024 |
| Gaussian readout: MCC | 0.0088 | 0.0029 | 0.0067 |
| Gaussian readout: AUC | 0.4976 | 0.5185 | 0.5116 |
| Gaussian readout: Brier | 0.2512 | 0.2516 | 0.2519 |
| Gaussian readout: ECE | 0.0357 | 0.0436 | 0.0501 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.50 | 566.40 | 811.07 |
| RMSE ($), raw heads | 411.20 | 577.77 | 821.67 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.31 | 379.19 | 553.95 |
| MAE ($), raw heads | 279.90 | 387.66 | 564.51 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0060 | -0.0089 | -0.0085 |
| skill vs zero, raw heads | -0.0552 | -0.0498 | -0.0351 |
| EV, served | -0.0032 | -0.0034 | -0.0007 |
| EV, raw heads | -0.0265 | -0.0215 | -0.0078 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0423 | -0.0151 | 0.0136 |
| corr, Spearman, raw heads | -0.0119 | 0.0168 | 0.0041 |
| mean predicted ($), served | 12.88 | 25.50 | 40.59 |
| mean predicted ($), raw heads | 57.70 | 75.41 | 97.19 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9474 | 0.9022 | 0.9025 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.2233 | 0.3381 | 0.4176 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.70 | 285.68 | 419.66 |
| CRPSS vs constant variance | 0.0186 | 0.0306 | 0.0566 |
| NLL | 7.4341 | 7.8377 | 8.1678 |
| PIT KS | 0.0520 | 0.0589 | 0.0758 |
| var / err^2 Spearman | 0.0990 | -0.0443 | 0.0897 |
| coverage of the 90% interval | 0.9000 | 0.9006 | 0.8647 |
| width of the 90% interval ($) | 1209.08 | 1756.27 | 2418.87 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0111 | [-0.0406, 0.0181] | NOISE |
| h1 | -0.0015 | [-0.0354, 0.0304] | NOISE |
| h2 | -0.0091 | [-0.0351, 0.0145] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.223 / h1 0.338 / h2 0.418) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6420 | 0.8797 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.5861 | 0.8367 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.3309 | 0.7425 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5770 | 0.8130 | 0.8072 | 0.4045 |
| expected if the two signs were independent | 0.5925 | 0.7680 | 0.8190 | 0.4015 |

- P(up) unanimity (all three horizons call the same side): 0.4721

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0349 vs 0.0025 (-0.0374): does not beat, noise (boot z -1.32) | -0.0098 vs 0.0205 (-0.0303): does not beat, noise (boot z -0.96) | -0.0046 vs -0.0138 (+0.0093): beats, noise (boot z +0.42) |
| logreg_lags | direction/auc | 0.4820 vs 0.5320 (-0.0500): does not beat, significantly worse (boot z -2.01) | 0.5025 vs 0.5251 (-0.0226): does not beat, noise (boot z -1.03) | 0.5013 vs 0.5095 (-0.0082): does not beat, noise (boot z -0.45) |
| logreg_lags | direction/brier | 0.2553 vs 0.2536 (-0.0017): does not beat, noise (DM z -0.66) | 0.2560 vs 0.2575 (+0.0014): beats, noise (DM z +0.56) | 0.2589 vs 0.2702 (+0.0113): beats (DM z +2.61) |
| logreg_lags | direction/ece_pos | 0.0629 vs 0.0663 (+0.0034): beats, noise (boot z +0.20) | 0.0683 vs 0.0812 (+0.0129): beats, noise (boot z +1.87) | 0.0853 vs 0.1230 (+0.0377): beats (boot z +7.28) |
| logreg_lags | direction/acc | 0.4798 vs 0.4836 (-0.0038): does not beat, noise (DM z -0.22) | 0.4848 vs 0.4911 (-0.0063): does not beat, noise (DM z -0.52) | 0.4799 vs 0.4756 (+0.0043): beats, noise (DM z +0.60) |
| logreg_lags | direction/bal_acc | 0.4828 vs 0.5004 (-0.0176): does not beat, noise (boot z -1.61) | 0.4964 vs 0.5055 (-0.0091): does not beat, noise (boot z -0.92) | 0.4986 vs 0.4971 (+0.0015): beats, noise (boot z +0.27) |
| class_prior | direction/mcc | -0.0349 vs 0.0000 (-0.0349): does not beat, noise (boot z -1.59) | -0.0098 vs 0.0000 (-0.0098): does not beat, noise (boot z -0.55) | -0.0046 vs 0.0000 (-0.0046): does not beat, noise (boot z -0.28) |
| class_prior | direction/auc | 0.4820 vs 0.5000 (-0.0180): does not beat, noise (boot z -1.29) | 0.5025 vs 0.5000 (+0.0025): beats, noise (boot z +0.19) | 0.5013 vs 0.5000 (+0.0013): beats, noise (boot z +0.14) |
| class_prior | direction/brier | 0.2553 vs 0.2532 (-0.0021): does not beat, noise (DM z -1.02) | 0.2560 vs 0.2533 (-0.0027): does not beat, noise (DM z -1.81) | 0.2589 vs 0.2573 (-0.0016): does not beat, noise (DM z -1.43) |
| class_prior | direction/ece_pos | 0.0629 vs 0.0593 (-0.0035): does not beat, noise (boot z -0.21) | 0.0683 vs 0.0603 (-0.0080): does not beat, noise (boot z -1.25) | 0.0853 vs 0.0886 (+0.0033): beats, noise (boot z +0.71) |
| class_prior | direction/acc | 0.4798 vs 0.4824 (-0.0026): does not beat, noise (DM z -0.15) | 0.4848 vs 0.4829 (+0.0019): beats, noise (DM z +0.18) | 0.4799 vs 0.4763 (+0.0035): beats, noise (DM z +0.44) |
| class_prior | direction/bal_acc | 0.4828 vs 0.5000 (-0.0172): does not beat, noise (boot z -1.59) | 0.4964 vs 0.5000 (-0.0036): does not beat, noise (boot z -0.55) | 0.4986 vs 0.5000 (-0.0014): does not beat, noise (boot z -0.28) |
| zero_delta | delta/rmse | 401.50 vs 400.31 (-1.19, -0.30%): does not beat, noise (DM z -1.76) | 566.40 vs 563.88 (-2.51, -0.45%): does not beat, noise (DM z -1.50) | 811.07 vs 807.63 (-3.44, -0.43%): does not beat, noise (DM z -0.99) |
| zero_delta | delta/mae | 272.31 vs 271.46 (-0.86, -0.32%): does not beat, noise (DM z -1.93) | 379.19 vs 377.88 (-1.32, -0.35%): does not beat, noise (DM z -1.08) | 553.95 vs 550.98 (-2.97, -0.54%): does not beat, noise (DM z -1.20) |
| mean_delta | delta/rmse | 401.50 vs 405.60 (+4.10, +1.01%): beats (DM z +2.62) | 566.40 vs 578.56 (+12.17, +2.10%): beats (DM z +3.12) | 811.07 vs 847.03 (+35.97, +4.25%): beats (DM z +3.13) |
| mean_delta | delta/mae | 272.31 vs 277.50 (+5.18, +1.87%): beats (DM z +4.27) | 379.19 vs 393.77 (+14.58, +3.70%): beats (DM z +4.32) | 553.95 vs 595.01 (+41.07, +6.90%): beats (DM z +4.11) |
| const_var | variance/crps | 202.70 vs 206.55 (+3.85, +1.86%): beats (DM z +5.26) | 285.68 vs 294.69 (+9.01, +3.06%): beats (DM z +4.34) | 419.66 vs 444.83 (+25.16, +5.66%): beats (DM z +3.96) |
| const_var | variance/nll | 7.4341 vs 7.4448 (+0.0107): beats, noise (DM z +1.28) | 7.8377 vs 7.8022 (-0.0355): does not beat, noise (DM z -1.13) | 8.1678 vs 8.2104 (+0.0426): beats, noise (DM z +1.74) |
| const_var | variance/pit_ks | 0.0520 vs 0.1064 (+0.0544): beats (boot z +23.68) | 0.0589 vs 0.1404 (+0.0815): beats (boot z +25.55) | 0.0758 vs 0.1810 (+0.1053): beats (boot z +31.89) |
| const_var | variance/corr_var_err2_spearman | 0.0990 vs 0.0000 (+0.0990): beats (boot z +4.89) | -0.0443 vs 0.0000 (-0.0443): does not beat, noise (boot z -1.70) | 0.0897 vs 0.0000 (+0.0897): beats (boot z +3.70) |

## Backtest (costs included)

- n_trades: 1256
- total_return: -0.9627
- sharpe_net: -118.6181
- sharpe_gross: 1.4286
- sortino: -137.7155
- max_drawdown: 0.9627
- hit_rate: 0.0549
- hit_rate_gross: 0.4801
- profit_factor: 0.0379
- avg_hold_bars: 11.3447
- exposure: 0.3298
- turnover: 748.7173
- fees_paid: 7487.1847
- traded_notional: 7487184.6621
- breakeven_cost_bps: 0.2851
- gross_edge_per_trade_bps: -0.1232
- costs_paid: 9733.3401
- gross_pnl: 106.7326
- net_pnl: -9626.6075

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6853, long_above 0.5661, short_below 0.4836, median 0.5276. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.27% | -118.62 | +96.27% | 1256 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.83% .. -95.79%) | -96.30% | -136.69 | | |

The random null enters at the strategy's rate (0.0434 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 56% of its seeds on net return, 100% on net Sharpe and 66% on gross return.
