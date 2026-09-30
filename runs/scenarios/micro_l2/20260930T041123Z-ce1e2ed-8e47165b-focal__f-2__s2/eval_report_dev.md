# Evaluation report - dev split - run `20260930T041123Z-ce1e2ed-8e47165b-focal__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.6692 | 0.7880 | 0.7922 |
| accuracy | 0.4821 | 0.4922 | 0.4795 |
| balanced accuracy | 0.4880 | 0.5020 | 0.4933 |
| precision (up) | 0.4735 | 0.4842 | 0.4721 |
| recall / sensitivity (up) | 0.6568 | 0.7901 | 0.7851 |
| specificity (down) | 0.3192 | 0.2139 | 0.2014 |
| F1 (up) | 0.5503 | 0.6005 | 0.5896 |
| MCC | -0.0255 | 0.0050 | -0.0165 |
| AUC | 0.4902 | 0.5154 | 0.4936 |
| Brier | 0.2552 | 0.2564 | 0.2598 |
| ECE (positive class) | 0.0605 | 0.0677 | 0.0812 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 11499 / 12788 / 5996 / 6009 | 14426 / 15366 / 4182 / 3832 | 14751 / 16495 / 4161 / 4037 |
| Gaussian readout: calls up | 0.7904 | 0.8388 | 0.7792 |
| Gaussian readout: MCC | -0.0136 | 0.0048 | 0.0126 |
| Gaussian readout: AUC | 0.5026 | 0.5085 | 0.5123 |
| Gaussian readout: Brier | 0.2504 | 0.2510 | 0.2519 |
| Gaussian readout: ECE | 0.0258 | 0.0356 | 0.0455 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.69 | 565.81 | 812.21 |
| RMSE ($), raw heads | 417.49 | 581.69 | 838.70 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.78 | 379.14 | 554.97 |
| MAE ($), raw heads | 286.75 | 392.88 | 582.45 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0019 | -0.0068 | -0.0114 |
| skill vs zero, raw heads | -0.0877 | -0.0642 | -0.0784 |
| EV, served | -0.0008 | -0.0021 | -0.0034 |
| EV, raw heads | -0.0453 | -0.0249 | -0.0352 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0128 | -0.0118 | 0.0013 |
| corr, Spearman, raw heads | -0.0122 | -0.0044 | -0.0001 |
| mean predicted ($), served | 6.41 | 22.48 | 41.13 |
| mean predicted ($), raw heads | 72.18 | 91.80 | 130.61 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.7789 | 0.8304 | 0.7771 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0888 | 0.2449 | 0.3149 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 201.88 | 285.71 | 419.49 |
| CRPSS vs constant variance | 0.0226 | 0.0305 | 0.0570 |
| NLL | 7.3916 | 7.7630 | 8.1798 |
| PIT KS | 0.0564 | 0.0759 | 0.0706 |
| var / err^2 Spearman | 0.1808 | 0.1210 | 0.1363 |
| coverage of the 90% interval | 0.9020 | 0.9011 | 0.8619 |
| width of the 90% interval ($) | 1210.25 | 1756.63 | 2391.73 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0053 | [-0.0306, 0.0190] | NOISE |
| h1 | 0.0038 | [-0.0261, 0.0352] | NOISE |
| h2 | -0.0031 | [-0.0363, 0.0297] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.089 / h1 0.245 / h2 0.315) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6044 | 0.9164 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.9040 | 0.9407 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.5458 | 0.8606 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5966 | 0.8213 | 0.7219 | 0.4240 |
| expected if the two signs were independent | 0.5974 | 0.6853 | 0.6613 | 0.3388 |

- P(up) unanimity (all three horizons call the same side): 0.4605

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0255 vs 0.0025 (-0.0280): does not beat, noise (boot z -1.02) | 0.0050 vs 0.0205 (-0.0156): does not beat, noise (boot z -0.56) | -0.0165 vs -0.0138 (-0.0027): does not beat, noise (boot z -0.10) |
| logreg_lags | direction/auc | 0.4902 vs 0.5320 (-0.0418): does not beat, noise (boot z -1.93) | 0.5154 vs 0.5251 (-0.0097): does not beat, noise (boot z -0.80) | 0.4936 vs 0.5095 (-0.0159): does not beat, noise (boot z -0.75) |
| logreg_lags | direction/brier | 0.2552 vs 0.2536 (-0.0016): does not beat, noise (DM z -0.97) | 0.2564 vs 0.2575 (+0.0010): beats, noise (DM z +0.63) | 0.2598 vs 0.2702 (+0.0104): beats (DM z +2.17) |
| logreg_lags | direction/ece_pos | 0.0605 vs 0.0663 (+0.0058): beats, noise (boot z +0.48) | 0.0677 vs 0.0812 (+0.0135): beats, noise (boot z +1.93) | 0.0812 vs 0.1230 (+0.0418): beats (boot z +4.76) |
| logreg_lags | direction/acc | 0.4821 vs 0.4836 (-0.0016): does not beat, noise (DM z -0.12) | 0.4922 vs 0.4911 (+0.0011): beats, noise (DM z +0.08) | 0.4795 vs 0.4756 (+0.0039): beats, noise (DM z +0.28) |
| logreg_lags | direction/bal_acc | 0.4880 vs 0.5004 (-0.0124): does not beat, noise (boot z -1.43) | 0.5020 vs 0.5055 (-0.0035): does not beat, noise (boot z -0.36) | 0.4933 vs 0.4971 (-0.0039): does not beat, noise (boot z -0.45) |
| class_prior | direction/mcc | -0.0255 vs 0.0000 (-0.0255): does not beat, noise (boot z -1.55) | 0.0050 vs 0.0000 (+0.0050): beats, noise (boot z +0.24) | -0.0165 vs 0.0000 (-0.0165): does not beat, noise (boot z -0.98) |
| class_prior | direction/auc | 0.4902 vs 0.5000 (-0.0098): does not beat, noise (boot z -0.90) | 0.5154 vs 0.5000 (+0.0154): beats, noise (boot z +1.18) | 0.4936 vs 0.5000 (-0.0064): does not beat, noise (boot z -0.51) |
| class_prior | direction/brier | 0.2552 vs 0.2532 (-0.0020): does not beat, noise (DM z -1.47) | 0.2564 vs 0.2533 (-0.0031): does not beat, noise (DM z -1.63) | 0.2598 vs 0.2573 (-0.0025): does not beat, noise (DM z -1.41) |
| class_prior | direction/ece_pos | 0.0605 vs 0.0593 (-0.0012): does not beat, noise (boot z -0.10) | 0.0677 vs 0.0603 (-0.0073): does not beat, noise (boot z -1.01) | 0.0812 vs 0.0886 (+0.0074): beats, noise (boot z +0.86) |
| class_prior | direction/acc | 0.4821 vs 0.4824 (-0.0004): does not beat, noise (DM z -0.03) | 0.4922 vs 0.4829 (+0.0093): beats, noise (DM z +0.63) | 0.4795 vs 0.4763 (+0.0031): beats, noise (DM z +0.21) |
| class_prior | direction/bal_acc | 0.4880 vs 0.5000 (-0.0120): does not beat, noise (boot z -1.55) | 0.5020 vs 0.5000 (+0.0020): beats, noise (boot z +0.24) | 0.4933 vs 0.5000 (-0.0067): does not beat, noise (boot z -0.97) |
| zero_delta | delta/rmse | 400.69 vs 400.31 (-0.38, -0.10%): does not beat, noise (DM z -1.20) | 565.81 vs 563.88 (-1.93, -0.34%): does not beat, noise (DM z -1.33) | 812.21 vs 807.63 (-4.58, -0.57%): does not beat, noise (DM z -1.22) |
| zero_delta | delta/mae | 271.78 vs 271.46 (-0.32, -0.12%): does not beat, noise (DM z -1.36) | 379.14 vs 377.88 (-1.26, -0.33%): does not beat, noise (DM z -1.20) | 554.97 vs 550.98 (-3.99, -0.72%): does not beat, noise (DM z -1.43) |
| mean_delta | delta/rmse | 400.69 vs 405.60 (+4.91, +1.21%): beats (DM z +3.09) | 565.81 vs 578.56 (+12.75, +2.20%): beats (DM z +3.20) | 812.21 vs 847.03 (+34.83, +4.11%): beats (DM z +3.06) |
| mean_delta | delta/mae | 271.78 vs 277.50 (+5.72, +2.06%): beats (DM z +4.17) | 379.14 vs 393.77 (+14.63, +3.72%): beats (DM z +4.21) | 554.97 vs 595.01 (+40.04, +6.73%): beats (DM z +3.98) |
| const_var | variance/crps | 201.88 vs 206.55 (+4.67, +2.26%): beats (DM z +5.23) | 285.71 vs 294.69 (+8.98, +3.05%): beats (DM z +4.08) | 419.49 vs 444.83 (+25.34, +5.70%): beats (DM z +3.88) |
| const_var | variance/nll | 7.3916 vs 7.4448 (+0.0533): beats (DM z +4.20) | 7.7630 vs 7.8022 (+0.0392): beats (DM z +3.26) | 8.1798 vs 8.2104 (+0.0307): beats, noise (DM z +1.11) |
| const_var | variance/pit_ks | 0.0564 vs 0.1064 (+0.0500): beats (boot z +10.81) | 0.0759 vs 0.1404 (+0.0645): beats (boot z +15.20) | 0.0706 vs 0.1810 (+0.1104): beats (boot z +23.22) |
| const_var | variance/corr_var_err2_spearman | 0.1808 vs 0.0000 (+0.1808): beats (boot z +9.19) | 0.1210 vs 0.0000 (+0.1210): beats (boot z +4.95) | 0.1363 vs 0.0000 (+0.1363): beats (boot z +5.26) |

## Backtest (costs included)

- n_trades: 1295
- total_return: -0.9652
- sharpe_net: -122.4984
- sharpe_gross: 2.1765
- sortino: -138.3839
- max_drawdown: 0.9652
- hit_rate: 0.0641
- hit_rate_gross: 0.5058
- profit_factor: 0.0279
- avg_hold_bars: 11.6340
- exposure: 0.3488
- turnover: 755.2343
- fees_paid: 7552.3666
- traded_notional: 7552366.6249
- breakeven_cost_bps: 0.4406
- gross_edge_per_trade_bps: 0.1272
- costs_paid: 9818.0766
- gross_pnl: 166.3740
- net_pnl: -9651.7027

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8511, long_above 0.5770, short_below 0.4874, median 0.5232. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.52% | -122.50 | +96.52% | 1295 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.91% .. -95.87%) | -96.41% | -134.83 | | |

The random null enters at the strategy's rate (0.0460 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 35% of its seeds on net return, 100% on net Sharpe and 82% on gross return.
