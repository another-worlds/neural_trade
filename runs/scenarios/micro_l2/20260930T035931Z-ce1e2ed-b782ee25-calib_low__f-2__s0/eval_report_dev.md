# Evaluation report - dev split - run `20260930T035931Z-ce1e2ed-b782ee25-calib_low__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.7345 | 0.7619 | 0.7903 |
| accuracy | 0.4814 | 0.4927 | 0.4833 |
| balanced accuracy | 0.4896 | 0.5017 | 0.4970 |
| precision (up) | 0.4754 | 0.4840 | 0.4744 |
| recall / sensitivity (up) | 0.7238 | 0.7636 | 0.7872 |
| specificity (down) | 0.2555 | 0.2397 | 0.2069 |
| F1 (up) | 0.5738 | 0.5925 | 0.5921 |
| MCC | -0.0235 | 0.0039 | -0.0073 |
| AUC | 0.4962 | 0.4996 | 0.5062 |
| Brier | 0.2583 | 0.2536 | 0.2592 |
| ECE (positive class) | 0.0826 | 0.0479 | 0.0809 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 12672 / 13985 / 4799 / 4836 | 13942 / 14862 / 4686 / 4316 | 14790 / 16383 / 4273 / 3998 |
| Gaussian readout: calls up | 0.6191 | 0.5959 | 0.4941 |
| Gaussian readout: MCC | -0.0270 | 0.0030 | 0.0054 |
| Gaussian readout: AUC | 0.4900 | 0.4972 | 0.5016 |
| Gaussian readout: Brier | 0.2524 | 0.2512 | 0.2526 |
| Gaussian readout: ECE | 0.0461 | 0.0272 | 0.0351 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 402.49 | 566.02 | 813.24 |
| RMSE ($), raw heads | 410.74 | 581.73 | 838.96 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 273.13 | 379.53 | 556.32 |
| MAE ($), raw heads | 280.04 | 395.57 | 582.54 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0109 | -0.0076 | -0.0140 |
| skill vs zero, raw heads | -0.0528 | -0.0643 | -0.0791 |
| EV, served | -0.0079 | -0.0053 | -0.0101 |
| EV, raw heads | -0.0390 | -0.0497 | -0.0652 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0377 | -0.0246 | -0.0097 |
| corr, Spearman, raw heads | -0.0350 | -0.0198 | -0.0080 |
| mean predicted ($), served | 13.53 | 12.93 | 23.10 |
| mean predicted ($), raw heads | 37.40 | 49.70 | 62.10 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.6069 | 0.5916 | 0.4907 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.3617 | 0.2602 | 0.3720 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 203.46 | 285.51 | 420.76 |
| CRPSS vs constant variance | 0.0149 | 0.0312 | 0.0541 |
| NLL | 7.4378 | 7.7886 | 8.1606 |
| PIT KS | 0.0515 | 0.0567 | 0.0658 |
| var / err^2 Spearman | 0.0358 | 0.0944 | 0.1138 |
| coverage of the 90% interval | 0.9002 | 0.9027 | 0.8599 |
| width of the 90% interval ($) | 1215.01 | 1768.23 | 2366.44 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0067 | [-0.0229, 0.0366] | NOISE |
| h1 | -0.0206 | [-0.0508, 0.0115] | NOISE |
| h2 | 0.0143 | [-0.0201, 0.0512] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.362 / h1 0.260 / h2 0.372) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8279 | 0.6869 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.7960 | 0.8708 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.6333 | 0.5719 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5735 | 0.7050 | 0.6113 | 0.3302 |
| expected if the two signs were independent | 0.5480 | 0.5487 | 0.4946 | 0.2369 |

- P(up) unanimity (all three horizons call the same side): 0.4848

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0235 vs 0.0025 (-0.0260): does not beat, noise (boot z -0.89) | 0.0039 vs 0.0205 (-0.0166): does not beat, noise (boot z -0.59) | -0.0073 vs -0.0138 (+0.0065): beats, noise (boot z +0.23) |
| logreg_lags | direction/auc | 0.4962 vs 0.5320 (-0.0358): does not beat, noise (boot z -1.60) | 0.4996 vs 0.5251 (-0.0255): does not beat, noise (boot z -1.24) | 0.5062 vs 0.5095 (-0.0032): does not beat, noise (boot z -0.29) |
| logreg_lags | direction/brier | 0.2583 vs 0.2536 (-0.0047): does not beat, significantly worse (DM z -2.21) | 0.2536 vs 0.2575 (+0.0038): beats, noise (DM z +1.53) | 0.2592 vs 0.2702 (+0.0110): beats (DM z +3.08) |
| logreg_lags | direction/ece_pos | 0.0826 vs 0.0663 (-0.0164): does not beat, noise (boot z -1.53) | 0.0479 vs 0.0812 (+0.0333): beats (boot z +4.98) | 0.0809 vs 0.1230 (+0.0421): beats (boot z +5.11) |
| logreg_lags | direction/acc | 0.4814 vs 0.4836 (-0.0022): does not beat, noise (DM z -0.18) | 0.4927 vs 0.4911 (+0.0016): beats, noise (DM z +0.12) | 0.4833 vs 0.4756 (+0.0077): beats, noise (DM z +0.44) |
| logreg_lags | direction/bal_acc | 0.4896 vs 0.5004 (-0.0107): does not beat, noise (boot z -1.16) | 0.5017 vs 0.5055 (-0.0038): does not beat, noise (boot z -0.41) | 0.4970 vs 0.4971 (-0.0001): does not beat, noise (boot z -0.01) |
| class_prior | direction/mcc | -0.0235 vs 0.0000 (-0.0235): does not beat, noise (boot z -1.23) | 0.0039 vs 0.0000 (+0.0039): beats, noise (boot z +0.26) | -0.0073 vs 0.0000 (-0.0073): does not beat, noise (boot z -0.31) |
| class_prior | direction/auc | 0.4962 vs 0.5000 (-0.0038): does not beat, noise (boot z -0.29) | 0.4996 vs 0.5000 (-0.0004): does not beat, noise (boot z -0.04) | 0.5062 vs 0.5000 (+0.0062): beats, noise (boot z +0.39) |
| class_prior | direction/brier | 0.2583 vs 0.2532 (-0.0051): does not beat, significantly worse (DM z -2.98) | 0.2536 vs 0.2533 (-0.0003): does not beat, noise (DM z -0.26) | 0.2592 vs 0.2573 (-0.0019): does not beat, noise (DM z -0.77) |
| class_prior | direction/ece_pos | 0.0826 vs 0.0593 (-0.0233): does not beat, significantly worse (boot z -2.19) | 0.0479 vs 0.0603 (+0.0124): beats, noise (boot z +1.92) | 0.0809 vs 0.0886 (+0.0077): beats, noise (boot z +0.90) |
| class_prior | direction/acc | 0.4814 vs 0.4824 (-0.0010): does not beat, noise (DM z -0.08) | 0.4927 vs 0.4829 (+0.0098): beats, noise (DM z +0.74) | 0.4833 vs 0.4763 (+0.0070): beats, noise (DM z +0.37) |
| class_prior | direction/bal_acc | 0.4896 vs 0.5000 (-0.0104): does not beat, noise (boot z -1.23) | 0.5017 vs 0.5000 (+0.0017): beats, noise (boot z +0.26) | 0.4970 vs 0.5000 (-0.0030): does not beat, noise (boot z -0.31) |
| zero_delta | delta/rmse | 402.49 vs 400.31 (-2.18, -0.55%): does not beat, significantly worse (DM z -2.77) | 566.02 vs 563.88 (-2.13, -0.38%): does not beat, significantly worse (DM z -2.38) | 813.24 vs 807.63 (-5.62, -0.70%): does not beat, significantly worse (DM z -2.07) |
| zero_delta | delta/mae | 273.13 vs 271.46 (-1.68, -0.62%): does not beat, significantly worse (DM z -2.80) | 379.53 vs 377.88 (-1.66, -0.44%): does not beat, significantly worse (DM z -2.05) | 556.32 vs 550.98 (-5.34, -0.97%): does not beat, significantly worse (DM z -2.26) |
| mean_delta | delta/rmse | 402.49 vs 405.60 (+3.11, +0.77%): beats (DM z +2.18) | 566.02 vs 578.56 (+12.55, +2.17%): beats (DM z +2.72) | 813.24 vs 847.03 (+33.79, +3.99%): beats (DM z +2.61) |
| mean_delta | delta/mae | 273.13 vs 277.50 (+4.37, +1.57%): beats (DM z +3.33) | 379.53 vs 393.77 (+14.23, +3.61%): beats (DM z +3.65) | 556.32 vs 595.01 (+38.70, +6.50%): beats (DM z +3.48) |
| const_var | variance/crps | 203.46 vs 206.55 (+3.08, +1.49%): beats (DM z +3.78) | 285.51 vs 294.69 (+9.18, +3.12%): beats (DM z +3.74) | 420.76 vs 444.83 (+24.06, +5.41%): beats (DM z +3.39) |
| const_var | variance/nll | 7.4378 vs 7.4448 (+0.0070): beats, noise (DM z +0.69) | 7.7886 vs 7.8022 (+0.0136): beats, noise (DM z +0.90) | 8.1606 vs 8.2104 (+0.0499): beats (DM z +2.12) |
| const_var | variance/pit_ks | 0.0515 vs 0.1064 (+0.0549): beats (boot z +19.69) | 0.0567 vs 0.1404 (+0.0837): beats (boot z +14.15) | 0.0658 vs 0.1810 (+0.1152): beats (boot z +21.60) |
| const_var | variance/corr_var_err2_spearman | 0.0358 vs 0.0000 (+0.0358): beats, noise (boot z +1.59) | 0.0944 vs 0.0000 (+0.0944): beats (boot z +3.91) | 0.1138 vs 0.0000 (+0.1138): beats (boot z +5.11) |

## Backtest (costs included)

- n_trades: 1391
- total_return: -0.9751
- sharpe_net: -131.9475
- sharpe_gross: -2.3371
- sortino: -148.0003
- max_drawdown: 0.9751
- hit_rate: 0.0597
- hit_rate_gross: 0.4385
- profit_factor: 0.0345
- avg_hold_bars: 11.8792
- exposure: 0.3825
- turnover: 736.9615
- fees_paid: 7370.0861
- traded_notional: 7370086.0800
- breakeven_cost_bps: -0.4615
- gross_edge_per_trade_bps: -0.5008
- costs_paid: 9581.1119
- gross_pnl: -170.0597
- net_pnl: -9751.1716

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7606, long_above 0.5902, short_below 0.4905, median 0.5337. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.51% | -131.95 | +97.51% | 1391 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.62% .. -96.87%) | -97.26% | -140.49 | | |

The random null enters at the strategy's rate (0.0521 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 16% of its seeds on net return, 100% on net Sharpe and 27% on gross return.
