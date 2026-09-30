# Evaluation report - dev split - run `20260930T081536Z-dce15ed-e1204c70-default__f-3__s0`

n = 46544 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26329 | 29695 | 31930 |
| n_eff of the scored moves (n scored // bars ahead) | 2632 | 1979 | 1596 |
| true up-rate | 0.5030 | 0.5065 | 0.5063 |
| calls up (predicted up-rate) | 0.5931 | 0.5486 | 0.6332 |
| accuracy | 0.5001 | 0.4947 | 0.4972 |
| balanced accuracy | 0.4995 | 0.4941 | 0.4955 |
| precision (up) | 0.5026 | 0.5011 | 0.5028 |
| recall / sensitivity (up) | 0.5926 | 0.5428 | 0.6287 |
| specificity (down) | 0.4065 | 0.4454 | 0.3623 |
| F1 (up) | 0.5439 | 0.5211 | 0.5588 |
| MCC | -0.0009 | -0.0119 | -0.0093 |
| AUC | 0.4970 | 0.4978 | 0.4931 |
| Brier | 0.2509 | 0.2500 | 0.2506 |
| ECE (positive class) | 0.0191 | 0.0092 | 0.0173 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0030 | 0.0065 | 0.0063 |
| TP / FP / TN / FN | 7848 / 7767 / 5319 / 5395 | 8163 / 8128 / 6527 / 6877 | 10165 / 10052 / 5711 / 6002 |
| Gaussian readout: calls up | 0.5198 | 0.4161 | 0.4065 |
| Gaussian readout: MCC | 0.0198 | 0.0102 | 0.0016 |
| Gaussian readout: AUC | 0.5122 | 0.5059 | 0.4979 |
| Gaussian readout: Brier | 0.2499 | 0.2500 | 0.2501 |
| Gaussian readout: ECE | 0.0033 | 0.0078 | 0.0080 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 151.15 | 186.55 | 212.86 |
| RMSE ($), raw heads | 151.79 | 234.41 | 215.73 |
| RMSE ($), zero prediction | 151.14 | 184.66 | 212.80 |
| MAE ($), served | 101.83 | 126.13 | 145.26 |
| MAE ($), raw heads | 102.11 | 129.94 | 146.66 |
| MAE ($), zero prediction | 101.84 | 125.61 | 145.22 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | -0.0206 | -0.0006 |
| skill vs zero, raw heads | -0.0086 | -0.6114 | -0.0278 |
| EV, served | -0.0002 | -0.0207 | -0.0005 |
| EV, raw heads | -0.0087 | -0.6121 | -0.0277 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0130 | 0.0072 | 0.0105 |
| corr, Spearman, raw heads | 0.0133 | 0.0078 | -0.0056 |
| mean predicted ($), served | 0.28 | 0.50 | -0.12 |
| mean predicted ($), raw heads | 0.96 | 2.60 | -0.57 |
| mean realised ($) | 2.58 | 3.89 | 5.19 |
| share predicted up, raw heads | 0.5257 | 0.4004 | 0.3968 |
| share realised up | 0.4954 | 0.4994 | 0.5014 |
| shrink beta (served = beta x raw, fit on cal) | 0.2927 | 0.1915 | 0.2038 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 74.39 | 91.94 | 105.50 |
| CRPSS vs constant variance | 0.0806 | 0.0730 | 0.0772 |
| NLL | 6.2673 | 6.4858 | 6.6359 |
| PIT KS | 0.0271 | 0.0232 | 0.0229 |
| var / err^2 Spearman | 0.4190 | 0.4194 | 0.4196 |
| coverage of the 90% interval | 0.8872 | 0.8836 | 0.8846 |
| width of the 90% interval ($) | 437.52 | 531.75 | 611.56 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0048 | [-0.0265, 0.0139] | NOISE |
| h1 | 0.0068 | [-0.0116, 0.0266] | NOISE |
| h2 | -0.0063 | [-0.0262, 0.0144] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.293 / h1 0.192 / h2 0.204) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7615 | 0.5931 | 0.6235 |
| abs(d h1) <= abs(d h2) | 0.7768 | 0.8032 | 0.6030 |
| full chain h0 <= h1 <= h2 | 0.5920 | 0.4785 | 0.3489 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5583 | 0.4742 | 0.5649 | 0.1611 |
| expected if the two signs were independent | 0.5022 | 0.4883 | 0.4810 | 0.1314 |

- P(up) unanimity (all three horizons call the same side): 0.3166

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0009 vs 0.0074 (-0.0083): does not beat, noise (boot z -0.48) | -0.0119 vs 0.0163 (-0.0282): does not beat, noise (boot z -1.41) | -0.0093 vs 0.0186 (-0.0279): does not beat, noise (boot z -1.46) |
| logreg_lags | direction/auc | 0.4970 vs 0.5121 (-0.0151): does not beat, noise (boot z -1.30) | 0.4978 vs 0.5163 (-0.0185): does not beat, noise (boot z -1.43) | 0.4931 vs 0.5158 (-0.0228): does not beat, noise (boot z -1.90) |
| logreg_lags | direction/brier | 0.2509 vs 0.2498 (-0.0011): does not beat, significantly worse (DM z -2.71) | 0.2500 vs 0.2497 (-0.0003): does not beat, noise (DM z -1.46) | 0.2506 vs 0.2497 (-0.0009): does not beat, significantly worse (DM z -2.70) |
| logreg_lags | direction/ece_pos | 0.0191 vs 0.0045 (-0.0146): does not beat, noise (boot z -1.80) | 0.0092 vs 0.0063 (-0.0029): does not beat, noise (boot z -0.50) | 0.0173 vs 0.0058 (-0.0115): does not beat, noise (boot z -1.23) |
| logreg_lags | direction/acc | 0.5001 vs 0.5035 (-0.0034): does not beat, noise (DM z -0.39) | 0.4947 vs 0.5078 (-0.0131): does not beat, noise (DM z -1.36) | 0.4972 vs 0.5090 (-0.0118): does not beat, noise (DM z -1.19) |
| logreg_lags | direction/bal_acc | 0.4995 vs 0.5037 (-0.0041): does not beat, noise (boot z -0.48) | 0.4941 vs 0.5081 (-0.0141): does not beat, noise (boot z -1.41) | 0.4955 vs 0.5093 (-0.0138): does not beat, noise (boot z -1.46) |
| class_prior | direction/mcc | -0.0009 vs 0.0000 (-0.0009): does not beat, noise (boot z -0.07) | -0.0119 vs 0.0000 (-0.0119): does not beat, noise (boot z -0.82) | -0.0093 vs 0.0000 (-0.0093): does not beat, noise (boot z -0.60) |
| class_prior | direction/auc | 0.4970 vs 0.5000 (-0.0030): does not beat, noise (boot z -0.33) | 0.4978 vs 0.5000 (-0.0022): does not beat, noise (boot z -0.24) | 0.4931 vs 0.5000 (-0.0069): does not beat, noise (boot z -0.68) |
| class_prior | direction/brier | 0.2509 vs 0.2500 (-0.0009): does not beat, significantly worse (DM z -2.42) | 0.2500 vs 0.2500 (-0.0000): does not beat, noise (DM z -0.10) | 0.2506 vs 0.2500 (-0.0006): does not beat, significantly worse (DM z -2.08) |
| class_prior | direction/ece_pos | 0.0191 vs 0.0007 (-0.0184): does not beat, significantly worse (boot z -2.34) | 0.0092 vs 0.0032 (-0.0060): does not beat, noise (boot z -1.13) | 0.0173 vs 0.0018 (-0.0155): does not beat, noise (boot z -1.85) |
| class_prior | direction/acc | 0.5001 vs 0.5030 (-0.0029): does not beat, noise (DM z -0.32) | 0.4947 vs 0.5065 (-0.0118): does not beat, noise (DM z -1.07) | 0.4972 vs 0.5063 (-0.0091): does not beat, noise (DM z -0.88) |
| class_prior | direction/bal_acc | 0.4995 vs 0.5000 (-0.0005): does not beat, noise (boot z -0.07) | 0.4941 vs 0.5000 (-0.0059): does not beat, noise (boot z -0.82) | 0.4955 vs 0.5000 (-0.0045): does not beat, noise (boot z -0.60) |
| zero_delta | delta/rmse | 151.15 vs 151.14 (-0.01, -0.01%): does not beat, noise (DM z -0.10) | 186.55 vs 184.66 (-1.89, -1.02%): does not beat, noise (DM z -1.17) | 212.86 vs 212.80 (-0.06, -0.03%): does not beat, noise (DM z -0.44) |
| zero_delta | delta/mae | 101.83 vs 101.84 (+0.00, +0.00%): beats, noise (DM z +0.04) | 126.13 vs 125.61 (-0.52, -0.41%): does not beat, noise (DM z -1.13) | 145.26 vs 145.22 (-0.05, -0.03%): does not beat, noise (DM z -0.50) |
| mean_delta | delta/rmse | 151.15 vs 151.13 (-0.02, -0.01%): does not beat, noise (DM z -0.22) | 186.55 vs 184.64 (-1.91, -1.03%): does not beat, noise (DM z -1.18) | 212.86 vs 212.77 (-0.09, -0.04%): does not beat, noise (DM z -0.61) |
| mean_delta | delta/mae | 101.83 vs 101.84 (+0.01, +0.01%): beats, noise (DM z +0.15) | 126.13 vs 125.61 (-0.52, -0.41%): does not beat, noise (DM z -1.13) | 145.26 vs 145.21 (-0.05, -0.03%): does not beat, noise (DM z -0.52) |
| const_var | variance/crps | 74.39 vs 80.91 (+6.52, +8.06%): beats (DM z +27.85) | 91.94 vs 99.18 (+7.24, +7.30%): beats (DM z +13.12) | 105.50 vs 114.33 (+8.82, +7.72%): beats (DM z +19.81) |
| const_var | variance/nll | 6.2673 vs 6.4734 (+0.2061): beats (DM z +12.66) | 6.4858 vs 6.6734 (+0.1875): beats (DM z +9.04) | 6.6359 vs 6.8153 (+0.1794): beats (DM z +7.14) |
| const_var | variance/pit_ks | 0.0271 vs 0.1172 (+0.0901): beats (boot z +21.68) | 0.0232 vs 0.1152 (+0.0920): beats (boot z +18.35) | 0.0229 vs 0.1158 (+0.0929): beats (boot z +18.46) |
| const_var | variance/corr_var_err2_spearman | 0.4190 vs 0.0000 (+0.4190): beats (boot z +32.01) | 0.4194 vs 0.0000 (+0.4194): beats (boot z +29.93) | 0.4196 vs 0.0000 (+0.4196): beats (boot z +28.71) |

## Backtest (costs included)

- n_trades: 1825
- total_return: -0.9911
- sharpe_net: -160.6205
- sharpe_gross: 3.4024
- sortino: -172.6605
- max_drawdown: 0.9911
- hit_rate: 0.0373
- hit_rate_gross: 0.5140
- profit_factor: 0.0103
- avg_hold_bars: 10.9397
- exposure: 0.4289
- turnover: 775.3472
- fees_paid: 7753.6362
- traded_notional: 7753636.1805
- breakeven_cost_bps: 0.4349
- gross_edge_per_trade_bps: 0.1654
- costs_paid: 10079.7270
- gross_pnl: 168.6087
- net_pnl: -9911.1184

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -3 (TimeSeriesSplit fold 12, 13 usable folds); this report scores the fold's out-of-sample block: 46544 sequences, 2025-06-25T00:26:00 .. 2025-07-27T08:09:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4664, long_above 0.5204, short_below 0.4883, median 0.5044. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.11% | -160.62 | +99.11% | 1825 |
| buy and hold | +11.00% | +4.11 | +6.87% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.24% .. -98.95%) | -99.12% | -167.11 | | |

The random null enters at the strategy's rate (0.0687 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 59% of its seeds on net return, 98% on net Sharpe and 80% on gross return.
