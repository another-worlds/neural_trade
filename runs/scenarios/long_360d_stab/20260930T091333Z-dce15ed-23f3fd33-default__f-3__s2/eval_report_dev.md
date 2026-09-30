# Evaluation report - dev split - run `20260930T091333Z-dce15ed-23f3fd33-default__f-3__s2`

n = 46544 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26329 | 29695 | 31930 |
| n_eff of the scored moves (n scored // bars ahead) | 2632 | 1979 | 1596 |
| true up-rate | 0.5030 | 0.5065 | 0.5063 |
| calls up (predicted up-rate) | 0.6158 | 0.7362 | 0.4861 |
| accuracy | 0.5055 | 0.5143 | 0.4975 |
| balanced accuracy | 0.5048 | 0.5113 | 0.4977 |
| precision (up) | 0.5069 | 0.5141 | 0.5039 |
| recall / sensitivity (up) | 0.6206 | 0.7473 | 0.4838 |
| specificity (down) | 0.3890 | 0.2753 | 0.5116 |
| F1 (up) | 0.5580 | 0.6092 | 0.4936 |
| MCC | 0.0099 | 0.0256 | -0.0047 |
| AUC | 0.5210 | 0.5187 | 0.4976 |
| Brier | 0.2496 | 0.2498 | 0.2504 |
| ECE (positive class) | 0.0014 | 0.0019 | 0.0187 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0030 | 0.0065 | 0.0063 |
| TP / FP / TN / FN | 8218 / 7995 / 5091 / 5025 | 11239 / 10621 / 4034 / 3801 | 7821 / 7699 / 8064 / 8346 |
| Gaussian readout: calls up | 0.4696 | 0.5094 | 0.4362 |
| Gaussian readout: MCC | -0.0098 | 0.0024 | -0.0096 |
| Gaussian readout: AUC | 0.4950 | 0.5015 | 0.4925 |
| Gaussian readout: Brier | 0.2502 | 0.2500 | 0.2502 |
| Gaussian readout: ECE | 0.0122 | 0.0064 | 0.0131 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 151.10 | 184.63 | 212.81 |
| RMSE ($), raw heads | 151.88 | 187.04 | 215.57 |
| RMSE ($), zero prediction | 151.14 | 184.66 | 212.80 |
| MAE ($), served | 101.84 | 125.61 | 145.24 |
| MAE ($), raw heads | 102.31 | 126.81 | 146.95 |
| MAE ($), zero prediction | 101.84 | 125.61 | 145.22 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0005 | 0.0003 | -0.0002 |
| skill vs zero, raw heads | -0.0098 | -0.0260 | -0.0263 |
| EV, served | 0.0005 | 0.0002 | -0.0001 |
| EV, raw heads | -0.0098 | -0.0262 | -0.0257 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0229 | 0.0176 | 0.0111 |
| corr, Spearman, raw heads | -0.0042 | -0.0041 | -0.0113 |
| mean predicted ($), served | -0.01 | 0.19 | -0.32 |
| mean predicted ($), raw heads | -0.05 | 1.30 | -2.13 |
| mean realised ($) | 2.58 | 3.89 | 5.19 |
| share predicted up, raw heads | 0.4511 | 0.5052 | 0.4312 |
| share realised up | 0.4954 | 0.4994 | 0.5014 |
| shrink beta (served = beta x raw, fit on cal) | 0.2232 | 0.1442 | 0.1523 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 74.40 | 91.41 | 105.53 |
| CRPSS vs constant variance | 0.0805 | 0.0783 | 0.0769 |
| NLL | 6.2757 | 6.4822 | 6.6365 |
| PIT KS | 0.0251 | 0.0219 | 0.0204 |
| var / err^2 Spearman | 0.4168 | 0.4168 | 0.4187 |
| coverage of the 90% interval | 0.8872 | 0.8835 | 0.8848 |
| width of the 90% interval ($) | 437.77 | 532.17 | 612.17 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0425 | [0.0201, 0.0652] | WORKS |
| h1 | 0.0114 | [-0.0088, 0.0323] | NOISE |
| h2 | 0.0009 | [-0.0203, 0.0199] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.223 / h1 0.144 / h2 0.152) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8351 | 0.5697 | 0.6235 |
| abs(d h1) <= abs(d h2) | 0.8010 | 0.8200 | 0.6030 |
| full chain h0 <= h1 <= h2 | 0.6606 | 0.4419 | 0.3489 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.3756 | 0.5064 | 0.5664 | 0.1281 |
| expected if the two signs were independent | 0.4846 | 0.5021 | 0.5008 | 0.1172 |

- P(up) unanimity (all three horizons call the same side): 0.2385

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0099 vs 0.0074 (+0.0025): beats, noise (boot z +0.17) | 0.0256 vs 0.0163 (+0.0093): beats, noise (boot z +0.53) | -0.0047 vs 0.0186 (-0.0233): does not beat, noise (boot z -1.24) |
| logreg_lags | direction/auc | 0.5210 vs 0.5121 (+0.0089): beats, noise (boot z +1.07) | 0.5187 vs 0.5163 (+0.0024): beats, noise (boot z +0.24) | 0.4976 vs 0.5158 (-0.0182): does not beat, noise (boot z -1.53) |
| logreg_lags | direction/brier | 0.2496 vs 0.2498 (+0.0002): beats, noise (DM z +1.37) | 0.2498 vs 0.2497 (-0.0002): does not beat, noise (DM z -0.61) | 0.2504 vs 0.2497 (-0.0008): does not beat, significantly worse (DM z -2.08) |
| logreg_lags | direction/ece_pos | 0.0014 vs 0.0045 (+0.0031): beats, noise (boot z +0.70) | 0.0019 vs 0.0063 (+0.0044): beats, noise (boot z +0.68) | 0.0187 vs 0.0058 (-0.0129): does not beat, noise (boot z -1.71) |
| logreg_lags | direction/acc | 0.5055 vs 0.5035 (+0.0020): beats, noise (DM z +0.28) | 0.5143 vs 0.5078 (+0.0065): beats, noise (DM z +0.69) | 0.4975 vs 0.5090 (-0.0115): does not beat, noise (DM z -1.23) |
| logreg_lags | direction/bal_acc | 0.5048 vs 0.5037 (+0.0011): beats, noise (boot z +0.15) | 0.5113 vs 0.5081 (+0.0031): beats, noise (boot z +0.37) | 0.4977 vs 0.5093 (-0.0116): does not beat, noise (boot z -1.24) |
| class_prior | direction/mcc | 0.0099 vs 0.0000 (+0.0099): beats, noise (boot z +0.66) | 0.0256 vs 0.0000 (+0.0256): beats (boot z +2.03) | -0.0047 vs 0.0000 (-0.0047): does not beat, noise (boot z -0.32) |
| class_prior | direction/auc | 0.5210 vs 0.5000 (+0.0210): beats (boot z +2.22) | 0.5187 vs 0.5000 (+0.0187): beats (boot z +2.31) | 0.4976 vs 0.5000 (-0.0024): does not beat, noise (boot z -0.25) |
| class_prior | direction/brier | 0.2496 vs 0.2500 (+0.0004): beats (DM z +2.65) | 0.2498 vs 0.2500 (+0.0001): beats, noise (DM z +0.50) | 0.2504 vs 0.2500 (-0.0005): does not beat, noise (DM z -1.27) |
| class_prior | direction/ece_pos | 0.0014 vs 0.0007 (-0.0007): does not beat, noise (boot z -0.18) | 0.0019 vs 0.0032 (+0.0012): beats, noise (boot z +0.26) | 0.0187 vs 0.0018 (-0.0169): does not beat, significantly worse (boot z -2.13) |
| class_prior | direction/acc | 0.5055 vs 0.5030 (+0.0025): beats, noise (DM z +0.27) | 0.5143 vs 0.5065 (+0.0078): beats, noise (DM z +1.09) | 0.4975 vs 0.5063 (-0.0088): does not beat, noise (DM z -0.68) |
| class_prior | direction/bal_acc | 0.5048 vs 0.5000 (+0.0048): beats, noise (boot z +0.66) | 0.5113 vs 0.5000 (+0.0113): beats (boot z +2.04) | 0.4977 vs 0.5000 (-0.0023): does not beat, noise (boot z -0.32) |
| zero_delta | delta/rmse | 151.10 vs 151.14 (+0.04, +0.02%): beats, noise (DM z +0.46) | 184.63 vs 184.66 (+0.03, +0.01%): beats, noise (DM z +0.25) | 212.81 vs 212.80 (-0.02, -0.01%): does not beat, noise (DM z -0.16) |
| zero_delta | delta/mae | 101.84 vs 101.84 (-0.00, -0.00%): does not beat, noise (DM z -0.11) | 125.61 vs 125.61 (+0.00, +0.00%): beats, noise (DM z +0.00) | 145.24 vs 145.22 (-0.02, -0.02%): does not beat, noise (DM z -0.33) |
| mean_delta | delta/rmse | 151.10 vs 151.13 (+0.03, +0.02%): beats, noise (DM z +0.36) | 184.63 vs 184.64 (+0.01, +0.01%): beats, noise (DM z +0.09) | 212.81 vs 212.77 (-0.04, -0.02%): does not beat, noise (DM z -0.36) |
| mean_delta | delta/mae | 101.84 vs 101.84 (+0.00, +0.00%): beats, noise (DM z +0.02) | 125.61 vs 125.61 (+0.00, +0.00%): beats, noise (DM z +0.02) | 145.24 vs 145.21 (-0.02, -0.02%): does not beat, noise (DM z -0.35) |
| const_var | variance/crps | 74.40 vs 80.91 (+6.51, +8.05%): beats (DM z +28.32) | 91.41 vs 99.18 (+7.77, +7.83%): beats (DM z +23.17) | 105.53 vs 114.33 (+8.79, +7.69%): beats (DM z +20.36) |
| const_var | variance/nll | 6.2757 vs 6.4734 (+0.1977): beats (DM z +11.86) | 6.4822 vs 6.6734 (+0.1912): beats (DM z +9.83) | 6.6365 vs 6.8153 (+0.1788): beats (DM z +7.47) |
| const_var | variance/pit_ks | 0.0251 vs 0.1172 (+0.0921): beats (boot z +21.20) | 0.0219 vs 0.1152 (+0.0933): beats (boot z +19.04) | 0.0204 vs 0.1158 (+0.0953): beats (boot z +18.73) |
| const_var | variance/corr_var_err2_spearman | 0.4168 vs 0.0000 (+0.4168): beats (boot z +31.80) | 0.4168 vs 0.0000 (+0.4168): beats (boot z +29.61) | 0.4187 vs 0.0000 (+0.4187): beats (boot z +28.93) |

## Backtest (costs included)

- n_trades: 1398
- total_return: -0.9729
- sharpe_net: -128.6505
- sharpe_gross: 1.5902
- sortino: -142.1824
- max_drawdown: 0.9730
- hit_rate: 0.0486
- hit_rate_gross: 0.5393
- profit_factor: 0.0127
- avg_hold_bars: 11.6102
- exposure: 0.3487
- turnover: 755.7231
- fees_paid: 7557.4386
- traded_notional: 7557438.5766
- breakeven_cost_bps: 0.2528
- gross_edge_per_trade_bps: 0.2346
- costs_paid: 9824.6701
- gross_pnl: 95.5211
- net_pnl: -9729.1490

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -3 (TimeSeriesSplit fold 12, 13 usable folds); this report scores the fold's out-of-sample block: 46544 sequences, 2025-06-25T00:26:00 .. 2025-07-27T08:09:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4654, long_above 0.5124, short_below 0.4962, median 0.5032. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.29% | -128.65 | +97.30% | 1398 |
| buy and hold | +11.00% | +4.11 | +6.87% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.54% .. -96.77%) | -97.23% | -142.35 | | |

The random null enters at the strategy's rate (0.0461 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 42% of its seeds on net return, 100% on net Sharpe and 76% on gross return.
