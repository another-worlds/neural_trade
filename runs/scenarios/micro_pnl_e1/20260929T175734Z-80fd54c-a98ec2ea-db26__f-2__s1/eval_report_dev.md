# Evaluation report - dev split - run `20260929T175734Z-80fd54c-a98ec2ea-db26__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 26 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 26 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 13621 | 19285 | 24800 |
| n_eff of the scored moves (n scored // bars ahead) | 227 | 160 | 103 |
| true up-rate | 0.4907 | 0.4802 | 0.4752 |
| calls up (predicted up-rate) | 0.7565 | 0.9599 | 0.9799 |
| accuracy | 0.4865 | 0.4829 | 0.4724 |
| balanced accuracy | 0.4913 | 0.5011 | 0.4962 |
| precision (up) | 0.4850 | 0.4808 | 0.4732 |
| recall / sensitivity (up) | 0.7476 | 0.9610 | 0.9759 |
| specificity (down) | 0.2350 | 0.0411 | 0.0165 |
| F1 (up) | 0.5883 | 0.6409 | 0.6374 |
| MCC | -0.0203 | 0.0054 | -0.0270 |
| AUC | 0.4702 | 0.5103 | 0.4955 |
| Brier | 0.2601 | 0.2598 | 0.2654 |
| ECE (positive class) | 0.0777 | 0.0969 | 0.1183 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0093 | 0.0198 | 0.0248 |
| TP / FP / TN / FN | 4997 / 5307 / 1630 / 1687 | 8900 / 9612 / 412 / 361 | 11500 / 12801 / 215 / 284 |
| Gaussian readout: calls up | 0.9506 | 0.8868 | 0.8400 |
| Gaussian readout: MCC | -0.0235 | -0.0309 | 0.0167 |
| Gaussian readout: AUC | 0.4962 | 0.4982 | 0.4964 |
| Gaussian readout: Brier | 0.2532 | 0.2527 | 0.2537 |
| Gaussian readout: ECE | 0.0532 | 0.0555 | 0.0582 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.49 | 565.86 | 811.46 |
| RMSE ($), raw heads | 418.51 | 587.50 | 846.69 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.38 | 379.21 | 554.57 |
| MAE ($), raw heads | 286.64 | 398.18 | 590.58 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0059 | -0.0070 | -0.0095 |
| skill vs zero, raw heads | -0.0930 | -0.0855 | -0.0991 |
| EV, served | -0.0027 | -0.0032 | -0.0017 |
| EV, raw heads | -0.0410 | -0.0411 | -0.0351 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0273 | -0.0289 | 0.0051 |
| corr, Spearman, raw heads | -0.0113 | -0.0047 | -0.0024 |
| mean predicted ($), served | 14.32 | 19.34 | 40.40 |
| mean predicted ($), raw heads | 81.04 | 98.86 | 166.00 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9239 | 0.8392 | 0.8169 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1767 | 0.1956 | 0.2434 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 203.55 | 287.07 | 419.52 |
| CRPSS vs constant variance | 0.0145 | 0.0259 | 0.0569 |
| NLL | 7.4508 | 7.7956 | 8.1604 |
| PIT KS | 0.0566 | 0.0715 | 0.0746 |
| var / err^2 Spearman | -0.0613 | -0.0154 | 0.1209 |
| coverage of the 90% interval | 0.9002 | 0.9005 | 0.8636 |
| width of the 90% interval ($) | 1209.37 | 1749.46 | 2411.25 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0222 | [-0.0683, 0.0222] | NOISE |
| h1 | 0.0037 | [-0.0350, 0.0461] | NOISE |
| h2 | 0.0036 | [-0.0334, 0.0429] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.177 / h1 0.196 / h2 0.243) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8707 | 0.8925 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.7721 | 0.8632 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.6740 | 0.7750 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7864 | 0.8221 | 0.8054 | 0.6325 |
| expected if the two signs were independent | 0.7887 | 0.8174 | 0.8065 | 0.6437 |

- P(up) unanimity (all three horizons call the same side): 0.8235

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0203 vs 0.0249 (-0.0452): does not beat, noise (boot z -1.09) | 0.0054 vs 0.0086 (-0.0032): does not beat, noise (boot z -0.11) | -0.0270 vs -0.0220 (-0.0049): does not beat, noise (boot z -0.18) |
| logreg_lags | direction/auc | 0.4702 vs 0.4982 (-0.0280): does not beat, noise (boot z -1.08) | 0.5103 vs 0.5132 (-0.0029): does not beat, noise (boot z -0.13) | 0.4955 vs 0.5054 (-0.0099): does not beat, noise (boot z -0.43) |
| logreg_lags | direction/brier | 0.2601 vs 0.2667 (+0.0066): beats, noise (DM z +1.35) | 0.2598 vs 0.2794 (+0.0196): beats (DM z +3.55) | 0.2654 vs 0.2971 (+0.0318): beats (DM z +3.61) |
| logreg_lags | direction/ece_pos | 0.0777 vs 0.1201 (+0.0424): beats (boot z +2.41) | 0.0969 vs 0.1682 (+0.0712): beats (boot z +18.81) | 0.1183 vs 0.2020 (+0.0837): beats (boot z +19.23) |
| logreg_lags | direction/acc | 0.4865 vs 0.4967 (-0.0102): does not beat, noise (DM z -0.55) | 0.4829 vs 0.4822 (+0.0006): beats, noise (DM z +0.14) | 0.4724 vs 0.4742 (-0.0018): does not beat, noise (DM z -0.50) |
| logreg_lags | direction/bal_acc | 0.4913 vs 0.5052 (-0.0139): does not beat, noise (boot z -1.05) | 0.5011 vs 0.5012 (-0.0002): does not beat, noise (boot z -0.03) | 0.4962 vs 0.4989 (-0.0026): does not beat, noise (boot z -0.98) |
| class_prior | direction/mcc | -0.0203 vs 0.0000 (-0.0203): does not beat, noise (boot z -0.62) | 0.0054 vs 0.0000 (+0.0054): beats, noise (boot z +0.23) | -0.0270 vs 0.0000 (-0.0270): does not beat, noise (boot z -1.58) |
| class_prior | direction/auc | 0.4702 vs 0.5000 (-0.0298): does not beat, noise (boot z -1.36) | 0.5103 vs 0.5000 (+0.0103): beats, noise (boot z +0.77) | 0.4955 vs 0.5000 (-0.0045): does not beat, noise (boot z -0.38) |
| class_prior | direction/brier | 0.2601 vs 0.2657 (+0.0056): beats, noise (DM z +1.08) | 0.2598 vs 0.2738 (+0.0140): beats (DM z +3.52) | 0.2654 vs 0.2756 (+0.0102): beats (DM z +2.47) |
| class_prior | direction/ece_pos | 0.0777 vs 0.1255 (+0.0478): beats (boot z +2.34) | 0.0969 vs 0.1556 (+0.0586): beats (boot z +18.15) | 0.1183 vs 0.1618 (+0.0435): beats (boot z +16.21) |
| class_prior | direction/acc | 0.4865 vs 0.4907 (-0.0042): does not beat, noise (DM z -0.22) | 0.4829 vs 0.4802 (+0.0026): beats, noise (DM z +0.55) | 0.4724 vs 0.4752 (-0.0028): does not beat, noise (DM z -0.78) |
| class_prior | direction/bal_acc | 0.4913 vs 0.5000 (-0.0087): does not beat, noise (boot z -0.62) | 0.5011 vs 0.5000 (+0.0011): beats, noise (boot z +0.23) | 0.4962 vs 0.5000 (-0.0038): does not beat, noise (boot z -1.49) |
| zero_delta | delta/rmse | 401.49 vs 400.31 (-1.18, -0.30%): does not beat, noise (DM z -1.75) | 565.86 vs 563.88 (-1.98, -0.35%): does not beat, noise (DM z -1.78) | 811.46 vs 807.63 (-3.84, -0.48%): does not beat, noise (DM z -1.15) |
| zero_delta | delta/mae | 272.38 vs 271.46 (-0.92, -0.34%): does not beat, noise (DM z -1.92) | 379.21 vs 377.88 (-1.33, -0.35%): does not beat, noise (DM z -1.52) | 554.57 vs 550.98 (-3.59, -0.65%): does not beat, noise (DM z -1.45) |
| mean_delta | delta/rmse | 401.49 vs 405.60 (+4.11, +1.01%): beats (DM z +2.97) | 565.86 vs 578.56 (+12.70, +2.20%): beats (DM z +2.95) | 811.46 vs 847.03 (+35.57, +4.20%): beats (DM z +3.06) |
| mean_delta | delta/mae | 272.38 vs 277.50 (+5.12, +1.84%): beats (DM z +4.31) | 379.21 vs 393.77 (+14.56, +3.70%): beats (DM z +4.05) | 554.57 vs 595.01 (+40.44, +6.80%): beats (DM z +4.05) |
| const_var | variance/crps | 203.55 vs 206.55 (+3.00, +1.45%): beats (DM z +3.94) | 287.07 vs 294.69 (+7.62, +2.59%): beats (DM z +3.38) | 419.52 vs 444.83 (+25.31, +5.69%): beats (DM z +3.95) |
| const_var | variance/nll | 7.4508 vs 7.4448 (-0.0060): does not beat, noise (DM z -0.56) | 7.7956 vs 7.8022 (+0.0066): beats, noise (DM z +0.55) | 8.1604 vs 8.2104 (+0.0501): beats (DM z +2.32) |
| const_var | variance/pit_ks | 0.0566 vs 0.1064 (+0.0498): beats (boot z +23.30) | 0.0715 vs 0.1404 (+0.0688): beats (boot z +16.23) | 0.0746 vs 0.1810 (+0.1065): beats (boot z +30.95) |
| const_var | variance/corr_var_err2_spearman | -0.0613 vs 0.0000 (-0.0613): does not beat, significantly worse (boot z -2.84) | -0.0154 vs 0.0000 (-0.0154): does not beat, noise (boot z -0.58) | 0.1209 vs 0.0000 (+0.1209): beats (boot z +4.97) |

## Backtest (costs included)

- n_trades: 1223
- total_return: -0.9601
- sharpe_net: -118.5625
- sharpe_gross: 0.2796
- sortino: -137.5125
- max_drawdown: 0.9601
- hit_rate: 0.0621
- hit_rate_gross: 0.4890
- profit_factor: 0.0229
- avg_hold_bars: 11.9297
- exposure: 0.3377
- turnover: 739.9353
- fees_paid: 7399.3825
- traded_notional: 7399382.4690
- breakeven_cost_bps: 0.0480
- gross_edge_per_trade_bps: -0.2983
- costs_paid: 9619.1972
- gross_pnl: 17.7594
- net_pnl: -9601.4378

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8433, long_above 0.6118, short_below 0.5213, median 0.5679. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.01% | -118.56 | +96.01% | 1223 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.37% .. -95.16%) | -95.80% | -131.51 | | |

The random null enters at the strategy's rate (0.0427 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 24% of its seeds on net return, 100% on net Sharpe and 52% on gross return.
