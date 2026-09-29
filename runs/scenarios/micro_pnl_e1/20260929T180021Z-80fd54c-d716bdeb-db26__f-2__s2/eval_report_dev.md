# Evaluation report - dev split - run `20260929T180021Z-80fd54c-d716bdeb-db26__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 26 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 26 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 13621 | 19285 | 24800 |
| n_eff of the scored moves (n scored // bars ahead) | 227 | 160 | 103 |
| true up-rate | 0.4907 | 0.4802 | 0.4752 |
| calls up (predicted up-rate) | 0.8496 | 0.9721 | 0.9041 |
| accuracy | 0.4832 | 0.4778 | 0.4692 |
| balanced accuracy | 0.4897 | 0.4965 | 0.4893 |
| precision (up) | 0.4847 | 0.4784 | 0.4692 |
| recall / sensitivity (up) | 0.8392 | 0.9685 | 0.8928 |
| specificity (down) | 0.1403 | 0.0245 | 0.0857 |
| F1 (up) | 0.6144 | 0.6405 | 0.6152 |
| MCC | -0.0288 | -0.0212 | -0.0364 |
| AUC | 0.4706 | 0.4998 | 0.4882 |
| Brier | 0.2604 | 0.2652 | 0.2626 |
| ECE (positive class) | 0.0837 | 0.1141 | 0.1056 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0093 | 0.0198 | 0.0248 |
| TP / FP / TN / FN | 5609 / 5964 / 973 / 1075 | 8969 / 9778 / 246 / 292 | 10521 / 11900 / 1116 / 1263 |
| Gaussian readout: calls up | 0.9359 | 0.8595 | 0.9213 |
| Gaussian readout: MCC | -0.0351 | -0.0120 | -0.0034 |
| Gaussian readout: AUC | 0.4804 | 0.4822 | 0.4993 |
| Gaussian readout: Brier | 0.2521 | 0.2507 | 0.2529 |
| Gaussian readout: ECE | 0.0441 | 0.0278 | 0.0556 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.20 | 564.53 | 811.17 |
| RMSE ($), raw heads | 423.38 | 572.22 | 829.52 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.18 | 378.32 | 553.94 |
| MAE ($), raw heads | 291.02 | 385.47 | 573.06 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0045 | -0.0023 | -0.0088 |
| skill vs zero, raw heads | -0.1186 | -0.0298 | -0.0550 |
| EV, served | -0.0017 | -0.0010 | -0.0012 |
| EV, raw heads | -0.0485 | -0.0150 | -0.0124 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0171 | -0.0208 | -0.0013 |
| corr, Spearman, raw heads | -0.0186 | -0.0174 | -0.0007 |
| mean predicted ($), served | 12.81 | 8.08 | 39.44 |
| mean predicted ($), raw heads | 95.64 | 49.94 | 129.18 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9052 | 0.8214 | 0.9123 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1340 | 0.1617 | 0.3053 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.72 | 290.91 | 419.70 |
| CRPSS vs constant variance | 0.0185 | 0.0128 | 0.0565 |
| NLL | 7.3909 | 7.7623 | 8.1507 |
| PIT KS | 0.0677 | 0.0990 | 0.0816 |
| var / err^2 Spearman | 0.1834 | 0.0295 | 0.0997 |
| coverage of the 90% interval | 0.8996 | 0.9016 | 0.8634 |
| width of the 90% interval ($) | 1204.57 | 1752.36 | 2409.08 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0263 | [-0.0733, 0.0228] | NOISE |
| h1 | 0.0075 | [-0.0404, 0.0580] | NOISE |
| h2 | 0.0046 | [-0.0270, 0.0432] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.134 / h1 0.162 / h2 0.305) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.2087 | 0.2590 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.9121 | 0.9702 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.1547 | 0.2425 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.8100 | 0.8008 | 0.8431 | 0.6162 |
| expected if the two signs were independent | 0.8249 | 0.8050 | 0.8427 | 0.6305 |

- P(up) unanimity (all three horizons call the same side): 0.8150

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0288 vs 0.0249 (-0.0537): does not beat, noise (boot z -1.23) | -0.0212 vs 0.0086 (-0.0298): does not beat, noise (boot z -1.62) | -0.0364 vs -0.0220 (-0.0143): does not beat, noise (boot z -0.47) |
| logreg_lags | direction/auc | 0.4706 vs 0.4982 (-0.0276): does not beat, noise (boot z -0.86) | 0.4998 vs 0.5132 (-0.0134): does not beat, noise (boot z -0.64) | 0.4882 vs 0.5054 (-0.0172): does not beat, noise (boot z -0.63) |
| logreg_lags | direction/brier | 0.2604 vs 0.2667 (+0.0063): beats, noise (DM z +1.40) | 0.2652 vs 0.2794 (+0.0142): beats (DM z +3.20) | 0.2626 vs 0.2971 (+0.0345): beats (DM z +3.15) |
| logreg_lags | direction/ece_pos | 0.0837 vs 0.1201 (+0.0364): beats (boot z +2.95) | 0.1141 vs 0.1682 (+0.0541): beats (boot z +20.13) | 0.1056 vs 0.2020 (+0.0964): beats (boot z +11.81) |
| logreg_lags | direction/acc | 0.4832 vs 0.4967 (-0.0135): does not beat, noise (DM z -0.96) | 0.4778 vs 0.4822 (-0.0044): does not beat, noise (DM z -1.55) | 0.4692 vs 0.4742 (-0.0049): does not beat, noise (DM z -0.47) |
| logreg_lags | direction/bal_acc | 0.4897 vs 0.5052 (-0.0155): does not beat, noise (boot z -1.31) | 0.4965 vs 0.5012 (-0.0047): does not beat, noise (boot z -1.77) | 0.4893 vs 0.4989 (-0.0096): does not beat, noise (boot z -1.59) |
| class_prior | direction/mcc | -0.0288 vs 0.0000 (-0.0288): does not beat, noise (boot z -0.97) | -0.0212 vs 0.0000 (-0.0212): does not beat, noise (boot z -0.93) | -0.0364 vs 0.0000 (-0.0364): does not beat, noise (boot z -1.83) |
| class_prior | direction/auc | 0.4706 vs 0.5000 (-0.0294): does not beat, noise (boot z -1.52) | 0.4998 vs 0.5000 (-0.0002): does not beat, noise (boot z -0.01) | 0.4882 vs 0.5000 (-0.0118): does not beat, noise (boot z -0.86) |
| class_prior | direction/brier | 0.2604 vs 0.2657 (+0.0053): beats, noise (DM z +1.28) | 0.2652 vs 0.2738 (+0.0087): beats (DM z +2.33) | 0.2626 vs 0.2756 (+0.0129): beats (DM z +2.21) |
| class_prior | direction/ece_pos | 0.0837 vs 0.1255 (+0.0418): beats (boot z +3.14) | 0.1141 vs 0.1556 (+0.0415): beats (boot z +10.58) | 0.1056 vs 0.1618 (+0.0562): beats (boot z +7.60) |
| class_prior | direction/acc | 0.4832 vs 0.4907 (-0.0075): does not beat, noise (DM z -0.56) | 0.4778 vs 0.4802 (-0.0024): does not beat, noise (DM z -0.53) | 0.4692 vs 0.4752 (-0.0059): does not beat, noise (DM z -0.55) |
| class_prior | direction/bal_acc | 0.4897 vs 0.5000 (-0.0103): does not beat, noise (boot z -0.97) | 0.4965 vs 0.5000 (-0.0035): does not beat, noise (boot z -0.90) | 0.4893 vs 0.5000 (-0.0107): does not beat, noise (boot z -1.80) |
| zero_delta | delta/rmse | 401.20 vs 400.31 (-0.89, -0.22%): does not beat, noise (DM z -1.46) | 564.53 vs 563.88 (-0.65, -0.12%): does not beat, noise (DM z -1.40) | 811.17 vs 807.63 (-3.54, -0.44%): does not beat, noise (DM z -1.13) |
| zero_delta | delta/mae | 272.18 vs 271.46 (-0.73, -0.27%): does not beat, noise (DM z -1.69) | 378.32 vs 377.88 (-0.45, -0.12%): does not beat, noise (DM z -1.21) | 553.94 vs 550.98 (-2.96, -0.54%): does not beat, noise (DM z -1.26) |
| mean_delta | delta/rmse | 401.20 vs 405.60 (+4.40, +1.09%): beats (DM z +3.21) | 564.53 vs 578.56 (+14.03, +2.43%): beats (DM z +2.88) | 811.17 vs 847.03 (+35.87, +4.23%): beats (DM z +3.06) |
| mean_delta | delta/mae | 272.18 vs 277.50 (+5.32, +1.92%): beats (DM z +4.40) | 378.32 vs 393.77 (+15.45, +3.92%): beats (DM z +3.88) | 553.94 vs 595.01 (+41.07, +6.90%): beats (DM z +4.12) |
| const_var | variance/crps | 202.72 vs 206.55 (+3.82, +1.85%): beats (DM z +5.06) | 290.91 vs 294.69 (+3.78, +1.28%): beats, noise (DM z +1.35) | 419.70 vs 444.83 (+25.13, +5.65%): beats (DM z +3.85) |
| const_var | variance/nll | 7.3909 vs 7.4448 (+0.0539): beats (DM z +4.16) | 7.7623 vs 7.8022 (+0.0399): beats, noise (DM z +1.43) | 8.1507 vs 8.2104 (+0.0597): beats (DM z +2.69) |
| const_var | variance/pit_ks | 0.0677 vs 0.1064 (+0.0387): beats (boot z +11.16) | 0.0990 vs 0.1404 (+0.0414): beats (boot z +5.51) | 0.0816 vs 0.1810 (+0.0995): beats (boot z +25.18) |
| const_var | variance/corr_var_err2_spearman | 0.1834 vs 0.0000 (+0.1834): beats (boot z +9.69) | 0.0295 vs 0.0000 (+0.0295): beats, noise (boot z +1.30) | 0.0997 vs 0.0000 (+0.0997): beats (boot z +4.22) |

## Backtest (costs included)

- n_trades: 1249
- total_return: -0.9630
- sharpe_net: -121.6838
- sharpe_gross: -1.4253
- sortino: -138.5934
- max_drawdown: 0.9630
- hit_rate: 0.0544
- hit_rate_gross: 0.4932
- profit_factor: 0.0194
- avg_hold_bars: 11.4243
- exposure: 0.3303
- turnover: 732.5038
- fees_paid: 7324.9297
- traded_notional: 7324929.7427
- breakeven_cost_bps: -0.2932
- gross_edge_per_trade_bps: -0.3388
- costs_paid: 9522.4087
- gross_pnl: -107.3715
- net_pnl: -9629.7802

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.165, long_above 0.6048, short_below 0.5340, median 0.5679. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.30% | -121.68 | +96.30% | 1249 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.78% .. -95.74%) | -96.26% | -136.47 | | |

The random null enters at the strategy's rate (0.0432 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 44% of its seeds on net return, 100% on net Sharpe and 30% on gross return.
