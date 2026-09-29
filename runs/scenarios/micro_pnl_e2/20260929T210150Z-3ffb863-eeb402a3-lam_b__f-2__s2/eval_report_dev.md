# Evaluation report - dev split - run `20260929T210150Z-3ffb863-eeb402a3-lam_b__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.3585 | 0.6630 | 0.7130 |
| accuracy | 0.5157 | 0.4994 | 0.4805 |
| balanced accuracy | 0.5107 | 0.5049 | 0.4905 |
| precision (up) | 0.4974 | 0.4867 | 0.4697 |
| recall / sensitivity (up) | 0.3697 | 0.6681 | 0.7031 |
| specificity (down) | 0.6518 | 0.3417 | 0.2780 |
| F1 (up) | 0.4241 | 0.5631 | 0.5632 |
| MCC | 0.0224 | 0.0104 | -0.0209 |
| AUC | 0.5024 | 0.5139 | 0.4911 |
| Brier | 0.2530 | 0.2551 | 0.2577 |
| ECE (positive class) | 0.0240 | 0.0494 | 0.0742 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 6472 / 6540 / 12244 / 11036 | 12199 / 12868 / 6680 / 6059 | 13209 / 14914 / 5742 / 5579 |
| Gaussian readout: calls up | 0.7603 | 0.8212 | 0.8260 |
| Gaussian readout: MCC | -0.0197 | 0.0021 | 0.0198 |
| Gaussian readout: AUC | 0.4959 | 0.5157 | 0.5154 |
| Gaussian readout: Brier | 0.2506 | 0.2504 | 0.2515 |
| Gaussian readout: ECE | 0.0306 | 0.0278 | 0.0461 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.94 | 564.98 | 811.38 |
| RMSE ($), raw heads | 416.79 | 581.17 | 835.02 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.94 | 378.49 | 553.71 |
| MAE ($), raw heads | 284.54 | 392.59 | 577.15 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0032 | -0.0039 | -0.0093 |
| skill vs zero, raw heads | -0.0841 | -0.0623 | -0.0690 |
| EV, served | -0.0014 | -0.0014 | -0.0022 |
| EV, raw heads | -0.0482 | -0.0289 | -0.0248 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0078 | -0.0137 | -0.0023 |
| corr, Spearman, raw heads | -0.0169 | 0.0064 | 0.0075 |
| mean predicted ($), served | 9.31 | 13.81 | 37.81 |
| mean predicted ($), raw heads | 65.70 | 83.23 | 132.35 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.7528 | 0.8128 | 0.8239 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1417 | 0.1660 | 0.2856 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.22 | 287.19 | 419.28 |
| CRPSS vs constant variance | 0.0210 | 0.0255 | 0.0574 |
| NLL | 7.3922 | 7.7528 | 8.1583 |
| PIT KS | 0.0611 | 0.0854 | 0.0772 |
| var / err^2 Spearman | 0.1870 | 0.1217 | 0.1108 |
| coverage of the 90% interval | 0.9019 | 0.9025 | 0.8635 |
| width of the 90% interval ($) | 1211.26 | 1762.56 | 2403.90 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0135 | [-0.0364, 0.0105] | NOISE |
| h1 | 0.0033 | [-0.0212, 0.0320] | NOISE |
| h2 | -0.0091 | [-0.0372, 0.0193] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.142 / h1 0.166 / h2 0.286) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7358 | 0.8211 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.8367 | 0.9373 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.6042 | 0.7721 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4749 | 0.6556 | 0.7288 | 0.2293 |
| expected if the two signs were independent | 0.4278 | 0.5985 | 0.6376 | 0.1710 |

- P(up) unanimity (all three horizons call the same side): 0.2417

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0224 vs 0.0025 (+0.0199): beats, noise (boot z +0.70) | 0.0104 vs 0.0205 (-0.0101): does not beat, noise (boot z -0.46) | -0.0209 vs -0.0138 (-0.0071): does not beat, noise (boot z -0.25) |
| logreg_lags | direction/auc | 0.5024 vs 0.5320 (-0.0296): does not beat, noise (boot z -1.39) | 0.5139 vs 0.5251 (-0.0111): does not beat, noise (boot z -1.11) | 0.4911 vs 0.5095 (-0.0184): does not beat, noise (boot z -0.89) |
| logreg_lags | direction/brier | 0.2530 vs 0.2536 (+0.0006): beats, noise (DM z +0.25) | 0.2551 vs 0.2575 (+0.0023): beats, noise (DM z +1.16) | 0.2577 vs 0.2702 (+0.0125): beats (DM z +2.23) |
| logreg_lags | direction/ece_pos | 0.0240 vs 0.0663 (+0.0423): beats (boot z +3.09) | 0.0494 vs 0.0812 (+0.0318): beats (boot z +4.09) | 0.0742 vs 0.1230 (+0.0488): beats (boot z +4.28) |
| logreg_lags | direction/acc | 0.5157 vs 0.4836 (+0.0321): beats, noise (DM z +1.47) | 0.4994 vs 0.4911 (+0.0083): beats, noise (DM z +0.53) | 0.4805 vs 0.4756 (+0.0049): beats, noise (DM z +0.26) |
| logreg_lags | direction/bal_acc | 0.5107 vs 0.5004 (+0.0104): beats, noise (boot z +1.18) | 0.5049 vs 0.5055 (-0.0006): does not beat, noise (boot z -0.07) | 0.4905 vs 0.4971 (-0.0066): does not beat, noise (boot z -0.70) |
| class_prior | direction/mcc | 0.0224 vs 0.0000 (+0.0224): beats, noise (boot z +1.44) | 0.0104 vs 0.0000 (+0.0104): beats, noise (boot z +0.61) | -0.0209 vs 0.0000 (-0.0209): does not beat, noise (boot z -1.23) |
| class_prior | direction/auc | 0.5024 vs 0.5000 (+0.0024): beats, noise (boot z +0.24) | 0.5139 vs 0.5000 (+0.0139): beats, noise (boot z +1.20) | 0.4911 vs 0.5000 (-0.0089): does not beat, noise (boot z -0.75) |
| class_prior | direction/brier | 0.2530 vs 0.2532 (+0.0002): beats, noise (DM z +0.11) | 0.2551 vs 0.2533 (-0.0018): does not beat, noise (DM z -0.85) | 0.2577 vs 0.2573 (-0.0004): does not beat, noise (DM z -0.17) |
| class_prior | direction/ece_pos | 0.0240 vs 0.0593 (+0.0354): beats (boot z +2.58) | 0.0494 vs 0.0603 (+0.0109): beats, noise (boot z +1.31) | 0.0742 vs 0.0886 (+0.0144): beats, noise (boot z +1.27) |
| class_prior | direction/acc | 0.5157 vs 0.4824 (+0.0333): beats, noise (DM z +1.52) | 0.4994 vs 0.4829 (+0.0164): beats, noise (DM z +0.93) | 0.4805 vs 0.4763 (+0.0041): beats, noise (DM z +0.21) |
| class_prior | direction/bal_acc | 0.5107 vs 0.5000 (+0.0107): beats, noise (boot z +1.44) | 0.5049 vs 0.5000 (+0.0049): beats, noise (boot z +0.61) | 0.4905 vs 0.5000 (-0.0095): does not beat, noise (boot z -1.22) |
| zero_delta | delta/rmse | 400.94 vs 400.31 (-0.64, -0.16%): does not beat, noise (DM z -1.20) | 564.98 vs 563.88 (-1.10, -0.20%): does not beat, noise (DM z -1.17) | 811.38 vs 807.63 (-3.75, -0.46%): does not beat, noise (DM z -1.16) |
| zero_delta | delta/mae | 271.94 vs 271.46 (-0.48, -0.18%): does not beat, noise (DM z -1.32) | 378.49 vs 377.88 (-0.61, -0.16%): does not beat, noise (DM z -0.88) | 553.71 vs 550.98 (-2.73, -0.50%): does not beat, noise (DM z -1.13) |
| mean_delta | delta/rmse | 400.94 vs 405.60 (+4.66, +1.15%): beats (DM z +3.18) | 564.98 vs 578.56 (+13.58, +2.35%): beats (DM z +3.06) | 811.38 vs 847.03 (+35.65, +4.21%): beats (DM z +3.04) |
| mean_delta | delta/mae | 271.94 vs 277.50 (+5.56, +2.00%): beats (DM z +4.21) | 378.49 vs 393.77 (+15.28, +3.88%): beats (DM z +4.05) | 553.71 vs 595.01 (+41.30, +6.94%): beats (DM z +4.08) |
| const_var | variance/crps | 202.22 vs 206.55 (+4.33, +2.10%): beats (DM z +5.00) | 287.19 vs 294.69 (+7.50, +2.55%): beats (DM z +3.05) | 419.28 vs 444.83 (+25.55, +5.74%): beats (DM z +3.92) |
| const_var | variance/nll | 7.3922 vs 7.4448 (+0.0526): beats (DM z +4.25) | 7.7528 vs 7.8022 (+0.0494): beats (DM z +2.82) | 8.1583 vs 8.2104 (+0.0522): beats (DM z +2.16) |
| const_var | variance/pit_ks | 0.0611 vs 0.1064 (+0.0453): beats (boot z +10.43) | 0.0854 vs 0.1404 (+0.0550): beats (boot z +9.16) | 0.0772 vs 0.1810 (+0.1039): beats (boot z +26.11) |
| const_var | variance/corr_var_err2_spearman | 0.1870 vs 0.0000 (+0.1870): beats (boot z +8.93) | 0.1217 vs 0.0000 (+0.1217): beats (boot z +4.97) | 0.1108 vs 0.0000 (+0.1108): beats (boot z +4.55) |

## Backtest (costs included)

- n_trades: 1145
- total_return: -0.9489
- sharpe_net: -110.3084
- sharpe_gross: 1.5445
- sortino: -127.1212
- max_drawdown: 0.9490
- hit_rate: 0.0629
- hit_rate_gross: 0.5188
- profit_factor: 0.0265
- avg_hold_bars: 12.0218
- exposure: 0.3186
- turnover: 739.2329
- fees_paid: 7392.3300
- traded_notional: 7392329.9768
- breakeven_cost_bps: 0.3266
- gross_edge_per_trade_bps: 0.0749
- costs_paid: 9610.0290
- gross_pnl: 120.7307
- net_pnl: -9489.2983

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.008, long_above 0.5468, short_below 0.4646, median 0.5024. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -94.89% | -110.31 | +94.90% | 1145 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -95.62% .. -94.16%) | -94.91% | -127.51 | | |

The random null enters at the strategy's rate (0.0389 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 54% of its seeds on net return, 100% on net Sharpe and 75% on gross return.
