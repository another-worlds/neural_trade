# Evaluation report - dev split - run `20260929T205408Z-3ffb863-4f910d32-lam_a__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.4514 | 0.7287 | 0.4649 |
| accuracy | 0.5083 | 0.5044 | 0.4958 |
| balanced accuracy | 0.5066 | 0.5122 | 0.4941 |
| precision (up) | 0.4897 | 0.4913 | 0.4700 |
| recall / sensitivity (up) | 0.4583 | 0.7414 | 0.4588 |
| specificity (down) | 0.5549 | 0.2830 | 0.5295 |
| F1 (up) | 0.4735 | 0.5910 | 0.4643 |
| MCC | 0.0133 | 0.0274 | -0.0118 |
| AUC | 0.5023 | 0.5239 | 0.4882 |
| Brier | 0.2532 | 0.2558 | 0.2553 |
| ECE (positive class) | 0.0304 | 0.0615 | 0.0522 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 8024 / 8360 / 10424 / 9484 | 13536 / 14015 / 5533 / 4722 | 8619 / 9719 / 10937 / 10169 |
| Gaussian readout: calls up | 0.9607 | 0.5966 | 0.5858 |
| Gaussian readout: MCC | -0.0023 | 0.0137 | 0.0247 |
| Gaussian readout: AUC | 0.4918 | 0.5107 | 0.5131 |
| Gaussian readout: Brier | 0.2510 | 0.2504 | 0.2514 |
| Gaussian readout: ECE | 0.0340 | 0.0210 | 0.0333 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.11 | 565.50 | 811.22 |
| RMSE ($), raw heads | 414.65 | 577.21 | 826.30 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.18 | 378.75 | 553.80 |
| MAE ($), raw heads | 283.76 | 389.68 | 569.80 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0040 | -0.0058 | -0.0089 |
| skill vs zero, raw heads | -0.0729 | -0.0478 | -0.0468 |
| EV, served | -0.0013 | -0.0048 | -0.0060 |
| EV, raw heads | -0.0271 | -0.0434 | -0.0381 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0135 | -0.0158 | 0.0043 |
| corr, Spearman, raw heads | -0.0267 | -0.0013 | 0.0047 |
| mean predicted ($), served | 12.60 | 6.16 | 18.10 |
| mean predicted ($), raw heads | 75.48 | 21.56 | 44.00 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9606 | 0.5874 | 0.5824 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1669 | 0.2857 | 0.4115 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 201.46 | 285.27 | 419.13 |
| CRPSS vs constant variance | 0.0246 | 0.0320 | 0.0578 |
| NLL | 7.4158 | 7.7611 | 8.2191 |
| PIT KS | 0.0469 | 0.0645 | 0.0436 |
| var / err^2 Spearman | 0.1866 | 0.1496 | 0.1164 |
| coverage of the 90% interval | 0.9005 | 0.9037 | 0.8591 |
| width of the 90% interval ($) | 1206.79 | 1776.97 | 2370.78 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0134 | [-0.0357, 0.0088] | NOISE |
| h1 | 0.0001 | [-0.0301, 0.0332] | NOISE |
| h2 | -0.0144 | [-0.0391, 0.0099] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.167 / h1 0.286 / h2 0.411) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6358 | 0.8807 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.8016 | 0.8984 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.5046 | 0.7942 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4725 | 0.5836 | 0.6817 | 0.1745 |
| expected if the two signs were independent | 0.4599 | 0.5390 | 0.4939 | 0.1102 |

- P(up) unanimity (all three horizons call the same side): 0.2275

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0133 vs 0.0025 (+0.0108): beats, noise (boot z +0.40) | 0.0274 vs 0.0205 (+0.0069): beats, noise (boot z +0.32) | -0.0118 vs -0.0138 (+0.0020): beats, noise (boot z +0.06) |
| logreg_lags | direction/auc | 0.5023 vs 0.5320 (-0.0298): does not beat, noise (boot z -1.42) | 0.5239 vs 0.5251 (-0.0012): does not beat, noise (boot z -0.14) | 0.4882 vs 0.5095 (-0.0213): does not beat, noise (boot z -0.93) |
| logreg_lags | direction/brier | 0.2532 vs 0.2536 (+0.0004): beats, noise (DM z +0.18) | 0.2558 vs 0.2575 (+0.0016): beats, noise (DM z +1.14) | 0.2553 vs 0.2702 (+0.0149): beats (DM z +2.01) |
| logreg_lags | direction/ece_pos | 0.0304 vs 0.0663 (+0.0359): beats (boot z +2.86) | 0.0615 vs 0.0812 (+0.0197): beats (boot z +4.74) | 0.0522 vs 0.1230 (+0.0708): beats (boot z +3.92) |
| logreg_lags | direction/acc | 0.5083 vs 0.4836 (+0.0247): beats, noise (DM z +1.30) | 0.5044 vs 0.4911 (+0.0133): beats, noise (DM z +1.00) | 0.4958 vs 0.4756 (+0.0202): beats, noise (DM z +0.60) |
| logreg_lags | direction/bal_acc | 0.5066 vs 0.5004 (+0.0062): beats, noise (boot z +0.73) | 0.5122 vs 0.5055 (+0.0067): beats, noise (boot z +0.85) | 0.4941 vs 0.4971 (-0.0030): does not beat, noise (boot z -0.27) |
| class_prior | direction/mcc | 0.0133 vs 0.0000 (+0.0133): beats, noise (boot z +0.87) | 0.0274 vs 0.0000 (+0.0274): beats, noise (boot z +1.51) | -0.0118 vs 0.0000 (-0.0118): does not beat, noise (boot z -0.64) |
| class_prior | direction/auc | 0.5023 vs 0.5000 (+0.0023): beats, noise (boot z +0.24) | 0.5239 vs 0.5000 (+0.0239): beats, noise (boot z +1.80) | 0.4882 vs 0.5000 (-0.0118): does not beat, noise (boot z -0.97) |
| class_prior | direction/brier | 0.2532 vs 0.2532 (+0.0000): beats, noise (DM z +0.00) | 0.2558 vs 0.2533 (-0.0025): does not beat, noise (DM z -1.19) | 0.2553 vs 0.2573 (+0.0020): beats, noise (DM z +0.45) |
| class_prior | direction/ece_pos | 0.0304 vs 0.0593 (+0.0289): beats (boot z +2.30) | 0.0615 vs 0.0603 (-0.0012): does not beat, noise (boot z -0.24) | 0.0522 vs 0.0886 (+0.0364): beats (boot z +1.99) |
| class_prior | direction/acc | 0.5083 vs 0.4824 (+0.0259): beats, noise (DM z +1.36) | 0.5044 vs 0.4829 (+0.0215): beats, noise (DM z +1.41) | 0.4958 vs 0.4763 (+0.0195): beats, noise (DM z +0.55) |
| class_prior | direction/bal_acc | 0.5066 vs 0.5000 (+0.0066): beats, noise (boot z +0.87) | 0.5122 vs 0.5000 (+0.0122): beats, noise (boot z +1.51) | 0.4941 vs 0.5000 (-0.0059): does not beat, noise (boot z -0.64) |
| zero_delta | delta/rmse | 401.11 vs 400.31 (-0.81, -0.20%): does not beat, noise (DM z -1.39) | 565.50 vs 563.88 (-1.62, -0.29%): does not beat, noise (DM z -1.50) | 811.22 vs 807.63 (-3.59, -0.44%): does not beat, noise (DM z -1.19) |
| zero_delta | delta/mae | 272.18 vs 271.46 (-0.72, -0.26%): does not beat, noise (DM z -1.79) | 378.75 vs 377.88 (-0.87, -0.23%): does not beat, noise (DM z -0.90) | 553.80 vs 550.98 (-2.82, -0.51%): does not beat, noise (DM z -1.13) |
| mean_delta | delta/rmse | 401.11 vs 405.60 (+4.49, +1.11%): beats (DM z +3.23) | 565.50 vs 578.56 (+13.06, +2.26%): beats (DM z +2.79) | 811.22 vs 847.03 (+35.82, +4.23%): beats (DM z +2.79) |
| mean_delta | delta/mae | 272.18 vs 277.50 (+5.32, +1.92%): beats (DM z +4.35) | 378.75 vs 393.77 (+15.02, +3.81%): beats (DM z +3.58) | 553.80 vs 595.01 (+41.21, +6.93%): beats (DM z +3.56) |
| const_var | variance/crps | 201.46 vs 206.55 (+5.09, +2.46%): beats (DM z +6.15) | 285.27 vs 294.69 (+9.42, +3.20%): beats (DM z +3.65) | 419.13 vs 444.83 (+25.70, +5.78%): beats (DM z +3.51) |
| const_var | variance/nll | 7.4158 vs 7.4448 (+0.0291): beats (DM z +1.98) | 7.7611 vs 7.8022 (+0.0411): beats (DM z +3.26) | 8.2191 vs 8.2104 (-0.0086): does not beat, noise (DM z -0.20) |
| const_var | variance/pit_ks | 0.0469 vs 0.1064 (+0.0596): beats (boot z +21.08) | 0.0645 vs 0.1404 (+0.0759): beats (boot z +9.50) | 0.0436 vs 0.1810 (+0.1374): beats (boot z +14.07) |
| const_var | variance/corr_var_err2_spearman | 0.1866 vs 0.0000 (+0.1866): beats (boot z +8.68) | 0.1496 vs 0.0000 (+0.1496): beats (boot z +6.25) | 0.1164 vs 0.0000 (+0.1164): beats (boot z +4.77) |

## Backtest (costs included)

- n_trades: 1129
- total_return: -0.9437
- sharpe_net: -107.2423
- sharpe_gross: 4.1077
- sortino: -124.0197
- max_drawdown: 0.9437
- hit_rate: 0.0638
- hit_rate_gross: 0.5288
- profit_factor: 0.0320
- avg_hold_bars: 12.2347
- exposure: 0.3197
- turnover: 751.9625
- fees_paid: 7519.6661
- traded_notional: 7519666.0894
- breakeven_cost_bps: 0.9006
- gross_edge_per_trade_bps: 0.5697
- costs_paid: 9775.5659
- gross_pnl: 338.5991
- net_pnl: -9436.9668

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8563, long_above 0.5524, short_below 0.4643, median 0.5040. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -94.37% | -107.24 | +94.37% | 1129 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -95.60% .. -94.00%) | -94.79% | -126.93 | | |

The random null enters at the strategy's rate (0.0384 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 82% of its seeds on net return, 100% on net Sharpe and 96% on gross return.
