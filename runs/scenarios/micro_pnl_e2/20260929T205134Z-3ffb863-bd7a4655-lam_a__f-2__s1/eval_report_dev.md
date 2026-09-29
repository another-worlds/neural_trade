# Evaluation report - dev split - run `20260929T205134Z-3ffb863-bd7a4655-lam_a__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.6768 | 0.8998 | 0.8732 |
| accuracy | 0.4774 | 0.4835 | 0.4814 |
| balanced accuracy | 0.4835 | 0.4971 | 0.4990 |
| precision (up) | 0.4703 | 0.4813 | 0.4758 |
| recall / sensitivity (up) | 0.6598 | 0.8968 | 0.8722 |
| specificity (down) | 0.3073 | 0.0975 | 0.1259 |
| F1 (up) | 0.5492 | 0.6264 | 0.6157 |
| MCC | -0.0352 | -0.0096 | -0.0029 |
| AUC | 0.4810 | 0.5060 | 0.5049 |
| Brier | 0.2551 | 0.2560 | 0.2572 |
| ECE (positive class) | 0.0645 | 0.0719 | 0.0760 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 11552 / 13012 / 5772 / 5956 | 16373 / 17643 / 1905 / 1885 | 16386 / 18055 / 2601 / 2402 |
| Gaussian readout: calls up | 0.8891 | 0.9250 | 0.8002 |
| Gaussian readout: MCC | -0.0124 | 0.0016 | 0.0173 |
| Gaussian readout: AUC | 0.4905 | 0.5115 | 0.5047 |
| Gaussian readout: Brier | 0.2513 | 0.2515 | 0.2527 |
| Gaussian readout: ECE | 0.0361 | 0.0420 | 0.0517 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 401.56 | 566.10 | 813.02 |
| RMSE ($), raw heads | 414.06 | 585.71 | 832.70 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.44 | 379.30 | 555.73 |
| MAE ($), raw heads | 282.68 | 395.45 | 575.94 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0063 | -0.0079 | -0.0134 |
| skill vs zero, raw heads | -0.0699 | -0.0789 | -0.0631 |
| EV, served | -0.0037 | -0.0028 | -0.0042 |
| EV, raw heads | -0.0382 | -0.0311 | -0.0253 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0442 | -0.0210 | -0.0082 |
| corr, Spearman, raw heads | -0.0249 | 0.0032 | -0.0075 |
| mean predicted ($), served | 12.10 | 23.76 | 45.57 |
| mean predicted ($), raw heads | 61.15 | 103.23 | 119.95 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.8825 | 0.9197 | 0.7991 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.1979 | 0.2301 | 0.3799 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.48 | 286.10 | 420.81 |
| CRPSS vs constant variance | 0.0197 | 0.0291 | 0.0540 |
| NLL | 7.4343 | 7.8548 | 8.1677 |
| PIT KS | 0.0493 | 0.0568 | 0.0782 |
| var / err^2 Spearman | 0.1306 | -0.0379 | 0.0978 |
| coverage of the 90% interval | 0.9006 | 0.9006 | 0.8610 |
| width of the 90% interval ($) | 1210.72 | 1755.66 | 2388.50 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0058 | [-0.0322, 0.0217] | NOISE |
| h1 | 0.0048 | [-0.0284, 0.0367] | NOISE |
| h2 | -0.0068 | [-0.0296, 0.0168] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.198 / h1 0.230 / h2 0.380) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.9039 | 0.9289 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.5561 | 0.7948 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.4758 | 0.7291 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6308 | 0.8587 | 0.7151 | 0.4503 |
| expected if the two signs were independent | 0.6466 | 0.8326 | 0.7240 | 0.4466 |

- P(up) unanimity (all three horizons call the same side): 0.5807

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0352 vs 0.0025 (-0.0377): does not beat, noise (boot z -1.40) | -0.0096 vs 0.0205 (-0.0301): does not beat, noise (boot z -1.01) | -0.0029 vs -0.0138 (+0.0109): beats, noise (boot z +0.48) |
| logreg_lags | direction/auc | 0.4810 vs 0.5320 (-0.0511): does not beat, significantly worse (boot z -2.10) | 0.5060 vs 0.5251 (-0.0191): does not beat, noise (boot z -0.89) | 0.5049 vs 0.5095 (-0.0046): does not beat, noise (boot z -0.27) |
| logreg_lags | direction/brier | 0.2551 vs 0.2536 (-0.0015): does not beat, noise (DM z -0.68) | 0.2560 vs 0.2575 (+0.0014): beats, noise (DM z +0.58) | 0.2572 vs 0.2702 (+0.0130): beats (DM z +2.83) |
| logreg_lags | direction/ece_pos | 0.0645 vs 0.0663 (+0.0018): beats, noise (boot z +0.13) | 0.0719 vs 0.0812 (+0.0093): beats, noise (boot z +1.71) | 0.0760 vs 0.1230 (+0.0470): beats (boot z +8.77) |
| logreg_lags | direction/acc | 0.4774 vs 0.4836 (-0.0063): does not beat, noise (DM z -0.45) | 0.4835 vs 0.4911 (-0.0076): does not beat, noise (DM z -0.82) | 0.4814 vs 0.4756 (+0.0058): beats, noise (DM z +0.69) |
| logreg_lags | direction/bal_acc | 0.4835 vs 0.5004 (-0.0168): does not beat, noise (boot z -1.73) | 0.4971 vs 0.5055 (-0.0084): does not beat, noise (boot z -1.00) | 0.4990 vs 0.4971 (+0.0019): beats, noise (boot z +0.33) |
| class_prior | direction/mcc | -0.0352 vs 0.0000 (-0.0352): does not beat, noise (boot z -1.71) | -0.0096 vs 0.0000 (-0.0096): does not beat, noise (boot z -0.59) | -0.0029 vs 0.0000 (-0.0029): does not beat, noise (boot z -0.20) |
| class_prior | direction/auc | 0.4810 vs 0.5000 (-0.0190): does not beat, noise (boot z -1.44) | 0.5060 vs 0.5000 (+0.0060): beats, noise (boot z +0.48) | 0.5049 vs 0.5000 (+0.0049): beats, noise (boot z +0.64) |
| class_prior | direction/brier | 0.2551 vs 0.2532 (-0.0019): does not beat, noise (DM z -1.12) | 0.2560 vs 0.2533 (-0.0027): does not beat, significantly worse (DM z -1.98) | 0.2572 vs 0.2573 (+0.0001): beats, noise (DM z +0.05) |
| class_prior | direction/ece_pos | 0.0645 vs 0.0593 (-0.0052): does not beat, noise (boot z -0.37) | 0.0719 vs 0.0603 (-0.0115): does not beat, significantly worse (boot z -2.39) | 0.0760 vs 0.0886 (+0.0126): beats (boot z +2.59) |
| class_prior | direction/acc | 0.4774 vs 0.4824 (-0.0051): does not beat, noise (DM z -0.36) | 0.4835 vs 0.4829 (+0.0005): beats, noise (DM z +0.07) | 0.4814 vs 0.4763 (+0.0050): beats, noise (DM z +0.54) |
| class_prior | direction/bal_acc | 0.4835 vs 0.5000 (-0.0165): does not beat, noise (boot z -1.71) | 0.4971 vs 0.5000 (-0.0029): does not beat, noise (boot z -0.59) | 0.4990 vs 0.5000 (-0.0010): does not beat, noise (boot z -0.20) |
| zero_delta | delta/rmse | 401.56 vs 400.31 (-1.25, -0.31%): does not beat, noise (DM z -1.95) | 566.10 vs 563.88 (-2.22, -0.39%): does not beat, noise (DM z -1.54) | 813.02 vs 807.63 (-5.39, -0.67%): does not beat, noise (DM z -1.48) |
| zero_delta | delta/mae | 272.44 vs 271.46 (-0.98, -0.36%): does not beat, significantly worse (DM z -2.26) | 379.30 vs 377.88 (-1.42, -0.38%): does not beat, noise (DM z -1.33) | 555.73 vs 550.98 (-4.76, -0.86%): does not beat, noise (DM z -1.68) |
| mean_delta | delta/rmse | 401.56 vs 405.60 (+4.05, +1.00%): beats (DM z +2.65) | 566.10 vs 578.56 (+12.46, +2.15%): beats (DM z +3.10) | 813.02 vs 847.03 (+34.02, +4.02%): beats (DM z +2.98) |
| mean_delta | delta/mae | 272.44 vs 277.50 (+5.06, +1.82%): beats (DM z +4.08) | 379.30 vs 393.77 (+14.47, +3.68%): beats (DM z +4.24) | 555.73 vs 595.01 (+39.28, +6.60%): beats (DM z +4.03) |
| const_var | variance/crps | 202.48 vs 206.55 (+4.07, +1.97%): beats (DM z +5.28) | 286.10 vs 294.69 (+8.58, +2.91%): beats (DM z +4.03) | 420.81 vs 444.83 (+24.01, +5.40%): beats (DM z +3.87) |
| const_var | variance/nll | 7.4343 vs 7.4448 (+0.0105): beats, noise (DM z +1.06) | 7.8548 vs 7.8022 (-0.0525): does not beat, noise (DM z -1.50) | 8.1677 vs 8.2104 (+0.0427): beats, noise (DM z +1.75) |
| const_var | variance/pit_ks | 0.0493 vs 0.1064 (+0.0571): beats (boot z +22.09) | 0.0568 vs 0.1404 (+0.0836): beats (boot z +23.16) | 0.0782 vs 0.1810 (+0.1029): beats (boot z +31.11) |
| const_var | variance/corr_var_err2_spearman | 0.1306 vs 0.0000 (+0.1306): beats (boot z +6.16) | -0.0379 vs 0.0000 (-0.0379): does not beat, noise (boot z -1.44) | 0.0978 vs 0.0000 (+0.0978): beats (boot z +3.89) |

## Backtest (costs included)

- n_trades: 1384
- total_return: -0.9745
- sharpe_net: -128.6490
- sharpe_gross: -1.5643
- sortino: -147.2723
- max_drawdown: 0.9745
- hit_rate: 0.0549
- hit_rate_gross: 0.4899
- profit_factor: 0.0255
- avg_hold_bars: 10.4379
- exposure: 0.3344
- turnover: 740.4038
- fees_paid: 7404.0811
- traded_notional: 7404081.0622
- breakeven_cost_bps: -0.3229
- gross_edge_per_trade_bps: -0.4531
- costs_paid: 9625.3054
- gross_pnl: -119.5479
- net_pnl: -9744.8532

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.679, long_above 0.5688, short_below 0.4924, median 0.5314. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.45% | -128.65 | +97.45% | 1384 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.71% .. -97.05%) | -97.39% | -147.00 | | |

The random null enters at the strategy's rate (0.0481 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 46% of its seeds on net return, 100% on net Sharpe and 30% on gross return.
