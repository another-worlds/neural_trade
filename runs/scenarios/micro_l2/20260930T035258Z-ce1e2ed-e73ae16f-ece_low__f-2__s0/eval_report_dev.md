# Evaluation report - dev split - run `20260930T035258Z-ce1e2ed-e73ae16f-ece_low__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.5777 | 0.7365 | 0.6393 |
| accuracy | 0.4884 | 0.4797 | 0.4971 |
| balanced accuracy | 0.4911 | 0.4878 | 0.5037 |
| precision (up) | 0.4747 | 0.4747 | 0.4792 |
| recall / sensitivity (up) | 0.5685 | 0.7239 | 0.6432 |
| specificity (down) | 0.4137 | 0.2517 | 0.3642 |
| F1 (up) | 0.5174 | 0.5734 | 0.5492 |
| MCC | -0.0180 | -0.0277 | 0.0077 |
| AUC | 0.4952 | 0.4937 | 0.5037 |
| Brier | 0.2561 | 0.2522 | 0.2587 |
| ECE (positive class) | 0.0677 | 0.0487 | 0.0645 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 9953 / 11013 / 7771 / 7555 | 13217 / 14628 / 4920 / 5041 | 12085 / 13133 / 7523 / 6703 |
| Gaussian readout: calls up | 0.9698 | 0.6004 | 0.6171 |
| Gaussian readout: MCC | -0.0170 | -0.0054 | -0.0023 |
| Gaussian readout: AUC | 0.4824 | 0.4915 | 0.4927 |
| Gaussian readout: Brier | 0.2506 | 0.2507 | 0.2539 |
| Gaussian readout: ECE | 0.0289 | 0.0229 | 0.0477 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.96 | 565.14 | 815.21 |
| RMSE ($), raw heads | 421.31 | 589.88 | 851.82 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.02 | 378.74 | 558.78 |
| MAE ($), raw heads | 291.03 | 402.77 | 596.78 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0032 | -0.0045 | -0.0189 |
| skill vs zero, raw heads | -0.1077 | -0.0943 | -0.1124 |
| EV, served | -0.0019 | -0.0031 | -0.0101 |
| EV, raw heads | -0.0480 | -0.0716 | -0.0712 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0626 | -0.0383 | -0.0133 |
| corr, Spearman, raw heads | -0.0585 | -0.0275 | -0.0192 |
| mean predicted ($), served | 7.28 | 8.32 | 44.13 |
| mean predicted ($), raw heads | 87.43 | 65.89 | 127.10 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9683 | 0.5941 | 0.6148 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0833 | 0.1263 | 0.3472 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 203.42 | 284.85 | 422.43 |
| CRPSS vs constant variance | 0.0151 | 0.0334 | 0.0504 |
| NLL | 7.4467 | 7.7976 | 8.1366 |
| PIT KS | 0.0492 | 0.0545 | 0.0853 |
| var / err^2 Spearman | 0.0092 | 0.1072 | 0.1250 |
| coverage of the 90% interval | 0.9013 | 0.9024 | 0.8600 |
| width of the 90% interval ($) | 1208.88 | 1759.79 | 2381.40 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0005 | [-0.0250, 0.0263] | NOISE |
| h1 | 0.0015 | [-0.0219, 0.0256] | NOISE |
| h2 | -0.0075 | [-0.0385, 0.0261] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.083 / h1 0.126 / h2 0.347) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6577 | 0.8113 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.8747 | 0.9737 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.6007 | 0.8000 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5772 | 0.6048 | 0.7490 | 0.3065 |
| expected if the two signs were independent | 0.5608 | 0.5459 | 0.5312 | 0.2137 |

- P(up) unanimity (all three horizons call the same side): 0.3695

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0180 vs 0.0025 (-0.0205): does not beat, noise (boot z -0.67) | -0.0277 vs 0.0205 (-0.0482): does not beat, noise (boot z -1.90) | 0.0077 vs -0.0138 (+0.0215): beats, noise (boot z +0.75) |
| logreg_lags | direction/auc | 0.4952 vs 0.5320 (-0.0368): does not beat, noise (boot z -1.66) | 0.4937 vs 0.5251 (-0.0314): does not beat, noise (boot z -1.53) | 0.5037 vs 0.5095 (-0.0058): does not beat, noise (boot z -0.46) |
| logreg_lags | direction/brier | 0.2561 vs 0.2536 (-0.0025): does not beat, noise (DM z -1.07) | 0.2522 vs 0.2575 (+0.0053): beats, noise (DM z +1.85) | 0.2587 vs 0.2702 (+0.0115): beats (DM z +2.54) |
| logreg_lags | direction/ece_pos | 0.0677 vs 0.0663 (-0.0014): does not beat, noise (boot z -0.09) | 0.0487 vs 0.0812 (+0.0325): beats (boot z +3.00) | 0.0645 vs 0.1230 (+0.0585): beats (boot z +5.19) |
| logreg_lags | direction/acc | 0.4884 vs 0.4836 (+0.0047): beats, noise (DM z +0.27) | 0.4797 vs 0.4911 (-0.0114): does not beat, noise (DM z -0.91) | 0.4971 vs 0.4756 (+0.0215): beats, noise (DM z +0.84) |
| logreg_lags | direction/bal_acc | 0.4911 vs 0.5004 (-0.0093): does not beat, noise (boot z -0.85) | 0.4878 vs 0.5055 (-0.0177): does not beat, significantly worse (boot z -2.07) | 0.5037 vs 0.4971 (+0.0066): beats, noise (boot z +0.60) |
| class_prior | direction/mcc | -0.0180 vs 0.0000 (-0.0180): does not beat, noise (boot z -0.90) | -0.0277 vs 0.0000 (-0.0277): does not beat, noise (boot z -1.85) | 0.0077 vs 0.0000 (+0.0077): beats, noise (boot z +0.36) |
| class_prior | direction/auc | 0.4952 vs 0.5000 (-0.0048): does not beat, noise (boot z -0.37) | 0.4937 vs 0.5000 (-0.0063): does not beat, noise (boot z -0.60) | 0.5037 vs 0.5000 (+0.0037): beats, noise (boot z +0.24) |
| class_prior | direction/brier | 0.2561 vs 0.2532 (-0.0029): does not beat, noise (DM z -1.52) | 0.2522 vs 0.2533 (+0.0012): beats, noise (DM z +0.88) | 0.2587 vs 0.2573 (-0.0014): does not beat, noise (DM z -0.43) |
| class_prior | direction/ece_pos | 0.0677 vs 0.0593 (-0.0083): does not beat, noise (boot z -0.56) | 0.0487 vs 0.0603 (+0.0117): beats, noise (boot z +1.08) | 0.0645 vs 0.0886 (+0.0241): beats (boot z +2.02) |
| class_prior | direction/acc | 0.4884 vs 0.4824 (+0.0060): beats, noise (DM z +0.35) | 0.4797 vs 0.4829 (-0.0032): does not beat, noise (DM z -0.23) | 0.4971 vs 0.4763 (+0.0208): beats, noise (DM z +0.76) |
| class_prior | direction/bal_acc | 0.4911 vs 0.5000 (-0.0089): does not beat, noise (boot z -0.90) | 0.4878 vs 0.5000 (-0.0122): does not beat, noise (boot z -1.85) | 0.5037 vs 0.5000 (+0.0037): beats, noise (boot z +0.36) |
| zero_delta | delta/rmse | 400.96 vs 400.31 (-0.65, -0.16%): does not beat, significantly worse (DM z -2.34) | 565.14 vs 563.88 (-1.26, -0.22%): does not beat, significantly worse (DM z -2.26) | 815.21 vs 807.63 (-7.58, -0.94%): does not beat, noise (DM z -1.94) |
| zero_delta | delta/mae | 272.02 vs 271.46 (-0.56, -0.21%): does not beat, significantly worse (DM z -2.56) | 378.74 vs 377.88 (-0.86, -0.23%): does not beat, noise (DM z -1.74) | 558.78 vs 550.98 (-7.80, -1.42%): does not beat, significantly worse (DM z -2.45) |
| mean_delta | delta/rmse | 400.96 vs 405.60 (+4.65, +1.15%): beats (DM z +2.87) | 565.14 vs 578.56 (+13.42, +2.32%): beats (DM z +2.75) | 815.21 vs 847.03 (+31.83, +3.76%): beats (DM z +2.78) |
| mean_delta | delta/mae | 272.02 vs 277.50 (+5.48, +1.98%): beats (DM z +4.06) | 378.74 vs 393.77 (+15.03, +3.82%): beats (DM z +3.76) | 558.78 vs 595.01 (+36.23, +6.09%): beats (DM z +3.67) |
| const_var | variance/crps | 203.42 vs 206.55 (+3.13, +1.51%): beats (DM z +3.27) | 284.85 vs 294.69 (+9.83, +3.34%): beats (DM z +3.73) | 422.43 vs 444.83 (+22.40, +5.04%): beats (DM z +3.53) |
| const_var | variance/nll | 7.4467 vs 7.4448 (-0.0019): does not beat, noise (DM z -0.14) | 7.7976 vs 7.8022 (+0.0046): beats, noise (DM z +0.19) | 8.1366 vs 8.2104 (+0.0738): beats (DM z +3.45) |
| const_var | variance/pit_ks | 0.0492 vs 0.1064 (+0.0572): beats (boot z +12.83) | 0.0545 vs 0.1404 (+0.0859): beats (boot z +12.28) | 0.0853 vs 0.1810 (+0.0958): beats (boot z +21.50) |
| const_var | variance/corr_var_err2_spearman | 0.0092 vs 0.0000 (+0.0092): beats, noise (boot z +0.41) | 0.1072 vs 0.0000 (+0.1072): beats (boot z +4.40) | 0.1250 vs 0.0000 (+0.1250): beats (boot z +5.21) |

## Backtest (costs included)

- n_trades: 1557
- total_return: -0.9835
- sharpe_net: -141.5525
- sharpe_gross: -2.4076
- sortino: -157.5870
- max_drawdown: 0.9835
- hit_rate: 0.0636
- hit_rate_gross: 0.4419
- profit_factor: 0.0353
- avg_hold_bars: 11.0951
- exposure: 0.3999
- turnover: 743.1284
- fees_paid: 7431.5873
- traded_notional: 7431587.2747
- breakeven_cost_bps: -0.4682
- gross_edge_per_trade_bps: -0.3110
- costs_paid: 9661.0635
- gross_pnl: -173.9738
- net_pnl: -9835.0372

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8026, long_above 0.5667, short_below 0.4712, median 0.5135. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.35% | -141.55 | +98.35% | 1557 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.48% .. -98.01%) | -98.27% | -152.22 | | |

The random null enters at the strategy's rate (0.0601 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 24% of its seeds on net return, 100% on net Sharpe and 23% on gross return.
