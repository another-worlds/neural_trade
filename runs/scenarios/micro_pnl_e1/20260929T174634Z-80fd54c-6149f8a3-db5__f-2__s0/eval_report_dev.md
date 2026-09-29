# Evaluation report - dev split - run `20260929T174634Z-80fd54c-6149f8a3-db5__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.6069 | 0.9105 | 0.7535 |
| accuracy | 0.4899 | 0.4790 | 0.4934 |
| balanced accuracy | 0.4936 | 0.4930 | 0.5054 |
| precision (up) | 0.4772 | 0.4791 | 0.4799 |
| recall / sensitivity (up) | 0.6003 | 0.9033 | 0.7592 |
| specificity (down) | 0.3870 | 0.0827 | 0.2516 |
| F1 (up) | 0.5317 | 0.6261 | 0.5881 |
| MCC | -0.0130 | -0.0245 | 0.0126 |
| AUC | 0.4930 | 0.4938 | 0.5076 |
| Brier | 0.2574 | 0.2535 | 0.2566 |
| ECE (positive class) | 0.0667 | 0.0587 | 0.0653 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 10510 / 11515 / 7269 / 6998 | 16492 / 17931 / 1617 / 1766 | 14264 / 15458 / 5198 / 4524 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.9902 | 0.9518 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0066 | -0.0014 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4919 | 0.4971 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2506 | 0.2520 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0273 | 0.0463 |
| Gaussian readout of the raw heads: calls up | 0.9493 | 0.9902 | 0.9518 |
| Gaussian readout of the raw heads: MCC | -0.0297 | -0.0066 | -0.0014 |
| Gaussian readout of the raw heads: AUC | 0.4619 | 0.4919 | 0.4971 |
| Gaussian readout of the raw heads: Brier | 0.2794 | 0.2638 | 0.2745 |
| Gaussian readout of the raw heads: ECE | 0.1429 | 0.1063 | 0.1328 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.31 | 564.78 | 810.84 |
| RMSE ($), raw heads | 421.04 | 583.73 | 845.93 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.46 | 378.56 | 553.91 |
| MAE ($), raw heads | 293.11 | 397.59 | 590.22 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0032 | -0.0080 |
| skill vs zero, raw heads | -0.1063 | -0.0716 | -0.0971 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0011 | -0.0019 |
| EV, raw heads | -0.0440 | -0.0207 | -0.0319 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0710 | -0.0398 | -0.0110 |
| corr, Spearman, raw heads | -0.0722 | -0.0277 | -0.0136 |
| mean predicted ($), served | 0.00 | 11.75 | 33.48 |
| mean predicted ($), raw heads | 89.57 | 107.09 | 167.90 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9514 | 0.9898 | 0.9519 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1097 | 0.1994 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.09 | 286.12 | 419.46 |
| CRPSS vs constant variance | 0.0216 | 0.0291 | 0.0570 |
| NLL | 7.4324 | 7.7696 | 8.1980 |
| PIT KS | 0.0387 | 0.0720 | 0.0624 |
| var / err^2 Spearman | 0.1181 | 0.0436 | 0.0771 |
| coverage of the 90% interval | 0.9028 | 0.9007 | 0.8637 |
| width of the 90% interval ($) | 1212.02 | 1749.57 | 2412.57 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0130 | [-0.0399, 0.0142] | NOISE |
| h1 | -0.0058 | [-0.0311, 0.0185] | NOISE |
| h2 | -0.0105 | [-0.0424, 0.0241] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.110 / h2 0.199) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5657 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.6930 | 0.8421 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.4731 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5816 | 0.9137 | 0.7380 | 0.4333 |
| expected if the two signs were independent | 0.5843 | 0.9057 | 0.7272 | 0.4232 |

- P(up) unanimity (all three horizons call the same side): 0.4605

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0130 vs 0.0025 (-0.0155): does not beat, noise (boot z -0.48) | -0.0245 vs 0.0205 (-0.0450): does not beat, noise (boot z -1.94) | 0.0126 vs -0.0138 (+0.0264): beats, noise (boot z +0.88) |
| logreg_lags | direction/auc | 0.4930 vs 0.5320 (-0.0390): does not beat, noise (boot z -1.66) | 0.4938 vs 0.5251 (-0.0313): does not beat, noise (boot z -1.63) | 0.5076 vs 0.5095 (-0.0019): does not beat, noise (boot z -0.18) |
| logreg_lags | direction/brier | 0.2574 vs 0.2536 (-0.0038): does not beat, noise (DM z -1.59) | 0.2535 vs 0.2575 (+0.0040): beats, noise (DM z +1.66) | 0.2566 vs 0.2702 (+0.0136): beats (DM z +3.13) |
| logreg_lags | direction/ece_pos | 0.0667 vs 0.0663 (-0.0004): does not beat, noise (boot z -0.03) | 0.0587 vs 0.0812 (+0.0225): beats (boot z +4.43) | 0.0653 vs 0.1230 (+0.0577): beats (boot z +8.47) |
| logreg_lags | direction/acc | 0.4899 vs 0.4836 (+0.0063): beats, noise (DM z +0.37) | 0.4790 vs 0.4911 (-0.0121): does not beat, noise (DM z -1.82) | 0.4934 vs 0.4756 (+0.0178): beats, noise (DM z +0.88) |
| logreg_lags | direction/bal_acc | 0.4936 vs 0.5004 (-0.0067): does not beat, noise (boot z -0.59) | 0.4930 vs 0.5055 (-0.0125): does not beat, noise (boot z -1.95) | 0.5054 vs 0.4971 (+0.0083): beats, noise (boot z +0.78) |
| class_prior | direction/mcc | -0.0130 vs 0.0000 (-0.0130): does not beat, noise (boot z -0.62) | -0.0245 vs 0.0000 (-0.0245): does not beat, noise (boot z -1.62) | 0.0126 vs 0.0000 (+0.0126): beats, noise (boot z +0.54) |
| class_prior | direction/auc | 0.4930 vs 0.5000 (-0.0070): does not beat, noise (boot z -0.50) | 0.4938 vs 0.5000 (-0.0062): does not beat, noise (boot z -0.67) | 0.5076 vs 0.5000 (+0.0076): beats, noise (boot z +0.47) |
| class_prior | direction/brier | 0.2574 vs 0.2532 (-0.0041): does not beat, significantly worse (DM z -2.14) | 0.2535 vs 0.2533 (-0.0001): does not beat, noise (DM z -0.20) | 0.2566 vs 0.2573 (+0.0007): beats, noise (DM z +0.25) |
| class_prior | direction/ece_pos | 0.0667 vs 0.0593 (-0.0073): does not beat, noise (boot z -0.52) | 0.0587 vs 0.0603 (+0.0016): beats, noise (boot z +0.33) | 0.0653 vs 0.0886 (+0.0233): beats (boot z +3.39) |
| class_prior | direction/acc | 0.4899 vs 0.4824 (+0.0075): beats, noise (DM z +0.45) | 0.4790 vs 0.4829 (-0.0039): does not beat, noise (DM z -0.62) | 0.4934 vs 0.4763 (+0.0171): beats, noise (DM z +0.79) |
| class_prior | direction/bal_acc | 0.4936 vs 0.5000 (-0.0064): does not beat, noise (boot z -0.62) | 0.4930 vs 0.5000 (-0.0070): does not beat, noise (boot z -1.61) | 0.5054 vs 0.5000 (+0.0054): beats, noise (boot z +0.54) |
| zero_delta | delta/rmse | 400.31 vs 400.31 (+0.00, +0.00%): does not beat | 564.78 vs 563.88 (-0.89, -0.16%): does not beat, noise (DM z -1.54) | 810.84 vs 807.63 (-3.21, -0.40%): does not beat, noise (DM z -1.25) |
| zero_delta | delta/mae | 271.46 vs 271.46 (+0.00, +0.00%): does not beat | 378.56 vs 377.88 (-0.68, -0.18%): does not beat, noise (DM z -1.44) | 553.91 vs 550.98 (-2.93, -0.53%): does not beat, noise (DM z -1.48) |
| mean_delta | delta/rmse | 400.31 vs 405.60 (+5.29, +1.31%): beats (DM z +2.85) | 564.78 vs 578.56 (+13.79, +2.38%): beats (DM z +2.92) | 810.84 vs 847.03 (+36.20, +4.27%): beats (DM z +2.96) |
| mean_delta | delta/mae | 271.46 vs 277.50 (+6.04, +2.18%): beats (DM z +3.94) | 378.56 vs 393.77 (+15.21, +3.86%): beats (DM z +3.98) | 553.91 vs 595.01 (+41.10, +6.91%): beats (DM z +4.01) |
| const_var | variance/crps | 202.09 vs 206.55 (+4.45, +2.16%): beats (DM z +4.71) | 286.12 vs 294.69 (+8.57, +2.91%): beats (DM z +3.47) | 419.46 vs 444.83 (+25.37, +5.70%): beats (DM z +3.84) |
| const_var | variance/nll | 7.4324 vs 7.4448 (+0.0124): beats, noise (DM z +1.17) | 7.7696 vs 7.8022 (+0.0326): beats (DM z +2.73) | 8.1980 vs 8.2104 (+0.0124): beats, noise (DM z +0.37) |
| const_var | variance/pit_ks | 0.0387 vs 0.1064 (+0.0678): beats (boot z +9.48) | 0.0720 vs 0.1404 (+0.0683): beats (boot z +11.69) | 0.0624 vs 0.1810 (+0.1186): beats (boot z +26.42) |
| const_var | variance/corr_var_err2_spearman | 0.1181 vs 0.0000 (+0.1181): beats (boot z +6.88) | 0.0436 vs 0.0000 (+0.0436): beats, noise (boot z +1.87) | 0.0771 vs 0.0000 (+0.0771): beats (boot z +4.04) |

## Backtest (costs included)

- n_trades: 1568
- total_return: -0.9850
- sharpe_net: -145.9499
- sharpe_gross: -5.4104
- sortino: -161.3267
- max_drawdown: 0.9850
- hit_rate: 0.0580
- hit_rate_gross: 0.4356
- profit_factor: 0.0309
- avg_hold_bars: 11.1460
- exposure: 0.4046
- turnover: 729.8349
- fees_paid: 7298.4885
- traded_notional: 7298488.4635
- breakeven_cost_bps: -0.9924
- gross_edge_per_trade_bps: -0.7402
- costs_paid: 9488.0350
- gross_pnl: -362.1549
- net_pnl: -9850.1899

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8791, long_above 0.5804, short_below 0.4830, median 0.5275. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.50% | -145.95 | +98.50% | 1568 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.56% .. -98.08%) | -98.34% | -153.03 | | |

The random null enters at the strategy's rate (0.0610 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 12% of its seeds on net return, 97% on net Sharpe and 7% on gross return.
