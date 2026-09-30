# Evaluation report - dev split - run `20260930T034544Z-ce1e2ed-6b4a65ac-clip100__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.8424 | 0.9160 | 0.8881 |
| accuracy | 0.4803 | 0.4739 | 0.4773 |
| balanced accuracy | 0.4924 | 0.4881 | 0.4957 |
| precision (up) | 0.4779 | 0.4765 | 0.4739 |
| recall / sensitivity (up) | 0.8345 | 0.9037 | 0.8836 |
| specificity (down) | 0.1502 | 0.0725 | 0.1077 |
| F1 (up) | 0.6078 | 0.6240 | 0.6169 |
| MCC | -0.0210 | -0.0428 | -0.0138 |
| AUC | 0.4963 | 0.4823 | 0.5025 |
| Brier | 0.2596 | 0.2536 | 0.2591 |
| ECE (positive class) | 0.0866 | 0.0609 | 0.0861 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 14611 / 15963 / 2821 / 2897 | 16500 / 18130 / 1418 / 1758 | 16601 / 18431 / 2225 / 2187 |
| Gaussian readout: calls up | 0.9917 | 0.7678 | 0.6928 |
| Gaussian readout: MCC | 0.0001 | -0.0173 | 0.0029 |
| Gaussian readout: AUC | 0.4712 | 0.4855 | 0.5003 |
| Gaussian readout: Brier | 0.2509 | 0.2507 | 0.2516 |
| Gaussian readout: ECE | 0.0284 | 0.0260 | 0.0405 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.92 | 564.94 | 811.45 |
| RMSE ($), raw heads | 423.10 | 583.47 | 830.42 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 272.02 | 378.64 | 554.10 |
| MAE ($), raw heads | 292.91 | 396.03 | 573.32 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0031 | -0.0038 | -0.0095 |
| skill vs zero, raw heads | -0.1171 | -0.0707 | -0.0572 |
| EV, served | -0.0017 | -0.0023 | -0.0042 |
| EV, raw heads | -0.0444 | -0.0434 | -0.0317 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0661 | -0.0488 | -0.0139 |
| corr, Spearman, raw heads | -0.0624 | -0.0380 | -0.0125 |
| mean predicted ($), served | 7.44 | 8.78 | 29.79 |
| mean predicted ($), raw heads | 97.58 | 73.76 | 93.37 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9911 | 0.7624 | 0.6912 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0762 | 0.1191 | 0.3190 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.43 | 284.81 | 420.04 |
| CRPSS vs constant variance | 0.0200 | 0.0335 | 0.0557 |
| NLL | 7.4543 | 7.8017 | 8.1514 |
| PIT KS | 0.0384 | 0.0539 | 0.0766 |
| var / err^2 Spearman | 0.0654 | 0.0694 | 0.0890 |
| coverage of the 90% interval | 0.9012 | 0.9018 | 0.8615 |
| width of the 90% interval ($) | 1207.47 | 1756.06 | 2385.45 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0047 | [-0.0241, 0.0382] | NOISE |
| h1 | -0.0063 | [-0.0287, 0.0151] | NOISE |
| h2 | 0.0107 | [-0.0271, 0.0488] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.076 / h1 0.119 / h2 0.319) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.3385 | 0.6627 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.7379 | 0.9277 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.2224 | 0.6175 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.8302 | 0.7184 | 0.6718 | 0.5125 |
| expected if the two signs were independent | 0.8252 | 0.7202 | 0.6478 | 0.4750 |

- P(up) unanimity (all three horizons call the same side): 0.7063

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0210 vs 0.0025 (-0.0235): does not beat, noise (boot z -0.75) | -0.0428 vs 0.0205 (-0.0633): does not beat, significantly worse (boot z -2.46) | -0.0138 vs -0.0138 (+0.0000): beats, noise (boot z +0.00) |
| logreg_lags | direction/auc | 0.4963 vs 0.5320 (-0.0357): does not beat, noise (boot z -1.55) | 0.4823 vs 0.5251 (-0.0427): does not beat, significantly worse (boot z -1.97) | 0.5025 vs 0.5095 (-0.0070): does not beat, noise (boot z -0.77) |
| logreg_lags | direction/brier | 0.2596 vs 0.2536 (-0.0060): does not beat, significantly worse (DM z -3.11) | 0.2536 vs 0.2575 (+0.0039): beats, noise (DM z +1.53) | 0.2591 vs 0.2702 (+0.0111): beats (DM z +3.33) |
| logreg_lags | direction/ece_pos | 0.0866 vs 0.0663 (-0.0203): does not beat, significantly worse (boot z -2.55) | 0.0609 vs 0.0812 (+0.0203): beats (boot z +3.78) | 0.0861 vs 0.1230 (+0.0369): beats (boot z +7.16) |
| logreg_lags | direction/acc | 0.4803 vs 0.4836 (-0.0033): does not beat, noise (DM z -0.34) | 0.4739 vs 0.4911 (-0.0172): does not beat, significantly worse (DM z -2.39) | 0.4773 vs 0.4756 (+0.0017): beats, noise (DM z +0.19) |
| logreg_lags | direction/bal_acc | 0.4924 vs 0.5004 (-0.0080): does not beat, noise (boot z -0.92) | 0.4881 vs 0.5055 (-0.0174): does not beat, significantly worse (boot z -2.47) | 0.4957 vs 0.4971 (-0.0015): does not beat, noise (boot z -0.24) |
| class_prior | direction/mcc | -0.0210 vs 0.0000 (-0.0210): does not beat, noise (boot z -1.00) | -0.0428 vs 0.0000 (-0.0428): does not beat, significantly worse (boot z -3.07) | -0.0138 vs 0.0000 (-0.0138): does not beat, noise (boot z -0.72) |
| class_prior | direction/auc | 0.4963 vs 0.5000 (-0.0037): does not beat, noise (boot z -0.26) | 0.4823 vs 0.5000 (-0.0177): does not beat, significantly worse (boot z -2.22) | 0.5025 vs 0.5000 (+0.0025): beats, noise (boot z +0.16) |
| class_prior | direction/brier | 0.2596 vs 0.2532 (-0.0064): does not beat, significantly worse (DM z -3.99) | 0.2536 vs 0.2533 (-0.0003): does not beat, noise (DM z -0.38) | 0.2591 vs 0.2573 (-0.0018): does not beat, noise (DM z -0.93) |
| class_prior | direction/ece_pos | 0.0866 vs 0.0593 (-0.0273): does not beat, significantly worse (boot z -3.50) | 0.0609 vs 0.0603 (-0.0005): does not beat, noise (boot z -0.11) | 0.0861 vs 0.0886 (+0.0025): beats, noise (boot z +0.46) |
| class_prior | direction/acc | 0.4803 vs 0.4824 (-0.0021): does not beat, noise (DM z -0.22) | 0.4739 vs 0.4829 (-0.0090): does not beat, noise (DM z -1.59) | 0.4773 vs 0.4763 (+0.0010): beats, noise (DM z +0.09) |
| class_prior | direction/bal_acc | 0.4924 vs 0.5000 (-0.0076): does not beat, noise (boot z -1.00) | 0.4881 vs 0.5000 (-0.0119): does not beat, significantly worse (boot z -3.04) | 0.4957 vs 0.5000 (-0.0043): does not beat, noise (boot z -0.72) |
| zero_delta | delta/rmse | 400.92 vs 400.31 (-0.61, -0.15%): does not beat, significantly worse (DM z -2.20) | 564.94 vs 563.88 (-1.06, -0.19%): does not beat, significantly worse (DM z -2.17) | 811.45 vs 807.63 (-3.82, -0.47%): does not beat, noise (DM z -1.57) |
| zero_delta | delta/mae | 272.02 vs 271.46 (-0.56, -0.21%): does not beat, significantly worse (DM z -2.58) | 378.64 vs 377.88 (-0.76, -0.20%): does not beat, noise (DM z -1.87) | 554.10 vs 550.98 (-3.12, -0.57%): does not beat, noise (DM z -1.62) |
| mean_delta | delta/rmse | 400.92 vs 405.60 (+4.68, +1.15%): beats (DM z +2.92) | 564.94 vs 578.56 (+13.62, +2.35%): beats (DM z +2.80) | 811.45 vs 847.03 (+35.59, +4.20%): beats (DM z +2.83) |
| mean_delta | delta/mae | 272.02 vs 277.50 (+5.48, +1.97%): beats (DM z +4.08) | 378.64 vs 393.77 (+15.13, +3.84%): beats (DM z +3.81) | 554.10 vs 595.01 (+40.91, +6.88%): beats (DM z +3.87) |
| const_var | variance/crps | 202.43 vs 206.55 (+4.12, +2.00%): beats (DM z +4.95) | 284.81 vs 294.69 (+9.88, +3.35%): beats (DM z +3.89) | 420.04 vs 444.83 (+24.79, +5.57%): beats (DM z +3.60) |
| const_var | variance/nll | 7.4543 vs 7.4448 (-0.0095): does not beat, noise (DM z -0.68) | 7.8017 vs 7.8022 (+0.0005): beats, noise (DM z +0.02) | 8.1514 vs 8.2104 (+0.0590): beats (DM z +2.61) |
| const_var | variance/pit_ks | 0.0384 vs 0.1064 (+0.0681): beats (boot z +18.64) | 0.0539 vs 0.1404 (+0.0865): beats (boot z +13.78) | 0.0766 vs 0.1810 (+0.1045): beats (boot z +23.77) |
| const_var | variance/corr_var_err2_spearman | 0.0654 vs 0.0000 (+0.0654): beats (boot z +4.00) | 0.0694 vs 0.0000 (+0.0694): beats (boot z +2.98) | 0.0890 vs 0.0000 (+0.0890): beats (boot z +4.02) |

## Backtest (costs included)

- n_trades: 1402
- total_return: -0.9759
- sharpe_net: -134.0439
- sharpe_gross: -4.7430
- sortino: -149.6858
- max_drawdown: 0.9759
- hit_rate: 0.0599
- hit_rate_gross: 0.4408
- profit_factor: 0.0315
- avg_hold_bars: 11.5100
- exposure: 0.3736
- turnover: 725.2518
- fees_paid: 7252.8556
- traded_notional: 7252855.5542
- breakeven_cost_bps: -0.9108
- gross_edge_per_trade_bps: -0.5209
- costs_paid: 9428.7122
- gross_pnl: -330.2911
- net_pnl: -9759.0034

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7283, long_above 0.5963, short_below 0.5049, median 0.5460. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.59% | -134.04 | +97.59% | 1402 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -97.59% .. -96.81%) | -97.22% | -140.20 | | |

The random null enters at the strategy's rate (0.0518 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 5% of its seeds on net return, 95% on net Sharpe and 7% on gross return.
