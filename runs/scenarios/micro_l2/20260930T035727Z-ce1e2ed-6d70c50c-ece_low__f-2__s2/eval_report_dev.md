# Evaluation report - dev split - run `20260930T035727Z-ce1e2ed-6d70c50c-ece_low__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.7897 | 0.6920 | 0.6764 |
| accuracy | 0.4774 | 0.5077 | 0.4867 |
| balanced accuracy | 0.4876 | 0.5142 | 0.4951 |
| precision (up) | 0.4746 | 0.4932 | 0.4727 |
| recall / sensitivity (up) | 0.7769 | 0.7068 | 0.6713 |
| specificity (down) | 0.1983 | 0.3217 | 0.3189 |
| F1 (up) | 0.5892 | 0.5810 | 0.5548 |
| MCC | -0.0304 | 0.0308 | -0.0105 |
| AUC | 0.4942 | 0.5262 | 0.4936 |
| Brier | 0.2572 | 0.2575 | 0.2557 |
| ECE (positive class) | 0.0792 | 0.0672 | 0.0616 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 13602 / 15059 / 3725 / 3906 | 12904 / 13259 / 6289 / 5354 | 12612 / 14069 / 6587 / 6176 |
| Gaussian readout: calls up | 0.8783 | 0.9199 | 0.7410 |
| Gaussian readout: MCC | -0.0022 | 0.0115 | 0.0250 |
| Gaussian readout: AUC | 0.4876 | 0.5047 | 0.5107 |
| Gaussian readout: Brier | 0.2502 | 0.2504 | 0.2519 |
| Gaussian readout: ECE | 0.0208 | 0.0260 | 0.0461 |

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.47 | 564.95 | 811.53 |
| RMSE ($), raw heads | 411.13 | 576.85 | 820.74 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.60 | 378.61 | 554.26 |
| MAE ($), raw heads | 281.05 | 389.11 | 563.50 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0008 | -0.0038 | -0.0097 |
| skill vs zero, raw heads | -0.0548 | -0.0465 | -0.0327 |
| EV, served | -0.0004 | -0.0012 | -0.0028 |
| EV, raw heads | -0.0235 | -0.0182 | -0.0136 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0309 | -0.0175 | 0.0040 |
| corr, Spearman, raw heads | -0.0297 | -0.0036 | 0.0027 |
| mean predicted ($), served | 2.78 | 13.88 | 36.57 |
| mean predicted ($), raw heads | 60.68 | 75.45 | 76.87 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.8691 | 0.9160 | 0.7380 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0458 | 0.1840 | 0.4758 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 203.31 | 294.27 | 419.02 |
| CRPSS vs constant variance | 0.0157 | 0.0014 | 0.0580 |
| NLL | 7.3881 | 7.7662 | 8.1868 |
| PIT KS | 0.0689 | 0.1199 | 0.0627 |
| var / err^2 Spearman | 0.2062 | 0.1680 | 0.1331 |
| coverage of the 90% interval | 0.9029 | 0.9012 | 0.8638 |
| width of the 90% interval ($) | 1213.24 | 1753.04 | 2403.29 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0116 | [-0.0157, 0.0375] | NOISE |
| h1 | 0.0025 | [-0.0293, 0.0380] | NOISE |
| h2 | -0.0137 | [-0.0463, 0.0184] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.046 / h1 0.184 / h2 0.476) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5855 | 0.9348 | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.7042 | 0.9376 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.3387 | 0.8731 | 0.3608 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7002 | 0.7004 | 0.4922 | 0.3074 |
| expected if the two signs were independent | 0.7213 | 0.6526 | 0.5843 | 0.3027 |

- P(up) unanimity (all three horizons call the same side): 0.4187

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0304 vs 0.0025 (-0.0329): does not beat, noise (boot z -1.23) | 0.0308 vs 0.0205 (+0.0103): beats, noise (boot z +0.41) | -0.0105 vs -0.0138 (+0.0033): beats, noise (boot z +0.11) |
| logreg_lags | direction/auc | 0.4942 vs 0.5320 (-0.0378): does not beat, noise (boot z -1.66) | 0.5262 vs 0.5251 (+0.0012): beats, noise (boot z +0.12) | 0.4936 vs 0.5095 (-0.0159): does not beat, noise (boot z -0.55) |
| logreg_lags | direction/brier | 0.2572 vs 0.2536 (-0.0036): does not beat, significantly worse (DM z -2.28) | 0.2575 vs 0.2575 (-0.0001): does not beat, noise (DM z -0.04) | 0.2557 vs 0.2702 (+0.0145): beats (DM z +2.04) |
| logreg_lags | direction/ece_pos | 0.0792 vs 0.0663 (-0.0129): does not beat, noise (boot z -1.28) | 0.0672 vs 0.0812 (+0.0140): beats (boot z +2.56) | 0.0616 vs 0.1230 (+0.0615): beats (boot z +4.75) |
| logreg_lags | direction/acc | 0.4774 vs 0.4836 (-0.0062): does not beat, noise (DM z -0.63) | 0.5077 vs 0.4911 (+0.0166): beats, noise (DM z +1.01) | 0.4867 vs 0.4756 (+0.0112): beats, noise (DM z +0.49) |
| logreg_lags | direction/bal_acc | 0.4876 vs 0.5004 (-0.0128): does not beat, noise (boot z -1.61) | 0.5142 vs 0.5055 (+0.0087): beats, noise (boot z +0.88) | 0.4951 vs 0.4971 (-0.0021): does not beat, noise (boot z -0.18) |
| class_prior | direction/mcc | -0.0304 vs 0.0000 (-0.0304): does not beat, noise (boot z -1.67) | 0.0308 vs 0.0000 (+0.0308): beats, noise (boot z +1.48) | -0.0105 vs 0.0000 (-0.0105): does not beat, noise (boot z -0.44) |
| class_prior | direction/auc | 0.4942 vs 0.5000 (-0.0058): does not beat, noise (boot z -0.57) | 0.5262 vs 0.5000 (+0.0262): beats, noise (boot z +1.79) | 0.4936 vs 0.5000 (-0.0064): does not beat, noise (boot z -0.39) |
| class_prior | direction/brier | 0.2572 vs 0.2532 (-0.0040): does not beat, significantly worse (DM z -3.70) | 0.2575 vs 0.2533 (-0.0042): does not beat, noise (DM z -1.52) | 0.2557 vs 0.2573 (+0.0016): beats, noise (DM z +0.46) |
| class_prior | direction/ece_pos | 0.0792 vs 0.0593 (-0.0198): does not beat, significantly worse (boot z -1.96) | 0.0672 vs 0.0603 (-0.0069): does not beat, noise (boot z -1.10) | 0.0616 vs 0.0886 (+0.0271): beats (boot z +2.08) |
| class_prior | direction/acc | 0.4774 vs 0.4824 (-0.0050): does not beat, noise (DM z -0.51) | 0.5077 vs 0.4829 (+0.0247): beats, noise (DM z +1.37) | 0.4867 vs 0.4763 (+0.0104): beats, noise (DM z +0.43) |
| class_prior | direction/bal_acc | 0.4876 vs 0.5000 (-0.0124): does not beat, noise (boot z -1.67) | 0.5142 vs 0.5000 (+0.0142): beats, noise (boot z +1.48) | 0.4951 vs 0.5000 (-0.0049): does not beat, noise (boot z -0.44) |
| zero_delta | delta/rmse | 400.47 vs 400.31 (-0.16, -0.04%): does not beat, noise (DM z -1.29) | 564.95 vs 563.88 (-1.06, -0.19%): does not beat, noise (DM z -1.30) | 811.53 vs 807.63 (-3.90, -0.48%): does not beat, noise (DM z -1.14) |
| zero_delta | delta/mae | 271.60 vs 271.46 (-0.14, -0.05%): does not beat, noise (DM z -1.57) | 378.61 vs 377.88 (-0.74, -0.19%): does not beat, noise (DM z -1.21) | 554.26 vs 550.98 (-3.28, -0.60%): does not beat, noise (DM z -1.31) |
| mean_delta | delta/rmse | 400.47 vs 405.60 (+5.13, +1.27%): beats (DM z +2.94) | 564.95 vs 578.56 (+13.62, +2.35%): beats (DM z +3.00) | 811.53 vs 847.03 (+35.50, +4.19%): beats (DM z +3.03) |
| mean_delta | delta/mae | 271.60 vs 277.50 (+5.90, +2.13%): beats (DM z +4.04) | 378.61 vs 393.77 (+15.16, +3.85%): beats (DM z +4.02) | 554.26 vs 595.01 (+40.75, +6.85%): beats (DM z +3.96) |
| const_var | variance/crps | 203.31 vs 206.55 (+3.24, +1.57%): beats (DM z +3.36) | 294.27 vs 294.69 (+0.42, +0.14%): beats, noise (DM z +0.15) | 419.02 vs 444.83 (+25.80, +5.80%): beats (DM z +3.95) |
| const_var | variance/nll | 7.3881 vs 7.4448 (+0.0567): beats (DM z +3.38) | 7.7662 vs 7.8022 (+0.0360): beats, noise (DM z +0.90) | 8.1868 vs 8.2104 (+0.0236): beats, noise (DM z +0.76) |
| const_var | variance/pit_ks | 0.0689 vs 0.1064 (+0.0375): beats (boot z +5.31) | 0.1199 vs 0.1404 (+0.0205): beats (boot z +3.07) | 0.0627 vs 0.1810 (+0.1184): beats (boot z +25.16) |
| const_var | variance/corr_var_err2_spearman | 0.2062 vs 0.0000 (+0.2062): beats (boot z +10.63) | 0.1680 vs 0.0000 (+0.1680): beats (boot z +7.08) | 0.1331 vs 0.0000 (+0.1331): beats (boot z +5.29) |

## Backtest (costs included)

- n_trades: 1087
- total_return: -0.9385
- sharpe_net: -105.3083
- sharpe_gross: -0.0703
- sortino: -121.8703
- max_drawdown: 0.9385
- hit_rate: 0.0727
- hit_rate_gross: 0.5271
- profit_factor: 0.0274
- avg_hold_bars: 12.5639
- exposure: 0.3161
- turnover: 721.1685
- fees_paid: 7211.6549
- traded_notional: 7211654.9163
- breakeven_cost_bps: -0.0281
- gross_edge_per_trade_bps: 0.3965
- costs_paid: 9375.1514
- gross_pnl: -10.1307
- net_pnl: -9385.2821

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.267, long_above 0.5772, short_below 0.4842, median 0.5394. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -93.85% | -105.31 | +93.85% | 1087 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -94.72% .. -92.97%) | -93.87% | -120.97 | | |

The random null enters at the strategy's rate (0.0368 per flat bar), holds 13 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 52% of its seeds on net return, 100% on net Sharpe and 55% on gross return.
