# Evaluation report - dev split - run `20260930T234243Z-fb840fd-f1fd90cf-control__f-35__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 17597 | 18683 | 19416 |
| n_eff of the scored moves (n scored // bars ahead) | 1759 | 1245 | 970 |
| true up-rate | 0.5230 | 0.5219 | 0.5231 |
| calls up (predicted up-rate) | 0.5484 | 0.6542 | 0.5855 |
| accuracy | 0.4923 | 0.5136 | 0.5007 |
| balanced accuracy | 0.4901 | 0.5069 | 0.4968 |
| precision (up) | 0.5139 | 0.5272 | 0.5203 |
| recall / sensitivity (up) | 0.5390 | 0.6608 | 0.5824 |
| specificity (down) | 0.4411 | 0.3530 | 0.4111 |
| F1 (up) | 0.5261 | 0.5864 | 0.5496 |
| MCC | -0.0200 | 0.0144 | -0.0066 |
| AUC | 0.4914 | 0.5132 | 0.4932 |
| Brier | 0.2575 | 0.2510 | 0.2572 |
| ECE (positive class) | 0.0636 | 0.0186 | 0.0527 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0230 | 0.0219 | 0.0231 |
| TP / FP / TN / FN | 4960 / 4691 / 3703 / 4243 | 6443 / 5779 / 3153 / 3308 | 5915 / 5453 / 3807 / 4241 |
| Gaussian readout: calls up | 0.3542 | 0.3328 | 0.3600 |
| Gaussian readout: MCC | 0.0128 | 0.0359 | 0.0402 |
| Gaussian readout: AUC | 0.5124 | 0.5189 | 0.5177 |
| Gaussian readout: Brier | 0.2501 | 0.2507 | 0.2516 |
| Gaussian readout: ECE | 0.0272 | 0.0302 | 0.0361 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 135.31 | 167.58 | 192.93 |
| RMSE ($), raw heads | 136.25 | 174.46 | 197.16 |
| RMSE ($), zero prediction | 135.21 | 166.70 | 192.12 |
| MAE ($), served | 82.09 | 101.10 | 117.46 |
| MAE ($), raw heads | 82.55 | 103.87 | 119.95 |
| MAE ($), zero prediction | 82.02 | 100.58 | 116.69 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0016 | -0.0106 | -0.0084 |
| skill vs zero, raw heads | -0.0155 | -0.0953 | -0.0532 |
| EV, served | -0.0013 | -0.0100 | -0.0083 |
| EV, raw heads | -0.0148 | -0.0937 | -0.0530 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0199 | 0.0194 | 0.0263 |
| corr, Spearman, raw heads | 0.0207 | 0.0306 | 0.0265 |
| mean predicted ($), served | -0.49 | -1.00 | -0.21 |
| mean predicted ($), raw heads | -1.14 | -2.69 | -0.45 |
| mean realised ($) | 5.17 | 7.76 | 10.35 |
| share predicted up, raw heads | 0.3372 | 0.3147 | 0.3449 |
| share realised up | 0.5125 | 0.5147 | 0.5157 |
| shrink beta (served = beta x raw, fit on cal) | 0.4268 | 0.3722 | 0.4693 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 60.29 | 74.67 | 86.88 |
| CRPSS vs constant variance | 0.0461 | 0.0385 | 0.0358 |
| NLL | 6.0639 | 6.3196 | 6.4898 |
| PIT KS | 0.0256 | 0.0322 | 0.0390 |
| var / err^2 Spearman | 0.4307 | 0.4120 | 0.4135 |
| coverage of the 90% interval | 0.9065 | 0.9066 | 0.9075 |
| width of the 90% interval ($) | 364.52 | 452.65 | 527.54 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0005 | [-0.0217, 0.0182] | NOISE |
| h1 | 0.0165 | [-0.0055, 0.0389] | NOISE |
| h2 | -0.0099 | [-0.0328, 0.0115] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.427 / h1 0.372 / h2 0.469) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8037 | 0.7644 | 0.6121 |
| abs(d h1) <= abs(d h2) | 0.6221 | 0.7640 | 0.5941 |
| full chain h0 <= h1 <= h2 | 0.4556 | 0.5585 | 0.3332 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6046 | 0.4924 | 0.5946 | 0.2188 |
| expected if the two signs were independent | 0.4906 | 0.4345 | 0.4795 | 0.1402 |

- P(up) unanimity (all three horizons call the same side): 0.3754

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0200 vs 0.0198 (-0.0398): does not beat, noise (boot z -1.77) | 0.0144 vs 0.0222 (-0.0077): does not beat, noise (boot z -0.40) | -0.0066 vs 0.0218 (-0.0284): does not beat, noise (boot z -1.18) |
| logreg_lags | direction/auc | 0.4914 vs 0.5173 (-0.0259): does not beat, significantly worse (boot z -2.11) | 0.5132 vs 0.5197 (-0.0065): does not beat, noise (boot z -0.57) | 0.4932 vs 0.5204 (-0.0272): does not beat, significantly worse (boot z -2.00) |
| logreg_lags | direction/brier | 0.2575 vs 0.2500 (-0.0075): does not beat, significantly worse (DM z -4.81) | 0.2510 vs 0.2501 (-0.0010): does not beat, noise (DM z -1.05) | 0.2572 vs 0.2500 (-0.0072): does not beat, significantly worse (DM z -4.16) |
| logreg_lags | direction/ece_pos | 0.0636 vs 0.0150 (-0.0486): does not beat, significantly worse (boot z -4.68) | 0.0186 vs 0.0140 (-0.0045): does not beat, noise (boot z -0.84) | 0.0527 vs 0.0146 (-0.0381): does not beat, significantly worse (boot z -3.66) |
| logreg_lags | direction/acc | 0.4923 vs 0.5172 (-0.0249): does not beat, significantly worse (DM z -2.37) | 0.5136 vs 0.5176 (-0.0040): does not beat, noise (DM z -0.42) | 0.5007 vs 0.5185 (-0.0178): does not beat, noise (DM z -1.52) |
| logreg_lags | direction/bal_acc | 0.4901 vs 0.5093 (-0.0193): does not beat, noise (boot z -1.79) | 0.5069 vs 0.5105 (-0.0036): does not beat, noise (boot z -0.39) | 0.4968 vs 0.5102 (-0.0134): does not beat, noise (boot z -1.18) |
| class_prior | direction/mcc | -0.0200 vs 0.0000 (-0.0200): does not beat, noise (boot z -1.53) | 0.0144 vs 0.0000 (+0.0144): beats, noise (boot z +0.88) | -0.0066 vs 0.0000 (-0.0066): does not beat, noise (boot z -0.45) |
| class_prior | direction/auc | 0.4914 vs 0.5000 (-0.0086): does not beat, noise (boot z -1.04) | 0.5132 vs 0.5000 (+0.0132): beats, noise (boot z +1.28) | 0.4932 vs 0.5000 (-0.0068): does not beat, noise (boot z -0.65) |
| class_prior | direction/brier | 0.2575 vs 0.2496 (-0.0079): does not beat, significantly worse (DM z -5.29) | 0.2510 vs 0.2496 (-0.0014): does not beat, noise (DM z -1.95) | 0.2572 vs 0.2496 (-0.0076): does not beat, significantly worse (DM z -4.47) |
| class_prior | direction/ece_pos | 0.0636 vs 0.0122 (-0.0513): does not beat, significantly worse (boot z -4.62) | 0.0186 vs 0.0103 (-0.0083): does not beat, noise (boot z -1.45) | 0.0527 vs 0.0109 (-0.0418): does not beat, significantly worse (boot z -3.54) |
| class_prior | direction/acc | 0.4923 vs 0.5230 (-0.0307): does not beat, significantly worse (DM z -2.70) | 0.5136 vs 0.5219 (-0.0083): does not beat, noise (DM z -0.76) | 0.5007 vs 0.5231 (-0.0224): does not beat, noise (DM z -1.65) |
| class_prior | direction/bal_acc | 0.4901 vs 0.5000 (-0.0099): does not beat, noise (boot z -1.53) | 0.5069 vs 0.5000 (+0.0069): beats, noise (boot z +0.88) | 0.4968 vs 0.5000 (-0.0032): does not beat, noise (boot z -0.45) |
| zero_delta | delta/rmse | 135.31 vs 135.21 (-0.11, -0.08%): does not beat, noise (DM z -0.38) | 167.58 vs 166.70 (-0.88, -0.53%): does not beat, noise (DM z -0.99) | 192.93 vs 192.12 (-0.80, -0.42%): does not beat, noise (DM z -0.57) |
| zero_delta | delta/mae | 82.09 vs 82.02 (-0.07, -0.09%): does not beat, noise (DM z -0.68) | 101.10 vs 100.58 (-0.52, -0.51%): does not beat, noise (DM z -1.75) | 117.46 vs 116.69 (-0.77, -0.66%): does not beat, noise (DM z -1.89) |
| mean_delta | delta/rmse | 135.31 vs 135.18 (-0.13, -0.10%): does not beat, noise (DM z -0.47) | 167.58 vs 166.65 (-0.93, -0.56%): does not beat, noise (DM z -1.04) | 192.93 vs 192.04 (-0.88, -0.46%): does not beat, noise (DM z -0.62) |
| mean_delta | delta/mae | 82.09 vs 82.00 (-0.09, -0.11%): does not beat, noise (DM z -0.84) | 101.10 vs 100.55 (-0.55, -0.55%): does not beat, noise (DM z -1.85) | 117.46 vs 116.65 (-0.81, -0.70%): does not beat, significantly worse (DM z -1.99) |
| const_var | variance/crps | 60.29 vs 63.20 (+2.91, +4.61%): beats (DM z +11.62) | 74.67 vs 77.66 (+2.99, +3.85%): beats (DM z +7.63) | 86.88 vs 90.10 (+3.22, +3.58%): beats (DM z +6.80) |
| const_var | variance/nll | 6.0639 vs 6.6403 (+0.5764): beats (DM z +3.59) | 6.3196 vs 6.8650 (+0.5454): beats (DM z +2.92) | 6.4898 vs 7.0192 (+0.5295): beats (DM z +2.89) |
| const_var | variance/pit_ks | 0.0256 vs 0.0393 (+0.0137): beats (boot z +3.35) | 0.0322 vs 0.0425 (+0.0103): beats (boot z +2.30) | 0.0390 vs 0.0458 (+0.0069): beats, noise (boot z +1.43) |
| const_var | variance/corr_var_err2_spearman | 0.4307 vs 0.0000 (+0.4307): beats (boot z +21.69) | 0.4120 vs 0.0000 (+0.4120): beats (boot z +19.14) | 0.4135 vs 0.0000 (+0.4135): beats (boot z +17.79) |

## Backtest (costs included)

- n_trades: 1654
- total_return: -0.0438
- sharpe_net: -2.0024
- sharpe_gross: -2.0024
- sortino: -2.7716
- max_drawdown: 0.1336
- hit_rate: 0.5230
- hit_rate_gross: 0.5230
- profit_factor: 0.9645
- avg_hold_bars: 7.1886
- exposure: 0.4875
- turnover: 3355.3433
- fees_paid: 0.0000
- traded_notional: 33555088.6498
- breakeven_cost_bps: -0.2612
- gross_edge_per_trade_bps: -0.2464
- costs_paid: 0.0000
- gross_pnl: -438.2817
- net_pnl: -438.2817

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -35 (TimeSeriesSplit fold 6, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-02-14T08:48:00 .. 2024-03-02T07:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4706, long_above 0.5356, short_below 0.4675, median 0.5016. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -4.38% | -2.00 | +13.36% | 1654 |
| buy and hold | +25.30% | +9.18 | +6.96% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -12.67% .. +12.41%) | +0.23% | +0.16 | | |

The random null enters at the strategy's rate (0.1323 per flat bar), holds 7 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 23% of its seeds on net return, 29% on net Sharpe and 23% on gross return.
