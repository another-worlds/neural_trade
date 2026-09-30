# Evaluation report - dev split - run `20260930T094257Z-dce15ed-e3669618-default__f-2__s0`

n = 46544 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 28800 | 32081 | 33877 |
| n_eff of the scored moves (n scored // bars ahead) | 2880 | 2138 | 1693 |
| true up-rate | 0.4906 | 0.4906 | 0.4890 |
| calls up (predicted up-rate) | 0.4653 | 0.7152 | 0.6899 |
| accuracy | 0.5193 | 0.5043 | 0.5095 |
| balanced accuracy | 0.5187 | 0.5084 | 0.5137 |
| precision (up) | 0.5106 | 0.4965 | 0.4989 |
| recall / sensitivity (up) | 0.4844 | 0.7237 | 0.7040 |
| specificity (down) | 0.5530 | 0.2930 | 0.3235 |
| F1 (up) | 0.4971 | 0.5889 | 0.5840 |
| MCC | 0.0374 | 0.0186 | 0.0297 |
| AUC | 0.5196 | 0.5184 | 0.5245 |
| Brier | 0.2502 | 0.2499 | 0.2503 |
| ECE (positive class) | 0.0111 | 0.0177 | 0.0266 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0094 | 0.0094 | 0.0110 |
| TP / FP / TN / FN | 6843 / 6559 / 8113 / 7285 | 11391 / 11553 / 4789 / 4348 | 11661 / 11711 / 5601 / 4904 |
| Gaussian readout: calls up | 0.8649 | 0.7865 | 0.7533 |
| Gaussian readout: MCC | 0.0072 | 0.0193 | 0.0011 |
| Gaussian readout: AUC | 0.5031 | 0.5111 | 0.5071 |
| Gaussian readout: Brier | 0.2502 | 0.2499 | 0.2500 |
| Gaussian readout: ECE | 0.0162 | 0.0127 | 0.0121 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 164.27 | 198.27 | 228.32 |
| RMSE ($), raw heads | 164.44 | 203.18 | 228.84 |
| RMSE ($), zero prediction | 164.21 | 198.38 | 228.36 |
| MAE ($), served | 112.69 | 137.97 | 158.98 |
| MAE ($), raw heads | 112.83 | 139.26 | 159.26 |
| MAE ($), zero prediction | 112.64 | 138.04 | 158.99 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0007 | 0.0011 | 0.0003 |
| skill vs zero, raw heads | -0.0027 | -0.0489 | -0.0042 |
| EV, served | -0.0004 | 0.0013 | 0.0004 |
| EV, raw heads | -0.0017 | -0.0463 | -0.0022 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0039 | 0.0363 | 0.0345 |
| corr, Spearman, raw heads | 0.0077 | 0.0183 | 0.0066 |
| mean predicted ($), served | 1.88 | 1.43 | 0.55 |
| mean predicted ($), raw heads | 4.32 | 8.56 | 8.07 |
| mean realised ($) | -1.18 | -1.76 | -2.32 |
| share predicted up, raw heads | 0.8642 | 0.8001 | 0.7438 |
| share realised up | 0.4897 | 0.4876 | 0.4888 |
| shrink beta (served = beta x raw, fit on cal) | 0.4356 | 0.1666 | 0.0684 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 82.08 | 100.22 | 115.52 |
| CRPSS vs constant variance | 0.0457 | 0.0437 | 0.0435 |
| NLL | 6.4015 | 6.5935 | 6.7376 |
| PIT KS | 0.0446 | 0.0433 | 0.0441 |
| var / err^2 Spearman | 0.3359 | 0.3300 | 0.3348 |
| coverage of the 90% interval | 0.9072 | 0.9081 | 0.9079 |
| width of the 90% interval ($) | 519.12 | 635.23 | 729.30 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0118 | [-0.0049, 0.0287] | NOISE |
| h1 | 0.0130 | [-0.0061, 0.0315] | NOISE |
| h2 | 0.0203 | [-0.0015, 0.0429] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.436 / h1 0.167 / h2 0.068) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5989 | 0.3159 | 0.6180 |
| abs(d h1) <= abs(d h2) | 0.5726 | 0.3179 | 0.5923 |
| full chain h0 <= h1 <= h2 | 0.2512 | 0.0347 | 0.3360 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4830 | 0.7418 | 0.5326 | 0.2212 |
| expected if the two signs were independent | 0.4671 | 0.6382 | 0.5857 | 0.1900 |

- P(up) unanimity (all three horizons call the same side): 0.3583

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0374 vs 0.0344 (+0.0030): beats, noise (boot z +0.19) | 0.0186 vs 0.0378 (-0.0192): does not beat, noise (boot z -1.33) | 0.0297 vs 0.0411 (-0.0114): does not beat, noise (boot z -0.99) |
| logreg_lags | direction/auc | 0.5196 vs 0.5243 (-0.0047): does not beat, noise (boot z -0.46) | 0.5184 vs 0.5272 (-0.0088): does not beat, noise (boot z -1.02) | 0.5245 vs 0.5314 (-0.0069): does not beat, noise (boot z -0.95) |
| logreg_lags | direction/brier | 0.2502 vs 0.2497 (-0.0005): does not beat, noise (DM z -1.12) | 0.2499 vs 0.2497 (-0.0001): does not beat, noise (DM z -0.46) | 0.2503 vs 0.2497 (-0.0006): does not beat, noise (DM z -1.36) |
| logreg_lags | direction/ece_pos | 0.0111 vs 0.0112 (+0.0001): beats, noise (boot z +0.03) | 0.0177 vs 0.0116 (-0.0061): does not beat, noise (boot z -1.01) | 0.0266 vs 0.0137 (-0.0128): does not beat, noise (boot z -1.77) |
| logreg_lags | direction/acc | 0.5193 vs 0.5165 (+0.0028): beats, noise (DM z +0.36) | 0.5043 vs 0.5178 (-0.0134): does not beat, noise (DM z -1.85) | 0.5095 vs 0.5191 (-0.0096): does not beat, noise (DM z -1.50) |
| logreg_lags | direction/bal_acc | 0.5187 vs 0.5171 (+0.0015): beats, noise (boot z +0.19) | 0.5084 vs 0.5188 (-0.0104): does not beat, noise (boot z -1.49) | 0.5137 vs 0.5204 (-0.0067): does not beat, noise (boot z -1.17) |
| class_prior | direction/mcc | 0.0374 vs 0.0000 (+0.0374): beats (boot z +3.33) | 0.0186 vs 0.0000 (+0.0186): beats, noise (boot z +1.53) | 0.0297 vs 0.0000 (+0.0297): beats (boot z +2.20) |
| class_prior | direction/auc | 0.5196 vs 0.5000 (+0.0196): beats (boot z +2.69) | 0.5184 vs 0.5000 (+0.0184): beats (boot z +2.15) | 0.5245 vs 0.5000 (+0.0245): beats (boot z +2.61) |
| class_prior | direction/brier | 0.2502 vs 0.2501 (-0.0002): does not beat, noise (DM z -0.39) | 0.2499 vs 0.2501 (+0.0002): beats, noise (DM z +1.03) | 0.2503 vs 0.2501 (-0.0002): does not beat, noise (DM z -0.37) |
| class_prior | direction/ece_pos | 0.0111 vs 0.0123 (+0.0012): beats, noise (boot z +0.32) | 0.0177 vs 0.0134 (-0.0043): does not beat, significantly worse (boot z -2.58) | 0.0266 vs 0.0164 (-0.0102): does not beat, significantly worse (boot z -3.66) |
| class_prior | direction/acc | 0.5193 vs 0.4906 (+0.0287): beats (DM z +2.91) | 0.5043 vs 0.4906 (+0.0137): beats, noise (DM z +1.84) | 0.5095 vs 0.4890 (+0.0206): beats (DM z +2.33) |
| class_prior | direction/bal_acc | 0.5187 vs 0.5000 (+0.0187): beats (boot z +3.33) | 0.5084 vs 0.5000 (+0.0084): beats, noise (boot z +1.53) | 0.5137 vs 0.5000 (+0.0137): beats (boot z +2.20) |
| zero_delta | delta/rmse | 164.27 vs 164.21 (-0.06, -0.03%): does not beat, noise (DM z -0.98) | 198.27 vs 198.38 (+0.11, +0.06%): beats, noise (DM z +0.51) | 228.32 vs 228.36 (+0.04, +0.02%): beats, noise (DM z +0.70) |
| zero_delta | delta/mae | 112.69 vs 112.64 (-0.05, -0.05%): does not beat, noise (DM z -1.55) | 137.97 vs 138.04 (+0.07, +0.05%): beats, noise (DM z +0.76) | 158.98 vs 158.99 (+0.02, +0.01%): beats, noise (DM z +0.66) |
| mean_delta | delta/rmse | 164.27 vs 164.22 (-0.05, -0.03%): does not beat, noise (DM z -0.92) | 198.27 vs 198.40 (+0.12, +0.06%): beats, noise (DM z +0.58) | 228.32 vs 228.38 (+0.06, +0.03%): beats, noise (DM z +1.18) |
| mean_delta | delta/mae | 112.69 vs 112.65 (-0.03, -0.03%): does not beat, noise (DM z -1.26) | 137.97 vs 138.07 (+0.10, +0.07%): beats, noise (DM z +1.12) | 158.98 vs 159.03 (+0.06, +0.03%): beats, noise (DM z +1.94) |
| const_var | variance/crps | 82.08 vs 86.01 (+3.93, +4.57%): beats (DM z +21.02) | 100.22 vs 104.80 (+4.58, +4.37%): beats (DM z +16.63) | 115.52 vs 120.77 (+5.25, +4.35%): beats (DM z +15.06) |
| const_var | variance/nll | 6.4015 vs 6.5314 (+0.1298): beats (DM z +8.07) | 6.5935 vs 6.7225 (+0.1290): beats (DM z +8.86) | 6.7376 vs 6.8636 (+0.1260): beats (DM z +8.45) |
| const_var | variance/pit_ks | 0.0446 vs 0.0915 (+0.0469): beats (boot z +12.37) | 0.0433 vs 0.0894 (+0.0460): beats (boot z +12.70) | 0.0441 vs 0.0910 (+0.0470): beats (boot z +12.42) |
| const_var | variance/corr_var_err2_spearman | 0.3359 vs 0.0000 (+0.3359): beats (boot z +24.51) | 0.3300 vs 0.0000 (+0.3300): beats (boot z +22.23) | 0.3348 vs 0.0000 (+0.3348): beats (boot z +21.75) |

## Backtest (costs included)

- n_trades: 1968
- total_return: -0.9930
- sharpe_net: -160.1782
- sharpe_gross: 7.5855
- sortino: -172.1878
- max_drawdown: 0.9930
- hit_rate: 0.0305
- hit_rate_gross: 0.5467
- profit_factor: 0.0109
- avg_hold_bars: 8.3186
- exposure: 0.3518
- turnover: 795.5989
- fees_paid: 7956.2802
- traded_notional: 7956280.2320
- breakeven_cost_bps: 1.0374
- gross_edge_per_trade_bps: 0.7962
- costs_paid: 10343.1643
- gross_pnl: 412.6918
- net_pnl: -9930.4725

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 13, 13 usable folds); this report scores the fold's out-of-sample block: 46544 sequences, 2025-07-27T08:10:00 .. 2025-08-28T15:53:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4708, long_above 0.5276, short_below 0.4766, median 0.5032. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.30% | -160.18 | +99.30% | 1968 |
| buy and hold | -5.10% | -1.62 | +12.57% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.52% .. -99.38%) | -99.44% | -178.87 | | |

The random null enters at the strategy's rate (0.0652 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 100% of its seeds on net return, 100% on net Sharpe and 99% on gross return.
