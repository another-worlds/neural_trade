# Evaluation report - dev split - run `20261006T094854Z-61014d0-e76a3912-linear_indicators__f-96__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10701 | 11457 | 11840 |
| n_eff of the scored moves (n scored // bars ahead) | 1070 | 763 | 592 |
| true up-rate | 0.5381 | 0.5412 | 0.5428 |
| calls up (predicted up-rate) | 0.5880 | 0.3991 | 0.6767 |
| accuracy | 0.5174 | 0.4748 | 0.5171 |
| balanced accuracy | 0.5108 | 0.4830 | 0.5020 |
| precision (up) | 0.5472 | 0.5200 | 0.5443 |
| recall / sensitivity (up) | 0.5980 | 0.3835 | 0.6785 |
| specificity (down) | 0.4236 | 0.5825 | 0.3255 |
| F1 (up) | 0.5715 | 0.4415 | 0.6041 |
| MCC | 0.0219 | -0.0346 | 0.0043 |
| AUC | 0.5138 | 0.4829 | 0.4990 |
| Brier | 0.2780 | 0.2743 | 0.2726 |
| ECE (positive class) | 0.1161 | 0.1248 | 0.0971 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0381 | 0.0412 | 0.0428 |
| TP / FP / TN / FN | 3443 / 2849 / 2094 / 2315 | 2378 / 2195 / 3062 / 3822 | 4361 / 3651 / 1762 / 2066 |
| Gaussian readout: calls up | 0.8634 | 0.8271 | 0.7041 |
| Gaussian readout: MCC | 0.0026 | 0.0023 | 0.0316 |
| Gaussian readout: AUC | 0.5132 | 0.5111 | 0.5236 |
| Gaussian readout: Brier | 0.2490 | 0.2491 | 0.2495 |
| Gaussian readout: ECE | 0.0251 | 0.0296 | 0.0389 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 53.77 | 65.60 | 76.26 |
| RMSE ($), raw heads | 53.84 | 65.58 | 76.33 |
| RMSE ($), zero prediction | 53.79 | 65.65 | 76.27 |
| MAE ($), served | 31.85 | 39.49 | 45.89 |
| MAE ($), raw heads | 31.88 | 39.49 | 45.89 |
| MAE ($), zero prediction | 31.90 | 39.54 | 45.92 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0007 | 0.0015 | 0.0004 |
| skill vs zero, raw heads | -0.0022 | 0.0022 | -0.0015 |
| EV, served | -0.0003 | 0.0004 | -0.0001 |
| EV, raw heads | -0.0045 | -0.0010 | -0.0045 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0032 | 0.0199 | -0.0008 |
| corr, Spearman, raw heads | 0.0087 | 0.0114 | 0.0214 |
| mean predicted ($), served | 0.59 | 0.66 | 0.27 |
| mean predicted ($), raw heads | 2.90 | 2.62 | 2.07 |
| mean realised ($) | 2.61 | 3.92 | 5.24 |
| share predicted up, raw heads | 0.8745 | 0.8343 | 0.7132 |
| share realised up | 0.5262 | 0.5317 | 0.5289 |
| shrink beta (served = beta x raw, fit on cal) | 0.2049 | 0.2504 | 0.1326 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 24.82 | 30.75 | 35.83 |
| CRPSS vs constant variance | -0.0009 | 0.0012 | -0.0010 |
| NLL | 6.3709 | 6.3921 | 6.6140 |
| PIT KS | 0.0862 | 0.0865 | 0.0892 |
| var / err^2 Spearman | 0.0959 | 0.1743 | 0.2077 |
| coverage of the 90% interval | 0.9318 | 0.9354 | 0.9340 |
| width of the 90% interval ($) | 172.31 | 220.16 | 259.63 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0340 | [0.0059, 0.0619] | WORKS |
| h1 | -0.0085 | [-0.0351, 0.0191] | NOISE |
| h2 | 0.0107 | [-0.0202, 0.0420] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.205 / h1 0.250 / h2 0.133) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4227 | 0.5210 | 0.6136 |
| abs(d h1) <= abs(d h2) | 0.5087 | 0.3005 | 0.5890 |
| full chain h0 <= h1 <= h2 | 0.1520 | 0.0763 | 0.3253 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6225 | 0.4640 | 0.6486 | 0.2204 |
| expected if the two signs were independent | 0.5675 | 0.4337 | 0.5741 | 0.1752 |

- P(up) unanimity (all three horizons call the same side): 0.4062

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0219 vs 0.0451 (-0.0233): does not beat, noise (boot z -1.27) | -0.0346 vs 0.0522 (-0.0868): does not beat, significantly worse (boot z -3.11) | 0.0043 vs 0.0519 (-0.0475): does not beat, significantly worse (boot z -2.17) |
| logreg_lags | direction/auc | 0.5138 vs 0.5317 (-0.0179): does not beat, noise (boot z -1.82) | 0.4829 vs 0.5327 (-0.0498): does not beat, significantly worse (boot z -2.61) | 0.4990 vs 0.5366 (-0.0376): does not beat, significantly worse (boot z -2.86) |
| logreg_lags | direction/brier | 0.2780 vs 0.2535 (-0.0245): does not beat, significantly worse (DM z -7.94) | 0.2743 vs 0.2535 (-0.0208): does not beat, significantly worse (DM z -5.42) | 0.2726 vs 0.2522 (-0.0204): does not beat, significantly worse (DM z -6.35) |
| logreg_lags | direction/ece_pos | 0.1161 vs 0.0381 (-0.0780): does not beat, significantly worse (boot z -6.12) | 0.1248 vs 0.0453 (-0.0796): does not beat, significantly worse (boot z -6.29) | 0.0971 vs 0.0445 (-0.0526): does not beat, significantly worse (boot z -3.11) |
| logreg_lags | direction/acc | 0.5174 vs 0.5300 (-0.0126): does not beat, noise (DM z -1.51) | 0.4748 vs 0.5360 (-0.0612): does not beat, significantly worse (DM z -4.24) | 0.5171 vs 0.5375 (-0.0204): does not beat, noise (DM z -1.96) |
| logreg_lags | direction/bal_acc | 0.5108 vs 0.5221 (-0.0113): does not beat, noise (boot z -1.26) | 0.4830 vs 0.5253 (-0.0423): does not beat, significantly worse (boot z -3.12) | 0.5020 vs 0.5248 (-0.0228): does not beat, significantly worse (boot z -2.18) |
| class_prior | direction/mcc | 0.0219 vs 0.0000 (+0.0219): beats, noise (boot z +1.16) | -0.0346 vs 0.0000 (-0.0346): does not beat, significantly worse (boot z -2.22) | 0.0043 vs 0.0000 (+0.0043): beats, noise (boot z +0.20) |
| class_prior | direction/auc | 0.5138 vs 0.5000 (+0.0138): beats, noise (boot z +1.14) | 0.4829 vs 0.5000 (-0.0171): does not beat, noise (boot z -1.59) | 0.4990 vs 0.5000 (-0.0010): does not beat, noise (boot z -0.07) |
| class_prior | direction/brier | 0.2780 vs 0.2494 (-0.0287): does not beat, significantly worse (DM z -6.36) | 0.2743 vs 0.2491 (-0.0252): does not beat, significantly worse (DM z -8.30) | 0.2726 vs 0.2489 (-0.0237): does not beat, significantly worse (DM z -5.58) |
| class_prior | direction/ece_pos | 0.1161 vs 0.0286 (-0.0876): does not beat, significantly worse (boot z -5.60) | 0.1248 vs 0.0280 (-0.0968): does not beat, significantly worse (boot z -7.89) | 0.0971 vs 0.0278 (-0.0693): does not beat, significantly worse (boot z -3.51) |
| class_prior | direction/acc | 0.5174 vs 0.5381 (-0.0207): does not beat, noise (DM z -1.50) | 0.4748 vs 0.5412 (-0.0663): does not beat, significantly worse (DM z -3.57) | 0.5171 vs 0.5428 (-0.0257): does not beat, noise (DM z -1.84) |
| class_prior | direction/bal_acc | 0.5108 vs 0.5000 (+0.0108): beats, noise (boot z +1.16) | 0.4830 vs 0.5000 (-0.0170): does not beat, significantly worse (boot z -2.22) | 0.5020 vs 0.5000 (+0.0020): beats, noise (boot z +0.20) |
| zero_delta | delta/rmse | 53.77 vs 53.79 (+0.02, +0.03%): beats, noise (DM z +0.99) | 65.60 vs 65.65 (+0.05, +0.07%): beats, noise (DM z +1.46) | 76.26 vs 76.27 (+0.01, +0.02%): beats, noise (DM z +0.43) |
| zero_delta | delta/mae | 31.85 vs 31.90 (+0.05, +0.15%): beats (DM z +2.85) | 39.49 vs 39.54 (+0.05, +0.13%): beats (DM z +2.24) | 45.89 vs 45.92 (+0.03, +0.06%): beats, noise (DM z +1.57) |
| mean_delta | delta/rmse | 53.77 vs 53.75 (-0.02, -0.03%): does not beat, noise (DM z -0.76) | 65.60 vs 65.59 (-0.02, -0.03%): does not beat, noise (DM z -0.51) | 76.26 vs 76.18 (-0.08, -0.11%): does not beat, noise (DM z -1.36) |
| mean_delta | delta/mae | 31.85 vs 31.86 (+0.00, +0.00%): beats, noise (DM z +0.11) | 39.49 vs 39.46 (-0.03, -0.07%): does not beat, noise (DM z -1.21) | 45.89 vs 45.83 (-0.06, -0.13%): does not beat, noise (DM z -1.47) |
| const_var | variance/crps | 24.82 vs 24.80 (-0.02, -0.09%): does not beat, noise (DM z -0.80) | 30.75 vs 30.79 (+0.04, +0.12%): beats, noise (DM z +1.03) | 35.83 vs 35.80 (-0.03, -0.10%): does not beat, noise (DM z -0.74) |
| const_var | variance/nll | 6.3709 vs 6.4212 (+0.0503): beats, noise (DM z +0.36) | 6.3921 vs 6.5631 (+0.1710): beats, noise (DM z +1.58) | 6.6140 vs 6.6903 (+0.0763): beats, noise (DM z +0.85) |
| const_var | variance/pit_ks | 0.0862 vs 0.0754 (-0.0107): does not beat, significantly worse (boot z -6.31) | 0.0865 vs 0.0767 (-0.0098): does not beat, significantly worse (boot z -4.55) | 0.0892 vs 0.0747 (-0.0145): does not beat, significantly worse (boot z -6.29) |
| const_var | variance/corr_var_err2_spearman | 0.0959 vs 0.0000 (+0.0959): beats (boot z +3.44) | 0.1743 vs 0.0000 (+0.1743): beats (boot z +5.94) | 0.2077 vs 0.0000 (+0.2077): beats (boot z +7.93) |

## Backtest (costs included)

- n_trades: 1137
- total_return: 0.1210
- sharpe_net: 8.4807
- sharpe_gross: 8.4807
- sortino: 13.2076
- max_drawdown: 0.0397
- hit_rate: 0.4855
- hit_rate_gross: 0.4855
- profit_factor: 1.1473
- avg_hold_bars: 8.6922
- exposure: 0.6655
- turnover: 2396.2723
- fees_paid: 0.0000
- traded_notional: 23963507.8744
- breakeven_cost_bps: 1.0095
- gross_edge_per_trade_bps: 1.0331
- costs_paid: 0.0000
- gross_pnl: 1209.6062
- net_pnl: 1209.6062

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 30.2476, indicator 19.4260 (clip 20).
Clipped steps over the run: main 4.0000, indicator 0.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 30.2476 / 8.9011 | 4.0000 / 0.0000 | 28.6% / 0.0% | 0.0000 | 1150.0000 / 1301.0000 / 1363.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 8.0247 / 2.7316 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1128.0000 / 1245.0000 / 1386.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 9.5368 / 19.4260 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1749.0000 / 1938.0000 / 2098.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 4.5160 / 5.6497 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1147.0000 / 1305.0000 / 1377.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 3.1935 / 7.7295 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1144.0000 / 1304.0000 / 1386.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 2.5908 / 5.1384 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1169.0000 / 1293.0000 / 1397.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 2.6031 / 0.8405 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1129.0000 / 1279.0000 / 1371.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 2.6152 / 1.4898 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1679.0000 / 1925.0000 / 2046.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 3.4131 / 2.0783 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1196.0000 / 1305.0000 / 1390.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 2.4762 / 1.2851 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1173.0000 / 1292.0000 / 1376.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 3.4023 / 3.2476 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1104.0000 / 1312.0000 / 1384.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 3.7698 / 2.3129 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1152.0000 / 1279.0000 / 1408.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 4.3863 / 6.0380 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1715.0000 / 1932.0000 / 2062.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 2.9030 / 3.6911 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1129.0000 / 1310.0000 / 1399.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
Learned periods sitting at their configured bound: period/obv_period_2=60.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=1.069 (corr skip/tower=-0.556), h1=0.578 (corr skip/tower=-0.486), h2=0.762 (corr skip/tower=-0.492).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -96 (TimeSeriesSplit fold 5, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-01-13T02:42:00 .. 2023-01-23T10:12:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8111, long_above 0.5553, short_below 0.4420, median 0.4944. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +12.10% | +8.48 | +3.97% | 1137 |
| buy and hold | +20.64% | +11.65 | +5.41% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -12.73% .. +12.23%) | -0.34% | -0.28 | | |

The random null enters at the strategy's rate (0.2289 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 93% of its seeds on net return, 92% on net Sharpe and 93% on gross return.
