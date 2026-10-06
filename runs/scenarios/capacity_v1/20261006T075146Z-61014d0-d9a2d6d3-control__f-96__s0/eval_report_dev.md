# Evaluation report - dev split - run `20261006T075146Z-61014d0-d9a2d6d3-control__f-96__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10701 | 11457 | 11840 |
| n_eff of the scored moves (n scored // bars ahead) | 1070 | 763 | 592 |
| true up-rate | 0.5381 | 0.5412 | 0.5428 |
| calls up (predicted up-rate) | 0.3557 | 0.4556 | 0.4229 |
| accuracy | 0.4812 | 0.4976 | 0.5078 |
| balanced accuracy | 0.4921 | 0.5013 | 0.5145 |
| precision (up) | 0.5271 | 0.5425 | 0.5598 |
| recall / sensitivity (up) | 0.3484 | 0.4568 | 0.4361 |
| specificity (down) | 0.6358 | 0.5457 | 0.5928 |
| F1 (up) | 0.4195 | 0.4960 | 0.4903 |
| MCC | -0.0164 | 0.0025 | 0.0292 |
| AUC | 0.4946 | 0.5094 | 0.5192 |
| Brier | 0.2951 | 0.2577 | 0.2685 |
| ECE (positive class) | 0.1717 | 0.0751 | 0.1061 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0381 | 0.0412 | 0.0428 |
| TP / FP / TN / FN | 2006 / 1800 / 3143 / 3752 | 2832 / 2388 / 2869 / 3368 | 2803 / 2204 / 3209 / 3624 |
| Gaussian readout: calls up | 0.4080 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | -0.0131 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.4968 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2501 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0389 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.4080 | 0.4499 | 0.4277 |
| Gaussian readout of the raw heads: MCC | -0.0131 | -0.0259 | -0.0298 |
| Gaussian readout of the raw heads: AUC | 0.4968 | 0.4874 | 0.4916 |
| Gaussian readout of the raw heads: Brier | 0.2662 | 0.2752 | 0.2739 |
| Gaussian readout of the raw heads: ECE | 0.1099 | 0.1279 | 0.1404 |

beta = 0 for h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 53.78 | 65.65 | 76.27 |
| RMSE ($), raw heads | 54.17 | 66.29 | 77.64 |
| RMSE ($), zero prediction | 53.79 | 65.65 | 76.27 |
| MAE ($), served | 31.91 | 39.54 | 45.92 |
| MAE ($), raw heads | 32.62 | 40.49 | 47.61 |
| MAE ($), zero prediction | 31.90 | 39.54 | 45.92 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0001 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0141 | -0.0197 | -0.0361 |
| EV, served | 0.0001 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0105 | -0.0156 | -0.0319 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0236 | 0.0065 | -0.0025 |
| corr, Spearman, raw heads | -0.0043 | -0.0161 | -0.0160 |
| mean predicted ($), served | -0.04 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -1.55 | -1.84 | -2.03 |
| mean realised ($) | 2.61 | 3.92 | 5.24 |
| share predicted up, raw heads | 0.4058 | 0.4501 | 0.4261 |
| share realised up | 0.5262 | 0.5317 | 0.5289 |
| shrink beta (served = beta x raw, fit on cal) | 0.0262 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 24.40 | 30.68 | 35.37 |
| CRPSS vs constant variance | 0.0161 | 0.0037 | 0.0117 |
| NLL | 5.6955 | 6.3333 | 6.2985 |
| PIT KS | 0.0644 | 0.0955 | 0.0834 |
| var / err^2 Spearman | 0.2135 | 0.2011 | 0.2305 |
| coverage of the 90% interval | 0.9315 | 0.9337 | 0.9329 |
| width of the 90% interval ($) | 171.96 | 217.87 | 258.11 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0071 | [-0.0361, 0.0243] | NOISE |
| h1 | 0.0252 | [0.0014, 0.0532] | WORKS |
| h2 | 0.0099 | [-0.0174, 0.0372] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.026 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6193 | n/a (beta = 0: served delta is 0) | 0.6136 |
| abs(d h1) <= abs(d h2) | 0.7684 | n/a (beta = 0: served delta is 0) | 0.5890 |
| full chain h0 <= h1 <= h2 | 0.4517 | n/a (beta = 0: served delta is 0) | 0.3253 |

beta = 0 for h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6783 | 0.6441 | 0.6299 | 0.3495 |
| expected if the two signs were independent | 0.5264 | 0.5054 | 0.5106 | 0.2096 |

- P(up) unanimity (all three horizons call the same side): 0.4664

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0164 vs 0.0451 (-0.0616): does not beat, noise (boot z -1.79) | 0.0025 vs 0.0522 (-0.0497): does not beat, noise (boot z -1.55) | 0.0292 vs 0.0519 (-0.0227): does not beat, noise (boot z -0.69) |
| logreg_lags | direction/auc | 0.4946 vs 0.5317 (-0.0371): does not beat, noise (boot z -1.65) | 0.5094 vs 0.5327 (-0.0233): does not beat, noise (boot z -1.26) | 0.5192 vs 0.5366 (-0.0174): does not beat, noise (boot z -0.79) |
| logreg_lags | direction/brier | 0.2951 vs 0.2535 (-0.0416): does not beat, significantly worse (DM z -7.28) | 0.2577 vs 0.2535 (-0.0042): does not beat, noise (DM z -1.57) | 0.2685 vs 0.2522 (-0.0163): does not beat, significantly worse (DM z -3.57) |
| logreg_lags | direction/ece_pos | 0.1717 vs 0.0381 (-0.1337): does not beat, significantly worse (boot z -9.74) | 0.0751 vs 0.0453 (-0.0298): does not beat, significantly worse (boot z -2.39) | 0.1061 vs 0.0445 (-0.0616): does not beat, significantly worse (boot z -4.72) |
| logreg_lags | direction/acc | 0.4812 vs 0.5300 (-0.0489): does not beat, significantly worse (DM z -2.91) | 0.4976 vs 0.5360 (-0.0384): does not beat, significantly worse (DM z -2.38) | 0.5078 vs 0.5375 (-0.0297): does not beat, noise (DM z -1.70) |
| logreg_lags | direction/bal_acc | 0.4921 vs 0.5221 (-0.0300): does not beat, noise (boot z -1.81) | 0.5013 vs 0.5253 (-0.0240): does not beat, noise (boot z -1.53) | 0.5145 vs 0.5248 (-0.0103): does not beat, noise (boot z -0.65) |
| class_prior | direction/mcc | -0.0164 vs 0.0000 (-0.0164): does not beat, noise (boot z -0.77) | 0.0025 vs 0.0000 (+0.0025): beats, noise (boot z +0.13) | 0.0292 vs 0.0000 (+0.0292): beats, noise (boot z +1.45) |
| class_prior | direction/auc | 0.4946 vs 0.5000 (-0.0054): does not beat, noise (boot z -0.41) | 0.5094 vs 0.5000 (+0.0094): beats, noise (boot z +0.77) | 0.5192 vs 0.5000 (+0.0192): beats, noise (boot z +1.50) |
| class_prior | direction/brier | 0.2951 vs 0.2494 (-0.0457): does not beat, significantly worse (DM z -9.75) | 0.2577 vs 0.2491 (-0.0086): does not beat, significantly worse (DM z -4.05) | 0.2685 vs 0.2489 (-0.0196): does not beat, significantly worse (DM z -5.49) |
| class_prior | direction/ece_pos | 0.1717 vs 0.0286 (-0.1432): does not beat, significantly worse (boot z -10.72) | 0.0751 vs 0.0280 (-0.0470): does not beat, significantly worse (boot z -3.49) | 0.1061 vs 0.0278 (-0.0783): does not beat, significantly worse (boot z -6.20) |
| class_prior | direction/acc | 0.4812 vs 0.5381 (-0.0569): does not beat, significantly worse (DM z -3.09) | 0.4976 vs 0.5412 (-0.0436): does not beat, significantly worse (DM z -2.35) | 0.5078 vs 0.5428 (-0.0351): does not beat, noise (DM z -1.65) |
| class_prior | direction/bal_acc | 0.4921 vs 0.5000 (-0.0079): does not beat, noise (boot z -0.77) | 0.5013 vs 0.5000 (+0.0013): beats, noise (boot z +0.13) | 0.5145 vs 0.5000 (+0.0145): beats, noise (boot z +1.45) |
| zero_delta | delta/rmse | 53.78 vs 53.79 (+0.00, +0.00%): beats, noise (DM z +0.40) | 65.65 vs 65.65 (+0.00, +0.00%): does not beat | 76.27 vs 76.27 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 31.91 vs 31.90 (-0.01, -0.02%): does not beat, significantly worse (DM z -2.25) | 39.54 vs 39.54 (+0.00, +0.00%): does not beat | 45.92 vs 45.92 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 53.78 vs 53.75 (-0.03, -0.06%): does not beat, noise (DM z -1.59) | 65.65 vs 65.59 (-0.06, -0.10%): does not beat, noise (DM z -1.69) | 76.27 vs 76.18 (-0.10, -0.13%): does not beat, noise (DM z -1.68) |
| mean_delta | delta/mae | 31.91 vs 31.86 (-0.05, -0.16%): does not beat, significantly worse (DM z -2.89) | 39.54 vs 39.46 (-0.08, -0.19%): does not beat, significantly worse (DM z -2.48) | 45.92 vs 45.83 (-0.09, -0.19%): does not beat, noise (DM z -1.89) |
| const_var | variance/crps | 24.40 vs 24.80 (+0.40, +1.61%): beats (DM z +3.83) | 30.68 vs 30.79 (+0.12, +0.37%): beats, noise (DM z +1.44) | 35.37 vs 35.80 (+0.42, +1.17%): beats (DM z +2.91) |
| const_var | variance/nll | 5.6955 vs 6.4212 (+0.7257): beats (DM z +3.03) | 6.3333 vs 6.5631 (+0.2298): beats, noise (DM z +1.37) | 6.2985 vs 6.6903 (+0.3917): beats, noise (DM z +1.79) |
| const_var | variance/pit_ks | 0.0644 vs 0.0754 (+0.0110): beats, noise (boot z +1.94) | 0.0955 vs 0.0767 (-0.0188): does not beat, significantly worse (boot z -3.93) | 0.0834 vs 0.0747 (-0.0087): does not beat, noise (boot z -1.39) |
| const_var | variance/corr_var_err2_spearman | 0.2135 vs 0.0000 (+0.2135): beats (boot z +8.27) | 0.2011 vs 0.0000 (+0.2011): beats (boot z +7.10) | 0.2305 vs 0.0000 (+0.2305): beats (boot z +8.06) |

## Backtest (costs included)

- n_trades: 890
- total_return: 0.0879
- sharpe_net: 6.5383
- sharpe_gross: 6.5383
- sortino: 10.7154
- max_drawdown: 0.0385
- hit_rate: 0.4798
- hit_rate_gross: 0.4798
- profit_factor: 1.1314
- avg_hold_bars: 8.7180
- exposure: 0.5225
- turnover: 1899.9538
- fees_paid: 0.0000
- traded_notional: 19001033.9721
- breakeven_cost_bps: 0.9253
- gross_edge_per_trade_bps: 0.9773
- costs_paid: 0.0000
- gross_pnl: 879.0950
- net_pnl: 879.0950

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 182.4872, indicator 143.8307 (clip 20).
Clipped steps over the run: main 17.0000, indicator 18.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 46.2862 / 51.3694 | 1.0000 / 2.0000 | 7.1% / 14.3% | 0.0000 | 1150.0000 / 1301.0000 / 1363.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 2.7626 / 5.5700 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1128.0000 / 1245.0000 / 1386.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 3.1178 / 2.8394 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1749.0000 / 1938.0000 / 2098.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 4.1133 / 7.0209 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1147.0000 / 1305.0000 / 1377.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 3.4476 / 7.1433 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1144.0000 / 1304.0000 / 1386.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 7.2693 / 5.2774 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1169.0000 / 1293.0000 / 1397.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 23.8367 / 83.6821 | 1.0000 / 2.0000 | 7.1% / 14.3% | 0.0000 | 1129.0000 / 1279.0000 / 1371.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 21.1298 / 41.3769 | 1.0000 / 2.0000 | 7.1% / 14.3% | 0.0000 | 1679.0000 / 1925.0000 / 2046.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 20.7757 / 12.3662 | 2.0000 / 0.0000 | 14.3% / 0.0% | 0.0000 | 1196.0000 / 1305.0000 / 1390.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 21.7640 / 64.1161 | 1.0000 / 2.0000 | 7.1% / 14.3% | 0.0000 | 1173.0000 / 1292.0000 / 1376.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 24.4528 / 17.7430 | 1.0000 / 0.0000 | 7.1% / 0.0% | 0.0000 | 1104.0000 / 1312.0000 / 1384.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 32.1907 / 94.2263 | 3.0000 / 2.0000 | 21.4% / 14.3% | 0.0000 | 1152.0000 / 1279.0000 / 1408.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 182.4872 / 143.8307 | 3.0000 / 3.0000 | 21.4% / 21.4% | 0.0000 | 1715.0000 / 1932.0000 / 2062.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 41.3620 / 115.6617 | 4.0000 / 5.0000 | 28.6% / 35.7% | 0.0000 | 1129.0000 / 1310.0000 / 1399.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.756 (corr skip/tower=-0.694), h1=0.147 (corr skip/tower=-0.027), h2=0.202 (corr skip/tower=-0.490).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -96 (TimeSeriesSplit fold 5, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-01-13T02:42:00 .. 2023-01-23T10:12:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.3738, long_above 0.6014, short_below 0.4032, median 0.4998. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +8.79% | +6.54 | +3.85% | 890 |
| buy and hold | +20.64% | +11.65 | +5.41% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -11.71% .. +11.89%) | -0.95% | -0.82 | | |

The random null enters at the strategy's rate (0.1255 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 88% of its seeds on net return, 88% on net Sharpe and 88% on gross return.
