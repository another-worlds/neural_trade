# Evaluation report - dev split - run `20261006T081903Z-61014d0-2ef82d96-control__f-93__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10324 | 10946 | 11486 |
| n_eff of the scored moves (n scored // bars ahead) | 1032 | 729 | 574 |
| true up-rate | 0.5120 | 0.5097 | 0.5145 |
| calls up (predicted up-rate) | 0.6373 | 0.5463 | 0.6346 |
| accuracy | 0.5019 | 0.5121 | 0.5070 |
| balanced accuracy | 0.4986 | 0.5112 | 0.5031 |
| precision (up) | 0.5109 | 0.5199 | 0.5169 |
| recall / sensitivity (up) | 0.6360 | 0.5573 | 0.6376 |
| specificity (down) | 0.3613 | 0.4651 | 0.3685 |
| F1 (up) | 0.5667 | 0.5379 | 0.5710 |
| MCC | -0.0028 | 0.0224 | 0.0063 |
| AUC | 0.5005 | 0.5164 | 0.5027 |
| Brier | 0.2778 | 0.2566 | 0.2673 |
| ECE (positive class) | 0.1246 | 0.0648 | 0.1019 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0120 | 0.0097 | 0.0145 |
| TP / FP / TN / FN | 3362 / 3218 / 1820 / 1924 | 3109 / 2871 / 2496 / 2470 | 3768 / 3521 / 2055 / 2142 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.6602 | 0.6711 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | 0.0295 | 0.0141 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.5185 | 0.5072 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2499 | 0.2498 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0148 | 0.0089 |
| Gaussian readout of the raw heads: calls up | 0.7081 | 0.6602 | 0.6711 |
| Gaussian readout of the raw heads: MCC | 0.0346 | 0.0295 | 0.0141 |
| Gaussian readout of the raw heads: AUC | 0.5200 | 0.5185 | 0.5073 |
| Gaussian readout of the raw heads: Brier | 0.2544 | 0.2540 | 0.2593 |
| Gaussian readout of the raw heads: ECE | 0.0482 | 0.0504 | 0.0832 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 55.38 | 68.23 | 79.33 |
| RMSE ($), raw heads | 55.45 | 68.46 | 79.83 |
| RMSE ($), zero prediction | 55.38 | 68.23 | 79.34 |
| MAE ($), served | 34.92 | 42.60 | 49.47 |
| MAE ($), raw heads | 35.06 | 42.81 | 50.01 |
| MAE ($), zero prediction | 34.92 | 42.60 | 49.49 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | 0.0000 | 0.0002 |
| skill vs zero, raw heads | -0.0024 | -0.0066 | -0.0124 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0000 | -0.0003 |
| EV, raw heads | -0.0030 | -0.0080 | -0.0141 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0218 | -0.0046 | -0.0045 |
| corr, Spearman, raw heads | 0.0295 | 0.0165 | 0.0107 |
| mean predicted ($), served | 0.00 | 0.07 | 0.48 |
| mean predicted ($), raw heads | 2.58 | 2.07 | 4.26 |
| mean realised ($) | 1.67 | 2.49 | 3.33 |
| share predicted up, raw heads | 0.6954 | 0.6505 | 0.6596 |
| share realised up | 0.4982 | 0.5021 | 0.5025 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0331 | 0.1133 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 26.61 | 32.69 | 37.92 |
| CRPSS vs constant variance | 0.0536 | 0.0488 | 0.0496 |
| NLL | 5.5837 | 5.8382 | 5.9730 |
| PIT KS | 0.0559 | 0.0578 | 0.0513 |
| var / err^2 Spearman | 0.2942 | 0.3131 | 0.2925 |
| coverage of the 90% interval | 0.8955 | 0.8931 | 0.8922 |
| width of the 90% interval ($) | 153.25 | 185.24 | 213.87 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0155 | [-0.0193, 0.0521] | NOISE |
| h1 | 0.0138 | [-0.0145, 0.0449] | NOISE |
| h2 | 0.0041 | [-0.0247, 0.0351] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.033 / h2 0.113) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7033 | n/a (beta = 0: served delta is 0) | 0.5973 |
| abs(d h1) <= abs(d h2) | 0.8409 | 0.9689 | 0.5845 |
| full chain h0 <= h1 <= h2 | 0.5751 | n/a (beta = 0: served delta is 0) | 0.3105 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7319 | 0.7635 | 0.7683 | 0.4992 |
| expected if the two signs were independent | 0.5576 | 0.5102 | 0.5398 | 0.2781 |

- P(up) unanimity (all three horizons call the same side): 0.5575

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0028 vs 0.0593 (-0.0621): does not beat, significantly worse (boot z -2.09) | 0.0224 vs 0.0353 (-0.0129): does not beat, noise (boot z -0.50) | 0.0063 vs 0.0287 (-0.0224): does not beat, noise (boot z -0.80) |
| logreg_lags | direction/auc | 0.5005 vs 0.5347 (-0.0342): does not beat, noise (boot z -1.70) | 0.5164 vs 0.5204 (-0.0040): does not beat, noise (boot z -0.24) | 0.5027 vs 0.5116 (-0.0089): does not beat, noise (boot z -0.51) |
| logreg_lags | direction/brier | 0.2778 vs 0.2515 (-0.0263): does not beat, significantly worse (DM z -5.41) | 0.2566 vs 0.2531 (-0.0035): does not beat, noise (DM z -1.39) | 0.2673 vs 0.2539 (-0.0133): does not beat, significantly worse (DM z -3.73) |
| logreg_lags | direction/ece_pos | 0.1246 vs 0.0169 (-0.1077): does not beat, significantly worse (boot z -6.80) | 0.0648 vs 0.0293 (-0.0355): does not beat, significantly worse (boot z -2.60) | 0.1019 vs 0.0324 (-0.0695): does not beat, significantly worse (boot z -4.63) |
| logreg_lags | direction/acc | 0.5019 vs 0.5302 (-0.0283): does not beat, noise (DM z -1.94) | 0.5121 vs 0.5182 (-0.0061): does not beat, noise (DM z -0.52) | 0.5070 vs 0.5163 (-0.0093): does not beat, noise (DM z -0.69) |
| logreg_lags | direction/bal_acc | 0.4986 vs 0.5296 (-0.0310): does not beat, significantly worse (boot z -2.12) | 0.5112 vs 0.5176 (-0.0065): does not beat, noise (boot z -0.50) | 0.5031 vs 0.5142 (-0.0112): does not beat, noise (boot z -0.82) |
| class_prior | direction/mcc | -0.0028 vs 0.0000 (-0.0028): does not beat, noise (boot z -0.13) | 0.0224 vs 0.0000 (+0.0224): beats, noise (boot z +0.97) | 0.0063 vs 0.0000 (+0.0063): beats, noise (boot z +0.27) |
| class_prior | direction/auc | 0.5005 vs 0.5000 (+0.0005): beats, noise (boot z +0.04) | 0.5164 vs 0.5000 (+0.0164): beats, noise (boot z +1.14) | 0.5027 vs 0.5000 (+0.0027): beats, noise (boot z +0.19) |
| class_prior | direction/brier | 0.2778 vs 0.2499 (-0.0279): does not beat, significantly worse (DM z -6.85) | 0.2566 vs 0.2499 (-0.0066): does not beat, significantly worse (DM z -2.87) | 0.2673 vs 0.2498 (-0.0175): does not beat, significantly worse (DM z -5.25) |
| class_prior | direction/ece_pos | 0.1246 vs 0.0060 (-0.1186): does not beat, significantly worse (boot z -7.25) | 0.0648 vs 0.0044 (-0.0604): does not beat, significantly worse (boot z -3.79) | 0.1019 vs 0.0064 (-0.0956): does not beat, significantly worse (boot z -5.22) |
| class_prior | direction/acc | 0.5019 vs 0.5120 (-0.0101): does not beat, noise (DM z -0.78) | 0.5121 vs 0.5097 (+0.0024): beats, noise (DM z +0.14) | 0.5070 vs 0.5145 (-0.0076): does not beat, noise (DM z -0.46) |
| class_prior | direction/bal_acc | 0.4986 vs 0.5000 (-0.0014): does not beat, noise (boot z -0.13) | 0.5112 vs 0.5000 (+0.0112): beats, noise (boot z +0.97) | 0.5031 vs 0.5000 (+0.0031): beats, noise (boot z +0.27) |
| zero_delta | delta/rmse | 55.38 vs 55.38 (+0.00, +0.00%): does not beat | 68.23 vs 68.23 (+0.00, +0.00%): beats, noise (DM z +0.22) | 79.33 vs 79.34 (+0.01, +0.01%): beats, noise (DM z +0.19) |
| zero_delta | delta/mae | 34.92 vs 34.92 (+0.00, +0.00%): does not beat | 42.60 vs 42.60 (+0.00, +0.01%): beats, noise (DM z +1.10) | 49.47 vs 49.49 (+0.02, +0.04%): beats, noise (DM z +0.61) |
| mean_delta | delta/rmse | 55.38 vs 55.38 (-0.00, -0.01%): does not beat, noise (DM z -1.18) | 68.23 vs 68.22 (-0.01, -0.01%): does not beat, noise (DM z -0.87) | 79.33 vs 79.32 (-0.00, -0.01%): does not beat, noise (DM z -0.13) |
| mean_delta | delta/mae | 34.92 vs 34.92 (+0.00, +0.00%): beats, noise (DM z +0.19) | 42.60 vs 42.60 (+0.00, +0.01%): beats, noise (DM z +0.69) | 49.47 vs 49.49 (+0.02, +0.03%): beats, noise (DM z +0.63) |
| const_var | variance/crps | 26.61 vs 28.12 (+1.51, +5.36%): beats (DM z +11.20) | 32.69 vs 34.37 (+1.68, +4.88%): beats (DM z +11.32) | 37.92 vs 39.90 (+1.98, +4.96%): beats (DM z +9.20) |
| const_var | variance/nll | 5.5837 vs 7.4610 (+1.8773): beats (DM z +7.63) | 5.8382 vs 7.6534 (+1.8152): beats (DM z +7.05) | 5.9730 vs 7.8083 (+1.8353): beats (DM z +6.23) |
| const_var | variance/pit_ks | 0.0559 vs 0.1245 (+0.0686): beats (boot z +12.66) | 0.0578 vs 0.1208 (+0.0630): beats (boot z +11.60) | 0.0513 vs 0.1242 (+0.0728): beats (boot z +11.15) |
| const_var | variance/corr_var_err2_spearman | 0.2942 vs 0.0000 (+0.2942): beats (boot z +11.60) | 0.3131 vs 0.0000 (+0.3131): beats (boot z +11.66) | 0.2925 vs 0.0000 (+0.2925): beats (boot z +9.61) |

## Backtest (costs included)

- n_trades: 689
- total_return: 0.0707
- sharpe_net: 6.7687
- sharpe_gross: 6.7687
- sortino: 9.9432
- max_drawdown: 0.0324
- hit_rate: 0.5254
- hit_rate_gross: 0.5254
- profit_factor: 1.1302
- avg_hold_bars: 9.2467
- exposure: 0.4290
- turnover: 1445.4716
- fees_paid: 0.0000
- traded_notional: 14455066.8594
- breakeven_cost_bps: 0.9778
- gross_edge_per_trade_bps: 1.0185
- costs_paid: 0.0000
- gross_pnl: 706.7125
- net_pnl: 706.7125

## Training health

11 epoch(s). Pre-clip gradient norm maximum over the run: main 119.2073, indicator 679.1854 (clip 20).
Clipped steps over the run: main 43.0000, indicator 100.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 45.7181 / 51.2640 | 1.0000 / 2.0000 | 1.7% / 3.4% | 0.0000 | 2763.0000 / 3243.0000 / 3538.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 26.3159 / 56.9328 | 1.0000 / 1.0000 | 1.7% / 1.7% | 0.0000 | 3263.0000 / 3763.0000 / 4132.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 12.9210 / 40.2852 | 0.0000 / 1.0000 | 0.0% / 1.7% | 0.0000 | 3233.0000 / 3789.0000 / 4047.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 11.9739 / 223.2186 | 0.0000 / 1.0000 | 0.0% / 1.7% | 0.0000 | 3163.0000 / 3666.0000 / 3994.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 19.3673 / 64.3846 | 0.0000 / 8.0000 | 0.0% / 13.8% | 0.0000 | 2758.0000 / 3194.0000 / 3497.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 18.8202 / 84.0920 | 0.0000 / 6.0000 | 0.0% / 10.3% | 0.0000 | 2684.0000 / 3159.0000 / 3405.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 40.6328 / 190.7979 | 3.0000 / 9.0000 | 5.2% / 15.5% | 0.0000 | 3162.0000 / 3762.0000 / 4092.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 119.2073 / 679.1854 | 15.0000 / 20.0000 | 25.9% / 34.5% | 0.0000 | 3188.0000 / 3720.0000 / 4056.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 95.9816 / 348.0687 | 6.0000 / 15.0000 | 10.3% / 25.9% | 0.0000 | 3135.0000 / 3687.0000 / 4079.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 110.8123 / 453.6232 | 7.0000 / 20.0000 | 12.1% / 34.5% | 0.0000 | 2705.0000 / 3155.0000 / 3448.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 29.4290 / 253.0830 | 10.0000 / 17.0000 | 17.2% / 29.3% | 0.0000 | 2693.0000 / 3146.0000 / 3493.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=1.011 (corr skip/tower=-0.798), h1=0.172 (corr skip/tower=-0.284), h2=0.443 (corr skip/tower=-0.601).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -93 (TimeSeriesSplit fold 8, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-13T01:15:00 .. 2023-02-23T08:45:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.638, long_above 0.6622, short_below 0.4296, median 0.5334. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +7.07% | +6.77 | +3.24% | 689 |
| buy and hold | +11.40% | +7.64 | +7.31% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -8.04% .. +7.69%) | +0.15% | +0.15 | | |

The random null enters at the strategy's rate (0.0813 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 94% of its seeds on net return, 91% on net Sharpe and 94% on gross return.
