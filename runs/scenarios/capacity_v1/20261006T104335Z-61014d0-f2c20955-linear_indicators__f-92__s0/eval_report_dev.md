# Evaluation report - dev split - run `20261006T104335Z-61014d0-f2c20955-linear_indicators__f-92__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 9392 | 10127 | 10722 |
| n_eff of the scored moves (n scored // bars ahead) | 939 | 675 | 536 |
| true up-rate | 0.4856 | 0.4794 | 0.4808 |
| calls up (predicted up-rate) | 0.6019 | 0.5011 | 0.6918 |
| accuracy | 0.5181 | 0.5225 | 0.5096 |
| balanced accuracy | 0.5210 | 0.5225 | 0.5170 |
| precision (up) | 0.5031 | 0.5019 | 0.4931 |
| recall / sensitivity (up) | 0.6235 | 0.5246 | 0.7094 |
| specificity (down) | 0.4185 | 0.5205 | 0.3246 |
| F1 (up) | 0.5569 | 0.5130 | 0.5818 |
| MCC | 0.0430 | 0.0451 | 0.0368 |
| AUC | 0.5355 | 0.5307 | 0.5319 |
| Brier | 0.2533 | 0.2525 | 0.2531 |
| ECE (positive class) | 0.0489 | 0.0425 | 0.0560 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0144 | 0.0206 | 0.0192 |
| TP / FP / TN / FN | 2844 / 2809 / 2022 / 1717 | 2547 / 2528 / 2744 / 2308 | 3657 / 3760 / 1807 / 1498 |
| Gaussian readout: calls up | 0.6874 | 0.5159 | 0.5158 |
| Gaussian readout: MCC | 0.0247 | 0.0633 | 0.0670 |
| Gaussian readout: AUC | 0.5290 | 0.5432 | 0.5477 |
| Gaussian readout: Brier | 0.2497 | 0.2490 | 0.2492 |
| Gaussian readout: ECE | 0.0210 | 0.0262 | 0.0248 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 43.33 | 54.28 | 63.38 |
| RMSE ($), raw heads | 43.35 | 54.28 | 63.40 |
| RMSE ($), zero prediction | 43.32 | 54.31 | 63.40 |
| MAE ($), served | 26.27 | 31.92 | 36.75 |
| MAE ($), raw heads | 26.27 | 31.90 | 36.67 |
| MAE ($), zero prediction | 26.28 | 31.98 | 36.80 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | 0.0011 | 0.0006 |
| skill vs zero, raw heads | -0.0010 | 0.0011 | 0.0001 |
| EV, served | 0.0003 | 0.0014 | 0.0006 |
| EV, raw heads | 0.0001 | 0.0017 | 0.0005 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0183 | 0.0412 | 0.0317 |
| corr, Spearman, raw heads | 0.0434 | 0.0567 | 0.0587 |
| mean predicted ($), served | 0.22 | 0.21 | 0.06 |
| mean predicted ($), raw heads | 0.62 | 0.38 | 0.25 |
| mean realised ($) | -1.39 | -2.08 | -2.78 |
| share predicted up, raw heads | 0.6902 | 0.4956 | 0.5023 |
| share realised up | 0.4734 | 0.4754 | 0.4770 |
| shrink beta (served = beta x raw, fit on cal) | 0.3552 | 0.5650 | 0.2256 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 20.10 | 24.51 | 28.31 |
| CRPSS vs constant variance | 0.0223 | 0.0231 | 0.0185 |
| NLL | 5.6137 | 5.9337 | 6.2236 |
| PIT KS | 0.0525 | 0.0519 | 0.0520 |
| var / err^2 Spearman | 0.2822 | 0.2921 | 0.2593 |
| coverage of the 90% interval | 0.9030 | 0.8997 | 0.8975 |
| width of the 90% interval ($) | 120.17 | 145.43 | 166.74 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0384 | [0.0087, 0.0689] | WORKS |
| h1 | 0.0209 | [-0.0061, 0.0461] | NOISE |
| h2 | 0.0257 | [-0.0045, 0.0551] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.355 / h1 0.565 / h2 0.226) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5462 | 0.6942 | 0.5983 |
| abs(d h1) <= abs(d h2) | 0.7006 | 0.3050 | 0.5786 |
| full chain h0 <= h1 <= h2 | 0.3425 | 0.1147 | 0.3054 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6179 | 0.5998 | 0.6548 | 0.2776 |
| expected if the two signs were independent | 0.5402 | 0.5000 | 0.5009 | 0.1700 |

- P(up) unanimity (all three horizons call the same side): 0.4097

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0430 vs 0.0646 (-0.0216): does not beat, noise (boot z -1.17) | 0.0451 vs 0.0750 (-0.0299): does not beat, noise (boot z -1.49) | 0.0368 vs 0.0755 (-0.0387): does not beat, significantly worse (boot z -2.13) |
| logreg_lags | direction/auc | 0.5355 vs 0.5596 (-0.0241): does not beat, significantly worse (boot z -2.01) | 0.5307 vs 0.5695 (-0.0389): does not beat, significantly worse (boot z -3.10) | 0.5319 vs 0.5730 (-0.0411): does not beat, significantly worse (boot z -3.78) |
| logreg_lags | direction/brier | 0.2533 vs 0.2488 (-0.0045): does not beat, significantly worse (DM z -2.49) | 0.2525 vs 0.2491 (-0.0034): does not beat, noise (DM z -1.75) | 0.2531 vs 0.2489 (-0.0042): does not beat, significantly worse (DM z -2.36) |
| logreg_lags | direction/ece_pos | 0.0489 vs 0.0315 (-0.0175): does not beat, significantly worse (boot z -2.01) | 0.0425 vs 0.0409 (-0.0016): does not beat, noise (boot z -0.13) | 0.0560 vs 0.0445 (-0.0115): does not beat, noise (boot z -1.72) |
| logreg_lags | direction/acc | 0.5181 vs 0.5244 (-0.0063): does not beat, noise (DM z -0.63) | 0.5225 vs 0.5260 (-0.0036): does not beat, noise (DM z -0.30) | 0.5096 vs 0.5237 (-0.0141): does not beat, noise (DM z -1.59) |
| logreg_lags | direction/bal_acc | 0.5210 vs 0.5299 (-0.0088): does not beat, noise (boot z -1.01) | 0.5225 vs 0.5344 (-0.0118): does not beat, noise (boot z -1.23) | 0.5170 vs 0.5330 (-0.0160): does not beat, significantly worse (boot z -1.97) |
| class_prior | direction/mcc | 0.0430 vs 0.0000 (+0.0430): beats (boot z +2.38) | 0.0451 vs 0.0000 (+0.0451): beats (boot z +2.71) | 0.0368 vs 0.0000 (+0.0368): beats, noise (boot z +1.73) |
| class_prior | direction/auc | 0.5355 vs 0.5000 (+0.0355): beats (boot z +3.07) | 0.5307 vs 0.5000 (+0.0307): beats (boot z +2.90) | 0.5319 vs 0.5000 (+0.0319): beats (boot z +2.25) |
| class_prior | direction/brier | 0.2533 vs 0.2505 (-0.0028): does not beat, noise (DM z -1.44) | 0.2525 vs 0.2506 (-0.0019): does not beat, noise (DM z -1.12) | 0.2531 vs 0.2507 (-0.0024): does not beat, noise (DM z -1.38) |
| class_prior | direction/ece_pos | 0.0489 vs 0.0260 (-0.0229): does not beat, significantly worse (boot z -2.65) | 0.0425 vs 0.0322 (-0.0103): does not beat, noise (boot z -0.82) | 0.0560 vs 0.0332 (-0.0229): does not beat, significantly worse (boot z -3.60) |
| class_prior | direction/acc | 0.5181 vs 0.4856 (+0.0325): beats (DM z +2.41) | 0.5225 vs 0.4794 (+0.0431): beats (DM z +2.49) | 0.5096 vs 0.4808 (+0.0288): beats (DM z +2.03) |
| class_prior | direction/bal_acc | 0.5210 vs 0.5000 (+0.0210): beats (boot z +2.38) | 0.5225 vs 0.5000 (+0.0225): beats (boot z +2.71) | 0.5170 vs 0.5000 (+0.0170): beats, noise (boot z +1.73) |
| zero_delta | delta/rmse | 43.33 vs 43.32 (-0.00, -0.00%): does not beat, noise (DM z -0.10) | 54.28 vs 54.31 (+0.03, +0.06%): beats, noise (DM z +0.80) | 63.38 vs 63.40 (+0.02, +0.03%): beats, noise (DM z +0.82) |
| zero_delta | delta/mae | 26.27 vs 26.28 (+0.01, +0.02%): beats, noise (DM z +0.67) | 31.92 vs 31.98 (+0.06, +0.19%): beats (DM z +2.27) | 36.75 vs 36.80 (+0.04, +0.12%): beats (DM z +2.73) |
| mean_delta | delta/rmse | 43.33 vs 43.34 (+0.01, +0.02%): beats, noise (DM z +0.75) | 54.28 vs 54.33 (+0.05, +0.10%): beats, noise (DM z +1.38) | 63.38 vs 63.43 (+0.05, +0.08%): beats, noise (DM z +1.83) |
| mean_delta | delta/mae | 26.27 vs 26.29 (+0.02, +0.09%): beats (DM z +2.72) | 31.92 vs 32.01 (+0.08, +0.26%): beats (DM z +3.12) | 36.75 vs 36.83 (+0.07, +0.20%): beats (DM z +3.20) |
| const_var | variance/crps | 20.10 vs 20.56 (+0.46, +2.23%): beats (DM z +11.04) | 24.51 vs 25.09 (+0.58, +2.31%): beats (DM z +9.14) | 28.31 vs 28.85 (+0.53, +1.85%): beats (DM z +7.69) |
| const_var | variance/nll | 5.6137 vs 6.2346 (+0.6210): beats (DM z +3.77) | 5.9337 vs 6.5179 (+0.5842): beats (DM z +3.68) | 6.2236 vs 6.6968 (+0.4732): beats (DM z +3.54) |
| const_var | variance/pit_ks | 0.0525 vs 0.0828 (+0.0303): beats (boot z +9.03) | 0.0519 vs 0.0747 (+0.0229): beats (boot z +5.81) | 0.0520 vs 0.0725 (+0.0205): beats (boot z +5.75) |
| const_var | variance/corr_var_err2_spearman | 0.2822 vs 0.0000 (+0.2822): beats (boot z +10.71) | 0.2921 vs 0.0000 (+0.2921): beats (boot z +10.00) | 0.2593 vs 0.0000 (+0.2593): beats (boot z +7.91) |

## Backtest (costs included)

- n_trades: 668
- total_return: 0.1195
- sharpe_net: 13.7005
- sharpe_gross: 13.7005
- sortino: 20.7574
- max_drawdown: 0.0191
- hit_rate: 0.5539
- hit_rate_gross: 0.5539
- profit_factor: 1.3038
- avg_hold_bars: 7.8757
- exposure: 0.3543
- turnover: 1406.2524
- fees_paid: 0.0000
- traded_notional: 14062510.8676
- breakeven_cost_bps: 1.6990
- gross_edge_per_trade_bps: 1.7099
- costs_paid: 0.0000
- gross_pnl: 1194.6401
- net_pnl: 1194.6401

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 30.6503, indicator 43.2308 (clip 20).
Clipped steps over the run: main 4.0000, indicator 5.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 30.6503 / 27.6818 | 4.0000 / 1.0000 | 5.6% / 1.4% | 0.0000 | 3512.0000 / 4092.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 3.8296 / 13.1945 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3452.0000 / 4067.0000 / 4454.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 10.3322 / 22.2469 | 0.0000 / 1.0000 | 0.0% / 1.4% | 0.0000 | 3414.0000 / 4008.0000 / 4421.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 6.8672 / 17.0834 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3493.0000 / 4116.0000 / 4488.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 5.3706 / 38.5982 | 0.0000 / 1.0000 | 0.0% / 1.4% | 0.0000 | 3444.0000 / 4007.0000 / 4375.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 6.3892 / 15.1371 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3468.0000 / 4051.0000 / 4457.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 13.8628 / 43.2308 | 0.0000 / 1.0000 | 0.0% / 1.4% | 0.0000 | 3459.0000 / 4101.0000 / 4492.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 5.6888 / 36.7153 | 0.0000 / 1.0000 | 0.0% / 1.4% | 0.0000 | 3495.0000 / 4140.0000 / 4527.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 6.9384 / 9.0241 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3460.0000 / 4074.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 4.3000 / 9.3927 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3503.0000 / 4005.0000 / 4505.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 6.3348 / 11.6554 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3507.0000 / 4058.0000 / 4513.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 6.5015 / 9.0197 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3437.0000 / 4007.0000 / 4382.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 6.3398 / 8.0059 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3427.0000 / 4093.0000 / 4540.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 11.1958 / 14.8421 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3463.0000 / 4091.0000 / 4446.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
Learned periods sitting at their configured bound: period/atr_period_1=2, period/keltner_0_atr_period=2, period/keltner_1_atr_period=2, period/keltner_2_atr_period=59.9843, period/macd_0_slow=60, period/macd_2_signal=60, period/obv_period_1=60, period/obv_period_2=60.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=1.109 (corr skip/tower=-0.612), h1=0.873 (corr skip/tower=-0.264), h2=0.900 (corr skip/tower=-0.506).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -92 (TimeSeriesSplit fold 9, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-23T08:46:00 .. 2023-03-05T16:16:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.187, long_above 0.5797, short_below 0.4494, median 0.5125. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +11.95% | +13.70 | +1.91% | 668 |
| buy and hold | -8.52% | -7.39 | +9.73% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -6.57% .. +7.30%) | -0.04% | -0.06 | | |

The random null enters at the strategy's rate (0.0697 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 100% of its seeds on net return, 99% on net Sharpe and 100% on gross return.
