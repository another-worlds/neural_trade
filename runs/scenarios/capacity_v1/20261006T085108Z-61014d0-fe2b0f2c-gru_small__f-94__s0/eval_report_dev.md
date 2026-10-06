# Evaluation report - dev split - run `20261006T085108Z-61014d0-fe2b0f2c-gru_small__f-94__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 8849 | 9593 | 9933 |
| n_eff of the scored moves (n scored // bars ahead) | 884 | 639 | 496 |
| true up-rate | 0.4864 | 0.4793 | 0.4801 |
| calls up (predicted up-rate) | 0.5135 | 0.4521 | 0.4207 |
| accuracy | 0.5276 | 0.5354 | 0.5290 |
| balanced accuracy | 0.5280 | 0.5335 | 0.5259 |
| precision (up) | 0.5136 | 0.5163 | 0.5109 |
| recall / sensitivity (up) | 0.5423 | 0.4870 | 0.4477 |
| specificity (down) | 0.5138 | 0.5800 | 0.6042 |
| F1 (up) | 0.5276 | 0.5012 | 0.4772 |
| MCC | 0.0560 | 0.0672 | 0.0525 |
| AUC | 0.5375 | 0.5487 | 0.5332 |
| Brier | 0.2537 | 0.2514 | 0.2532 |
| ECE (positive class) | 0.0464 | 0.0437 | 0.0399 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0136 | 0.0207 | 0.0199 |
| TP / FP / TN / FN | 2334 / 2210 / 2335 / 1970 | 2239 / 2098 / 2897 / 2359 | 2135 / 2044 / 3120 / 2634 |
| Gaussian readout: calls up | 0.6124 | 0.5842 | 0.5465 |
| Gaussian readout: MCC | 0.0665 | 0.0542 | 0.0506 |
| Gaussian readout: AUC | 0.5353 | 0.5281 | 0.5311 |
| Gaussian readout: Brier | 0.2496 | 0.2500 | 0.2496 |
| Gaussian readout: ECE | 0.0186 | 0.0247 | 0.0225 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 35.60 | 43.77 | 50.42 |
| RMSE ($), raw heads | 36.10 | 44.89 | 51.82 |
| RMSE ($), zero prediction | 35.60 | 43.71 | 50.34 |
| MAE ($), served | 22.55 | 27.22 | 30.76 |
| MAE ($), raw heads | 22.93 | 27.99 | 31.80 |
| MAE ($), zero prediction | 22.55 | 27.19 | 30.71 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0004 | -0.0025 | -0.0034 |
| skill vs zero, raw heads | -0.0279 | -0.0545 | -0.0598 |
| EV, served | 0.0008 | -0.0020 | -0.0028 |
| EV, raw heads | -0.0241 | -0.0509 | -0.0561 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0290 | -0.0092 | -0.0091 |
| corr, Spearman, raw heads | 0.0502 | 0.0366 | 0.0362 |
| mean predicted ($), served | 0.21 | 0.24 | 0.30 |
| mean predicted ($), raw heads | 1.30 | 1.42 | 1.52 |
| mean realised ($) | -1.20 | -1.81 | -2.42 |
| share predicted up, raw heads | 0.5964 | 0.5715 | 0.5348 |
| share realised up | 0.4785 | 0.4800 | 0.4785 |
| shrink beta (served = beta x raw, fit on cal) | 0.1628 | 0.1685 | 0.1971 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 16.96 | 20.61 | 23.44 |
| CRPSS vs constant variance | 0.0187 | 0.0172 | 0.0192 |
| NLL | 5.1923 | 5.3798 | 5.6276 |
| PIT KS | 0.0581 | 0.0503 | 0.0528 |
| var / err^2 Spearman | 0.3301 | 0.3172 | 0.3131 |
| coverage of the 90% interval | 0.9077 | 0.9106 | 0.9156 |
| width of the 90% interval ($) | 102.00 | 124.97 | 146.06 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0274 | [-0.0035, 0.0577] | NOISE |
| h1 | 0.0490 | [0.0176, 0.0805] | WORKS |
| h2 | 0.0230 | [-0.0143, 0.0594] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.163 / h1 0.169 / h2 0.197) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7314 | 0.7451 | 0.6023 |
| abs(d h1) <= abs(d h2) | 0.7387 | 0.8033 | 0.5711 |
| full chain h0 <= h1 <= h2 | 0.5123 | 0.5818 | 0.3135 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7569 | 0.7457 | 0.7197 | 0.4957 |
| expected if the two signs were independent | 0.5014 | 0.4925 | 0.4937 | 0.2441 |

- P(up) unanimity (all three horizons call the same side): 0.5823

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0560 vs 0.0988 (-0.0428): does not beat, noise (boot z -1.73) | 0.0672 vs 0.0921 (-0.0250): does not beat, noise (boot z -0.97) | 0.0525 vs 0.0686 (-0.0161): does not beat, noise (boot z -0.51) |
| logreg_lags | direction/auc | 0.5375 vs 0.5568 (-0.0194): does not beat, noise (boot z -1.24) | 0.5487 vs 0.5638 (-0.0151): does not beat, noise (boot z -0.89) | 0.5332 vs 0.5593 (-0.0260): does not beat, noise (boot z -1.32) |
| logreg_lags | direction/brier | 0.2537 vs 0.2483 (-0.0055): does not beat, significantly worse (DM z -2.54) | 0.2514 vs 0.2479 (-0.0035): does not beat, noise (DM z -1.39) | 0.2532 vs 0.2483 (-0.0049): does not beat, noise (DM z -1.76) |
| logreg_lags | direction/ece_pos | 0.0464 vs 0.0242 (-0.0222): does not beat, noise (boot z -1.82) | 0.0437 vs 0.0319 (-0.0118): does not beat, noise (boot z -0.74) | 0.0399 vs 0.0350 (-0.0049): does not beat, noise (boot z -0.26) |
| logreg_lags | direction/acc | 0.5276 vs 0.5430 (-0.0154): does not beat, noise (DM z -1.18) | 0.5354 vs 0.5389 (-0.0035): does not beat, noise (DM z -0.24) | 0.5290 vs 0.5243 (+0.0047): beats, noise (DM z +0.26) |
| logreg_lags | direction/bal_acc | 0.5280 vs 0.5471 (-0.0191): does not beat, noise (boot z -1.58) | 0.5335 vs 0.5445 (-0.0110): does not beat, noise (boot z -0.87) | 0.5259 vs 0.5318 (-0.0059): does not beat, noise (boot z -0.39) |
| class_prior | direction/mcc | 0.0560 vs 0.0000 (+0.0560): beats (boot z +2.72) | 0.0672 vs 0.0000 (+0.0672): beats (boot z +3.26) | 0.0525 vs 0.0000 (+0.0525): beats (boot z +2.17) |
| class_prior | direction/auc | 0.5375 vs 0.5000 (+0.0375): beats (boot z +2.91) | 0.5487 vs 0.5000 (+0.0487): beats (boot z +3.65) | 0.5332 vs 0.5000 (+0.0332): beats (boot z +2.14) |
| class_prior | direction/brier | 0.2537 vs 0.2503 (-0.0034): does not beat, noise (DM z -1.63) | 0.2514 vs 0.2504 (-0.0010): does not beat, noise (DM z -0.42) | 0.2532 vs 0.2505 (-0.0027): does not beat, noise (DM z -1.07) |
| class_prior | direction/ece_pos | 0.0464 vs 0.0224 (-0.0241): does not beat, significantly worse (boot z -1.97) | 0.0437 vs 0.0282 (-0.0155): does not beat, noise (boot z -0.92) | 0.0399 vs 0.0298 (-0.0100): does not beat, noise (boot z -0.51) |
| class_prior | direction/acc | 0.5276 vs 0.4864 (+0.0412): beats (DM z +2.57) | 0.5354 vs 0.4793 (+0.0561): beats (DM z +2.89) | 0.5290 vs 0.4801 (+0.0489): beats (DM z +2.13) |
| class_prior | direction/bal_acc | 0.5280 vs 0.5000 (+0.0280): beats (boot z +2.72) | 0.5335 vs 0.5000 (+0.0335): beats (boot z +3.26) | 0.5259 vs 0.5000 (+0.0259): beats (boot z +2.17) |
| zero_delta | delta/rmse | 35.60 vs 35.60 (+0.01, +0.02%): beats, noise (DM z +0.27) | 43.77 vs 43.71 (-0.05, -0.12%): does not beat, noise (DM z -1.26) | 50.42 vs 50.34 (-0.09, -0.17%): does not beat, noise (DM z -1.25) |
| zero_delta | delta/mae | 22.55 vs 22.55 (-0.00, -0.01%): does not beat, noise (DM z -0.09) | 27.22 vs 27.19 (-0.03, -0.11%): does not beat, noise (DM z -1.02) | 30.76 vs 30.71 (-0.05, -0.18%): does not beat, noise (DM z -1.16) |
| mean_delta | delta/rmse | 35.60 vs 35.61 (+0.02, +0.04%): beats, noise (DM z +0.59) | 43.77 vs 43.73 (-0.04, -0.09%): does not beat, noise (DM z -0.90) | 50.42 vs 50.36 (-0.06, -0.13%): does not beat, noise (DM z -0.90) |
| mean_delta | delta/mae | 22.55 vs 22.56 (+0.01, +0.04%): beats, noise (DM z +0.49) | 27.22 vs 27.20 (-0.02, -0.06%): does not beat, noise (DM z -0.59) | 30.76 vs 30.72 (-0.04, -0.11%): does not beat, noise (DM z -0.76) |
| const_var | variance/crps | 16.96 vs 17.28 (+0.32, +1.87%): beats (DM z +4.53) | 20.61 vs 20.97 (+0.36, +1.72%): beats (DM z +4.67) | 23.44 vs 23.90 (+0.46, +1.92%): beats (DM z +3.88) |
| const_var | variance/nll | 5.1923 vs 5.2417 (+0.0494): beats, noise (DM z +0.58) | 5.3798 vs 5.4377 (+0.0578): beats, noise (DM z +0.79) | 5.6276 vs 5.5668 (-0.0608): does not beat, noise (DM z -0.68) |
| const_var | variance/pit_ks | 0.0581 vs 0.0392 (-0.0189): does not beat, significantly worse (boot z -2.77) | 0.0503 vs 0.0494 (-0.0009): does not beat, noise (boot z -0.08) | 0.0528 vs 0.0598 (+0.0070): beats, noise (boot z +0.53) |
| const_var | variance/corr_var_err2_spearman | 0.3301 vs 0.0000 (+0.3301): beats (boot z +13.24) | 0.3172 vs 0.0000 (+0.3172): beats (boot z +12.20) | 0.3131 vs 0.0000 (+0.3131): beats (boot z +11.18) |

## Backtest (costs included)

- n_trades: 553
- total_return: 0.0829
- sharpe_net: 12.2465
- sharpe_gross: 12.2465
- sortino: 18.4282
- max_drawdown: 0.0257
- hit_rate: 0.5497
- hit_rate_gross: 0.5497
- profit_factor: 1.2794
- avg_hold_bars: 9.6745
- exposure: 0.3602
- turnover: 1157.8283
- fees_paid: 0.0000
- traded_notional: 11577697.2478
- breakeven_cost_bps: 1.4321
- gross_edge_per_trade_bps: 1.4534
- costs_paid: 0.0000
- gross_pnl: 829.0398
- net_pnl: 829.0398

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 82.6258, indicator 431.8115 (clip 20).
Clipped steps over the run: main 9.0000, indicator 27.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 82.6258 / 238.4556 | 5.0000 / 2.0000 | 11.6% / 4.7% | 0.0000 | 2605.0000 / 2952.0000 / 3194.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 8.7685 / 431.8115 | 0.0000 / 1.0000 | 0.0% / 2.3% | 0.0000 | 2610.0000 / 3022.0000 / 3243.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 3.7113 / 6.9855 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2526.0000 / 2918.0000 / 3192.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 9.4963 / 20.9453 | 0.0000 / 1.0000 | 0.0% / 2.3% | 0.0000 | 3121.0000 / 3591.0000 / 3884.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 9.5926 / 41.6198 | 0.0000 / 1.0000 | 0.0% / 2.3% | 0.0000 | 2572.0000 / 2911.0000 / 3162.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 8.5829 / 61.2265 | 0.0000 / 1.0000 | 0.0% / 2.3% | 0.0000 | 2588.0000 / 2972.0000 / 3137.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 66.8323 / 154.6339 | 1.0000 / 3.0000 | 2.3% / 7.0% | 0.0000 | 2957.0000 / 3360.0000 / 3589.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 7.0145 / 25.6237 | 0.0000 / 1.0000 | 0.0% / 2.3% | 0.0000 | 2639.0000 / 2964.0000 / 3218.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 24.1900 / 56.8662 | 1.0000 / 1.0000 | 2.3% / 2.3% | 0.0000 | 2587.0000 / 3003.0000 / 3212.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 9.6396 / 52.9863 | 0.0000 / 3.0000 | 0.0% / 7.0% | 0.0000 | 2551.0000 / 2956.0000 / 3195.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 11.2775 / 25.6201 | 0.0000 / 2.0000 | 0.0% / 4.7% | 0.0000 | 2523.0000 / 2910.0000 / 3171.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 12.8316 / 18.1814 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2645.0000 / 2973.0000 / 3210.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 29.7216 / 65.5240 | 1.0000 / 5.0000 | 2.3% / 11.6% | 0.0000 | 2602.0000 / 2933.0000 / 3194.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 34.6364 / 98.6972 | 1.0000 / 6.0000 | 2.3% / 14.0% | 0.0000 | 3118.0000 / 3548.0000 / 3866.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.108 (corr skip/tower=-0.307), h1=0.227 (corr skip/tower=-0.354), h2=0.390 (corr skip/tower=-0.453).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -94 (TimeSeriesSplit fold 7, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-02T17:44:00 .. 2023-02-13T01:14:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.9668, long_above 0.5950, short_below 0.3929, median 0.4888. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +8.29% | +12.25 | +2.57% | 553 |
| buy and hold | -7.43% | -7.41 | +10.90% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -6.42% .. +6.42%) | +0.59% | +1.00 | | |

The random null enters at the strategy's rate (0.0582 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 97% of its seeds on net return, 97% on net Sharpe and 97% on gross return.
