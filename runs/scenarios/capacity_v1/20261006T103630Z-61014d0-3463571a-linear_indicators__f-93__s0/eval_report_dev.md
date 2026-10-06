# Evaluation report - dev split - run `20261006T103630Z-61014d0-3463571a-linear_indicators__f-93__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10324 | 10946 | 11486 |
| n_eff of the scored moves (n scored // bars ahead) | 1032 | 729 | 574 |
| true up-rate | 0.5120 | 0.5097 | 0.5145 |
| calls up (predicted up-rate) | 0.5646 | 0.4520 | 0.6086 |
| accuracy | 0.5187 | 0.5033 | 0.5050 |
| balanced accuracy | 0.5172 | 0.5042 | 0.5018 |
| precision (up) | 0.5272 | 0.5143 | 0.5160 |
| recall / sensitivity (up) | 0.5813 | 0.4562 | 0.6103 |
| specificity (down) | 0.4530 | 0.5523 | 0.3933 |
| F1 (up) | 0.5529 | 0.4835 | 0.5592 |
| MCC | 0.0346 | 0.0085 | 0.0037 |
| AUC | 0.5273 | 0.5054 | 0.4923 |
| Brier | 0.2574 | 0.2628 | 0.2608 |
| ECE (positive class) | 0.0669 | 0.0811 | 0.0645 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0120 | 0.0097 | 0.0145 |
| TP / FP / TN / FN | 3073 / 2756 / 2282 / 2213 | 2545 / 2403 / 2964 / 3034 | 3607 / 3383 / 2193 / 2303 |
| Gaussian readout: calls up | 0.4992 | 0.4211 | 0.3951 |
| Gaussian readout: MCC | 0.0442 | 0.0259 | 0.0346 |
| Gaussian readout: AUC | 0.5341 | 0.5143 | 0.5187 |
| Gaussian readout: Brier | 0.2494 | 0.2512 | 0.2522 |
| Gaussian readout: ECE | 0.0124 | 0.0204 | 0.0333 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 55.43 | 68.32 | 79.57 |
| RMSE ($), raw heads | 55.43 | 68.32 | 79.57 |
| RMSE ($), zero prediction | 55.38 | 68.23 | 79.34 |
| MAE ($), served | 34.90 | 42.64 | 49.59 |
| MAE ($), raw heads | 34.90 | 42.64 | 49.59 |
| MAE ($), zero prediction | 34.92 | 42.60 | 49.49 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0016 | -0.0026 | -0.0058 |
| skill vs zero, raw heads | -0.0016 | -0.0026 | -0.0058 |
| EV, served | -0.0018 | -0.0026 | -0.0052 |
| EV, raw heads | -0.0018 | -0.0026 | -0.0052 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0124 | -0.0128 | -0.0254 |
| corr, Spearman, raw heads | 0.0257 | 0.0124 | 0.0173 |
| mean predicted ($), served | 0.17 | -0.01 | -0.59 |
| mean predicted ($), raw heads | 0.17 | -0.01 | -0.59 |
| mean realised ($) | 1.67 | 2.49 | 3.33 |
| share predicted up, raw heads | 0.5025 | 0.4159 | 0.3882 |
| share realised up | 0.4982 | 0.5021 | 0.5025 |
| shrink beta (served = beta x raw, fit on cal) | 1.0000 | 1.0000 | 1.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 27.30 | 33.39 | 38.96 |
| CRPSS vs constant variance | 0.0292 | 0.0283 | 0.0235 |
| NLL | 6.3268 | 6.4553 | 6.7389 |
| PIT KS | 0.1070 | 0.0986 | 0.1056 |
| var / err^2 Spearman | 0.3275 | 0.3283 | 0.3311 |
| coverage of the 90% interval | 0.8952 | 0.8939 | 0.8917 |
| width of the 90% interval ($) | 153.23 | 185.86 | 213.63 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0309 | [0.0037, 0.0598] | WORKS |
| h1 | 0.0022 | [-0.0196, 0.0257] | NOISE |
| h2 | -0.0251 | [-0.0591, 0.0044] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 1.000 / h1 1.000 / h2 1.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6206 | 0.6206 | 0.5973 |
| abs(d h1) <= abs(d h2) | 0.7346 | 0.7346 | 0.5845 |
| full chain h0 <= h1 <= h2 | 0.4253 | 0.4253 | 0.3105 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6390 | 0.6336 | 0.6073 | 0.2947 |
| expected if the two signs were independent | 0.5003 | 0.5073 | 0.4757 | 0.1605 |

- P(up) unanimity (all three horizons call the same side): 0.3936

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0346 vs 0.0593 (-0.0247): does not beat, noise (boot z -1.26) | 0.0085 vs 0.0353 (-0.0268): does not beat, noise (boot z -1.44) | 0.0037 vs 0.0287 (-0.0250): does not beat, noise (boot z -1.28) |
| logreg_lags | direction/auc | 0.5273 vs 0.5347 (-0.0075): does not beat, noise (boot z -0.62) | 0.5054 vs 0.5204 (-0.0150): does not beat, noise (boot z -1.35) | 0.4923 vs 0.5116 (-0.0193): does not beat, noise (boot z -1.70) |
| logreg_lags | direction/brier | 0.2574 vs 0.2515 (-0.0059): does not beat, significantly worse (DM z -2.57) | 0.2628 vs 0.2531 (-0.0098): does not beat, significantly worse (DM z -5.32) | 0.2608 vs 0.2539 (-0.0068): does not beat, significantly worse (DM z -3.42) |
| logreg_lags | direction/ece_pos | 0.0669 vs 0.0169 (-0.0500): does not beat, significantly worse (boot z -4.27) | 0.0811 vs 0.0293 (-0.0518): does not beat, significantly worse (boot z -6.02) | 0.0645 vs 0.0324 (-0.0320): does not beat, significantly worse (boot z -3.22) |
| logreg_lags | direction/acc | 0.5187 vs 0.5302 (-0.0115): does not beat, noise (DM z -1.15) | 0.5033 vs 0.5182 (-0.0149): does not beat, noise (DM z -1.47) | 0.5050 vs 0.5163 (-0.0113): does not beat, noise (DM z -1.10) |
| logreg_lags | direction/bal_acc | 0.5172 vs 0.5296 (-0.0125): does not beat, noise (boot z -1.27) | 0.5042 vs 0.5176 (-0.0134): does not beat, noise (boot z -1.44) | 0.5018 vs 0.5142 (-0.0124): does not beat, noise (boot z -1.29) |
| class_prior | direction/mcc | 0.0346 vs 0.0000 (+0.0346): beats (boot z +1.99) | 0.0085 vs 0.0000 (+0.0085): beats, noise (boot z +0.49) | 0.0037 vs 0.0000 (+0.0037): beats, noise (boot z +0.17) |
| class_prior | direction/auc | 0.5273 vs 0.5000 (+0.0273): beats (boot z +2.36) | 0.5054 vs 0.5000 (+0.0054): beats, noise (boot z +0.51) | 0.4923 vs 0.5000 (-0.0077): does not beat, noise (boot z -0.52) |
| class_prior | direction/brier | 0.2574 vs 0.2499 (-0.0075): does not beat, significantly worse (DM z -2.83) | 0.2628 vs 0.2499 (-0.0129): does not beat, significantly worse (DM z -5.75) | 0.2608 vs 0.2498 (-0.0110): does not beat, significantly worse (DM z -4.16) |
| class_prior | direction/ece_pos | 0.0669 vs 0.0060 (-0.0609): does not beat, significantly worse (boot z -4.53) | 0.0811 vs 0.0044 (-0.0766): does not beat, significantly worse (boot z -6.17) | 0.0645 vs 0.0064 (-0.0581): does not beat, significantly worse (boot z -3.79) |
| class_prior | direction/acc | 0.5187 vs 0.5120 (+0.0067): beats, noise (DM z +0.47) | 0.5033 vs 0.5097 (-0.0064): does not beat, noise (DM z -0.36) | 0.5050 vs 0.5145 (-0.0096): does not beat, noise (DM z -0.63) |
| class_prior | direction/bal_acc | 0.5172 vs 0.5000 (+0.0172): beats (boot z +1.99) | 0.5042 vs 0.5000 (+0.0042): beats, noise (boot z +0.49) | 0.5018 vs 0.5000 (+0.0018): beats, noise (boot z +0.17) |
| zero_delta | delta/rmse | 55.43 vs 55.38 (-0.04, -0.08%): does not beat, noise (DM z -0.94) | 68.32 vs 68.23 (-0.09, -0.13%): does not beat, noise (DM z -0.97) | 79.57 vs 79.34 (-0.23, -0.29%): does not beat, noise (DM z -1.44) |
| zero_delta | delta/mae | 34.90 vs 34.92 (+0.02, +0.07%): beats, noise (DM z +0.81) | 42.64 vs 42.60 (-0.03, -0.08%): does not beat, noise (DM z -0.62) | 49.59 vs 49.49 (-0.11, -0.21%): does not beat, noise (DM z -1.09) |
| mean_delta | delta/rmse | 55.43 vs 55.38 (-0.05, -0.09%): does not beat, noise (DM z -1.02) | 68.32 vs 68.22 (-0.10, -0.14%): does not beat, noise (DM z -1.06) | 79.57 vs 79.32 (-0.24, -0.31%): does not beat, noise (DM z -1.50) |
| mean_delta | delta/mae | 34.90 vs 34.92 (+0.02, +0.07%): beats, noise (DM z +0.83) | 42.64 vs 42.60 (-0.04, -0.08%): does not beat, noise (DM z -0.63) | 49.59 vs 49.49 (-0.11, -0.22%): does not beat, noise (DM z -1.09) |
| const_var | variance/crps | 27.30 vs 28.12 (+0.82, +2.92%): beats (DM z +9.98) | 33.39 vs 34.37 (+0.97, +2.83%): beats (DM z +8.93) | 38.96 vs 39.90 (+0.94, +2.35%): beats (DM z +7.00) |
| const_var | variance/nll | 6.3268 vs 7.4610 (+1.1343): beats (DM z +6.28) | 6.4553 vs 7.6534 (+1.1981): beats (DM z +6.20) | 6.7389 vs 7.8083 (+1.0694): beats (DM z +5.52) |
| const_var | variance/pit_ks | 0.1070 vs 0.1245 (+0.0174): beats (boot z +4.66) | 0.0986 vs 0.1208 (+0.0222): beats (boot z +5.54) | 0.1056 vs 0.1242 (+0.0185): beats (boot z +4.83) |
| const_var | variance/corr_var_err2_spearman | 0.3275 vs 0.0000 (+0.3275): beats (boot z +12.08) | 0.3283 vs 0.0000 (+0.3283): beats (boot z +10.67) | 0.3311 vs 0.0000 (+0.3311): beats (boot z +10.54) |

## Backtest (costs included)

- n_trades: 944
- total_return: 0.0369
- sharpe_net: 3.3790
- sharpe_gross: 3.3790
- sortino: 5.0578
- max_drawdown: 0.0925
- hit_rate: 0.5064
- hit_rate_gross: 0.5064
- profit_factor: 1.0506
- avg_hold_bars: 7.1663
- exposure: 0.4555
- turnover: 1911.5873
- fees_paid: 0.0000
- traded_notional: 19115861.2556
- breakeven_cost_bps: 0.3863
- gross_edge_per_trade_bps: 0.4084
- costs_paid: 0.0000
- gross_pnl: 369.2277
- net_pnl: 369.2277

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 30.7198, indicator 21.6138 (clip 20).
Clipped steps over the run: main 4.0000, indicator 1.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 30.7198 / 12.7344 | 4.0000 / 0.0000 | 6.9% / 0.0% | 0.0000 | 2763.0000 / 3243.0000 / 3538.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 4.5189 / 5.8863 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3263.0000 / 3763.0000 / 4132.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 4.1213 / 21.6138 | 0.0000 / 1.0000 | 0.0% / 1.7% | 0.0000 | 3233.0000 / 3789.0000 / 4047.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 3.5037 / 7.3404 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3163.0000 / 3666.0000 / 3994.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 7.0613 / 9.8263 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2758.0000 / 3194.0000 / 3497.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 9.9929 / 16.4371 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2684.0000 / 3159.0000 / 3405.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 4.0000 / 7.3661 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3162.0000 / 3762.0000 / 4092.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 5.7799 / 5.8730 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3188.0000 / 3720.0000 / 4056.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 4.3972 / 16.1707 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3135.0000 / 3687.0000 / 4079.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 5.0733 / 4.2932 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2705.0000 / 3155.0000 / 3448.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 5.1385 / 8.1190 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2693.0000 / 3146.0000 / 3493.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 5.7426 / 2.5446 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3212.0000 / 3775.0000 / 4146.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 13.8154 / 10.5767 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3204.0000 / 3754.0000 / 4090.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 7.0087 / 5.1025 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3258.0000 / 3731.0000 / 4042.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
Learned periods sitting at their configured bound: period/keltner_0_atr_period=2, period/keltner_0_period=60, period/keltner_1_atr_period=2, period/keltner_2_atr_period=59.9911, period/macd_0_slow=60, period/obv_period_1=59.9692, period/obv_period_2=60.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=1.015 (corr skip/tower=-0.575), h1=0.848 (corr skip/tower=-0.270), h2=0.806 (corr skip/tower=-0.622).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -93 (TimeSeriesSplit fold 8, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-13T01:15:00 .. 2023-02-23T08:45:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.9296, long_above 0.5819, short_below 0.4473, median 0.5132. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +3.69% | +3.38 | +9.25% | 944 |
| buy and hold | +11.40% | +7.64 | +7.31% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -8.29% .. +9.19%) | -0.15% | -0.15 | | |

The random null enters at the strategy's rate (0.1167 per flat bar), holds 7 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 77% of its seeds on net return, 72% on net Sharpe and 77% on gross return.
