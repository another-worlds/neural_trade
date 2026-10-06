# Evaluation report - dev split - run `20261006T103020Z-61014d0-037741e4-linear_indicators__f-94__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 8849 | 9593 | 9933 |
| n_eff of the scored moves (n scored // bars ahead) | 884 | 639 | 496 |
| true up-rate | 0.4864 | 0.4793 | 0.4801 |
| calls up (predicted up-rate) | 0.5337 | 0.3435 | 0.6240 |
| accuracy | 0.5158 | 0.5283 | 0.5074 |
| balanced accuracy | 0.5167 | 0.5219 | 0.5123 |
| precision (up) | 0.5020 | 0.5111 | 0.4900 |
| recall / sensitivity (up) | 0.5509 | 0.3662 | 0.6368 |
| specificity (down) | 0.4825 | 0.6775 | 0.3879 |
| F1 (up) | 0.5253 | 0.4267 | 0.5538 |
| MCC | 0.0335 | 0.0460 | 0.0255 |
| AUC | 0.5274 | 0.5270 | 0.5280 |
| Brier | 0.2536 | 0.2523 | 0.2537 |
| ECE (positive class) | 0.0474 | 0.0297 | 0.0505 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0136 | 0.0207 | 0.0199 |
| TP / FP / TN / FN | 2371 / 2352 / 2193 / 1933 | 1684 / 1611 / 3384 / 2914 | 3037 / 3161 / 2003 / 1732 |
| Gaussian readout: calls up | 0.5714 | 0.4148 | 0.3505 |
| Gaussian readout: MCC | 0.0758 | 0.0783 | 0.0685 |
| Gaussian readout: AUC | 0.5450 | 0.5559 | 0.5555 |
| Gaussian readout: Brier | 0.2490 | 0.2487 | 0.2489 |
| Gaussian readout: ECE | 0.0217 | 0.0303 | 0.0288 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 35.58 | 43.66 | 50.29 |
| RMSE ($), raw heads | 35.59 | 43.59 | 50.22 |
| RMSE ($), zero prediction | 35.60 | 43.71 | 50.34 |
| MAE ($), served | 22.52 | 27.14 | 30.65 |
| MAE ($), raw heads | 22.52 | 27.08 | 30.58 |
| MAE ($), zero prediction | 22.55 | 27.19 | 30.71 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0012 | 0.0024 | 0.0020 |
| skill vs zero, raw heads | 0.0008 | 0.0054 | 0.0047 |
| EV, served | 0.0017 | 0.0024 | 0.0018 |
| EV, raw heads | 0.0019 | 0.0055 | 0.0037 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0456 | 0.0755 | 0.0629 |
| corr, Spearman, raw heads | 0.0681 | 0.0833 | 0.0800 |
| mean predicted ($), served | 0.21 | 0.02 | -0.13 |
| mean predicted ($), raw heads | 0.48 | 0.07 | -0.60 |
| mean realised ($) | -1.20 | -1.81 | -2.42 |
| share predicted up, raw heads | 0.5641 | 0.4112 | 0.3398 |
| share realised up | 0.4785 | 0.4800 | 0.4785 |
| shrink beta (served = beta x raw, fit on cal) | 0.4331 | 0.2951 | 0.2106 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 17.15 | 20.71 | 23.61 |
| CRPSS vs constant variance | 0.0077 | 0.0123 | 0.0121 |
| NLL | 5.2720 | 5.4326 | 5.5472 |
| PIT KS | 0.0553 | 0.0477 | 0.0426 |
| var / err^2 Spearman | 0.2415 | 0.2873 | 0.2321 |
| coverage of the 90% interval | 0.9088 | 0.9119 | 0.9158 |
| width of the 90% interval ($) | 102.08 | 125.36 | 146.21 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0283 | [-0.0068, 0.0625] | NOISE |
| h1 | 0.0350 | [0.0057, 0.0646] | WORKS |
| h2 | 0.0287 | [-0.0071, 0.0630] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.433 / h1 0.295 / h2 0.211) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5929 | 0.4603 | 0.6023 |
| abs(d h1) <= abs(d h2) | 0.6785 | 0.5383 | 0.5711 |
| full chain h0 <= h1 <= h2 | 0.3815 | 0.2157 | 0.3135 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5803 | 0.5950 | 0.5992 | 0.2343 |
| expected if the two signs were independent | 0.5026 | 0.5274 | 0.4672 | 0.1479 |

- P(up) unanimity (all three horizons call the same side): 0.3639

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0335 vs 0.0988 (-0.0654): does not beat, significantly worse (boot z -2.48) | 0.0460 vs 0.0921 (-0.0461): does not beat, significantly worse (boot z -2.17) | 0.0255 vs 0.0686 (-0.0431): does not beat, noise (boot z -1.74) |
| logreg_lags | direction/auc | 0.5274 vs 0.5568 (-0.0294): does not beat, noise (boot z -1.76) | 0.5270 vs 0.5638 (-0.0368): does not beat, significantly worse (boot z -2.82) | 0.5280 vs 0.5593 (-0.0313): does not beat, significantly worse (boot z -2.34) |
| logreg_lags | direction/brier | 0.2536 vs 0.2483 (-0.0054): does not beat, significantly worse (DM z -2.59) | 0.2523 vs 0.2479 (-0.0044): does not beat, significantly worse (DM z -2.52) | 0.2537 vs 0.2483 (-0.0054): does not beat, significantly worse (DM z -3.81) |
| logreg_lags | direction/ece_pos | 0.0474 vs 0.0242 (-0.0232): does not beat, noise (boot z -1.85) | 0.0297 vs 0.0319 (+0.0022): beats, noise (boot z +0.15) | 0.0505 vs 0.0350 (-0.0155): does not beat, noise (boot z -1.85) |
| logreg_lags | direction/acc | 0.5158 vs 0.5430 (-0.0272): does not beat, significantly worse (DM z -2.14) | 0.5283 vs 0.5389 (-0.0106): does not beat, noise (DM z -0.77) | 0.5074 vs 0.5243 (-0.0169): does not beat, noise (DM z -1.43) |
| logreg_lags | direction/bal_acc | 0.5167 vs 0.5471 (-0.0304): does not beat, significantly worse (boot z -2.37) | 0.5219 vs 0.5445 (-0.0226): does not beat, significantly worse (boot z -2.20) | 0.5123 vs 0.5318 (-0.0195): does not beat, noise (boot z -1.66) |
| class_prior | direction/mcc | 0.0335 vs 0.0000 (+0.0335): beats, noise (boot z +1.55) | 0.0460 vs 0.0000 (+0.0460): beats (boot z +2.70) | 0.0255 vs 0.0000 (+0.0255): beats, noise (boot z +1.11) |
| class_prior | direction/auc | 0.5274 vs 0.5000 (+0.0274): beats, noise (boot z +1.84) | 0.5270 vs 0.5000 (+0.0270): beats (boot z +2.51) | 0.5280 vs 0.5000 (+0.0280): beats, noise (boot z +1.91) |
| class_prior | direction/brier | 0.2536 vs 0.2503 (-0.0033): does not beat, noise (DM z -1.52) | 0.2523 vs 0.2504 (-0.0019): does not beat, noise (DM z -1.14) | 0.2537 vs 0.2505 (-0.0032): does not beat, noise (DM z -1.64) |
| class_prior | direction/ece_pos | 0.0474 vs 0.0224 (-0.0250): does not beat, noise (boot z -1.90) | 0.0297 vs 0.0282 (-0.0015): does not beat, noise (boot z -0.09) | 0.0505 vs 0.0298 (-0.0207): does not beat, significantly worse (boot z -2.48) |
| class_prior | direction/acc | 0.5158 vs 0.4864 (+0.0294): beats, noise (DM z +1.88) | 0.5283 vs 0.4793 (+0.0490): beats (DM z +2.34) | 0.5074 vs 0.4801 (+0.0273): beats, noise (DM z +1.66) |
| class_prior | direction/bal_acc | 0.5167 vs 0.5000 (+0.0167): beats, noise (boot z +1.55) | 0.5219 vs 0.5000 (+0.0219): beats (boot z +2.70) | 0.5123 vs 0.5000 (+0.0123): beats, noise (boot z +1.11) |
| zero_delta | delta/rmse | 35.58 vs 35.60 (+0.02, +0.06%): beats, noise (DM z +0.98) | 43.66 vs 43.71 (+0.05, +0.12%): beats (DM z +2.71) | 50.29 vs 50.34 (+0.05, +0.10%): beats (DM z +2.26) |
| zero_delta | delta/mae | 22.52 vs 22.55 (+0.03, +0.14%): beats (DM z +2.20) | 27.14 vs 27.19 (+0.05, +0.18%): beats (DM z +3.72) | 30.65 vs 30.71 (+0.06, +0.18%): beats (DM z +3.56) |
| mean_delta | delta/rmse | 35.58 vs 35.61 (+0.03, +0.08%): beats, noise (DM z +1.44) | 43.66 vs 43.73 (+0.07, +0.15%): beats (DM z +3.17) | 50.29 vs 50.36 (+0.07, +0.15%): beats (DM z +2.85) |
| mean_delta | delta/mae | 22.52 vs 22.56 (+0.04, +0.18%): beats (DM z +3.01) | 27.14 vs 27.20 (+0.06, +0.23%): beats (DM z +4.12) | 30.65 vs 30.72 (+0.07, +0.24%): beats (DM z +3.61) |
| const_var | variance/crps | 17.15 vs 17.28 (+0.13, +0.77%): beats (DM z +3.25) | 20.71 vs 20.97 (+0.26, +1.23%): beats (DM z +4.90) | 23.61 vs 23.90 (+0.29, +1.21%): beats (DM z +4.72) |
| const_var | variance/nll | 5.2720 vs 5.2417 (-0.0303): does not beat, noise (DM z -0.71) | 5.4326 vs 5.4377 (+0.0051): beats, noise (DM z +0.10) | 5.5472 vs 5.5668 (+0.0195): beats, noise (DM z +0.39) |
| const_var | variance/pit_ks | 0.0553 vs 0.0392 (-0.0161): does not beat, significantly worse (boot z -2.06) | 0.0477 vs 0.0494 (+0.0017): beats, noise (boot z +0.13) | 0.0426 vs 0.0598 (+0.0173): beats, noise (boot z +1.34) |
| const_var | variance/corr_var_err2_spearman | 0.2415 vs 0.0000 (+0.2415): beats (boot z +9.21) | 0.2873 vs 0.0000 (+0.2873): beats (boot z +10.21) | 0.2321 vs 0.0000 (+0.2321): beats (boot z +7.69) |

## Backtest (costs included)

- n_trades: 410
- total_return: 0.0333
- sharpe_net: 5.1355
- sharpe_gross: 5.1355
- sortino: 7.5502
- max_drawdown: 0.0666
- hit_rate: 0.5341
- hit_rate_gross: 0.5341
- profit_factor: 1.1188
- avg_hold_bars: 9.0951
- exposure: 0.2511
- turnover: 858.0787
- fees_paid: 0.0000
- traded_notional: 8580907.7382
- breakeven_cost_bps: 0.7754
- gross_edge_per_trade_bps: 0.8191
- costs_paid: 0.0000
- gross_pnl: 332.6651
- net_pnl: 332.6651

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 30.5815, indicator 49.0829 (clip 20).
Clipped steps over the run: main 5.0000, indicator 3.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 30.5815 / 9.1335 | 4.0000 / 0.0000 | 9.3% / 0.0% | 0.0000 | 2605.0000 / 2952.0000 / 3194.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 4.2821 / 16.0879 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2610.0000 / 3022.0000 / 3243.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 3.9430 / 8.2059 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2526.0000 / 2918.0000 / 3192.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 3.6281 / 2.4224 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3121.0000 / 3591.0000 / 3884.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 4.3088 / 2.5328 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2572.0000 / 2911.0000 / 3162.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 5.4977 / 2.0744 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2588.0000 / 2972.0000 / 3137.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 4.6274 / 3.3306 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2957.0000 / 3360.0000 / 3589.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 3.5016 / 3.1809 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2639.0000 / 2964.0000 / 3218.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 3.9848 / 5.7026 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2587.0000 / 3003.0000 / 3212.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 4.7249 / 15.4299 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2551.0000 / 2956.0000 / 3195.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 4.8924 / 24.3658 | 0.0000 / 2.0000 | 0.0% / 4.7% | 0.0000 | 2523.0000 / 2910.0000 / 3171.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 20.0940 / 49.0829 | 1.0000 / 1.0000 | 2.3% / 2.3% | 0.0000 | 2645.0000 / 2973.0000 / 3210.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 4.0622 / 3.0515 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2602.0000 / 2933.0000 / 3194.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 5.4482 / 11.3446 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3118.0000 / 3548.0000 / 3866.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
Learned periods sitting at their configured bound: period/keltner_0_atr_period=2, period/keltner_2_period=60, period/macd_1_slow=60, period/obv_period_2=60.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.457 (corr skip/tower=-0.556), h1=0.384 (corr skip/tower=-0.327), h2=0.247 (corr skip/tower=-0.641).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -94 (TimeSeriesSplit fold 7, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-02T17:44:00 .. 2023-02-13T01:14:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8451, long_above 0.5667, short_below 0.4234, median 0.4926. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +3.33% | +5.14 | +6.66% | 410 |
| buy and hold | -7.43% | -7.41 | +10.90% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -4.85% .. +5.76%) | +0.57% | +1.18 | | |

The random null enters at the strategy's rate (0.0369 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 78% of its seeds on net return, 68% on net Sharpe and 78% on gross return.
