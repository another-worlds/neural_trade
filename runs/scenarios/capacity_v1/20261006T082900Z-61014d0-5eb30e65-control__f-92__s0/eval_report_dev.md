# Evaluation report - dev split - run `20261006T082900Z-61014d0-5eb30e65-control__f-92__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 9392 | 10127 | 10722 |
| n_eff of the scored moves (n scored // bars ahead) | 939 | 675 | 536 |
| true up-rate | 0.4856 | 0.4794 | 0.4808 |
| calls up (predicted up-rate) | 0.5636 | 0.5016 | 0.5585 |
| accuracy | 0.4876 | 0.5200 | 0.4983 |
| balanced accuracy | 0.4895 | 0.5201 | 0.5006 |
| precision (up) | 0.4763 | 0.4994 | 0.4813 |
| recall / sensitivity (up) | 0.5527 | 0.5226 | 0.5591 |
| specificity (down) | 0.4262 | 0.5176 | 0.4421 |
| F1 (up) | 0.5117 | 0.5107 | 0.5173 |
| MCC | -0.0212 | 0.0402 | 0.0011 |
| AUC | 0.4857 | 0.5318 | 0.5098 |
| Brier | 0.2688 | 0.2518 | 0.2594 |
| ECE (positive class) | 0.1018 | 0.0478 | 0.0822 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0144 | 0.0206 | 0.0192 |
| TP / FP / TN / FN | 2521 / 2772 / 2059 / 2040 | 2537 / 2543 / 2729 / 2318 | 2882 / 3106 / 2461 / 2273 |
| Gaussian readout: calls up | 0.6776 | 0.6250 | 0.6126 |
| Gaussian readout: MCC | 0.0080 | 0.0265 | 0.0227 |
| Gaussian readout: AUC | 0.5274 | 0.5329 | 0.5312 |
| Gaussian readout: Brier | 0.2499 | 0.2495 | 0.2496 |
| Gaussian readout: ECE | 0.0154 | 0.0267 | 0.0220 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 43.33 | 54.39 | 63.47 |
| RMSE ($), raw heads | 43.58 | 54.78 | 64.30 |
| RMSE ($), zero prediction | 43.32 | 54.31 | 63.40 |
| MAE ($), served | 26.27 | 31.96 | 36.78 |
| MAE ($), raw heads | 26.40 | 32.13 | 37.20 |
| MAE ($), zero prediction | 26.28 | 31.98 | 36.80 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | -0.0030 | -0.0021 |
| skill vs zero, raw heads | -0.0121 | -0.0174 | -0.0285 |
| EV, served | -0.0000 | -0.0023 | -0.0017 |
| EV, raw heads | -0.0081 | -0.0147 | -0.0248 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0030 | -0.0273 | -0.0386 |
| corr, Spearman, raw heads | 0.0289 | 0.0294 | 0.0163 |
| mean predicted ($), served | 0.05 | 0.42 | 0.28 |
| mean predicted ($), raw heads | 1.69 | 1.43 | 1.96 |
| mean realised ($) | -1.39 | -2.08 | -2.78 |
| share predicted up, raw heads | 0.6540 | 0.5962 | 0.5939 |
| share realised up | 0.4734 | 0.4754 | 0.4770 |
| shrink beta (served = beta x raw, fit on cal) | 0.0281 | 0.2921 | 0.1440 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 19.85 | 24.26 | 28.02 |
| CRPSS vs constant variance | 0.0345 | 0.0332 | 0.0288 |
| NLL | 5.3692 | 5.6784 | 5.9843 |
| PIT KS | 0.0291 | 0.0339 | 0.0311 |
| var / err^2 Spearman | 0.3064 | 0.3071 | 0.2831 |
| coverage of the 90% interval | 0.9029 | 0.8998 | 0.8976 |
| width of the 90% interval ($) | 120.16 | 145.33 | 167.12 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0142 | [-0.0460, 0.0127] | NOISE |
| h1 | 0.0421 | [0.0131, 0.0680] | WORKS |
| h2 | 0.0272 | [-0.0052, 0.0581] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.028 / h1 0.292 / h2 0.144) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7394 | 0.9842 | 0.5983 |
| abs(d h1) <= abs(d h2) | 0.7344 | 0.2507 | 0.5786 |
| full chain h0 <= h1 <= h2 | 0.5165 | 0.2359 | 0.3054 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7140 | 0.7612 | 0.7595 | 0.4872 |
| expected if the two signs were independent | 0.5109 | 0.4968 | 0.5070 | 0.2421 |

- P(up) unanimity (all three horizons call the same side): 0.5500

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0212 vs 0.0646 (-0.0858): does not beat, significantly worse (boot z -2.99) | 0.0402 vs 0.0750 (-0.0348): does not beat, noise (boot z -1.57) | 0.0011 vs 0.0755 (-0.0743): does not beat, significantly worse (boot z -2.60) |
| logreg_lags | direction/auc | 0.4857 vs 0.5596 (-0.0739): does not beat, significantly worse (boot z -3.95) | 0.5318 vs 0.5695 (-0.0378): does not beat, significantly worse (boot z -2.86) | 0.5098 vs 0.5730 (-0.0632): does not beat, significantly worse (boot z -3.68) |
| logreg_lags | direction/brier | 0.2688 vs 0.2488 (-0.0200): does not beat, significantly worse (DM z -5.44) | 0.2518 vs 0.2491 (-0.0027): does not beat, noise (DM z -1.33) | 0.2594 vs 0.2489 (-0.0105): does not beat, significantly worse (DM z -3.66) |
| logreg_lags | direction/ece_pos | 0.1018 vs 0.0315 (-0.0703): does not beat, significantly worse (boot z -4.52) | 0.0478 vs 0.0409 (-0.0070): does not beat, noise (boot z -0.54) | 0.0822 vs 0.0445 (-0.0377): does not beat, significantly worse (boot z -2.21) |
| logreg_lags | direction/acc | 0.4876 vs 0.5244 (-0.0367): does not beat, significantly worse (DM z -2.47) | 0.5200 vs 0.5260 (-0.0060): does not beat, noise (DM z -0.46) | 0.4983 vs 0.5237 (-0.0254): does not beat, noise (DM z -1.62) |
| logreg_lags | direction/bal_acc | 0.4895 vs 0.5299 (-0.0404): does not beat, significantly worse (boot z -2.92) | 0.5201 vs 0.5344 (-0.0143): does not beat, noise (boot z -1.32) | 0.5006 vs 0.5330 (-0.0325): does not beat, significantly worse (boot z -2.43) |
| class_prior | direction/mcc | -0.0212 vs 0.0000 (-0.0212): does not beat, noise (boot z -0.99) | 0.0402 vs 0.0000 (+0.0402): beats, noise (boot z +1.86) | 0.0011 vs 0.0000 (+0.0011): beats, noise (boot z +0.05) |
| class_prior | direction/auc | 0.4857 vs 0.5000 (-0.0143): does not beat, noise (boot z -1.06) | 0.5318 vs 0.5000 (+0.0318): beats (boot z +2.39) | 0.5098 vs 0.5000 (+0.0098): beats, noise (boot z +0.73) |
| class_prior | direction/brier | 0.2688 vs 0.2505 (-0.0183): does not beat, significantly worse (DM z -6.01) | 0.2518 vs 0.2506 (-0.0012): does not beat, noise (DM z -0.64) | 0.2594 vs 0.2507 (-0.0086): does not beat, significantly worse (DM z -3.57) |
| class_prior | direction/ece_pos | 0.1018 vs 0.0260 (-0.0757): does not beat, significantly worse (boot z -4.93) | 0.0478 vs 0.0322 (-0.0156): does not beat, noise (boot z -1.28) | 0.0822 vs 0.0332 (-0.0490): does not beat, significantly worse (boot z -2.94) |
| class_prior | direction/acc | 0.4876 vs 0.4856 (+0.0020): beats, noise (DM z +0.13) | 0.5200 vs 0.4794 (+0.0406): beats (DM z +2.16) | 0.4983 vs 0.4808 (+0.0175): beats, noise (DM z +0.94) |
| class_prior | direction/bal_acc | 0.4895 vs 0.5000 (-0.0105): does not beat, noise (boot z -0.99) | 0.5201 vs 0.5000 (+0.0201): beats, noise (boot z +1.86) | 0.5006 vs 0.5000 (+0.0006): beats, noise (boot z +0.05) |
| zero_delta | delta/rmse | 43.33 vs 43.32 (-0.00, -0.00%): does not beat, noise (DM z -0.56) | 54.39 vs 54.31 (-0.08, -0.15%): does not beat, noise (DM z -1.64) | 63.47 vs 63.40 (-0.07, -0.10%): does not beat, noise (DM z -1.72) |
| zero_delta | delta/mae | 26.27 vs 26.28 (+0.00, +0.01%): beats, noise (DM z +0.73) | 31.96 vs 31.98 (+0.03, +0.09%): beats, noise (DM z +0.87) | 36.78 vs 36.80 (+0.02, +0.04%): beats, noise (DM z +0.60) |
| mean_delta | delta/rmse | 43.33 vs 43.34 (+0.01, +0.02%): beats, noise (DM z +1.52) | 54.39 vs 54.33 (-0.06, -0.11%): does not beat, noise (DM z -1.40) | 63.47 vs 63.43 (-0.03, -0.05%): does not beat, noise (DM z -1.09) |
| mean_delta | delta/mae | 26.27 vs 26.29 (+0.02, +0.07%): beats (DM z +3.40) | 31.96 vs 32.01 (+0.05, +0.16%): beats, noise (DM z +1.70) | 36.78 vs 36.83 (+0.05, +0.12%): beats, noise (DM z +1.65) |
| const_var | variance/crps | 19.85 vs 20.56 (+0.71, +3.45%): beats (DM z +9.11) | 24.26 vs 25.09 (+0.83, +3.32%): beats (DM z +7.25) | 28.02 vs 28.85 (+0.83, +2.88%): beats (DM z +5.88) |
| const_var | variance/nll | 5.3692 vs 6.2346 (+0.8654): beats (DM z +4.52) | 5.6784 vs 6.5179 (+0.8395): beats (DM z +4.22) | 5.9843 vs 6.6968 (+0.7125): beats (DM z +4.35) |
| const_var | variance/pit_ks | 0.0291 vs 0.0828 (+0.0537): beats (boot z +7.71) | 0.0339 vs 0.0747 (+0.0408): beats (boot z +3.59) | 0.0311 vs 0.0725 (+0.0414): beats (boot z +3.92) |
| const_var | variance/corr_var_err2_spearman | 0.3064 vs 0.0000 (+0.3064): beats (boot z +10.91) | 0.3071 vs 0.0000 (+0.3071): beats (boot z +10.32) | 0.2831 vs 0.0000 (+0.2831): beats (boot z +8.68) |

## Backtest (costs included)

- n_trades: 575
- total_return: 0.0167
- sharpe_net: 2.1057
- sharpe_gross: 2.1057
- sortino: 3.1921
- max_drawdown: 0.0450
- hit_rate: 0.4887
- hit_rate_gross: 0.4887
- profit_factor: 1.0466
- avg_hold_bars: 9.5078
- exposure: 0.3681
- turnover: 1157.5086
- fees_paid: 0.0000
- traded_notional: 11574253.0741
- breakeven_cost_bps: 0.2878
- gross_edge_per_trade_bps: 0.3098
- costs_paid: 0.0000
- gross_pnl: 166.5490
- net_pnl: 166.5490

## Training health

11 epoch(s). Pre-clip gradient norm maximum over the run: main 882.5237, indicator 575.2389 (clip 20).
Clipped steps over the run: main 98.0000, indicator 109.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 43.5957 / 29.1370 | 1.0000 / 1.0000 | 1.4% / 1.4% | 0.0000 | 3512.0000 / 4092.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 19.7966 / 51.3877 | 0.0000 / 1.0000 | 0.0% / 1.4% | 0.0000 | 3452.0000 / 4067.0000 / 4454.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 25.5785 / 69.2465 | 1.0000 / 3.0000 | 1.4% / 4.2% | 0.0000 | 3414.0000 / 4008.0000 / 4421.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 55.3853 / 85.5001 | 1.0000 / 3.0000 | 1.4% / 4.2% | 0.0000 | 3493.0000 / 4116.0000 / 4488.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 17.2593 / 49.5663 | 0.0000 / 6.0000 | 0.0% / 8.3% | 0.0000 | 3444.0000 / 4007.0000 / 4375.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 21.3675 / 21.0781 | 1.0000 / 3.0000 | 1.4% / 4.2% | 0.0000 | 3468.0000 / 4051.0000 / 4457.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 882.5237 / 446.3612 | 12.0000 / 20.0000 | 16.7% / 27.8% | 0.0000 | 3459.0000 / 4101.0000 / 4492.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 36.7685 / 126.3413 | 17.0000 / 14.0000 | 23.6% / 19.4% | 0.0000 | 3495.0000 / 4140.0000 / 4527.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 203.9247 / 575.2389 | 13.0000 / 19.0000 | 18.1% / 26.4% | 0.0000 | 3460.0000 / 4074.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 98.5875 / 133.6139 | 23.0000 / 22.0000 | 31.9% / 30.6% | 0.0000 | 3503.0000 / 4005.0000 / 4505.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 92.0729 / 142.8723 | 29.0000 / 17.0000 | 40.3% / 23.6% | 0.0000 | 3507.0000 / 4058.0000 / 4513.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
Learned periods sitting at their configured bound: period/donchian_period_2=60.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=1.298 (corr skip/tower=-0.845), h1=0.283 (corr skip/tower=-0.351), h2=0.668 (corr skip/tower=-0.598).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -92 (TimeSeriesSplit fold 9, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-23T08:46:00 .. 2023-03-05T16:16:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.667, long_above 0.6196, short_below 0.4097, median 0.5047. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +1.67% | +2.11 | +4.50% | 575 |
| buy and hold | -8.52% | -7.39 | +9.73% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -6.57% .. +6.86%) | +0.08% | +0.11 | | |

The random null enters at the strategy's rate (0.0613 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 65% of its seeds on net return, 62% on net Sharpe and 65% on gross return.
