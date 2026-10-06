# Evaluation report - dev split - run `20261006T102426Z-61014d0-314b70d4-linear_indicators__f-95__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10047 | 10743 | 11232 |
| n_eff of the scored moves (n scored // bars ahead) | 1004 | 716 | 561 |
| true up-rate | 0.5088 | 0.5070 | 0.5105 |
| calls up (predicted up-rate) | 0.5672 | 0.3839 | 0.6694 |
| accuracy | 0.5309 | 0.4960 | 0.5222 |
| balanced accuracy | 0.5297 | 0.4976 | 0.5186 |
| precision (up) | 0.5350 | 0.5039 | 0.5244 |
| recall / sensitivity (up) | 0.5964 | 0.3815 | 0.6877 |
| specificity (down) | 0.4630 | 0.6137 | 0.3496 |
| F1 (up) | 0.5641 | 0.4342 | 0.5950 |
| MCC | 0.0600 | -0.0050 | 0.0396 |
| AUC | 0.5409 | 0.4968 | 0.5279 |
| Brier | 0.2563 | 0.2597 | 0.2550 |
| ECE (positive class) | 0.0526 | 0.0753 | 0.0513 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0088 | 0.0070 | 0.0105 |
| TP / FP / TN / FN | 3049 / 2650 / 2285 / 2063 | 2078 / 2046 / 3250 / 3369 | 3943 / 3576 / 1922 / 1791 |
| Gaussian readout: calls up | 0.6879 | 0.5804 | 0.4808 |
| Gaussian readout: MCC | 0.0755 | 0.0731 | 0.1028 |
| Gaussian readout: AUC | 0.5496 | 0.5467 | 0.5696 |
| Gaussian readout: Brier | 0.2482 | 0.2485 | 0.2493 |
| Gaussian readout: ECE | 0.0120 | 0.0191 | 0.0462 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 45.59 | 55.66 | 63.63 |
| RMSE ($), raw heads | 45.58 | 55.66 | 63.49 |
| RMSE ($), zero prediction | 45.63 | 55.69 | 63.66 |
| MAE ($), served | 28.71 | 35.05 | 40.40 |
| MAE ($), raw heads | 28.69 | 35.01 | 40.21 |
| MAE ($), zero prediction | 28.77 | 35.11 | 40.44 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0018 | 0.0011 | 0.0008 |
| skill vs zero, raw heads | 0.0021 | 0.0010 | 0.0053 |
| EV, served | 0.0016 | 0.0008 | 0.0008 |
| EV, raw heads | 0.0019 | 0.0007 | 0.0052 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0438 | 0.0311 | 0.0727 |
| corr, Spearman, raw heads | 0.0727 | 0.0632 | 0.1048 |
| mean predicted ($), served | 0.49 | 0.36 | 0.02 |
| mean predicted ($), raw heads | 0.97 | 0.86 | 0.21 |
| mean realised ($) | 0.71 | 1.07 | 1.42 |
| share predicted up, raw heads | 0.6905 | 0.5817 | 0.4755 |
| share realised up | 0.4979 | 0.4947 | 0.4994 |
| shrink beta (served = beta x raw, fit on cal) | 0.5039 | 0.4221 | 0.0957 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 22.28 | 27.22 | 31.53 |
| CRPSS vs constant variance | 0.0021 | 0.0038 | -0.0012 |
| NLL | 6.0848 | 6.2061 | 6.4643 |
| PIT KS | 0.0961 | 0.0913 | 0.0943 |
| var / err^2 Spearman | 0.1375 | 0.2005 | 0.2061 |
| coverage of the 90% interval | 0.9091 | 0.9110 | 0.9128 |
| width of the 90% interval ($) | 139.11 | 172.43 | 203.47 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0271 | [-0.0002, 0.0563] | NOISE |
| h1 | -0.0106 | [-0.0365, 0.0161] | NOISE |
| h2 | 0.0209 | [-0.0092, 0.0542] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.504 / h1 0.422 / h2 0.096) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5321 | 0.4634 | 0.6071 |
| abs(d h1) <= abs(d h2) | 0.6360 | 0.1634 | 0.5872 |
| full chain h0 <= h1 <= h2 | 0.2957 | 0.0219 | 0.3254 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6225 | 0.5294 | 0.6184 | 0.2348 |
| expected if the two signs were independent | 0.5221 | 0.4808 | 0.4918 | 0.1466 |

- P(up) unanimity (all three horizons call the same side): 0.3812

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0600 vs 0.0878 (-0.0279): does not beat, noise (boot z -1.14) | -0.0050 vs 0.1036 (-0.1086): does not beat, significantly worse (boot z -4.62) | 0.0396 vs 0.0915 (-0.0519): does not beat, significantly worse (boot z -2.13) |
| logreg_lags | direction/auc | 0.5409 vs 0.5646 (-0.0237): does not beat, noise (boot z -1.62) | 0.4968 vs 0.5749 (-0.0781): does not beat, significantly worse (boot z -4.94) | 0.5279 vs 0.5760 (-0.0481): does not beat, significantly worse (boot z -3.23) |
| logreg_lags | direction/brier | 0.2563 vs 0.2478 (-0.0086): does not beat, significantly worse (DM z -3.50) | 0.2597 vs 0.2466 (-0.0131): does not beat, significantly worse (DM z -5.28) | 0.2550 vs 0.2460 (-0.0090): does not beat, significantly worse (DM z -3.71) |
| logreg_lags | direction/ece_pos | 0.0526 vs 0.0151 (-0.0375): does not beat, significantly worse (boot z -3.18) | 0.0753 vs 0.0181 (-0.0572): does not beat, significantly worse (boot z -3.94) | 0.0513 vs 0.0083 (-0.0430): does not beat, significantly worse (boot z -3.32) |
| logreg_lags | direction/acc | 0.5309 vs 0.5446 (-0.0137): does not beat, noise (DM z -1.21) | 0.4960 vs 0.5521 (-0.0561): does not beat, significantly worse (DM z -4.17) | 0.5222 vs 0.5468 (-0.0247): does not beat, significantly worse (DM z -2.22) |
| logreg_lags | direction/bal_acc | 0.5297 vs 0.5422 (-0.0125): does not beat, noise (boot z -1.05) | 0.4976 vs 0.5505 (-0.0529): does not beat, significantly worse (boot z -4.62) | 0.5186 vs 0.5439 (-0.0253): does not beat, significantly worse (boot z -2.18) |
| class_prior | direction/mcc | 0.0600 vs 0.0000 (+0.0600): beats (boot z +3.17) | -0.0050 vs 0.0000 (-0.0050): does not beat, noise (boot z -0.32) | 0.0396 vs 0.0000 (+0.0396): beats, noise (boot z +1.86) |
| class_prior | direction/auc | 0.5409 vs 0.5000 (+0.0409): beats (boot z +3.36) | 0.4968 vs 0.5000 (-0.0032): does not beat, noise (boot z -0.31) | 0.5279 vs 0.5000 (+0.0279): beats, noise (boot z +1.94) |
| class_prior | direction/brier | 0.2563 vs 0.2499 (-0.0064): does not beat, significantly worse (DM z -2.41) | 0.2597 vs 0.2500 (-0.0098): does not beat, significantly worse (DM z -4.78) | 0.2550 vs 0.2499 (-0.0051): does not beat, significantly worse (DM z -1.97) |
| class_prior | direction/ece_pos | 0.0526 vs 0.0002 (-0.0524): does not beat, significantly worse (boot z -4.98) | 0.0753 vs 0.0002 (-0.0751): does not beat, significantly worse (boot z -6.47) | 0.0513 vs 0.0011 (-0.0502): does not beat, significantly worse (boot z -3.80) |
| class_prior | direction/acc | 0.5309 vs 0.5088 (+0.0221): beats, noise (DM z +1.52) | 0.4960 vs 0.5070 (-0.0111): does not beat, noise (DM z -0.56) | 0.5222 vs 0.5105 (+0.0117): beats, noise (DM z +0.80) |
| class_prior | direction/bal_acc | 0.5297 vs 0.5000 (+0.0297): beats (boot z +3.17) | 0.4976 vs 0.5000 (-0.0024): does not beat, noise (boot z -0.32) | 0.5186 vs 0.5000 (+0.0186): beats, noise (boot z +1.86) |
| zero_delta | delta/rmse | 45.59 vs 45.63 (+0.04, +0.09%): beats, noise (DM z +1.71) | 55.66 vs 55.69 (+0.03, +0.05%): beats, noise (DM z +0.98) | 63.63 vs 63.66 (+0.03, +0.04%): beats, noise (DM z +1.86) |
| zero_delta | delta/mae | 28.71 vs 28.77 (+0.06, +0.22%): beats (DM z +3.07) | 35.05 vs 35.11 (+0.06, +0.17%): beats (DM z +2.67) | 40.40 vs 40.44 (+0.03, +0.08%): beats (DM z +3.83) |
| mean_delta | delta/rmse | 45.59 vs 45.63 (+0.04, +0.08%): beats, noise (DM z +1.64) | 55.66 vs 55.68 (+0.02, +0.04%): beats, noise (DM z +0.63) | 63.63 vs 63.64 (+0.01, +0.02%): beats, noise (DM z +0.41) |
| mean_delta | delta/mae | 28.71 vs 28.77 (+0.06, +0.22%): beats (DM z +3.57) | 35.05 vs 35.12 (+0.07, +0.19%): beats (DM z +3.12) | 40.40 vs 40.44 (+0.03, +0.08%): beats, noise (DM z +1.42) |
| const_var | variance/crps | 22.28 vs 22.33 (+0.05, +0.21%): beats, noise (DM z +1.39) | 27.22 vs 27.32 (+0.10, +0.38%): beats (DM z +2.39) | 31.53 vs 31.49 (-0.04, -0.12%): does not beat, noise (DM z -0.77) |
| const_var | variance/nll | 6.0848 vs 6.2119 (+0.1271): beats, noise (DM z +1.12) | 6.2061 vs 6.3726 (+0.1665): beats, noise (DM z +1.51) | 6.4643 vs 6.4597 (-0.0046): does not beat, noise (DM z -0.05) |
| const_var | variance/pit_ks | 0.0961 vs 0.0879 (-0.0082): does not beat, significantly worse (boot z -4.97) | 0.0913 vs 0.0814 (-0.0098): does not beat, significantly worse (boot z -4.38) | 0.0943 vs 0.0786 (-0.0156): does not beat, significantly worse (boot z -4.30) |
| const_var | variance/corr_var_err2_spearman | 0.1375 vs 0.0000 (+0.1375): beats (boot z +5.19) | 0.2005 vs 0.0000 (+0.2005): beats (boot z +7.32) | 0.2061 vs 0.0000 (+0.2061): beats (boot z +7.41) |

## Backtest (costs included)

- n_trades: 733
- total_return: 0.0686
- sharpe_net: 7.0608
- sharpe_gross: 7.0608
- sortino: 10.3943
- max_drawdown: 0.0433
- hit_rate: 0.5143
- hit_rate_gross: 0.5143
- profit_factor: 1.1371
- avg_hold_bars: 9.0900
- exposure: 0.4487
- turnover: 1508.0657
- fees_paid: 0.0000
- traded_notional: 15080538.0999
- breakeven_cost_bps: 0.9097
- gross_edge_per_trade_bps: 0.9279
- costs_paid: 0.0000
- gross_pnl: 685.9294
- net_pnl: 685.9294

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 30.1840, indicator 28.0696 (clip 20).
Clipped steps over the run: main 4.0000, indicator 1.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 30.1840 / 5.1152 | 4.0000 / 0.0000 | 13.8% / 0.0% | 0.0000 | 1516.0000 / 1721.0000 / 1859.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 9.8678 / 5.3663 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2038.0000 / 2299.0000 / 2479.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 3.3051 / 3.3087 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2036.0000 / 2337.0000 / 2525.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 4.7965 / 4.1008 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2013.0000 / 2310.0000 / 2493.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 3.4835 / 7.4350 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2063.0000 / 2359.0000 / 2513.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 7.3290 / 4.1825 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2034.0000 / 2349.0000 / 2490.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 3.2366 / 5.4647 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2009.0000 / 2333.0000 / 2488.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 4.4888 / 3.6139 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2060.0000 / 2313.0000 / 2508.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 4.7541 / 1.8100 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1661.0000 / 1922.0000 / 2116.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 4.0688 / 10.7822 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1516.0000 / 1791.0000 / 1924.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 3.5535 / 5.1644 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1478.0000 / 1764.0000 / 1893.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 9.6381 / 28.0696 | 0.0000 / 1.0000 | 0.0% / 3.4% | 0.0000 | 2055.0000 / 2388.0000 / 2552.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 3.2211 / 3.8409 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2023.0000 / 2352.0000 / 2496.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 4.4434 / 3.6202 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2020.0000 / 2337.0000 / 2528.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
Learned periods sitting at their configured bound: period/keltner_2_period=60, period/macd_0_slow=60, period/obv_period_2=60, period/vwap_period_2=60.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.910 (corr skip/tower=-0.565), h1=0.472 (corr skip/tower=-0.377), h2=0.585 (corr skip/tower=-0.535).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -95 (TimeSeriesSplit fold 6, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-01-23T10:13:00 .. 2023-02-02T17:43:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8286, long_above 0.5629, short_below 0.4332, median 0.4948. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +6.86% | +7.06 | +4.33% | 733 |
| buy and hold | +4.56% | +3.67 | +5.61% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -7.55% .. +7.88%) | +0.31% | +0.39 | | |

The random null enters at the strategy's rate (0.0895 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 91% of its seeds on net return, 88% on net Sharpe and 91% on gross return.
