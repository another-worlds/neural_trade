# Evaluation report - dev split - run `20261006T084444Z-61014d0-6a6535c4-gru_small__f-95__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10047 | 10743 | 11232 |
| n_eff of the scored moves (n scored // bars ahead) | 1004 | 716 | 561 |
| true up-rate | 0.5088 | 0.5070 | 0.5105 |
| calls up (predicted up-rate) | 0.4306 | 0.3852 | 0.3832 |
| accuracy | 0.5069 | 0.5287 | 0.5304 |
| balanced accuracy | 0.5081 | 0.5303 | 0.5329 |
| precision (up) | 0.5183 | 0.5464 | 0.5534 |
| recall / sensitivity (up) | 0.4386 | 0.4151 | 0.4154 |
| specificity (down) | 0.5777 | 0.6456 | 0.6504 |
| F1 (up) | 0.4751 | 0.4718 | 0.4746 |
| MCC | 0.0164 | 0.0623 | 0.0677 |
| AUC | 0.5078 | 0.5378 | 0.5444 |
| Brier | 0.2611 | 0.2552 | 0.2551 |
| ECE (positive class) | 0.0743 | 0.0551 | 0.0599 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0088 | 0.0070 | 0.0105 |
| TP / FP / TN / FN | 2242 / 2084 / 2851 / 2870 | 2261 / 1877 / 3419 / 3186 | 2382 / 1922 / 3576 / 3352 |
| Gaussian readout: calls up | 0.6138 | 0.5049 | 0.5465 |
| Gaussian readout: MCC | 0.0475 | 0.0361 | 0.0428 |
| Gaussian readout: AUC | 0.5318 | 0.5236 | 0.5320 |
| Gaussian readout: Brier | 0.2495 | 0.2501 | 0.2493 |
| Gaussian readout: ECE | 0.0141 | 0.0122 | 0.0094 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 45.62 | 55.71 | 63.64 |
| RMSE ($), raw heads | 45.88 | 56.22 | 64.17 |
| RMSE ($), zero prediction | 45.63 | 55.69 | 63.66 |
| MAE ($), served | 28.77 | 35.15 | 40.41 |
| MAE ($), raw heads | 29.03 | 35.65 | 40.88 |
| MAE ($), zero prediction | 28.77 | 35.11 | 40.44 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0005 | -0.0009 | 0.0006 |
| skill vs zero, raw heads | -0.0108 | -0.0193 | -0.0161 |
| EV, served | 0.0004 | -0.0011 | 0.0005 |
| EV, raw heads | -0.0110 | -0.0196 | -0.0166 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0197 | 0.0090 | 0.0226 |
| corr, Spearman, raw heads | 0.0329 | 0.0151 | 0.0327 |
| mean predicted ($), served | 0.14 | 0.22 | 0.17 |
| mean predicted ($), raw heads | 1.19 | 0.78 | 1.46 |
| mean realised ($) | 0.71 | 1.07 | 1.42 |
| share predicted up, raw heads | 0.6027 | 0.4973 | 0.5439 |
| share realised up | 0.4979 | 0.4947 | 0.4994 |
| shrink beta (served = beta x raw, fit on cal) | 0.1196 | 0.2872 | 0.1134 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 22.00 | 27.09 | 31.09 |
| CRPSS vs constant variance | 0.0149 | 0.0085 | 0.0126 |
| NLL | 5.7597 | 6.0478 | 6.1338 |
| PIT KS | 0.0822 | 0.0878 | 0.0767 |
| var / err^2 Spearman | 0.2177 | 0.2052 | 0.2087 |
| coverage of the 90% interval | 0.9065 | 0.9131 | 0.9129 |
| width of the 90% interval ($) | 137.73 | 174.23 | 204.12 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0007 | [-0.0310, 0.0324] | NOISE |
| h1 | 0.0252 | [-0.0037, 0.0537] | NOISE |
| h2 | 0.0329 | [0.0053, 0.0660] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.120 / h1 0.287 / h2 0.113) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6738 | 0.8878 | 0.6071 |
| abs(d h1) <= abs(d h2) | 0.7129 | 0.1755 | 0.5872 |
| full chain h0 <= h1 <= h2 | 0.4453 | 0.0923 | 0.3254 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6806 | 0.6905 | 0.7009 | 0.4035 |
| expected if the two signs were independent | 0.4858 | 0.5006 | 0.4896 | 0.2006 |

- P(up) unanimity (all three horizons call the same side): 0.4895

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0164 vs 0.0878 (-0.0714): does not beat, significantly worse (boot z -2.90) | 0.0623 vs 0.1036 (-0.0413): does not beat, noise (boot z -1.76) | 0.0677 vs 0.0915 (-0.0238): does not beat, noise (boot z -0.80) |
| logreg_lags | direction/auc | 0.5078 vs 0.5646 (-0.0568): does not beat, significantly worse (boot z -3.73) | 0.5378 vs 0.5749 (-0.0371): does not beat, significantly worse (boot z -2.57) | 0.5444 vs 0.5760 (-0.0315): does not beat, significantly worse (boot z -1.97) |
| logreg_lags | direction/brier | 0.2611 vs 0.2478 (-0.0134): does not beat, significantly worse (DM z -4.85) | 0.2552 vs 0.2466 (-0.0086): does not beat, significantly worse (DM z -3.19) | 0.2551 vs 0.2460 (-0.0091): does not beat, significantly worse (DM z -2.96) |
| logreg_lags | direction/ece_pos | 0.0743 vs 0.0151 (-0.0593): does not beat, significantly worse (boot z -3.98) | 0.0551 vs 0.0181 (-0.0370): does not beat, significantly worse (boot z -2.40) | 0.0599 vs 0.0083 (-0.0516): does not beat, significantly worse (boot z -3.03) |
| logreg_lags | direction/acc | 0.5069 vs 0.5446 (-0.0377): does not beat, significantly worse (DM z -2.74) | 0.5287 vs 0.5521 (-0.0234): does not beat, noise (DM z -1.61) | 0.5304 vs 0.5468 (-0.0164): does not beat, noise (DM z -0.95) |
| logreg_lags | direction/bal_acc | 0.5081 vs 0.5422 (-0.0341): does not beat, significantly worse (boot z -2.83) | 0.5303 vs 0.5505 (-0.0202): does not beat, noise (boot z -1.75) | 0.5329 vs 0.5439 (-0.0110): does not beat, noise (boot z -0.76) |
| class_prior | direction/mcc | 0.0164 vs 0.0000 (+0.0164): beats, noise (boot z +0.85) | 0.0623 vs 0.0000 (+0.0623): beats (boot z +3.11) | 0.0677 vs 0.0000 (+0.0677): beats (boot z +2.64) |
| class_prior | direction/auc | 0.5078 vs 0.5000 (+0.0078): beats, noise (boot z +0.63) | 0.5378 vs 0.5000 (+0.0378): beats (boot z +2.91) | 0.5444 vs 0.5000 (+0.0444): beats (boot z +2.71) |
| class_prior | direction/brier | 0.2611 vs 0.2499 (-0.0112): does not beat, significantly worse (DM z -4.72) | 0.2552 vs 0.2500 (-0.0053): does not beat, significantly worse (DM z -2.15) | 0.2551 vs 0.2499 (-0.0052): does not beat, noise (DM z -1.67) |
| class_prior | direction/ece_pos | 0.0743 vs 0.0002 (-0.0741): does not beat, significantly worse (boot z -6.18) | 0.0551 vs 0.0002 (-0.0549): does not beat, significantly worse (boot z -4.44) | 0.0599 vs 0.0011 (-0.0588): does not beat, significantly worse (boot z -4.11) |
| class_prior | direction/acc | 0.5069 vs 0.5088 (-0.0019): does not beat, noise (DM z -0.11) | 0.5287 vs 0.5070 (+0.0217): beats, noise (DM z +1.05) | 0.5304 vs 0.5105 (+0.0199): beats, noise (DM z +0.85) |
| class_prior | direction/bal_acc | 0.5081 vs 0.5000 (+0.0081): beats, noise (boot z +0.85) | 0.5303 vs 0.5000 (+0.0303): beats (boot z +3.11) | 0.5329 vs 0.5000 (+0.0329): beats (boot z +2.65) |
| zero_delta | delta/rmse | 45.62 vs 45.63 (+0.01, +0.02%): beats, noise (DM z +0.53) | 55.71 vs 55.69 (-0.03, -0.05%): does not beat, noise (DM z -0.34) | 63.64 vs 63.66 (+0.02, +0.03%): beats, noise (DM z +0.58) |
| zero_delta | delta/mae | 28.77 vs 28.77 (+0.00, +0.00%): beats, noise (DM z +0.07) | 35.15 vs 35.11 (-0.04, -0.11%): does not beat, noise (DM z -0.81) | 40.41 vs 40.44 (+0.03, +0.07%): beats, noise (DM z +1.05) |
| mean_delta | delta/rmse | 45.62 vs 45.63 (+0.01, +0.01%): beats, noise (DM z +0.31) | 55.71 vs 55.68 (-0.03, -0.06%): does not beat, noise (DM z -0.48) | 63.64 vs 63.64 (+0.01, +0.01%): beats, noise (DM z +0.15) |
| mean_delta | delta/mae | 28.77 vs 28.77 (+0.00, +0.01%): beats, noise (DM z +0.20) | 35.15 vs 35.12 (-0.03, -0.09%): does not beat, noise (DM z -0.64) | 40.41 vs 40.44 (+0.03, +0.07%): beats, noise (DM z +0.92) |
| const_var | variance/crps | 22.00 vs 22.33 (+0.33, +1.49%): beats (DM z +4.82) | 27.09 vs 27.32 (+0.23, +0.85%): beats (DM z +2.90) | 31.09 vs 31.49 (+0.40, +1.26%): beats (DM z +3.70) |
| const_var | variance/nll | 5.7597 vs 6.2119 (+0.4523): beats (DM z +3.00) | 6.0478 vs 6.3726 (+0.3249): beats (DM z +2.26) | 6.1338 vs 6.4597 (+0.3259): beats (DM z +2.26) |
| const_var | variance/pit_ks | 0.0822 vs 0.0879 (+0.0057): beats, noise (boot z +1.49) | 0.0878 vs 0.0814 (-0.0063): does not beat, noise (boot z -1.45) | 0.0767 vs 0.0786 (+0.0019): beats, noise (boot z +0.45) |
| const_var | variance/corr_var_err2_spearman | 0.2177 vs 0.0000 (+0.2177): beats (boot z +8.98) | 0.2052 vs 0.0000 (+0.2052): beats (boot z +7.75) | 0.2087 vs 0.0000 (+0.2087): beats (boot z +7.51) |

## Backtest (costs included)

- n_trades: 560
- total_return: 0.0060
- sharpe_net: 0.8962
- sharpe_gross: 0.8962
- sortino: 1.3194
- max_drawdown: 0.0464
- hit_rate: 0.5125
- hit_rate_gross: 0.5125
- profit_factor: 1.0158
- avg_hold_bars: 9.8982
- exposure: 0.3732
- turnover: 1126.5910
- fees_paid: 0.0000
- traded_notional: 11266159.0848
- breakeven_cost_bps: 0.1069
- gross_edge_per_trade_bps: 0.1255
- costs_paid: 0.0000
- gross_pnl: 60.2017
- net_pnl: 60.2017

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 71.9111, indicator 168.4499 (clip 20).
Clipped steps over the run: main 7.0000, indicator 14.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 71.9111 / 26.9453 | 5.0000 / 1.0000 | 17.2% / 3.4% | 0.0000 | 1516.0000 / 1721.0000 / 1859.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 6.2790 / 29.2273 | 0.0000 / 1.0000 | 0.0% / 3.4% | 0.0000 | 2038.0000 / 2299.0000 / 2479.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 4.3640 / 18.7033 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2036.0000 / 2337.0000 / 2525.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 50.5770 / 91.9838 | 1.0000 / 1.0000 | 3.4% / 3.4% | 0.0000 | 2013.0000 / 2310.0000 / 2493.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 6.1330 / 12.7754 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2063.0000 / 2359.0000 / 2513.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 9.1606 / 13.2949 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2034.0000 / 2349.0000 / 2490.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 8.1868 / 11.9574 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2009.0000 / 2333.0000 / 2488.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 8.2415 / 17.5880 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2060.0000 / 2313.0000 / 2508.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 14.8344 / 40.9852 | 0.0000 / 2.0000 | 0.0% / 6.9% | 0.0000 | 1661.0000 / 1922.0000 / 2116.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 9.3261 / 25.8350 | 0.0000 / 2.0000 | 0.0% / 6.9% | 0.0000 | 1516.0000 / 1791.0000 / 1924.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 13.8977 / 29.2471 | 0.0000 / 1.0000 | 0.0% / 3.4% | 0.0000 | 1478.0000 / 1764.0000 / 1893.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 17.5365 / 54.7866 | 0.0000 / 2.0000 | 0.0% / 6.9% | 0.0000 | 2055.0000 / 2388.0000 / 2552.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 15.3399 / 32.3553 | 0.0000 / 2.0000 | 0.0% / 6.9% | 0.0000 | 2023.0000 / 2352.0000 / 2496.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 27.2930 / 168.4499 | 1.0000 / 2.0000 | 3.4% / 6.9% | 0.0000 | 2020.0000 / 2337.0000 / 2528.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
Learned periods sitting at their configured bound: period/macd_1_slow=60.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.320 (corr skip/tower=-0.447), h1=0.513 (corr skip/tower=-0.253), h2=0.488 (corr skip/tower=-0.331).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -95 (TimeSeriesSplit fold 6, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-01-23T10:13:00 .. 2023-02-02T17:43:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7119, long_above 0.5972, short_below 0.4041, median 0.5005. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +0.60% | +0.90 | +4.64% | 560 |
| buy and hold | +4.56% | +3.67 | +5.61% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -7.79% .. +6.95%) | -0.55% | -0.67 | | |

The random null enters at the strategy's rate (0.0602 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 59% of its seeds on net return, 59% on net Sharpe and 59% on gross return.
