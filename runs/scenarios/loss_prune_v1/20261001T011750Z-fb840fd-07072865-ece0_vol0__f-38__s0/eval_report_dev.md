# Evaluation report - dev split - run `20261001T011750Z-fb840fd-07072865-ece0_vol0__f-38__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 18047 | 19287 | 19803 |
| n_eff of the scored moves (n scored // bars ahead) | 1804 | 1285 | 990 |
| true up-rate | 0.5140 | 0.5117 | 0.5132 |
| calls up (predicted up-rate) | 0.4978 | 0.5833 | 0.5131 |
| accuracy | 0.4800 | 0.5047 | 0.5072 |
| balanced accuracy | 0.4800 | 0.5027 | 0.5069 |
| precision (up) | 0.4940 | 0.5140 | 0.5199 |
| recall / sensitivity (up) | 0.4784 | 0.5860 | 0.5198 |
| specificity (down) | 0.4816 | 0.4195 | 0.4940 |
| F1 (up) | 0.4861 | 0.5477 | 0.5199 |
| MCC | -0.0400 | 0.0056 | 0.0138 |
| AUC | 0.4718 | 0.5080 | 0.5088 |
| Brier | 0.2768 | 0.2562 | 0.2615 |
| ECE (positive class) | 0.1124 | 0.0563 | 0.0757 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0140 | 0.0117 | 0.0132 |
| TP / FP / TN / FN | 4438 / 4546 / 4224 / 4839 | 5783 / 5467 / 3951 / 4086 | 5283 / 4878 / 4762 / 4880 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.3024 | 0.4487 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | 0.0579 | 0.0664 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.5412 | 0.5472 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2496 | 0.2500 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0179 | 0.0316 |
| Gaussian readout of the raw heads: calls up | 0.5573 | 0.3024 | 0.4487 |
| Gaussian readout of the raw heads: MCC | 0.0089 | 0.0579 | 0.0664 |
| Gaussian readout of the raw heads: AUC | 0.5059 | 0.5412 | 0.5472 |
| Gaussian readout of the raw heads: Brier | 0.2510 | 0.2514 | 0.2492 |
| Gaussian readout of the raw heads: ECE | 0.0276 | 0.0472 | 0.0238 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 105.50 | 129.43 | 146.87 |
| RMSE ($), raw heads | 105.64 | 129.65 | 147.30 |
| RMSE ($), zero prediction | 105.50 | 129.48 | 146.87 |
| MAE ($), served | 66.17 | 81.10 | 92.15 |
| MAE ($), raw heads | 66.35 | 81.23 | 92.02 |
| MAE ($), zero prediction | 66.17 | 81.17 | 92.15 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | 0.0009 | 0.0000 |
| skill vs zero, raw heads | -0.0027 | -0.0025 | -0.0059 |
| EV, served | n/a (beta = 0: served delta is 0) | 0.0011 | 0.0000 |
| EV, raw heads | -0.0027 | 0.0005 | -0.0057 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0028 | 0.0453 | 0.0317 |
| corr, Spearman, raw heads | 0.0011 | 0.0648 | 0.0728 |
| mean predicted ($), served | 0.00 | -0.97 | -0.00 |
| mean predicted ($), raw heads | 0.41 | -5.68 | -0.82 |
| mean realised ($) | 1.08 | 1.63 | 2.19 |
| share predicted up, raw heads | 0.5588 | 0.2977 | 0.4389 |
| share realised up | 0.5061 | 0.5062 | 0.5070 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1712 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 49.22 | 60.16 | 68.70 |
| CRPSS vs constant variance | 0.0186 | 0.0194 | 0.0167 |
| NLL | 6.0766 | 6.2869 | 6.3760 |
| PIT KS | 0.0313 | 0.0338 | 0.0231 |
| var / err^2 Spearman | 0.2778 | 0.2883 | 0.2634 |
| coverage of the 90% interval | 0.9019 | 0.9036 | 0.9103 |
| width of the 90% interval ($) | 303.67 | 376.30 | 443.48 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0196 | [-0.0438, 0.0046] | NOISE |
| h1 | 0.0221 | [-0.0021, 0.0436] | NOISE |
| h2 | 0.0025 | [-0.0202, 0.0255] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.171 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7534 | n/a (beta = 0: served delta is 0) | 0.6144 |
| abs(d h1) <= abs(d h2) | 0.6492 | 0.0001 | 0.5797 |
| full chain h0 <= h1 <= h2 | 0.4371 | n/a (beta = 0: served delta is 0) | 0.3220 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5020 | 0.5515 | 0.6656 | 0.2028 |
| expected if the two signs were independent | 0.5007 | 0.4660 | 0.4980 | 0.1336 |

- P(up) unanimity (all three horizons call the same side): 0.2807

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0400 vs 0.0353 (-0.0752): does not beat, significantly worse (boot z -2.77) | 0.0056 vs 0.0473 (-0.0418): does not beat, significantly worse (boot z -3.26) | 0.0138 vs 0.0546 (-0.0408): does not beat, significantly worse (boot z -2.35) |
| logreg_lags | direction/auc | 0.4718 vs 0.5324 (-0.0606): does not beat, significantly worse (boot z -3.17) | 0.5080 vs 0.5407 (-0.0327): does not beat, significantly worse (boot z -4.03) | 0.5088 vs 0.5445 (-0.0357): does not beat, significantly worse (boot z -3.34) |
| logreg_lags | direction/brier | 0.2768 vs 0.2498 (-0.0271): does not beat, significantly worse (DM z -8.34) | 0.2562 vs 0.2495 (-0.0067): does not beat, significantly worse (DM z -6.03) | 0.2615 vs 0.2493 (-0.0121): does not beat, significantly worse (DM z -5.94) |
| logreg_lags | direction/ece_pos | 0.1124 vs 0.0113 (-0.1012): does not beat, significantly worse (boot z -9.34) | 0.0563 vs 0.0116 (-0.0448): does not beat, significantly worse (boot z -4.72) | 0.0757 vs 0.0111 (-0.0646): does not beat, significantly worse (boot z -6.39) |
| logreg_lags | direction/acc | 0.4800 vs 0.5199 (-0.0400): does not beat, significantly worse (DM z -2.90) | 0.5047 vs 0.5255 (-0.0208): does not beat, significantly worse (DM z -3.41) | 0.5072 vs 0.5292 (-0.0219): does not beat, significantly worse (DM z -2.47) |
| logreg_lags | direction/bal_acc | 0.4800 vs 0.5173 (-0.0373): does not beat, significantly worse (boot z -2.77) | 0.5027 vs 0.5232 (-0.0205): does not beat, significantly worse (boot z -3.26) | 0.5069 vs 0.5269 (-0.0200): does not beat, significantly worse (boot z -2.33) |
| class_prior | direction/mcc | -0.0400 vs 0.0000 (-0.0400): does not beat, significantly worse (boot z -2.73) | 0.0056 vs 0.0000 (+0.0056): beats, noise (boot z +0.33) | 0.0138 vs 0.0000 (+0.0138): beats, noise (boot z +0.88) |
| class_prior | direction/auc | 0.4718 vs 0.5000 (-0.0282): does not beat, significantly worse (boot z -2.86) | 0.5080 vs 0.5000 (+0.0080): beats, noise (boot z +0.78) | 0.5088 vs 0.5000 (+0.0088): beats, noise (boot z +0.87) |
| class_prior | direction/brier | 0.2768 vs 0.2499 (-0.0270): does not beat, significantly worse (DM z -10.16) | 0.2562 vs 0.2499 (-0.0063): does not beat, significantly worse (DM z -4.09) | 0.2615 vs 0.2498 (-0.0116): does not beat, significantly worse (DM z -5.05) |
| class_prior | direction/ece_pos | 0.1124 vs 0.0074 (-0.1050): does not beat, significantly worse (boot z -9.68) | 0.0563 vs 0.0035 (-0.0528): does not beat, significantly worse (boot z -4.47) | 0.0757 vs 0.0049 (-0.0708): does not beat, significantly worse (boot z -6.36) |
| class_prior | direction/acc | 0.4800 vs 0.5140 (-0.0341): does not beat, significantly worse (DM z -2.86) | 0.5047 vs 0.5117 (-0.0070): does not beat, noise (DM z -0.56) | 0.5072 vs 0.5132 (-0.0060): does not beat, noise (DM z -0.41) |
| class_prior | direction/bal_acc | 0.4800 vs 0.5000 (-0.0200): does not beat, significantly worse (boot z -2.73) | 0.5027 vs 0.5000 (+0.0027): beats, noise (boot z +0.33) | 0.5069 vs 0.5000 (+0.0069): beats, noise (boot z +0.88) |
| zero_delta | delta/rmse | 105.50 vs 105.50 (+0.00, +0.00%): does not beat | 129.43 vs 129.48 (+0.06, +0.04%): beats, noise (DM z +0.96) | 146.87 vs 146.87 (+0.00, +0.00%): beats, noise (DM z +1.12) |
| zero_delta | delta/mae | 66.17 vs 66.17 (+0.00, +0.00%): does not beat | 81.10 vs 81.17 (+0.07, +0.08%): beats, noise (DM z +1.81) | 92.15 vs 92.15 (+0.00, +0.00%): beats (DM z +3.26) |
| mean_delta | delta/rmse | 105.50 vs 105.50 (-0.01, -0.00%): does not beat, noise (DM z -0.35) | 129.43 vs 129.47 (+0.05, +0.04%): beats, noise (DM z +0.70) | 146.87 vs 146.86 (-0.02, -0.01%): does not beat, noise (DM z -0.36) |
| mean_delta | delta/mae | 66.17 vs 66.16 (-0.01, -0.02%): does not beat, noise (DM z -0.78) | 81.10 vs 81.15 (+0.05, +0.07%): beats, noise (DM z +1.03) | 92.15 vs 92.13 (-0.02, -0.02%): does not beat, noise (DM z -0.52) |
| const_var | variance/crps | 49.22 vs 50.16 (+0.93, +1.86%): beats (DM z +8.44) | 60.16 vs 61.35 (+1.19, +1.94%): beats (DM z +7.59) | 68.70 vs 69.87 (+1.17, +1.67%): beats (DM z +4.89) |
| const_var | variance/nll | 6.0766 vs 6.4056 (+0.3289): beats (DM z +4.10) | 6.2869 vs 6.6267 (+0.3398): beats (DM z +3.15) | 6.3760 vs 6.7358 (+0.3598): beats (DM z +2.94) |
| const_var | variance/pit_ks | 0.0313 vs 0.0380 (+0.0068): beats (boot z +2.26) | 0.0338 vs 0.0386 (+0.0048): beats, noise (boot z +1.06) | 0.0231 vs 0.0388 (+0.0157): beats (boot z +3.09) |
| const_var | variance/corr_var_err2_spearman | 0.2778 vs 0.0000 (+0.2778): beats (boot z +14.46) | 0.2883 vs 0.0000 (+0.2883): beats (boot z +13.61) | 0.2634 vs 0.0000 (+0.2634): beats (boot z +12.18) |

## Backtest (costs included)

- n_trades: 1244
- total_return: -0.0597
- sharpe_net: -2.4190
- sharpe_gross: -2.4190
- sortino: -3.5996
- max_drawdown: 0.1510
- hit_rate: 0.4349
- hit_rate_gross: 0.4349
- profit_factor: 0.9420
- avg_hold_bars: 8.3746
- exposure: 0.4271
- turnover: 2485.3869
- fees_paid: 0.0000
- traded_notional: 24854628.8335
- breakeven_cost_bps: -0.4805
- gross_edge_per_trade_bps: -0.4582
- costs_paid: 0.0000
- gross_pnl: -597.1593
- net_pnl: -597.1593

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -38 (TimeSeriesSplit fold 3, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2023-12-25T13:18:00 .. 2024-01-11T11:47:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7844, long_above 0.5541, short_below 0.4367, median 0.5021. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -5.97% | -2.42 | +15.10% | 1244 |
| buy and hold | +6.12% | +2.44 | +9.52% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -12.03% .. +10.98%) | +0.05% | +0.08 | | |

The random null enters at the strategy's rate (0.0890 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 24% of its seeds on net return, 26% on net Sharpe and 24% on gross return.
