# Evaluation report - dev split - run `20260930T230540Z-fb840fd-9bd6073c-control__f-38__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 18047 | 19287 | 19803 |
| n_eff of the scored moves (n scored // bars ahead) | 1804 | 1285 | 990 |
| true up-rate | 0.5140 | 0.5117 | 0.5132 |
| calls up (predicted up-rate) | 0.7881 | 0.5773 | 0.6839 |
| accuracy | 0.5167 | 0.5106 | 0.5142 |
| balanced accuracy | 0.5086 | 0.5088 | 0.5094 |
| precision (up) | 0.5195 | 0.5193 | 0.5200 |
| recall / sensitivity (up) | 0.7965 | 0.5859 | 0.6930 |
| specificity (down) | 0.2208 | 0.4317 | 0.3257 |
| F1 (up) | 0.6289 | 0.5506 | 0.5942 |
| MCC | 0.0211 | 0.0178 | 0.0201 |
| AUC | 0.5211 | 0.5115 | 0.5126 |
| Brier | 0.2620 | 0.2549 | 0.2588 |
| ECE (positive class) | 0.0769 | 0.0441 | 0.0623 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0140 | 0.0117 | 0.0132 |
| TP / FP / TN / FN | 7389 / 6834 / 1936 / 1888 | 5782 / 5352 / 4066 / 4087 | 7043 / 6500 / 3140 / 3120 |
| Gaussian readout: calls up | 0.7258 | 0.6696 | 0.6021 |
| Gaussian readout: MCC | 0.0317 | 0.0422 | 0.0326 |
| Gaussian readout: AUC | 0.5210 | 0.5311 | 0.5287 |
| Gaussian readout: Brier | 0.2496 | 0.2494 | 0.2495 |
| Gaussian readout: ECE | 0.0122 | 0.0148 | 0.0107 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 105.47 | 129.39 | 146.77 |
| RMSE ($), raw heads | 106.66 | 134.33 | 151.62 |
| RMSE ($), zero prediction | 105.50 | 129.48 | 146.87 |
| MAE ($), served | 66.12 | 81.05 | 92.02 |
| MAE ($), raw heads | 67.02 | 84.93 | 96.08 |
| MAE ($), zero prediction | 66.17 | 81.17 | 92.15 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0006 | 0.0015 | 0.0014 |
| skill vs zero, raw heads | -0.0222 | -0.0763 | -0.0657 |
| EV, served | 0.0005 | 0.0014 | 0.0012 |
| EV, raw heads | -0.0172 | -0.0645 | -0.0602 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0278 | 0.0472 | 0.0420 |
| corr, Spearman, raw heads | 0.0484 | 0.0594 | 0.0527 |
| mean predicted ($), served | 0.63 | 0.92 | 0.87 |
| mean predicted ($), raw heads | 8.60 | 15.76 | 13.34 |
| mean realised ($) | 1.08 | 1.63 | 2.19 |
| share predicted up, raw heads | 0.7247 | 0.6706 | 0.5979 |
| share realised up | 0.5061 | 0.5062 | 0.5070 |
| shrink beta (served = beta x raw, fit on cal) | 0.0738 | 0.0585 | 0.0651 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 49.00 | 60.08 | 68.47 |
| CRPSS vs constant variance | 0.0230 | 0.0208 | 0.0199 |
| NLL | 6.0876 | 6.2957 | 6.4227 |
| PIT KS | 0.0391 | 0.0345 | 0.0304 |
| var / err^2 Spearman | 0.2996 | 0.2883 | 0.2659 |
| coverage of the 90% interval | 0.9023 | 0.9035 | 0.9103 |
| width of the 90% interval ($) | 303.52 | 375.64 | 443.62 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0289 | [0.0079, 0.0546] | WORKS |
| h1 | -0.0012 | [-0.0224, 0.0217] | NOISE |
| h2 | 0.0072 | [-0.0152, 0.0288] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.074 / h1 0.058 / h2 0.065) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8106 | 0.7185 | 0.6144 |
| abs(d h1) <= abs(d h2) | 0.6265 | 0.7153 | 0.5797 |
| full chain h0 <= h1 <= h2 | 0.4794 | 0.4945 | 0.3220 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7138 | 0.6533 | 0.6837 | 0.3841 |
| expected if the two signs were independent | 0.6296 | 0.5264 | 0.5361 | 0.2620 |

- P(up) unanimity (all three horizons call the same side): 0.4703

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0211 vs 0.0353 (-0.0142): does not beat, noise (boot z -0.61) | 0.0178 vs 0.0473 (-0.0295): does not beat, noise (boot z -1.84) | 0.0201 vs 0.0546 (-0.0345): does not beat, noise (boot z -1.80) |
| logreg_lags | direction/auc | 0.5211 vs 0.5324 (-0.0113): does not beat, noise (boot z -0.81) | 0.5115 vs 0.5407 (-0.0292): does not beat, significantly worse (boot z -2.93) | 0.5126 vs 0.5445 (-0.0319): does not beat, significantly worse (boot z -2.86) |
| logreg_lags | direction/brier | 0.2620 vs 0.2498 (-0.0122): does not beat, significantly worse (DM z -4.86) | 0.2549 vs 0.2495 (-0.0054): does not beat, significantly worse (DM z -3.95) | 0.2588 vs 0.2493 (-0.0094): does not beat, significantly worse (DM z -4.88) |
| logreg_lags | direction/ece_pos | 0.0769 vs 0.0113 (-0.0656): does not beat, significantly worse (boot z -5.09) | 0.0441 vs 0.0116 (-0.0325): does not beat, significantly worse (boot z -3.66) | 0.0623 vs 0.0111 (-0.0512): does not beat, significantly worse (boot z -4.08) |
| logreg_lags | direction/acc | 0.5167 vs 0.5199 (-0.0032): does not beat, noise (DM z -0.29) | 0.5106 vs 0.5255 (-0.0149): does not beat, noise (DM z -1.93) | 0.5142 vs 0.5292 (-0.0149): does not beat, noise (DM z -1.52) |
| logreg_lags | direction/bal_acc | 0.5086 vs 0.5173 (-0.0087): does not beat, noise (boot z -0.81) | 0.5088 vs 0.5232 (-0.0144): does not beat, noise (boot z -1.83) | 0.5094 vs 0.5269 (-0.0175): does not beat, noise (boot z -1.89) |
| class_prior | direction/mcc | 0.0211 vs 0.0000 (+0.0211): beats, noise (boot z +1.36) | 0.0178 vs 0.0000 (+0.0178): beats, noise (boot z +1.24) | 0.0201 vs 0.0000 (+0.0201): beats, noise (boot z +1.30) |
| class_prior | direction/auc | 0.5211 vs 0.5000 (+0.0211): beats (boot z +2.15) | 0.5115 vs 0.5000 (+0.0115): beats, noise (boot z +1.21) | 0.5126 vs 0.5000 (+0.0126): beats, noise (boot z +1.19) |
| class_prior | direction/brier | 0.2620 vs 0.2499 (-0.0121): does not beat, significantly worse (DM z -5.34) | 0.2549 vs 0.2499 (-0.0050): does not beat, significantly worse (DM z -3.58) | 0.2588 vs 0.2498 (-0.0089): does not beat, significantly worse (DM z -4.32) |
| class_prior | direction/ece_pos | 0.0769 vs 0.0074 (-0.0695): does not beat, significantly worse (boot z -5.01) | 0.0441 vs 0.0035 (-0.0405): does not beat, significantly worse (boot z -3.88) | 0.0623 vs 0.0049 (-0.0574): does not beat, significantly worse (boot z -4.19) |
| class_prior | direction/acc | 0.5167 vs 0.5140 (+0.0027): beats, noise (DM z +0.37) | 0.5106 vs 0.5117 (-0.0011): does not beat, noise (DM z -0.09) | 0.5142 vs 0.5132 (+0.0010): beats, noise (DM z +0.10) |
| class_prior | direction/bal_acc | 0.5086 vs 0.5000 (+0.0086): beats, noise (boot z +1.35) | 0.5088 vs 0.5000 (+0.0088): beats, noise (boot z +1.24) | 0.5094 vs 0.5000 (+0.0094): beats, noise (boot z +1.30) |
| zero_delta | delta/rmse | 105.47 vs 105.50 (+0.03, +0.03%): beats, noise (DM z +1.04) | 129.39 vs 129.48 (+0.10, +0.07%): beats, noise (DM z +1.67) | 146.77 vs 146.87 (+0.10, +0.07%): beats, noise (DM z +1.38) |
| zero_delta | delta/mae | 66.12 vs 66.17 (+0.05, +0.07%): beats (DM z +2.36) | 81.05 vs 81.17 (+0.12, +0.14%): beats (DM z +2.87) | 92.02 vs 92.15 (+0.12, +0.13%): beats (DM z +2.37) |
| mean_delta | delta/rmse | 105.47 vs 105.50 (+0.03, +0.03%): beats, noise (DM z +0.92) | 129.39 vs 129.47 (+0.09, +0.07%): beats, noise (DM z +1.54) | 146.77 vs 146.86 (+0.09, +0.06%): beats, noise (DM z +1.11) |
| mean_delta | delta/mae | 66.12 vs 66.16 (+0.04, +0.06%): beats (DM z +2.06) | 81.05 vs 81.15 (+0.10, +0.13%): beats (DM z +2.71) | 92.02 vs 92.13 (+0.11, +0.11%): beats (DM z +2.00) |
| const_var | variance/crps | 49.00 vs 50.16 (+1.15, +2.30%): beats (DM z +8.00) | 60.08 vs 61.35 (+1.27, +2.08%): beats (DM z +5.73) | 68.47 vs 69.87 (+1.39, +1.99%): beats (DM z +5.61) |
| const_var | variance/nll | 6.0876 vs 6.4056 (+0.3179): beats (DM z +3.16) | 6.2957 vs 6.6267 (+0.3310): beats (DM z +2.42) | 6.4227 vs 6.7358 (+0.3131): beats (DM z +2.36) |
| const_var | variance/pit_ks | 0.0391 vs 0.0380 (-0.0010): does not beat, noise (boot z -0.24) | 0.0345 vs 0.0386 (+0.0041): beats, noise (boot z +0.93) | 0.0304 vs 0.0388 (+0.0084): beats (boot z +2.16) |
| const_var | variance/corr_var_err2_spearman | 0.2996 vs 0.0000 (+0.2996): beats (boot z +15.46) | 0.2883 vs 0.0000 (+0.2883): beats (boot z +13.99) | 0.2659 vs 0.0000 (+0.2659): beats (boot z +11.88) |

## Backtest (costs included)

- n_trades: 1114
- total_return: 0.1216
- sharpe_net: 5.6371
- sharpe_gross: 5.6371
- sortino: 8.1503
- max_drawdown: 0.0722
- hit_rate: 0.5251
- hit_rate_gross: 0.5251
- profit_factor: 1.1202
- avg_hold_bars: 8.9524
- exposure: 0.4089
- turnover: 2388.1012
- fees_paid: 0.0000
- traded_notional: 23881901.4275
- breakeven_cost_bps: 1.0182
- gross_edge_per_trade_bps: 1.0723
- costs_paid: 0.0000
- gross_pnl: 1215.8825
- net_pnl: 1215.8825

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -38 (TimeSeriesSplit fold 3, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2023-12-25T13:18:00 .. 2024-01-11T11:47:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7206, long_above 0.6174, short_below 0.4755, median 0.5265. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +12.16% | +5.64 | +7.22% | 1114 |
| buy and hold | +6.12% | +2.44 | +9.52% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -12.26% .. +13.04%) | +0.18% | +0.13 | | |

The random null enters at the strategy's rate (0.0773 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 93% of its seeds on net return, 89% on net Sharpe and 93% on gross return.
