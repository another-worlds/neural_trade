# Evaluation report - dev split - run `20260929T124027Z-9446405-4feea012-h_15m__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.4196 | 0.6985 | 0.5443 |
| accuracy | 0.4975 | 0.4973 | 0.5023 |
| balanced accuracy | 0.4958 | 0.5014 | 0.5034 |
| precision (up) | 0.4847 | 0.4906 | 0.4910 |
| recall / sensitivity (up) | 0.4154 | 0.7000 | 0.5478 |
| specificity (down) | 0.5763 | 0.3028 | 0.4590 |
| F1 (up) | 0.4474 | 0.5769 | 0.5178 |
| MCC | -0.0085 | 0.0031 | 0.0068 |
| AUC | 0.4963 | 0.5044 | 0.5122 |
| Brier | 0.2582 | 0.2537 | 0.2549 |
| ECE (positive class) | 0.0638 | 0.0435 | 0.0547 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 5422 / 5764 / 7839 / 7631 | 10195 / 10587 / 4599 / 4370 | 8394 / 8703 / 7385 / 6930 |
| Gaussian readout: calls up | 0.7130 | 0.6814 | 0.6498 |
| Gaussian readout: MCC | -0.0207 | 0.0069 | -0.0156 |
| Gaussian readout: AUC | 0.5041 | 0.5117 | 0.5033 |
| Gaussian readout: Brier | 0.2505 | 0.2500 | 0.2502 |
| Gaussian readout: ECE | 0.0317 | 0.0186 | 0.0226 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.17 | 199.83 | 230.29 |
| RMSE ($), raw heads | 167.06 | 205.24 | 236.04 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.41 | 137.87 | 158.93 |
| MAE ($), raw heads | 114.11 | 141.34 | 162.98 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0010 | -0.0006 | -0.0011 |
| skill vs zero, raw heads | -0.0240 | -0.0555 | -0.0517 |
| EV, served | -0.0003 | -0.0000 | -0.0006 |
| EV, raw heads | -0.0168 | -0.0434 | -0.0432 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0134 | 0.0169 | 0.0102 |
| corr, Spearman, raw heads | -0.0007 | 0.0079 | -0.0005 |
| mean predicted ($), served | 3.00 | 3.02 | 3.04 |
| mean predicted ($), raw heads | 12.44 | 19.75 | 18.34 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.6940 | 0.6664 | 0.6354 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.2413 | 0.1530 | 0.1660 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 83.35 | 101.64 | 116.88 |
| CRPSS vs constant variance | 0.0096 | 0.0123 | 0.0168 |
| NLL | 6.5259 | 6.7253 | 6.8293 |
| PIT KS | 0.0389 | 0.0306 | 0.0393 |
| var / err^2 Spearman | 0.2211 | 0.1964 | 0.2321 |
| coverage of the 90% interval | 0.9050 | 0.9044 | 0.8978 |
| width of the 90% interval ($) | 512.59 | 623.60 | 698.14 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0070 | [-0.0093, 0.0264] | NOISE |
| h1 | -0.0016 | [-0.0194, 0.0155] | NOISE |
| h2 | 0.0186 | [0.0018, 0.0367] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.241 / h1 0.153 / h2 0.166) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6047 | 0.4067 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.5558 | 0.5898 | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.2667 | 0.1907 | 0.3339 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4841 | 0.6502 | 0.6465 | 0.2155 |
| expected if the two signs were independent | 0.4686 | 0.5666 | 0.5084 | 0.1520 |

- P(up) unanimity (all three horizons call the same side): 0.3181

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0085 vs 0.0011 (-0.0096): does not beat, noise (boot z -0.46) | 0.0031 vs 0.0027 (+0.0004): beats, noise (boot z +0.02) | 0.0068 vs -0.0018 (+0.0087): beats, noise (boot z +0.41) |
| logreg_lags | direction/auc | 0.4963 vs 0.5085 (-0.0122): does not beat, noise (boot z -0.91) | 0.5044 vs 0.5137 (-0.0094): does not beat, noise (boot z -0.80) | 0.5122 vs 0.5174 (-0.0053): does not beat, noise (boot z -0.39) |
| logreg_lags | direction/brier | 0.2582 vs 0.2513 (-0.0069): does not beat, significantly worse (DM z -4.79) | 0.2537 vs 0.2517 (-0.0021): does not beat, significantly worse (DM z -2.71) | 0.2549 vs 0.2518 (-0.0031): does not beat, significantly worse (DM z -2.26) |
| logreg_lags | direction/ece_pos | 0.0638 vs 0.0347 (-0.0291): does not beat, significantly worse (boot z -2.77) | 0.0435 vs 0.0387 (-0.0048): does not beat, noise (boot z -0.71) | 0.0547 vs 0.0432 (-0.0115): does not beat, noise (boot z -1.06) |
| logreg_lags | direction/acc | 0.4975 vs 0.4921 (+0.0054): beats, noise (DM z +0.52) | 0.4973 vs 0.4923 (+0.0050): beats, noise (DM z +0.66) | 0.5023 vs 0.4895 (+0.0129): beats, noise (DM z +1.18) |
| logreg_lags | direction/bal_acc | 0.4958 vs 0.5003 (-0.0045): does not beat, noise (boot z -0.56) | 0.5014 vs 0.5008 (+0.0006): beats, noise (boot z +0.09) | 0.5034 vs 0.4995 (+0.0039): beats, noise (boot z +0.50) |
| class_prior | direction/mcc | -0.0085 vs 0.0000 (-0.0085): does not beat, noise (boot z -0.69) | 0.0031 vs 0.0000 (+0.0031): beats, noise (boot z +0.30) | 0.0068 vs 0.0000 (+0.0068): beats, noise (boot z +0.55) |
| class_prior | direction/auc | 0.4963 vs 0.5000 (-0.0037): does not beat, noise (boot z -0.47) | 0.5044 vs 0.5000 (+0.0044): beats, noise (boot z +0.64) | 0.5122 vs 0.5000 (+0.0122): beats, noise (boot z +1.49) |
| class_prior | direction/brier | 0.2582 vs 0.2508 (-0.0074): does not beat, significantly worse (DM z -5.64) | 0.2537 vs 0.2510 (-0.0027): does not beat, significantly worse (DM z -3.52) | 0.2549 vs 0.2512 (-0.0037): does not beat, significantly worse (DM z -2.79) |
| class_prior | direction/ece_pos | 0.0638 vs 0.0297 (-0.0341): does not beat, significantly worse (boot z -3.15) | 0.0435 vs 0.0335 (-0.0101): does not beat, noise (boot z -1.44) | 0.0547 vs 0.0371 (-0.0176): does not beat, noise (boot z -1.59) |
| class_prior | direction/acc | 0.4975 vs 0.4897 (+0.0078): beats, noise (DM z +0.73) | 0.4973 vs 0.4896 (+0.0077): beats, noise (DM z +1.05) | 0.5023 vs 0.4878 (+0.0145): beats, noise (DM z +1.29) |
| class_prior | direction/bal_acc | 0.4958 vs 0.5000 (-0.0042): does not beat, noise (boot z -0.69) | 0.5014 vs 0.5000 (+0.0014): beats, noise (boot z +0.30) | 0.5034 vs 0.5000 (+0.0034): beats, noise (boot z +0.55) |
| zero_delta | delta/rmse | 165.17 vs 165.09 (-0.08, -0.05%): does not beat, noise (DM z -0.75) | 199.83 vs 199.76 (-0.06, -0.03%): does not beat, noise (DM z -0.39) | 230.29 vs 230.16 (-0.13, -0.06%): does not beat, noise (DM z -0.57) |
| zero_delta | delta/mae | 112.41 vs 112.32 (-0.09, -0.08%): does not beat, noise (DM z -1.26) | 137.87 vs 137.87 (-0.00, -0.00%): does not beat, noise (DM z -0.04) | 158.93 vs 158.85 (-0.08, -0.05%): does not beat, noise (DM z -0.58) |
| mean_delta | delta/rmse | 165.17 vs 165.39 (+0.22, +0.13%): beats (DM z +2.14) | 199.83 vs 200.31 (+0.49, +0.24%): beats (DM z +2.54) | 230.29 vs 231.00 (+0.71, +0.31%): beats (DM z +2.40) |
| mean_delta | delta/mae | 112.41 vs 112.73 (+0.32, +0.29%): beats (DM z +3.51) | 137.87 vs 138.56 (+0.69, +0.50%): beats (DM z +4.07) | 158.93 vs 159.88 (+0.95, +0.59%): beats (DM z +3.59) |
| const_var | variance/crps | 83.35 vs 84.15 (+0.81, +0.96%): beats (DM z +5.39) | 101.64 vs 102.91 (+1.27, +1.23%): beats (DM z +7.03) | 116.88 vs 118.88 (+2.00, +1.68%): beats (DM z +7.30) |
| const_var | variance/nll | 6.5259 vs 6.5630 (+0.0371): beats (DM z +2.21) | 6.7253 vs 6.7470 (+0.0217): beats, noise (DM z +1.35) | 6.8293 vs 6.8845 (+0.0552): beats (DM z +3.76) |
| const_var | variance/pit_ks | 0.0389 vs 0.0612 (+0.0223): beats (boot z +10.39) | 0.0306 vs 0.0652 (+0.0346): beats (boot z +15.60) | 0.0393 vs 0.0725 (+0.0332): beats (boot z +12.68) |
| const_var | variance/corr_var_err2_spearman | 0.2211 vs 0.0000 (+0.2211): beats (boot z +14.96) | 0.1964 vs 0.0000 (+0.1964): beats (boot z +11.90) | 0.2321 vs 0.0000 (+0.2321): beats (boot z +14.09) |

## Backtest (costs included)

- n_trades: 2166
- total_return: -0.9965
- sharpe_net: -181.5073
- sharpe_gross: -10.2765
- sortino: -194.5235
- max_drawdown: 0.9965
- hit_rate: 0.0346
- hit_rate_gross: 0.4875
- profit_factor: 0.0129
- avg_hold_bars: 7.4986
- exposure: 0.3760
- turnover: 721.3307
- fees_paid: 7213.5376
- traded_notional: 7213537.6283
- breakeven_cost_bps: -1.6280
- gross_edge_per_trade_bps: -0.0328
- costs_paid: 9377.5989
- gross_pnl: -587.1714
- net_pnl: -9964.7703

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.9019, long_above 0.5486, short_below 0.4571, median 0.5013. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.65% | -181.51 | +99.65% | 2166 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.74% .. -99.64%) | -99.69% | -200.21 | | |

The random null enters at the strategy's rate (0.0804 per flat bar), holds 7 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 88% of its seeds on net return, 100% on net Sharpe and 0% on gross return.
