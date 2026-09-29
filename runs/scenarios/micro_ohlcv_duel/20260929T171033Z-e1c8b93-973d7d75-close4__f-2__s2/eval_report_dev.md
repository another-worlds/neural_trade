# Evaluation report - dev split - run `20260929T171033Z-e1c8b93-973d7d75-close4__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.3956 | 0.4926 | 0.5328 |
| accuracy | 0.5121 | 0.5000 | 0.4995 |
| balanced accuracy | 0.5100 | 0.4999 | 0.5003 |
| precision (up) | 0.5023 | 0.4894 | 0.4881 |
| recall / sensitivity (up) | 0.4057 | 0.4925 | 0.5331 |
| specificity (down) | 0.6142 | 0.5072 | 0.4675 |
| F1 (up) | 0.4489 | 0.4909 | 0.5096 |
| MCC | 0.0204 | -0.0003 | 0.0006 |
| AUC | 0.5062 | 0.4980 | 0.5043 |
| Brier | 0.2571 | 0.2610 | 0.2556 |
| ECE (positive class) | 0.0524 | 0.0685 | 0.0569 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 5296 / 5248 / 8355 / 7757 | 7173 / 7483 / 7703 / 7392 | 8169 / 8567 / 7521 / 7155 |
| Gaussian readout: calls up | 0.6185 | 0.6328 | 0.5915 |
| Gaussian readout: MCC | 0.0090 | -0.0046 | 0.0090 |
| Gaussian readout: AUC | 0.5096 | 0.5003 | 0.5023 |
| Gaussian readout: Brier | 0.2500 | 0.2500 | 0.2500 |
| Gaussian readout: ECE | 0.0118 | 0.0105 | 0.0121 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.07 | 199.76 | 230.16 |
| RMSE ($), raw heads | 166.18 | 205.67 | 237.25 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.31 | 137.87 | 158.85 |
| MAE ($), raw heads | 113.79 | 143.76 | 165.64 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0003 | 0.0000 | 0.0000 |
| skill vs zero, raw heads | -0.0132 | -0.0600 | -0.0626 |
| EV, served | 0.0003 | 0.0000 | 0.0000 |
| EV, raw heads | -0.0113 | -0.0571 | -0.0595 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0245 | 0.0042 | 0.0097 |
| corr, Spearman, raw heads | 0.0141 | -0.0058 | 0.0005 |
| mean predicted ($), served | 0.33 | 0.02 | 0.06 |
| mean predicted ($), raw heads | 5.91 | 8.69 | 10.03 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.6272 | 0.6034 | 0.5719 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.0552 | 0.0024 | 0.0055 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 82.91 | 101.56 | 116.89 |
| CRPSS vs constant variance | 0.0147 | 0.0131 | 0.0167 |
| NLL | 6.5199 | 6.7280 | 6.8284 |
| PIT KS | 0.0277 | 0.0222 | 0.0312 |
| var / err^2 Spearman | 0.2383 | 0.2116 | 0.2354 |
| coverage of the 90% interval | 0.9062 | 0.9033 | 0.8971 |
| width of the 90% interval ($) | 513.73 | 621.13 | 696.40 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0041 | [-0.0222, 0.0148] | NOISE |
| h1 | 0.0019 | [-0.0152, 0.0189] | NOISE |
| h2 | 0.0058 | [-0.0091, 0.0233] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.055 / h1 0.002 / h2 0.005) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7579 | 0.0444 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.6331 | 0.9232 | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.4553 | 0.0412 | 0.3339 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5558 | 0.6503 | 0.6038 | 0.2215 |
| expected if the two signs were independent | 0.4716 | 0.4956 | 0.5054 | 0.1204 |

- P(up) unanimity (all three horizons call the same side): 0.2915

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0204 vs 0.0011 (+0.0193): beats, noise (boot z +0.86) | -0.0003 vs 0.0027 (-0.0030): does not beat, noise (boot z -0.16) | 0.0006 vs -0.0018 (+0.0024): beats, noise (boot z +0.12) |
| logreg_lags | direction/auc | 0.5062 vs 0.5085 (-0.0024): does not beat, noise (boot z -0.17) | 0.4980 vs 0.5137 (-0.0157): does not beat, noise (boot z -1.34) | 0.5043 vs 0.5174 (-0.0132): does not beat, noise (boot z -1.07) |
| logreg_lags | direction/brier | 0.2571 vs 0.2513 (-0.0058): does not beat, significantly worse (DM z -3.84) | 0.2610 vs 0.2517 (-0.0093): does not beat, significantly worse (DM z -5.75) | 0.2556 vs 0.2518 (-0.0038): does not beat, significantly worse (DM z -3.56) |
| logreg_lags | direction/ece_pos | 0.0524 vs 0.0347 (-0.0177): does not beat, noise (boot z -1.59) | 0.0685 vs 0.0387 (-0.0298): does not beat, significantly worse (boot z -2.79) | 0.0569 vs 0.0432 (-0.0136): does not beat, noise (boot z -1.27) |
| logreg_lags | direction/acc | 0.5121 vs 0.4921 (+0.0200): beats, noise (DM z +1.79) | 0.5000 vs 0.4923 (+0.0077): beats, noise (DM z +0.76) | 0.4995 vs 0.4895 (+0.0100): beats, noise (DM z +0.96) |
| logreg_lags | direction/bal_acc | 0.5100 vs 0.5003 (+0.0096): beats, noise (boot z +1.10) | 0.4999 vs 0.5008 (-0.0009): does not beat, noise (boot z -0.14) | 0.5003 vs 0.4995 (+0.0008): beats, noise (boot z +0.12) |
| class_prior | direction/mcc | 0.0204 vs 0.0000 (+0.0204): beats, noise (boot z +1.56) | -0.0003 vs 0.0000 (-0.0003): does not beat, noise (boot z -0.02) | 0.0006 vs 0.0000 (+0.0006): beats, noise (boot z +0.05) |
| class_prior | direction/auc | 0.5062 vs 0.5000 (+0.0062): beats, noise (boot z +0.74) | 0.4980 vs 0.5000 (-0.0020): does not beat, noise (boot z -0.25) | 0.5043 vs 0.5000 (+0.0043): beats, noise (boot z +0.64) |
| class_prior | direction/brier | 0.2571 vs 0.2508 (-0.0063): does not beat, significantly worse (DM z -4.50) | 0.2610 vs 0.2510 (-0.0099): does not beat, significantly worse (DM z -5.63) | 0.2556 vs 0.2512 (-0.0044): does not beat, significantly worse (DM z -4.15) |
| class_prior | direction/ece_pos | 0.0524 vs 0.0297 (-0.0227): does not beat, significantly worse (boot z -2.01) | 0.0685 vs 0.0335 (-0.0350): does not beat, significantly worse (boot z -3.07) | 0.0569 vs 0.0371 (-0.0198): does not beat, noise (boot z -1.78) |
| class_prior | direction/acc | 0.5121 vs 0.4897 (+0.0224): beats (DM z +1.97) | 0.5000 vs 0.4896 (+0.0105): beats, noise (DM z +0.96) | 0.4995 vs 0.4878 (+0.0117): beats, noise (DM z +1.04) |
| class_prior | direction/bal_acc | 0.5100 vs 0.5000 (+0.0100): beats, noise (boot z +1.56) | 0.4999 vs 0.5000 (-0.0001): does not beat, noise (boot z -0.02) | 0.5003 vs 0.5000 (+0.0003): beats, noise (boot z +0.05) |
| zero_delta | delta/rmse | 165.07 vs 165.09 (+0.02, +0.01%): beats, noise (DM z +1.27) | 199.76 vs 199.76 (+0.00, +0.00%): beats, noise (DM z +0.10) | 230.16 vs 230.16 (+0.00, +0.00%): beats, noise (DM z +0.32) |
| zero_delta | delta/mae | 112.31 vs 112.32 (+0.00, +0.00%): beats, noise (DM z +0.25) | 137.87 vs 137.87 (+0.00, +0.00%): beats, noise (DM z +0.04) | 158.85 vs 158.85 (+0.00, +0.00%): beats, noise (DM z +0.10) |
| mean_delta | delta/rmse | 165.07 vs 165.39 (+0.32, +0.19%): beats (DM z +2.87) | 199.76 vs 200.31 (+0.55, +0.27%): beats (DM z +2.58) | 230.16 vs 231.00 (+0.84, +0.36%): beats (DM z +2.58) |
| mean_delta | delta/mae | 112.31 vs 112.73 (+0.41, +0.37%): beats (DM z +4.22) | 137.87 vs 138.56 (+0.69, +0.50%): beats (DM z +3.78) | 158.85 vs 159.88 (+1.03, +0.64%): beats (DM z +3.70) |
| const_var | variance/crps | 82.91 vs 84.15 (+1.24, +1.47%): beats (DM z +9.65) | 101.56 vs 102.91 (+1.35, +1.31%): beats (DM z +7.60) | 116.89 vs 118.88 (+1.99, +1.67%): beats (DM z +6.98) |
| const_var | variance/nll | 6.5199 vs 6.5630 (+0.0430): beats, noise (DM z +1.76) | 6.7280 vs 6.7470 (+0.0190): beats, noise (DM z +0.92) | 6.8284 vs 6.8845 (+0.0561): beats (DM z +2.74) |
| const_var | variance/pit_ks | 0.0277 vs 0.0612 (+0.0335): beats (boot z +12.01) | 0.0222 vs 0.0652 (+0.0431): beats (boot z +10.79) | 0.0312 vs 0.0725 (+0.0413): beats (boot z +10.49) |
| const_var | variance/corr_var_err2_spearman | 0.2383 vs 0.0000 (+0.2383): beats (boot z +16.67) | 0.2116 vs 0.0000 (+0.2116): beats (boot z +13.22) | 0.2354 vs 0.0000 (+0.2354): beats (boot z +14.18) |

## Backtest (costs included)

- n_trades: 1761
- total_return: -0.9896
- sharpe_net: -156.1982
- sharpe_gross: 2.5585
- sortino: -170.3551
- max_drawdown: 0.9897
- hit_rate: 0.0398
- hit_rate_gross: 0.4855
- profit_factor: 0.0290
- avg_hold_bars: 8.0068
- exposure: 0.3264
- turnover: 773.6745
- fees_paid: 7737.1174
- traded_notional: 7737117.3863
- breakeven_cost_bps: 0.4182
- gross_edge_per_trade_bps: 0.0940
- costs_paid: 10058.2526
- gross_pnl: 161.7988
- net_pnl: -9896.4538

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8229, long_above 0.5600, short_below 0.4326, median 0.4912. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.96% | -156.20 | +98.97% | 1761 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.12% .. -98.83%) | -98.97% | -172.86 | | |

The random null enters at the strategy's rate (0.0605 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 51% of its seeds on net return, 100% on net Sharpe and 82% on gross return.
