# Evaluation report - dev split - run `20260929T172649Z-e1c8b93-dirty-31c032db-ohlcv14__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.4608 | 0.5882 | 0.5395 |
| accuracy | 0.5109 | 0.5080 | 0.5110 |
| balanced accuracy | 0.5101 | 0.5098 | 0.5119 |
| precision (up) | 0.5006 | 0.4979 | 0.4989 |
| recall / sensitivity (up) | 0.4711 | 0.5982 | 0.5517 |
| specificity (down) | 0.5491 | 0.4214 | 0.4721 |
| F1 (up) | 0.4854 | 0.5435 | 0.5240 |
| MCC | 0.0202 | 0.0200 | 0.0239 |
| AUC | 0.5137 | 0.5166 | 0.5142 |
| Brier | 0.2571 | 0.2614 | 0.2546 |
| ECE (positive class) | 0.0552 | 0.0701 | 0.0466 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 6149 / 6134 / 7469 / 6904 | 8713 / 8786 / 6400 / 5852 | 8455 / 8493 / 7595 / 6869 |
| Gaussian readout: calls up | 0.6986 | n/a (beta = 0: readout is the constant 0.5) | 0.4466 |
| Gaussian readout: MCC | -0.0095 | n/a (beta = 0: readout is the constant 0.5) | 0.0148 |
| Gaussian readout: AUC | 0.4942 | n/a (beta = 0: readout is the constant 0.5) | 0.5106 |
| Gaussian readout: Brier | 0.2503 | n/a (beta = 0: readout is the constant 0.5) | 0.2499 |
| Gaussian readout: ECE | 0.0166 | n/a (beta = 0: readout is the constant 0.5) | 0.0112 |
| Gaussian readout of the raw heads: calls up | 0.6986 | 0.6016 | 0.4466 |
| Gaussian readout of the raw heads: MCC | -0.0095 | -0.0083 | 0.0148 |
| Gaussian readout of the raw heads: AUC | 0.4942 | 0.4966 | 0.5106 |
| Gaussian readout of the raw heads: Brier | 0.2594 | 0.2573 | 0.2576 |
| Gaussian readout of the raw heads: ECE | 0.0761 | 0.0699 | 0.0686 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.14 | 199.76 | 230.15 |
| RMSE ($), raw heads | 166.72 | 201.62 | 232.14 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.38 | 137.87 | 158.84 |
| MAE ($), raw heads | 114.25 | 139.70 | 161.21 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0006 | n/a (beta = 0: served delta is 0) | 0.0001 |
| skill vs zero, raw heads | -0.0198 | -0.0186 | -0.0173 |
| EV, served | -0.0004 | n/a (beta = 0: served delta is 0) | 0.0001 |
| EV, raw heads | -0.0146 | -0.0182 | -0.0174 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0080 | -0.0033 | 0.0087 |
| corr, Spearman, raw heads | -0.0050 | -0.0079 | 0.0110 |
| mean predicted ($), served | 1.22 | 0.00 | -0.20 |
| mean predicted ($), raw heads | 10.44 | 2.45 | -5.38 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.7209 | 0.5786 | 0.4208 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.1169 | 0.0000 | 0.0380 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 83.29 | 102.44 | 117.18 |
| CRPSS vs constant variance | 0.0103 | 0.0045 | 0.0143 |
| NLL | 6.5416 | 6.7501 | 6.8868 |
| PIT KS | 0.0294 | 0.0367 | 0.0249 |
| var / err^2 Spearman | 0.2185 | 0.0672 | 0.2096 |
| coverage of the 90% interval | 0.9053 | 0.9033 | 0.8969 |
| width of the 90% interval ($) | 512.73 | 621.21 | 695.92 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0088 | [-0.0107, 0.0276] | NOISE |
| h1 | 0.0195 | [-0.0024, 0.0398] | NOISE |
| h2 | 0.0069 | [-0.0098, 0.0230] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.117 / h1 0.000 / h2 0.038) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5613 | n/a (beta = 0: served delta is 0) | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.5772 | n/a (beta = 0: served delta is 0) | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.2322 | n/a (beta = 0: served delta is 0) | 0.3339 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4976 | 0.6225 | 0.5136 | 0.1560 |
| expected if the two signs were independent | 0.4875 | 0.5128 | 0.4911 | 0.1250 |

- P(up) unanimity (all three horizons call the same side): 0.2999

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0202 vs 0.0011 (+0.0191): beats, noise (boot z +0.85) | 0.0200 vs 0.0027 (+0.0173): beats, noise (boot z +1.12) | 0.0239 vs -0.0018 (+0.0257): beats, noise (boot z +1.50) |
| logreg_lags | direction/auc | 0.5137 vs 0.5085 (+0.0052): beats, noise (boot z +0.35) | 0.5166 vs 0.5137 (+0.0029): beats, noise (boot z +0.37) | 0.5142 vs 0.5174 (-0.0032): does not beat, noise (boot z -0.30) |
| logreg_lags | direction/brier | 0.2571 vs 0.2513 (-0.0058): does not beat, significantly worse (DM z -3.34) | 0.2614 vs 0.2517 (-0.0097): does not beat, significantly worse (DM z -5.46) | 0.2546 vs 0.2518 (-0.0027): does not beat, significantly worse (DM z -2.59) |
| logreg_lags | direction/ece_pos | 0.0552 vs 0.0347 (-0.0206): does not beat, significantly worse (boot z -1.97) | 0.0701 vs 0.0387 (-0.0314): does not beat, significantly worse (boot z -3.46) | 0.0466 vs 0.0432 (-0.0034): does not beat, noise (boot z -0.37) |
| logreg_lags | direction/acc | 0.5109 vs 0.4921 (+0.0188): beats, noise (DM z +1.75) | 0.5080 vs 0.4923 (+0.0157): beats, noise (DM z +1.89) | 0.5110 vs 0.4895 (+0.0215): beats (DM z +2.24) |
| logreg_lags | direction/bal_acc | 0.5101 vs 0.5003 (+0.0097): beats, noise (boot z +1.10) | 0.5098 vs 0.5008 (+0.0090): beats, noise (boot z +1.54) | 0.5119 vs 0.4995 (+0.0124): beats (boot z +2.11) |
| class_prior | direction/mcc | 0.0202 vs 0.0000 (+0.0202): beats, noise (boot z +1.60) | 0.0200 vs 0.0000 (+0.0200): beats, noise (boot z +1.55) | 0.0239 vs 0.0000 (+0.0239): beats (boot z +2.37) |
| class_prior | direction/auc | 0.5137 vs 0.5000 (+0.0137): beats, noise (boot z +1.65) | 0.5166 vs 0.5000 (+0.0166): beats, noise (boot z +1.86) | 0.5142 vs 0.5000 (+0.0142): beats (boot z +2.16) |
| class_prior | direction/brier | 0.2571 vs 0.2508 (-0.0063): does not beat, significantly worse (DM z -4.16) | 0.2614 vs 0.2510 (-0.0104): does not beat, significantly worse (DM z -4.95) | 0.2546 vs 0.2512 (-0.0033): does not beat, significantly worse (DM z -3.06) |
| class_prior | direction/ece_pos | 0.0552 vs 0.0297 (-0.0256): does not beat, significantly worse (boot z -2.42) | 0.0701 vs 0.0335 (-0.0366): does not beat, significantly worse (boot z -3.62) | 0.0466 vs 0.0371 (-0.0095): does not beat, noise (boot z -0.97) |
| class_prior | direction/acc | 0.5109 vs 0.4897 (+0.0212): beats (DM z +2.03) | 0.5080 vs 0.4896 (+0.0184): beats, noise (DM z +1.91) | 0.5110 vs 0.4878 (+0.0231): beats (DM z +2.13) |
| class_prior | direction/bal_acc | 0.5101 vs 0.5000 (+0.0101): beats, noise (boot z +1.60) | 0.5098 vs 0.5000 (+0.0098): beats, noise (boot z +1.55) | 0.5119 vs 0.5000 (+0.0119): beats (boot z +2.37) |
| zero_delta | delta/rmse | 165.14 vs 165.09 (-0.05, -0.03%): does not beat, noise (DM z -1.61) | 199.76 vs 199.76 (+0.00, +0.00%): does not beat | 230.15 vs 230.16 (+0.01, +0.00%): beats, noise (DM z +0.60) |
| zero_delta | delta/mae | 112.38 vs 112.32 (-0.06, -0.05%): does not beat, significantly worse (DM z -2.37) | 137.87 vs 137.87 (+0.00, +0.00%): does not beat | 158.84 vs 158.85 (+0.01, +0.01%): beats, noise (DM z +0.77) |
| mean_delta | delta/rmse | 165.14 vs 165.39 (+0.25, +0.15%): beats (DM z +2.38) | 199.76 vs 200.31 (+0.55, +0.27%): beats (DM z +2.57) | 230.15 vs 231.00 (+0.85, +0.37%): beats (DM z +2.60) |
| mean_delta | delta/mae | 112.38 vs 112.73 (+0.35, +0.31%): beats (DM z +3.92) | 137.87 vs 138.56 (+0.69, +0.50%): beats (DM z +3.78) | 158.84 vs 159.88 (+1.04, +0.65%): beats (DM z +3.68) |
| const_var | variance/crps | 83.29 vs 84.15 (+0.87, +1.03%): beats (DM z +11.55) | 102.44 vs 102.91 (+0.46, +0.45%): beats (DM z +3.41) | 117.18 vs 118.88 (+1.70, +1.43%): beats (DM z +7.39) |
| const_var | variance/nll | 6.5416 vs 6.5630 (+0.0213): beats (DM z +2.84) | 6.7501 vs 6.7470 (-0.0031): does not beat, noise (DM z -0.42) | 6.8868 vs 6.8845 (-0.0023): does not beat, noise (DM z -0.18) |
| const_var | variance/pit_ks | 0.0294 vs 0.0612 (+0.0318): beats (boot z +25.27) | 0.0367 vs 0.0652 (+0.0286): beats (boot z +10.08) | 0.0249 vs 0.0725 (+0.0477): beats (boot z +6.11) |
| const_var | variance/corr_var_err2_spearman | 0.2185 vs 0.0000 (+0.2185): beats (boot z +16.89) | 0.0672 vs 0.0000 (+0.0672): beats (boot z +4.46) | 0.2096 vs 0.0000 (+0.2096): beats (boot z +13.59) |

## Backtest (costs included)

- n_trades: 1522
- total_return: -0.9798
- sharpe_net: -137.2932
- sharpe_gross: 1.1984
- sortino: -151.9434
- max_drawdown: 0.9798
- hit_rate: 0.0631
- hit_rate_gross: 0.5164
- profit_factor: 0.0340
- avg_hold_bars: 9.9915
- exposure: 0.3520
- turnover: 760.0329
- fees_paid: 7600.5307
- traded_notional: 7600530.6941
- breakeven_cost_bps: 0.2177
- gross_edge_per_trade_bps: 0.4115
- costs_paid: 9880.6899
- gross_pnl: 82.7388
- net_pnl: -9797.9511

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8184, long_above 0.5677, short_below 0.4449, median 0.5035. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.98% | -137.29 | +97.98% | 1522 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.36% .. -97.80%) | -98.09% | -153.58 | | |

The random null enters at the strategy's rate (0.0544 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 74% of its seeds on net return, 100% on net Sharpe and 69% on gross return.
