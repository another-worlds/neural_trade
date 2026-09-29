# Evaluation report - dev split - run `20260929T193152Z-60ad158-23d6bef8-ohlcv14__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.3342 | 0.4595 | 0.3632 |
| accuracy | 0.5051 | 0.5063 | 0.5045 |
| balanced accuracy | 0.5017 | 0.5055 | 0.5011 |
| precision (up) | 0.4922 | 0.4955 | 0.4894 |
| recall / sensitivity (up) | 0.3359 | 0.4651 | 0.3643 |
| specificity (down) | 0.6674 | 0.5458 | 0.6379 |
| F1 (up) | 0.3993 | 0.4798 | 0.4177 |
| MCC | 0.0036 | 0.0110 | 0.0023 |
| AUC | 0.5063 | 0.5045 | 0.5001 |
| Brier | 0.2574 | 0.2562 | 0.2585 |
| ECE (positive class) | 0.0651 | 0.0554 | 0.0636 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 4385 / 4524 / 9079 / 8668 | 6774 / 6897 / 8289 / 7791 | 5583 / 5825 / 10263 / 9741 |
| Gaussian readout: calls up | 0.3076 | 0.3270 | 0.3794 |
| Gaussian readout: MCC | 0.0225 | 0.0138 | -0.0101 |
| Gaussian readout: AUC | 0.5119 | 0.5025 | 0.4985 |
| Gaussian readout: Brier | 0.2499 | 0.2501 | 0.2509 |
| Gaussian readout: ECE | 0.0076 | 0.0049 | 0.0289 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.08 | 199.83 | 230.50 |
| RMSE ($), raw heads | 166.27 | 202.72 | 234.55 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.28 | 137.89 | 159.11 |
| MAE ($), raw heads | 113.28 | 140.63 | 162.87 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0001 | -0.0006 | -0.0030 |
| skill vs zero, raw heads | -0.0143 | -0.0298 | -0.0385 |
| EV, served | 0.0001 | -0.0008 | -0.0031 |
| EV, raw heads | -0.0123 | -0.0261 | -0.0383 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0080 | 0.0016 | -0.0069 |
| corr, Spearman, raw heads | 0.0193 | 0.0010 | -0.0108 |
| mean predicted ($), served | -0.83 | -2.68 | -2.03 |
| mean predicted ($), raw heads | -9.19 | -14.94 | -7.72 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.3066 | 0.3234 | 0.3716 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.0905 | 0.1792 | 0.2628 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 82.83 | 101.59 | 117.05 |
| CRPSS vs constant variance | 0.0157 | 0.0128 | 0.0153 |
| NLL | 6.5101 | 6.6782 | 6.8632 |
| PIT KS | 0.0183 | 0.0277 | 0.0205 |
| var / err^2 Spearman | 0.2472 | 0.2096 | 0.2279 |
| coverage of the 90% interval | 0.9059 | 0.9037 | 0.8961 |
| width of the 90% interval ($) | 513.66 | 622.59 | 696.07 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0176 | [-0.0012, 0.0356] | NOISE |
| h1 | -0.0050 | [-0.0223, 0.0135] | NOISE |
| h2 | 0.0118 | [-0.0089, 0.0304] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.091 / h1 0.179 / h2 0.263) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7440 | 0.8989 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.6045 | 0.7684 | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.4114 | 0.6739 | 0.3339 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6885 | 0.5806 | 0.6137 | 0.2983 |
| expected if the two signs were independent | 0.5569 | 0.5110 | 0.5341 | 0.2062 |

- P(up) unanimity (all three horizons call the same side): 0.4070

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0036 vs 0.0011 (+0.0024): beats, noise (boot z +0.11) | 0.0110 vs 0.0027 (+0.0083): beats, noise (boot z +0.37) | 0.0023 vs -0.0018 (+0.0042): beats, noise (boot z +0.20) |
| logreg_lags | direction/auc | 0.5063 vs 0.5085 (-0.0022): does not beat, noise (boot z -0.16) | 0.5045 vs 0.5137 (-0.0093): does not beat, noise (boot z -0.62) | 0.5001 vs 0.5174 (-0.0173): does not beat, noise (boot z -1.34) |
| logreg_lags | direction/brier | 0.2574 vs 0.2513 (-0.0061): does not beat, significantly worse (DM z -3.65) | 0.2562 vs 0.2517 (-0.0045): does not beat, significantly worse (DM z -2.69) | 0.2585 vs 0.2518 (-0.0067): does not beat, significantly worse (DM z -3.68) |
| logreg_lags | direction/ece_pos | 0.0651 vs 0.0347 (-0.0304): does not beat, significantly worse (boot z -2.71) | 0.0554 vs 0.0387 (-0.0167): does not beat, noise (boot z -1.58) | 0.0636 vs 0.0432 (-0.0204): does not beat, noise (boot z -1.49) |
| logreg_lags | direction/acc | 0.5051 vs 0.4921 (+0.0130): beats, noise (DM z +1.11) | 0.5063 vs 0.4923 (+0.0140): beats, noise (DM z +1.24) | 0.5045 vs 0.4895 (+0.0150): beats, noise (DM z +1.09) |
| logreg_lags | direction/bal_acc | 0.5017 vs 0.5003 (+0.0013): beats, noise (boot z +0.16) | 0.5055 vs 0.5008 (+0.0047): beats, noise (boot z +0.54) | 0.5011 vs 0.4995 (+0.0017): beats, noise (boot z +0.22) |
| class_prior | direction/mcc | 0.0036 vs 0.0000 (+0.0036): beats, noise (boot z +0.28) | 0.0110 vs 0.0000 (+0.0110): beats, noise (boot z +0.85) | 0.0023 vs 0.0000 (+0.0023): beats, noise (boot z +0.20) |
| class_prior | direction/auc | 0.5063 vs 0.5000 (+0.0063): beats, noise (boot z +0.76) | 0.5045 vs 0.5000 (+0.0045): beats, noise (boot z +0.54) | 0.5001 vs 0.5000 (+0.0001): beats, noise (boot z +0.02) |
| class_prior | direction/brier | 0.2574 vs 0.2508 (-0.0067): does not beat, significantly worse (DM z -4.33) | 0.2562 vs 0.2510 (-0.0052): does not beat, significantly worse (DM z -3.62) | 0.2585 vs 0.2512 (-0.0073): does not beat, significantly worse (DM z -3.91) |
| class_prior | direction/ece_pos | 0.0651 vs 0.0297 (-0.0354): does not beat, significantly worse (boot z -3.10) | 0.0554 vs 0.0335 (-0.0219): does not beat, significantly worse (boot z -2.07) | 0.0636 vs 0.0371 (-0.0266): does not beat, noise (boot z -1.88) |
| class_prior | direction/acc | 0.5051 vs 0.4897 (+0.0154): beats, noise (DM z +1.27) | 0.5063 vs 0.4896 (+0.0167): beats, noise (DM z +1.44) | 0.5045 vs 0.4878 (+0.0166): beats, noise (DM z +1.14) |
| class_prior | direction/bal_acc | 0.5017 vs 0.5000 (+0.0017): beats, noise (boot z +0.28) | 0.5055 vs 0.5000 (+0.0055): beats, noise (boot z +0.85) | 0.5011 vs 0.5000 (+0.0011): beats, noise (boot z +0.20) |
| zero_delta | delta/rmse | 165.08 vs 165.09 (+0.01, +0.01%): beats, noise (DM z +0.40) | 199.83 vs 199.76 (-0.06, -0.03%): does not beat, noise (DM z -0.58) | 230.50 vs 230.16 (-0.34, -0.15%): does not beat, noise (DM z -1.54) |
| zero_delta | delta/mae | 112.28 vs 112.32 (+0.04, +0.04%): beats, noise (DM z +1.89) | 137.89 vs 137.87 (-0.02, -0.01%): does not beat, noise (DM z -0.22) | 159.11 vs 158.85 (-0.26, -0.16%): does not beat, noise (DM z -1.52) |
| mean_delta | delta/rmse | 165.08 vs 165.39 (+0.31, +0.19%): beats (DM z +2.36) | 199.83 vs 200.31 (+0.49, +0.24%): beats, noise (DM z +1.75) | 230.50 vs 231.00 (+0.50, +0.22%): beats, noise (DM z +1.19) |
| mean_delta | delta/mae | 112.28 vs 112.73 (+0.45, +0.40%): beats (DM z +4.03) | 137.89 vs 138.56 (+0.67, +0.49%): beats (DM z +2.87) | 159.11 vs 159.88 (+0.77, +0.48%): beats (DM z +2.16) |
| const_var | variance/crps | 82.83 vs 84.15 (+1.32, +1.57%): beats (DM z +10.58) | 101.59 vs 102.91 (+1.32, +1.28%): beats (DM z +6.52) | 117.05 vs 118.88 (+1.82, +1.53%): beats (DM z +6.45) |
| const_var | variance/nll | 6.5101 vs 6.5630 (+0.0529): beats (DM z +3.72) | 6.6782 vs 6.7470 (+0.0688): beats (DM z +4.65) | 6.8632 vs 6.8845 (+0.0212): beats, noise (DM z +1.54) |
| const_var | variance/pit_ks | 0.0183 vs 0.0612 (+0.0429): beats (boot z +7.96) | 0.0277 vs 0.0652 (+0.0376): beats (boot z +6.18) | 0.0205 vs 0.0725 (+0.0520): beats (boot z +6.24) |
| const_var | variance/corr_var_err2_spearman | 0.2472 vs 0.0000 (+0.2472): beats (boot z +17.62) | 0.2096 vs 0.0000 (+0.2096): beats (boot z +15.21) | 0.2279 vs 0.0000 (+0.2279): beats (boot z +13.83) |

## Backtest (costs included)

- n_trades: 1753
- total_return: -0.9889
- sharpe_net: -151.4321
- sharpe_gross: 0.4274
- sortino: -167.9097
- max_drawdown: 0.9889
- hit_rate: 0.0399
- hit_rate_gross: 0.5265
- profit_factor: 0.0204
- avg_hold_bars: 9.2407
- exposure: 0.3750
- turnover: 762.6641
- fees_paid: 7626.7764
- traded_notional: 7626776.3572
- breakeven_cost_bps: 0.0675
- gross_edge_per_trade_bps: 0.3671
- costs_paid: 9914.8093
- gross_pnl: 25.7280
- net_pnl: -9889.0812

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7884, long_above 0.5462, short_below 0.4085, median 0.4852. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.89% | -151.43 | +98.89% | 1753 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.14% .. -98.85%) | -99.00% | -169.75 | | |

The random null enters at the strategy's rate (0.0649 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 85% of its seeds on net return, 100% on net Sharpe and 54% on gross return.
