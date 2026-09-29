# Evaluation report - dev split - run `20260929T170336Z-e1c8b93-c3f9b9a6-close4__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.4184 | 0.6603 | 0.5530 |
| accuracy | 0.5031 | 0.5002 | 0.5064 |
| balanced accuracy | 0.5014 | 0.5035 | 0.5077 |
| precision (up) | 0.4913 | 0.4922 | 0.4948 |
| recall / sensitivity (up) | 0.4198 | 0.6639 | 0.5609 |
| specificity (down) | 0.5830 | 0.3431 | 0.4546 |
| F1 (up) | 0.4528 | 0.5653 | 0.5258 |
| MCC | 0.0028 | 0.0074 | 0.0155 |
| AUC | 0.5009 | 0.5061 | 0.5170 |
| Brier | 0.2581 | 0.2531 | 0.2544 |
| ECE (positive class) | 0.0621 | 0.0388 | 0.0494 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 5480 / 5673 / 7930 / 7573 | 9670 / 9976 / 5210 / 4895 | 8595 / 8775 / 7313 / 6729 |
| Gaussian readout: calls up | 0.7161 | 0.6838 | 0.6103 |
| Gaussian readout: MCC | -0.0071 | -0.0018 | -0.0075 |
| Gaussian readout: AUC | 0.5019 | 0.5108 | 0.5011 |
| Gaussian readout: Brier | 0.2507 | 0.2500 | 0.2502 |
| Gaussian readout: ECE | 0.0269 | 0.0185 | 0.0179 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.16 | 199.80 | 230.28 |
| RMSE ($), raw heads | 166.84 | 203.59 | 236.33 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.41 | 137.87 | 158.93 |
| MAE ($), raw heads | 113.97 | 140.54 | 163.39 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0008 | -0.0004 | -0.0011 |
| skill vs zero, raw heads | -0.0213 | -0.0386 | -0.0543 |
| EV, served | -0.0000 | 0.0002 | -0.0006 |
| EV, raw heads | -0.0147 | -0.0288 | -0.0463 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0176 | 0.0197 | 0.0082 |
| corr, Spearman, raw heads | 0.0025 | 0.0034 | -0.0063 |
| mean predicted ($), served | 3.09 | 3.00 | 2.68 |
| mean predicted ($), raw heads | 11.89 | 17.60 | 17.63 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.6947 | 0.6590 | 0.5902 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.2599 | 0.1702 | 0.1521 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 83.46 | 101.58 | 116.98 |
| CRPSS vs constant variance | 0.0082 | 0.0129 | 0.0160 |
| NLL | 6.5423 | 6.7254 | 6.8450 |
| PIT KS | 0.0351 | 0.0288 | 0.0334 |
| var / err^2 Spearman | 0.2168 | 0.2042 | 0.2224 |
| coverage of the 90% interval | 0.9054 | 0.9047 | 0.8976 |
| width of the 90% interval ($) | 513.64 | 623.91 | 697.89 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0086 | [-0.0094, 0.0281] | NOISE |
| h1 | 0.0009 | [-0.0157, 0.0170] | NOISE |
| h2 | 0.0224 | [0.0044, 0.0413] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.260 / h1 0.170 / h2 0.152) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6300 | 0.4742 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.5885 | 0.5431 | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.3017 | 0.1852 | 0.3339 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5001 | 0.6155 | 0.5998 | 0.2101 |
| expected if the two signs were independent | 0.4649 | 0.5505 | 0.5075 | 0.1502 |

- P(up) unanimity (all three horizons call the same side): 0.3286

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0028 vs 0.0011 (+0.0017): beats, noise (boot z +0.08) | 0.0074 vs 0.0027 (+0.0047): beats, noise (boot z +0.24) | 0.0155 vs -0.0018 (+0.0174): beats, noise (boot z +0.90) |
| logreg_lags | direction/auc | 0.5009 vs 0.5085 (-0.0076): does not beat, noise (boot z -0.59) | 0.5061 vs 0.5137 (-0.0076): does not beat, noise (boot z -0.67) | 0.5170 vs 0.5174 (-0.0004): does not beat, noise (boot z -0.03) |
| logreg_lags | direction/brier | 0.2581 vs 0.2513 (-0.0068): does not beat, significantly worse (DM z -4.68) | 0.2531 vs 0.2517 (-0.0015): does not beat, significantly worse (DM z -2.08) | 0.2544 vs 0.2518 (-0.0026): does not beat, significantly worse (DM z -2.03) |
| logreg_lags | direction/ece_pos | 0.0621 vs 0.0347 (-0.0274): does not beat, significantly worse (boot z -2.56) | 0.0388 vs 0.0387 (-0.0001): does not beat, noise (boot z -0.02) | 0.0494 vs 0.0432 (-0.0062): does not beat, noise (boot z -0.64) |
| logreg_lags | direction/acc | 0.5031 vs 0.4921 (+0.0110): beats, noise (DM z +1.07) | 0.5002 vs 0.4923 (+0.0079): beats, noise (DM z +0.99) | 0.5064 vs 0.4895 (+0.0170): beats, noise (DM z +1.69) |
| logreg_lags | direction/bal_acc | 0.5014 vs 0.5003 (+0.0011): beats, noise (boot z +0.14) | 0.5035 vs 0.5008 (+0.0027): beats, noise (boot z +0.41) | 0.5077 vs 0.4995 (+0.0082): beats, noise (boot z +1.18) |
| class_prior | direction/mcc | 0.0028 vs 0.0000 (+0.0028): beats, noise (boot z +0.23) | 0.0074 vs 0.0000 (+0.0074): beats, noise (boot z +0.73) | 0.0155 vs 0.0000 (+0.0155): beats, noise (boot z +1.33) |
| class_prior | direction/auc | 0.5009 vs 0.5000 (+0.0009): beats, noise (boot z +0.11) | 0.5061 vs 0.5000 (+0.0061): beats, noise (boot z +0.95) | 0.5170 vs 0.5000 (+0.0170): beats (boot z +2.13) |
| class_prior | direction/brier | 0.2581 vs 0.2508 (-0.0073): does not beat, significantly worse (DM z -5.37) | 0.2531 vs 0.2510 (-0.0021): does not beat, significantly worse (DM z -3.19) | 0.2544 vs 0.2512 (-0.0032): does not beat, significantly worse (DM z -2.44) |
| class_prior | direction/ece_pos | 0.0621 vs 0.0297 (-0.0324): does not beat, significantly worse (boot z -2.91) | 0.0388 vs 0.0335 (-0.0054): does not beat, noise (boot z -0.74) | 0.0494 vs 0.0371 (-0.0124): does not beat, noise (boot z -1.23) |
| class_prior | direction/acc | 0.5031 vs 0.4897 (+0.0134): beats, noise (DM z +1.24) | 0.5002 vs 0.4896 (+0.0106): beats, noise (DM z +1.33) | 0.5064 vs 0.4878 (+0.0186): beats, noise (DM z +1.73) |
| class_prior | direction/bal_acc | 0.5014 vs 0.5000 (+0.0014): beats, noise (boot z +0.23) | 0.5035 vs 0.5000 (+0.0035): beats, noise (boot z +0.73) | 0.5077 vs 0.5000 (+0.0077): beats, noise (boot z +1.33) |
| zero_delta | delta/rmse | 165.16 vs 165.09 (-0.06, -0.04%): does not beat, noise (DM z -0.57) | 199.80 vs 199.76 (-0.04, -0.02%): does not beat, noise (DM z -0.25) | 230.28 vs 230.16 (-0.12, -0.05%): does not beat, noise (DM z -0.60) |
| zero_delta | delta/mae | 112.41 vs 112.32 (-0.09, -0.08%): does not beat, noise (DM z -1.21) | 137.87 vs 137.87 (-0.00, -0.00%): does not beat, noise (DM z -0.01) | 158.93 vs 158.85 (-0.08, -0.05%): does not beat, noise (DM z -0.60) |
| mean_delta | delta/rmse | 165.16 vs 165.39 (+0.23, +0.14%): beats (DM z +2.23) | 199.80 vs 200.31 (+0.51, +0.26%): beats (DM z +2.89) | 230.28 vs 231.00 (+0.72, +0.31%): beats (DM z +2.44) |
| mean_delta | delta/mae | 112.41 vs 112.73 (+0.32, +0.29%): beats (DM z +3.46) | 137.87 vs 138.56 (+0.69, +0.50%): beats (DM z +4.16) | 158.93 vs 159.88 (+0.95, +0.60%): beats (DM z +3.58) |
| const_var | variance/crps | 83.46 vs 84.15 (+0.69, +0.82%): beats (DM z +4.47) | 101.58 vs 102.91 (+1.32, +1.29%): beats (DM z +7.04) | 116.98 vs 118.88 (+1.90, +1.60%): beats (DM z +6.85) |
| const_var | variance/nll | 6.5423 vs 6.5630 (+0.0206): beats, noise (DM z +1.06) | 6.7254 vs 6.7470 (+0.0216): beats, noise (DM z +1.21) | 6.8450 vs 6.8845 (+0.0394): beats (DM z +2.28) |
| const_var | variance/pit_ks | 0.0351 vs 0.0612 (+0.0261): beats (boot z +11.79) | 0.0288 vs 0.0652 (+0.0365): beats (boot z +14.80) | 0.0334 vs 0.0725 (+0.0391): beats (boot z +13.51) |
| const_var | variance/corr_var_err2_spearman | 0.2168 vs 0.0000 (+0.2168): beats (boot z +14.72) | 0.2042 vs 0.0000 (+0.2042): beats (boot z +12.42) | 0.2224 vs 0.0000 (+0.2224): beats (boot z +13.19) |

## Backtest (costs included)

- n_trades: 1909
- total_return: -0.9934
- sharpe_net: -166.3992
- sharpe_gross: -8.1913
- sortino: -180.4051
- max_drawdown: 0.9934
- hit_rate: 0.0424
- hit_rate_gross: 0.4798
- profit_factor: 0.0158
- avg_hold_bars: 7.9497
- exposure: 0.3513
- turnover: 725.3230
- fees_paid: 7253.3842
- traded_notional: 7253384.1909
- breakeven_cost_bps: -1.3909
- gross_edge_per_trade_bps: -0.2407
- costs_paid: 9429.3994
- gross_pnl: -504.4332
- net_pnl: -9933.8326

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8495, long_above 0.5523, short_below 0.4516, median 0.4982. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.34% | -166.40 | +99.34% | 1909 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.39% .. -99.20%) | -99.29% | -180.30 | | |

The random null enters at the strategy's rate (0.0681 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 25% of its seeds on net return, 100% on net Sharpe and 0% on gross return.
