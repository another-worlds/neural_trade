# Evaluation report - dev split - run `20260929T192706Z-60ad158-773c434e-ohlcv14__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.4771 | 0.7339 | 0.7149 |
| accuracy | 0.4957 | 0.4954 | 0.4906 |
| balanced accuracy | 0.4952 | 0.5003 | 0.4958 |
| precision (up) | 0.4847 | 0.4897 | 0.4849 |
| recall / sensitivity (up) | 0.4722 | 0.7342 | 0.7106 |
| specificity (down) | 0.5182 | 0.2664 | 0.2810 |
| F1 (up) | 0.4784 | 0.5875 | 0.5765 |
| MCC | -0.0096 | 0.0006 | -0.0093 |
| AUC | 0.4895 | 0.5048 | 0.5021 |
| Brier | 0.2632 | 0.2545 | 0.2574 |
| ECE (positive class) | 0.0799 | 0.0539 | 0.0717 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 6164 / 6554 / 7049 / 6889 | 10693 / 11141 / 4045 / 3872 | 10890 / 11568 / 4520 / 4434 |
| Gaussian readout: calls up | 0.8259 | 0.8253 | 0.7981 |
| Gaussian readout: MCC | -0.0004 | 0.0003 | -0.0110 |
| Gaussian readout: AUC | 0.4983 | 0.5132 | 0.5040 |
| Gaussian readout: Brier | 0.2502 | 0.2503 | 0.2508 |
| Gaussian readout: ECE | 0.0153 | 0.0235 | 0.0326 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.11 | 199.83 | 230.35 |
| RMSE ($), raw heads | 166.74 | 203.69 | 234.15 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.34 | 137.97 | 159.12 |
| MAE ($), raw heads | 114.30 | 141.83 | 163.20 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0002 | -0.0006 | -0.0017 |
| skill vs zero, raw heads | -0.0201 | -0.0397 | -0.0350 |
| EV, served | -0.0000 | 0.0003 | -0.0002 |
| EV, raw heads | -0.0104 | -0.0183 | -0.0178 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0003 | 0.0169 | 0.0136 |
| corr, Spearman, raw heads | -0.0068 | 0.0168 | 0.0065 |
| mean predicted ($), served | 0.92 | 3.93 | 6.13 |
| mean predicted ($), raw heads | 14.74 | 26.89 | 27.16 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.8210 | 0.8205 | 0.7967 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.0623 | 0.1460 | 0.2257 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 83.76 | 101.57 | 117.33 |
| CRPSS vs constant variance | 0.0046 | 0.0130 | 0.0130 |
| NLL | 6.6484 | 6.7169 | 6.8529 |
| PIT KS | 0.0310 | 0.0318 | 0.0349 |
| var / err^2 Spearman | 0.1429 | 0.1960 | 0.2036 |
| coverage of the 90% interval | 0.9063 | 0.9041 | 0.8977 |
| width of the 90% interval ($) | 513.94 | 622.64 | 700.09 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0163 | [-0.0325, 0.0008] | NOISE |
| h1 | 0.0049 | [-0.0128, 0.0222] | NOISE |
| h2 | 0.0105 | [-0.0093, 0.0296] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.062 / h1 0.146 / h2 0.226) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6781 | 0.8670 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.5501 | 0.7524 | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.2986 | 0.6317 | 0.3339 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5410 | 0.7109 | 0.7012 | 0.3019 |
| expected if the two signs were independent | 0.4849 | 0.6559 | 0.6212 | 0.2462 |

- P(up) unanimity (all three horizons call the same side): 0.3939

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0096 vs 0.0011 (-0.0107): does not beat, noise (boot z -0.53) | 0.0006 vs 0.0027 (-0.0021): does not beat, noise (boot z -0.11) | -0.0093 vs -0.0018 (-0.0075): does not beat, noise (boot z -0.37) |
| logreg_lags | direction/auc | 0.4895 vs 0.5085 (-0.0190): does not beat, noise (boot z -1.47) | 0.5048 vs 0.5137 (-0.0090): does not beat, noise (boot z -0.78) | 0.5021 vs 0.5174 (-0.0153): does not beat, noise (boot z -1.19) |
| logreg_lags | direction/brier | 0.2632 vs 0.2513 (-0.0119): does not beat, significantly worse (DM z -7.49) | 0.2545 vs 0.2517 (-0.0028): does not beat, significantly worse (DM z -3.57) | 0.2574 vs 0.2518 (-0.0055): does not beat, significantly worse (DM z -4.94) |
| logreg_lags | direction/ece_pos | 0.0799 vs 0.0347 (-0.0452): does not beat, significantly worse (boot z -4.77) | 0.0539 vs 0.0387 (-0.0152): does not beat, significantly worse (boot z -2.45) | 0.0717 vs 0.0432 (-0.0285): does not beat, significantly worse (boot z -3.66) |
| logreg_lags | direction/acc | 0.4957 vs 0.4921 (+0.0036): beats, noise (DM z +0.36) | 0.4954 vs 0.4923 (+0.0031): beats, noise (DM z +0.43) | 0.4906 vs 0.4895 (+0.0011): beats, noise (DM z +0.14) |
| logreg_lags | direction/bal_acc | 0.4952 vs 0.5003 (-0.0051): does not beat, noise (boot z -0.65) | 0.5003 vs 0.5008 (-0.0005): does not beat, noise (boot z -0.08) | 0.4958 vs 0.4995 (-0.0037): does not beat, noise (boot z -0.54) |
| class_prior | direction/mcc | -0.0096 vs 0.0000 (-0.0096): does not beat, noise (boot z -0.81) | 0.0006 vs 0.0000 (+0.0006): beats, noise (boot z +0.06) | -0.0093 vs 0.0000 (-0.0093): does not beat, noise (boot z -0.90) |
| class_prior | direction/auc | 0.4895 vs 0.5000 (-0.0105): does not beat, noise (boot z -1.37) | 0.5048 vs 0.5000 (+0.0048): beats, noise (boot z +0.68) | 0.5021 vs 0.5000 (+0.0021): beats, noise (boot z +0.29) |
| class_prior | direction/brier | 0.2632 vs 0.2508 (-0.0125): does not beat, significantly worse (DM z -8.43) | 0.2545 vs 0.2510 (-0.0035): does not beat, significantly worse (DM z -4.77) | 0.2574 vs 0.2512 (-0.0061): does not beat, significantly worse (DM z -5.71) |
| class_prior | direction/ece_pos | 0.0799 vs 0.0297 (-0.0503): does not beat, significantly worse (boot z -5.14) | 0.0539 vs 0.0335 (-0.0204): does not beat, significantly worse (boot z -3.30) | 0.0717 vs 0.0371 (-0.0346): does not beat, significantly worse (boot z -4.53) |
| class_prior | direction/acc | 0.4957 vs 0.4897 (+0.0060): beats, noise (DM z +0.59) | 0.4954 vs 0.4896 (+0.0058): beats, noise (DM z +0.83) | 0.4906 vs 0.4878 (+0.0027): beats, noise (DM z +0.36) |
| class_prior | direction/bal_acc | 0.4952 vs 0.5000 (-0.0048): does not beat, noise (boot z -0.81) | 0.5003 vs 0.5000 (+0.0003): beats, noise (boot z +0.06) | 0.4958 vs 0.5000 (-0.0042): does not beat, noise (boot z -0.90) |
| zero_delta | delta/rmse | 165.11 vs 165.09 (-0.01, -0.01%): does not beat, noise (DM z -0.75) | 199.83 vs 199.76 (-0.06, -0.03%): does not beat, noise (DM z -0.60) | 230.35 vs 230.16 (-0.19, -0.08%): does not beat, noise (DM z -1.03) |
| zero_delta | delta/mae | 112.34 vs 112.32 (-0.03, -0.02%): does not beat, noise (DM z -1.70) | 137.97 vs 137.87 (-0.10, -0.08%): does not beat, noise (DM z -1.33) | 159.12 vs 158.85 (-0.27, -0.17%): does not beat, noise (DM z -1.89) |
| mean_delta | delta/rmse | 165.11 vs 165.39 (+0.28, +0.17%): beats (DM z +2.77) | 199.83 vs 200.31 (+0.49, +0.24%): beats (DM z +3.14) | 230.35 vs 231.00 (+0.65, +0.28%): beats (DM z +2.54) |
| mean_delta | delta/mae | 112.34 vs 112.73 (+0.39, +0.34%): beats (DM z +4.26) | 137.97 vs 138.56 (+0.59, +0.42%): beats (DM z +4.29) | 159.12 vs 159.88 (+0.76, +0.47%): beats (DM z +3.62) |
| const_var | variance/crps | 83.76 vs 84.15 (+0.39, +0.46%): beats (DM z +3.03) | 101.57 vs 102.91 (+1.34, +1.30%): beats (DM z +9.35) | 117.33 vs 118.88 (+1.55, +1.30%): beats (DM z +5.98) |
| const_var | variance/nll | 6.6484 vs 6.5630 (-0.0854): does not beat, significantly worse (DM z -3.87) | 6.7169 vs 6.7470 (+0.0301): beats (DM z +2.46) | 6.8529 vs 6.8845 (+0.0316): beats, noise (DM z +1.73) |
| const_var | variance/pit_ks | 0.0310 vs 0.0612 (+0.0302): beats (boot z +4.39) | 0.0318 vs 0.0652 (+0.0334): beats (boot z +20.26) | 0.0349 vs 0.0725 (+0.0376): beats (boot z +15.33) |
| const_var | variance/corr_var_err2_spearman | 0.1429 vs 0.0000 (+0.1429): beats (boot z +10.47) | 0.1960 vs 0.0000 (+0.1960): beats (boot z +11.77) | 0.2036 vs 0.0000 (+0.2036): beats (boot z +12.15) |

## Backtest (costs included)

- n_trades: 1951
- total_return: -0.9939
- sharpe_net: -167.5900
- sharpe_gross: -9.0057
- sortino: -182.9478
- max_drawdown: 0.9939
- hit_rate: 0.0482
- hit_rate_gross: 0.4505
- profit_factor: 0.0228
- avg_hold_bars: 8.1230
- exposure: 0.3669
- turnover: 721.2000
- fees_paid: 7212.0430
- traded_notional: 7212043.0088
- breakeven_cost_bps: -1.5610
- gross_edge_per_trade_bps: -0.0547
- costs_paid: 9375.6559
- gross_pnl: -562.8991
- net_pnl: -9938.5550

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8104, long_above 0.5824, short_below 0.4514, median 0.5186. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.39% | -167.59 | +99.39% | 1951 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.47% .. -99.30%) | -99.39% | -183.07 | | |

The random null enters at the strategy's rate (0.0713 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 55% of its seeds on net return, 100% on net Sharpe and 0% on gross return.
