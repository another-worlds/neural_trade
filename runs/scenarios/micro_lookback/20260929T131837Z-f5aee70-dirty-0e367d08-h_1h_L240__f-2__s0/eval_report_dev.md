# Evaluation report - dev split - run `20260929T131837Z-f5aee70-dirty-0e367d08-h_1h_L240__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (40 bars) | h1 (60 bars) | h2 (80 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 34654 | 36286 | 36934 |
| n_eff of the scored moves (n scored // bars ahead) | 866 | 604 | 461 |
| true up-rate | 0.4870 | 0.4827 | 0.4851 |
| calls up (predicted up-rate) | 0.5236 | 0.7379 | 0.7190 |
| accuracy | 0.4924 | 0.4920 | 0.4998 |
| balanced accuracy | 0.4930 | 0.5002 | 0.5064 |
| precision (up) | 0.4803 | 0.4829 | 0.4895 |
| recall / sensitivity (up) | 0.5164 | 0.7382 | 0.7256 |
| specificity (down) | 0.4696 | 0.2623 | 0.2872 |
| F1 (up) | 0.4977 | 0.5838 | 0.5846 |
| MCC | -0.0140 | 0.0005 | 0.0142 |
| AUC | 0.4948 | 0.5040 | 0.5182 |
| Brier | 0.2632 | 0.2534 | 0.2567 |
| ECE (positive class) | 0.0896 | 0.0511 | 0.0668 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0130 | 0.0173 | 0.0149 |
| TP / FP / TN / FN | 8715 / 9429 / 8348 / 8162 | 12930 / 13847 / 4923 / 4586 | 12999 / 13556 / 5462 / 4917 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.7402 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0159 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4966 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2500 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0178 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.8224 | 0.7402 | 0.6576 |
| Gaussian readout of the raw heads: MCC | -0.0310 | -0.0159 | -0.0255 |
| Gaussian readout of the raw heads: AUC | 0.4975 | 0.4966 | 0.4908 |
| Gaussian readout of the raw heads: Brier | 0.2703 | 0.2852 | 0.2773 |
| Gaussian readout of the raw heads: ECE | 0.1329 | 0.1658 | 0.1518 |

beta = 0 for h0, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (40 bars) | h1 (60 bars) | h2 (80 bars) |
|---|---|---|---|
| RMSE ($), served | 326.58 | 399.96 | 462.20 |
| RMSE ($), raw heads | 333.98 | 419.20 | 487.94 |
| RMSE ($), zero prediction | 326.58 | 399.93 | 462.20 |
| MAE ($), served | 224.08 | 270.93 | 310.92 |
| MAE ($), raw heads | 232.31 | 289.41 | 335.43 |
| MAE ($), zero prediction | 224.08 | 270.91 | 310.92 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0001 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0458 | -0.0987 | -0.1144 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0001 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0241 | -0.0709 | -0.0872 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0074 | -0.0321 | -0.0299 |
| corr, Spearman, raw heads | -0.0108 | -0.0231 | -0.0256 |
| mean predicted ($), served | 0.00 | 0.32 | 0.00 |
| mean predicted ($), raw heads | 41.61 | 57.08 | 64.02 |
| mean realised ($) | -7.01 | -10.43 | -13.64 |
| share predicted up, raw heads | 0.8134 | 0.7291 | 0.6472 |
| share realised up | 0.4863 | 0.4823 | 0.4847 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0055 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (40 bars) | h1 (60 bars) | h2 (80 bars) |
|---|---|---|---|
| CRPS ($) | 168.53 | 204.86 | 232.38 |
| CRPSS vs constant variance | 0.0020 | 0.0062 | 0.0265 |
| NLL | 7.4295 | 7.6577 | 7.5919 |
| PIT KS | 0.0334 | 0.0359 | 0.0296 |
| var / err^2 Spearman | 0.0346 | 0.0526 | 0.1267 |
| coverage of the 90% interval | 0.9043 | 0.9035 | 0.9092 |
| width of the 90% interval ($) | 1008.96 | 1210.00 | 1427.63 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0055 | [-0.0215, 0.0308] | NOISE |
| h1 | -0.0062 | [-0.0350, 0.0206] | NOISE |
| h2 | 0.0169 | [-0.0139, 0.0483] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.006 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8234 | n/a (beta = 0: served delta is 0) | 0.5993 |
| abs(d h1) <= abs(d h2) | 0.6941 | n/a (beta = 0: served delta is 0) | 0.5794 |
| full chain h0 <= h1 <= h2 | 0.5591 | n/a (beta = 0: served delta is 0) | 0.3136 |

beta = 0 for h0, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6078 | 0.6184 | 0.5951 | 0.2884 |
| expected if the two signs were independent | 0.5103 | 0.6119 | 0.5623 | 0.2540 |

- P(up) unanimity (all three horizons call the same side): 0.4404

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (40 bars) | h1 (60 bars) | h2 (80 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0140 vs -0.0176 (+0.0035): beats, noise (boot z +0.12) | 0.0005 vs -0.0044 (+0.0049): beats, noise (boot z +0.20) | 0.0142 vs 0.0150 (-0.0008): does not beat, noise (boot z -0.03) |
| logreg_lags | direction/auc | 0.4948 vs 0.5264 (-0.0316): does not beat, noise (boot z -1.60) | 0.5040 vs 0.5385 (-0.0344): does not beat, noise (boot z -1.76) | 0.5182 vs 0.5300 (-0.0118): does not beat, noise (boot z -0.87) |
| logreg_lags | direction/brier | 0.2632 vs 0.2534 (-0.0098): does not beat, significantly worse (DM z -3.15) | 0.2534 vs 0.2541 (+0.0007): beats, noise (DM z +0.42) | 0.2567 vs 0.2568 (+0.0000): beats, noise (DM z +0.02) |
| logreg_lags | direction/ece_pos | 0.0896 vs 0.0645 (-0.0252): does not beat, noise (boot z -1.70) | 0.0511 vs 0.0724 (+0.0213): beats (boot z +2.61) | 0.0668 vs 0.0768 (+0.0100): beats, noise (boot z +1.27) |
| logreg_lags | direction/acc | 0.4924 vs 0.4856 (+0.0068): beats, noise (DM z +0.42) | 0.4920 vs 0.4827 (+0.0093): beats, noise (DM z +0.83) | 0.4998 vs 0.4943 (+0.0055): beats, noise (DM z +0.49) |
| logreg_lags | direction/bal_acc | 0.4930 vs 0.4983 (-0.0053): does not beat, noise (boot z -0.54) | 0.5002 vs 0.4994 (+0.0008): beats, noise (boot z +0.10) | 0.5064 vs 0.5052 (+0.0012): beats, noise (boot z +0.12) |
| class_prior | direction/mcc | -0.0140 vs 0.0000 (-0.0140): does not beat, noise (boot z -0.74) | 0.0005 vs 0.0000 (+0.0005): beats, noise (boot z +0.03) | 0.0142 vs 0.0000 (+0.0142): beats, noise (boot z +0.70) |
| class_prior | direction/auc | 0.4948 vs 0.5000 (-0.0052): does not beat, noise (boot z -0.42) | 0.5040 vs 0.5000 (+0.0040): beats, noise (boot z +0.33) | 0.5182 vs 0.5000 (+0.0182): beats, noise (boot z +1.21) |
| class_prior | direction/brier | 0.2632 vs 0.2524 (-0.0108): does not beat, significantly worse (DM z -4.00) | 0.2534 vs 0.2532 (-0.0003): does not beat, noise (DM z -0.22) | 0.2567 vs 0.2525 (-0.0043): does not beat, significantly worse (DM z -2.02) |
| class_prior | direction/ece_pos | 0.0896 vs 0.0502 (-0.0394): does not beat, significantly worse (boot z -2.62) | 0.0511 vs 0.0591 (+0.0080): beats, noise (boot z +0.95) | 0.0668 vs 0.0518 (-0.0150): does not beat, noise (boot z -1.68) |
| class_prior | direction/acc | 0.4924 vs 0.4870 (+0.0054): beats, noise (DM z +0.33) | 0.4920 vs 0.4827 (+0.0093): beats, noise (DM z +0.81) | 0.4998 vs 0.4851 (+0.0148): beats, noise (DM z +1.08) |
| class_prior | direction/bal_acc | 0.4930 vs 0.5000 (-0.0070): does not beat, noise (boot z -0.74) | 0.5002 vs 0.5000 (+0.0002): beats, noise (boot z +0.03) | 0.5064 vs 0.5000 (+0.0064): beats, noise (boot z +0.70) |
| zero_delta | delta/rmse | 326.58 vs 326.58 (+0.00, +0.00%): does not beat | 399.96 vs 399.93 (-0.03, -0.01%): does not beat, noise (DM z -1.24) | 462.20 vs 462.20 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 224.08 vs 224.08 (+0.00, +0.00%): does not beat | 270.93 vs 270.91 (-0.02, -0.01%): does not beat, noise (DM z -1.02) | 310.92 vs 310.92 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 326.58 vs 329.37 (+2.79, +0.85%): beats (DM z +2.78) | 399.96 vs 404.98 (+5.03, +1.24%): beats (DM z +2.78) | 462.20 vs 469.89 (+7.68, +1.64%): beats (DM z +2.75) |
| mean_delta | delta/mae | 224.08 vs 227.27 (+3.19, +1.40%): beats (DM z +3.83) | 270.93 vs 276.75 (+5.83, +2.11%): beats (DM z +3.90) | 310.92 vs 319.40 (+8.49, +2.66%): beats (DM z +3.75) |
| const_var | variance/crps | 168.53 vs 168.88 (+0.35, +0.20%): beats, noise (DM z +0.50) | 204.86 vs 206.14 (+1.28, +0.62%): beats, noise (DM z +1.13) | 232.38 vs 238.71 (+6.33, +2.65%): beats (DM z +3.88) |
| const_var | variance/nll | 7.4295 vs 7.2371 (-0.1924): does not beat, significantly worse (DM z -3.30) | 7.6577 vs 7.4415 (-0.2162): does not beat, significantly worse (DM z -3.05) | 7.5919 vs 7.5918 (-0.0001): does not beat, noise (DM z -0.00) |
| const_var | variance/pit_ks | 0.0334 vs 0.0886 (+0.0552): beats (boot z +4.94) | 0.0359 vs 0.1060 (+0.0701): beats (boot z +5.96) | 0.0296 vs 0.1195 (+0.0899): beats (boot z +11.48) |
| const_var | variance/corr_var_err2_spearman | 0.0346 vs 0.0000 (+0.0346): beats, noise (boot z +1.51) | 0.0526 vs 0.0000 (+0.0526): beats (boot z +1.98) | 0.1267 vs 0.0000 (+0.1267): beats (boot z +5.05) |

## Backtest (costs included)

- n_trades: 1499
- total_return: -0.9795
- sharpe_net: -135.2311
- sharpe_gross: -1.7159
- sortino: -151.7809
- max_drawdown: 0.9795
- hit_rate: 0.0627
- hit_rate_gross: 0.4663
- profit_factor: 0.0391
- avg_hold_bars: 11.3289
- exposure: 0.3931
- turnover: 743.7520
- fees_paid: 7437.8863
- traded_notional: 7437886.3372
- breakeven_cost_bps: -0.3377
- gross_edge_per_trade_bps: 0.1221
- costs_paid: 9669.2522
- gross_pnl: -125.6067
- net_pnl: -9794.8589

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T22:38:00 .. 2025-08-30T22:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7135, long_above 0.5850, short_below 0.4515, median 0.5238. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -97.95% | -135.23 | +97.95% | 1499 |
| buy and hold | -6.74% | -2.35 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.32% .. -97.75%) | -98.04% | -150.02 | | |

The random null enters at the strategy's rate (0.0572 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 70% of its seeds on net return, 100% on net Sharpe and 32% on gross return.
