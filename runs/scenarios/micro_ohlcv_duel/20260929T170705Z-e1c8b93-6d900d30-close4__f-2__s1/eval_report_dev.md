# Evaluation report - dev split - run `20260929T170705Z-e1c8b93-6d900d30-close4__f-2__s1`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.3534 | 0.4884 | 0.6295 |
| accuracy | 0.5054 | 0.5030 | 0.4996 |
| balanced accuracy | 0.5023 | 0.5028 | 0.5027 |
| precision (up) | 0.4930 | 0.4924 | 0.4900 |
| recall / sensitivity (up) | 0.3558 | 0.4912 | 0.6323 |
| specificity (down) | 0.6489 | 0.5143 | 0.3731 |
| F1 (up) | 0.4133 | 0.4918 | 0.5522 |
| MCC | 0.0049 | 0.0055 | 0.0057 |
| AUC | 0.5054 | 0.5085 | 0.5093 |
| Brier | 0.2565 | 0.2549 | 0.2583 |
| ECE (positive class) | 0.0584 | 0.0571 | 0.0648 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 4644 / 4776 / 8827 / 8409 | 7155 / 7376 / 7810 / 7410 | 9690 / 10085 / 6003 / 5634 |
| Gaussian readout: calls up | 0.7222 | 0.5925 | 0.4781 |
| Gaussian readout: MCC | -0.0066 | -0.0055 | -0.0028 |
| Gaussian readout: AUC | 0.5007 | 0.5028 | 0.5012 |
| Gaussian readout: Brier | 0.2504 | 0.2506 | 0.2502 |
| Gaussian readout: ECE | 0.0205 | 0.0196 | 0.0150 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.14 | 199.88 | 230.23 |
| RMSE ($), raw heads | 167.73 | 206.20 | 236.00 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.36 | 137.96 | 158.90 |
| MAE ($), raw heads | 114.53 | 142.40 | 163.20 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0006 | -0.0012 | -0.0006 |
| skill vs zero, raw heads | -0.0322 | -0.0655 | -0.0514 |
| EV, served | -0.0002 | -0.0008 | -0.0004 |
| EV, raw heads | -0.0240 | -0.0587 | -0.0475 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0092 | 0.0056 | 0.0097 |
| corr, Spearman, raw heads | 0.0008 | -0.0007 | -0.0031 |
| mean predicted ($), served | 2.13 | 2.02 | 1.60 |
| mean predicted ($), raw heads | 13.40 | 14.24 | 11.42 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.7230 | 0.5848 | 0.4700 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.1593 | 0.1418 | 0.1404 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 83.43 | 102.07 | 117.34 |
| CRPSS vs constant variance | 0.0086 | 0.0081 | 0.0129 |
| NLL | 6.5654 | 6.7567 | 6.8784 |
| PIT KS | 0.0292 | 0.0235 | 0.0243 |
| var / err^2 Spearman | 0.1800 | 0.1547 | 0.1978 |
| coverage of the 90% interval | 0.9055 | 0.9036 | 0.8982 |
| width of the 90% interval ($) | 513.34 | 622.92 | 699.72 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0216 | [0.0023, 0.0411] | WORKS |
| h1 | 0.0132 | [-0.0047, 0.0311] | NOISE |
| h2 | 0.0083 | [-0.0119, 0.0256] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.159 / h1 0.142 / h2 0.140) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6142 | 0.5675 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.5829 | 0.5752 | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.3302 | 0.2944 | 0.3339 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5469 | 0.6512 | 0.5554 | 0.2700 |
| expected if the two signs were independent | 0.4414 | 0.4984 | 0.4918 | 0.1619 |

- P(up) unanimity (all three horizons call the same side): 0.4361

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0049 vs 0.0011 (+0.0038): beats, noise (boot z +0.17) | 0.0055 vs 0.0027 (+0.0028): beats, noise (boot z +0.13) | 0.0057 vs -0.0018 (+0.0075): beats, noise (boot z +0.40) |
| logreg_lags | direction/auc | 0.5054 vs 0.5085 (-0.0032): does not beat, noise (boot z -0.23) | 0.5085 vs 0.5137 (-0.0053): does not beat, noise (boot z -0.35) | 0.5093 vs 0.5174 (-0.0082): does not beat, noise (boot z -0.78) |
| logreg_lags | direction/brier | 0.2565 vs 0.2513 (-0.0052): does not beat, significantly worse (DM z -3.59) | 0.2549 vs 0.2517 (-0.0033): does not beat, significantly worse (DM z -2.25) | 0.2583 vs 0.2518 (-0.0065): does not beat, significantly worse (DM z -4.61) |
| logreg_lags | direction/ece_pos | 0.0584 vs 0.0347 (-0.0237): does not beat, significantly worse (boot z -2.05) | 0.0571 vs 0.0387 (-0.0184): does not beat, noise (boot z -1.68) | 0.0648 vs 0.0432 (-0.0216): does not beat, significantly worse (boot z -2.58) |
| logreg_lags | direction/acc | 0.5054 vs 0.4921 (+0.0133): beats, noise (DM z +1.17) | 0.5030 vs 0.4923 (+0.0107): beats, noise (DM z +0.97) | 0.4996 vs 0.4895 (+0.0101): beats, noise (DM z +1.13) |
| logreg_lags | direction/bal_acc | 0.5023 vs 0.5003 (+0.0020): beats, noise (boot z +0.24) | 0.5028 vs 0.5008 (+0.0020): beats, noise (boot z +0.23) | 0.5027 vs 0.4995 (+0.0033): beats, noise (boot z +0.50) |
| class_prior | direction/mcc | 0.0049 vs 0.0000 (+0.0049): beats, noise (boot z +0.39) | 0.0055 vs 0.0000 (+0.0055): beats, noise (boot z +0.46) | 0.0057 vs 0.0000 (+0.0057): beats, noise (boot z +0.51) |
| class_prior | direction/auc | 0.5054 vs 0.5000 (+0.0054): beats, noise (boot z +0.67) | 0.5085 vs 0.5000 (+0.0085): beats, noise (boot z +1.07) | 0.5093 vs 0.5000 (+0.0093): beats, noise (boot z +1.21) |
| class_prior | direction/brier | 0.2565 vs 0.2508 (-0.0058): does not beat, significantly worse (DM z -4.22) | 0.2549 vs 0.2510 (-0.0039): does not beat, significantly worse (DM z -3.22) | 0.2583 vs 0.2512 (-0.0071): does not beat, significantly worse (DM z -4.27) |
| class_prior | direction/ece_pos | 0.0584 vs 0.0297 (-0.0287): does not beat, significantly worse (boot z -2.44) | 0.0571 vs 0.0335 (-0.0236): does not beat, significantly worse (boot z -2.15) | 0.0648 vs 0.0371 (-0.0277): does not beat, significantly worse (boot z -3.12) |
| class_prior | direction/acc | 0.5054 vs 0.4897 (+0.0157): beats, noise (DM z +1.32) | 0.5030 vs 0.4896 (+0.0134): beats, noise (DM z +1.20) | 0.4996 vs 0.4878 (+0.0117): beats, noise (DM z +1.22) |
| class_prior | direction/bal_acc | 0.5023 vs 0.5000 (+0.0023): beats, noise (boot z +0.39) | 0.5028 vs 0.5000 (+0.0028): beats, noise (boot z +0.46) | 0.5027 vs 0.5000 (+0.0027): beats, noise (boot z +0.51) |
| zero_delta | delta/rmse | 165.14 vs 165.09 (-0.05, -0.03%): does not beat, noise (DM z -0.66) | 199.88 vs 199.76 (-0.12, -0.06%): does not beat, noise (DM z -0.77) | 230.23 vs 230.16 (-0.07, -0.03%): does not beat, noise (DM z -0.43) |
| zero_delta | delta/mae | 112.36 vs 112.32 (-0.04, -0.04%): does not beat, noise (DM z -0.82) | 137.96 vs 137.87 (-0.10, -0.07%): does not beat, noise (DM z -0.99) | 158.90 vs 158.85 (-0.05, -0.03%): does not beat, noise (DM z -0.41) |
| mean_delta | delta/rmse | 165.14 vs 165.39 (+0.25, +0.15%): beats (DM z +2.48) | 199.88 vs 200.31 (+0.43, +0.21%): beats (DM z +2.08) | 230.23 vs 231.00 (+0.77, +0.33%): beats (DM z +2.43) |
| mean_delta | delta/mae | 112.36 vs 112.73 (+0.37, +0.33%): beats (DM z +4.13) | 137.96 vs 138.56 (+0.60, +0.43%): beats (DM z +3.27) | 158.90 vs 159.88 (+0.98, +0.61%): beats (DM z +3.54) |
| const_var | variance/crps | 83.43 vs 84.15 (+0.73, +0.86%): beats (DM z +5.01) | 102.07 vs 102.91 (+0.84, +0.81%): beats (DM z +4.36) | 117.34 vs 118.88 (+1.54, +1.29%): beats (DM z +6.19) |
| const_var | variance/nll | 6.5654 vs 6.5630 (-0.0024): does not beat, noise (DM z -0.15) | 6.7567 vs 6.7470 (-0.0097): does not beat, noise (DM z -0.62) | 6.8784 vs 6.8845 (+0.0061): beats, noise (DM z +0.42) |
| const_var | variance/pit_ks | 0.0292 vs 0.0612 (+0.0320): beats (boot z +14.63) | 0.0235 vs 0.0652 (+0.0418): beats (boot z +9.76) | 0.0243 vs 0.0725 (+0.0482): beats (boot z +11.54) |
| const_var | variance/corr_var_err2_spearman | 0.1800 vs 0.0000 (+0.1800): beats (boot z +13.00) | 0.1547 vs 0.0000 (+0.1547): beats (boot z +10.52) | 0.1978 vs 0.0000 (+0.1978): beats (boot z +11.71) |

## Backtest (costs included)

- n_trades: 2005
- total_return: -0.9939
- sharpe_net: -167.3229
- sharpe_gross: 7.7770
- sortino: -181.6360
- max_drawdown: 0.9939
- hit_rate: 0.0339
- hit_rate_gross: 0.5337
- profit_factor: 0.0194
- avg_hold_bars: 7.7551
- exposure: 0.3600
- turnover: 800.4238
- fees_paid: 8004.2652
- traded_notional: 8004265.1930
- breakeven_cost_bps: 1.1667
- gross_edge_per_trade_bps: 0.6411
- costs_paid: 10405.5448
- gross_pnl: 466.9207
- net_pnl: -9938.6240

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8172, long_above 0.5601, short_below 0.4255, median 0.4964. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.39% | -167.32 | +99.39% | 2005 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.50% .. -99.34%) | -99.42% | -184.01 | | |

The random null enters at the strategy's rate (0.0725 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 66% of its seeds on net return, 100% on net Sharpe and 99% on gross return.
