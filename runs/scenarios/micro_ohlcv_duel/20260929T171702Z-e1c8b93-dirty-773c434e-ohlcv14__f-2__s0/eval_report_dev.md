# Evaluation report - dev split - run `20260929T171702Z-e1c8b93-dirty-773c434e-ohlcv14__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.3831 | 0.7357 | 0.5507 |
| accuracy | 0.4918 | 0.4979 | 0.5010 |
| balanced accuracy | 0.4894 | 0.5028 | 0.5023 |
| precision (up) | 0.4759 | 0.4915 | 0.4899 |
| recall / sensitivity (up) | 0.3723 | 0.7386 | 0.5531 |
| specificity (down) | 0.6065 | 0.2671 | 0.4515 |
| F1 (up) | 0.4178 | 0.5902 | 0.5196 |
| MCC | -0.0218 | 0.0064 | 0.0045 |
| AUC | 0.4846 | 0.5069 | 0.5093 |
| Brier | 0.2628 | 0.2556 | 0.2561 |
| ECE (positive class) | 0.0869 | 0.0619 | 0.0626 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 4860 / 5353 / 8250 / 8193 | 10757 / 11130 / 4056 / 3808 | 8475 / 8825 / 7263 / 6849 |
| Gaussian readout: calls up | 0.7559 | 0.4991 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | -0.0030 | -0.0011 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5036 | 0.5063 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2500 | 0.2500 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0104 | 0.0101 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.7559 | 0.4991 | 0.5411 |
| Gaussian readout of the raw heads: MCC | -0.0030 | -0.0011 | -0.0096 |
| Gaussian readout of the raw heads: AUC | 0.5036 | 0.5063 | 0.4973 |
| Gaussian readout of the raw heads: Brier | 0.2719 | 0.2796 | 0.2696 |
| Gaussian readout of the raw heads: ECE | 0.1173 | 0.1412 | 0.1146 |

beta = 0 for h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.09 | 199.74 | 230.16 |
| RMSE ($), raw heads | 167.94 | 207.11 | 237.60 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.32 | 137.86 | 158.85 |
| MAE ($), raw heads | 115.73 | 145.90 | 166.73 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0000 | 0.0002 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0348 | -0.0749 | -0.0657 |
| EV, served | 0.0000 | 0.0002 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0230 | -0.0750 | -0.0635 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0080 | 0.0199 | 0.0143 |
| corr, Spearman, raw heads | -0.0036 | 0.0062 | -0.0072 |
| mean predicted ($), served | 0.02 | -0.00 | 0.00 |
| mean predicted ($), raw heads | 16.39 | -0.10 | 8.29 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.7443 | 0.4883 | 0.5348 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.0010 | 0.0198 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 83.84 | 101.39 | 117.35 |
| CRPSS vs constant variance | 0.0037 | 0.0147 | 0.0129 |
| NLL | 6.5963 | 6.7003 | 6.8028 |
| PIT KS | 0.0250 | 0.0230 | 0.0412 |
| var / err^2 Spearman | 0.1846 | 0.2243 | 0.2433 |
| coverage of the 90% interval | 0.9060 | 0.9039 | 0.8971 |
| width of the 90% interval ($) | 513.51 | 622.19 | 696.68 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0083 | [-0.0259, 0.0102] | NOISE |
| h1 | -0.0042 | [-0.0228, 0.0154] | NOISE |
| h2 | 0.0199 | [0.0001, 0.0385] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.001 / h1 0.020 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6889 | 0.9837 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.4870 | n/a (beta = 0: served delta is 0) | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.2608 | n/a (beta = 0: served delta is 0) | 0.3339 |

beta = 0 for h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5696 | 0.5556 | 0.5997 | 0.2317 |
| expected if the two signs were independent | 0.4428 | 0.4944 | 0.5028 | 0.1491 |

- P(up) unanimity (all three horizons call the same side): 0.3702

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0218 vs 0.0011 (-0.0229): does not beat, noise (boot z -1.08) | 0.0064 vs 0.0027 (+0.0037): beats, noise (boot z +0.19) | 0.0045 vs -0.0018 (+0.0064): beats, noise (boot z +0.31) |
| logreg_lags | direction/auc | 0.4846 vs 0.5085 (-0.0240): does not beat, noise (boot z -1.78) | 0.5069 vs 0.5137 (-0.0068): does not beat, noise (boot z -0.64) | 0.5093 vs 0.5174 (-0.0081): does not beat, noise (boot z -0.68) |
| logreg_lags | direction/brier | 0.2628 vs 0.2513 (-0.0115): does not beat, significantly worse (DM z -6.91) | 0.2556 vs 0.2517 (-0.0040): does not beat, significantly worse (DM z -4.27) | 0.2561 vs 0.2518 (-0.0043): does not beat, significantly worse (DM z -3.50) |
| logreg_lags | direction/ece_pos | 0.0869 vs 0.0347 (-0.0522): does not beat, significantly worse (boot z -4.91) | 0.0619 vs 0.0387 (-0.0232): does not beat, significantly worse (boot z -3.74) | 0.0626 vs 0.0432 (-0.0194): does not beat, noise (boot z -1.89) |
| logreg_lags | direction/acc | 0.4918 vs 0.4921 (-0.0003): does not beat, noise (DM z -0.02) | 0.4979 vs 0.4923 (+0.0056): beats, noise (DM z +0.76) | 0.5010 vs 0.4895 (+0.0116): beats, noise (DM z +1.12) |
| logreg_lags | direction/bal_acc | 0.4894 vs 0.5003 (-0.0109): does not beat, noise (boot z -1.33) | 0.5028 vs 0.5008 (+0.0020): beats, noise (boot z +0.30) | 0.5023 vs 0.4995 (+0.0028): beats, noise (boot z +0.36) |
| class_prior | direction/mcc | -0.0218 vs 0.0000 (-0.0218): does not beat, noise (boot z -1.77) | 0.0064 vs 0.0000 (+0.0064): beats, noise (boot z +0.55) | 0.0045 vs 0.0000 (+0.0045): beats, noise (boot z +0.36) |
| class_prior | direction/auc | 0.4846 vs 0.5000 (-0.0154): does not beat, noise (boot z -1.95) | 0.5069 vs 0.5000 (+0.0069): beats, noise (boot z +0.91) | 0.5093 vs 0.5000 (+0.0093): beats, noise (boot z +1.13) |
| class_prior | direction/brier | 0.2628 vs 0.2508 (-0.0121): does not beat, significantly worse (DM z -7.83) | 0.2556 vs 0.2510 (-0.0046): does not beat, significantly worse (DM z -4.98) | 0.2561 vs 0.2512 (-0.0049): does not beat, significantly worse (DM z -3.88) |
| class_prior | direction/ece_pos | 0.0869 vs 0.0297 (-0.0572): does not beat, significantly worse (boot z -5.24) | 0.0619 vs 0.0335 (-0.0284): does not beat, significantly worse (boot z -4.53) | 0.0626 vs 0.0371 (-0.0256): does not beat, significantly worse (boot z -2.45) |
| class_prior | direction/acc | 0.4918 vs 0.4897 (+0.0021): beats, noise (DM z +0.19) | 0.4979 vs 0.4896 (+0.0083): beats, noise (DM z +1.15) | 0.5010 vs 0.4878 (+0.0132): beats, noise (DM z +1.21) |
| class_prior | direction/bal_acc | 0.4894 vs 0.5000 (-0.0106): does not beat, noise (boot z -1.77) | 0.5028 vs 0.5000 (+0.0028): beats, noise (boot z +0.55) | 0.5023 vs 0.5000 (+0.0023): beats, noise (boot z +0.36) |
| zero_delta | delta/rmse | 165.09 vs 165.09 (+0.00, +0.00%): beats, noise (DM z +0.12) | 199.74 vs 199.76 (+0.02, +0.01%): beats, noise (DM z +1.08) | 230.16 vs 230.16 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 112.32 vs 112.32 (-0.00, -0.00%): does not beat, noise (DM z -1.04) | 137.86 vs 137.87 (+0.00, +0.00%): beats, noise (DM z +0.33) | 158.85 vs 158.85 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 165.09 vs 165.39 (+0.30, +0.18%): beats (DM z +2.58) | 199.74 vs 200.31 (+0.57, +0.28%): beats (DM z +2.69) | 230.16 vs 231.00 (+0.84, +0.36%): beats (DM z +2.55) |
| mean_delta | delta/mae | 112.32 vs 112.73 (+0.41, +0.36%): beats (DM z +4.09) | 137.86 vs 138.56 (+0.70, +0.50%): beats (DM z +3.79) | 158.85 vs 159.88 (+1.03, +0.64%): beats (DM z +3.69) |
| const_var | variance/crps | 83.84 vs 84.15 (+0.31, +0.37%): beats (DM z +1.99) | 101.39 vs 102.91 (+1.51, +1.47%): beats (DM z +7.76) | 117.35 vs 118.88 (+1.53, +1.29%): beats (DM z +4.45) |
| const_var | variance/nll | 6.5963 vs 6.5630 (-0.0333): does not beat, noise (DM z -1.67) | 6.7003 vs 6.7470 (+0.0467): beats (DM z +3.11) | 6.8028 vs 6.8845 (+0.0817): beats (DM z +3.93) |
| const_var | variance/pit_ks | 0.0250 vs 0.0612 (+0.0362): beats (boot z +9.75) | 0.0230 vs 0.0652 (+0.0422): beats (boot z +10.95) | 0.0412 vs 0.0725 (+0.0313): beats (boot z +6.07) |
| const_var | variance/corr_var_err2_spearman | 0.1846 vs 0.0000 (+0.1846): beats (boot z +12.76) | 0.2243 vs 0.0000 (+0.2243): beats (boot z +13.86) | 0.2433 vs 0.0000 (+0.2433): beats (boot z +14.68) |

## Backtest (costs included)

- n_trades: 2048
- total_return: -0.9953
- sharpe_net: -174.9112
- sharpe_gross: -8.2735
- sortino: -188.4142
- max_drawdown: 0.9953
- hit_rate: 0.0347
- hit_rate_gross: 0.4937
- profit_factor: 0.0175
- avg_hold_bars: 8.2822
- exposure: 0.3927
- turnover: 727.7081
- fees_paid: 7277.2391
- traded_notional: 7277239.0663
- breakeven_cost_bps: -1.3526
- gross_edge_per_trade_bps: -0.0819
- costs_paid: 9460.4108
- gross_pnl: -492.1495
- net_pnl: -9952.5603

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.84, long_above 0.5724, short_below 0.4370, median 0.5036. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -99.53% | -174.91 | +99.53% | 2048 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -99.61% .. -99.48%) | -99.54% | -188.65 | | |

The random null enters at the strategy's rate (0.0781 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 63% of its seeds on net return, 100% on net Sharpe and 0% on gross return.
