# Evaluation report - dev split - run `20260930T232532Z-fb840fd-f2ddc1ec-control__f-36__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 16483 | 17728 | 18448 |
| n_eff of the scored moves (n scored // bars ahead) | 1648 | 1181 | 922 |
| true up-rate | 0.5096 | 0.5110 | 0.5149 |
| calls up (predicted up-rate) | 0.3613 | 0.5674 | 0.3911 |
| accuracy | 0.5109 | 0.5173 | 0.5130 |
| balanced accuracy | 0.5135 | 0.5158 | 0.5162 |
| precision (up) | 0.5283 | 0.5250 | 0.5356 |
| recall / sensitivity (up) | 0.3746 | 0.5828 | 0.4068 |
| specificity (down) | 0.6525 | 0.4488 | 0.6256 |
| F1 (up) | 0.4383 | 0.5524 | 0.4624 |
| MCC | 0.0282 | 0.0320 | 0.0332 |
| AUC | 0.5197 | 0.5261 | 0.5265 |
| Brier | 0.2531 | 0.2495 | 0.2517 |
| ECE (positive class) | 0.0421 | 0.0077 | 0.0323 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0096 | 0.0110 | 0.0149 |
| TP / FP / TN / FN | 3146 / 2809 / 5275 / 5253 | 5280 / 4778 / 3891 / 3779 | 3864 / 3351 / 5599 / 5634 |
| Gaussian readout: calls up | 0.6063 | 0.4664 | 0.3748 |
| Gaussian readout: MCC | 0.0183 | 0.0511 | 0.0472 |
| Gaussian readout: AUC | 0.5116 | 0.5306 | 0.5326 |
| Gaussian readout: Brier | 0.2499 | 0.2495 | 0.2496 |
| Gaussian readout: ECE | 0.0046 | 0.0126 | 0.0171 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 76.11 | 92.64 | 106.01 |
| RMSE ($), raw heads | 76.31 | 94.12 | 107.22 |
| RMSE ($), zero prediction | 76.07 | 92.59 | 105.96 |
| MAE ($), served | 52.50 | 63.98 | 73.26 |
| MAE ($), raw heads | 52.59 | 64.72 | 73.72 |
| MAE ($), zero prediction | 52.53 | 64.07 | 73.33 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0009 | -0.0011 | -0.0009 |
| skill vs zero, raw heads | -0.0062 | -0.0335 | -0.0238 |
| EV, served | -0.0015 | -0.0012 | -0.0005 |
| EV, raw heads | -0.0073 | -0.0339 | -0.0213 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0082 | 0.0087 | 0.0036 |
| corr, Spearman, raw heads | 0.0190 | 0.0384 | 0.0400 |
| mean predicted ($), served | 0.66 | 0.09 | -0.38 |
| mean predicted ($), raw heads | 1.64 | 0.41 | -2.17 |
| mean realised ($) | 2.79 | 4.20 | 5.63 |
| share predicted up, raw heads | 0.6290 | 0.4799 | 0.3803 |
| share realised up | 0.5013 | 0.5012 | 0.5102 |
| shrink beta (served = beta x raw, fit on cal) | 0.4037 | 0.2283 | 0.1753 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 38.61 | 47.42 | 54.22 |
| CRPSS vs constant variance | 0.0228 | 0.0157 | 0.0180 |
| NLL | 5.6305 | 5.8473 | 5.9790 |
| PIT KS | 0.0588 | 0.0748 | 0.0719 |
| var / err^2 Spearman | 0.3101 | 0.2995 | 0.2953 |
| coverage of the 90% interval | 0.8866 | 0.8866 | 0.8806 |
| width of the 90% interval ($) | 218.79 | 270.06 | 305.81 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0157 | [-0.0085, 0.0387] | NOISE |
| h1 | 0.0276 | [0.0044, 0.0491] | WORKS |
| h2 | 0.0233 | [0.0008, 0.0455] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.404 / h1 0.228 / h2 0.175) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7614 | 0.6079 | 0.6105 |
| abs(d h1) <= abs(d h2) | 0.5015 | 0.3305 | 0.5875 |
| full chain h0 <= h1 <= h2 | 0.3406 | 0.1403 | 0.3272 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5206 | 0.6522 | 0.6834 | 0.2501 |
| expected if the two signs were independent | 0.4588 | 0.4968 | 0.5274 | 0.1435 |

- P(up) unanimity (all three horizons call the same side): 0.4005

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0282 vs 0.0600 (-0.0318): does not beat, noise (boot z -1.44) | 0.0320 vs 0.0619 (-0.0299): does not beat, noise (boot z -1.61) | 0.0332 vs 0.0325 (+0.0007): beats, noise (boot z +0.03) |
| logreg_lags | direction/auc | 0.5197 vs 0.5395 (-0.0197): does not beat, noise (boot z -1.53) | 0.5261 vs 0.5444 (-0.0183): does not beat, noise (boot z -1.68) | 0.5265 vs 0.5410 (-0.0145): does not beat, noise (boot z -1.06) |
| logreg_lags | direction/brier | 0.2531 vs 0.2493 (-0.0037): does not beat, significantly worse (DM z -2.75) | 0.2495 vs 0.2493 (-0.0002): does not beat, noise (DM z -0.29) | 0.2517 vs 0.2493 (-0.0024): does not beat, significantly worse (DM z -2.14) |
| logreg_lags | direction/ece_pos | 0.0421 vs 0.0118 (-0.0303): does not beat, significantly worse (boot z -2.83) | 0.0077 vs 0.0108 (+0.0031): beats, noise (boot z +0.40) | 0.0323 vs 0.0073 (-0.0249): does not beat, significantly worse (boot z -2.49) |
| logreg_lags | direction/acc | 0.5109 vs 0.5315 (-0.0206): does not beat, noise (DM z -1.75) | 0.5173 vs 0.5327 (-0.0154): does not beat, noise (DM z -1.78) | 0.5130 vs 0.5198 (-0.0069): does not beat, noise (DM z -0.52) |
| logreg_lags | direction/bal_acc | 0.5135 vs 0.5289 (-0.0153): does not beat, noise (boot z -1.44) | 0.5158 vs 0.5299 (-0.0140): does not beat, noise (boot z -1.57) | 0.5162 vs 0.5156 (+0.0006): beats, noise (boot z +0.06) |
| class_prior | direction/mcc | 0.0282 vs 0.0000 (+0.0282): beats, noise (boot z +1.80) | 0.0320 vs 0.0000 (+0.0320): beats (boot z +2.23) | 0.0332 vs 0.0000 (+0.0332): beats (boot z +2.06) |
| class_prior | direction/auc | 0.5197 vs 0.5000 (+0.0197): beats, noise (boot z +1.92) | 0.5261 vs 0.5000 (+0.0261): beats (boot z +2.75) | 0.5265 vs 0.5000 (+0.0265): beats (boot z +2.60) |
| class_prior | direction/brier | 0.2531 vs 0.2499 (-0.0032): does not beat, significantly worse (DM z -2.33) | 0.2495 vs 0.2499 (+0.0003): beats, noise (DM z +0.60) | 0.2517 vs 0.2498 (-0.0019): does not beat, noise (DM z -1.49) |
| class_prior | direction/ece_pos | 0.0421 vs 0.0001 (-0.0420): does not beat, significantly worse (boot z -4.34) | 0.0077 vs 0.0002 (-0.0075): does not beat, noise (boot z -1.31) | 0.0323 vs 0.0038 (-0.0285): does not beat, significantly worse (boot z -3.13) |
| class_prior | direction/acc | 0.5109 vs 0.5096 (+0.0013): beats, noise (DM z +0.09) | 0.5173 vs 0.5110 (+0.0063): beats, noise (DM z +0.51) | 0.5130 vs 0.5149 (-0.0019): does not beat, noise (DM z -0.10) |
| class_prior | direction/bal_acc | 0.5135 vs 0.5000 (+0.0135): beats, noise (boot z +1.80) | 0.5158 vs 0.5000 (+0.0158): beats (boot z +2.23) | 0.5162 vs 0.5000 (+0.0162): beats (boot z +2.06) |
| zero_delta | delta/rmse | 76.11 vs 76.07 (-0.04, -0.05%): does not beat, noise (DM z -0.64) | 92.64 vs 92.59 (-0.05, -0.05%): does not beat, noise (DM z -0.46) | 106.01 vs 105.96 (-0.05, -0.04%): does not beat, noise (DM z -0.59) |
| zero_delta | delta/mae | 52.50 vs 52.53 (+0.03, +0.05%): beats, noise (DM z +0.77) | 63.98 vs 64.07 (+0.09, +0.13%): beats, noise (DM z +1.35) | 73.26 vs 73.33 (+0.07, +0.09%): beats, noise (DM z +1.40) |
| mean_delta | delta/rmse | 76.11 vs 76.05 (-0.06, -0.08%): does not beat, noise (DM z -1.16) | 92.64 vs 92.53 (-0.10, -0.11%): does not beat, noise (DM z -0.89) | 106.01 vs 105.88 (-0.13, -0.12%): does not beat, noise (DM z -1.19) |
| mean_delta | delta/mae | 52.50 vs 52.53 (+0.02, +0.05%): beats, noise (DM z +0.72) | 63.98 vs 64.07 (+0.09, +0.13%): beats, noise (DM z +1.24) | 73.26 vs 73.30 (+0.04, +0.05%): beats, noise (DM z +0.57) |
| const_var | variance/crps | 38.61 vs 39.51 (+0.90, +2.28%): beats (DM z +8.20) | 47.42 vs 48.18 (+0.76, +1.57%): beats (DM z +5.26) | 54.22 vs 55.21 (+0.99, +1.80%): beats (DM z +5.57) |
| const_var | variance/nll | 5.6305 vs 5.7555 (+0.1250): beats (DM z +6.57) | 5.8473 vs 5.9518 (+0.1045): beats (DM z +5.18) | 5.9790 vs 6.0862 (+0.1072): beats (DM z +4.83) |
| const_var | variance/pit_ks | 0.0588 vs 0.0612 (+0.0023): beats, noise (boot z +0.61) | 0.0748 vs 0.0602 (-0.0147): does not beat, significantly worse (boot z -3.13) | 0.0719 vs 0.0622 (-0.0097): does not beat, significantly worse (boot z -2.03) |
| const_var | variance/corr_var_err2_spearman | 0.3101 vs 0.0000 (+0.3101): beats (boot z +15.77) | 0.2995 vs 0.0000 (+0.2995): beats (boot z +13.41) | 0.2953 vs 0.0000 (+0.2953): beats (boot z +12.43) |

## Backtest (costs included)

- n_trades: 911
- total_return: 0.0060
- sharpe_net: 0.6154
- sharpe_gross: 0.6154
- sortino: 0.8516
- max_drawdown: 0.0707
- hit_rate: 0.5521
- hit_rate_gross: 0.5521
- profit_factor: 1.0095
- avg_hold_bars: 9.6103
- exposure: 0.3590
- turnover: 1881.8741
- fees_paid: 0.0000
- traded_notional: 18819561.6081
- breakeven_cost_bps: 0.0638
- gross_edge_per_trade_bps: 0.0848
- costs_paid: 0.0000
- gross_pnl: 60.0507
- net_pnl: 60.0507

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -36 (TimeSeriesSplit fold 5, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-01-28T10:18:00 .. 2024-02-14T08:47:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.02, long_above 0.5558, short_below 0.4466, median 0.4884. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +0.60% | +0.62 | +7.07% | 911 |
| buy and hold | +15.72% | +8.30 | +4.49% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -7.69% .. +8.47%) | -0.07% | -0.04 | | |

The random null enters at the strategy's rate (0.0583 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 55% of its seeds on net return, 55% on net Sharpe and 55% on gross return.
