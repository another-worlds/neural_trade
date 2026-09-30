# Evaluation report - dev split - run `20260930T033844Z-ce1e2ed-26a4b7e5-control__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 36292 | 37806 | 39444 |
| n_eff of the scored moves (n scored // bars ahead) | 604 | 315 | 164 |
| true up-rate | 0.4824 | 0.4829 | 0.4763 |
| calls up (predicted up-rate) | 0.6109 | 0.9085 | 0.7510 |
| accuracy | 0.4893 | 0.4790 | 0.4935 |
| balanced accuracy | 0.4932 | 0.4930 | 0.5054 |
| precision (up) | 0.4769 | 0.4791 | 0.4799 |
| recall / sensitivity (up) | 0.6039 | 0.9012 | 0.7567 |
| specificity (down) | 0.3826 | 0.0847 | 0.2542 |
| F1 (up) | 0.5329 | 0.6256 | 0.5873 |
| MCC | -0.0139 | -0.0244 | 0.0125 |
| AUC | 0.4932 | 0.4938 | 0.5076 |
| Brier | 0.2574 | 0.2535 | 0.2566 |
| ECE (positive class) | 0.0673 | 0.0584 | 0.0652 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0176 | 0.0171 | 0.0237 |
| TP / FP / TN / FN | 10573 / 11598 / 7186 / 6935 | 16454 / 17892 / 1656 / 1804 | 14216 / 15406 / 5250 / 4572 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.9901 | 0.9505 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0061 | -0.0022 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4921 | 0.4974 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2506 | 0.2520 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0274 | 0.0463 |
| Gaussian readout of the raw heads: calls up | 0.9508 | 0.9901 | 0.9505 |
| Gaussian readout of the raw heads: MCC | -0.0304 | -0.0061 | -0.0022 |
| Gaussian readout of the raw heads: AUC | 0.4618 | 0.4921 | 0.4974 |
| Gaussian readout of the raw heads: Brier | 0.2793 | 0.2639 | 0.2744 |
| Gaussian readout of the raw heads: ECE | 0.1431 | 0.1071 | 0.1326 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| RMSE ($), served | 400.31 | 564.79 | 810.83 |
| RMSE ($), raw heads | 420.98 | 583.99 | 845.90 |
| RMSE ($), zero prediction | 400.31 | 563.88 | 807.63 |
| MAE ($), served | 271.46 | 378.57 | 553.90 |
| MAE ($), raw heads | 293.03 | 397.85 | 590.16 |
| MAE ($), zero prediction | 271.46 | 377.88 | 550.98 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0032 | -0.0079 |
| skill vs zero, raw heads | -0.1059 | -0.0726 | -0.0970 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0011 | -0.0018 |
| EV, raw heads | -0.0437 | -0.0208 | -0.0321 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0714 | -0.0397 | -0.0105 |
| corr, Spearman, raw heads | -0.0724 | -0.0276 | -0.0132 |
| mean predicted ($), served | 0.00 | 11.91 | 33.44 |
| mean predicted ($), raw heads | 89.54 | 108.12 | 167.57 |
| mean realised ($) | -10.96 | -22.09 | -42.85 |
| share predicted up, raw heads | 0.9529 | 0.9898 | 0.9507 |
| share realised up | 0.4823 | 0.4850 | 0.4782 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1102 | 0.1995 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|
| CRPS ($) | 202.08 | 286.14 | 419.45 |
| CRPSS vs constant variance | 0.0217 | 0.0290 | 0.0570 |
| NLL | 7.4328 | 7.7697 | 8.1985 |
| PIT KS | 0.0384 | 0.0722 | 0.0623 |
| var / err^2 Spearman | 0.1201 | 0.0416 | 0.0780 |
| coverage of the 90% interval | 0.9028 | 0.9007 | 0.8637 |
| width of the 90% interval ($) | 1212.02 | 1749.83 | 2412.31 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0113 | [-0.0382, 0.0169] | NOISE |
| h1 | -0.0045 | [-0.0298, 0.0199] | NOISE |
| h2 | -0.0097 | [-0.0414, 0.0251] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.110 / h2 0.200) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5751 | n/a (beta = 0: served delta is 0) | 0.6247 |
| abs(d h1) <= abs(d h2) | 0.6849 | 0.8366 | 0.6431 |
| full chain h0 <= h1 <= h2 | 0.4757 | n/a (beta = 0: served delta is 0) | 0.3608 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5852 | 0.9122 | 0.7358 | 0.4334 |
| expected if the two signs were independent | 0.5883 | 0.9037 | 0.7243 | 0.4233 |

- P(up) unanimity (all three horizons call the same side): 0.4610

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (60 bars) | h1 (120 bars) | h2 (240 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0139 vs 0.0025 (-0.0164): does not beat, noise (boot z -0.51) | -0.0244 vs 0.0205 (-0.0449): does not beat, noise (boot z -1.94) | 0.0125 vs -0.0138 (+0.0263): beats, noise (boot z +0.87) |
| logreg_lags | direction/auc | 0.4932 vs 0.5320 (-0.0388): does not beat, noise (boot z -1.66) | 0.4938 vs 0.5251 (-0.0313): does not beat, noise (boot z -1.63) | 0.5076 vs 0.5095 (-0.0018): does not beat, noise (boot z -0.17) |
| logreg_lags | direction/brier | 0.2574 vs 0.2536 (-0.0038): does not beat, noise (DM z -1.60) | 0.2535 vs 0.2575 (+0.0040): beats, noise (DM z +1.66) | 0.2566 vs 0.2702 (+0.0136): beats (DM z +3.12) |
| logreg_lags | direction/ece_pos | 0.0673 vs 0.0663 (-0.0010): does not beat, noise (boot z -0.07) | 0.0584 vs 0.0812 (+0.0228): beats (boot z +4.43) | 0.0652 vs 0.1230 (+0.0578): beats (boot z +8.41) |
| logreg_lags | direction/acc | 0.4893 vs 0.4836 (+0.0057): beats, noise (DM z +0.34) | 0.4790 vs 0.4911 (-0.0121): does not beat, noise (DM z -1.81) | 0.4935 vs 0.4756 (+0.0179): beats, noise (DM z +0.88) |
| logreg_lags | direction/bal_acc | 0.4932 vs 0.5004 (-0.0072): does not beat, noise (boot z -0.63) | 0.4930 vs 0.5055 (-0.0126): does not beat, noise (boot z -1.95) | 0.5054 vs 0.4971 (+0.0083): beats, noise (boot z +0.77) |
| class_prior | direction/mcc | -0.0139 vs 0.0000 (-0.0139): does not beat, noise (boot z -0.66) | -0.0244 vs 0.0000 (-0.0244): does not beat, noise (boot z -1.61) | 0.0125 vs 0.0000 (+0.0125): beats, noise (boot z +0.54) |
| class_prior | direction/auc | 0.4932 vs 0.5000 (-0.0068): does not beat, noise (boot z -0.49) | 0.4938 vs 0.5000 (-0.0062): does not beat, noise (boot z -0.67) | 0.5076 vs 0.5000 (+0.0076): beats, noise (boot z +0.48) |
| class_prior | direction/brier | 0.2574 vs 0.2532 (-0.0042): does not beat, significantly worse (DM z -2.16) | 0.2535 vs 0.2533 (-0.0001): does not beat, noise (DM z -0.15) | 0.2566 vs 0.2573 (+0.0007): beats, noise (DM z +0.25) |
| class_prior | direction/ece_pos | 0.0673 vs 0.0593 (-0.0079): does not beat, noise (boot z -0.57) | 0.0584 vs 0.0603 (+0.0019): beats, noise (boot z +0.38) | 0.0652 vs 0.0886 (+0.0234): beats (boot z +3.37) |
| class_prior | direction/acc | 0.4893 vs 0.4824 (+0.0069): beats, noise (DM z +0.42) | 0.4790 vs 0.4829 (-0.0039): does not beat, noise (DM z -0.61) | 0.4935 vs 0.4763 (+0.0172): beats, noise (DM z +0.79) |
| class_prior | direction/bal_acc | 0.4932 vs 0.5000 (-0.0068): does not beat, noise (boot z -0.66) | 0.4930 vs 0.5000 (-0.0070): does not beat, noise (boot z -1.60) | 0.5054 vs 0.5000 (+0.0054): beats, noise (boot z +0.54) |
| zero_delta | delta/rmse | 400.31 vs 400.31 (+0.00, +0.00%): does not beat | 564.79 vs 563.88 (-0.90, -0.16%): does not beat, noise (DM z -1.54) | 810.83 vs 807.63 (-3.20, -0.40%): does not beat, noise (DM z -1.25) |
| zero_delta | delta/mae | 271.46 vs 271.46 (+0.00, +0.00%): does not beat | 378.57 vs 377.88 (-0.69, -0.18%): does not beat, noise (DM z -1.44) | 553.90 vs 550.98 (-2.92, -0.53%): does not beat, noise (DM z -1.47) |
| mean_delta | delta/rmse | 400.31 vs 405.60 (+5.29, +1.31%): beats (DM z +2.85) | 564.79 vs 578.56 (+13.78, +2.38%): beats (DM z +2.92) | 810.83 vs 847.03 (+36.21, +4.27%): beats (DM z +2.96) |
| mean_delta | delta/mae | 271.46 vs 277.50 (+6.04, +2.18%): beats (DM z +3.94) | 378.57 vs 393.77 (+15.20, +3.86%): beats (DM z +3.98) | 553.90 vs 595.01 (+41.11, +6.91%): beats (DM z +4.01) |
| const_var | variance/crps | 202.08 vs 206.55 (+4.47, +2.17%): beats (DM z +4.72) | 286.14 vs 294.69 (+8.55, +2.90%): beats (DM z +3.47) | 419.45 vs 444.83 (+25.38, +5.70%): beats (DM z +3.84) |
| const_var | variance/nll | 7.4328 vs 7.4448 (+0.0121): beats, noise (DM z +1.12) | 7.7697 vs 7.8022 (+0.0325): beats (DM z +2.74) | 8.1985 vs 8.2104 (+0.0120): beats, noise (DM z +0.35) |
| const_var | variance/pit_ks | 0.0384 vs 0.1064 (+0.0680): beats (boot z +9.53) | 0.0722 vs 0.1404 (+0.0682): beats (boot z +11.75) | 0.0623 vs 0.1810 (+0.1188): beats (boot z +26.26) |
| const_var | variance/corr_var_err2_spearman | 0.1201 vs 0.0000 (+0.1201): beats (boot z +6.98) | 0.0416 vs 0.0000 (+0.0416): beats, noise (boot z +1.78) | 0.0780 vs 0.0000 (+0.0780): beats (boot z +4.08) |

## Backtest (costs included)

- n_trades: 1567
- total_return: -0.9849
- sharpe_net: -145.5703
- sharpe_gross: -5.0260
- sortino: -161.0264
- max_drawdown: 0.9849
- hit_rate: 0.0593
- hit_rate_gross: 0.4371
- profit_factor: 0.0303
- avg_hold_bars: 11.1532
- exposure: 0.4046
- turnover: 731.2466
- fees_paid: 7312.6151
- traded_notional: 7312615.0846
- breakeven_cost_bps: -0.9365
- gross_edge_per_trade_bps: -0.6987
- costs_paid: 9506.3996
- gross_pnl: -342.4055
- net_pnl: -9848.8052

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T19:58:00 .. 2025-08-30T19:57:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.8804, long_above 0.5805, short_below 0.4832, median 0.5277. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.49% | -145.57 | +98.49% | 1567 |
| buy and hold | -7.06% | -2.46 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.56% .. -98.09%) | -98.33% | -152.99 | | |

The random null enters at the strategy's rate (0.0609 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 14% of its seeds on net return, 98% on net Sharpe and 7% on gross return.
