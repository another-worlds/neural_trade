# Evaluation report - dev split - run `20260929T124929Z-9446405-3d56c738-h_4h__f-2__s0`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 38831 | 39457 | 40128 |
| n_eff of the scored moves (n scored // bars ahead) | 242 | 164 | 125 |
| true up-rate | 0.4792 | 0.4762 | 0.4740 |
| calls up (predicted up-rate) | 0.7128 | 0.9704 | 0.9306 |
| accuracy | 0.4986 | 0.4748 | 0.4735 |
| balanced accuracy | 0.5075 | 0.4972 | 0.4959 |
| precision (up) | 0.4845 | 0.4747 | 0.4718 |
| recall / sensitivity (up) | 0.7206 | 0.9675 | 0.9263 |
| specificity (down) | 0.2943 | 0.0270 | 0.0655 |
| F1 (up) | 0.5794 | 0.6369 | 0.6251 |
| MCC | 0.0165 | -0.0163 | -0.0161 |
| AUC | 0.5171 | 0.4906 | 0.5031 |
| Brier | 0.2547 | 0.2557 | 0.2598 |
| ECE (positive class) | 0.0593 | 0.0744 | 0.0935 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0208 | 0.0238 | 0.0260 |
| TP / FP / TN / FN | 13410 / 14270 / 5952 / 5199 | 18177 / 20111 / 558 / 611 | 17617 / 19726 / 1383 / 1402 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.9397 | 0.8246 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | -0.0095 | -0.0330 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.4923 | 0.4837 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2512 | 0.2519 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0390 | 0.0485 |
| Gaussian readout of the raw heads: calls up | 1.0000 | 0.9397 | 0.8246 |
| Gaussian readout of the raw heads: MCC | 0.0000 | -0.0095 | -0.0330 |
| Gaussian readout of the raw heads: AUC | 0.4775 | 0.4923 | 0.4837 |
| Gaussian readout of the raw heads: Brier | 0.2767 | 0.2742 | 0.2800 |
| Gaussian readout of the raw heads: ECE | 0.1543 | 0.1321 | 0.1474 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| RMSE ($), served | 652.16 | 812.25 | 953.93 |
| RMSE ($), raw heads | 686.24 | 861.39 | 1022.10 |
| RMSE ($), zero prediction | 652.16 | 809.64 | 949.34 |
| MAE ($), served | 441.66 | 555.38 | 667.49 |
| MAE ($), raw heads | 477.54 | 606.73 | 732.85 |
| MAE ($), zero prediction | 441.66 | 553.16 | 663.87 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | -0.0065 | -0.0097 |
| skill vs zero, raw heads | -0.1073 | -0.1319 | -0.1592 |
| EV, served | n/a (beta = 0: served delta is 0) | -0.0018 | -0.0039 |
| EV, raw heads | -0.0128 | -0.0477 | -0.0743 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0368 | -0.0184 | -0.0350 |
| corr, Spearman, raw heads | -0.0407 | -0.0246 | -0.0374 |
| mean predicted ($), served | 0.00 | 26.48 | 34.41 |
| mean predicted ($), raw heads | 172.29 | 194.45 | 224.42 |
| mean realised ($) | -30.45 | -45.03 | -58.66 |
| share predicted up, raw heads | 1.0000 | 0.9402 | 0.8244 |
| share realised up | 0.4790 | 0.4781 | 0.4760 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1362 | 0.1533 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|
| CRPS ($) | 332.25 | 421.32 | 502.12 |
| CRPSS vs constant variance | 0.0401 | 0.0525 | 0.0674 |
| NLL | 7.9330 | 8.1516 | 8.3211 |
| PIT KS | 0.0517 | 0.0796 | 0.0685 |
| var / err^2 Spearman | 0.0884 | 0.0498 | 0.0844 |
| coverage of the 90% interval | 0.8907 | 0.8646 | 0.8496 |
| width of the 90% interval ($) | 1977.60 | 2432.40 | 2829.22 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0118 | [-0.0229, 0.0485] | NOISE |
| h1 | -0.0091 | [-0.0387, 0.0197] | NOISE |
| h2 | 0.0124 | [-0.0334, 0.0550] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.136 / h2 0.153) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.4903 | n/a (beta = 0: served delta is 0) | 0.6030 |
| abs(d h1) <= abs(d h2) | 0.7569 | 0.8039 | 0.6010 |
| full chain h0 <= h1 <= h2 | 0.4501 | n/a (beta = 0: served delta is 0) | 0.3235 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7069 | 0.9135 | 0.8119 | 0.6062 |
| expected if the two signs were independent | 0.7069 | 0.9148 | 0.7792 | 0.5527 |

- P(up) unanimity (all three horizons call the same side): 0.6709

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (160 bars) | h1 (240 bars) | h2 (320 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0165 vs 0.0132 (+0.0033): beats, noise (boot z +0.09) | -0.0163 vs -0.0218 (+0.0055): beats, noise (boot z +0.20) | -0.0161 vs 0.0136 (-0.0297): does not beat, noise (boot z -1.15) |
| logreg_lags | direction/auc | 0.5171 vs 0.5230 (-0.0059): does not beat, noise (boot z -0.29) | 0.4906 vs 0.5053 (-0.0147): does not beat, noise (boot z -0.59) | 0.5031 vs 0.5191 (-0.0160): does not beat, significantly worse (boot z -2.11) |
| logreg_lags | direction/brier | 0.2547 vs 0.2584 (+0.0037): beats, noise (DM z +1.19) | 0.2557 vs 0.2683 (+0.0126): beats (DM z +2.83) | 0.2598 vs 0.2773 (+0.0175): beats (DM z +3.31) |
| logreg_lags | direction/ece_pos | 0.0593 vs 0.0842 (+0.0249): beats (boot z +2.87) | 0.0744 vs 0.1158 (+0.0414): beats (boot z +7.29) | 0.0935 vs 0.1534 (+0.0599): beats (boot z +12.17) |
| logreg_lags | direction/acc | 0.4986 vs 0.4870 (+0.0116): beats, noise (DM z +0.57) | 0.4748 vs 0.4739 (+0.0009): beats, noise (DM z +0.13) | 0.4735 vs 0.4746 (-0.0011): does not beat, noise (DM z -0.12) |
| logreg_lags | direction/bal_acc | 0.5075 vs 0.5039 (+0.0036): beats, noise (boot z +0.27) | 0.4972 vs 0.4945 (+0.0028): beats, noise (boot z +0.42) | 0.4959 vs 0.5006 (-0.0047): does not beat, noise (boot z -0.76) |
| class_prior | direction/mcc | 0.0165 vs 0.0000 (+0.0165): beats, noise (boot z +0.73) | -0.0163 vs 0.0000 (-0.0163): does not beat, noise (boot z -1.20) | -0.0161 vs 0.0000 (-0.0161): does not beat, noise (boot z -0.66) |
| class_prior | direction/auc | 0.5171 vs 0.5000 (+0.0171): beats, noise (boot z +1.08) | 0.4906 vs 0.5000 (-0.0094): does not beat, noise (boot z -0.92) | 0.5031 vs 0.5000 (+0.0031): beats, noise (boot z +0.18) |
| class_prior | direction/brier | 0.2547 vs 0.2534 (-0.0014): does not beat, noise (DM z -0.62) | 0.2557 vs 0.2559 (+0.0002): beats, noise (DM z +0.33) | 0.2598 vs 0.2626 (+0.0028): beats, noise (DM z +1.00) |
| class_prior | direction/ece_pos | 0.0593 vs 0.0618 (+0.0025): beats, noise (boot z +0.30) | 0.0744 vs 0.0806 (+0.0063): beats (boot z +2.82) | 0.0935 vs 0.1152 (+0.0217): beats (boot z +4.48) |
| class_prior | direction/acc | 0.4986 vs 0.4792 (+0.0194): beats, noise (DM z +0.95) | 0.4748 vs 0.4762 (-0.0013): does not beat, noise (DM z -0.43) | 0.4735 vs 0.4740 (-0.0005): does not beat, noise (DM z -0.05) |
| class_prior | direction/bal_acc | 0.5075 vs 0.5000 (+0.0075): beats, noise (boot z +0.73) | 0.4972 vs 0.5000 (-0.0028): does not beat, noise (boot z -1.18) | 0.4959 vs 0.5000 (-0.0041): does not beat, noise (boot z -0.66) |
| zero_delta | delta/rmse | 652.16 vs 652.16 (+0.00, +0.00%): does not beat | 812.25 vs 809.64 (-2.61, -0.32%): does not beat, noise (DM z -1.30) | 953.93 vs 949.34 (-4.60, -0.48%): does not beat, noise (DM z -1.56) |
| zero_delta | delta/mae | 441.66 vs 441.66 (+0.00, +0.00%): does not beat | 555.38 vs 553.16 (-2.22, -0.40%): does not beat, noise (DM z -1.40) | 667.49 vs 663.87 (-3.61, -0.54%): does not beat, noise (DM z -1.57) |
| mean_delta | delta/rmse | 652.16 vs 672.36 (+20.20, +3.00%): beats (DM z +2.63) | 812.25 vs 846.83 (+34.58, +4.08%): beats (DM z +2.83) | 953.93 vs 1006.31 (+52.38, +5.21%): beats (DM z +2.81) |
| mean_delta | delta/mae | 441.66 vs 462.93 (+21.27, +4.60%): beats (DM z +3.38) | 555.38 vs 594.10 (+38.72, +6.52%): beats (DM z +3.81) | 667.49 vs 725.63 (+58.14, +8.01%): beats (DM z +3.82) |
| const_var | variance/crps | 332.25 vs 346.13 (+13.87, +4.01%): beats (DM z +3.42) | 421.32 vs 444.65 (+23.33, +5.25%): beats (DM z +3.46) | 502.12 vs 538.40 (+36.28, +6.74%): beats (DM z +3.53) |
| const_var | variance/nll | 7.9330 vs 7.9640 (+0.0309): beats, noise (DM z +1.55) | 8.1516 vs 8.2102 (+0.0587): beats (DM z +2.36) | 8.3211 vs 8.3837 (+0.0626): beats (DM z +2.06) |
| const_var | variance/pit_ks | 0.0517 vs 0.1451 (+0.0934): beats (boot z +12.69) | 0.0796 vs 0.1760 (+0.0964): beats (boot z +25.91) | 0.0685 vs 0.1923 (+0.1238): beats (boot z +30.93) |
| const_var | variance/corr_var_err2_spearman | 0.0884 vs 0.0000 (+0.0884): beats (boot z +3.96) | 0.0498 vs 0.0000 (+0.0498): beats (boot z +1.99) | 0.0844 vs 0.0000 (+0.0844): beats (boot z +3.59) |

## Backtest (costs included)

- n_trades: 1288
- total_return: -0.9670
- sharpe_net: -127.2524
- sharpe_gross: -3.0269
- sortino: -142.5802
- max_drawdown: 0.9670
- hit_rate: 0.0637
- hit_rate_gross: 0.4573
- profit_factor: 0.0325
- avg_hold_bars: 12.2547
- exposure: 0.3654
- turnover: 727.6129
- fees_paid: 7276.0918
- traded_notional: 7276091.7642
- breakeven_cost_bps: -0.5808
- gross_edge_per_trade_bps: -0.4382
- costs_paid: 9458.9193
- gross_pnl: -211.3138
- net_pnl: -9670.2331

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T18:38:00 .. 2025-08-30T18:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.9334, long_above 0.5921, short_below 0.5020, median 0.5455. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -96.70% | -127.25 | +96.70% | 1288 |
| buy and hold | -8.03% | -2.84 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -96.94% .. -96.01%) | -96.53% | -135.30 | | |

The random null enters at the strategy's rate (0.0470 per flat bar), holds 12 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 27% of its seeds on net return, 98% on net Sharpe and 19% on gross return.
