# Evaluation report - dev split - run `20261001T001645Z-fb840fd-68e19787-ece0__f-38__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 18047 | 19287 | 19803 |
| n_eff of the scored moves (n scored // bars ahead) | 1804 | 1285 | 990 |
| true up-rate | 0.5140 | 0.5117 | 0.5132 |
| calls up (predicted up-rate) | 0.7165 | 0.5863 | 0.5718 |
| accuracy | 0.4994 | 0.5191 | 0.5085 |
| balanced accuracy | 0.4933 | 0.5171 | 0.5066 |
| precision (up) | 0.5094 | 0.5263 | 0.5189 |
| recall / sensitivity (up) | 0.7100 | 0.6030 | 0.5782 |
| specificity (down) | 0.2766 | 0.4312 | 0.4350 |
| F1 (up) | 0.5932 | 0.5620 | 0.5470 |
| MCC | -0.0148 | 0.0347 | 0.0133 |
| AUC | 0.4885 | 0.5239 | 0.5083 |
| Brier | 0.2698 | 0.2545 | 0.2608 |
| ECE (positive class) | 0.1005 | 0.0435 | 0.0732 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0140 | 0.0117 | 0.0132 |
| TP / FP / TN / FN | 6587 / 6344 / 2426 / 2690 | 5951 / 5357 / 4061 / 3918 | 5876 / 5447 / 4193 / 4287 |
| Gaussian readout: calls up | 0.6800 | n/a (beta = 0: readout is the constant 0.5) | 0.4873 |
| Gaussian readout: MCC | -0.0003 | n/a (beta = 0: readout is the constant 0.5) | 0.0751 |
| Gaussian readout: AUC | 0.5006 | n/a (beta = 0: readout is the constant 0.5) | 0.5452 |
| Gaussian readout: Brier | 0.2500 | n/a (beta = 0: readout is the constant 0.5) | 0.2499 |
| Gaussian readout: ECE | 0.0126 | n/a (beta = 0: readout is the constant 0.5) | 0.0358 |
| Gaussian readout of the raw heads: calls up | 0.6800 | 0.5408 | 0.4873 |
| Gaussian readout of the raw heads: MCC | -0.0003 | 0.0452 | 0.0751 |
| Gaussian readout of the raw heads: AUC | 0.5006 | 0.5359 | 0.5452 |
| Gaussian readout of the raw heads: Brier | 0.2511 | 0.2516 | 0.2503 |
| Gaussian readout of the raw heads: ECE | 0.0243 | 0.0417 | 0.0272 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 105.51 | 129.48 | 146.87 |
| RMSE ($), raw heads | 105.73 | 130.88 | 148.61 |
| RMSE ($), zero prediction | 105.50 | 129.48 | 146.87 |
| MAE ($), served | 66.17 | 81.17 | 92.11 |
| MAE ($), raw heads | 66.31 | 81.76 | 92.38 |
| MAE ($), zero prediction | 66.17 | 81.17 | 92.15 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | n/a (beta = 0: served delta is 0) | 0.0000 |
| skill vs zero, raw heads | -0.0043 | -0.0217 | -0.0238 |
| EV, served | -0.0001 | n/a (beta = 0: served delta is 0) | 0.0000 |
| EV, raw heads | -0.0042 | -0.0213 | -0.0241 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0095 | 0.0279 | 0.0067 |
| corr, Spearman, raw heads | 0.0070 | 0.0625 | 0.0766 |
| mean predicted ($), served | 0.21 | 0.00 | 0.04 |
| mean predicted ($), raw heads | 2.42 | 4.50 | 1.93 |
| mean realised ($) | 1.08 | 1.63 | 2.19 |
| share predicted up, raw heads | 0.6828 | 0.5317 | 0.4788 |
| share realised up | 0.5061 | 0.5062 | 0.5070 |
| shrink beta (served = beta x raw, fit on cal) | 0.0861 | 0.0000 | 0.0227 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 49.06 | 60.23 | 68.72 |
| CRPSS vs constant variance | 0.0219 | 0.0183 | 0.0164 |
| NLL | 5.9504 | 6.1472 | 6.2974 |
| PIT KS | 0.0295 | 0.0411 | 0.0405 |
| var / err^2 Spearman | 0.3027 | 0.2963 | 0.2770 |
| coverage of the 90% interval | 0.9017 | 0.9035 | 0.9107 |
| width of the 90% interval ($) | 303.59 | 375.79 | 443.92 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0087 | [-0.0321, 0.0186] | NOISE |
| h1 | 0.0220 | [-0.0020, 0.0464] | NOISE |
| h2 | -0.0005 | [-0.0223, 0.0234] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.086 / h1 0.000 / h2 0.023) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7672 | n/a (beta = 0: served delta is 0) | 0.6144 |
| abs(d h1) <= abs(d h2) | 0.5765 | n/a (beta = 0: served delta is 0) | 0.5797 |
| full chain h0 <= h1 <= h2 | 0.4010 | n/a (beta = 0: served delta is 0) | 0.3220 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6150 | 0.6898 | 0.6544 | 0.3034 |
| expected if the two signs were independent | 0.5825 | 0.5054 | 0.4968 | 0.1934 |

- P(up) unanimity (all three horizons call the same side): 0.4549

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0148 vs 0.0353 (-0.0500): does not beat, significantly worse (boot z -2.15) | 0.0347 vs 0.0473 (-0.0126): does not beat, noise (boot z -0.91) | 0.0133 vs 0.0546 (-0.0413): does not beat, significantly worse (boot z -2.40) |
| logreg_lags | direction/auc | 0.4885 vs 0.5324 (-0.0439): does not beat, significantly worse (boot z -2.79) | 0.5239 vs 0.5407 (-0.0168): does not beat, significantly worse (boot z -2.46) | 0.5083 vs 0.5445 (-0.0362): does not beat, significantly worse (boot z -3.47) |
| logreg_lags | direction/brier | 0.2698 vs 0.2498 (-0.0201): does not beat, significantly worse (DM z -6.94) | 0.2545 vs 0.2495 (-0.0050): does not beat, significantly worse (DM z -4.71) | 0.2608 vs 0.2493 (-0.0115): does not beat, significantly worse (DM z -6.01) |
| logreg_lags | direction/ece_pos | 0.1005 vs 0.0113 (-0.0892): does not beat, significantly worse (boot z -7.10) | 0.0435 vs 0.0116 (-0.0320): does not beat, significantly worse (boot z -3.93) | 0.0732 vs 0.0111 (-0.0621): does not beat, significantly worse (boot z -5.52) |
| logreg_lags | direction/acc | 0.4994 vs 0.5199 (-0.0205): does not beat, noise (DM z -1.80) | 0.5191 vs 0.5255 (-0.0064): does not beat, noise (DM z -0.99) | 0.5085 vs 0.5292 (-0.0207): does not beat, significantly worse (DM z -2.39) |
| logreg_lags | direction/bal_acc | 0.4933 vs 0.5173 (-0.0240): does not beat, significantly worse (boot z -2.18) | 0.5171 vs 0.5232 (-0.0061): does not beat, noise (boot z -0.90) | 0.5066 vs 0.5269 (-0.0203): does not beat, significantly worse (boot z -2.39) |
| class_prior | direction/mcc | -0.0148 vs 0.0000 (-0.0148): does not beat, noise (boot z -0.93) | 0.0347 vs 0.0000 (+0.0347): beats (boot z +2.23) | 0.0133 vs 0.0000 (+0.0133): beats, noise (boot z +0.80) |
| class_prior | direction/auc | 0.4885 vs 0.5000 (-0.0115): does not beat, noise (boot z -1.14) | 0.5239 vs 0.5000 (+0.0239): beats (boot z +2.33) | 0.5083 vs 0.5000 (+0.0083): beats, noise (boot z +0.78) |
| class_prior | direction/brier | 0.2698 vs 0.2499 (-0.0200): does not beat, significantly worse (DM z -8.00) | 0.2545 vs 0.2499 (-0.0046): does not beat, significantly worse (DM z -2.71) | 0.2608 vs 0.2498 (-0.0110): does not beat, significantly worse (DM z -4.97) |
| class_prior | direction/ece_pos | 0.1005 vs 0.0074 (-0.0930): does not beat, significantly worse (boot z -6.94) | 0.0435 vs 0.0035 (-0.0400): does not beat, significantly worse (boot z -3.78) | 0.0732 vs 0.0049 (-0.0683): does not beat, significantly worse (boot z -5.58) |
| class_prior | direction/acc | 0.4994 vs 0.5140 (-0.0146): does not beat, noise (DM z -1.69) | 0.5191 vs 0.5117 (+0.0074): beats, noise (DM z +0.61) | 0.5085 vs 0.5132 (-0.0047): does not beat, noise (DM z -0.35) |
| class_prior | direction/bal_acc | 0.4933 vs 0.5000 (-0.0067): does not beat, noise (boot z -0.93) | 0.5171 vs 0.5000 (+0.0171): beats (boot z +2.23) | 0.5066 vs 0.5000 (+0.0066): beats, noise (boot z +0.80) |
| zero_delta | delta/rmse | 105.51 vs 105.50 (-0.00, -0.00%): does not beat, noise (DM z -0.25) | 129.48 vs 129.48 (+0.00, +0.00%): does not beat | 146.87 vs 146.87 (+0.00, +0.00%): beats, noise (DM z +0.16) |
| zero_delta | delta/mae | 66.17 vs 66.17 (+0.00, +0.00%): beats, noise (DM z +0.07) | 81.17 vs 81.17 (+0.00, +0.00%): does not beat | 92.11 vs 92.15 (+0.03, +0.04%): beats (DM z +3.28) |
| mean_delta | delta/rmse | 105.51 vs 105.50 (-0.01, -0.01%): does not beat, noise (DM z -0.53) | 129.48 vs 129.47 (-0.01, -0.01%): does not beat, noise (DM z -0.36) | 146.87 vs 146.86 (-0.01, -0.01%): does not beat, noise (DM z -0.31) |
| mean_delta | delta/mae | 66.17 vs 66.16 (-0.01, -0.01%): does not beat, noise (DM z -0.80) | 81.17 vs 81.15 (-0.01, -0.02%): does not beat, noise (DM z -0.58) | 92.11 vs 92.13 (+0.02, +0.02%): beats, noise (DM z +0.46) |
| const_var | variance/crps | 49.06 vs 50.16 (+1.10, +2.19%): beats (DM z +6.27) | 60.23 vs 61.35 (+1.12, +1.83%): beats (DM z +4.44) | 68.72 vs 69.87 (+1.14, +1.64%): beats (DM z +3.50) |
| const_var | variance/nll | 5.9504 vs 6.4056 (+0.4551): beats (DM z +4.28) | 6.1472 vs 6.6267 (+0.4795): beats (DM z +3.50) | 6.2974 vs 6.7358 (+0.4384): beats (DM z +2.94) |
| const_var | variance/pit_ks | 0.0295 vs 0.0380 (+0.0085): beats, noise (boot z +1.00) | 0.0411 vs 0.0386 (-0.0025): does not beat, noise (boot z -0.29) | 0.0405 vs 0.0388 (-0.0017): does not beat, noise (boot z -0.17) |
| const_var | variance/corr_var_err2_spearman | 0.3027 vs 0.0000 (+0.3027): beats (boot z +16.44) | 0.2963 vs 0.0000 (+0.2963): beats (boot z +14.70) | 0.2770 vs 0.0000 (+0.2770): beats (boot z +12.49) |

## Backtest (costs included)

- n_trades: 1131
- total_return: 0.0359
- sharpe_net: 1.8259
- sharpe_gross: 1.8259
- sortino: 2.5947
- max_drawdown: 0.0928
- hit_rate: 0.5349
- hit_rate_gross: 0.5349
- profit_factor: 1.0350
- avg_hold_bars: 8.5022
- exposure: 0.3943
- turnover: 2386.1685
- fees_paid: 0.0000
- traded_notional: 23863245.5477
- breakeven_cost_bps: 0.3008
- gross_edge_per_trade_bps: 0.3494
- costs_paid: 0.0000
- gross_pnl: 358.9537
- net_pnl: 358.9537

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -38 (TimeSeriesSplit fold 3, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2023-12-25T13:18:00 .. 2024-01-11T11:47:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.206, long_above 0.5984, short_below 0.4411, median 0.5275. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +3.59% | +1.83 | +9.28% | 1131 |
| buy and hold | +6.12% | +2.44 | +9.52% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -12.01% .. +12.49%) | +0.17% | +0.12 | | |

The random null enters at the strategy's rate (0.0766 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 70% of its seeds on net return, 70% on net Sharpe and 70% on gross return.
