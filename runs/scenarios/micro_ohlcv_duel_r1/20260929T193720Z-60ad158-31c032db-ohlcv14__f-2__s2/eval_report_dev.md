# Evaluation report - dev split - run `20260929T193720Z-60ad158-31c032db-ohlcv14__f-2__s2`

n = 43200 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26656 | 29751 | 31412 |
| n_eff of the scored moves (n scored // bars ahead) | 2665 | 1983 | 1570 |
| true up-rate | 0.4897 | 0.4896 | 0.4878 |
| calls up (predicted up-rate) | 0.4578 | 0.6107 | 0.5546 |
| accuracy | 0.5110 | 0.5084 | 0.4997 |
| balanced accuracy | 0.5101 | 0.5107 | 0.5011 |
| precision (up) | 0.5007 | 0.4983 | 0.4888 |
| recall / sensitivity (up) | 0.4681 | 0.6216 | 0.5557 |
| specificity (down) | 0.5522 | 0.3998 | 0.4465 |
| F1 (up) | 0.4839 | 0.5532 | 0.5201 |
| MCC | 0.0203 | 0.0219 | 0.0022 |
| AUC | 0.5115 | 0.5165 | 0.5012 |
| Brier | 0.2576 | 0.2623 | 0.2565 |
| ECE (positive class) | 0.0574 | 0.0714 | 0.0605 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0103 | 0.0104 | 0.0122 |
| TP / FP / TN / FN | 6110 / 6092 / 7511 / 6943 | 9054 / 9115 / 6071 / 5511 | 8515 / 8905 / 7183 / 6809 |
| Gaussian readout: calls up | 0.5732 | 0.6808 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | -0.0108 | 0.0055 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.4936 | 0.5069 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2501 | 0.2501 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0109 | 0.0141 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.5732 | 0.6808 | 0.4493 |
| Gaussian readout of the raw heads: MCC | -0.0108 | 0.0055 | 0.0017 |
| Gaussian readout of the raw heads: AUC | 0.4936 | 0.5069 | 0.5022 |
| Gaussian readout of the raw heads: Brier | 0.2549 | 0.2560 | 0.2538 |
| Gaussian readout of the raw heads: ECE | 0.0573 | 0.0642 | 0.0520 |

beta = 0 for h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 165.10 | 199.77 | 230.16 |
| RMSE ($), raw heads | 166.01 | 201.91 | 231.57 |
| RMSE ($), zero prediction | 165.09 | 199.76 | 230.16 |
| MAE ($), served | 112.33 | 137.89 | 158.85 |
| MAE ($), raw heads | 113.48 | 140.13 | 160.37 |
| MAE ($), zero prediction | 112.32 | 137.87 | 158.85 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | -0.0001 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0112 | -0.0216 | -0.0123 |
| EV, served | -0.0000 | 0.0001 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0103 | -0.0163 | -0.0125 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0010 | 0.0113 | 0.0007 |
| corr, Spearman, raw heads | -0.0036 | 0.0022 | -0.0019 |
| mean predicted ($), served | 0.19 | 1.29 | 0.00 |
| mean predicted ($), raw heads | 3.46 | 12.34 | -3.96 |
| mean realised ($) | -1.61 | -2.42 | -3.21 |
| share predicted up, raw heads | 0.5984 | 0.6530 | 0.4466 |
| share realised up | 0.4899 | 0.4879 | 0.4886 |
| shrink beta (served = beta x raw, fit on cal) | 0.0545 | 0.1042 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 83.83 | 103.09 | 117.83 |
| CRPSS vs constant variance | 0.0038 | -0.0018 | 0.0088 |
| NLL | 6.5527 | 6.7476 | 6.8626 |
| PIT KS | 0.0407 | 0.0512 | 0.0385 |
| var / err^2 Spearman | 0.0822 | 0.0211 | 0.1297 |
| coverage of the 90% interval | 0.9056 | 0.9043 | 0.8971 |
| width of the 90% interval ($) | 513.27 | 623.02 | 696.68 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0084 | [-0.0100, 0.0270] | NOISE |
| h1 | 0.0144 | [-0.0059, 0.0352] | NOISE |
| h2 | -0.0023 | [-0.0177, 0.0115] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.054 / h1 0.104 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6590 | 0.8070 | 0.6173 |
| abs(d h1) <= abs(d h2) | 0.5110 | n/a (beta = 0: served delta is 0) | 0.5917 |
| full chain h0 <= h1 <= h2 | 0.2378 | n/a (beta = 0: served delta is 0) | 0.3339 |

beta = 0 for h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4396 | 0.6077 | 0.4661 | 0.1277 |
| expected if the two signs were independent | 0.4901 | 0.5294 | 0.4918 | 0.1277 |

- P(up) unanimity (all three horizons call the same side): 0.2931

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0203 vs 0.0011 (+0.0192): beats, noise (boot z +0.80) | 0.0219 vs 0.0027 (+0.0192): beats, noise (boot z +1.13) | 0.0022 vs -0.0018 (+0.0040): beats, noise (boot z +0.21) |
| logreg_lags | direction/auc | 0.5115 vs 0.5085 (+0.0030): beats, noise (boot z +0.19) | 0.5165 vs 0.5137 (+0.0028): beats, noise (boot z +0.28) | 0.5012 vs 0.5174 (-0.0162): does not beat, noise (boot z -1.17) |
| logreg_lags | direction/brier | 0.2576 vs 0.2513 (-0.0063): does not beat, significantly worse (DM z -3.57) | 0.2623 vs 0.2517 (-0.0106): does not beat, significantly worse (DM z -5.58) | 0.2565 vs 0.2518 (-0.0046): does not beat, significantly worse (DM z -3.77) |
| logreg_lags | direction/ece_pos | 0.0574 vs 0.0347 (-0.0228): does not beat, significantly worse (boot z -2.04) | 0.0714 vs 0.0387 (-0.0327): does not beat, significantly worse (boot z -3.74) | 0.0605 vs 0.0432 (-0.0173): does not beat, noise (boot z -1.66) |
| logreg_lags | direction/acc | 0.5110 vs 0.4921 (+0.0189): beats, noise (DM z +1.71) | 0.5084 vs 0.4923 (+0.0161): beats, noise (DM z +1.93) | 0.4997 vs 0.4895 (+0.0103): beats, noise (DM z +1.03) |
| logreg_lags | direction/bal_acc | 0.5101 vs 0.5003 (+0.0098): beats, noise (boot z +1.03) | 0.5107 vs 0.5008 (+0.0099): beats, noise (boot z +1.56) | 0.5011 vs 0.4995 (+0.0016): beats, noise (boot z +0.23) |
| class_prior | direction/mcc | 0.0203 vs 0.0000 (+0.0203): beats, noise (boot z +1.50) | 0.0219 vs 0.0000 (+0.0219): beats, noise (boot z +1.74) | 0.0022 vs 0.0000 (+0.0022): beats, noise (boot z +0.20) |
| class_prior | direction/auc | 0.5115 vs 0.5000 (+0.0115): beats, noise (boot z +1.32) | 0.5165 vs 0.5000 (+0.0165): beats, noise (boot z +1.89) | 0.5012 vs 0.5000 (+0.0012): beats, noise (boot z +0.18) |
| class_prior | direction/brier | 0.2576 vs 0.2508 (-0.0068): does not beat, significantly worse (DM z -4.44) | 0.2623 vs 0.2510 (-0.0112): does not beat, significantly worse (DM z -5.20) | 0.2565 vs 0.2512 (-0.0052): does not beat, significantly worse (DM z -4.85) |
| class_prior | direction/ece_pos | 0.0574 vs 0.0297 (-0.0278): does not beat, significantly worse (boot z -2.49) | 0.0714 vs 0.0335 (-0.0379): does not beat, significantly worse (boot z -3.97) | 0.0605 vs 0.0371 (-0.0235): does not beat, significantly worse (boot z -2.17) |
| class_prior | direction/acc | 0.5110 vs 0.4897 (+0.0213): beats (DM z +2.01) | 0.5084 vs 0.4896 (+0.0188): beats (DM z +2.03) | 0.4997 vs 0.4878 (+0.0119): beats, noise (DM z +1.11) |
| class_prior | direction/bal_acc | 0.5101 vs 0.5000 (+0.0101): beats, noise (boot z +1.50) | 0.5107 vs 0.5000 (+0.0107): beats, noise (boot z +1.74) | 0.5011 vs 0.5000 (+0.0011): beats, noise (boot z +0.20) |
| zero_delta | delta/rmse | 165.10 vs 165.09 (-0.01, -0.00%): does not beat, noise (DM z -0.53) | 199.77 vs 199.76 (-0.01, -0.00%): does not beat, noise (DM z -0.14) | 230.16 vs 230.16 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 112.33 vs 112.32 (-0.01, -0.01%): does not beat, noise (DM z -0.98) | 137.89 vs 137.87 (-0.03, -0.02%): does not beat, noise (DM z -0.64) | 158.85 vs 158.85 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 165.10 vs 165.39 (+0.29, +0.18%): beats (DM z +2.53) | 199.77 vs 200.31 (+0.54, +0.27%): beats (DM z +2.96) | 230.16 vs 231.00 (+0.84, +0.36%): beats (DM z +2.55) |
| mean_delta | delta/mae | 112.33 vs 112.73 (+0.40, +0.36%): beats (DM z +4.08) | 137.89 vs 138.56 (+0.67, +0.48%): beats (DM z +3.95) | 158.85 vs 159.88 (+1.03, +0.64%): beats (DM z +3.69) |
| const_var | variance/crps | 83.83 vs 84.15 (+0.32, +0.38%): beats (DM z +4.28) | 103.09 vs 102.91 (-0.18, -0.18%): does not beat, noise (DM z -1.23) | 117.83 vs 118.88 (+1.05, +0.88%): beats (DM z +5.54) |
| const_var | variance/nll | 6.5527 vs 6.5630 (+0.0102): beats, noise (DM z +1.57) | 6.7476 vs 6.7470 (-0.0006): does not beat, noise (DM z -0.08) | 6.8626 vs 6.8845 (+0.0219): beats (DM z +3.03) |
| const_var | variance/pit_ks | 0.0407 vs 0.0612 (+0.0205): beats (boot z +11.30) | 0.0512 vs 0.0652 (+0.0141): beats (boot z +5.24) | 0.0385 vs 0.0725 (+0.0341): beats (boot z +10.06) |
| const_var | variance/corr_var_err2_spearman | 0.0822 vs 0.0000 (+0.0822): beats (boot z +7.44) | 0.0211 vs 0.0000 (+0.0211): beats, noise (boot z +1.27) | 0.1297 vs 0.0000 (+0.1297): beats (boot z +8.54) |

## Backtest (costs included)

- n_trades: 1546
- total_return: -0.9805
- sharpe_net: -137.5564
- sharpe_gross: 5.5187
- sortino: -152.5260
- max_drawdown: 0.9805
- hit_rate: 0.0563
- hit_rate_gross: 0.5259
- profit_factor: 0.0309
- avg_hold_bars: 9.8952
- exposure: 0.3541
- turnover: 784.1602
- fees_paid: 7841.9022
- traded_notional: 7841902.1973
- breakeven_cost_bps: 0.9936
- gross_edge_per_trade_bps: 0.5827
- costs_paid: 10194.4729
- gross_pnl: 389.5854
- net_pnl: -9804.8874

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -2 (TimeSeriesSplit fold 1, 2 usable folds); this report scores the fold's out-of-sample block: 43200 sequences, 2025-07-31T23:38:00 .. 2025-08-30T23:37:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.9838, long_above 0.5755, short_below 0.4436, median 0.5069. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.05% | -137.56 | +98.05% | 1546 |
| buy and hold | -6.39% | -2.21 | +13.68% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.41% .. -97.90%) | -98.18% | -154.53 | | |

The random null enters at the strategy's rate (0.0554 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 77% of its seeds on net return, 100% on net Sharpe and 100% on gross return.
