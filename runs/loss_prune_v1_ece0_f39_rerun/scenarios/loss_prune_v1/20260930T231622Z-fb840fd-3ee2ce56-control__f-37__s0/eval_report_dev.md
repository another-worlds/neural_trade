# Evaluation report - dev split - run `20260930T231622Z-fb840fd-3ee2ce56-control__f-37__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 17330 | 18405 | 19141 |
| n_eff of the scored moves (n scored // bars ahead) | 1733 | 1227 | 957 |
| true up-rate | 0.4990 | 0.4981 | 0.4981 |
| calls up (predicted up-rate) | 0.3949 | 0.4862 | 0.5738 |
| accuracy | 0.5162 | 0.5103 | 0.5051 |
| balanced accuracy | 0.5160 | 0.5102 | 0.5054 |
| precision (up) | 0.5193 | 0.5086 | 0.5029 |
| recall / sensitivity (up) | 0.4110 | 0.4965 | 0.5792 |
| specificity (down) | 0.6211 | 0.5240 | 0.4316 |
| F1 (up) | 0.4588 | 0.5025 | 0.5384 |
| MCC | 0.0327 | 0.0205 | 0.0110 |
| AUC | 0.5159 | 0.5089 | 0.5001 |
| Brier | 0.2545 | 0.2509 | 0.2551 |
| ECE (positive class) | 0.0324 | 0.0145 | 0.0403 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0010 | 0.0019 | 0.0019 |
| TP / FP / TN / FN | 3554 / 3290 / 5392 / 5094 | 4551 / 4397 / 4841 / 4616 | 5523 / 5460 / 4146 / 4012 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.7041 | 0.5841 | 0.6411 |
| Gaussian readout of the raw heads: MCC | -0.0023 | 0.0082 | 0.0038 |
| Gaussian readout of the raw heads: AUC | 0.5025 | 0.5017 | 0.5027 |
| Gaussian readout of the raw heads: Brier | 0.2529 | 0.2563 | 0.2558 |
| Gaussian readout of the raw heads: ECE | 0.0468 | 0.0571 | 0.0589 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 90.11 | 109.57 | 126.22 |
| RMSE ($), raw heads | 91.06 | 114.37 | 130.04 |
| RMSE ($), zero prediction | 90.11 | 109.57 | 126.22 |
| MAE ($), served | 59.37 | 71.99 | 82.66 |
| MAE ($), raw heads | 59.73 | 73.74 | 84.13 |
| MAE ($), zero prediction | 59.37 | 71.99 | 82.66 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0212 | -0.0896 | -0.0614 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0177 | -0.0840 | -0.0556 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0365 | -0.0733 | -0.0661 |
| corr, Spearman, raw heads | -0.0001 | 0.0004 | 0.0014 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 4.24 | 6.46 | 7.32 |
| mean realised ($) | -1.30 | -1.98 | -2.67 |
| share predicted up, raw heads | 0.6963 | 0.5663 | 0.6296 |
| share realised up | 0.5005 | 0.4945 | 0.4976 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 43.37 | 52.62 | 60.58 |
| CRPSS vs constant variance | 0.0377 | 0.0396 | 0.0374 |
| NLL | 5.7166 | 5.8992 | 6.0513 |
| PIT KS | 0.0189 | 0.0293 | 0.0249 |
| var / err^2 Spearman | 0.3914 | 0.3930 | 0.3876 |
| coverage of the 90% interval | 0.9068 | 0.9062 | 0.9070 |
| width of the 90% interval ($) | 258.40 | 318.72 | 362.88 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0031 | [-0.0263, 0.0229] | NOISE |
| h1 | -0.0086 | [-0.0293, 0.0142] | NOISE |
| h2 | -0.0192 | [-0.0427, 0.0060] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7090 | n/a (beta = 0: served delta is 0) | 0.6066 |
| abs(d h1) <= abs(d h2) | 0.5859 | n/a (beta = 0: served delta is 0) | 0.5887 |
| full chain h0 <= h1 <= h2 | 0.3833 | n/a (beta = 0: served delta is 0) | 0.3212 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5216 | 0.5511 | 0.6275 | 0.2191 |
| expected if the two signs were independent | 0.4554 | 0.5011 | 0.5198 | 0.1535 |

- P(up) unanimity (all three horizons call the same side): 0.3517

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0327 vs 0.0245 (+0.0083): beats, noise (boot z +0.37) | 0.0205 vs 0.0175 (+0.0030): beats, noise (boot z +0.17) | 0.0110 vs 0.0080 (+0.0029): beats, noise (boot z +0.16) |
| logreg_lags | direction/auc | 0.5159 vs 0.5197 (-0.0038): does not beat, noise (boot z -0.27) | 0.5089 vs 0.5155 (-0.0066): does not beat, noise (boot z -0.55) | 0.5001 vs 0.5134 (-0.0132): does not beat, noise (boot z -1.17) |
| logreg_lags | direction/brier | 0.2545 vs 0.2505 (-0.0040): does not beat, significantly worse (DM z -2.30) | 0.2509 vs 0.2515 (+0.0006): beats, noise (DM z +0.51) | 0.2551 vs 0.2515 (-0.0035): does not beat, significantly worse (DM z -3.67) |
| logreg_lags | direction/ece_pos | 0.0324 vs 0.0173 (-0.0151): does not beat, noise (boot z -1.15) | 0.0145 vs 0.0253 (+0.0108): beats, noise (boot z +1.02) | 0.0403 vs 0.0290 (-0.0113): does not beat, noise (boot z -1.20) |
| logreg_lags | direction/acc | 0.5162 vs 0.5106 (+0.0057): beats, noise (DM z +0.46) | 0.5103 vs 0.5070 (+0.0033): beats, noise (DM z +0.33) | 0.5051 vs 0.5028 (+0.0023): beats, noise (DM z +0.25) |
| logreg_lags | direction/bal_acc | 0.5160 vs 0.5110 (+0.0050): beats, noise (boot z +0.49) | 0.5102 vs 0.5079 (+0.0024): beats, noise (boot z +0.29) | 0.5054 vs 0.5036 (+0.0018): beats, noise (boot z +0.21) |
| class_prior | direction/mcc | 0.0327 vs 0.0000 (+0.0327): beats (boot z +2.33) | 0.0205 vs 0.0000 (+0.0205): beats, noise (boot z +1.59) | 0.0110 vs 0.0000 (+0.0110): beats, noise (boot z +0.69) |
| class_prior | direction/auc | 0.5159 vs 0.5000 (+0.0159): beats, noise (boot z +1.68) | 0.5089 vs 0.5000 (+0.0089): beats, noise (boot z +1.11) | 0.5001 vs 0.5000 (+0.0001): beats, noise (boot z +0.01) |
| class_prior | direction/brier | 0.2545 vs 0.2502 (-0.0043): does not beat, significantly worse (DM z -2.82) | 0.2509 vs 0.2503 (-0.0006): does not beat, noise (DM z -0.92) | 0.2551 vs 0.2502 (-0.0048): does not beat, significantly worse (DM z -3.66) |
| class_prior | direction/ece_pos | 0.0324 vs 0.0133 (-0.0191): does not beat, noise (boot z -1.38) | 0.0145 vs 0.0162 (+0.0017): beats, noise (boot z +0.15) | 0.0403 vs 0.0157 (-0.0246): does not beat, significantly worse (boot z -2.09) |
| class_prior | direction/acc | 0.5162 vs 0.4990 (+0.0172): beats, noise (DM z +1.25) | 0.5103 vs 0.4981 (+0.0122): beats, noise (DM z +0.90) | 0.5051 vs 0.4981 (+0.0070): beats, noise (DM z +0.51) |
| class_prior | direction/bal_acc | 0.5160 vs 0.5000 (+0.0160): beats (boot z +2.33) | 0.5102 vs 0.5000 (+0.0102): beats, noise (boot z +1.59) | 0.5054 vs 0.5000 (+0.0054): beats, noise (boot z +0.69) |
| zero_delta | delta/rmse | 90.11 vs 90.11 (+0.00, +0.00%): does not beat | 109.57 vs 109.57 (+0.00, +0.00%): does not beat | 126.22 vs 126.22 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 59.37 vs 59.37 (+0.00, +0.00%): does not beat | 71.99 vs 71.99 (+0.00, +0.00%): does not beat | 82.66 vs 82.66 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 90.11 vs 90.14 (+0.03, +0.04%): beats, noise (DM z +1.22) | 109.57 vs 109.62 (+0.06, +0.05%): beats, noise (DM z +1.22) | 126.22 vs 126.32 (+0.09, +0.07%): beats, noise (DM z +1.21) |
| mean_delta | delta/mae | 59.37 vs 59.38 (+0.00, +0.01%): beats, noise (DM z +0.21) | 71.99 vs 72.03 (+0.04, +0.05%): beats, noise (DM z +0.97) | 82.66 vs 82.70 (+0.04, +0.05%): beats, noise (DM z +0.64) |
| const_var | variance/crps | 43.37 vs 45.08 (+1.70, +3.77%): beats (DM z +12.92) | 52.62 vs 54.79 (+2.17, +3.96%): beats (DM z +9.43) | 60.58 vs 62.93 (+2.35, +3.74%): beats (DM z +9.26) |
| const_var | variance/nll | 5.7166 vs 5.9949 (+0.2783): beats (DM z +7.19) | 5.8992 vs 6.1909 (+0.2916): beats (DM z +5.65) | 6.0513 vs 6.3349 (+0.2836): beats (DM z +4.93) |
| const_var | variance/pit_ks | 0.0189 vs 0.0468 (+0.0279): beats (boot z +4.44) | 0.0293 vs 0.0515 (+0.0223): beats (boot z +3.14) | 0.0249 vs 0.0525 (+0.0276): beats (boot z +3.39) |
| const_var | variance/corr_var_err2_spearman | 0.3914 vs 0.0000 (+0.3914): beats (boot z +19.59) | 0.3930 vs 0.0000 (+0.3930): beats (boot z +18.60) | 0.3876 vs 0.0000 (+0.3876): beats (boot z +17.04) |

## Backtest (costs included)

- n_trades: 921
- total_return: 0.0432
- sharpe_net: 2.6545
- sharpe_gross: 2.6545
- sortino: 3.8057
- max_drawdown: 0.0596
- hit_rate: 0.5483
- hit_rate_gross: 0.5483
- profit_factor: 1.0543
- avg_hold_bars: 8.5429
- exposure: 0.3226
- turnover: 1874.6580
- fees_paid: 0.0000
- traded_notional: 18746098.2055
- breakeven_cost_bps: 0.4613
- gross_edge_per_trade_bps: 0.4915
- costs_paid: 0.0000
- gross_pnl: 432.4198
- net_pnl: 432.4198

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -37 (TimeSeriesSplit fold 4, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-01-11T11:48:00 .. 2024-01-28T10:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.134, long_above 0.5395, short_below 0.4479, median 0.4952. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +4.32% | +2.65 | +5.96% | 921 |
| buy and hold | -7.01% | -2.90 | +21.43% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -7.98% .. +9.52%) | -0.29% | -0.19 | | |

The random null enters at the strategy's rate (0.0557 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 82% of its seeds on net return, 79% on net Sharpe and 82% on gross return.
