# Evaluation report - dev split - run `20261001T004350Z-fb840fd-622d7752-ece0__f-35__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 17597 | 18683 | 19416 |
| n_eff of the scored moves (n scored // bars ahead) | 1759 | 1245 | 970 |
| true up-rate | 0.5230 | 0.5219 | 0.5231 |
| calls up (predicted up-rate) | 0.4456 | 0.3947 | 0.3962 |
| accuracy | 0.5062 | 0.5094 | 0.5001 |
| balanced accuracy | 0.5087 | 0.5141 | 0.5049 |
| precision (up) | 0.5327 | 0.5397 | 0.5292 |
| recall / sensitivity (up) | 0.4539 | 0.4082 | 0.4008 |
| specificity (down) | 0.5635 | 0.6200 | 0.6089 |
| F1 (up) | 0.4901 | 0.4648 | 0.4562 |
| MCC | 0.0175 | 0.0288 | 0.0099 |
| AUC | 0.5089 | 0.5233 | 0.5046 |
| Brier | 0.2641 | 0.2554 | 0.2611 |
| ECE (positive class) | 0.0822 | 0.0534 | 0.0766 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0230 | 0.0219 | 0.0231 |
| TP / FP / TN / FN | 4177 / 3664 / 4730 / 5026 | 3980 / 3394 / 5538 / 5771 | 4071 / 3622 / 5638 / 6085 |
| Gaussian readout: calls up | 0.3509 | n/a (beta = 0: readout is the constant 0.5) | 0.4009 |
| Gaussian readout: MCC | 0.0211 | n/a (beta = 0: readout is the constant 0.5) | 0.0191 |
| Gaussian readout: AUC | 0.5129 | n/a (beta = 0: readout is the constant 0.5) | 0.5129 |
| Gaussian readout: Brier | 0.2500 | n/a (beta = 0: readout is the constant 0.5) | 0.2500 |
| Gaussian readout: ECE | 0.0248 | n/a (beta = 0: readout is the constant 0.5) | 0.0231 |
| Gaussian readout of the raw heads: calls up | 0.3509 | 0.2695 | 0.4009 |
| Gaussian readout of the raw heads: MCC | 0.0211 | 0.0075 | 0.0191 |
| Gaussian readout of the raw heads: AUC | 0.5129 | 0.5085 | 0.5129 |
| Gaussian readout of the raw heads: Brier | 0.2528 | 0.2643 | 0.2619 |
| Gaussian readout of the raw heads: ECE | 0.0430 | 0.0911 | 0.0837 |

beta = 0 for h1: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 135.11 | 166.70 | 192.12 |
| RMSE ($), raw heads | 135.58 | 175.52 | 197.39 |
| RMSE ($), zero prediction | 135.21 | 166.70 | 192.12 |
| MAE ($), served | 82.01 | 100.58 | 116.69 |
| MAE ($), raw heads | 82.67 | 106.23 | 121.00 |
| MAE ($), zero prediction | 82.02 | 100.58 | 116.69 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0015 | n/a (beta = 0: served delta is 0) | 0.0000 |
| skill vs zero, raw heads | -0.0056 | -0.1086 | -0.0556 |
| EV, served | 0.0017 | n/a (beta = 0: served delta is 0) | 0.0000 |
| EV, raw heads | -0.0026 | -0.0959 | -0.0496 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0654 | 0.0519 | 0.0525 |
| corr, Spearman, raw heads | 0.0228 | 0.0069 | 0.0173 |
| mean predicted ($), served | -0.38 | 0.00 | -0.01 |
| mean predicted ($), raw heads | -3.80 | -12.70 | -8.00 |
| mean realised ($) | 5.17 | 7.76 | 10.35 |
| share predicted up, raw heads | 0.3464 | 0.2563 | 0.3976 |
| share realised up | 0.5125 | 0.5147 | 0.5157 |
| shrink beta (served = beta x raw, fit on cal) | 0.0995 | 0.0000 | 0.0012 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 60.16 | 74.30 | 86.08 |
| CRPSS vs constant variance | 0.0481 | 0.0433 | 0.0446 |
| NLL | 6.0508 | 6.3168 | 6.4532 |
| PIT KS | 0.0227 | 0.0280 | 0.0284 |
| var / err^2 Spearman | 0.4366 | 0.4150 | 0.4217 |
| coverage of the 90% interval | 0.9059 | 0.9064 | 0.9071 |
| width of the 90% interval ($) | 364.62 | 451.58 | 526.92 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0046 | [-0.0204, 0.0284] | NOISE |
| h1 | 0.0331 | [0.0056, 0.0582] | WORKS |
| h2 | -0.0081 | [-0.0359, 0.0151] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.099 / h1 0.000 / h2 0.001) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8508 | n/a (beta = 0: served delta is 0) | 0.6121 |
| abs(d h1) <= abs(d h2) | 0.6591 | n/a (beta = 0: served delta is 0) | 0.5941 |
| full chain h0 <= h1 <= h2 | 0.5311 | n/a (beta = 0: served delta is 0) | 0.3332 |

beta = 0 for h1: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6941 | 0.8045 | 0.7880 | 0.5091 |
| expected if the two signs were independent | 0.5191 | 0.5553 | 0.5234 | 0.2748 |

- P(up) unanimity (all three horizons call the same side): 0.6165

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0175 vs 0.0198 (-0.0024): does not beat, noise (boot z -0.09) | 0.0288 vs 0.0222 (+0.0066): beats, noise (boot z +0.35) | 0.0099 vs 0.0218 (-0.0119): does not beat, noise (boot z -0.52) |
| logreg_lags | direction/auc | 0.5089 vs 0.5173 (-0.0083): does not beat, noise (boot z -0.61) | 0.5233 vs 0.5197 (+0.0036): beats, noise (boot z +0.38) | 0.5046 vs 0.5204 (-0.0158): does not beat, noise (boot z -1.32) |
| logreg_lags | direction/brier | 0.2641 vs 0.2500 (-0.0141): does not beat, significantly worse (DM z -5.75) | 0.2554 vs 0.2501 (-0.0053): does not beat, significantly worse (DM z -3.40) | 0.2611 vs 0.2500 (-0.0111): does not beat, significantly worse (DM z -5.76) |
| logreg_lags | direction/ece_pos | 0.0822 vs 0.0150 (-0.0672): does not beat, significantly worse (boot z -6.52) | 0.0534 vs 0.0140 (-0.0393): does not beat, significantly worse (boot z -4.44) | 0.0766 vs 0.0146 (-0.0620): does not beat, significantly worse (boot z -6.51) |
| logreg_lags | direction/acc | 0.5062 vs 0.5172 (-0.0110): does not beat, noise (DM z -0.96) | 0.5094 vs 0.5176 (-0.0082): does not beat, noise (DM z -0.73) | 0.5001 vs 0.5185 (-0.0184): does not beat, noise (DM z -1.42) |
| logreg_lags | direction/bal_acc | 0.5087 vs 0.5093 (-0.0006): does not beat, noise (boot z -0.05) | 0.5141 vs 0.5105 (+0.0036): beats, noise (boot z +0.39) | 0.5049 vs 0.5102 (-0.0053): does not beat, noise (boot z -0.49) |
| class_prior | direction/mcc | 0.0175 vs 0.0000 (+0.0175): beats, noise (boot z +1.03) | 0.0288 vs 0.0000 (+0.0288): beats, noise (boot z +1.55) | 0.0099 vs 0.0000 (+0.0099): beats, noise (boot z +0.57) |
| class_prior | direction/auc | 0.5089 vs 0.5000 (+0.0089): beats, noise (boot z +0.84) | 0.5233 vs 0.5000 (+0.0233): beats (boot z +1.96) | 0.5046 vs 0.5000 (+0.0046): beats, noise (boot z +0.41) |
| class_prior | direction/brier | 0.2641 vs 0.2496 (-0.0145): does not beat, significantly worse (DM z -6.26) | 0.2554 vs 0.2496 (-0.0058): does not beat, significantly worse (DM z -3.04) | 0.2611 vs 0.2496 (-0.0115): does not beat, significantly worse (DM z -5.13) |
| class_prior | direction/ece_pos | 0.0822 vs 0.0122 (-0.0700): does not beat, significantly worse (boot z -6.61) | 0.0534 vs 0.0103 (-0.0431): does not beat, significantly worse (boot z -4.36) | 0.0766 vs 0.0109 (-0.0658): does not beat, significantly worse (boot z -6.47) |
| class_prior | direction/acc | 0.5062 vs 0.5230 (-0.0168): does not beat, noise (DM z -1.26) | 0.5094 vs 0.5219 (-0.0125): does not beat, noise (DM z -0.72) | 0.5001 vs 0.5231 (-0.0230): does not beat, noise (DM z -1.26) |
| class_prior | direction/bal_acc | 0.5087 vs 0.5000 (+0.0087): beats, noise (boot z +1.03) | 0.5141 vs 0.5000 (+0.0141): beats, noise (boot z +1.55) | 0.5049 vs 0.5000 (+0.0049): beats, noise (boot z +0.57) |
| zero_delta | delta/rmse | 135.11 vs 135.21 (+0.10, +0.07%): beats, noise (DM z +0.90) | 166.70 vs 166.70 (+0.00, +0.00%): does not beat | 192.12 vs 192.12 (+0.00, +0.00%): beats, noise (DM z +0.85) |
| zero_delta | delta/mae | 82.01 vs 82.02 (+0.00, +0.00%): beats, noise (DM z +0.11) | 100.58 vs 100.58 (+0.00, +0.00%): does not beat | 116.69 vs 116.69 (+0.00, +0.00%): beats, noise (DM z +0.31) |
| mean_delta | delta/rmse | 135.11 vs 135.18 (+0.07, +0.05%): beats, noise (DM z +0.63) | 166.70 vs 166.65 (-0.05, -0.03%): does not beat, noise (DM z -1.95) | 192.12 vs 192.04 (-0.07, -0.04%): does not beat, noise (DM z -1.85) |
| mean_delta | delta/mae | 82.01 vs 82.00 (-0.02, -0.02%): does not beat, noise (DM z -0.50) | 100.58 vs 100.55 (-0.03, -0.03%): does not beat, noise (DM z -1.44) | 116.69 vs 116.65 (-0.04, -0.04%): does not beat, noise (DM z -1.27) |
| const_var | variance/crps | 60.16 vs 63.20 (+3.04, +4.81%): beats (DM z +11.41) | 74.30 vs 77.66 (+3.36, +4.33%): beats (DM z +9.97) | 86.08 vs 90.10 (+4.02, +4.46%): beats (DM z +9.17) |
| const_var | variance/nll | 6.0508 vs 6.6403 (+0.5894): beats (DM z +3.59) | 6.3168 vs 6.8650 (+0.5482): beats (DM z +3.25) | 6.4532 vs 7.0192 (+0.5660): beats (DM z +3.10) |
| const_var | variance/pit_ks | 0.0227 vs 0.0393 (+0.0166): beats (boot z +4.09) | 0.0280 vs 0.0425 (+0.0146): beats (boot z +3.41) | 0.0284 vs 0.0458 (+0.0174): beats (boot z +4.06) |
| const_var | variance/corr_var_err2_spearman | 0.4366 vs 0.0000 (+0.4366): beats (boot z +22.06) | 0.4150 vs 0.0000 (+0.4150): beats (boot z +19.19) | 0.4217 vs 0.0000 (+0.4217): beats (boot z +18.28) |

## Backtest (costs included)

- n_trades: 1250
- total_return: 0.0858
- sharpe_net: 4.3203
- sharpe_gross: 4.3203
- sortino: 6.1729
- max_drawdown: 0.0469
- hit_rate: 0.5464
- hit_rate_gross: 0.5464
- profit_factor: 1.0792
- avg_hold_bars: 9.1224
- exposure: 0.4676
- turnover: 2544.4177
- fees_paid: 0.0000
- traded_notional: 25445135.1897
- breakeven_cost_bps: 0.6748
- gross_edge_per_trade_bps: 0.6906
- costs_paid: 0.0000
- gross_pnl: 858.4596
- net_pnl: 858.4596

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -35 (TimeSeriesSplit fold 6, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-02-14T08:48:00 .. 2024-03-02T07:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4969, long_above 0.5524, short_below 0.4133, median 0.4864. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +8.58% | +4.32 | +4.69% | 1250 |
| buy and hold | +25.30% | +9.18 | +6.96% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -14.51% .. +12.55%) | -0.73% | -0.42 | | |

The random null enters at the strategy's rate (0.0963 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 88% of its seeds on net return, 82% on net Sharpe and 88% on gross return.
