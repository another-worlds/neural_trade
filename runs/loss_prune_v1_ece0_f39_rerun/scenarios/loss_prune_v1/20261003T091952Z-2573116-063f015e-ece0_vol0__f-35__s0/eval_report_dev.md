# Evaluation report - dev split - run `20261003T091952Z-2573116-063f015e-ece0_vol0__f-35__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 17597 | 18683 | 19416 |
| n_eff of the scored moves (n scored // bars ahead) | 1759 | 1245 | 970 |
| true up-rate | 0.5230 | 0.5219 | 0.5231 |
| calls up (predicted up-rate) | 0.5373 | 0.4717 | 0.4733 |
| accuracy | 0.4988 | 0.5035 | 0.4939 |
| balanced accuracy | 0.4971 | 0.5047 | 0.4951 |
| precision (up) | 0.5203 | 0.5269 | 0.5180 |
| recall / sensitivity (up) | 0.5345 | 0.4762 | 0.4687 |
| specificity (down) | 0.4597 | 0.5333 | 0.5216 |
| F1 (up) | 0.5273 | 0.5002 | 0.4921 |
| MCC | -0.0058 | 0.0094 | -0.0097 |
| AUC | 0.5012 | 0.5060 | 0.4969 |
| Brier | 0.2648 | 0.2576 | 0.2629 |
| ECE (positive class) | 0.0930 | 0.0646 | 0.0886 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0230 | 0.0219 | 0.0231 |
| TP / FP / TN / FN | 4919 / 4535 / 3859 / 4284 | 4643 / 4169 / 4763 / 5108 | 4760 / 4430 / 4830 / 5396 |
| Gaussian readout: calls up | 0.4534 | 0.3467 | 0.3728 |
| Gaussian readout: MCC | -0.0043 | -0.0047 | 0.0023 |
| Gaussian readout: AUC | 0.4948 | 0.4929 | 0.4930 |
| Gaussian readout: Brier | 0.2504 | 0.2503 | 0.2500 |
| Gaussian readout: ECE | 0.0230 | 0.0251 | 0.0236 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 135.35 | 166.75 | 192.14 |
| RMSE ($), raw heads | 137.11 | 172.20 | 200.85 |
| RMSE ($), zero prediction | 135.21 | 166.70 | 192.12 |
| MAE ($), served | 82.11 | 100.62 | 116.71 |
| MAE ($), raw heads | 83.11 | 104.44 | 122.35 |
| MAE ($), zero prediction | 82.02 | 100.58 | 116.69 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0021 | -0.0005 | -0.0002 |
| skill vs zero, raw heads | -0.0283 | -0.0670 | -0.0929 |
| EV, served | -0.0021 | -0.0002 | -0.0001 |
| EV, raw heads | -0.0279 | -0.0598 | -0.0826 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0091 | 0.0040 | -0.0077 |
| corr, Spearman, raw heads | -0.0090 | -0.0103 | -0.0131 |
| mean predicted ($), served | -0.17 | -0.62 | -0.18 |
| mean predicted ($), raw heads | -0.70 | -8.46 | -11.93 |
| mean realised ($) | 5.17 | 7.76 | 10.35 |
| share predicted up, raw heads | 0.4571 | 0.3414 | 0.3737 |
| share realised up | 0.5125 | 0.5147 | 0.5157 |
| shrink beta (served = beta x raw, fit on cal) | 0.2344 | 0.0728 | 0.0151 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 60.29 | 74.42 | 86.21 |
| CRPSS vs constant variance | 0.0461 | 0.0417 | 0.0432 |
| NLL | 6.0820 | 6.3424 | 6.4792 |
| PIT KS | 0.0276 | 0.0333 | 0.0339 |
| var / err^2 Spearman | 0.4422 | 0.4232 | 0.4248 |
| coverage of the 90% interval | 0.9054 | 0.9059 | 0.9072 |
| width of the 90% interval ($) | 364.04 | 451.42 | 526.90 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0058 | [-0.0195, 0.0281] | NOISE |
| h1 | -0.0059 | [-0.0360, 0.0200] | NOISE |
| h2 | 0.0062 | [-0.0198, 0.0306] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.234 / h1 0.073 / h2 0.015) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8596 | 0.3408 | 0.6121 |
| abs(d h1) <= abs(d h2) | 0.7324 | 0.0419 | 0.5941 |
| full chain h0 <= h1 <= h2 | 0.6162 | 0.0013 | 0.3332 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7390 | 0.8003 | 0.7893 | 0.5524 |
| expected if the two signs were independent | 0.4978 | 0.5077 | 0.5067 | 0.2784 |

- P(up) unanimity (all three horizons call the same side): 0.6413

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0058 vs 0.0198 (-0.0256): does not beat, noise (boot z -1.05) | 0.0094 vs 0.0222 (-0.0128): does not beat, noise (boot z -0.64) | -0.0097 vs 0.0218 (-0.0316): does not beat, noise (boot z -1.35) |
| logreg_lags | direction/auc | 0.5012 vs 0.5173 (-0.0160): does not beat, noise (boot z -1.18) | 0.5060 vs 0.5197 (-0.0138): does not beat, noise (boot z -1.21) | 0.4969 vs 0.5204 (-0.0235): does not beat, noise (boot z -1.76) |
| logreg_lags | direction/brier | 0.2648 vs 0.2500 (-0.0148): does not beat, significantly worse (DM z -6.09) | 0.2576 vs 0.2501 (-0.0076): does not beat, significantly worse (DM z -4.49) | 0.2629 vs 0.2500 (-0.0129): does not beat, significantly worse (DM z -5.98) |
| logreg_lags | direction/ece_pos | 0.0930 vs 0.0150 (-0.0781): does not beat, significantly worse (boot z -6.92) | 0.0646 vs 0.0140 (-0.0506): does not beat, significantly worse (boot z -4.85) | 0.0886 vs 0.0146 (-0.0740): does not beat, significantly worse (boot z -6.72) |
| logreg_lags | direction/acc | 0.4988 vs 0.5172 (-0.0184): does not beat, noise (DM z -1.66) | 0.5035 vs 0.5176 (-0.0142): does not beat, noise (DM z -1.38) | 0.4939 vs 0.5185 (-0.0246): does not beat, significantly worse (DM z -2.05) |
| logreg_lags | direction/bal_acc | 0.4971 vs 0.5093 (-0.0122): does not beat, noise (boot z -1.04) | 0.5047 vs 0.5105 (-0.0058): does not beat, noise (boot z -0.60) | 0.4951 vs 0.5102 (-0.0151): does not beat, noise (boot z -1.35) |
| class_prior | direction/mcc | -0.0058 vs 0.0000 (-0.0058): does not beat, noise (boot z -0.36) | 0.0094 vs 0.0000 (+0.0094): beats, noise (boot z +0.50) | -0.0097 vs 0.0000 (-0.0097): does not beat, noise (boot z -0.55) |
| class_prior | direction/auc | 0.5012 vs 0.5000 (+0.0012): beats, noise (boot z +0.12) | 0.5060 vs 0.5000 (+0.0060): beats, noise (boot z +0.49) | 0.4969 vs 0.5000 (-0.0031): does not beat, noise (boot z -0.26) |
| class_prior | direction/brier | 0.2648 vs 0.2496 (-0.0152): does not beat, significantly worse (DM z -6.65) | 0.2576 vs 0.2496 (-0.0080): does not beat, significantly worse (DM z -3.99) | 0.2629 vs 0.2496 (-0.0133): does not beat, significantly worse (DM z -5.75) |
| class_prior | direction/ece_pos | 0.0930 vs 0.0122 (-0.0808): does not beat, significantly worse (boot z -6.73) | 0.0646 vs 0.0103 (-0.0543): does not beat, significantly worse (boot z -4.46) | 0.0886 vs 0.0109 (-0.0777): does not beat, significantly worse (boot z -6.28) |
| class_prior | direction/acc | 0.4988 vs 0.5230 (-0.0242): does not beat, significantly worse (DM z -1.97) | 0.5035 vs 0.5219 (-0.0185): does not beat, noise (DM z -1.18) | 0.4939 vs 0.5231 (-0.0292): does not beat, noise (DM z -1.75) |
| class_prior | direction/bal_acc | 0.4971 vs 0.5000 (-0.0029): does not beat, noise (boot z -0.36) | 0.5047 vs 0.5000 (+0.0047): beats, noise (boot z +0.50) | 0.4951 vs 0.5000 (-0.0049): does not beat, noise (boot z -0.55) |
| zero_delta | delta/rmse | 135.35 vs 135.21 (-0.14, -0.11%): does not beat, noise (DM z -0.78) | 166.75 vs 166.70 (-0.05, -0.03%): does not beat, noise (DM z -0.38) | 192.14 vs 192.12 (-0.02, -0.01%): does not beat, noise (DM z -0.50) |
| zero_delta | delta/mae | 82.11 vs 82.02 (-0.09, -0.11%): does not beat, noise (DM z -1.30) | 100.62 vs 100.58 (-0.04, -0.04%): does not beat, noise (DM z -0.75) | 116.71 vs 116.69 (-0.02, -0.01%): does not beat, noise (DM z -0.98) |
| mean_delta | delta/rmse | 135.35 vs 135.18 (-0.17, -0.13%): does not beat, noise (DM z -0.91) | 166.75 vs 166.65 (-0.09, -0.06%): does not beat, noise (DM z -0.73) | 192.14 vs 192.04 (-0.09, -0.05%): does not beat, noise (DM z -1.49) |
| mean_delta | delta/mae | 82.11 vs 82.00 (-0.11, -0.14%): does not beat, noise (DM z -1.52) | 100.62 vs 100.55 (-0.07, -0.07%): does not beat, noise (DM z -1.15) | 116.71 vs 116.65 (-0.06, -0.05%): does not beat, noise (DM z -1.43) |
| const_var | variance/crps | 60.29 vs 63.20 (+2.91, +4.61%): beats (DM z +12.53) | 74.42 vs 77.66 (+3.24, +4.17%): beats (DM z +10.74) | 86.21 vs 90.10 (+3.89, +4.32%): beats (DM z +9.74) |
| const_var | variance/nll | 6.0820 vs 6.6403 (+0.5583): beats (DM z +3.62) | 6.3424 vs 6.8650 (+0.5225): beats (DM z +3.37) | 6.4792 vs 7.0192 (+0.5400): beats (DM z +3.35) |
| const_var | variance/pit_ks | 0.0276 vs 0.0393 (+0.0117): beats (boot z +2.61) | 0.0333 vs 0.0425 (+0.0093): beats (boot z +2.00) | 0.0339 vs 0.0458 (+0.0119): beats (boot z +2.73) |
| const_var | variance/corr_var_err2_spearman | 0.4422 vs 0.0000 (+0.4422): beats (boot z +22.94) | 0.4232 vs 0.0000 (+0.4232): beats (boot z +20.23) | 0.4248 vs 0.0000 (+0.4248): beats (boot z +18.80) |

## Backtest (costs included)

- n_trades: 1358
- total_return: -0.1076
- sharpe_net: -5.5204
- sharpe_gross: -5.5204
- sortino: -7.6226
- max_drawdown: 0.1521
- hit_rate: 0.5442
- hit_rate_gross: 0.5442
- profit_factor: 0.9085
- avg_hold_bars: 8.7194
- exposure: 0.4855
- turnover: 2638.4139
- fees_paid: 0.0000
- traded_notional: 26385826.3667
- breakeven_cost_bps: -0.8154
- gross_edge_per_trade_bps: -0.8090
- costs_paid: 0.0000
- gross_pnl: -1075.6990
- net_pnl: -1075.6990

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -35 (TimeSeriesSplit fold 6, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-02-14T08:48:00 .. 2024-03-02T07:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.4659, long_above 0.5886, short_below 0.4260, median 0.4998. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -10.76% | -5.52 | +15.21% | 1358 |
| buy and hold | +25.30% | +9.18 | +6.96% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -13.92% .. +13.09%) | -0.80% | -0.44 | | |

The random null enters at the strategy's rate (0.1082 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 9% of its seeds on net return, 14% on net Sharpe and 9% on gross return.
