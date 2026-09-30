# Evaluation report - dev split - run `20260930T084838Z-dce15ed-10eeccf2-default__f-3__s1`

n = 46544 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 26329 | 29695 | 31930 |
| n_eff of the scored moves (n scored // bars ahead) | 2632 | 1979 | 1596 |
| true up-rate | 0.5030 | 0.5065 | 0.5063 |
| calls up (predicted up-rate) | 0.7019 | 0.8983 | 0.9211 |
| accuracy | 0.5067 | 0.5056 | 0.5133 |
| balanced accuracy | 0.5055 | 0.5004 | 0.5080 |
| precision (up) | 0.5069 | 0.5067 | 0.5107 |
| recall / sensitivity (up) | 0.7074 | 0.8987 | 0.9290 |
| specificity (down) | 0.3036 | 0.1021 | 0.0870 |
| F1 (up) | 0.5906 | 0.6480 | 0.6590 |
| MCC | 0.0120 | 0.0014 | 0.0296 |
| AUC | 0.5129 | 0.5104 | 0.5159 |
| Brier | 0.2500 | 0.2499 | 0.2498 |
| ECE (positive class) | 0.0123 | 0.0082 | 0.0078 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0030 | 0.0065 | 0.0063 |
| TP / FP / TN / FN | 9368 / 9113 / 3973 / 3875 | 13516 / 13158 / 1497 / 1524 | 15019 / 14392 / 1371 / 1148 |
| Gaussian readout: calls up | 0.6565 | 0.4958 | 0.4743 |
| Gaussian readout: MCC | 0.0206 | 0.0113 | -0.0001 |
| Gaussian readout: AUC | 0.5147 | 0.5058 | 0.4995 |
| Gaussian readout: Brier | 0.2499 | 0.2500 | 0.2501 |
| Gaussian readout: ECE | 0.0026 | 0.0069 | 0.0069 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 151.17 | 184.80 | 212.93 |
| RMSE ($), raw heads | 151.44 | 187.85 | 215.52 |
| RMSE ($), zero prediction | 151.14 | 184.66 | 212.80 |
| MAE ($), served | 101.85 | 125.67 | 145.29 |
| MAE ($), raw heads | 102.01 | 126.70 | 146.70 |
| MAE ($), zero prediction | 101.84 | 125.61 | 145.22 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0004 | -0.0015 | -0.0013 |
| skill vs zero, raw heads | -0.0039 | -0.0349 | -0.0257 |
| EV, served | -0.0005 | -0.0014 | -0.0012 |
| EV, raw heads | -0.0042 | -0.0346 | -0.0253 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0035 | 0.0051 | -0.0122 |
| corr, Spearman, raw heads | 0.0158 | -0.0017 | -0.0122 |
| mean predicted ($), served | 0.63 | -0.26 | -0.31 |
| mean predicted ($), raw heads | 1.61 | -1.13 | -1.84 |
| mean realised ($) | 2.58 | 3.89 | 5.19 |
| share predicted up, raw heads | 0.7064 | 0.5155 | 0.4802 |
| share realised up | 0.4954 | 0.4994 | 0.5014 |
| shrink beta (served = beta x raw, fit on cal) | 0.3928 | 0.2270 | 0.1696 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 74.58 | 91.88 | 105.96 |
| CRPSS vs constant variance | 0.0782 | 0.0736 | 0.0732 |
| NLL | 6.2664 | 6.4790 | 6.6251 |
| PIT KS | 0.0404 | 0.0414 | 0.0478 |
| var / err^2 Spearman | 0.4209 | 0.4183 | 0.4204 |
| coverage of the 90% interval | 0.8872 | 0.8835 | 0.8845 |
| width of the 90% interval ($) | 437.62 | 531.65 | 611.65 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0123 | [-0.0076, 0.0356] | NOISE |
| h1 | 0.0244 | [-0.0006, 0.0468] | NOISE |
| h2 | 0.0012 | [-0.0292, 0.0301] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.393 / h1 0.227 / h2 0.170) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7711 | 0.5438 | 0.6235 |
| abs(d h1) <= abs(d h2) | 0.8528 | 0.6598 | 0.6030 |
| full chain h0 <= h1 <= h2 | 0.6763 | 0.3572 | 0.3489 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7128 | 0.4941 | 0.5023 | 0.3195 |
| expected if the two signs were independent | 0.5965 | 0.5116 | 0.4833 | 0.2770 |

- P(up) unanimity (all three horizons call the same side): 0.5921

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0120 vs 0.0074 (+0.0046): beats, noise (boot z +0.27) | 0.0014 vs 0.0163 (-0.0149): does not beat, noise (boot z -0.85) | 0.0296 vs 0.0186 (+0.0110): beats, noise (boot z +0.59) |
| logreg_lags | direction/auc | 0.5129 vs 0.5121 (+0.0008): beats, noise (boot z +0.08) | 0.5104 vs 0.5163 (-0.0059): does not beat, noise (boot z -0.53) | 0.5159 vs 0.5158 (+0.0001): beats, noise (boot z +0.01) |
| logreg_lags | direction/brier | 0.2500 vs 0.2498 (-0.0001): does not beat, noise (DM z -0.37) | 0.2499 vs 0.2497 (-0.0003): does not beat, noise (DM z -0.83) | 0.2498 vs 0.2497 (-0.0001): does not beat, noise (DM z -0.29) |
| logreg_lags | direction/ece_pos | 0.0123 vs 0.0045 (-0.0078): does not beat, noise (boot z -0.98) | 0.0082 vs 0.0063 (-0.0019): does not beat, noise (boot z -0.20) | 0.0078 vs 0.0058 (-0.0020): does not beat, noise (boot z -0.22) |
| logreg_lags | direction/acc | 0.5067 vs 0.5035 (+0.0032): beats, noise (DM z +0.37) | 0.5056 vs 0.5078 (-0.0022): does not beat, noise (DM z -0.20) | 0.5133 vs 0.5090 (+0.0043): beats, noise (DM z +0.35) |
| logreg_lags | direction/bal_acc | 0.5055 vs 0.5037 (+0.0018): beats, noise (boot z +0.22) | 0.5004 vs 0.5081 (-0.0077): does not beat, noise (boot z -0.98) | 0.5080 vs 0.5093 (-0.0013): does not beat, noise (boot z -0.16) |
| class_prior | direction/mcc | 0.0120 vs 0.0000 (+0.0120): beats, noise (boot z +0.89) | 0.0014 vs 0.0000 (+0.0014): beats, noise (boot z +0.11) | 0.0296 vs 0.0000 (+0.0296): beats (boot z +2.05) |
| class_prior | direction/auc | 0.5129 vs 0.5000 (+0.0129): beats, noise (boot z +1.48) | 0.5104 vs 0.5000 (+0.0104): beats, noise (boot z +1.23) | 0.5159 vs 0.5000 (+0.0159): beats, noise (boot z +1.53) |
| class_prior | direction/brier | 0.2500 vs 0.2500 (+0.0000): beats, noise (DM z +0.11) | 0.2499 vs 0.2500 (+0.0000): beats, noise (DM z +0.12) | 0.2498 vs 0.2500 (+0.0002): beats, noise (DM z +0.76) |
| class_prior | direction/ece_pos | 0.0123 vs 0.0007 (-0.0116): does not beat, noise (boot z -1.65) | 0.0082 vs 0.0032 (-0.0050): does not beat, noise (boot z -0.61) | 0.0078 vs 0.0018 (-0.0060): does not beat, noise (boot z -0.91) |
| class_prior | direction/acc | 0.5067 vs 0.5030 (+0.0037): beats, noise (DM z +0.48) | 0.5056 vs 0.5065 (-0.0009): does not beat, noise (DM z -0.21) | 0.5133 vs 0.5063 (+0.0070): beats, noise (DM z +1.77) |
| class_prior | direction/bal_acc | 0.5055 vs 0.5000 (+0.0055): beats, noise (boot z +0.89) | 0.5004 vs 0.5000 (+0.0004): beats, noise (boot z +0.11) | 0.5080 vs 0.5000 (+0.0080): beats (boot z +2.08) |
| zero_delta | delta/rmse | 151.17 vs 151.14 (-0.03, -0.02%): does not beat, noise (DM z -0.46) | 184.80 vs 184.66 (-0.14, -0.08%): does not beat, noise (DM z -0.97) | 212.93 vs 212.80 (-0.14, -0.07%): does not beat, noise (DM z -1.37) |
| zero_delta | delta/mae | 101.85 vs 101.84 (-0.01, -0.01%): does not beat, noise (DM z -0.26) | 125.67 vs 125.61 (-0.06, -0.05%): does not beat, noise (DM z -0.78) | 145.29 vs 145.22 (-0.07, -0.05%): does not beat, noise (DM z -1.16) |
| mean_delta | delta/rmse | 151.17 vs 151.13 (-0.04, -0.03%): does not beat, noise (DM z -0.58) | 184.80 vs 184.64 (-0.15, -0.08%): does not beat, noise (DM z -1.07) | 212.93 vs 212.77 (-0.16, -0.08%): does not beat, noise (DM z -1.54) |
| mean_delta | delta/mae | 101.85 vs 101.84 (-0.01, -0.01%): does not beat, noise (DM z -0.13) | 125.67 vs 125.61 (-0.06, -0.05%): does not beat, noise (DM z -0.75) | 145.29 vs 145.21 (-0.08, -0.05%): does not beat, noise (DM z -1.13) |
| const_var | variance/crps | 74.58 vs 80.91 (+6.33, +7.82%): beats (DM z +28.28) | 91.88 vs 99.18 (+7.30, +7.36%): beats (DM z +21.44) | 105.96 vs 114.33 (+8.36, +7.32%): beats (DM z +20.36) |
| const_var | variance/nll | 6.2664 vs 6.4734 (+0.2070): beats (DM z +14.00) | 6.4790 vs 6.6734 (+0.1943): beats (DM z +11.13) | 6.6251 vs 6.8153 (+0.1902): beats (DM z +9.42) |
| const_var | variance/pit_ks | 0.0404 vs 0.1172 (+0.0768): beats (boot z +18.54) | 0.0414 vs 0.1152 (+0.0738): beats (boot z +15.77) | 0.0478 vs 0.1158 (+0.0680): beats (boot z +14.66) |
| const_var | variance/corr_var_err2_spearman | 0.4209 vs 0.0000 (+0.4209): beats (boot z +32.44) | 0.4183 vs 0.0000 (+0.4183): beats (boot z +29.67) | 0.4204 vs 0.0000 (+0.4204): beats (boot z +28.80) |

## Backtest (costs included)

- n_trades: 1656
- total_return: -0.9846
- sharpe_net: -145.6194
- sharpe_gross: 3.6412
- sortino: -158.5389
- max_drawdown: 0.9846
- hit_rate: 0.0543
- hit_rate_gross: 0.5254
- profit_factor: 0.0132
- avg_hold_bars: 10.5006
- exposure: 0.3736
- turnover: 772.1813
- fees_paid: 7721.8885
- traded_notional: 7721888.4681
- breakeven_cost_bps: 0.4984
- gross_edge_per_trade_bps: 0.8421
- costs_paid: 10038.4550
- gross_pnl: 192.4441
- net_pnl: -9846.0110

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -3 (TimeSeriesSplit fold 12, 13 usable folds); this report scores the fold's out-of-sample block: 46544 sequences, 2025-06-25T00:26:00 .. 2025-07-27T08:09:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.536, long_above 0.5235, short_below 0.4991, median 0.5114. Costs per side: fee 10.0 bps + half-spread 1.0 bps + slippage 2.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -98.46% | -145.62 | +98.46% | 1656 |
| buy and hold | +11.00% | +4.11 | +6.87% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -98.72% .. -98.28%) | -98.54% | -157.53 | | |

The random null enters at the strategy's rate (0.0568 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 77% of its seeds on net return, 100% on net Sharpe and 89% on gross return.
