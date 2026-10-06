# Evaluation report - dev split - run `20261001T002324Z-fb840fd-8c5e4460-ece0__f-37__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 17330 | 18405 | 19141 |
| n_eff of the scored moves (n scored // bars ahead) | 1733 | 1227 | 957 |
| true up-rate | 0.4990 | 0.4981 | 0.4981 |
| calls up (predicted up-rate) | 0.5766 | 0.6333 | 0.6470 |
| accuracy | 0.5036 | 0.5142 | 0.5075 |
| balanced accuracy | 0.5037 | 0.5147 | 0.5081 |
| precision (up) | 0.5023 | 0.5097 | 0.5044 |
| recall / sensitivity (up) | 0.5804 | 0.6481 | 0.6552 |
| specificity (down) | 0.4271 | 0.3814 | 0.3610 |
| F1 (up) | 0.5385 | 0.5706 | 0.5700 |
| MCC | 0.0075 | 0.0306 | 0.0169 |
| AUC | 0.5081 | 0.5211 | 0.5097 |
| Brier | 0.2583 | 0.2519 | 0.2550 |
| ECE (positive class) | 0.0637 | 0.0310 | 0.0479 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0010 | 0.0019 | 0.0019 |
| TP / FP / TN / FN | 5019 / 4974 / 3708 / 3629 | 5941 / 5715 / 3523 / 3226 | 6247 / 6138 / 3468 / 3288 |
| Gaussian readout: calls up | 0.8743 | 0.7437 | 0.6987 |
| Gaussian readout: MCC | -0.0047 | 0.0157 | -0.0070 |
| Gaussian readout: AUC | 0.5085 | 0.5253 | 0.5036 |
| Gaussian readout: Brier | 0.2500 | 0.2498 | 0.2501 |
| Gaussian readout: ECE | 0.0108 | 0.0072 | 0.0102 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 90.09 | 109.84 | 126.43 |
| RMSE ($), raw heads | 90.23 | 113.20 | 129.92 |
| RMSE ($), zero prediction | 90.11 | 109.57 | 126.22 |
| MAE ($), served | 59.34 | 71.97 | 82.68 |
| MAE ($), raw heads | 59.41 | 72.99 | 83.84 |
| MAE ($), zero prediction | 59.37 | 71.99 | 82.66 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0004 | -0.0051 | -0.0033 |
| skill vs zero, raw heads | -0.0027 | -0.0674 | -0.0594 |
| EV, served | 0.0008 | -0.0046 | -0.0031 |
| EV, raw heads | 0.0011 | -0.0621 | -0.0552 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0387 | -0.0331 | -0.0618 |
| corr, Spearman, raw heads | 0.0142 | 0.0254 | 0.0024 |
| mean predicted ($), served | 0.90 | 1.21 | 0.69 |
| mean predicted ($), raw heads | 4.36 | 6.28 | 5.92 |
| mean realised ($) | -1.30 | -1.98 | -2.67 |
| share predicted up, raw heads | 0.8924 | 0.7455 | 0.6919 |
| share realised up | 0.5005 | 0.4945 | 0.4976 |
| shrink beta (served = beta x raw, fit on cal) | 0.2055 | 0.1930 | 0.1164 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 43.18 | 52.44 | 60.29 |
| CRPSS vs constant variance | 0.0421 | 0.0429 | 0.0419 |
| NLL | 5.6932 | 5.8746 | 6.0276 |
| PIT KS | 0.0212 | 0.0231 | 0.0161 |
| var / err^2 Spearman | 0.4071 | 0.4047 | 0.4058 |
| coverage of the 90% interval | 0.9077 | 0.9055 | 0.9065 |
| width of the 90% interval ($) | 258.72 | 317.53 | 362.00 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0133 | [-0.0087, 0.0381] | NOISE |
| h1 | 0.0123 | [-0.0121, 0.0351] | NOISE |
| h2 | 0.0044 | [-0.0155, 0.0271] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.206 / h1 0.193 / h2 0.116) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.5021 | 0.4804 | 0.6066 |
| abs(d h1) <= abs(d h2) | 0.6162 | 0.3968 | 0.5887 |
| full chain h0 <= h1 <= h2 | 0.2351 | 0.0820 | 0.3212 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6023 | 0.7401 | 0.6795 | 0.3407 |
| expected if the two signs were independent | 0.5649 | 0.5631 | 0.5588 | 0.2199 |

- P(up) unanimity (all three horizons call the same side): 0.4182

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0075 vs 0.0245 (-0.0169): does not beat, noise (boot z -0.74) | 0.0306 vs 0.0175 (+0.0131): beats, noise (boot z +0.99) | 0.0169 vs 0.0080 (+0.0089): beats, noise (boot z +0.45) |
| logreg_lags | direction/auc | 0.5081 vs 0.5197 (-0.0117): does not beat, noise (boot z -0.77) | 0.5211 vs 0.5155 (+0.0055): beats, noise (boot z +0.75) | 0.5097 vs 0.5134 (-0.0037): does not beat, noise (boot z -0.31) |
| logreg_lags | direction/brier | 0.2583 vs 0.2505 (-0.0078): does not beat, significantly worse (DM z -3.74) | 0.2519 vs 0.2515 (-0.0004): does not beat, noise (DM z -0.57) | 0.2550 vs 0.2515 (-0.0035): does not beat, significantly worse (DM z -3.11) |
| logreg_lags | direction/ece_pos | 0.0637 vs 0.0173 (-0.0465): does not beat, significantly worse (boot z -4.36) | 0.0310 vs 0.0253 (-0.0057): does not beat, noise (boot z -1.19) | 0.0479 vs 0.0290 (-0.0189): does not beat, significantly worse (boot z -2.31) |
| logreg_lags | direction/acc | 0.5036 vs 0.5106 (-0.0070): does not beat, noise (DM z -0.62) | 0.5142 vs 0.5070 (+0.0072): beats, noise (DM z +1.14) | 0.5075 vs 0.5028 (+0.0047): beats, noise (DM z +0.52) |
| logreg_lags | direction/bal_acc | 0.5037 vs 0.5110 (-0.0073): does not beat, noise (boot z -0.67) | 0.5147 vs 0.5079 (+0.0068): beats, noise (boot z +1.15) | 0.5081 vs 0.5036 (+0.0045): beats, noise (boot z +0.49) |
| class_prior | direction/mcc | 0.0075 vs 0.0000 (+0.0075): beats, noise (boot z +0.50) | 0.0306 vs 0.0000 (+0.0306): beats, noise (boot z +1.80) | 0.0169 vs 0.0000 (+0.0169): beats, noise (boot z +1.18) |
| class_prior | direction/auc | 0.5081 vs 0.5000 (+0.0081): beats, noise (boot z +0.84) | 0.5211 vs 0.5000 (+0.0211): beats (boot z +1.97) | 0.5097 vs 0.5000 (+0.0097): beats, noise (boot z +1.06) |
| class_prior | direction/brier | 0.2583 vs 0.2502 (-0.0081): does not beat, significantly worse (DM z -4.57) | 0.2519 vs 0.2503 (-0.0016): does not beat, noise (DM z -1.48) | 0.2550 vs 0.2502 (-0.0048): does not beat, significantly worse (DM z -3.79) |
| class_prior | direction/ece_pos | 0.0637 vs 0.0133 (-0.0505): does not beat, significantly worse (boot z -4.66) | 0.0310 vs 0.0162 (-0.0148): does not beat, significantly worse (boot z -2.01) | 0.0479 vs 0.0157 (-0.0322): does not beat, significantly worse (boot z -3.33) |
| class_prior | direction/acc | 0.5036 vs 0.4990 (+0.0046): beats, noise (DM z +0.41) | 0.5142 vs 0.4981 (+0.0161): beats, noise (DM z +1.33) | 0.5075 vs 0.4981 (+0.0094): beats, noise (DM z +0.80) |
| class_prior | direction/bal_acc | 0.5037 vs 0.5000 (+0.0037): beats, noise (boot z +0.50) | 0.5147 vs 0.5000 (+0.0147): beats, noise (boot z +1.80) | 0.5081 vs 0.5000 (+0.0081): beats, noise (boot z +1.18) |
| zero_delta | delta/rmse | 90.09 vs 90.11 (+0.02, +0.02%): beats, noise (DM z +0.44) | 109.84 vs 109.57 (-0.28, -0.25%): does not beat, noise (DM z -0.89) | 126.43 vs 126.22 (-0.21, -0.17%): does not beat, noise (DM z -1.11) |
| zero_delta | delta/mae | 59.34 vs 59.37 (+0.03, +0.05%): beats, noise (DM z +1.42) | 71.97 vs 71.99 (+0.01, +0.02%): beats, noise (DM z +0.16) | 82.68 vs 82.66 (-0.02, -0.03%): does not beat, noise (DM z -0.44) |
| mean_delta | delta/rmse | 90.09 vs 90.14 (+0.05, +0.06%): beats, noise (DM z +1.15) | 109.84 vs 109.62 (-0.22, -0.20%): does not beat, noise (DM z -0.75) | 126.43 vs 126.32 (-0.12, -0.09%): does not beat, noise (DM z -0.72) |
| mean_delta | delta/mae | 59.34 vs 59.38 (+0.03, +0.06%): beats, noise (DM z +1.78) | 71.97 vs 72.03 (+0.05, +0.07%): beats, noise (DM z +0.60) | 82.68 vs 82.70 (+0.02, +0.02%): beats, noise (DM z +0.22) |
| const_var | variance/crps | 43.18 vs 45.08 (+1.90, +4.21%): beats (DM z +12.77) | 52.44 vs 54.79 (+2.35, +4.29%): beats (DM z +10.41) | 60.29 vs 62.93 (+2.64, +4.19%): beats (DM z +9.86) |
| const_var | variance/nll | 5.6932 vs 5.9949 (+0.3017): beats (DM z +7.34) | 5.8746 vs 6.1909 (+0.3162): beats (DM z +5.77) | 6.0276 vs 6.3349 (+0.3073): beats (DM z +5.05) |
| const_var | variance/pit_ks | 0.0212 vs 0.0468 (+0.0256): beats (boot z +5.73) | 0.0231 vs 0.0515 (+0.0284): beats (boot z +4.98) | 0.0161 vs 0.0525 (+0.0364): beats (boot z +4.63) |
| const_var | variance/corr_var_err2_spearman | 0.4071 vs 0.0000 (+0.4071): beats (boot z +20.26) | 0.4047 vs 0.0000 (+0.4047): beats (boot z +18.71) | 0.4058 vs 0.0000 (+0.4058): beats (boot z +17.71) |

## Backtest (costs included)

- n_trades: 987
- total_return: -0.0207
- sharpe_net: -1.0440
- sharpe_gross: -1.0440
- sortino: -1.4932
- max_drawdown: 0.0647
- hit_rate: 0.5583
- hit_rate_gross: 0.5583
- profit_factor: 0.9741
- avg_hold_bars: 8.2948
- exposure: 0.3357
- turnover: 1996.8215
- fees_paid: 0.0000
- traded_notional: 19967723.6134
- breakeven_cost_bps: -0.2068
- gross_edge_per_trade_bps: -0.1850
- costs_paid: 0.0000
- gross_pnl: -206.5094
- net_pnl: -206.5094

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -37 (TimeSeriesSplit fold 4, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-01-11T11:48:00 .. 2024-01-28T10:17:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.017, long_above 0.5684, short_below 0.4433, median 0.5109. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -2.07% | -1.04 | +6.47% | 987 |
| buy and hold | -7.01% | -2.90 | +21.43% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -8.55% .. +9.03%) | -0.36% | -0.23 | | |

The random null enters at the strategy's rate (0.0609 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 37% of its seeds on net return, 42% on net Sharpe and 37% on gross return.
