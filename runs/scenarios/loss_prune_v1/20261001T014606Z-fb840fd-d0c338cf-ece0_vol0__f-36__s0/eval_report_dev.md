# Evaluation report - dev split - run `20261001T014606Z-fb840fd-d0c338cf-ece0_vol0__f-36__s0`

n = 24390 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 16483 | 17728 | 18448 |
| n_eff of the scored moves (n scored // bars ahead) | 1648 | 1181 | 922 |
| true up-rate | 0.5096 | 0.5110 | 0.5149 |
| calls up (predicted up-rate) | 0.4956 | 0.5333 | 0.6030 |
| accuracy | 0.5194 | 0.5256 | 0.5211 |
| balanced accuracy | 0.5195 | 0.5248 | 0.5181 |
| precision (up) | 0.5292 | 0.5343 | 0.5298 |
| recall / sensitivity (up) | 0.5147 | 0.5576 | 0.6206 |
| specificity (down) | 0.5242 | 0.4921 | 0.4156 |
| F1 (up) | 0.5218 | 0.5457 | 0.5716 |
| MCC | 0.0389 | 0.0498 | 0.0370 |
| AUC | 0.5262 | 0.5343 | 0.5264 |
| Brier | 0.2549 | 0.2497 | 0.2529 |
| ECE (positive class) | 0.0497 | 0.0141 | 0.0357 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0096 | 0.0110 | 0.0149 |
| TP / FP / TN / FN | 4323 / 3846 / 4238 / 4076 | 5051 / 4403 / 4266 / 4008 | 5894 / 5230 / 3720 / 3604 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | 0.7290 | 0.8411 |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | 0.0279 | 0.0377 |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | 0.5407 | 0.5391 |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | 0.2495 | 0.2493 |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | 0.0118 | 0.0123 |
| Gaussian readout of the raw heads: calls up | 0.7403 | 0.7290 | 0.8411 |
| Gaussian readout of the raw heads: MCC | 0.0238 | 0.0279 | 0.0377 |
| Gaussian readout of the raw heads: AUC | 0.5118 | 0.5407 | 0.5391 |
| Gaussian readout of the raw heads: Brier | 0.2500 | 0.2488 | 0.2495 |
| Gaussian readout of the raw heads: ECE | 0.0057 | 0.0152 | 0.0214 |

beta = 0 for h0: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 76.07 | 92.55 | 105.97 |
| RMSE ($), raw heads | 75.98 | 92.85 | 106.76 |
| RMSE ($), zero prediction | 76.07 | 92.59 | 105.96 |
| MAE ($), served | 52.53 | 64.01 | 73.23 |
| MAE ($), raw heads | 52.53 | 64.04 | 73.52 |
| MAE ($), zero prediction | 52.53 | 64.07 | 73.33 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | 0.0009 | -0.0003 |
| skill vs zero, raw heads | 0.0026 | -0.0058 | -0.0151 |
| EV, served | n/a (beta = 0: served delta is 0) | 0.0005 | -0.0014 |
| EV, raw heads | 0.0014 | -0.0074 | -0.0180 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0390 | 0.0226 | -0.0057 |
| corr, Spearman, raw heads | 0.0267 | 0.0487 | 0.0499 |
| mean predicted ($), served | 0.00 | 0.38 | 1.32 |
| mean predicted ($), raw heads | 1.75 | 2.18 | 5.20 |
| mean realised ($) | 2.79 | 4.20 | 5.63 |
| share predicted up, raw heads | 0.7399 | 0.7467 | 0.8554 |
| share realised up | 0.5013 | 0.5012 | 0.5102 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.1737 | 0.2539 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 38.36 | 46.94 | 53.80 |
| CRPSS vs constant variance | 0.0291 | 0.0257 | 0.0255 |
| NLL | 5.6210 | 5.8363 | 5.9693 |
| PIT KS | 0.0407 | 0.0354 | 0.0409 |
| var / err^2 Spearman | 0.3161 | 0.3044 | 0.2986 |
| coverage of the 90% interval | 0.8871 | 0.8875 | 0.8803 |
| width of the 90% interval ($) | 219.01 | 270.01 | 305.03 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0121 | [-0.0123, 0.0360] | NOISE |
| h1 | 0.0264 | [0.0030, 0.0515] | WORKS |
| h2 | 0.0172 | [-0.0068, 0.0440] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.174 / h2 0.254) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6913 | n/a (beta = 0: served delta is 0) | 0.6105 |
| abs(d h1) <= abs(d h2) | 0.8836 | 0.9469 | 0.5875 |
| full chain h0 <= h1 <= h2 | 0.5920 | n/a (beta = 0: served delta is 0) | 0.3272 |

beta = 0 for h0: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5643 | 0.5957 | 0.7019 | 0.2693 |
| expected if the two signs were independent | 0.4934 | 0.5184 | 0.5797 | 0.1871 |

- P(up) unanimity (all three horizons call the same side): 0.3999

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0389 vs 0.0600 (-0.0210): does not beat, noise (boot z -0.96) | 0.0498 vs 0.0619 (-0.0121): does not beat, noise (boot z -0.85) | 0.0370 vs 0.0325 (+0.0044): beats, noise (boot z +0.22) |
| logreg_lags | direction/auc | 0.5262 vs 0.5395 (-0.0133): does not beat, noise (boot z -1.08) | 0.5343 vs 0.5444 (-0.0101): does not beat, noise (boot z -1.37) | 0.5264 vs 0.5410 (-0.0145): does not beat, noise (boot z -1.36) |
| logreg_lags | direction/brier | 0.2549 vs 0.2493 (-0.0056): does not beat, significantly worse (DM z -3.12) | 0.2497 vs 0.2493 (-0.0004): does not beat, noise (DM z -0.50) | 0.2529 vs 0.2493 (-0.0037): does not beat, significantly worse (DM z -3.36) |
| logreg_lags | direction/ece_pos | 0.0497 vs 0.0118 (-0.0379): does not beat, significantly worse (boot z -3.23) | 0.0141 vs 0.0108 (-0.0033): does not beat, noise (boot z -0.35) | 0.0357 vs 0.0073 (-0.0283): does not beat, significantly worse (boot z -3.18) |
| logreg_lags | direction/acc | 0.5194 vs 0.5315 (-0.0121): does not beat, noise (DM z -1.10) | 0.5256 vs 0.5327 (-0.0072): does not beat, noise (DM z -0.98) | 0.5211 vs 0.5198 (+0.0013): beats, noise (DM z +0.14) |
| logreg_lags | direction/bal_acc | 0.5195 vs 0.5289 (-0.0094): does not beat, noise (boot z -0.88) | 0.5248 vs 0.5299 (-0.0051): does not beat, noise (boot z -0.73) | 0.5181 vs 0.5156 (+0.0025): beats, noise (boot z +0.26) |
| class_prior | direction/mcc | 0.0389 vs 0.0000 (+0.0389): beats (boot z +2.45) | 0.0498 vs 0.0000 (+0.0498): beats (boot z +2.68) | 0.0370 vs 0.0000 (+0.0370): beats (boot z +2.43) |
| class_prior | direction/auc | 0.5262 vs 0.5000 (+0.0262): beats (boot z +2.69) | 0.5343 vs 0.5000 (+0.0343): beats (boot z +2.97) | 0.5264 vs 0.5000 (+0.0264): beats (boot z +2.53) |
| class_prior | direction/brier | 0.2549 vs 0.2499 (-0.0050): does not beat, significantly worse (DM z -2.93) | 0.2497 vs 0.2499 (+0.0002): beats, noise (DM z +0.22) | 0.2529 vs 0.2498 (-0.0031): does not beat, significantly worse (DM z -2.07) |
| class_prior | direction/ece_pos | 0.0497 vs 0.0001 (-0.0496): does not beat, significantly worse (boot z -4.88) | 0.0141 vs 0.0002 (-0.0139): does not beat, significantly worse (boot z -2.08) | 0.0357 vs 0.0038 (-0.0319): does not beat, significantly worse (boot z -3.16) |
| class_prior | direction/acc | 0.5194 vs 0.5096 (+0.0098): beats, noise (DM z +0.75) | 0.5256 vs 0.5110 (+0.0146): beats, noise (DM z +1.01) | 0.5211 vs 0.5149 (+0.0063): beats, noise (DM z +0.47) |
| class_prior | direction/bal_acc | 0.5195 vs 0.5000 (+0.0195): beats (boot z +2.45) | 0.5248 vs 0.5000 (+0.0248): beats (boot z +2.68) | 0.5181 vs 0.5000 (+0.0181): beats (boot z +2.43) |
| zero_delta | delta/rmse | 76.07 vs 76.07 (+0.00, +0.00%): does not beat | 92.55 vs 92.59 (+0.04, +0.04%): beats, noise (DM z +0.72) | 105.97 vs 105.96 (-0.01, -0.01%): does not beat, noise (DM z -0.12) |
| zero_delta | delta/mae | 52.53 vs 52.53 (+0.00, +0.00%): does not beat | 64.01 vs 64.07 (+0.06, +0.09%): beats (DM z +2.00) | 73.23 vs 73.33 (+0.10, +0.13%): beats, noise (DM z +1.39) |
| mean_delta | delta/rmse | 76.07 vs 76.05 (-0.03, -0.04%): does not beat, noise (DM z -1.68) | 92.55 vs 92.53 (-0.01, -0.01%): does not beat, noise (DM z -0.21) | 105.97 vs 105.88 (-0.10, -0.09%): does not beat, noise (DM z -0.83) |
| mean_delta | delta/mae | 52.53 vs 52.53 (-0.00, -0.00%): does not beat, noise (DM z -0.16) | 64.01 vs 64.07 (+0.06, +0.09%): beats, noise (DM z +1.68) | 73.23 vs 73.30 (+0.07, +0.09%): beats, noise (DM z +1.01) |
| const_var | variance/crps | 38.36 vs 39.51 (+1.15, +2.91%): beats (DM z +10.55) | 46.94 vs 48.18 (+1.24, +2.57%): beats (DM z +8.08) | 53.80 vs 55.21 (+1.41, +2.55%): beats (DM z +7.31) |
| const_var | variance/nll | 5.6210 vs 5.7555 (+0.1345): beats (DM z +6.69) | 5.8363 vs 5.9518 (+0.1155): beats (DM z +5.68) | 5.9693 vs 6.0862 (+0.1169): beats (DM z +5.01) |
| const_var | variance/pit_ks | 0.0407 vs 0.0612 (+0.0205): beats (boot z +4.41) | 0.0354 vs 0.0602 (+0.0248): beats (boot z +5.20) | 0.0409 vs 0.0622 (+0.0213): beats (boot z +5.47) |
| const_var | variance/corr_var_err2_spearman | 0.3161 vs 0.0000 (+0.3161): beats (boot z +15.30) | 0.3044 vs 0.0000 (+0.3044): beats (boot z +13.40) | 0.2986 vs 0.0000 (+0.2986): beats (boot z +12.25) |

## Backtest (costs included)

- n_trades: 765
- total_return: 0.0148
- sharpe_net: 1.2911
- sharpe_gross: 1.2911
- sortino: 1.8469
- max_drawdown: 0.0871
- hit_rate: 0.5556
- hit_rate_gross: 0.5556
- profit_factor: 1.0254
- avg_hold_bars: 10.2458
- exposure: 0.3214
- turnover: 1548.0398
- fees_paid: 0.0000
- traded_notional: 15481584.7102
- breakeven_cost_bps: 0.1906
- gross_edge_per_trade_bps: 0.2125
- costs_paid: 0.0000
- gross_pnl: 147.5186
- net_pnl: 147.5186

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -36 (TimeSeriesSplit fold 5, 39 usable folds); this report scores the fold's out-of-sample block: 24390 sequences, 2024-01-28T10:18:00 .. 2024-02-14T08:47:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.7031, long_above 0.5845, short_below 0.4453, median 0.5031. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 minutes per year at 1.0-minute bars.

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +1.48% | +1.29 | +8.71% | 765 |
| buy and hold | +15.72% | +8.30 | +4.49% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -6.75% .. +8.22%) | +0.31% | +0.31 | | |

The random null enters at the strategy's rate (0.0462 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 59% of its seeds on net return, 59% on net Sharpe and 59% on gross return.
