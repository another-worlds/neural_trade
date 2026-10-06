# Evaluation report - dev split - run `20261006T073538Z-590829b-e6f31e0d-gru_small__f-92__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 9392 | 10127 | 10722 |
| n_eff of the scored moves (n scored // bars ahead) | 939 | 675 | 536 |
| true up-rate | 0.4856 | 0.4794 | 0.4808 |
| calls up (predicted up-rate) | 0.3968 | 0.3430 | 0.4554 |
| accuracy | 0.4821 | 0.5099 | 0.5029 |
| balanced accuracy | 0.4791 | 0.5035 | 0.5012 |
| precision (up) | 0.4594 | 0.4845 | 0.4821 |
| recall / sensitivity (up) | 0.3754 | 0.3467 | 0.4566 |
| specificity (down) | 0.5829 | 0.6603 | 0.5457 |
| F1 (up) | 0.4131 | 0.4041 | 0.4690 |
| MCC | -0.0426 | 0.0073 | 0.0024 |
| AUC | 0.4567 | 0.5045 | 0.5067 |
| Brier | 0.2706 | 0.2604 | 0.2605 |
| ECE (positive class) | 0.0988 | 0.0733 | 0.0801 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0144 | 0.0206 | 0.0192 |
| TP / FP / TN / FN | 1712 / 2015 / 2816 / 2849 | 1683 / 1791 / 3481 / 3172 | 2354 / 2529 / 3038 / 2801 |
| Gaussian readout: calls up | 0.6467 | 0.7415 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0126 | 0.0434 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5205 | 0.5369 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2500 | 0.2499 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0234 | 0.0225 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6467 | 0.7415 | 0.5526 |
| Gaussian readout of the raw heads: MCC | 0.0126 | 0.0434 | 0.0279 |
| Gaussian readout of the raw heads: AUC | 0.5205 | 0.5369 | 0.5194 |
| Gaussian readout of the raw heads: Brier | 0.2523 | 0.2540 | 0.2520 |
| Gaussian readout of the raw heads: ECE | 0.0455 | 0.0680 | 0.0393 |

beta = 0 for h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 43.35 | 54.31 | 63.40 |
| RMSE ($), raw heads | 43.43 | 54.50 | 63.28 |
| RMSE ($), zero prediction | 43.32 | 54.31 | 63.40 |
| MAE ($), served | 26.27 | 31.98 | 36.80 |
| MAE ($), raw heads | 26.30 | 32.10 | 36.85 |
| MAE ($), zero prediction | 26.28 | 31.98 | 36.80 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0014 | -0.0001 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0051 | -0.0069 | 0.0038 |
| EV, served | -0.0011 | 0.0000 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0040 | -0.0025 | 0.0043 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0248 | 0.0109 | 0.0652 |
| corr, Spearman, raw heads | 0.0252 | 0.0301 | 0.0258 |
| mean predicted ($), served | 0.22 | 0.09 | 0.00 |
| mean predicted ($), raw heads | 0.59 | 2.07 | 0.28 |
| mean realised ($) | -1.39 | -2.08 | -2.78 |
| share predicted up, raw heads | 0.6557 | 0.7221 | 0.5430 |
| share realised up | 0.4734 | 0.4754 | 0.4770 |
| shrink beta (served = beta x raw, fit on cal) | 0.3768 | 0.0413 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 20.37 | 24.83 | 28.49 |
| CRPSS vs constant variance | 0.0090 | 0.0102 | 0.0123 |
| NLL | 6.0188 | 6.2099 | 6.2675 |
| PIT KS | 0.0731 | 0.0643 | 0.0551 |
| var / err^2 Spearman | 0.2146 | 0.2227 | 0.2103 |
| coverage of the 90% interval | 0.9031 | 0.8992 | 0.8971 |
| width of the 90% interval ($) | 120.31 | 145.28 | 166.70 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0619 | [-0.0940, -0.0310] | INVERTED |
| h1 | 0.0012 | [-0.0285, 0.0311] | NOISE |
| h2 | 0.0071 | [-0.0227, 0.0380] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.377 / h1 0.041 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6668 | 0.1197 | 0.5983 |
| abs(d h1) <= abs(d h2) | 0.5240 | n/a (beta = 0: served delta is 0) | 0.5786 |
| full chain h0 <= h1 <= h2 | 0.2661 | n/a (beta = 0: served delta is 0) | 0.3054 |

beta = 0 for h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.4680 | 0.4320 | 0.5111 | 0.1066 |
| expected if the two signs were independent | 0.4753 | 0.4258 | 0.4958 | 0.1069 |

- P(up) unanimity (all three horizons call the same side): 0.2763

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0426 vs 0.0646 (-0.1072): does not beat, significantly worse (boot z -3.64) | 0.0073 vs 0.0750 (-0.0677): does not beat, significantly worse (boot z -2.76) | 0.0024 vs 0.0755 (-0.0731): does not beat, significantly worse (boot z -2.50) |
| logreg_lags | direction/auc | 0.4567 vs 0.5596 (-0.1029): does not beat, significantly worse (boot z -4.97) | 0.5045 vs 0.5695 (-0.0651): does not beat, significantly worse (boot z -4.14) | 0.5067 vs 0.5730 (-0.0662): does not beat, significantly worse (boot z -3.55) |
| logreg_lags | direction/brier | 0.2706 vs 0.2488 (-0.0218): does not beat, significantly worse (DM z -6.44) | 0.2604 vs 0.2491 (-0.0113): does not beat, significantly worse (DM z -4.09) | 0.2605 vs 0.2489 (-0.0116): does not beat, significantly worse (DM z -3.68) |
| logreg_lags | direction/ece_pos | 0.0988 vs 0.0315 (-0.0673): does not beat, significantly worse (boot z -4.03) | 0.0733 vs 0.0409 (-0.0325): does not beat, noise (boot z -1.67) | 0.0801 vs 0.0445 (-0.0356): does not beat, noise (boot z -1.89) |
| logreg_lags | direction/acc | 0.4821 vs 0.5244 (-0.0423): does not beat, significantly worse (DM z -2.59) | 0.5099 vs 0.5260 (-0.0161): does not beat, noise (DM z -0.98) | 0.5029 vs 0.5237 (-0.0208): does not beat, noise (DM z -1.17) |
| logreg_lags | direction/bal_acc | 0.4791 vs 0.5299 (-0.0508): does not beat, significantly worse (boot z -3.62) | 0.5035 vs 0.5344 (-0.0309): does not beat, significantly worse (boot z -2.72) | 0.5012 vs 0.5330 (-0.0319): does not beat, significantly worse (boot z -2.34) |
| class_prior | direction/mcc | -0.0426 vs 0.0000 (-0.0426): does not beat, significantly worse (boot z -2.29) | 0.0073 vs 0.0000 (+0.0073): beats, noise (boot z +0.44) | 0.0024 vs 0.0000 (+0.0024): beats, noise (boot z +0.11) |
| class_prior | direction/auc | 0.4567 vs 0.5000 (-0.0433): does not beat, significantly worse (boot z -3.43) | 0.5045 vs 0.5000 (+0.0045): beats, noise (boot z +0.41) | 0.5067 vs 0.5000 (+0.0067): beats, noise (boot z +0.49) |
| class_prior | direction/brier | 0.2706 vs 0.2505 (-0.0201): does not beat, significantly worse (DM z -7.43) | 0.2604 vs 0.2506 (-0.0098): does not beat, significantly worse (DM z -3.16) | 0.2605 vs 0.2507 (-0.0097): does not beat, significantly worse (DM z -2.77) |
| class_prior | direction/ece_pos | 0.0988 vs 0.0260 (-0.0728): does not beat, significantly worse (boot z -4.43) | 0.0733 vs 0.0322 (-0.0411): does not beat, significantly worse (boot z -2.14) | 0.0801 vs 0.0332 (-0.0469): does not beat, significantly worse (boot z -2.49) |
| class_prior | direction/acc | 0.4821 vs 0.4856 (-0.0035): does not beat, noise (DM z -0.20) | 0.5099 vs 0.4794 (+0.0305): beats, noise (DM z +1.42) | 0.5029 vs 0.4808 (+0.0221): beats, noise (DM z +1.00) |
| class_prior | direction/bal_acc | 0.4791 vs 0.5000 (-0.0209): does not beat, significantly worse (boot z -2.29) | 0.5035 vs 0.5000 (+0.0035): beats, noise (boot z +0.44) | 0.5012 vs 0.5000 (+0.0012): beats, noise (boot z +0.11) |
| zero_delta | delta/rmse | 43.35 vs 43.32 (-0.03, -0.07%): does not beat, noise (DM z -1.40) | 54.31 vs 54.31 (-0.00, -0.00%): does not beat, noise (DM z -0.54) | 63.40 vs 63.40 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 26.27 vs 26.28 (+0.00, +0.01%): beats, noise (DM z +0.13) | 31.98 vs 31.98 (+0.00, +0.00%): beats, noise (DM z +0.22) | 36.80 vs 36.80 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 43.35 vs 43.34 (-0.02, -0.04%): does not beat, noise (DM z -0.99) | 54.31 vs 54.33 (+0.02, +0.04%): beats, noise (DM z +1.52) | 63.40 vs 63.43 (+0.03, +0.05%): beats, noise (DM z +1.40) |
| mean_delta | delta/mae | 26.27 vs 26.29 (+0.02, +0.07%): beats, noise (DM z +1.70) | 31.98 vs 32.01 (+0.03, +0.08%): beats (DM z +2.49) | 36.80 vs 36.83 (+0.03, +0.08%): beats, noise (DM z +1.71) |
| const_var | variance/crps | 20.37 vs 20.56 (+0.18, +0.90%): beats (DM z +6.36) | 24.83 vs 25.09 (+0.26, +1.02%): beats (DM z +8.24) | 28.49 vs 28.85 (+0.36, +1.23%): beats (DM z +7.56) |
| const_var | variance/nll | 6.0188 vs 6.2346 (+0.2158): beats (DM z +2.99) | 6.2099 vs 6.5179 (+0.3081): beats (DM z +3.26) | 6.2675 vs 6.6968 (+0.4293): beats (DM z +3.04) |
| const_var | variance/pit_ks | 0.0731 vs 0.0828 (+0.0097): beats (boot z +2.77) | 0.0643 vs 0.0747 (+0.0104): beats (boot z +3.96) | 0.0551 vs 0.0725 (+0.0175): beats (boot z +5.61) |
| const_var | variance/corr_var_err2_spearman | 0.2146 vs 0.0000 (+0.2146): beats (boot z +8.24) | 0.2227 vs 0.0000 (+0.2227): beats (boot z +7.98) | 0.2103 vs 0.0000 (+0.2103): beats (boot z +6.91) |

## Backtest (costs included)

- n_trades: 473
- total_return: 0.0039
- sharpe_net: 0.6333
- sharpe_gross: 0.6333
- sortino: 0.9584
- max_drawdown: 0.0435
- hit_rate: 0.3953
- hit_rate_gross: 0.3953
- profit_factor: 1.0120
- avg_hold_bars: 10.0233
- exposure: 0.3192
- turnover: 945.3175
- fees_paid: 0.0000
- traded_notional: 9452684.1758
- breakeven_cost_bps: 0.0827
- gross_edge_per_trade_bps: 0.1049
- costs_paid: 0.0000
- gross_pnl: 39.0925
- net_pnl: 39.0925

## Training health

2 epoch(s). Pre-clip gradient norm maximum over the run: main 80.1248, indicator 51.7079 (clip 20).
Clipped steps over the run: main 4.0000, indicator 2.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 80.1248 / 51.7079 | 4.0000 / 2.0000 | 5.6% / 2.8% | 0.0000 | 3512.0000 / 4092.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 13.5711 / 16.5989 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3452.0000 / 4067.0000 / 4454.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=1.146 (corr skip/tower=-0.656), h1=1.088 (corr skip/tower=-0.527), h2=1.111 (corr skip/tower=-0.584).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -92 (TimeSeriesSplit fold 9, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-23T08:46:00 .. 2023-03-05T16:16:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.07, long_above 0.5347, short_below 0.4062, median 0.4816. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +0.39% | +0.63 | +4.35% | 473 |
| buy and hold | -8.52% | -7.39 | +9.73% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -6.38% .. +6.89%) | +0.01% | +0.01 | | |

The random null enters at the strategy's rate (0.0468 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 54% of its seeds on net return, 54% on net Sharpe and 54% on gross return.
