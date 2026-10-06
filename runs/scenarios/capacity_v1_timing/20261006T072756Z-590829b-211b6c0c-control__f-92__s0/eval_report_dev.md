# Evaluation report - dev split - run `20261006T072756Z-590829b-211b6c0c-control__f-92__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 9392 | 10127 | 10722 |
| n_eff of the scored moves (n scored // bars ahead) | 939 | 675 | 536 |
| true up-rate | 0.4856 | 0.4794 | 0.4808 |
| calls up (predicted up-rate) | 0.5867 | 0.5229 | 0.6461 |
| accuracy | 0.4782 | 0.4948 | 0.4916 |
| balanced accuracy | 0.4806 | 0.4957 | 0.4972 |
| precision (up) | 0.4691 | 0.4754 | 0.4786 |
| recall / sensitivity (up) | 0.5668 | 0.5184 | 0.6433 |
| specificity (down) | 0.3945 | 0.4731 | 0.3512 |
| F1 (up) | 0.5134 | 0.4960 | 0.5489 |
| MCC | -0.0393 | -0.0085 | -0.0058 |
| AUC | 0.4685 | 0.4988 | 0.4980 |
| Brier | 0.2764 | 0.2557 | 0.2611 |
| ECE (positive class) | 0.1038 | 0.0572 | 0.0812 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0144 | 0.0206 | 0.0192 |
| TP / FP / TN / FN | 2585 / 2925 / 1906 / 1976 | 2517 / 2778 / 2494 / 2338 | 3316 / 3612 / 1955 / 1839 |
| Gaussian readout: calls up | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6973 | 0.6087 | 0.3934 |
| Gaussian readout of the raw heads: MCC | -0.0377 | -0.0191 | 0.0596 |
| Gaussian readout of the raw heads: AUC | 0.4689 | 0.4935 | 0.5315 |
| Gaussian readout of the raw heads: Brier | 0.2536 | 0.2533 | 0.2492 |
| Gaussian readout of the raw heads: ECE | 0.0537 | 0.0483 | 0.0083 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 43.32 | 54.31 | 63.40 |
| RMSE ($), raw heads | 43.37 | 54.40 | 63.49 |
| RMSE ($), zero prediction | 43.32 | 54.31 | 63.40 |
| MAE ($), served | 26.28 | 31.98 | 36.80 |
| MAE ($), raw heads | 26.42 | 32.11 | 36.74 |
| MAE ($), zero prediction | 26.28 | 31.98 | 36.80 |
| skill vs zero (1 - MSE / MSE of 0), served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0021 | -0.0032 | -0.0027 |
| EV, served | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0007 | -0.0019 | -0.0038 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0087 | -0.0019 | -0.0134 |
| corr, Spearman, raw heads | -0.0462 | -0.0023 | 0.0422 |
| mean predicted ($), served | 0.00 | 0.00 | 0.00 |
| mean predicted ($), raw heads | 0.75 | 0.77 | -0.90 |
| mean realised ($) | -1.39 | -2.08 | -2.78 |
| share predicted up, raw heads | 0.7049 | 0.6559 | 0.3780 |
| share realised up | 0.4734 | 0.4754 | 0.4770 |
| shrink beta (served = beta x raw, fit on cal) | 0.0000 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h0, h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 19.93 | 24.30 | 28.08 |
| CRPSS vs constant variance | 0.0304 | 0.0313 | 0.0265 |
| NLL | 5.4379 | 5.5673 | 5.8891 |
| PIT KS | 0.0367 | 0.0298 | 0.0340 |
| var / err^2 Spearman | 0.2886 | 0.3044 | 0.2723 |
| coverage of the 90% interval | 0.9029 | 0.8991 | 0.8971 |
| width of the 90% interval ($) | 120.16 | 145.17 | 166.70 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0421 | [-0.0730, -0.0153] | INVERTED |
| h1 | 0.0120 | [-0.0120, 0.0449] | NOISE |
| h2 | -0.0081 | [-0.0358, 0.0170] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.000 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6484 | n/a (beta = 0: served delta is 0) | 0.5983 |
| abs(d h1) <= abs(d h2) | 0.5510 | n/a (beta = 0: served delta is 0) | 0.5786 |
| full chain h0 <= h1 <= h2 | 0.2780 | n/a (beta = 0: served delta is 0) | 0.3054 |

beta = 0 for h0, h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

No served delta varies, so there is no served ordering: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5697 | 0.5068 | 0.5344 | 0.1341 |
| expected if the two signs were independent | 0.5371 | 0.5086 | 0.4653 | 0.1101 |

- P(up) unanimity (all three horizons call the same side): 0.3549

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0393 vs 0.0646 (-0.1039): does not beat, significantly worse (boot z -3.35) | -0.0085 vs 0.0750 (-0.0835): does not beat, significantly worse (boot z -3.25) | -0.0058 vs 0.0755 (-0.0813): does not beat, significantly worse (boot z -3.34) |
| logreg_lags | direction/auc | 0.4685 vs 0.5596 (-0.0911): does not beat, significantly worse (boot z -4.38) | 0.4988 vs 0.5695 (-0.0707): does not beat, significantly worse (boot z -4.21) | 0.4980 vs 0.5730 (-0.0749): does not beat, significantly worse (boot z -4.84) |
| logreg_lags | direction/brier | 0.2764 vs 0.2488 (-0.0276): does not beat, significantly worse (DM z -6.42) | 0.2557 vs 0.2491 (-0.0066): does not beat, significantly worse (DM z -3.42) | 0.2611 vs 0.2489 (-0.0122): does not beat, significantly worse (DM z -4.46) |
| logreg_lags | direction/ece_pos | 0.1038 vs 0.0315 (-0.0724): does not beat, significantly worse (boot z -5.32) | 0.0572 vs 0.0409 (-0.0164): does not beat, noise (boot z -1.11) | 0.0812 vs 0.0445 (-0.0366): does not beat, significantly worse (boot z -2.68) |
| logreg_lags | direction/acc | 0.4782 vs 0.5244 (-0.0462): does not beat, significantly worse (DM z -3.12) | 0.4948 vs 0.5260 (-0.0312): does not beat, significantly worse (DM z -2.22) | 0.4916 vs 0.5237 (-0.0321): does not beat, significantly worse (DM z -2.61) |
| logreg_lags | direction/bal_acc | 0.4806 vs 0.5299 (-0.0492): does not beat, significantly worse (boot z -3.30) | 0.4957 vs 0.5344 (-0.0386): does not beat, significantly worse (boot z -3.16) | 0.4972 vs 0.5330 (-0.0358): does not beat, significantly worse (boot z -3.26) |
| class_prior | direction/mcc | -0.0393 vs 0.0000 (-0.0393): does not beat, noise (boot z -1.78) | -0.0085 vs 0.0000 (-0.0085): does not beat, noise (boot z -0.45) | -0.0058 vs 0.0000 (-0.0058): does not beat, noise (boot z -0.33) |
| class_prior | direction/auc | 0.4685 vs 0.5000 (-0.0315): does not beat, significantly worse (boot z -2.29) | 0.4988 vs 0.5000 (-0.0012): does not beat, noise (boot z -0.10) | 0.4980 vs 0.5000 (-0.0020): does not beat, noise (boot z -0.17) |
| class_prior | direction/brier | 0.2764 vs 0.2505 (-0.0259): does not beat, significantly worse (DM z -7.25) | 0.2557 vs 0.2506 (-0.0051): does not beat, significantly worse (DM z -2.93) | 0.2611 vs 0.2507 (-0.0104): does not beat, significantly worse (DM z -4.75) |
| class_prior | direction/ece_pos | 0.1038 vs 0.0260 (-0.0778): does not beat, significantly worse (boot z -5.78) | 0.0572 vs 0.0322 (-0.0250): does not beat, noise (boot z -1.75) | 0.0812 vs 0.0332 (-0.0480): does not beat, significantly worse (boot z -3.57) |
| class_prior | direction/acc | 0.4782 vs 0.4856 (-0.0075): does not beat, noise (DM z -0.52) | 0.4948 vs 0.4794 (+0.0154): beats, noise (DM z +0.90) | 0.4916 vs 0.4808 (+0.0108): beats, noise (DM z +0.72) |
| class_prior | direction/bal_acc | 0.4806 vs 0.5000 (-0.0194): does not beat, noise (boot z -1.78) | 0.4957 vs 0.5000 (-0.0043): does not beat, noise (boot z -0.45) | 0.4972 vs 0.5000 (-0.0028): does not beat, noise (boot z -0.33) |
| zero_delta | delta/rmse | 43.32 vs 43.32 (+0.00, +0.00%): does not beat | 54.31 vs 54.31 (+0.00, +0.00%): does not beat | 63.40 vs 63.40 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 26.28 vs 26.28 (+0.00, +0.00%): does not beat | 31.98 vs 31.98 (+0.00, +0.00%): does not beat | 36.80 vs 36.80 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 43.32 vs 43.34 (+0.01, +0.03%): beats, noise (DM z +1.44) | 54.31 vs 54.33 (+0.02, +0.04%): beats, noise (DM z +1.41) | 63.40 vs 63.43 (+0.03, +0.05%): beats, noise (DM z +1.40) |
| mean_delta | delta/mae | 26.28 vs 26.29 (+0.02, +0.07%): beats (DM z +2.85) | 31.98 vs 32.01 (+0.02, +0.08%): beats (DM z +2.13) | 36.80 vs 36.83 (+0.03, +0.08%): beats, noise (DM z +1.71) |
| const_var | variance/crps | 19.93 vs 20.56 (+0.63, +3.04%): beats (DM z +10.23) | 24.30 vs 25.09 (+0.79, +3.13%): beats (DM z +8.26) | 28.08 vs 28.85 (+0.76, +2.65%): beats (DM z +6.63) |
| const_var | variance/nll | 5.4379 vs 6.2346 (+0.7967): beats (DM z +4.70) | 5.5673 vs 6.5179 (+0.9506): beats (DM z +3.65) | 5.8891 vs 6.6968 (+0.8077): beats (DM z +3.68) |
| const_var | variance/pit_ks | 0.0367 vs 0.0828 (+0.0461): beats (boot z +7.58) | 0.0298 vs 0.0747 (+0.0449): beats (boot z +4.76) | 0.0340 vs 0.0725 (+0.0385): beats (boot z +5.05) |
| const_var | variance/corr_var_err2_spearman | 0.2886 vs 0.0000 (+0.2886): beats (boot z +10.19) | 0.3044 vs 0.0000 (+0.3044): beats (boot z +9.88) | 0.2723 vs 0.0000 (+0.2723): beats (boot z +8.00) |

## Backtest (costs included)

- n_trades: 596
- total_return: 0.0057
- sharpe_net: 0.7990
- sharpe_gross: 0.7990
- sortino: 1.2456
- max_drawdown: 0.0577
- hit_rate: 0.4530
- hit_rate_gross: 0.4530
- profit_factor: 1.0142
- avg_hold_bars: 8.0906
- exposure: 0.3248
- turnover: 1194.4825
- fees_paid: 0.0000
- traded_notional: 11944585.4892
- breakeven_cost_bps: 0.0957
- gross_edge_per_trade_bps: 0.1278
- costs_paid: 0.0000
- gross_pnl: 57.1662
- net_pnl: 57.1662

## Training health

2 epoch(s). Pre-clip gradient norm maximum over the run: main 43.5957, indicator 48.1500 (clip 20).
Clipped steps over the run: main 1.0000, indicator 4.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 43.5957 / 29.1370 | 1.0000 / 1.0000 | 1.4% / 1.4% | 0.0000 | 3512.0000 / 4092.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 12.6872 / 48.1500 | 0.0000 / 3.0000 | 0.0% / 4.2% | 0.0000 | 3452.0000 / 4067.0000 / 4454.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=1.268 (corr skip/tower=-0.761), h1=0.946 (corr skip/tower=-0.179), h2=0.930 (corr skip/tower=-0.580).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -92 (TimeSeriesSplit fold 9, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-23T08:46:00 .. 2023-03-05T16:16:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.656, long_above 0.5784, short_below 0.4457, median 0.5113. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +0.57% | +0.80 | +5.77% | 596 |
| buy and hold | -8.52% | -7.39 | +9.73% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -5.77% .. +8.40%) | +0.10% | +0.13 | | |

The random null enters at the strategy's rate (0.0594 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 56% of its seeds on net return, 53% on net Sharpe and 56% on gross return.
