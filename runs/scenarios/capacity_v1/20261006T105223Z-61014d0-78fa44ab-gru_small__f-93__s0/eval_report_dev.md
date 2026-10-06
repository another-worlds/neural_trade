# Evaluation report - dev split - run `20261006T105223Z-61014d0-78fa44ab-gru_small__f-93__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10324 | 10946 | 11486 |
| n_eff of the scored moves (n scored // bars ahead) | 1032 | 729 | 574 |
| true up-rate | 0.5120 | 0.5097 | 0.5145 |
| calls up (predicted up-rate) | 0.4332 | 0.3905 | 0.4155 |
| accuracy | 0.5002 | 0.5117 | 0.5079 |
| balanced accuracy | 0.5018 | 0.5138 | 0.5104 |
| precision (up) | 0.5141 | 0.5274 | 0.5270 |
| recall / sensitivity (up) | 0.4349 | 0.4040 | 0.4255 |
| specificity (down) | 0.5687 | 0.6236 | 0.5952 |
| F1 (up) | 0.4712 | 0.4575 | 0.4709 |
| MCC | 0.0036 | 0.0283 | 0.0211 |
| AUC | 0.4984 | 0.5152 | 0.5199 |
| Brier | 0.2683 | 0.2632 | 0.2607 |
| ECE (positive class) | 0.0937 | 0.0814 | 0.0836 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0120 | 0.0097 | 0.0145 |
| TP / FP / TN / FN | 2299 / 2173 / 2865 / 2987 | 2254 / 2020 / 3347 / 3325 | 2515 / 2257 / 3319 / 3395 |
| Gaussian readout: calls up | 0.6652 | 0.4347 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0105 | -0.0063 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5058 | 0.4909 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2500 | 0.2501 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0056 | 0.0097 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.6652 | 0.4347 | 0.4868 |
| Gaussian readout of the raw heads: MCC | 0.0105 | -0.0063 | -0.0261 |
| Gaussian readout of the raw heads: AUC | 0.5058 | 0.4909 | 0.4861 |
| Gaussian readout of the raw heads: Brier | 0.2530 | 0.2576 | 0.2610 |
| Gaussian readout of the raw heads: ECE | 0.0387 | 0.0680 | 0.0871 |

beta = 0 for h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 55.35 | 68.21 | 79.34 |
| RMSE ($), raw heads | 55.46 | 68.43 | 79.69 |
| RMSE ($), zero prediction | 55.38 | 68.23 | 79.34 |
| MAE ($), served | 34.92 | 42.60 | 49.49 |
| MAE ($), raw heads | 35.13 | 43.00 | 50.27 |
| MAE ($), zero prediction | 34.92 | 42.60 | 49.49 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0013 | 0.0006 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0029 | -0.0058 | -0.0088 |
| EV, served | 0.0009 | 0.0005 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0038 | -0.0062 | -0.0098 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0306 | 0.0330 | 0.0294 |
| corr, Spearman, raw heads | 0.0048 | -0.0103 | -0.0126 |
| mean predicted ($), served | 0.46 | 0.03 | 0.00 |
| mean predicted ($), raw heads | 1.79 | 0.34 | 1.10 |
| mean realised ($) | 1.67 | 2.49 | 3.33 |
| share predicted up, raw heads | 0.6801 | 0.4303 | 0.4750 |
| share realised up | 0.4982 | 0.5021 | 0.5025 |
| shrink beta (served = beta x raw, fit on cal) | 0.2562 | 0.0798 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 26.56 | 32.87 | 37.89 |
| CRPSS vs constant variance | 0.0555 | 0.0435 | 0.0504 |
| NLL | 5.5470 | 5.9528 | 6.0112 |
| PIT KS | 0.0596 | 0.0726 | 0.0588 |
| var / err^2 Spearman | 0.3122 | 0.2942 | 0.3038 |
| coverage of the 90% interval | 0.8968 | 0.8935 | 0.8922 |
| width of the 90% interval ($) | 153.82 | 185.53 | 213.30 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0099 | [-0.0337, 0.0130] | NOISE |
| h1 | -0.0027 | [-0.0299, 0.0251] | NOISE |
| h2 | 0.0279 | [0.0014, 0.0578] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.256 / h1 0.080 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6536 | 0.2616 | 0.5973 |
| abs(d h1) <= abs(d h2) | 0.7061 | n/a (beta = 0: served delta is 0) | 0.5845 |
| full chain h0 <= h1 <= h2 | 0.4212 | n/a (beta = 0: served delta is 0) | 0.3105 |

beta = 0 for h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5775 | 0.6039 | 0.6084 | 0.2264 |
| expected if the two signs were independent | 0.4748 | 0.5155 | 0.5047 | 0.1511 |

- P(up) unanimity (all three horizons call the same side): 0.3382

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0036 vs 0.0593 (-0.0557): does not beat, significantly worse (boot z -2.27) | 0.0283 vs 0.0353 (-0.0070): does not beat, noise (boot z -0.24) | 0.0211 vs 0.0287 (-0.0077): does not beat, noise (boot z -0.21) |
| logreg_lags | direction/auc | 0.4984 vs 0.5347 (-0.0363): does not beat, significantly worse (boot z -2.32) | 0.5152 vs 0.5204 (-0.0051): does not beat, noise (boot z -0.25) | 0.5199 vs 0.5116 (+0.0083): beats, noise (boot z +0.34) |
| logreg_lags | direction/brier | 0.2683 vs 0.2515 (-0.0168): does not beat, significantly worse (DM z -5.75) | 0.2632 vs 0.2531 (-0.0102): does not beat, significantly worse (DM z -3.05) | 0.2607 vs 0.2539 (-0.0068): does not beat, noise (DM z -1.69) |
| logreg_lags | direction/ece_pos | 0.0937 vs 0.0169 (-0.0768): does not beat, significantly worse (boot z -6.31) | 0.0814 vs 0.0293 (-0.0521): does not beat, significantly worse (boot z -4.04) | 0.0836 vs 0.0324 (-0.0512): does not beat, significantly worse (boot z -3.26) |
| logreg_lags | direction/acc | 0.5002 vs 0.5302 (-0.0300): does not beat, significantly worse (DM z -2.53) | 0.5117 vs 0.5182 (-0.0065): does not beat, noise (DM z -0.45) | 0.5079 vs 0.5163 (-0.0084): does not beat, noise (DM z -0.45) |
| logreg_lags | direction/bal_acc | 0.5018 vs 0.5296 (-0.0278): does not beat, significantly worse (boot z -2.28) | 0.5138 vs 0.5176 (-0.0038): does not beat, noise (boot z -0.26) | 0.5104 vs 0.5142 (-0.0038): does not beat, noise (boot z -0.21) |
| class_prior | direction/mcc | 0.0036 vs 0.0000 (+0.0036): beats, noise (boot z +0.19) | 0.0283 vs 0.0000 (+0.0283): beats, noise (boot z +1.59) | 0.0211 vs 0.0000 (+0.0211): beats, noise (boot z +0.98) |
| class_prior | direction/auc | 0.4984 vs 0.5000 (-0.0016): does not beat, noise (boot z -0.14) | 0.5152 vs 0.5000 (+0.0152): beats, noise (boot z +1.32) | 0.5199 vs 0.5000 (+0.0199): beats, noise (boot z +1.45) |
| class_prior | direction/brier | 0.2683 vs 0.2499 (-0.0184): does not beat, significantly worse (DM z -6.99) | 0.2632 vs 0.2499 (-0.0133): does not beat, significantly worse (DM z -4.90) | 0.2607 vs 0.2498 (-0.0109): does not beat, significantly worse (DM z -3.57) |
| class_prior | direction/ece_pos | 0.0937 vs 0.0060 (-0.0877): does not beat, significantly worse (boot z -6.85) | 0.0814 vs 0.0044 (-0.0770): does not beat, significantly worse (boot z -6.51) | 0.0836 vs 0.0064 (-0.0773): does not beat, significantly worse (boot z -5.78) |
| class_prior | direction/acc | 0.5002 vs 0.5120 (-0.0118): does not beat, noise (DM z -0.72) | 0.5117 vs 0.5097 (+0.0020): beats, noise (DM z +0.10) | 0.5079 vs 0.5145 (-0.0066): does not beat, noise (DM z -0.29) |
| class_prior | direction/bal_acc | 0.5018 vs 0.5000 (+0.0018): beats, noise (boot z +0.19) | 0.5138 vs 0.5000 (+0.0138): beats, noise (boot z +1.59) | 0.5104 vs 0.5000 (+0.0104): beats, noise (boot z +0.98) |
| zero_delta | delta/rmse | 55.35 vs 55.38 (+0.04, +0.07%): beats, noise (DM z +0.73) | 68.21 vs 68.23 (+0.02, +0.03%): beats, noise (DM z +0.71) | 79.34 vs 79.34 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 34.92 vs 34.92 (+0.00, +0.01%): beats, noise (DM z +0.14) | 42.60 vs 42.60 (-0.00, -0.00%): does not beat, noise (DM z -0.02) | 49.49 vs 49.49 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 55.35 vs 55.38 (+0.03, +0.06%): beats, noise (DM z +0.67) | 68.21 vs 68.22 (+0.01, +0.02%): beats, noise (DM z +0.43) | 79.34 vs 79.32 (-0.01, -0.02%): does not beat, noise (DM z -1.18) |
| mean_delta | delta/mae | 34.92 vs 34.92 (+0.00, +0.01%): beats, noise (DM z +0.16) | 42.60 vs 42.60 (-0.00, -0.00%): does not beat, noise (DM z -0.08) | 49.49 vs 49.49 (-0.00, -0.00%): does not beat, noise (DM z -0.18) |
| const_var | variance/crps | 26.56 vs 28.12 (+1.56, +5.55%): beats (DM z +11.88) | 32.87 vs 34.37 (+1.49, +4.35%): beats (DM z +10.82) | 37.89 vs 39.90 (+2.01, +5.04%): beats (DM z +8.67) |
| const_var | variance/nll | 5.5470 vs 7.4610 (+1.9140): beats (DM z +7.77) | 5.9528 vs 7.6534 (+1.7006): beats (DM z +6.82) | 6.0112 vs 7.8083 (+1.7971): beats (DM z +6.02) |
| const_var | variance/pit_ks | 0.0596 vs 0.1245 (+0.0648): beats (boot z +11.52) | 0.0726 vs 0.1208 (+0.0482): beats (boot z +9.86) | 0.0588 vs 0.1242 (+0.0654): beats (boot z +9.22) |
| const_var | variance/corr_var_err2_spearman | 0.3122 vs 0.0000 (+0.3122): beats (boot z +12.41) | 0.2942 vs 0.0000 (+0.2942): beats (boot z +10.27) | 0.3038 vs 0.0000 (+0.3038): beats (boot z +10.06) |

## Backtest (costs included)

- n_trades: 835
- total_return: 0.0417
- sharpe_net: 3.9550
- sharpe_gross: 3.9550
- sortino: 5.9029
- max_drawdown: 0.0645
- hit_rate: 0.5054
- hit_rate_gross: 0.5054
- profit_factor: 1.0653
- avg_hold_bars: 8.4048
- exposure: 0.4726
- turnover: 1722.5076
- fees_paid: 0.0000
- traded_notional: 17226455.0743
- breakeven_cost_bps: 0.4839
- gross_edge_per_trade_bps: 0.5143
- costs_paid: 0.0000
- gross_pnl: 416.8096
- net_pnl: 416.8096

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 78.8119, indicator 225.0406 (clip 20).
Clipped steps over the run: main 15.0000, indicator 50.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 78.8119 / 132.8972 | 6.0000 / 4.0000 | 10.3% / 6.9% | 0.0000 | 2763.0000 / 3243.0000 / 3538.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 14.7195 / 46.3037 | 0.0000 / 2.0000 | 0.0% / 3.4% | 0.0000 | 3263.0000 / 3763.0000 / 4132.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 9.9828 / 28.0132 | 0.0000 / 2.0000 | 0.0% / 3.4% | 0.0000 | 3233.0000 / 3789.0000 / 4047.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 42.8158 / 81.6431 | 1.0000 / 5.0000 | 1.7% / 8.6% | 0.0000 | 3163.0000 / 3666.0000 / 3994.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 14.2256 / 28.8766 | 0.0000 / 3.0000 | 0.0% / 5.2% | 0.0000 | 2758.0000 / 3194.0000 / 3497.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 25.1900 / 42.3245 | 1.0000 / 1.0000 | 1.7% / 1.7% | 0.0000 | 2684.0000 / 3159.0000 / 3405.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 19.1544 / 91.3166 | 0.0000 / 3.0000 | 0.0% / 5.2% | 0.0000 | 3162.0000 / 3762.0000 / 4092.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 52.9136 / 124.4897 | 1.0000 / 4.0000 | 1.7% / 6.9% | 0.0000 | 3188.0000 / 3720.0000 / 4056.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 45.6919 / 152.4872 | 2.0000 / 3.0000 | 3.4% / 5.2% | 0.0000 | 3135.0000 / 3687.0000 / 4079.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 22.6065 / 32.1986 | 1.0000 / 1.0000 | 1.7% / 1.7% | 0.0000 | 2705.0000 / 3155.0000 / 3448.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 19.3296 / 43.5782 | 0.0000 / 5.0000 | 0.0% / 8.6% | 0.0000 | 2693.0000 / 3146.0000 / 3493.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 52.3588 / 96.2833 | 1.0000 / 3.0000 | 1.7% / 5.2% | 0.0000 | 3212.0000 / 3775.0000 / 4146.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 53.2113 / 225.0406 | 2.0000 / 8.0000 | 3.4% / 13.8% | 0.0000 | 3204.0000 / 3754.0000 / 4090.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 17.5821 / 77.5744 | 0.0000 / 6.0000 | 0.0% / 10.3% | 0.0000 | 3258.0000 / 3731.0000 / 4042.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.883 (corr skip/tower=-0.513), h1=1.071 (corr skip/tower=-0.678), h2=0.999 (corr skip/tower=-0.730).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -93 (TimeSeriesSplit fold 8, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-13T01:15:00 .. 2023-02-23T08:45:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.408, long_above 0.5590, short_below 0.4289, median 0.4928. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +4.17% | +3.95 | +6.45% | 835 |
| buy and hold | +11.40% | +7.64 | +7.31% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -7.93% .. +9.33%) | -0.06% | -0.05 | | |

The random null enters at the strategy's rate (0.1066 per flat bar), holds 8 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 75% of its seeds on net return, 72% on net Sharpe and 75% on gross return.
