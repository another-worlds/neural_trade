# Evaluation report - dev split - run `20261006T093908Z-61014d0-b11ab9ff-gru_small__f-92__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 9392 | 10127 | 10722 |
| n_eff of the scored moves (n scored // bars ahead) | 939 | 675 | 536 |
| true up-rate | 0.4856 | 0.4794 | 0.4808 |
| calls up (predicted up-rate) | 0.5603 | 0.4391 | 0.4207 |
| accuracy | 0.5201 | 0.5079 | 0.5115 |
| balanced accuracy | 0.5219 | 0.5054 | 0.5084 |
| precision (up) | 0.5051 | 0.4855 | 0.4908 |
| recall / sensitivity (up) | 0.5828 | 0.4447 | 0.4295 |
| specificity (down) | 0.4610 | 0.5660 | 0.5874 |
| F1 (up) | 0.5412 | 0.4642 | 0.4581 |
| MCC | 0.0441 | 0.0108 | 0.0171 |
| AUC | 0.5237 | 0.5083 | 0.5102 |
| Brier | 0.2565 | 0.2589 | 0.2581 |
| ECE (positive class) | 0.0560 | 0.0697 | 0.0642 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0144 | 0.0206 | 0.0192 |
| TP / FP / TN / FN | 2658 / 2604 / 2227 / 1903 | 2159 / 2288 / 2984 / 2696 | 2214 / 2297 / 3270 / 2941 |
| Gaussian readout: calls up | 0.6240 | 0.5923 | 0.5558 |
| Gaussian readout: MCC | 0.0223 | 0.0022 | 0.0173 |
| Gaussian readout: AUC | 0.5069 | 0.5004 | 0.5050 |
| Gaussian readout: Brier | 0.2503 | 0.2502 | 0.2501 |
| Gaussian readout: ECE | 0.0205 | 0.0225 | 0.0206 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 43.32 | 54.31 | 63.40 |
| RMSE ($), raw heads | 43.65 | 55.13 | 64.39 |
| RMSE ($), zero prediction | 43.32 | 54.31 | 63.40 |
| MAE ($), served | 26.30 | 32.00 | 36.81 |
| MAE ($), raw heads | 26.69 | 32.83 | 38.06 |
| MAE ($), zero prediction | 26.28 | 31.98 | 36.80 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0000 | -0.0001 | 0.0001 |
| skill vs zero, raw heads | -0.0150 | -0.0305 | -0.0313 |
| EV, served | 0.0005 | 0.0000 | 0.0004 |
| EV, raw heads | -0.0107 | -0.0251 | -0.0268 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0228 | 0.0072 | 0.0212 |
| corr, Spearman, raw heads | 0.0007 | -0.0078 | 0.0011 |
| mean predicted ($), served | 0.31 | 0.13 | 0.18 |
| mean predicted ($), raw heads | 1.76 | 2.41 | 2.30 |
| mean realised ($) | -1.39 | -2.08 | -2.78 |
| share predicted up, raw heads | 0.6114 | 0.5814 | 0.5416 |
| share realised up | 0.4734 | 0.4754 | 0.4770 |
| shrink beta (served = beta x raw, fit on cal) | 0.1781 | 0.0545 | 0.0768 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 19.87 | 24.34 | 28.04 |
| CRPSS vs constant variance | 0.0336 | 0.0298 | 0.0278 |
| NLL | 5.2502 | 5.6412 | 5.7950 |
| PIT KS | 0.0304 | 0.0347 | 0.0301 |
| var / err^2 Spearman | 0.2961 | 0.2981 | 0.2824 |
| coverage of the 90% interval | 0.9034 | 0.8993 | 0.8968 |
| width of the 90% interval ($) | 120.10 | 145.35 | 166.76 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0020 | [-0.0263, 0.0315] | NOISE |
| h1 | 0.0138 | [-0.0159, 0.0429] | NOISE |
| h2 | 0.0039 | [-0.0330, 0.0366] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.178 / h1 0.055 / h2 0.077) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7468 | 0.1104 | 0.5983 |
| abs(d h1) <= abs(d h2) | 0.7379 | 0.8504 | 0.5786 |
| full chain h0 <= h1 <= h2 | 0.5173 | 0.0756 | 0.3054 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.7401 | 0.6913 | 0.7288 | 0.4464 |
| expected if the two signs were independent | 0.5102 | 0.4876 | 0.4929 | 0.2254 |

- P(up) unanimity (all three horizons call the same side): 0.5200

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0441 vs 0.0646 (-0.0206): does not beat, noise (boot z -0.83) | 0.0108 vs 0.0750 (-0.0642): does not beat, significantly worse (boot z -2.63) | 0.0171 vs 0.0755 (-0.0584): does not beat, noise (boot z -1.95) |
| logreg_lags | direction/auc | 0.5237 vs 0.5596 (-0.0359): does not beat, significantly worse (boot z -2.18) | 0.5083 vs 0.5695 (-0.0612): does not beat, significantly worse (boot z -4.06) | 0.5102 vs 0.5730 (-0.0628): does not beat, significantly worse (boot z -3.13) |
| logreg_lags | direction/brier | 0.2565 vs 0.2488 (-0.0078): does not beat, significantly worse (DM z -3.04) | 0.2589 vs 0.2491 (-0.0098): does not beat, significantly worse (DM z -3.89) | 0.2581 vs 0.2489 (-0.0092): does not beat, significantly worse (DM z -2.96) |
| logreg_lags | direction/ece_pos | 0.0560 vs 0.0315 (-0.0245): does not beat, significantly worse (boot z -2.42) | 0.0697 vs 0.0409 (-0.0288): does not beat, noise (boot z -1.52) | 0.0642 vs 0.0445 (-0.0196): does not beat, noise (boot z -0.97) |
| logreg_lags | direction/acc | 0.5201 vs 0.5244 (-0.0043): does not beat, noise (DM z -0.33) | 0.5079 vs 0.5260 (-0.0182): does not beat, noise (DM z -1.19) | 0.5115 vs 0.5237 (-0.0122): does not beat, noise (DM z -0.65) |
| logreg_lags | direction/bal_acc | 0.5219 vs 0.5299 (-0.0080): does not beat, noise (boot z -0.67) | 0.5054 vs 0.5344 (-0.0290): does not beat, significantly worse (boot z -2.49) | 0.5084 vs 0.5330 (-0.0246): does not beat, noise (boot z -1.77) |
| class_prior | direction/mcc | 0.0441 vs 0.0000 (+0.0441): beats (boot z +2.36) | 0.0108 vs 0.0000 (+0.0108): beats, noise (boot z +0.57) | 0.0171 vs 0.0000 (+0.0171): beats, noise (boot z +0.82) |
| class_prior | direction/auc | 0.5237 vs 0.5000 (+0.0237): beats, noise (boot z +1.95) | 0.5083 vs 0.5000 (+0.0083): beats, noise (boot z +0.68) | 0.5102 vs 0.5000 (+0.0102): beats, noise (boot z +0.69) |
| class_prior | direction/brier | 0.2565 vs 0.2505 (-0.0061): does not beat, significantly worse (DM z -2.89) | 0.2589 vs 0.2506 (-0.0083): does not beat, significantly worse (DM z -3.13) | 0.2581 vs 0.2507 (-0.0074): does not beat, significantly worse (DM z -2.41) |
| class_prior | direction/ece_pos | 0.0560 vs 0.0260 (-0.0299): does not beat, significantly worse (boot z -2.99) | 0.0697 vs 0.0322 (-0.0374): does not beat, significantly worse (boot z -2.00) | 0.0642 vs 0.0332 (-0.0310): does not beat, noise (boot z -1.55) |
| class_prior | direction/acc | 0.5201 vs 0.4856 (+0.0345): beats (DM z +2.31) | 0.5079 vs 0.4794 (+0.0284): beats, noise (DM z +1.43) | 0.5115 vs 0.4808 (+0.0307): beats, noise (DM z +1.35) |
| class_prior | direction/bal_acc | 0.5219 vs 0.5000 (+0.0219): beats (boot z +2.36) | 0.5054 vs 0.5000 (+0.0054): beats, noise (boot z +0.57) | 0.5084 vs 0.5000 (+0.0084): beats, noise (boot z +0.82) |
| zero_delta | delta/rmse | 43.32 vs 43.32 (+0.00, +0.00%): beats, noise (DM z +0.00) | 54.31 vs 54.31 (-0.00, -0.01%): does not beat, noise (DM z -0.26) | 63.40 vs 63.40 (+0.00, +0.01%): beats, noise (DM z +0.14) |
| zero_delta | delta/mae | 26.30 vs 26.28 (-0.02, -0.08%): does not beat, noise (DM z -1.17) | 32.00 vs 31.98 (-0.01, -0.04%): does not beat, noise (DM z -1.12) | 36.81 vs 36.80 (-0.01, -0.03%): does not beat, noise (DM z -0.59) |
| mean_delta | delta/rmse | 43.32 vs 43.34 (+0.01, +0.03%): beats, noise (DM z +0.39) | 54.31 vs 54.33 (+0.02, +0.03%): beats, noise (DM z +0.74) | 63.40 vs 63.43 (+0.04, +0.06%): beats, noise (DM z +0.77) |
| mean_delta | delta/mae | 26.30 vs 26.29 (-0.00, -0.01%): does not beat, noise (DM z -0.18) | 32.00 vs 32.01 (+0.01, +0.04%): beats, noise (DM z +0.96) | 36.81 vs 36.83 (+0.02, +0.05%): beats, noise (DM z +0.70) |
| const_var | variance/crps | 19.87 vs 20.56 (+0.69, +3.36%): beats (DM z +8.81) | 24.34 vs 25.09 (+0.75, +2.98%): beats (DM z +8.17) | 28.04 vs 28.85 (+0.80, +2.78%): beats (DM z +5.81) |
| const_var | variance/nll | 5.2502 vs 6.2346 (+0.9844): beats (DM z +3.75) | 5.6412 vs 6.5179 (+0.8768): beats (DM z +3.27) | 5.7950 vs 6.6968 (+0.9018): beats (DM z +2.73) |
| const_var | variance/pit_ks | 0.0304 vs 0.0828 (+0.0524): beats (boot z +6.60) | 0.0347 vs 0.0747 (+0.0400): beats (boot z +6.06) | 0.0301 vs 0.0725 (+0.0424): beats (boot z +4.35) |
| const_var | variance/corr_var_err2_spearman | 0.2961 vs 0.0000 (+0.2961): beats (boot z +11.10) | 0.2981 vs 0.0000 (+0.2981): beats (boot z +10.46) | 0.2824 vs 0.0000 (+0.2824): beats (boot z +8.97) |

## Backtest (costs included)

- n_trades: 666
- total_return: 0.0320
- sharpe_net: 4.5491
- sharpe_gross: 4.5491
- sortino: 6.4400
- max_drawdown: 0.0368
- hit_rate: 0.5255
- hit_rate_gross: 0.5255
- profit_factor: 1.0861
- avg_hold_bars: 8.7447
- exposure: 0.3922
- turnover: 1373.2395
- fees_paid: 0.0000
- traded_notional: 13732030.5108
- breakeven_cost_bps: 0.4662
- gross_edge_per_trade_bps: 0.4848
- costs_paid: 0.0000
- gross_pnl: 320.0819
- net_pnl: 320.0819

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 522.7101, indicator 596.9432 (clip 20).
Clipped steps over the run: main 18.0000, indicator 42.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 80.1248 / 51.7079 | 4.0000 / 2.0000 | 5.6% / 2.8% | 0.0000 | 3512.0000 / 4092.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 13.5711 / 16.5989 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3452.0000 / 4067.0000 / 4454.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 16.9923 / 28.2296 | 0.0000 / 1.0000 | 0.0% / 1.4% | 0.0000 | 3414.0000 / 4008.0000 / 4421.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 46.3314 / 76.7586 | 4.0000 / 6.0000 | 5.6% / 8.3% | 0.0000 | 3493.0000 / 4116.0000 / 4488.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 22.9153 / 27.7522 | 1.0000 / 1.0000 | 1.4% / 1.4% | 0.0000 | 3444.0000 / 4007.0000 / 4375.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 35.0764 / 166.6197 | 1.0000 / 4.0000 | 1.4% / 5.6% | 0.0000 | 3468.0000 / 4051.0000 / 4457.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 522.7101 / 596.9432 | 3.0000 / 7.0000 | 4.2% / 9.7% | 0.0000 | 3459.0000 / 4101.0000 / 4492.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 108.2619 / 140.1889 | 1.0000 / 3.0000 | 1.4% / 4.2% | 0.0000 | 3495.0000 / 4140.0000 / 4527.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 13.7989 / 60.9191 | 0.0000 / 2.0000 | 0.0% / 2.8% | 0.0000 | 3460.0000 / 4074.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 13.4491 / 35.6586 | 0.0000 / 1.0000 | 0.0% / 1.4% | 0.0000 | 3503.0000 / 4005.0000 / 4505.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 35.4700 / 69.7365 | 1.0000 / 3.0000 | 1.4% / 4.2% | 0.0000 | 3507.0000 / 4058.0000 / 4513.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 14.3313 / 68.8944 | 0.0000 / 4.0000 | 0.0% / 5.6% | 0.0000 | 3437.0000 / 4007.0000 / 4382.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 14.1748 / 30.3960 | 0.0000 / 2.0000 | 0.0% / 2.8% | 0.0000 | 3427.0000 / 4093.0000 / 4540.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 177.5924 / 377.6439 | 3.0000 / 6.0000 | 4.2% / 8.3% | 0.0000 | 3463.0000 / 4091.0000 / 4446.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
Learned periods sitting at their configured bound: period/vwap_period_2=60.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.580 (corr skip/tower=-0.391), h1=0.653 (corr skip/tower=-0.462), h2=0.487 (corr skip/tower=-0.512).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -92 (TimeSeriesSplit fold 9, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-23T08:46:00 .. 2023-03-05T16:16:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 1.599, long_above 0.5795, short_below 0.4064, median 0.4943. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +3.20% | +4.55 | +3.68% | 666 |
| buy and hold | -8.52% | -7.39 | +9.73% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -6.02% .. +7.19%) | +0.29% | +0.40 | | |

The random null enters at the strategy's rate (0.0738 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 76% of its seeds on net return, 77% on net Sharpe and 76% on gross return.
