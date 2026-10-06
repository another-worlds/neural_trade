# Evaluation report - dev split - run `20261006T081026Z-61014d0-9ec26d2c-control__f-94__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 8849 | 9593 | 9933 |
| n_eff of the scored moves (n scored // bars ahead) | 884 | 639 | 496 |
| true up-rate | 0.4864 | 0.4793 | 0.4801 |
| calls up (predicted up-rate) | 0.6875 | 0.6487 | 0.7410 |
| accuracy | 0.5163 | 0.5339 | 0.5098 |
| balanced accuracy | 0.5215 | 0.5402 | 0.5194 |
| precision (up) | 0.5020 | 0.5102 | 0.4932 |
| recall / sensitivity (up) | 0.7096 | 0.6905 | 0.7612 |
| specificity (down) | 0.3333 | 0.3898 | 0.2777 |
| F1 (up) | 0.5880 | 0.5868 | 0.5986 |
| MCC | 0.0463 | 0.0840 | 0.0443 |
| AUC | 0.5329 | 0.5572 | 0.5428 |
| Brier | 0.2633 | 0.2584 | 0.2637 |
| ECE (positive class) | 0.1012 | 0.0830 | 0.1119 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0136 | 0.0207 | 0.0199 |
| TP / FP / TN / FN | 3054 / 3030 / 1515 / 1250 | 3175 / 3048 / 1947 / 1423 | 3630 / 3730 / 1434 / 1139 |
| Gaussian readout: calls up | 0.6963 | 0.6799 | 0.6686 |
| Gaussian readout: MCC | 0.0831 | 0.0756 | 0.0674 |
| Gaussian readout: AUC | 0.5469 | 0.5534 | 0.5440 |
| Gaussian readout: Brier | 0.2503 | 0.2507 | 0.2518 |
| Gaussian readout: ECE | 0.0448 | 0.0531 | 0.0583 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 35.67 | 43.82 | 50.46 |
| RMSE ($), raw heads | 36.54 | 45.43 | 52.66 |
| RMSE ($), zero prediction | 35.60 | 43.71 | 50.34 |
| MAE ($), served | 22.50 | 27.09 | 30.67 |
| MAE ($), raw heads | 22.96 | 28.01 | 32.35 |
| MAE ($), zero prediction | 22.55 | 27.19 | 30.71 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0035 | -0.0050 | -0.0049 |
| skill vs zero, raw heads | -0.0535 | -0.0803 | -0.0945 |
| EV, served | -0.0020 | -0.0030 | -0.0023 |
| EV, raw heads | -0.0412 | -0.0671 | -0.0776 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0013 | 0.0086 | 0.0211 |
| corr, Spearman, raw heads | 0.0673 | 0.0763 | 0.0662 |
| mean predicted ($), served | 0.65 | 0.85 | 1.12 |
| mean predicted ($), raw heads | 2.92 | 3.55 | 4.59 |
| mean realised ($) | -1.20 | -1.81 | -2.42 |
| share predicted up, raw heads | 0.7111 | 0.6858 | 0.6728 |
| share realised up | 0.4785 | 0.4800 | 0.4785 |
| shrink beta (served = beta x raw, fit on cal) | 0.2225 | 0.2387 | 0.2446 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 17.19 | 20.90 | 23.77 |
| CRPSS vs constant variance | 0.0051 | 0.0034 | 0.0054 |
| NLL | 5.5420 | 5.9310 | 5.9808 |
| PIT KS | 0.0874 | 0.0931 | 0.0855 |
| var / err^2 Spearman | 0.3292 | 0.2979 | 0.2897 |
| coverage of the 90% interval | 0.9075 | 0.9104 | 0.9135 |
| width of the 90% interval ($) | 101.86 | 125.57 | 146.47 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0117 | [-0.0272, 0.0484] | NOISE |
| h1 | 0.0273 | [-0.0129, 0.0618] | NOISE |
| h2 | 0.0336 | [-0.0086, 0.0764] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.223 / h1 0.239 / h2 0.245) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.8092 | 0.8418 | 0.6023 |
| abs(d h1) <= abs(d h2) | 0.8113 | 0.8296 | 0.5711 |
| full chain h0 <= h1 <= h2 | 0.6573 | 0.7022 | 0.3135 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.8507 | 0.9334 | 0.8824 | 0.7615 |
| expected if the two signs were independent | 0.5847 | 0.5579 | 0.5848 | 0.4402 |

- P(up) unanimity (all three horizons call the same side): 0.7840

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0463 vs 0.0988 (-0.0526): does not beat, noise (boot z -1.78) | 0.0840 vs 0.0921 (-0.0081): does not beat, noise (boot z -0.30) | 0.0443 vs 0.0686 (-0.0243): does not beat, noise (boot z -0.82) |
| logreg_lags | direction/auc | 0.5329 vs 0.5568 (-0.0239): does not beat, noise (boot z -1.23) | 0.5572 vs 0.5638 (-0.0066): does not beat, noise (boot z -0.38) | 0.5428 vs 0.5593 (-0.0165): does not beat, noise (boot z -0.89) |
| logreg_lags | direction/brier | 0.2633 vs 0.2483 (-0.0151): does not beat, significantly worse (DM z -4.52) | 0.2584 vs 0.2479 (-0.0105): does not beat, significantly worse (DM z -2.97) | 0.2637 vs 0.2483 (-0.0154): does not beat, significantly worse (DM z -3.89) |
| logreg_lags | direction/ece_pos | 0.1012 vs 0.0242 (-0.0769): does not beat, significantly worse (boot z -7.06) | 0.0830 vs 0.0319 (-0.0511): does not beat, significantly worse (boot z -4.93) | 0.1119 vs 0.0350 (-0.0769): does not beat, significantly worse (boot z -8.19) |
| logreg_lags | direction/acc | 0.5163 vs 0.5430 (-0.0267): does not beat, significantly worse (DM z -2.04) | 0.5339 vs 0.5389 (-0.0050): does not beat, noise (DM z -0.40) | 0.5098 vs 0.5243 (-0.0145): does not beat, noise (DM z -1.11) |
| logreg_lags | direction/bal_acc | 0.5215 vs 0.5471 (-0.0257): does not beat, noise (boot z -1.86) | 0.5402 vs 0.5445 (-0.0043): does not beat, noise (boot z -0.33) | 0.5194 vs 0.5318 (-0.0124): does not beat, noise (boot z -0.93) |
| class_prior | direction/mcc | 0.0463 vs 0.0000 (+0.0463): beats, noise (boot z +1.83) | 0.0840 vs 0.0000 (+0.0840): beats (boot z +3.21) | 0.0443 vs 0.0000 (+0.0443): beats, noise (boot z +1.62) |
| class_prior | direction/auc | 0.5329 vs 0.5000 (+0.0329): beats (boot z +2.06) | 0.5572 vs 0.5000 (+0.0572): beats (boot z +3.45) | 0.5428 vs 0.5000 (+0.0428): beats (boot z +2.37) |
| class_prior | direction/brier | 0.2633 vs 0.2503 (-0.0130): does not beat, significantly worse (DM z -4.01) | 0.2584 vs 0.2504 (-0.0080): does not beat, significantly worse (DM z -2.13) | 0.2637 vs 0.2505 (-0.0132): does not beat, significantly worse (DM z -3.08) |
| class_prior | direction/ece_pos | 0.1012 vs 0.0224 (-0.0788): does not beat, significantly worse (boot z -10.06) | 0.0830 vs 0.0282 (-0.0548): does not beat, significantly worse (boot z -6.40) | 0.1119 vs 0.0298 (-0.0821): does not beat, significantly worse (boot z -9.16) |
| class_prior | direction/acc | 0.5163 vs 0.4864 (+0.0299): beats (DM z +2.34) | 0.5339 vs 0.4793 (+0.0546): beats (DM z +3.48) | 0.5098 vs 0.4801 (+0.0297): beats (DM z +2.16) |
| class_prior | direction/bal_acc | 0.5215 vs 0.5000 (+0.0215): beats, noise (boot z +1.83) | 0.5402 vs 0.5000 (+0.0402): beats (boot z +3.20) | 0.5194 vs 0.5000 (+0.0194): beats, noise (boot z +1.61) |
| zero_delta | delta/rmse | 35.67 vs 35.60 (-0.06, -0.18%): does not beat, noise (DM z -1.20) | 43.82 vs 43.71 (-0.11, -0.25%): does not beat, noise (DM z -1.04) | 50.46 vs 50.34 (-0.12, -0.25%): does not beat, noise (DM z -0.84) |
| zero_delta | delta/mae | 22.50 vs 22.55 (+0.05, +0.23%): beats, noise (DM z +1.62) | 27.09 vs 27.19 (+0.11, +0.39%): beats, noise (DM z +1.77) | 30.67 vs 30.71 (+0.03, +0.11%): beats, noise (DM z +0.40) |
| mean_delta | delta/rmse | 35.67 vs 35.61 (-0.05, -0.15%): does not beat, noise (DM z -1.09) | 43.82 vs 43.73 (-0.09, -0.21%): does not beat, noise (DM z -0.93) | 50.46 vs 50.36 (-0.10, -0.20%): does not beat, noise (DM z -0.72) |
| mean_delta | delta/mae | 22.50 vs 22.56 (+0.06, +0.27%): beats (DM z +2.01) | 27.09 vs 27.20 (+0.12, +0.44%): beats (DM z +2.07) | 30.67 vs 30.72 (+0.05, +0.17%): beats, noise (DM z +0.64) |
| const_var | variance/crps | 17.19 vs 17.28 (+0.09, +0.51%): beats, noise (DM z +1.30) | 20.90 vs 20.97 (+0.07, +0.34%): beats, noise (DM z +0.72) | 23.77 vs 23.90 (+0.13, +0.54%): beats, noise (DM z +1.02) |
| const_var | variance/nll | 5.5420 vs 5.2417 (-0.3003): does not beat, significantly worse (DM z -3.96) | 5.9310 vs 5.4377 (-0.4933): does not beat, significantly worse (DM z -3.95) | 5.9808 vs 5.5668 (-0.4140): does not beat, significantly worse (DM z -3.36) |
| const_var | variance/pit_ks | 0.0874 vs 0.0392 (-0.0482): does not beat, significantly worse (boot z -5.98) | 0.0931 vs 0.0494 (-0.0437): does not beat, significantly worse (boot z -3.75) | 0.0855 vs 0.0598 (-0.0257): does not beat, noise (boot z -1.84) |
| const_var | variance/corr_var_err2_spearman | 0.3292 vs 0.0000 (+0.3292): beats (boot z +13.51) | 0.2979 vs 0.0000 (+0.2979): beats (boot z +11.49) | 0.2897 vs 0.0000 (+0.2897): beats (boot z +10.30) |

## Backtest (costs included)

- n_trades: 412
- total_return: 0.0112
- sharpe_net: 1.9279
- sharpe_gross: 1.9279
- sortino: 2.7364
- max_drawdown: 0.0464
- hit_rate: 0.5437
- hit_rate_gross: 0.5437
- profit_factor: 1.0462
- avg_hold_bars: 10.1359
- exposure: 0.2812
- turnover: 811.7342
- fees_paid: 0.0000
- traded_notional: 8117172.7885
- breakeven_cost_bps: 0.2761
- gross_edge_per_trade_bps: 0.2835
- costs_paid: 0.0000
- gross_pnl: 112.0662
- net_pnl: 112.0662

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 1123.0648, indicator 1436.9249 (clip 20).
Clipped steps over the run: main 79.0000, indicator 114.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 40.9817 / 22.9473 | 1.0000 / 1.0000 | 2.3% / 2.3% | 0.0000 | 2605.0000 / 2952.0000 / 3194.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 3.5735 / 2.6436 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2610.0000 / 3022.0000 / 3243.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 10.7839 / 10.7443 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2526.0000 / 2918.0000 / 3192.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 15.8202 / 41.1348 | 0.0000 / 2.0000 | 0.0% / 4.7% | 0.0000 | 3121.0000 / 3591.0000 / 3884.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 17.2283 / 30.4412 | 0.0000 / 1.0000 | 0.0% / 2.3% | 0.0000 | 2572.0000 / 2911.0000 / 3162.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 31.3406 / 115.3467 | 4.0000 / 6.0000 | 9.3% / 14.0% | 0.0000 | 2588.0000 / 2972.0000 / 3137.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 34.9483 / 145.5073 | 2.0000 / 6.0000 | 4.7% / 14.0% | 0.0000 | 2957.0000 / 3360.0000 / 3589.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 51.5331 / 93.9342 | 11.0000 / 8.0000 | 25.6% / 18.6% | 0.0000 | 2639.0000 / 2964.0000 / 3218.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 40.3644 / 72.8978 | 6.0000 / 13.0000 | 14.0% / 30.2% | 0.0000 | 2587.0000 / 3003.0000 / 3212.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 1123.0648 / 1436.9249 | 8.0000 / 9.0000 | 18.6% / 20.9% | 0.0000 | 2551.0000 / 2956.0000 / 3195.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 203.7974 / 320.3801 | 9.0000 / 14.0000 | 20.9% / 32.6% | 0.0000 | 2523.0000 / 2910.0000 / 3171.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 80.1465 / 302.9352 | 7.0000 / 18.0000 | 16.3% / 41.9% | 0.0000 | 2645.0000 / 2973.0000 / 3210.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 102.2599 / 150.1620 | 11.0000 / 12.0000 | 25.6% / 27.9% | 0.0000 | 2602.0000 / 2933.0000 / 3194.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 87.8850 / 520.9717 | 20.0000 / 24.0000 | 46.5% / 55.8% | 0.0000 | 3118.0000 / 3548.0000 / 3866.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=-0.053 (corr skip/tower=-0.764), h1=-0.020 (corr skip/tower=-0.278), h2=0.021 (corr skip/tower=-0.474).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -94 (TimeSeriesSplit fold 7, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-02T17:44:00 .. 2023-02-13T01:14:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6115, long_above 0.7043, short_below 0.3941, median 0.5476. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +1.12% | +1.93 | +4.64% | 412 |
| buy and hold | -7.43% | -7.41 | +10.90% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -4.99% .. +5.53%) | +0.65% | +1.27 | | |

The random null enters at the strategy's rate (0.0386 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 52% of its seeds on net return, 52% on net Sharpe and 52% on gross return.
