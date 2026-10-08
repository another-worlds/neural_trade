# Evaluation report - dev split - run `20261006T080236Z-61014d0-c7bb80c5-control__f-95__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10047 | 10743 | 11232 |
| n_eff of the scored moves (n scored // bars ahead) | 1004 | 716 | 561 |
| true up-rate | 0.5088 | 0.5070 | 0.5105 |
| calls up (predicted up-rate) | 0.5632 | 0.4302 | 0.5846 |
| accuracy | 0.4822 | 0.5427 | 0.5262 |
| balanced accuracy | 0.4811 | 0.5437 | 0.5244 |
| precision (up) | 0.4920 | 0.5578 | 0.5314 |
| recall / sensitivity (up) | 0.5446 | 0.4733 | 0.6085 |
| specificity (down) | 0.4176 | 0.6140 | 0.4403 |
| F1 (up) | 0.5170 | 0.5121 | 0.5673 |
| MCC | -0.0381 | 0.0882 | 0.0495 |
| AUC | 0.4713 | 0.5588 | 0.5390 |
| Brier | 0.2803 | 0.2502 | 0.2577 |
| ECE (positive class) | 0.1218 | 0.0396 | 0.0641 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0088 | 0.0070 | 0.0105 |
| TP / FP / TN / FN | 2784 / 2874 / 2061 / 2328 | 2578 / 2044 / 3252 / 2869 | 3489 / 3077 / 2421 / 2245 |
| Gaussian readout: calls up | 0.3685 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0724 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5340 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2491 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0118 | n/a (beta = 0: readout is the constant 0.5) | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.3685 | 0.4132 | 0.3429 |
| Gaussian readout of the raw heads: MCC | 0.0724 | 0.0916 | 0.0730 |
| Gaussian readout of the raw heads: AUC | 0.5341 | 0.5548 | 0.5549 |
| Gaussian readout of the raw heads: Brier | 0.2544 | 0.2567 | 0.2591 |
| Gaussian readout of the raw heads: ECE | 0.0526 | 0.0683 | 0.0989 |

beta = 0 for h1, h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 45.62 | 55.69 | 63.66 |
| RMSE ($), raw heads | 45.69 | 55.81 | 64.02 |
| RMSE ($), zero prediction | 45.63 | 55.69 | 63.66 |
| MAE ($), served | 28.74 | 35.11 | 40.44 |
| MAE ($), raw heads | 28.74 | 35.10 | 40.59 |
| MAE ($), zero prediction | 28.77 | 35.11 | 40.44 |
| skill vs zero (1 - MSE / MSE of 0), served | 0.0005 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0025 | -0.0045 | -0.0116 |
| EV, served | 0.0006 | n/a (beta = 0: served delta is 0) | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0020 | -0.0030 | -0.0081 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0247 | 0.0423 | 0.0448 |
| corr, Spearman, raw heads | 0.0496 | 0.0815 | 0.0836 |
| mean predicted ($), served | -0.13 | 0.00 | 0.00 |
| mean predicted ($), raw heads | -0.48 | -1.37 | -2.60 |
| mean realised ($) | 0.71 | 1.07 | 1.42 |
| share predicted up, raw heads | 0.3553 | 0.4088 | 0.3328 |
| share realised up | 0.4979 | 0.4947 | 0.4994 |
| shrink beta (served = beta x raw, fit on cal) | 0.2679 | 0.0000 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h1, h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 22.25 | 27.20 | 31.22 |
| CRPSS vs constant variance | 0.0038 | 0.0044 | 0.0086 |
| NLL | 6.0854 | 6.2098 | 6.2500 |
| PIT KS | 0.1096 | 0.0981 | 0.0831 |
| var / err^2 Spearman | 0.2150 | 0.2283 | 0.2178 |
| coverage of the 90% interval | 0.9082 | 0.9115 | 0.9127 |
| width of the 90% interval ($) | 138.38 | 172.52 | 203.44 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0293 | [-0.0582, -0.0017] | INVERTED |
| h1 | 0.0426 | [0.0115, 0.0726] | WORKS |
| h2 | 0.0418 | [0.0133, 0.0691] | WORKS |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.268 / h1 0.000 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.7705 | n/a (beta = 0: served delta is 0) | 0.6071 |
| abs(d h1) <= abs(d h2) | 0.8392 | n/a (beta = 0: served delta is 0) | 0.5872 |
| full chain h0 <= h1 <= h2 | 0.6280 | n/a (beta = 0: served delta is 0) | 0.3254 |

beta = 0 for h1, h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve them are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.5170 | 0.8086 | 0.6629 | 0.3420 |
| expected if the two signs were independent | 0.4768 | 0.5154 | 0.4718 | 0.1820 |

- P(up) unanimity (all three horizons call the same side): 0.4198

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0381 vs 0.0878 (-0.1259): does not beat, significantly worse (boot z -3.84) | 0.0882 vs 0.1036 (-0.0154): does not beat, noise (boot z -0.62) | 0.0495 vs 0.0915 (-0.0420): does not beat, noise (boot z -1.56) |
| logreg_lags | direction/auc | 0.4713 vs 0.5646 (-0.0933): does not beat, significantly worse (boot z -4.28) | 0.5588 vs 0.5749 (-0.0160): does not beat, noise (boot z -1.06) | 0.5390 vs 0.5760 (-0.0369): does not beat, significantly worse (boot z -2.18) |
| logreg_lags | direction/brier | 0.2803 vs 0.2478 (-0.0325): does not beat, significantly worse (DM z -7.23) | 0.2502 vs 0.2466 (-0.0036): does not beat, noise (DM z -1.42) | 0.2577 vs 0.2460 (-0.0117): does not beat, significantly worse (DM z -3.63) |
| logreg_lags | direction/ece_pos | 0.1218 vs 0.0151 (-0.1067): does not beat, significantly worse (boot z -7.75) | 0.0396 vs 0.0181 (-0.0215): does not beat, noise (boot z -1.34) | 0.0641 vs 0.0083 (-0.0558): does not beat, significantly worse (boot z -4.00) |
| logreg_lags | direction/acc | 0.4822 vs 0.5446 (-0.0624): does not beat, significantly worse (DM z -3.80) | 0.5427 vs 0.5521 (-0.0094): does not beat, noise (DM z -0.69) | 0.5262 vs 0.5468 (-0.0207): does not beat, noise (DM z -1.50) |
| logreg_lags | direction/bal_acc | 0.4811 vs 0.5422 (-0.0611): does not beat, significantly worse (boot z -3.83) | 0.5437 vs 0.5505 (-0.0069): does not beat, noise (boot z -0.55) | 0.5244 vs 0.5439 (-0.0195): does not beat, noise (boot z -1.49) |
| class_prior | direction/mcc | -0.0381 vs 0.0000 (-0.0381): does not beat, noise (boot z -1.80) | 0.0882 vs 0.0000 (+0.0882): beats (boot z +3.61) | 0.0495 vs 0.0000 (+0.0495): beats (boot z +2.33) |
| class_prior | direction/auc | 0.4713 vs 0.5000 (-0.0287): does not beat, significantly worse (boot z -2.18) | 0.5588 vs 0.5000 (+0.0588): beats (boot z +3.80) | 0.5390 vs 0.5000 (+0.0390): beats (boot z +2.92) |
| class_prior | direction/brier | 0.2803 vs 0.2499 (-0.0304): does not beat, significantly worse (DM z -8.28) | 0.2502 vs 0.2500 (-0.0003): does not beat, noise (DM z -0.11) | 0.2577 vs 0.2499 (-0.0078): does not beat, significantly worse (DM z -2.70) |
| class_prior | direction/ece_pos | 0.1218 vs 0.0002 (-0.1215): does not beat, significantly worse (boot z -8.78) | 0.0396 vs 0.0002 (-0.0394): does not beat, significantly worse (boot z -3.12) | 0.0641 vs 0.0011 (-0.0630): does not beat, significantly worse (boot z -4.50) |
| class_prior | direction/acc | 0.4822 vs 0.5088 (-0.0266): does not beat, noise (DM z -1.79) | 0.5427 vs 0.5070 (+0.0357): beats, noise (DM z +1.79) | 0.5262 vs 0.5105 (+0.0157): beats, noise (DM z +0.96) |
| class_prior | direction/bal_acc | 0.4811 vs 0.5000 (-0.0189): does not beat, noise (boot z -1.80) | 0.5437 vs 0.5000 (+0.0437): beats (boot z +3.61) | 0.5244 vs 0.5000 (+0.0244): beats (boot z +2.33) |
| zero_delta | delta/rmse | 45.62 vs 45.63 (+0.01, +0.02%): beats, noise (DM z +0.48) | 55.69 vs 55.69 (+0.00, +0.00%): does not beat | 63.66 vs 63.66 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 28.74 vs 28.77 (+0.04, +0.12%): beats (DM z +2.27) | 35.11 vs 35.11 (+0.00, +0.00%): does not beat | 40.44 vs 40.44 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 45.62 vs 45.63 (+0.01, +0.01%): beats, noise (DM z +0.27) | 55.69 vs 55.68 (-0.01, -0.02%): does not beat, noise (DM z -0.48) | 63.66 vs 63.64 (-0.01, -0.02%): does not beat, noise (DM z -0.49) |
| mean_delta | delta/mae | 28.74 vs 28.77 (+0.04, +0.13%): beats (DM z +1.96) | 35.11 vs 35.12 (+0.01, +0.02%): beats, noise (DM z +0.46) | 40.44 vs 40.44 (+0.00, +0.00%): beats, noise (DM z +0.05) |
| const_var | variance/crps | 22.25 vs 22.33 (+0.09, +0.38%): beats, noise (DM z +1.14) | 27.20 vs 27.32 (+0.12, +0.44%): beats, noise (DM z +1.80) | 31.22 vs 31.49 (+0.27, +0.86%): beats (DM z +2.79) |
| const_var | variance/nll | 6.0854 vs 6.2119 (+0.1265): beats, noise (DM z +0.87) | 6.2098 vs 6.3726 (+0.1629): beats, noise (DM z +1.20) | 6.2500 vs 6.4597 (+0.2096): beats, noise (DM z +1.44) |
| const_var | variance/pit_ks | 0.1096 vs 0.0879 (-0.0217): does not beat, significantly worse (boot z -4.78) | 0.0981 vs 0.0814 (-0.0167): does not beat, significantly worse (boot z -3.43) | 0.0831 vs 0.0786 (-0.0045): does not beat, noise (boot z -0.89) |
| const_var | variance/corr_var_err2_spearman | 0.2150 vs 0.0000 (+0.2150): beats (boot z +8.19) | 0.2283 vs 0.0000 (+0.2283): beats (boot z +8.27) | 0.2178 vs 0.0000 (+0.2178): beats (boot z +7.51) |

## Backtest (costs included)

- n_trades: 796
- total_return: 0.0414
- sharpe_net: 4.3416
- sharpe_gross: 4.3416
- sortino: 6.5578
- max_drawdown: 0.0435
- hit_rate: 0.4736
- hit_rate_gross: 0.4736
- profit_factor: 1.0873
- avg_hold_bars: 8.6834
- exposure: 0.4655
- turnover: 1579.3090
- fees_paid: 0.0000
- traded_notional: 15793881.0353
- breakeven_cost_bps: 0.5237
- gross_edge_per_trade_bps: 0.5252
- costs_paid: 0.0000
- gross_pnl: 413.5841
- net_pnl: 413.5841

## Training health

11 epoch(s). Pre-clip gradient norm maximum over the run: main 143.0549, indicator 435.7000 (clip 20).
Clipped steps over the run: main 65.0000, indicator 47.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 41.0260 / 66.2103 | 1.0000 / 1.0000 | 3.4% / 3.4% | 0.0000 | 1516.0000 / 1721.0000 / 1859.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 3.6133 / 5.9636 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2038.0000 / 2299.0000 / 2479.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 8.6095 / 7.8953 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 2036.0000 / 2337.0000 / 2525.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 74.4585 / 160.5818 | 1.0000 / 4.0000 | 3.4% / 13.8% | 0.0000 | 2013.0000 / 2310.0000 / 2493.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 93.1404 / 273.3329 | 1.0000 / 3.0000 | 3.4% / 10.3% | 0.0000 | 2063.0000 / 2359.0000 / 2513.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 38.2396 / 160.2593 | 2.0000 / 3.0000 | 6.9% / 10.3% | 0.0000 | 2034.0000 / 2349.0000 / 2490.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 99.0642 / 114.4035 | 5.0000 / 6.0000 | 17.2% / 20.7% | 0.0000 | 2009.0000 / 2333.0000 / 2488.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 33.5757 / 397.3225 | 8.0000 / 3.0000 | 27.6% / 10.3% | 0.0000 | 2060.0000 / 2313.0000 / 2508.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 31.6096 / 79.7848 | 8.0000 / 9.0000 | 27.6% / 31.0% | 0.0000 | 1661.0000 / 1922.0000 / 2116.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 49.1863 / 59.5114 | 21.0000 / 6.0000 | 72.4% / 20.7% | 0.0000 | 1516.0000 / 1791.0000 / 1924.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 143.0549 / 435.7000 | 18.0000 / 12.0000 | 62.1% / 41.4% | 0.0000 | 1478.0000 / 1764.0000 / 1893.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.906 (corr skip/tower=-0.749), h1=0.116 (corr skip/tower=-0.017), h2=0.255 (corr skip/tower=-0.600).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -95 (TimeSeriesSplit fold 6, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-01-23T10:13:00 .. 2023-02-02T17:43:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.5053, long_above 0.5897, short_below 0.4046, median 0.4975. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +4.14% | +4.34 | +4.35% | 796 |
| buy and hold | +4.56% | +3.67 | +5.61% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -7.84% .. +8.68%) | +0.05% | +0.10 | | |

The random null enters at the strategy's rate (0.1003 per flat bar), holds 9 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 79% of its seeds on net return, 76% on net Sharpe and 79% on gross return.
