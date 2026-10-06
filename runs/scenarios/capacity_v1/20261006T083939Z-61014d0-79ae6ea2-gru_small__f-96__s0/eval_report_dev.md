# Evaluation report - dev split - run `20261006T083939Z-61014d0-79ae6ea2-gru_small__f-96__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 10701 | 11457 | 11840 |
| n_eff of the scored moves (n scored // bars ahead) | 1070 | 763 | 592 |
| true up-rate | 0.5381 | 0.5412 | 0.5428 |
| calls up (predicted up-rate) | 0.3752 | 0.2580 | 0.2992 |
| accuracy | 0.4889 | 0.4805 | 0.5030 |
| balanced accuracy | 0.4984 | 0.5004 | 0.5204 |
| precision (up) | 0.5360 | 0.5419 | 0.5766 |
| recall / sensitivity (up) | 0.3737 | 0.2584 | 0.3179 |
| specificity (down) | 0.6231 | 0.7424 | 0.7229 |
| F1 (up) | 0.4404 | 0.3499 | 0.4098 |
| MCC | -0.0032 | 0.0009 | 0.0443 |
| AUC | 0.4989 | 0.5030 | 0.5227 |
| Brier | 0.2748 | 0.2764 | 0.2774 |
| ECE (positive class) | 0.1184 | 0.1318 | 0.1391 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0381 | 0.0412 | 0.0428 |
| TP / FP / TN / FN | 2152 / 1863 / 3080 / 3606 | 1602 / 1354 / 3903 / 4598 | 2043 / 1500 / 3913 / 4384 |
| Gaussian readout: calls up | 0.4376 | 0.5511 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: MCC | 0.0197 | 0.0053 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: AUC | 0.5166 | 0.5035 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: Brier | 0.2500 | 0.2500 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout: ECE | 0.0383 | 0.0406 | n/a (beta = 0: readout is the constant 0.5) |
| Gaussian readout of the raw heads: calls up | 0.4376 | 0.5511 | 0.4219 |
| Gaussian readout of the raw heads: MCC | 0.0197 | 0.0053 | 0.0383 |
| Gaussian readout of the raw heads: AUC | 0.5166 | 0.5034 | 0.5225 |
| Gaussian readout of the raw heads: Brier | 0.2573 | 0.2575 | 0.2656 |
| Gaussian readout of the raw heads: ECE | 0.0707 | 0.0735 | 0.1007 |

beta = 0 for h2: the served delta is 0 there, so its Gaussian readout P(up | the move leaves the deadband) is the constant 0.5: it calls up on no bar and has AUC 0.5 and Brier 0.25 by construction, so those cells are n/a. The rows "Gaussian readout of the raw heads" score the raw price heads' readout (with the served variance).

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 53.79 | 65.65 | 76.27 |
| RMSE ($), raw heads | 54.04 | 65.78 | 76.88 |
| RMSE ($), zero prediction | 53.79 | 65.65 | 76.27 |
| MAE ($), served | 31.90 | 39.54 | 45.92 |
| MAE ($), raw heads | 32.11 | 39.84 | 46.47 |
| MAE ($), zero prediction | 31.90 | 39.54 | 45.92 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0001 | 0.0001 | n/a (beta = 0: served delta is 0) |
| skill vs zero, raw heads | -0.0095 | -0.0038 | -0.0159 |
| EV, served | -0.0001 | 0.0001 | n/a (beta = 0: served delta is 0) |
| EV, raw heads | -0.0091 | -0.0052 | -0.0122 |
| corr, Pearson, raw heads (the same for served while beta > 0) | -0.0169 | 0.0120 | 0.0005 |
| corr, Spearman, raw heads | 0.0137 | 0.0089 | 0.0303 |
| mean predicted ($), served | -0.00 | 0.03 | 0.00 |
| mean predicted ($), raw heads | -0.22 | 0.86 | -1.77 |
| mean realised ($) | 2.61 | 3.92 | 5.24 |
| share predicted up, raw heads | 0.4262 | 0.5424 | 0.4205 |
| share realised up | 0.5262 | 0.5317 | 0.5289 |
| shrink beta (served = beta x raw, fit on cal) | 0.0179 | 0.0329 | 0.0000 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0. beta = 0 for h2: the served delta is 0 there, the zero prediction (its errors are the zero prediction's, its skill vs zero and EV 0 by construction: n/a), so it has no correlation and no sign of its own.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 24.94 | 30.79 | 35.99 |
| CRPSS vs constant variance | -0.0057 | 0.0000 | -0.0055 |
| NLL | 6.4584 | 6.4373 | 6.7603 |
| PIT KS | 0.1081 | 0.0944 | 0.1058 |
| var / err^2 Spearman | 0.1194 | 0.1784 | 0.1402 |
| coverage of the 90% interval | 0.9318 | 0.9341 | 0.9329 |
| width of the 90% interval ($) | 172.18 | 218.23 | 258.11 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0097 | [-0.0337, 0.0169] | NOISE |
| h1 | -0.0197 | [-0.0470, 0.0044] | NOISE |
| h2 | -0.0193 | [-0.0501, 0.0106] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.018 / h1 0.033 / h2 0.000) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.6341 | 0.8034 | 0.6136 |
| abs(d h1) <= abs(d h2) | 0.7125 | n/a (beta = 0: served delta is 0) | 0.5890 |
| full chain h0 <= h1 <= h2 | 0.3890 | n/a (beta = 0: served delta is 0) | 0.3253 |

beta = 0 for h2: the served delta is 0 there, so it has no magnitude ordering (|0| <= |0| holds on every bar by ties) and no sign; the served checks that involve it are n/a; the sign checks below use the raw heads.

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6311 | 0.5763 | 0.6482 | 0.3095 |
| expected if the two signs were independent | 0.5193 | 0.4797 | 0.5315 | 0.1948 |

- P(up) unanimity (all three horizons call the same side): 0.4625

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | -0.0032 vs 0.0451 (-0.0484): does not beat, significantly worse (boot z -1.98) | 0.0009 vs 0.0522 (-0.0513): does not beat, significantly worse (boot z -2.11) | 0.0443 vs 0.0519 (-0.0075): does not beat, noise (boot z -0.25) |
| logreg_lags | direction/auc | 0.4989 vs 0.5317 (-0.0329): does not beat, significantly worse (boot z -2.04) | 0.5030 vs 0.5327 (-0.0298): does not beat, significantly worse (boot z -1.97) | 0.5227 vs 0.5366 (-0.0139): does not beat, noise (boot z -0.78) |
| logreg_lags | direction/brier | 0.2748 vs 0.2535 (-0.0213): does not beat, significantly worse (DM z -6.04) | 0.2764 vs 0.2535 (-0.0229): does not beat, significantly worse (DM z -6.50) | 0.2774 vs 0.2522 (-0.0252): does not beat, significantly worse (DM z -5.48) |
| logreg_lags | direction/ece_pos | 0.1184 vs 0.0381 (-0.0804): does not beat, significantly worse (boot z -9.05) | 0.1318 vs 0.0453 (-0.0865): does not beat, significantly worse (boot z -12.44) | 0.1391 vs 0.0445 (-0.0946): does not beat, significantly worse (boot z -11.13) |
| logreg_lags | direction/acc | 0.4889 vs 0.5300 (-0.0411): does not beat, significantly worse (DM z -3.30) | 0.4805 vs 0.5360 (-0.0555): does not beat, significantly worse (DM z -3.48) | 0.5030 vs 0.5375 (-0.0345): does not beat, noise (DM z -1.83) |
| logreg_lags | direction/bal_acc | 0.4984 vs 0.5221 (-0.0237): does not beat, significantly worse (boot z -1.98) | 0.5004 vs 0.5253 (-0.0249): does not beat, significantly worse (boot z -2.15) | 0.5204 vs 0.5248 (-0.0044): does not beat, noise (boot z -0.31) |
| class_prior | direction/mcc | -0.0032 vs 0.0000 (-0.0032): does not beat, noise (boot z -0.21) | 0.0009 vs 0.0000 (+0.0009): beats, noise (boot z +0.06) | 0.0443 vs 0.0000 (+0.0443): beats (boot z +2.20) |
| class_prior | direction/auc | 0.4989 vs 0.5000 (-0.0011): does not beat, noise (boot z -0.11) | 0.5030 vs 0.5000 (+0.0030): beats, noise (boot z +0.26) | 0.5227 vs 0.5000 (+0.0227): beats, noise (boot z +1.65) |
| class_prior | direction/brier | 0.2748 vs 0.2494 (-0.0254): does not beat, significantly worse (DM z -8.07) | 0.2764 vs 0.2491 (-0.0273): does not beat, significantly worse (DM z -7.24) | 0.2774 vs 0.2489 (-0.0285): does not beat, significantly worse (DM z -5.74) |
| class_prior | direction/ece_pos | 0.1184 vs 0.0286 (-0.0899): does not beat, significantly worse (boot z -9.80) | 0.1318 vs 0.0280 (-0.1037): does not beat, significantly worse (boot z -15.22) | 0.1391 vs 0.0278 (-0.1113): does not beat, significantly worse (boot z -13.26) |
| class_prior | direction/acc | 0.4889 vs 0.5381 (-0.0492): does not beat, significantly worse (DM z -2.82) | 0.4805 vs 0.5412 (-0.0607): does not beat, significantly worse (DM z -2.73) | 0.5030 vs 0.5428 (-0.0398): does not beat, noise (DM z -1.59) |
| class_prior | direction/bal_acc | 0.4984 vs 0.5000 (-0.0016): does not beat, noise (boot z -0.21) | 0.5004 vs 0.5000 (+0.0004): beats, noise (boot z +0.06) | 0.5204 vs 0.5000 (+0.0204): beats (boot z +2.19) |
| zero_delta | delta/rmse | 53.79 vs 53.79 (-0.00, -0.00%): does not beat, noise (DM z -0.59) | 65.65 vs 65.65 (+0.00, +0.01%): beats, noise (DM z +0.80) | 76.27 vs 76.27 (+0.00, +0.00%): does not beat |
| zero_delta | delta/mae | 31.90 vs 31.90 (-0.00, -0.00%): does not beat, noise (DM z -0.80) | 39.54 vs 39.54 (-0.00, -0.00%): does not beat, noise (DM z -0.08) | 45.92 vs 45.92 (+0.00, +0.00%): does not beat |
| mean_delta | delta/rmse | 53.79 vs 53.75 (-0.04, -0.07%): does not beat, noise (DM z -1.73) | 65.65 vs 65.59 (-0.06, -0.09%): does not beat, noise (DM z -1.67) | 76.27 vs 76.18 (-0.10, -0.13%): does not beat, noise (DM z -1.68) |
| mean_delta | delta/mae | 31.90 vs 31.86 (-0.05, -0.15%): does not beat, significantly worse (DM z -2.69) | 39.54 vs 39.46 (-0.08, -0.19%): does not beat, significantly worse (DM z -2.50) | 45.92 vs 45.83 (-0.09, -0.19%): does not beat, noise (DM z -1.89) |
| const_var | variance/crps | 24.94 vs 24.80 (-0.14, -0.57%): does not beat, significantly worse (DM z -3.84) | 30.79 vs 30.79 (+0.00, +0.00%): beats, noise (DM z +0.01) | 35.99 vs 35.80 (-0.20, -0.55%): does not beat, significantly worse (DM z -2.79) |
| const_var | variance/nll | 6.4584 vs 6.4212 (-0.0372): does not beat, noise (DM z -0.27) | 6.4373 vs 6.5631 (+0.1258): beats, noise (DM z +1.26) | 6.7603 vs 6.6903 (-0.0701): does not beat, noise (DM z -0.53) |
| const_var | variance/pit_ks | 0.1081 vs 0.0754 (-0.0327): does not beat, significantly worse (boot z -10.23) | 0.0944 vs 0.0767 (-0.0178): does not beat, significantly worse (boot z -5.60) | 0.1058 vs 0.0747 (-0.0311): does not beat, significantly worse (boot z -8.84) |
| const_var | variance/corr_var_err2_spearman | 0.1194 vs 0.0000 (+0.1194): beats (boot z +5.43) | 0.1784 vs 0.0000 (+0.1784): beats (boot z +7.37) | 0.1402 vs 0.0000 (+0.1402): beats (boot z +5.58) |

## Backtest (costs included)

- n_trades: 856
- total_return: -0.0335
- sharpe_net: -2.3037
- sharpe_gross: -2.3037
- sortino: -3.6889
- max_drawdown: 0.0830
- hit_rate: 0.4463
- hit_rate_gross: 0.4463
- profit_factor: 0.9536
- avg_hold_bars: 11.3400
- exposure: 0.6537
- turnover: 1717.6593
- fees_paid: 0.0000
- traded_notional: 17178078.3215
- breakeven_cost_bps: -0.3905
- gross_edge_per_trade_bps: -0.3633
- costs_paid: 0.0000
- gross_pnl: -335.4100
- net_pnl: -335.4100

## Training health

14 epoch(s). Pre-clip gradient norm maximum over the run: main 376.4302, indicator 661.5818 (clip 20).
Clipped steps over the run: main 8.0000, indicator 9.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 376.4302 / 661.5818 | 5.0000 / 2.0000 | 35.7% / 14.3% | 0.0000 | 1150.0000 / 1301.0000 / 1363.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 5.5343 / 18.2887 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1128.0000 / 1245.0000 / 1386.0000 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 2.6545 / 6.8846 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1749.0000 / 1938.0000 / 2098.0000 | 0.0000 / 0.0000 / 0.0000 |
| 3 | 2.8853 / 12.2191 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1147.0000 / 1305.0000 / 1377.0000 | 0.0000 / 0.0000 / 0.0000 |
| 4 | 9.8484 / 24.6722 | 0.0000 / 1.0000 | 0.0% / 7.1% | 0.0000 | 1144.0000 / 1304.0000 / 1386.0000 | 0.0000 / 0.0000 / 0.0000 |
| 5 | 3.4318 / 42.6970 | 0.0000 / 1.0000 | 0.0% / 7.1% | 0.0000 | 1169.0000 / 1293.0000 / 1397.0000 | 0.0000 / 0.0000 / 0.0000 |
| 6 | 9.1685 / 45.0136 | 0.0000 / 1.0000 | 0.0% / 7.1% | 0.0000 | 1129.0000 / 1279.0000 / 1371.0000 | 0.0000 / 0.0000 / 0.0000 |
| 7 | 3.9539 / 5.0382 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1679.0000 / 1925.0000 / 2046.0000 | 0.0000 / 0.0000 / 0.0000 |
| 8 | 6.2679 / 4.9801 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1196.0000 / 1305.0000 / 1390.0000 | 0.0000 / 0.0000 / 0.0000 |
| 9 | 11.8817 / 32.4584 | 0.0000 / 1.0000 | 0.0% / 7.1% | 0.0000 | 1173.0000 / 1292.0000 / 1376.0000 | 0.0000 / 0.0000 / 0.0000 |
| 10 | 5.4667 / 15.9603 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1104.0000 / 1312.0000 / 1384.0000 | 0.0000 / 0.0000 / 0.0000 |
| 11 | 108.7910 / 411.1827 | 2.0000 / 2.0000 | 14.3% / 14.3% | 0.0000 | 1152.0000 / 1279.0000 / 1408.0000 | 0.0000 / 0.0000 / 0.0000 |
| 12 | 7.5971 / 8.1463 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 1715.0000 / 1932.0000 / 2062.0000 | 0.0000 / 0.0000 / 0.0000 |
| 13 | 118.4393 / 215.5529 | 1.0000 / 1.0000 | 7.1% / 7.1% | 0.0000 | 1129.0000 / 1310.0000 / 1399.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=0.680 (corr skip/tower=-0.581), h1=0.754 (corr skip/tower=-0.234), h2=0.842 (corr skip/tower=-0.405).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -96 (TimeSeriesSplit fold 5, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-01-13T02:42:00 .. 2023-01-23T10:12:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.6601, long_above 0.6129, short_below 0.4476, median 0.5318. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | -3.35% | -2.30 | +8.30% | 856 |
| buy and hold | +20.64% | +11.65 | +5.41% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -10.36% .. +13.81%) | +0.53% | +0.40 | | |

The random null enters at the strategy's rate (0.1664 per flat bar), holds 11 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 35% of its seeds on net return, 35% on net Sharpe and 35% on gross return.
