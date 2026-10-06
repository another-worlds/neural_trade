# Evaluation report - dev split - run `20261006T074438Z-590829b-61336ed5-linear_indicators__f-92__s0`

n = 14851 samples, one per 1-minute bar. Direction metrics count only moves beyond 5 bps (the neutral mask). Consecutive samples share most of their target window, so n_eff = n // bars ahead counts the non-overlapping outcomes.

## Direction heads: P(up) > 0.5 on moves beyond 5 bps

Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the counts and rates are the same before and after calibration; Brier and ECE are not.

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| n scored (outside the deadband) | 9392 | 10127 | 10722 |
| n_eff of the scored moves (n scored // bars ahead) | 939 | 675 | 536 |
| true up-rate | 0.4856 | 0.4794 | 0.4808 |
| calls up (predicted up-rate) | 0.6568 | 0.4075 | 0.7429 |
| accuracy | 0.5077 | 0.4922 | 0.4982 |
| balanced accuracy | 0.5122 | 0.4884 | 0.5076 |
| precision (up) | 0.4949 | 0.4652 | 0.4859 |
| recall / sensitivity (up) | 0.6694 | 0.3955 | 0.7507 |
| specificity (down) | 0.3550 | 0.5814 | 0.2644 |
| F1 (up) | 0.5691 | 0.4275 | 0.5899 |
| MCC | 0.0257 | -0.0235 | 0.0173 |
| AUC | 0.5174 | 0.4787 | 0.5117 |
| Brier | 0.2671 | 0.2683 | 0.2662 |
| ECE (positive class) | 0.0969 | 0.0906 | 0.0979 |
| ECE of a constant 0.5 (= distance of the up-rate from 0.5) | 0.0144 | 0.0206 | 0.0192 |
| TP / FP / TN / FN | 3053 / 3116 / 1715 / 1508 | 1920 / 2207 / 3065 / 2935 | 3870 / 4095 / 1472 / 1285 |
| Gaussian readout: calls up | 0.8011 | 0.6444 | 0.5016 |
| Gaussian readout: MCC | 0.0188 | 0.0043 | 0.0360 |
| Gaussian readout: AUC | 0.5194 | 0.5174 | 0.5200 |
| Gaussian readout: Brier | 0.2500 | 0.2507 | 0.2501 |
| Gaussian readout: ECE | 0.0239 | 0.0376 | 0.0233 |

## Price heads (dollars)

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| RMSE ($), served | 43.33 | 54.32 | 63.46 |
| RMSE ($), raw heads | 43.52 | 54.35 | 63.62 |
| RMSE ($), zero prediction | 43.32 | 54.31 | 63.40 |
| MAE ($), served | 26.28 | 31.99 | 36.79 |
| MAE ($), raw heads | 26.48 | 32.05 | 36.87 |
| MAE ($), zero prediction | 26.28 | 31.98 | 36.80 |
| skill vs zero (1 - MSE / MSE of 0), served | -0.0004 | -0.0002 | -0.0017 |
| skill vs zero, raw heads | -0.0089 | -0.0016 | -0.0069 |
| EV, served | 0.0001 | 0.0011 | -0.0015 |
| EV, raw heads | -0.0035 | 0.0008 | -0.0064 |
| corr, Pearson, raw heads (the same for served while beta > 0) | 0.0120 | 0.0339 | -0.0154 |
| corr, Spearman, raw heads | 0.0262 | 0.0294 | 0.0235 |
| mean predicted ($), served | 0.31 | 0.78 | 0.13 |
| mean predicted ($), raw heads | 2.08 | 1.31 | 0.33 |
| mean realised ($) | -1.39 | -2.08 | -2.78 |
| share predicted up, raw heads | 0.8143 | 0.6621 | 0.5004 |
| share realised up | 0.4734 | 0.4754 | 0.4770 |
| shrink beta (served = beta x raw, fit on cal) | 0.1471 | 0.5954 | 0.3995 |

Correlations and the share predicted up are the raw heads': the served delta, beta x raw, has the same while beta > 0.

## Variance heads

| metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|
| CRPS ($) | 20.58 | 24.96 | 28.92 |
| CRPSS vs constant variance | -0.0010 | 0.0050 | -0.0024 |
| NLL | 6.5094 | 6.4089 | 7.0273 |
| PIT KS | 0.0816 | 0.0689 | 0.0807 |
| var / err^2 Spearman | 0.0220 | 0.1919 | 0.1193 |
| coverage of the 90% interval | 0.9023 | 0.9000 | 0.8964 |
| width of the 90% interval ($) | 120.21 | 145.86 | 166.44 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0107 | [-0.0200, 0.0447] | NOISE |
| h1 | -0.0280 | [-0.0584, 0.0010] | NOISE |
| h2 | -0.0167 | [-0.0536, 0.0201] | NOISE |

## Coherence across horizons

Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.

| check | raw price heads (the trained ordering) | served deltas (beta-shrunk: h0 0.147 / h1 0.595 / h2 0.399) | realised moves |
|---|---|---|---|
| abs(d h0) <= abs(d h1) | 0.3876 | 0.8160 | 0.5983 |
| abs(d h1) <= abs(d h2) | 0.5656 | 0.4473 | 0.5786 |
| full chain h0 <= h1 <= h2 | 0.1588 | 0.2945 | 0.3054 |

The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the trained constraint on the raw heads.

Sign agreement: sign(raw price head) against calibrated P(up) > 0.5 (the served delta, beta x raw, has the same sign while beta > 0).

| | h0 | h1 | h2 | all 3 |
|---|---|---|---|---|
| agree | 0.6451 | 0.4555 | 0.4970 | 0.1448 |
| expected if the two signs were independent | 0.5988 | 0.4751 | 0.5002 | 0.1485 |

- P(up) unanimity (all three horizons call the same side): 0.3943

## Against baselines (fit on the training block)

Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "noise": |z| < 1.96, so the ordering is not established; "significantly worse": the model loses with z <= -1.96. "DM z": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag 2 x bars ahead. "boot z": the margin over its standard error in a paired moving-block bootstrap (80-bar blocks, 500 resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, var / err^2 Spearman). Rows that only restate the RMSE verdict for a constant prediction (EV, corr and skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.

| baseline | metric | h0 (10 bars) | h1 (15 bars) | h2 (20 bars) |
|---|---|---|---|---|
| logreg_lags | direction/mcc | 0.0257 vs 0.0646 (-0.0389): does not beat, noise (boot z -1.68) | -0.0235 vs 0.0750 (-0.0986): does not beat, significantly worse (boot z -3.49) | 0.0173 vs 0.0755 (-0.0582): does not beat, significantly worse (boot z -2.24) |
| logreg_lags | direction/auc | 0.5174 vs 0.5596 (-0.0422): does not beat, significantly worse (boot z -2.84) | 0.4787 vs 0.5695 (-0.0909): does not beat, significantly worse (boot z -4.65) | 0.5117 vs 0.5730 (-0.0613): does not beat, significantly worse (boot z -3.73) |
| logreg_lags | direction/brier | 0.2671 vs 0.2488 (-0.0183): does not beat, significantly worse (DM z -5.87) | 0.2683 vs 0.2491 (-0.0192): does not beat, significantly worse (DM z -5.09) | 0.2662 vs 0.2489 (-0.0173): does not beat, significantly worse (DM z -5.34) |
| logreg_lags | direction/ece_pos | 0.0969 vs 0.0315 (-0.0654): does not beat, significantly worse (boot z -5.76) | 0.0906 vs 0.0409 (-0.0497): does not beat, significantly worse (boot z -2.72) | 0.0979 vs 0.0445 (-0.0534): does not beat, significantly worse (boot z -6.83) |
| logreg_lags | direction/acc | 0.5077 vs 0.5244 (-0.0167): does not beat, noise (DM z -1.49) | 0.4922 vs 0.5260 (-0.0338): does not beat, noise (DM z -1.96) | 0.4982 vs 0.5237 (-0.0255): does not beat, significantly worse (DM z -2.19) |
| logreg_lags | direction/bal_acc | 0.5122 vs 0.5299 (-0.0177): does not beat, noise (boot z -1.63) | 0.4884 vs 0.5344 (-0.0459): does not beat, significantly worse (boot z -3.43) | 0.5076 vs 0.5330 (-0.0255): does not beat, significantly worse (boot z -2.23) |
| class_prior | direction/mcc | 0.0257 vs 0.0000 (+0.0257): beats, noise (boot z +1.44) | -0.0235 vs 0.0000 (-0.0235): does not beat, noise (boot z -1.32) | 0.0173 vs 0.0000 (+0.0173): beats, noise (boot z +0.79) |
| class_prior | direction/auc | 0.5174 vs 0.5000 (+0.0174): beats, noise (boot z +1.53) | 0.4787 vs 0.5000 (-0.0213): does not beat, noise (boot z -1.75) | 0.5117 vs 0.5000 (+0.0117): beats, noise (boot z +0.78) |
| class_prior | direction/brier | 0.2671 vs 0.2505 (-0.0166): does not beat, significantly worse (DM z -5.07) | 0.2683 vs 0.2506 (-0.0177): does not beat, significantly worse (DM z -5.99) | 0.2662 vs 0.2507 (-0.0155): does not beat, significantly worse (DM z -5.12) |
| class_prior | direction/ece_pos | 0.0969 vs 0.0260 (-0.0708): does not beat, significantly worse (boot z -6.31) | 0.0906 vs 0.0322 (-0.0583): does not beat, significantly worse (boot z -3.29) | 0.0979 vs 0.0332 (-0.0647): does not beat, significantly worse (boot z -8.54) |
| class_prior | direction/acc | 0.5077 vs 0.4856 (+0.0220): beats, noise (DM z +1.76) | 0.4922 vs 0.4794 (+0.0128): beats, noise (DM z +0.64) | 0.4982 vs 0.4808 (+0.0174): beats, noise (DM z +1.34) |
| class_prior | direction/bal_acc | 0.5122 vs 0.5000 (+0.0122): beats, noise (boot z +1.44) | 0.4884 vs 0.5000 (-0.0116): does not beat, noise (boot z -1.32) | 0.5076 vs 0.5000 (+0.0076): beats, noise (boot z +0.79) |
| zero_delta | delta/rmse | 43.33 vs 43.32 (-0.01, -0.02%): does not beat, noise (DM z -0.79) | 54.32 vs 54.31 (-0.00, -0.01%): does not beat, noise (DM z -0.12) | 63.46 vs 63.40 (-0.05, -0.08%): does not beat, noise (DM z -0.88) |
| zero_delta | delta/mae | 26.28 vs 26.28 (-0.00, -0.01%): does not beat, noise (DM z -0.36) | 31.99 vs 31.98 (-0.01, -0.03%): does not beat, noise (DM z -0.26) | 36.79 vs 36.80 (+0.00, +0.01%): beats, noise (DM z +0.15) |
| mean_delta | delta/rmse | 43.33 vs 43.34 (+0.00, +0.01%): beats, noise (DM z +0.51) | 54.32 vs 54.33 (+0.02, +0.03%): beats, noise (DM z +0.42) | 63.46 vs 63.43 (-0.02, -0.03%): does not beat, noise (DM z -0.41) |
| mean_delta | delta/mae | 26.28 vs 26.29 (+0.01, +0.05%): beats (DM z +2.26) | 31.99 vs 32.01 (+0.02, +0.05%): beats, noise (DM z +0.54) | 36.79 vs 36.83 (+0.04, +0.10%): beats, noise (DM z +1.01) |
| const_var | variance/crps | 20.58 vs 20.56 (-0.02, -0.10%): does not beat, noise (DM z -1.58) | 24.96 vs 25.09 (+0.12, +0.50%): beats (DM z +4.67) | 28.92 vs 28.85 (-0.07, -0.24%): does not beat, noise (DM z -1.88) |
| const_var | variance/nll | 6.5094 vs 6.2346 (-0.2748): does not beat, noise (DM z -1.89) | 6.4089 vs 6.5179 (+0.1091): beats (DM z +4.74) | 7.0273 vs 6.6968 (-0.3305): does not beat, significantly worse (DM z -1.96) |
| const_var | variance/pit_ks | 0.0816 vs 0.0828 (+0.0012): beats, noise (boot z +0.87) | 0.0689 vs 0.0747 (+0.0058): beats (boot z +4.24) | 0.0807 vs 0.0725 (-0.0081): does not beat, significantly worse (boot z -5.74) |
| const_var | variance/corr_var_err2_spearman | 0.0220 vs 0.0000 (+0.0220): beats, noise (boot z +1.00) | 0.1919 vs 0.0000 (+0.1919): beats (boot z +8.24) | 0.1193 vs 0.0000 (+0.1193): beats (boot z +4.45) |

## Backtest (costs included)

- n_trades: 435
- total_return: 0.0280
- sharpe_net: 3.4788
- sharpe_gross: 3.4788
- sortino: 5.3759
- max_drawdown: 0.0419
- hit_rate: 0.4598
- hit_rate_gross: 0.4598
- profit_factor: 1.0816
- avg_hold_bars: 10.2644
- exposure: 0.3007
- turnover: 872.6647
- fees_paid: 0.0000
- traded_notional: 8726127.8347
- breakeven_cost_bps: 0.6429
- gross_edge_per_trade_bps: 0.6713
- costs_paid: 0.0000
- gross_pnl: 280.4996
- net_pnl: 280.4996

## Training health

2 epoch(s). Pre-clip gradient norm maximum over the run: main 30.6503, indicator 27.6818 (clip 20).
Clipped steps over the run: main 4.0000, indicator 1.0000.
Non-finite training steps over the run (the finite-gradient guard fired): 0.0000.

| epoch | grad norm max (main / indicator) | clipped steps (main / indicator) | clipped share (main / indicator) | non-finite steps | dir_n (h0/h1/h2) | var@floor (h0/h1/h2) |
|---|---|---|---|---|---|---|
| 0 | 30.6503 / 27.6818 | 4.0000 / 1.0000 | 5.6% / 1.4% | 0.0000 | 3512.0000 / 4092.0000 / 4440.0000 | 0.0000 / 0.0000 / 0.0000 |
| 1 | 3.8296 / 13.1945 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 | 3452.0000 / 4067.0000 / 4454.0000 | 0.0000 / 0.0000 / 0.0000 |

No loss term ever masked a non-finite value.
DIRECTION_SKIP logit's covariance share of the direction-logit variance (validation block, skip_share + tower_share = 1, NT-110): h0=1.092 (corr skip/tower=-0.590), h1=1.122 (corr skip/tower=-0.596), h2=1.090 (corr skip/tower=-0.538).

## Experiment engine: out-of-sample block and backtest

Role: **dev** (rows rank on it). Fold -92 (TimeSeriesSplit fold 9, 96 usable folds); this report scores the fold's out-of-sample block: 14851 sequences, 2023-02-23T08:46:00 .. 2023-03-05T16:16:00.

Strategy `calibrated_quantile`, knobs fitted on the calibration block only: var_scale 0.9961, long_above 0.6239, short_below 0.4301, median 0.5200. Costs per side: fee 0.0 bps + half-spread 0.0 bps + slippage 0.0 bps; fills at the next bar's open (next_open); stops on high_low; Sharpe annualised over 525600 periods per year at 1.0-minute bars (calendar '24/7').

| | net return | net Sharpe | max drawdown | trades |
|---|---|---|---|---|
| strategy | +2.80% | +3.48 | +4.19% | 435 |
| buy and hold | -8.52% | -7.39 | +9.73% | 1 |
| always flat | +0.00% | +0.00 | +0.00% | 0 |
| random null, mean of 100 seeds (p05 .. p95 of the net return: -7.20% .. +5.66%) | +0.10% | +0.12 | | |

The random null enters at the strategy's rate (0.0419 per flat bar), holds 10 bars and sizes each entry at the strategy's mean position size (1.000). The strategy beats 72% of its seeds on net return, 68% on net Sharpe and 72% on gross return.
