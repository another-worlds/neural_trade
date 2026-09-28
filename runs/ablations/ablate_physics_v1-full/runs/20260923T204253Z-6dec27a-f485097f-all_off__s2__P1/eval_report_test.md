# Evaluation report - test split - run `20260923T204253Z-6dec27a-f485097f-all_off__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0075 | -0.0395 | 0.0264 |
| direction AUC | 0.4850 | 0.4599 | 0.5330 |
| direction ECE | 0.0582 | 0.0940 | 0.0627 |
| Gaussian MCC | 0.0348 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5257 | 0.5000 | 0.5000 |
| delta EV | 0.0013 | 0.0000 | 0.0000 |
| delta corr | 0.0392 | 0.0000 | 0.0000 |
| skill vs zero | 0.0006 | 0.0000 | 0.0000 |
| CRPS ($) | 110.1265 | 135.2120 | 158.9634 |
| PIT KS | 0.0430 | 0.0435 | 0.0471 |
| var/err2 Spearman | 0.3797 | 0.3718 | 0.3539 |
| coverage 90% | 0.9230 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0278 | [-0.0857, 0.0247] | NOISE |
| h1 | -0.0669 | [-0.1123, -0.0148] | INVERTED |
| h2 | 0.0250 | [-0.0227, 0.0706] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.6481
- delta_dir_align_all: 0.0325
- coherence_primary: 0.0760

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | does not beat | does not beat |
| zero_delta | delta/mae | beats | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 253
- total_return: -0.4726
- sharpe_net: -105.7332
- sharpe_gross: 3.9453
- sortino: -127.0377
- max_drawdown: 0.4726
- hit_rate: 0.1067
- profit_factor: 0.0840
- avg_hold_bars: 12.2292
- exposure: 0.4276
- turnover: 373.1439
- fees_paid: 3731.2484
- costs_paid: 4850.6229
- gross_pnl: 124.1952
- net_pnl: -4726.4278
