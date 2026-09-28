# Evaluation report - test split - run `20260924T013613Z-6dec27a-d35ba113-only-LAMBDA_VAC__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0457 | -0.0284 | 0.0015 |
| direction AUC | 0.4629 | 0.4870 | 0.5048 |
| direction ECE | 0.0759 | 0.0798 | 0.0506 |
| Gaussian MCC | 0.0243 | 0.0250 | 0.0374 |
| Gaussian AUC | 0.5131 | 0.5213 | 0.5259 |
| delta EV | 0.0051 | 0.0010 | 0.0031 |
| delta corr | 0.0722 | 0.0757 | 0.0717 |
| skill vs zero | 0.0045 | 0.0010 | 0.0032 |
| CRPS ($) | 109.1802 | 134.8287 | 157.9002 |
| PIT KS | 0.0480 | 0.0621 | 0.0577 |
| var/err2 Spearman | 0.4291 | 0.3683 | 0.3758 |
| coverage 90% | 0.9233 | 0.9283 | 0.9256 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0539 | [-0.0957, -0.0114] | INVERTED |
| h1 | -0.0248 | [-0.0639, 0.0169] | NOISE |
| h2 | -0.0068 | [-0.0579, 0.0478] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0416
- unanimity: 0.4956
- delta_dir_align_all: 0.3021
- coherence_primary: 0.5579

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | beats |
| zero_delta | delta/mae | beats | beats | beats |
| zero_delta | delta/ev | beats | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | beats | beats | beats |
| mean_delta | delta/corr | beats | beats | beats |
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
| logreg_lags | direction/mcc | does not beat | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | does not beat | beats | beats |

## Backtest (costs included)

- n_trades: 331
- total_return: -0.5876
- sharpe_net: -141.4403
- sharpe_gross: -3.2739
- sortino: -161.3473
- max_drawdown: 0.5876
- hit_rate: 0.0785
- profit_factor: 0.0338
- avg_hold_bars: 9.2236
- exposure: 0.4219
- turnover: 444.4341
- fees_paid: 4444.0685
- costs_paid: 5777.2891
- gross_pnl: -98.6112
- net_pnl: -5875.9003
