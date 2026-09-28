# Evaluation report - test split - run `20260923T205517Z-6dec27a-1322b571-all_off__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0265 | -0.0030 | 0.0599 |
| direction AUC | 0.4845 | 0.4998 | 0.5212 |
| direction ECE | 0.0383 | 0.0286 | 0.0485 |
| Gaussian MCC | 0.0412 | -0.0138 | 0.0684 |
| Gaussian AUC | 0.5083 | 0.4765 | 0.5016 |
| delta EV | 0.0023 | -0.0026 | -0.0086 |
| delta corr | 0.0591 | 0.0229 | 0.0217 |
| skill vs zero | 0.0010 | -0.0026 | -0.0153 |
| CRPS ($) | 105.8454 | 127.5346 | 145.8748 |
| PIT KS | 0.0545 | 0.0425 | 0.0573 |
| var/err2 Spearman | 0.2437 | 0.2387 | 0.2345 |
| coverage 90% | 0.9031 | 0.9059 | 0.9063 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0122 | [-0.0623, 0.0315] | NOISE |
| h1 | 0.0013 | [-0.0406, 0.0396] | NOISE |
| h2 | -0.0301 | [-0.0785, 0.0320] | NOISE |

## Coherence across horizons

- mag_order_full: 0.2188
- unanimity: 0.2906
- delta_dir_align_all: 0.2587
- coherence_primary: 0.5474

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | does not beat | does not beat |
| zero_delta | delta/mae | beats | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
| mean_delta | delta/rmse | beats | does not beat | does not beat |
| mean_delta | delta/mae | beats | does not beat | does not beat |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 221
- total_return: -0.4295
- sharpe_net: -112.8190
- sharpe_gross: 4.7068
- sortino: -130.6029
- max_drawdown: 0.4302
- hit_rate: 0.0950
- profit_factor: 0.0352
- avg_hold_bars: 9.3937
- exposure: 0.2870
- turnover: 340.9930
- fees_paid: 3409.8049
- costs_paid: 4432.7463
- gross_pnl: 138.2387
- net_pnl: -4294.5076
