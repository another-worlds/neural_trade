# Evaluation report - test split - run `20260923T203258Z-6dec27a-ddfc658c-all_off__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0378 | 0.0175 | -0.0052 |
| direction AUC | 0.4885 | 0.5300 | 0.4993 |
| direction ECE | 0.0732 | 0.0522 | 0.0869 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 110.7762 | 136.9521 | 159.6151 |
| PIT KS | 0.0471 | 0.0768 | 0.0534 |
| var/err2 Spearman | 0.3767 | 0.3149 | 0.3651 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0114 | [-0.0361, 0.0660] | NOISE |
| h1 | 0.0058 | [-0.0387, 0.0540] | NOISE |
| h2 | -0.0014 | [-0.0581, 0.0549] | NOISE |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.4254
- delta_dir_align_all: 0.0162
- coherence_primary: 0.2148

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | beats | does not beat |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | beats | does not beat |
| class_prior | direction/auc | does not beat | beats | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | beats | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | beats |
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | beats | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | beats |

## Backtest (costs included)

- n_trades: 350
- total_return: -0.6151
- sharpe_net: -146.2270
- sharpe_gross: -8.7246
- sortino: -164.4300
- max_drawdown: 0.6153
- hit_rate: 0.0771
- profit_factor: 0.0308
- avg_hold_bars: 8.9971
- exposure: 0.4352
- turnover: 453.0942
- fees_paid: 4530.8706
- costs_paid: 5890.1318
- gross_pnl: -261.2607
- net_pnl: -6151.3925
