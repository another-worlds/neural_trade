# Evaluation report - test split - run `20260923T225652Z-6dec27a-716e2a67-only-LAMBDA_IFE__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0105 | -0.0029 | 0.0069 |
| direction AUC | 0.4922 | 0.5248 | 0.5132 |
| direction ECE | 0.0585 | 0.0589 | 0.0764 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 110.2768 | 136.1740 | 160.2105 |
| PIT KS | 0.0491 | 0.0754 | 0.0722 |
| var/err2 Spearman | 0.3708 | 0.3311 | 0.3208 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0007 | [-0.0433, 0.0435] | NOISE |
| h1 | 0.0222 | [-0.0212, 0.0659] | NOISE |
| h2 | 0.0005 | [-0.0474, 0.0480] | NOISE |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.4538
- delta_dir_align_all: 0.0224
- coherence_primary: 0.1515

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
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | does not beat | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | beats |
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 338
- total_return: -0.6054
- sharpe_net: -142.5534
- sharpe_gross: -9.2732
- sortino: -162.2418
- max_drawdown: 0.6058
- hit_rate: 0.0740
- profit_factor: 0.0329
- avg_hold_bars: 9.2663
- exposure: 0.4328
- turnover: 443.4997
- fees_paid: 4434.9310
- costs_paid: 5765.4103
- gross_pnl: -288.1800
- net_pnl: -6053.5903
