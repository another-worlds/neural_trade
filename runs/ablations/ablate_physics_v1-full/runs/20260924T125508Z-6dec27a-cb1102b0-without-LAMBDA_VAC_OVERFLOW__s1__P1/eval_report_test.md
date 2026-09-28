# Evaluation report - test split - run `20260924T125508Z-6dec27a-cb1102b0-without-LAMBDA_VAC_OVERFLOW__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0335 | 0.0003 | 0.0054 |
| direction AUC | 0.4932 | 0.4777 | 0.4703 |
| direction ECE | 0.0593 | 0.0875 | 0.0658 |
| Gaussian MCC | -0.0004 | -0.0243 | 0.0000 |
| Gaussian AUC | 0.4813 | 0.4740 | 0.5000 |
| delta EV | -0.0026 | 0.0016 | 0.0000 |
| delta corr | 0.0308 | 0.0465 | 0.0000 |
| skill vs zero | -0.0146 | -0.0042 | 0.0000 |
| CRPS ($) | 111.5769 | 136.4087 | 158.9510 |
| PIT KS | 0.0717 | 0.0672 | 0.0640 |
| var/err2 Spearman | 0.4011 | 0.4042 | 0.4043 |
| coverage 90% | 0.9208 | 0.9230 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0423 | [-0.0142, 0.0901] | NOISE |
| h1 | -0.0431 | [-0.0920, 0.0091] | NOISE |
| h2 | -0.0588 | [-0.1242, -0.0017] | INVERTED |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.6295
- delta_dir_align_all: 0.0343
- coherence_primary: 0.8777

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | beats | does not beat |
| zero_delta | delta/corr | beats | beats | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | beats |
| mean_delta | delta/mae | does not beat | does not beat | beats |
| mean_delta | delta/ev | does not beat | beats | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| class_prior | direction/mcc | does not beat | beats | beats |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | beats |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | beats |

## Backtest (costs included)

- n_trades: 305
- total_return: -0.5482
- sharpe_net: -125.9784
- sharpe_gross: -0.1862
- sortino: -146.9808
- max_drawdown: 0.5482
- hit_rate: 0.0918
- profit_factor: 0.0351
- avg_hold_bars: 12.2328
- exposure: 0.5156
- turnover: 421.0119
- fees_paid: 4209.9925
- costs_paid: 5472.9902
- gross_pnl: -8.7287
- net_pnl: -5481.7190
