# Evaluation report - test split - run `20260924T020800Z-6dec27a-1d21ad46-without-LAMBDA_T_PERP__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0144 | -0.0326 | -0.0079 |
| direction AUC | 0.5029 | 0.4841 | 0.4987 |
| direction ECE | 0.0666 | 0.0992 | 0.0676 |
| Gaussian MCC | 0.0148 | -0.0017 | 0.0008 |
| Gaussian AUC | 0.5048 | 0.5190 | 0.5327 |
| delta EV | 0.0015 | -0.0032 | 0.0013 |
| delta corr | 0.0542 | 0.0366 | 0.0370 |
| skill vs zero | -0.0023 | -0.0050 | 0.0009 |
| CRPS ($) | 109.9574 | 135.5131 | 158.7984 |
| PIT KS | 0.0542 | 0.0529 | 0.0496 |
| var/err2 Spearman | 0.4255 | 0.4173 | 0.4132 |
| coverage 90% | 0.9229 | 0.9227 | 0.9255 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0014 | [-0.0421, 0.0488] | NOISE |
| h1 | 0.0022 | [-0.0437, 0.0485] | NOISE |
| h2 | -0.0035 | [-0.0612, 0.0493] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0017
- unanimity: 0.7457
- delta_dir_align_all: 0.5112
- coherence_primary: 0.8903

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | beats |
| zero_delta | delta/mae | does not beat | does not beat | beats |
| zero_delta | delta/ev | beats | does not beat | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| mean_delta | delta/rmse | does not beat | does not beat | beats |
| mean_delta | delta/mae | does not beat | does not beat | beats |
| mean_delta | delta/ev | beats | does not beat | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | does not beat |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 309
- total_return: -0.5455
- sharpe_net: -124.6761
- sharpe_gross: 5.3780
- sortino: -144.2837
- max_drawdown: 0.5465
- hit_rate: 0.1197
- profit_factor: 0.0569
- avg_hold_bars: 11.3430
- exposure: 0.4844
- turnover: 431.6483
- fees_paid: 4316.3462
- costs_paid: 5611.2501
- gross_pnl: 156.5542
- net_pnl: -5454.6959
