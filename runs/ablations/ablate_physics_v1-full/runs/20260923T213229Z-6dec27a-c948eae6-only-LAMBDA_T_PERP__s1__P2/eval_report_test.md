# Evaluation report - test split - run `20260923T213229Z-6dec27a-c948eae6-only-LAMBDA_T_PERP__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0020 | 0.0152 | -0.0219 |
| direction AUC | 0.5056 | 0.4979 | 0.4911 |
| direction ECE | 0.0379 | 0.0347 | 0.0303 |
| Gaussian MCC | -0.0316 | -0.0485 | -0.0621 |
| Gaussian AUC | 0.4735 | 0.4692 | 0.4591 |
| delta EV | -0.0121 | -0.0205 | -0.0322 |
| delta corr | -0.0363 | -0.0619 | -0.0912 |
| skill vs zero | -0.0127 | -0.0213 | -0.0316 |
| CRPS ($) | 106.3815 | 128.4692 | 147.4979 |
| PIT KS | 0.0433 | 0.0391 | 0.0360 |
| var/err2 Spearman | 0.2352 | 0.2293 | 0.2195 |
| coverage 90% | 0.9045 | 0.9070 | 0.9088 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0026 | [-0.0504, 0.0518] | NOISE |
| h1 | -0.0237 | [-0.0569, 0.0168] | NOISE |
| h2 | -0.0002 | [-0.0426, 0.0365] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4566
- unanimity: 0.3075
- delta_dir_align_all: 0.1447
- coherence_primary: 0.4222

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | beats | beats | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 196
- total_return: -0.4020
- sharpe_net: -109.6895
- sharpe_gross: -1.0592
- sortino: -125.8802
- max_drawdown: 0.4020
- hit_rate: 0.0714
- profit_factor: 0.0324
- avg_hold_bars: 7.6020
- exposure: 0.2059
- turnover: 306.6781
- fees_paid: 3066.9915
- costs_paid: 3987.0890
- gross_pnl: -33.0753
- net_pnl: -4020.1642
