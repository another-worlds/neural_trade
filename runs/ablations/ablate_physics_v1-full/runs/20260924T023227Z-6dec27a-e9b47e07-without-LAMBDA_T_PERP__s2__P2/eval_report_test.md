# Evaluation report - test split - run `20260924T023227Z-6dec27a-e9b47e07-without-LAMBDA_T_PERP__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0135 | -0.0435 | -0.0124 |
| direction AUC | 0.5036 | 0.4648 | 0.4871 |
| direction ECE | 0.0184 | 0.0602 | 0.0407 |
| Gaussian MCC | -0.0153 | -0.0105 | -0.0212 |
| Gaussian AUC | 0.4859 | 0.4789 | 0.4789 |
| delta EV | -0.0117 | -0.0058 | -0.0167 |
| delta corr | -0.0209 | -0.0104 | -0.0262 |
| skill vs zero | -0.0138 | -0.0086 | -0.0203 |
| CRPS ($) | 106.2564 | 127.7184 | 146.5267 |
| PIT KS | 0.0507 | 0.0460 | 0.0504 |
| var/err2 Spearman | 0.2832 | 0.2632 | 0.2540 |
| coverage 90% | 0.9022 | 0.9077 | 0.9095 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0070 | [-0.0320, 0.0523] | NOISE |
| h1 | -0.0291 | [-0.0697, 0.0096] | NOISE |
| h2 | -0.0212 | [-0.0667, 0.0243] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3481
- unanimity: 0.2555
- delta_dir_align_all: 0.2355
- coherence_primary: 0.6511

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
| class_prior | direction/mcc | beats | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | beats | does not beat | does not beat |
| class_prior | direction/acc | beats | does not beat | beats |
| class_prior | direction/bal_acc | beats | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | beats | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 166
- total_return: -0.3576
- sharpe_net: -98.9854
- sharpe_gross: -2.1964
- sortino: -114.8304
- max_drawdown: 0.3584
- hit_rate: 0.0602
- profit_factor: 0.0281
- avg_hold_bars: 10.5723
- exposure: 0.2425
- turnover: 269.8131
- fees_paid: 2698.3699
- costs_paid: 3507.8808
- gross_pnl: -67.7507
- net_pnl: -3575.6315
