# Evaluation report - test split - run `20260923T202600Z-6dec27a-61cc47c8-all_on__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0343 | -0.0122 | -0.0290 |
| direction AUC | 0.4689 | 0.5007 | 0.4699 |
| direction ECE | 0.0358 | 0.0278 | 0.0546 |
| Gaussian MCC | 0.0098 | -0.0019 | -0.0032 |
| Gaussian AUC | 0.4985 | 0.4903 | 0.5000 |
| delta EV | -0.0236 | -0.0259 | -0.0351 |
| delta corr | -0.0337 | -0.0277 | -0.0244 |
| skill vs zero | -0.0266 | -0.0330 | -0.0444 |
| CRPS ($) | 106.5972 | 128.7963 | 148.0464 |
| PIT KS | 0.0287 | 0.0362 | 0.0487 |
| var/err2 Spearman | 0.2730 | 0.2649 | 0.2490 |
| coverage 90% | 0.9037 | 0.9053 | 0.9107 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0570 | [-0.1047, -0.0059] | INVERTED |
| h1 | 0.0083 | [-0.0356, 0.0503] | NOISE |
| h2 | -0.0731 | [-0.1347, -0.0065] | INVERTED |

## Coherence across horizons

- mag_order_full: 0.6417
- unanimity: 0.3190
- delta_dir_align_all: 0.1633
- coherence_primary: 0.4764

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
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | beats | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | beats | does not beat |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | does not beat |
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

- n_trades: 310
- total_return: -0.5465
- sharpe_net: -145.6243
- sharpe_gross: 4.1243
- sortino: -164.0796
- max_drawdown: 0.5477
- hit_rate: 0.0677
- profit_factor: 0.0443
- avg_hold_bars: 11.7452
- exposure: 0.5033
- turnover: 429.5928
- fees_paid: 4296.0214
- costs_paid: 5584.8279
- gross_pnl: 119.8591
- net_pnl: -5464.9688
