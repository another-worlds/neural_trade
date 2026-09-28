# Evaluation report - test split - run `20260924T114518Z-6dec27a-e1ace559-without-LAMBDA_HD__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0518 | -0.0070 | -0.0758 |
| direction AUC | 0.4761 | 0.4813 | 0.4483 |
| direction ECE | 0.0539 | 0.0282 | 0.0458 |
| Gaussian MCC | 0.0044 | -0.0340 | -0.0495 |
| Gaussian AUC | 0.4753 | 0.4599 | 0.4570 |
| delta EV | -0.0064 | -0.0137 | -0.0287 |
| delta corr | 0.0054 | -0.0497 | -0.0828 |
| skill vs zero | -0.0088 | -0.0147 | -0.0307 |
| CRPS ($) | 106.3642 | 128.2080 | 147.6354 |
| PIT KS | 0.0612 | 0.0467 | 0.0573 |
| var/err2 Spearman | 0.2352 | 0.2333 | 0.2331 |
| coverage 90% | 0.9004 | 0.9048 | 0.9121 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0102 | [-0.0613, 0.0427] | NOISE |
| h1 | -0.0505 | [-0.0861, -0.0144] | INVERTED |
| h2 | -0.0356 | [-0.0904, 0.0084] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4714
- unanimity: 0.4328
- delta_dir_align_all: 0.2783
- coherence_primary: 0.6175

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | beats | does not beat |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
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

- n_trades: 300
- total_return: -0.5582
- sharpe_net: -151.2273
- sharpe_gross: -10.3295
- sortino: -168.6964
- max_drawdown: 0.5590
- hit_rate: 0.0600
- profit_factor: 0.0162
- avg_hold_bars: 11.4433
- exposure: 0.4746
- turnover: 406.0522
- fees_paid: 4060.4082
- costs_paid: 5278.5307
- gross_pnl: -303.1539
- net_pnl: -5581.6846
