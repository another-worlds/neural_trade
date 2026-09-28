# Evaluation report - test split - run `20260923T233930Z-6dec27a-7a7cf27c-only-LAMBDA_VAC_OVERFLOW__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0076 | -0.0143 | 0.0036 |
| direction AUC | 0.4969 | 0.4800 | 0.4934 |
| direction ECE | 0.0487 | 0.0778 | 0.0581 |
| Gaussian MCC | 0.0226 | 0.0212 | 0.0357 |
| Gaussian AUC | 0.5143 | 0.5143 | 0.5287 |
| delta EV | 0.0066 | -0.0003 | 0.0102 |
| delta corr | 0.0866 | 0.0805 | 0.1145 |
| skill vs zero | 0.0018 | 0.0010 | 0.0129 |
| CRPS ($) | 110.7084 | 136.2024 | 159.5083 |
| PIT KS | 0.0503 | 0.0355 | 0.0436 |
| var/err2 Spearman | 0.4203 | 0.4111 | 0.3875 |
| coverage 90% | 0.9222 | 0.9261 | 0.9223 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0196 | [-0.0280, 0.0582] | NOISE |
| h1 | -0.0387 | [-0.0812, 0.0017] | NOISE |
| h2 | -0.0300 | [-0.0822, 0.0124] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3655
- unanimity: 0.5988
- delta_dir_align_all: 0.2852
- coherence_primary: 0.5847

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | beats |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | beats | does not beat | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | beats |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 333
- total_return: -0.5681
- sharpe_net: -131.1187
- sharpe_gross: 6.1936
- sortino: -153.5171
- max_drawdown: 0.5681
- hit_rate: 0.1021
- profit_factor: 0.0537
- avg_hold_bars: 10.4535
- exposure: 0.4811
- turnover: 451.0906
- fees_paid: 4510.6952
- costs_paid: 5863.9038
- gross_pnl: 183.1035
- net_pnl: -5680.8003
