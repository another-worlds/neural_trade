# Evaluation report - test split - run `20260924T122721Z-6dec27a-9bde89bf-without-LAMBDA_IFE__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0462 | 0.0248 | 0.0366 |
| direction AUC | 0.5242 | 0.5158 | 0.5176 |
| direction ECE | 0.0479 | 0.0237 | 0.0362 |
| Gaussian MCC | 0.0170 | 0.0204 | 0.0101 |
| Gaussian AUC | 0.5088 | 0.4990 | 0.5050 |
| delta EV | -0.0078 | -0.0167 | -0.0156 |
| delta corr | 0.0176 | -0.0081 | 0.0129 |
| skill vs zero | -0.0084 | -0.0201 | -0.0230 |
| CRPS ($) | 106.5041 | 128.9820 | 146.8814 |
| PIT KS | 0.0566 | 0.0608 | 0.0694 |
| var/err2 Spearman | 0.2609 | 0.2560 | 0.2519 |
| coverage 90% | 0.9028 | 0.9096 | 0.9093 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0027 | [-0.0413, 0.0360] | NOISE |
| h1 | 0.0213 | [-0.0204, 0.0620] | NOISE |
| h2 | -0.0062 | [-0.0438, 0.0344] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4956
- unanimity: 0.4967
- delta_dir_align_all: 0.3443
- coherence_primary: 0.6512

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | does not beat | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | beats | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | does not beat |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | does not beat |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 219
- total_return: -0.4212
- sharpe_net: -111.3562
- sharpe_gross: 5.4637
- sortino: -127.3515
- max_drawdown: 0.4215
- hit_rate: 0.0913
- profit_factor: 0.0328
- avg_hold_bars: 9.2648
- exposure: 0.2804
- turnover: 336.1893
- fees_paid: 3362.1662
- costs_paid: 4370.8161
- gross_pnl: 158.9421
- net_pnl: -4211.8740
