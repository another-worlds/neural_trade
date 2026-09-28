# Evaluation report - test split - run `20260924T132028Z-6dec27a-0091176b-without-LAMBDA_VAC_OVERFLOW__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0272 | -0.0053 | -0.0262 |
| direction AUC | 0.4939 | 0.4984 | 0.4913 |
| direction ECE | 0.0381 | 0.0226 | 0.0261 |
| Gaussian MCC | -0.0350 | -0.0723 | -0.0492 |
| Gaussian AUC | 0.4802 | 0.4485 | 0.4566 |
| delta EV | -0.0117 | -0.0221 | -0.0517 |
| delta corr | 0.0065 | -0.0396 | -0.0743 |
| skill vs zero | -0.0128 | -0.0222 | -0.0512 |
| CRPS ($) | 106.1218 | 128.2276 | 148.4394 |
| PIT KS | 0.0459 | 0.0281 | 0.0295 |
| var/err2 Spearman | 0.2540 | 0.2499 | 0.2468 |
| coverage 90% | 0.9011 | 0.9035 | 0.9084 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0389 | [-0.0036, 0.0780] | NOISE |
| h1 | 0.0009 | [-0.0436, 0.0428] | NOISE |
| h2 | 0.0114 | [-0.0251, 0.0530] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4069
- unanimity: 0.4098
- delta_dir_align_all: 0.2796
- coherence_primary: 0.6759

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
| class_prior | direction/acc | does not beat | beats | beats |
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

- n_trades: 308
- total_return: -0.5724
- sharpe_net: -153.3569
- sharpe_gross: -11.9459
- sortino: -169.2095
- max_drawdown: 0.5733
- hit_rate: 0.0649
- profit_factor: 0.0219
- avg_hold_bars: 10.7305
- exposure: 0.4569
- turnover: 413.6814
- fees_paid: 4136.9316
- costs_paid: 5378.0111
- gross_pnl: -346.0975
- net_pnl: -5724.1085
