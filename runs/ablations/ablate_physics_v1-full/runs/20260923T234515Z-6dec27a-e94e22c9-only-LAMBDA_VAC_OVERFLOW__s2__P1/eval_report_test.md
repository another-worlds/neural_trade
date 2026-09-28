# Evaluation report - test split - run `20260923T234515Z-6dec27a-e94e22c9-only-LAMBDA_VAC_OVERFLOW__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0394 | 0.0146 | -0.0008 |
| direction AUC | 0.4807 | 0.5080 | 0.5087 |
| direction ECE | 0.0735 | 0.0724 | 0.0552 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 109.4521 | 134.8012 | 157.8365 |
| PIT KS | 0.0461 | 0.0561 | 0.0521 |
| var/err2 Spearman | 0.4109 | 0.3561 | 0.3805 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0358 | [-0.0863, 0.0091] | NOISE |
| h1 | -0.0228 | [-0.0703, 0.0213] | NOISE |
| h2 | -0.0042 | [-0.0533, 0.0506] | NOISE |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.5158
- delta_dir_align_all: 0.0292
- coherence_primary: 0.1918

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
| class_prior | direction/mcc | does not beat | beats | does not beat |
| class_prior | direction/auc | does not beat | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | beats | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | beats |
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | beats |

## Backtest (costs included)

- n_trades: 334
- total_return: -0.5760
- sharpe_net: -138.8739
- sharpe_gross: 4.0082
- sortino: -159.5210
- max_drawdown: 0.5760
- hit_rate: 0.0808
- profit_factor: 0.0348
- avg_hold_bars: 9.1287
- exposure: 0.4215
- turnover: 451.8359
- fees_paid: 4518.2157
- costs_paid: 5873.6804
- gross_pnl: 113.7294
- net_pnl: -5759.9510
