# Evaluation report - test split - run `20260924T141821Z-6dec27a-e1c93970-without-LAMBDA_VAC__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0190 | -0.0304 | -0.0046 |
| direction AUC | 0.4929 | 0.4733 | 0.4856 |
| direction ECE | 0.0655 | 0.0987 | 0.0583 |
| Gaussian MCC | 0.0023 | -0.0279 | 0.0332 |
| Gaussian AUC | 0.5230 | 0.5169 | 0.5350 |
| delta EV | -0.0036 | -0.0169 | 0.0029 |
| delta corr | 0.0496 | 0.0484 | 0.0644 |
| skill vs zero | -0.0185 | -0.0261 | -0.0010 |
| CRPS ($) | 111.1115 | 136.8366 | 158.6381 |
| PIT KS | 0.0711 | 0.0662 | 0.0497 |
| var/err2 Spearman | 0.4299 | 0.4248 | 0.4043 |
| coverage 90% | 0.9219 | 0.9225 | 0.9244 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0380 | [-0.0797, 0.0058] | NOISE |
| h1 | -0.0391 | [-0.0936, 0.0119] | NOISE |
| h2 | -0.0337 | [-0.0746, 0.0087] | NOISE |

## Coherence across horizons

- mag_order_full: 0.1452
- unanimity: 0.7290
- delta_dir_align_all: 0.5296
- coherence_primary: 0.8409

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | beats |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 313
- total_return: -0.5189
- sharpe_net: -116.1338
- sharpe_gross: 17.0203
- sortino: -138.2317
- max_drawdown: 0.5194
- hit_rate: 0.1086
- profit_factor: 0.0634
- avg_hold_bars: 11.4058
- exposure: 0.4934
- turnover: 439.2483
- fees_paid: 4392.2279
- costs_paid: 5709.8963
- gross_pnl: 520.8114
- net_pnl: -5189.0849
