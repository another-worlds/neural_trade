# Evaluation report - test split - run `20260923T212531Z-6dec27a-ae8a00d9-only-LAMBDA_T_PERP__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0261 | 0.0551 | 0.0408 |
| direction AUC | 0.5289 | 0.5304 | 0.5248 |
| direction ECE | 0.0414 | 0.0229 | 0.0442 |
| Gaussian MCC | 0.0149 | 0.0255 | 0.0158 |
| Gaussian AUC | 0.5253 | 0.5182 | 0.5102 |
| delta EV | 0.0036 | 0.0022 | -0.0059 |
| delta corr | 0.0635 | 0.0580 | 0.0274 |
| skill vs zero | 0.0044 | 0.0034 | -0.0058 |
| CRPS ($) | 106.2357 | 128.0602 | 146.4401 |
| PIT KS | 0.0492 | 0.0514 | 0.0539 |
| var/err2 Spearman | 0.2448 | 0.2360 | 0.2247 |
| coverage 90% | 0.9071 | 0.9110 | 0.9142 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0385 | [-0.0047, 0.0813] | NOISE |
| h1 | 0.0101 | [-0.0430, 0.0645] | NOISE |
| h2 | 0.0046 | [-0.0392, 0.0456] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4225
- unanimity: 0.5902
- delta_dir_align_all: 0.2217
- coherence_primary: 0.4576

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | does not beat |
| zero_delta | delta/mae | beats | beats | does not beat |
| zero_delta | delta/ev | beats | beats | does not beat |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | beats | does not beat |
| mean_delta | delta/rmse | beats | beats | does not beat |
| mean_delta | delta/mae | beats | beats | does not beat |
| mean_delta | delta/ev | beats | beats | does not beat |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | beats | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | does not beat |
| logreg_lags | direction/auc | beats | beats | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | does not beat |

## Backtest (costs included)

- n_trades: 246
- total_return: -0.4580
- sharpe_net: -116.3352
- sharpe_gross: 8.4628
- sortino: -134.1481
- max_drawdown: 0.4583
- hit_rate: 0.0854
- profit_factor: 0.0337
- avg_hold_bars: 9.7602
- exposure: 0.3318
- turnover: 372.4340
- fees_paid: 3724.4086
- costs_paid: 4841.7312
- gross_pnl: 261.9809
- net_pnl: -4579.7503
