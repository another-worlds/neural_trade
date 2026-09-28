# Evaluation report - test split - run `20260923T201204Z-6dec27a-f2f93af0-all_on__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0436 | -0.0072 | 0.0454 |
| direction AUC | 0.5284 | 0.4933 | 0.5310 |
| direction ECE | 0.0359 | 0.0193 | 0.0203 |
| Gaussian MCC | -0.0357 | 0.0234 | 0.0249 |
| Gaussian AUC | 0.4878 | 0.5094 | 0.5112 |
| delta EV | -0.0086 | -0.0046 | -0.0265 |
| delta corr | -0.0070 | 0.0377 | 0.0179 |
| skill vs zero | -0.0101 | -0.0052 | -0.0370 |
| CRPS ($) | 106.4774 | 127.8430 | 147.5951 |
| PIT KS | 0.0526 | 0.0412 | 0.0642 |
| var/err2 Spearman | 0.2491 | 0.2401 | 0.2331 |
| coverage 90% | 0.9073 | 0.9088 | 0.9088 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0069 | [-0.0326, 0.0479] | NOISE |
| h1 | -0.0124 | [-0.0618, 0.0367] | NOISE |
| h2 | 0.0307 | [-0.0062, 0.0688] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4696
- unanimity: 0.3329
- delta_dir_align_all: 0.2324
- coherence_primary: 0.6516

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | beats | does not beat | beats |
| class_prior | direction/auc | beats | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | beats |
| class_prior | direction/ece_pos | does not beat | beats | beats |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | beats |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | beats | beats |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | does not beat | beats |

## Backtest (costs included)

- n_trades: 223
- total_return: -0.4089
- sharpe_net: -106.0231
- sharpe_gross: 13.8601
- sortino: -123.4606
- max_drawdown: 0.4089
- hit_rate: 0.1121
- profit_factor: 0.0417
- avg_hold_bars: 9.2018
- exposure: 0.2836
- turnover: 347.4393
- fees_paid: 3474.5061
- costs_paid: 4516.8580
- gross_pnl: 428.0890
- net_pnl: -4088.7689
