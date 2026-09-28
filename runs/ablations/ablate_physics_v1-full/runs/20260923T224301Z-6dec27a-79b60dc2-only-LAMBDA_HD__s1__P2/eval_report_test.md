# Evaluation report - test split - run `20260923T224301Z-6dec27a-79b60dc2-only-LAMBDA_HD__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0220 | -0.0373 | -0.0539 |
| direction AUC | 0.4868 | 0.4849 | 0.4703 |
| direction ECE | 0.0347 | 0.0401 | 0.0384 |
| Gaussian MCC | -0.0279 | -0.0096 | -0.0304 |
| Gaussian AUC | 0.4789 | 0.4799 | 0.4789 |
| delta EV | -0.0107 | -0.0136 | -0.0137 |
| delta corr | 0.0014 | -0.0344 | -0.0251 |
| skill vs zero | -0.0108 | -0.0142 | -0.0146 |
| CRPS ($) | 105.7827 | 127.4454 | 145.7146 |
| PIT KS | 0.0389 | 0.0329 | 0.0397 |
| var/err2 Spearman | 0.2604 | 0.2551 | 0.2412 |
| coverage 90% | 0.9028 | 0.9059 | 0.9091 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0249 | [-0.0763, 0.0268] | NOISE |
| h1 | 0.0125 | [-0.0236, 0.0460] | NOISE |
| h2 | -0.0081 | [-0.0579, 0.0350] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4631
- unanimity: 0.3774
- delta_dir_align_all: 0.2512
- coherence_primary: 0.6230

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
| class_prior | direction/acc | beats | does not beat | does not beat |
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

- n_trades: 245
- total_return: -0.5019
- sharpe_net: -140.9406
- sharpe_gross: -15.0596
- sortino: -157.7172
- max_drawdown: 0.5029
- hit_rate: 0.0531
- profit_factor: 0.0151
- avg_hold_bars: 10.1959
- exposure: 0.3454
- turnover: 353.6235
- fees_paid: 3535.9733
- costs_paid: 4596.7653
- gross_pnl: -422.6327
- net_pnl: -5019.3980
