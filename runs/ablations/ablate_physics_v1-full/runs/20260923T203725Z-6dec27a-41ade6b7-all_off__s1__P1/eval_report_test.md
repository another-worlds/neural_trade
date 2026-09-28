# Evaluation report - test split - run `20260923T203725Z-6dec27a-41ade6b7-all_off__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0086 | -0.0135 | -0.0083 |
| direction AUC | 0.5054 | 0.4884 | 0.4811 |
| direction ECE | 0.0497 | 0.0805 | 0.0656 |
| Gaussian MCC | 0.0000 | 0.0359 | 0.0423 |
| Gaussian AUC | 0.5000 | 0.5195 | 0.5330 |
| delta EV | 0.0000 | 0.0009 | 0.0077 |
| delta corr | 0.0000 | 0.0501 | 0.0881 |
| skill vs zero | 0.0000 | -0.0002 | 0.0056 |
| CRPS ($) | 109.9299 | 135.0499 | 158.2231 |
| PIT KS | 0.0402 | 0.0417 | 0.0441 |
| var/err2 Spearman | 0.4026 | 0.3876 | 0.3571 |
| coverage 90% | 0.9232 | 0.9286 | 0.9265 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0146 | [-0.0263, 0.0586] | NOISE |
| h1 | -0.0205 | [-0.0696, 0.0267] | NOISE |
| h2 | -0.0401 | [-0.0909, 0.0085] | NOISE |

## Coherence across horizons

- mag_order_full: 0.7956
- unanimity: 0.5786
- delta_dir_align_all: 0.1639
- coherence_primary: 0.5451

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | beats |
| zero_delta | delta/mae | does not beat | does not beat | beats |
| zero_delta | delta/ev | does not beat | beats | beats |
| zero_delta | delta/corr | does not beat | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | does not beat | beats | beats |
| mean_delta | delta/corr | does not beat | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | beats | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | beats | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | does not beat |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 321
- total_return: -0.5449
- sharpe_net: -126.6319
- sharpe_gross: 9.5305
- sortino: -147.6180
- max_drawdown: 0.5449
- hit_rate: 0.1121
- profit_factor: 0.0625
- avg_hold_bars: 9.3333
- exposure: 0.4140
- turnover: 441.8744
- fees_paid: 4418.7939
- costs_paid: 5744.4320
- gross_pnl: 295.2125
- net_pnl: -5449.2196
