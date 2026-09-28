# Evaluation report - test split - run `20260924T112550Z-6dec27a-f3f1e2f1-without-LAMBDA_HD__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0072 | -0.0151 | 0.0013 |
| direction AUC | 0.5012 | 0.4754 | 0.4696 |
| direction ECE | 0.0570 | 0.0838 | 0.0666 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 110.0940 | 135.1375 | 158.7727 |
| PIT KS | 0.0383 | 0.0422 | 0.0444 |
| var/err2 Spearman | 0.3977 | 0.4114 | 0.3917 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0073 | [-0.0409, 0.0514] | NOISE |
| h1 | -0.0466 | [-0.0980, -0.0002] | INVERTED |
| h2 | -0.0574 | [-0.1206, -0.0057] | INVERTED |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.6169
- delta_dir_align_all: 0.0296
- coherence_primary: 0.1017

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
| class_prior | direction/mcc | beats | does not beat | beats |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | beats |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 338
- total_return: -0.5734
- sharpe_net: -138.0316
- sharpe_gross: 5.6155
- sortino: -158.7976
- max_drawdown: 0.5734
- hit_rate: 0.1006
- profit_factor: 0.0493
- avg_hold_bars: 9.4467
- exposure: 0.4413
- turnover: 453.3179
- fees_paid: 4533.2678
- costs_paid: 5893.2481
- gross_pnl: 158.9690
- net_pnl: -5734.2791
