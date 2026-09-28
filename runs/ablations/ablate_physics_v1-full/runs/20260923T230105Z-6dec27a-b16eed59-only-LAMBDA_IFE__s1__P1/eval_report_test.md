# Evaluation report - test split - run `20260923T230105Z-6dec27a-b16eed59-only-LAMBDA_IFE__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0141 | -0.0384 | -0.0470 |
| direction AUC | 0.5065 | 0.4762 | 0.4743 |
| direction ECE | 0.0570 | 0.0971 | 0.0879 |
| Gaussian MCC | 0.0192 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5222 | 0.5000 | 0.5000 |
| delta EV | 0.0006 | 0.0000 | 0.0000 |
| delta corr | 0.0272 | 0.0000 | 0.0000 |
| skill vs zero | -0.0009 | 0.0000 | 0.0000 |
| CRPS ($) | 109.5112 | 134.1981 | 157.5730 |
| PIT KS | 0.0479 | 0.0387 | 0.0428 |
| var/err2 Spearman | 0.3993 | 0.4192 | 0.3915 |
| coverage 90% | 0.9219 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0207 | [-0.0630, 0.0249] | NOISE |
| h1 | -0.0290 | [-0.0755, 0.0121] | NOISE |
| h2 | 0.0013 | [-0.0541, 0.0508] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.6436
- delta_dir_align_all: 0.0478
- coherence_primary: 0.1466

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | beats | beats |
| class_prior | direction/mcc | beats | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | beats | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | does not beat |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 304
- total_return: -0.5614
- sharpe_net: -140.5433
- sharpe_gross: -8.5333
- sortino: -160.3117
- max_drawdown: 0.5614
- hit_rate: 0.0691
- profit_factor: 0.0261
- avg_hold_bars: 9.2566
- exposure: 0.3889
- turnover: 413.3948
- fees_paid: 4133.7664
- costs_paid: 5373.8963
- gross_pnl: -240.0891
- net_pnl: -5613.9854
