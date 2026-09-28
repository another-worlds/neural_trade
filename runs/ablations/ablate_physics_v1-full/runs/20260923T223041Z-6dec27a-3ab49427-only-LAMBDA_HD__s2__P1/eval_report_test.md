# Evaluation report - test split - run `20260923T223041Z-6dec27a-3ab49427-only-LAMBDA_HD__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0128 | -0.0251 | 0.0614 |
| direction AUC | 0.4875 | 0.4950 | 0.5364 |
| direction ECE | 0.0656 | 0.0900 | 0.0492 |
| Gaussian MCC | 0.0185 | 0.0684 | 0.0000 |
| Gaussian AUC | 0.5016 | 0.5436 | 0.5000 |
| delta EV | 0.0003 | 0.0015 | 0.0000 |
| delta corr | 0.0465 | 0.0898 | 0.0000 |
| skill vs zero | 0.0001 | 0.0012 | 0.0000 |
| CRPS ($) | 109.7384 | 134.5943 | 158.8278 |
| PIT KS | 0.0417 | 0.0407 | 0.0523 |
| var/err2 Spearman | 0.4356 | 0.4169 | 0.4052 |
| coverage 90% | 0.9232 | 0.9283 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0091 | [-0.0689, 0.0510] | NOISE |
| h1 | -0.0074 | [-0.0457, 0.0331] | NOISE |
| h2 | -0.0117 | [-0.0565, 0.0447] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.5050
- delta_dir_align_all: 0.1422
- coherence_primary: 0.6997

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | does not beat |
| zero_delta | delta/mae | does not beat | beats | does not beat |
| zero_delta | delta/ev | beats | beats | does not beat |
| zero_delta | delta/corr | beats | beats | does not beat |
| zero_delta | delta/skill_vs_zero | beats | beats | does not beat |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | beats | beats | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | beats |
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | beats |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 250
- total_return: -0.4825
- sharpe_net: -108.4548
- sharpe_gross: -0.3096
- sortino: -129.2090
- max_drawdown: 0.4825
- hit_rate: 0.1160
- profit_factor: 0.0742
- avg_hold_bars: 11.2400
- exposure: 0.3883
- turnover: 370.0627
- fees_paid: 3700.3877
- costs_paid: 4810.5040
- gross_pnl: -14.0480
- net_pnl: -4824.5521
