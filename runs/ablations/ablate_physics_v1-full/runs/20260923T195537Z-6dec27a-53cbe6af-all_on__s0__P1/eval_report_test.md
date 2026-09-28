# Evaluation report - test split - run `20260923T195537Z-6dec27a-53cbe6af-all_on__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0535 | -0.0186 | -0.0361 |
| direction AUC | 0.4773 | 0.4919 | 0.4725 |
| direction ECE | 0.0930 | 0.0550 | 0.0866 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 109.5203 | 133.8369 | 157.2644 |
| PIT KS | 0.0416 | 0.0498 | 0.0483 |
| var/err2 Spearman | 0.4165 | 0.4147 | 0.4135 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0175 | [-0.0666, 0.0213] | NOISE |
| h1 | -0.0310 | [-0.0699, 0.0161] | NOISE |
| h2 | -0.0601 | [-0.1138, -0.0049] | INVERTED |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.5209
- delta_dir_align_all: 0.0325
- coherence_primary: 0.2580

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
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | does not beat |
| logreg_lags | direction/auc | does not beat | beats | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | does not beat |

## Backtest (costs included)

- n_trades: 299
- total_return: -0.5395
- sharpe_net: -125.6883
- sharpe_gross: -0.2910
- sortino: -146.8887
- max_drawdown: 0.5396
- hit_rate: 0.1070
- profit_factor: 0.0458
- avg_hold_bars: 8.5217
- exposure: 0.3521
- turnover: 414.0088
- fees_paid: 4139.6926
- costs_paid: 5381.6004
- gross_pnl: -13.1208
- net_pnl: -5394.7212
