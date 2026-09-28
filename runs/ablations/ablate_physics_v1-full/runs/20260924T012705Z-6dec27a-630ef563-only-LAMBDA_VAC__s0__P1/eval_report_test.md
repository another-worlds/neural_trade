# Evaluation report - test split - run `20260924T012705Z-6dec27a-630ef563-only-LAMBDA_VAC__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0283 | 0.0066 | 0.0119 |
| direction AUC | 0.4849 | 0.5159 | 0.5144 |
| direction ECE | 0.0679 | 0.0614 | 0.0733 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 110.9527 | 136.7578 | 159.6201 |
| PIT KS | 0.0606 | 0.0790 | 0.0596 |
| var/err2 Spearman | 0.3633 | 0.3205 | 0.3675 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0249 | [-0.0680, 0.0170] | NOISE |
| h1 | 0.0000 | [-0.0384, 0.0452] | NOISE |
| h2 | 0.0075 | [-0.0417, 0.0565] | NOISE |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.4417
- delta_dir_align_all: 0.0390
- coherence_primary: 0.1910

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
| class_prior | direction/mcc | does not beat | beats | beats |
| class_prior | direction/auc | does not beat | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | beats | beats |
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

- n_trades: 327
- total_return: -0.6094
- sharpe_net: -152.7693
- sharpe_gross: -19.0434
- sortino: -169.4116
- max_drawdown: 0.6095
- hit_rate: 0.0795
- profit_factor: 0.0216
- avg_hold_bars: 7.8471
- exposure: 0.3546
- turnover: 425.4346
- fees_paid: 4254.1451
- costs_paid: 5530.3886
- gross_pnl: -563.3751
- net_pnl: -6093.7637
