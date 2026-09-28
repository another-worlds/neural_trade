# Evaluation report - test split - run `20260924T124807Z-6dec27a-b67b786b-without-LAMBDA_VAC_OVERFLOW__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0409 | -0.0058 | -0.0205 |
| direction AUC | 0.4827 | 0.5053 | 0.4902 |
| direction ECE | 0.0825 | 0.0473 | 0.0774 |
| Gaussian MCC | -0.0110 | 0.0141 | 0.0393 |
| Gaussian AUC | 0.5154 | 0.5250 | 0.5309 |
| delta EV | 0.0095 | 0.0074 | 0.0041 |
| delta corr | 0.0973 | 0.1348 | 0.0958 |
| skill vs zero | 0.0031 | 0.0068 | 0.0038 |
| CRPS ($) | 109.6184 | 133.6581 | 157.0625 |
| PIT KS | 0.0523 | 0.0333 | 0.0342 |
| var/err2 Spearman | 0.4286 | 0.4147 | 0.4134 |
| coverage 90% | 0.9263 | 0.9284 | 0.9270 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0154 | [-0.0619, 0.0271] | NOISE |
| h1 | 0.0021 | [-0.0454, 0.0497] | NOISE |
| h2 | -0.0126 | [-0.0659, 0.0421] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0691
- unanimity: 0.4789
- delta_dir_align_all: 0.2206
- coherence_primary: 0.6194

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | beats |
| zero_delta | delta/mae | does not beat | beats | beats |
| zero_delta | delta/ev | beats | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | beats | beats | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | beats | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | does not beat |
| logreg_lags | direction/auc | does not beat | beats | does not beat |
| logreg_lags | direction/brier | does not beat | beats | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | does not beat |

## Backtest (costs included)

- n_trades: 298
- total_return: -0.5533
- sharpe_net: -133.4258
- sharpe_gross: -8.0764
- sortino: -152.7500
- max_drawdown: 0.5533
- hit_rate: 0.0973
- profit_factor: 0.0409
- avg_hold_bars: 7.6577
- exposure: 0.3154
- turnover: 406.1059
- fees_paid: 4060.9163
- costs_paid: 5279.1912
- gross_pnl: -253.9562
- net_pnl: -5533.1475
