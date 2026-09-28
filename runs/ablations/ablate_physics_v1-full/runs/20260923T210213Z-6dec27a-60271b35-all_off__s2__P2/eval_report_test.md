# Evaluation report - test split - run `20260923T210213Z-6dec27a-60271b35-all_off__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0092 | 0.0026 | 0.0242 |
| direction AUC | 0.5088 | 0.4887 | 0.4947 |
| direction ECE | 0.0319 | 0.0226 | 0.0488 |
| Gaussian MCC | -0.0503 | -0.0360 | -0.0306 |
| Gaussian AUC | 0.4657 | 0.4792 | 0.4668 |
| delta EV | -0.0021 | -0.0067 | -0.0080 |
| delta corr | -0.0498 | -0.0224 | -0.0398 |
| skill vs zero | -0.0022 | -0.0074 | -0.0080 |
| CRPS ($) | 105.4737 | 127.4243 | 145.6356 |
| PIT KS | 0.0310 | 0.0258 | 0.0315 |
| var/err2 Spearman | 0.2551 | 0.2411 | 0.2245 |
| coverage 90% | 0.9028 | 0.9099 | 0.9135 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0187 | [-0.0654, 0.0199] | NOISE |
| h1 | -0.0511 | [-0.0805, -0.0232] | INVERTED |
| h2 | -0.0305 | [-0.0796, 0.0125] | NOISE |

## Coherence across horizons

- mag_order_full: 0.6090
- unanimity: 0.3451
- delta_dir_align_all: 0.1342
- coherence_primary: 0.4735

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
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
- total_return: -0.4721
- sharpe_net: -124.0924
- sharpe_gross: 2.0778
- sortino: -141.4152
- max_drawdown: 0.4729
- hit_rate: 0.1184
- profit_factor: 0.0593
- avg_hold_bars: 8.2204
- exposure: 0.2783
- turnover: 367.6861
- fees_paid: 3677.1897
- costs_paid: 4780.3466
- gross_pnl: 59.2512
- net_pnl: -4721.0954
