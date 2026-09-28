# Evaluation report - test split - run `20260924T011315Z-6dec27a-6907527e-only-LAMBDA_VAC_OVERFLOW__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0145 | 0.0107 | 0.0032 |
| direction AUC | 0.5078 | 0.5091 | 0.5037 |
| direction ECE | 0.0242 | 0.0284 | 0.0228 |
| Gaussian MCC | -0.0538 | -0.0620 | -0.0493 |
| Gaussian AUC | 0.4727 | 0.4579 | 0.4559 |
| delta EV | -0.0009 | -0.0034 | -0.0153 |
| delta corr | 0.0308 | 0.0067 | -0.0387 |
| skill vs zero | -0.0010 | -0.0032 | -0.0146 |
| CRPS ($) | 105.7025 | 127.2522 | 145.9776 |
| PIT KS | 0.0337 | 0.0210 | 0.0254 |
| var/err2 Spearman | 0.2434 | 0.2412 | 0.2328 |
| coverage 90% | 0.9017 | 0.9071 | 0.9109 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0063 | [-0.0421, 0.0502] | NOISE |
| h1 | 0.0194 | [-0.0210, 0.0554] | NOISE |
| h2 | 0.0061 | [-0.0384, 0.0424] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3983
- unanimity: 0.3991
- delta_dir_align_all: 0.2150
- coherence_primary: 0.4855

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | beats |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | beats |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 199
- total_return: -0.4034
- sharpe_net: -110.1807
- sharpe_gross: -0.1320
- sortino: -125.2556
- max_drawdown: 0.4034
- hit_rate: 0.0704
- profit_factor: 0.0296
- avg_hold_bars: 8.2161
- exposure: 0.2260
- turnover: 309.7714
- fees_paid: 3097.7541
- costs_paid: 4027.0804
- gross_pnl: -6.7757
- net_pnl: -4033.8560
