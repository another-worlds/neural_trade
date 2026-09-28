# Evaluation report - test split - run `20260924T020237Z-6dec27a-04b1da62-without-LAMBDA_T_PERP__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0167 | -0.0071 | -0.0446 |
| direction AUC | 0.4788 | 0.4893 | 0.4667 |
| direction ECE | 0.0743 | 0.0561 | 0.0857 |
| Gaussian MCC | 0.0339 | 0.0578 | 0.0761 |
| Gaussian AUC | 0.5205 | 0.5346 | 0.5498 |
| delta EV | 0.0064 | 0.0080 | 0.0097 |
| delta corr | 0.0800 | 0.1047 | 0.1117 |
| skill vs zero | 0.0034 | 0.0072 | 0.0087 |
| CRPS ($) | 109.6430 | 134.2752 | 157.0405 |
| PIT KS | 0.0460 | 0.0415 | 0.0428 |
| var/err2 Spearman | 0.4170 | 0.4056 | 0.4095 |
| coverage 90% | 0.9240 | 0.9287 | 0.9280 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0576 | [-0.1062, -0.0117] | INVERTED |
| h1 | -0.0373 | [-0.0834, 0.0158] | NOISE |
| h2 | -0.0545 | [-0.0977, -0.0089] | INVERTED |

## Coherence across horizons

- mag_order_full: 0.2355
- unanimity: 0.5207
- delta_dir_align_all: 0.2323
- coherence_primary: 0.4389

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
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | does not beat |

## Backtest (costs included)

- n_trades: 253
- total_return: -0.4755
- sharpe_net: -108.3875
- sharpe_gross: 1.9989
- sortino: -129.5957
- max_drawdown: 0.4755
- hit_rate: 0.1304
- profit_factor: 0.0515
- avg_hold_bars: 8.4625
- exposure: 0.2959
- turnover: 370.6661
- fees_paid: 3706.4949
- costs_paid: 4818.4433
- gross_pnl: 63.1939
- net_pnl: -4755.2494
