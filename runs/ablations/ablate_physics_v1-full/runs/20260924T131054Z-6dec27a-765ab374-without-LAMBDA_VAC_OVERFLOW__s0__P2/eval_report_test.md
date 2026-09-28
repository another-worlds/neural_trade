# Evaluation report - test split - run `20260924T131054Z-6dec27a-765ab374-without-LAMBDA_VAC_OVERFLOW__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0224 | 0.0134 | 0.0406 |
| direction AUC | 0.5204 | 0.5070 | 0.5336 |
| direction ECE | 0.0374 | 0.0164 | 0.0320 |
| Gaussian MCC | -0.0082 | 0.0341 | 0.0420 |
| Gaussian AUC | 0.5095 | 0.5219 | 0.5234 |
| delta EV | -0.0027 | -0.0044 | -0.0034 |
| delta corr | 0.0493 | 0.0388 | 0.0483 |
| skill vs zero | -0.0020 | -0.0065 | -0.0057 |
| CRPS ($) | 105.7303 | 127.6628 | 145.5809 |
| PIT KS | 0.0352 | 0.0482 | 0.0509 |
| var/err2 Spearman | 0.2641 | 0.2486 | 0.2516 |
| coverage 90% | 0.9073 | 0.9122 | 0.9171 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0196 | [-0.0203, 0.0570] | NOISE |
| h1 | -0.0069 | [-0.0512, 0.0322] | NOISE |
| h2 | 0.0318 | [-0.0069, 0.0771] | NOISE |

## Coherence across horizons

- mag_order_full: 0.2877
- unanimity: 0.3308
- delta_dir_align_all: 0.2098
- coherence_primary: 0.6014

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | beats | beats |
| class_prior | direction/ece_pos | does not beat | beats | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | beats | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 235
- total_return: -0.4309
- sharpe_net: -112.0804
- sharpe_gross: 12.4405
- sortino: -129.8499
- max_drawdown: 0.4317
- hit_rate: 0.1021
- profit_factor: 0.0468
- avg_hold_bars: 8.7574
- exposure: 0.2844
- turnover: 360.2488
- fees_paid: 3602.7257
- costs_paid: 4683.5434
- gross_pnl: 374.0871
- net_pnl: -4309.4563
