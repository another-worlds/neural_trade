# Evaluation report - test split - run `20260923T210911Z-6dec27a-8c390f51-only-LAMBDA_T_PERP__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0661 | -0.0312 | -0.0526 |
| direction AUC | 0.4676 | 0.4872 | 0.4783 |
| direction ECE | 0.0946 | 0.0514 | 0.0830 |
| Gaussian MCC | 0.0302 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5220 | 0.5000 | 0.5000 |
| delta EV | 0.0011 | 0.0000 | 0.0000 |
| delta corr | 0.0737 | 0.0000 | 0.0000 |
| skill vs zero | 0.0005 | 0.0000 | 0.0000 |
| CRPS ($) | 109.8378 | 134.4727 | 158.4149 |
| PIT KS | 0.0519 | 0.0439 | 0.0554 |
| var/err2 Spearman | 0.4062 | 0.3877 | 0.4062 |
| coverage 90% | 0.9236 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0199 | [-0.0751, 0.0277] | NOISE |
| h1 | -0.0069 | [-0.0505, 0.0394] | NOISE |
| h2 | -0.0134 | [-0.0599, 0.0332] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.5126
- delta_dir_align_all: 0.0480
- coherence_primary: 0.2537

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | does not beat | does not beat |
| zero_delta | delta/mae | beats | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
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
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 289
- total_return: -0.5380
- sharpe_net: -127.4915
- sharpe_gross: -5.4783
- sortino: -147.4262
- max_drawdown: 0.5380
- hit_rate: 0.1073
- profit_factor: 0.0448
- avg_hold_bars: 8.2007
- exposure: 0.3275
- turnover: 400.2607
- fees_paid: 4002.7254
- costs_paid: 5203.5430
- gross_pnl: -176.2779
- net_pnl: -5379.8208
