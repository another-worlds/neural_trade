# Evaluation report - test split - run `20260923T233345Z-6dec27a-f4413078-only-LAMBDA_VAC_OVERFLOW__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0761 | -0.0057 | -0.0149 |
| direction AUC | 0.4696 | 0.4905 | 0.4783 |
| direction ECE | 0.0978 | 0.0485 | 0.0715 |
| Gaussian MCC | 0.0134 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5193 | 0.5000 | 0.5000 |
| delta EV | 0.0073 | 0.0000 | 0.0000 |
| delta corr | 0.1014 | 0.0000 | 0.0000 |
| skill vs zero | 0.0050 | 0.0000 | 0.0000 |
| CRPS ($) | 109.4910 | 134.3510 | 157.6568 |
| PIT KS | 0.0531 | 0.0553 | 0.0564 |
| var/err2 Spearman | 0.4086 | 0.3854 | 0.3902 |
| coverage 90% | 0.9227 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0008 | [-0.0493, 0.0427] | NOISE |
| h1 | -0.0283 | [-0.0726, 0.0149] | NOISE |
| h2 | -0.0529 | [-0.1032, -0.0076] | INVERTED |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.4719
- delta_dir_align_all: 0.0688
- coherence_primary: 0.2870

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
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | does not beat |

## Backtest (costs included)

- n_trades: 286
- total_return: -0.5332
- sharpe_net: -126.8231
- sharpe_gross: -4.7656
- sortino: -146.4316
- max_drawdown: 0.5334
- hit_rate: 0.0944
- profit_factor: 0.0408
- avg_hold_bars: 8.6049
- exposure: 0.3401
- turnover: 398.2515
- fees_paid: 3982.2669
- costs_paid: 5176.9469
- gross_pnl: -155.1636
- net_pnl: -5332.1106
