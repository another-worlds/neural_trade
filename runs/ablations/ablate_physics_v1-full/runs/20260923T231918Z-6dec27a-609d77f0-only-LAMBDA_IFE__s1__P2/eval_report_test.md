# Evaluation report - test split - run `20260923T231918Z-6dec27a-609d77f0-only-LAMBDA_IFE__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0046 | -0.0025 | -0.0167 |
| direction AUC | 0.5024 | 0.4888 | 0.4830 |
| direction ECE | 0.0364 | 0.0356 | 0.0290 |
| Gaussian MCC | -0.0572 | -0.0316 | -0.0512 |
| Gaussian AUC | 0.4682 | 0.4682 | 0.4666 |
| delta EV | -0.0040 | -0.0045 | -0.0167 |
| delta corr | -0.0233 | -0.0225 | -0.0443 |
| skill vs zero | -0.0040 | -0.0044 | -0.0163 |
| CRPS ($) | 105.5243 | 126.9634 | 145.7688 |
| PIT KS | 0.0263 | 0.0228 | 0.0248 |
| var/err2 Spearman | 0.2698 | 0.2555 | 0.2516 |
| coverage 90% | 0.9024 | 0.9063 | 0.9104 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0005 | [-0.0479, 0.0463] | NOISE |
| h1 | -0.0319 | [-0.0711, 0.0044] | NOISE |
| h2 | -0.0301 | [-0.0752, 0.0148] | NOISE |

## Coherence across horizons

- mag_order_full: 0.5047
- unanimity: 0.4080
- delta_dir_align_all: 0.2289
- coherence_primary: 0.6191

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
| class_prior | direction/mcc | beats | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | does not beat | does not beat |
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

- n_trades: 256
- total_return: -0.4879
- sharpe_net: -130.9758
- sharpe_gross: -0.8712
- sortino: -146.8743
- max_drawdown: 0.4879
- hit_rate: 0.0664
- profit_factor: 0.0250
- avg_hold_bars: 8.1602
- exposure: 0.2887
- turnover: 373.0858
- fees_paid: 3731.3612
- costs_paid: 4850.7696
- gross_pnl: -27.8536
- net_pnl: -4878.6232
