# Evaluation report - test split - run `20260924T113200Z-6dec27a-e857c730-without-LAMBDA_HD__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0139 | -0.0151 | 0.0255 |
| direction AUC | 0.4815 | 0.4769 | 0.5166 |
| direction ECE | 0.0551 | 0.0804 | 0.0531 |
| Gaussian MCC | 0.0389 | 0.0276 | 0.0271 |
| Gaussian AUC | 0.5244 | 0.5214 | 0.5255 |
| delta EV | 0.0029 | 0.0065 | 0.0043 |
| delta corr | 0.0862 | 0.0806 | 0.0842 |
| skill vs zero | 0.0023 | 0.0067 | 0.0039 |
| CRPS ($) | 109.7227 | 134.2631 | 158.0690 |
| PIT KS | 0.0353 | 0.0369 | 0.0423 |
| var/err2 Spearman | 0.4029 | 0.3807 | 0.3596 |
| coverage 90% | 0.9226 | 0.9265 | 0.9255 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0312 | [-0.0839, 0.0163] | NOISE |
| h1 | -0.0382 | [-0.0822, 0.0109] | NOISE |
| h2 | -0.0012 | [-0.0505, 0.0470] | NOISE |

## Coherence across horizons

- mag_order_full: 0.1365
- unanimity: 0.5301
- delta_dir_align_all: 0.2004
- coherence_primary: 0.4110

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | beats |
| zero_delta | delta/mae | beats | beats | beats |
| zero_delta | delta/ev | beats | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | beats | beats | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 280
- total_return: -0.5110
- sharpe_net: -116.3191
- sharpe_gross: 3.2497
- sortino: -137.7925
- max_drawdown: 0.5110
- hit_rate: 0.1143
- profit_factor: 0.0604
- avg_hold_bars: 12.3214
- exposure: 0.4768
- turnover: 400.2873
- fees_paid: 4002.6142
- costs_paid: 5203.3984
- gross_pnl: 93.7219
- net_pnl: -5109.6765
