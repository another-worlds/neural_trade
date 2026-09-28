# Evaluation report - test split - run `20260924T143448Z-6dec27a-cdd868e4-without-LAMBDA_VAC__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0173 | -0.0253 | 0.0171 |
| direction AUC | 0.4805 | 0.4793 | 0.5136 |
| direction ECE | 0.0624 | 0.0775 | 0.0496 |
| Gaussian MCC | 0.0290 | 0.0587 | 0.0224 |
| Gaussian AUC | 0.5143 | 0.5383 | 0.5222 |
| delta EV | 0.0064 | 0.0077 | 0.0103 |
| delta corr | 0.0818 | 0.0891 | 0.1108 |
| skill vs zero | 0.0053 | 0.0076 | 0.0101 |
| CRPS ($) | 109.1812 | 133.4351 | 156.5828 |
| PIT KS | 0.0512 | 0.0482 | 0.0442 |
| var/err2 Spearman | 0.4279 | 0.4136 | 0.4186 |
| coverage 90% | 0.9245 | 0.9294 | 0.9284 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0331 | [-0.0707, 0.0124] | NOISE |
| h1 | -0.0105 | [-0.0507, 0.0349] | NOISE |
| h2 | 0.0163 | [-0.0335, 0.0653] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3011
- unanimity: 0.4649
- delta_dir_align_all: 0.2218
- coherence_primary: 0.5297

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | beats |
| zero_delta | delta/mae | does not beat | beats | beats |
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

- n_trades: 303
- total_return: -0.5394
- sharpe_net: -127.1732
- sharpe_gross: 2.3672
- sortino: -150.1828
- max_drawdown: 0.5394
- hit_rate: 0.1089
- profit_factor: 0.0570
- avg_hold_bars: 11.2343
- exposure: 0.4704
- turnover: 420.1528
- fees_paid: 4201.2340
- costs_paid: 5461.6041
- gross_pnl: 67.8081
- net_pnl: -5393.7961
