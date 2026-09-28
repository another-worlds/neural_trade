# Evaluation report - test split - run `20260923T204821Z-6dec27a-db86c2eb-all_off__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0414 | 0.0252 | 0.0689 |
| direction AUC | 0.5142 | 0.5216 | 0.5434 |
| direction ECE | 0.0191 | 0.0222 | 0.0333 |
| Gaussian MCC | 0.0127 | 0.0534 | 0.0417 |
| Gaussian AUC | 0.5069 | 0.5211 | 0.5149 |
| delta EV | -0.0077 | -0.0029 | -0.0022 |
| delta corr | 0.0232 | 0.0357 | 0.0455 |
| skill vs zero | -0.0103 | -0.0059 | -0.0078 |
| CRPS ($) | 106.0286 | 127.8985 | 145.8289 |
| PIT KS | 0.0391 | 0.0417 | 0.0519 |
| var/err2 Spearman | 0.2540 | 0.2291 | 0.2257 |
| coverage 90% | 0.9055 | 0.9139 | 0.9142 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0071 | [-0.0456, 0.0285] | NOISE |
| h1 | 0.0306 | [-0.0072, 0.0649] | NOISE |
| h2 | 0.0232 | [-0.0154, 0.0626] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4270
- unanimity: 0.3872
- delta_dir_align_all: 0.2506
- coherence_primary: 0.6050

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
| class_prior | direction/ece_pos | beats | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | beats | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | does not beat | beats |

## Backtest (costs included)

- n_trades: 256
- total_return: -0.4668
- sharpe_net: -124.7725
- sharpe_gross: 10.0876
- sortino: -142.2753
- max_drawdown: 0.4678
- hit_rate: 0.0742
- profit_factor: 0.0372
- avg_hold_bars: 8.7617
- exposure: 0.3100
- turnover: 381.3943
- fees_paid: 3814.1581
- costs_paid: 4958.4055
- gross_pnl: 290.8688
- net_pnl: -4667.5367
