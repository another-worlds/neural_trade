# Evaluation report - test split - run `20260924T012011Z-6dec27a-64666da6-only-LAMBDA_VAC_OVERFLOW__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0260 | -0.0490 | -0.0151 |
| direction AUC | 0.4917 | 0.4765 | 0.4694 |
| direction ECE | 0.0282 | 0.0438 | 0.0582 |
| Gaussian MCC | -0.0187 | -0.0240 | -0.0216 |
| Gaussian AUC | 0.4776 | 0.4753 | 0.4795 |
| delta EV | -0.0028 | 0.0009 | -0.0000 |
| delta corr | 0.0201 | 0.0299 | 0.0041 |
| skill vs zero | -0.0028 | 0.0007 | -0.0000 |
| CRPS ($) | 105.6195 | 127.1315 | 145.1887 |
| PIT KS | 0.0260 | 0.0266 | 0.0347 |
| var/err2 Spearman | 0.2533 | 0.2392 | 0.2311 |
| coverage 90% | 0.9048 | 0.9064 | 0.9125 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0218 | [-0.0275, 0.0683] | NOISE |
| h1 | 0.0045 | [-0.0447, 0.0463] | NOISE |
| h2 | -0.0379 | [-0.0897, 0.0064] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0130
- unanimity: 0.2605
- delta_dir_align_all: 0.2188
- coherence_primary: 0.5217

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | beats | does not beat |
| zero_delta | delta/mae | does not beat | beats | beats |
| zero_delta | delta/ev | does not beat | beats | does not beat |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | beats | does not beat |
| mean_delta | delta/rmse | does not beat | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | does not beat | beats | does not beat |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | beats | beats |
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
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 157
- total_return: -0.3415
- sharpe_net: -96.2873
- sharpe_gross: -3.4936
- sortino: -112.0758
- max_drawdown: 0.3421
- hit_rate: 0.1019
- profit_factor: 0.0477
- avg_hold_bars: 10.4459
- exposure: 0.2266
- turnover: 254.7782
- fees_paid: 2547.9814
- costs_paid: 3312.3758
- gross_pnl: -103.0433
- net_pnl: -3415.4191
