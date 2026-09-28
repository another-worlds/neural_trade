# Evaluation report - test split - run `20260924T010620Z-6dec27a-67ad3323-only-LAMBDA_VAC_OVERFLOW__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0381 | -0.0282 | 0.0510 |
| direction AUC | 0.5175 | 0.4818 | 0.5322 |
| direction ECE | 0.0351 | 0.0256 | 0.0261 |
| Gaussian MCC | 0.0516 | 0.0316 | 0.0597 |
| Gaussian AUC | 0.5204 | 0.5140 | 0.5311 |
| delta EV | 0.0030 | 0.0016 | 0.0021 |
| delta corr | 0.0554 | 0.0405 | 0.0459 |
| skill vs zero | 0.0029 | 0.0011 | 0.0004 |
| CRPS ($) | 105.3250 | 127.4697 | 145.1196 |
| PIT KS | 0.0413 | 0.0456 | 0.0494 |
| var/err2 Spearman | 0.2652 | 0.2458 | 0.2333 |
| coverage 90% | 0.9060 | 0.9103 | 0.9136 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0073 | [-0.0472, 0.0365] | NOISE |
| h1 | -0.0141 | [-0.0497, 0.0225] | NOISE |
| h2 | 0.0183 | [-0.0215, 0.0539] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3288
- unanimity: 0.3650
- delta_dir_align_all: 0.2711
- coherence_primary: 0.5896

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
| class_prior | direction/mcc | beats | does not beat | beats |
| class_prior | direction/auc | beats | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | beats |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | does not beat | does not beat | beats |

## Backtest (costs included)

- n_trades: 175
- total_return: -0.3623
- sharpe_net: -99.5703
- sharpe_gross: 1.8166
- sortino: -114.2214
- max_drawdown: 0.3623
- hit_rate: 0.1143
- profit_factor: 0.0243
- avg_hold_bars: 8.1314
- exposure: 0.1967
- turnover: 282.3430
- fees_paid: 2823.6171
- costs_paid: 3670.7023
- gross_pnl: 47.7275
- net_pnl: -3622.9748
