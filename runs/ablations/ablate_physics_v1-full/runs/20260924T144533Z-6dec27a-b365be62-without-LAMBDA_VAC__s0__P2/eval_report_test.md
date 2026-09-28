# Evaluation report - test split - run `20260924T144533Z-6dec27a-b365be62-without-LAMBDA_VAC__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0203 | 0.0170 | 0.0361 |
| direction AUC | 0.5025 | 0.5098 | 0.5189 |
| direction ECE | 0.0176 | 0.0201 | 0.0309 |
| Gaussian MCC | -0.0757 | -0.0500 | -0.0301 |
| Gaussian AUC | 0.4582 | 0.4716 | 0.4762 |
| delta EV | -0.0022 | 0.0002 | -0.0049 |
| delta corr | -0.0416 | 0.0196 | -0.0249 |
| skill vs zero | -0.0023 | 0.0001 | -0.0065 |
| CRPS ($) | 106.2551 | 127.6178 | 146.1145 |
| PIT KS | 0.0515 | 0.0412 | 0.0552 |
| var/err2 Spearman | 0.2456 | 0.2232 | 0.2052 |
| coverage 90% | 0.9038 | 0.9055 | 0.9103 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0142 | [-0.0542, 0.0232] | NOISE |
| h1 | 0.0184 | [-0.0316, 0.0594] | NOISE |
| h2 | 0.0208 | [-0.0216, 0.0662] | NOISE |

## Coherence across horizons

- mag_order_full: 0.5478
- unanimity: 0.4536
- delta_dir_align_all: 0.2546
- coherence_primary: 0.5659

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | beats | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | beats | does not beat |
| zero_delta | delta/corr | does not beat | beats | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | beats | does not beat |
| mean_delta | delta/rmse | does not beat | beats | does not beat |
| mean_delta | delta/mae | does not beat | beats | does not beat |
| mean_delta | delta/ev | does not beat | beats | does not beat |
| mean_delta | delta/corr | does not beat | beats | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | beats | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | beats | does not beat |
| class_prior | direction/ece_pos | beats | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | beats | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 280
- total_return: -0.5066
- sharpe_net: -131.4106
- sharpe_gross: 6.0669
- sortino: -150.4545
- max_drawdown: 0.5072
- hit_rate: 0.0821
- profit_factor: 0.0430
- avg_hold_bars: 9.1964
- exposure: 0.3559
- turnover: 403.7362
- fees_paid: 4037.5194
- costs_paid: 5248.7752
- gross_pnl: 182.4238
- net_pnl: -5066.3514
