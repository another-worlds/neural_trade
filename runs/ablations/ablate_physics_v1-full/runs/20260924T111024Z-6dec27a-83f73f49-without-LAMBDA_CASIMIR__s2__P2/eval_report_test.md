# Evaluation report - test split - run `20260924T111024Z-6dec27a-83f73f49-without-LAMBDA_CASIMIR__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0090 | -0.0204 | 0.0007 |
| direction AUC | 0.5086 | 0.4810 | 0.4982 |
| direction ECE | 0.0372 | 0.0455 | 0.0668 |
| Gaussian MCC | -0.0127 | -0.0840 | -0.0585 |
| Gaussian AUC | 0.4810 | 0.4563 | 0.4638 |
| delta EV | -0.0048 | -0.0068 | -0.0124 |
| delta corr | -0.0481 | -0.0770 | -0.0761 |
| skill vs zero | -0.0075 | -0.0075 | -0.0142 |
| CRPS ($) | 105.8657 | 127.4644 | 145.8710 |
| PIT KS | 0.0319 | 0.0199 | 0.0307 |
| var/err2 Spearman | 0.2442 | 0.2370 | 0.2261 |
| coverage 90% | 0.9013 | 0.9091 | 0.9121 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0204 | [-0.0768, 0.0227] | NOISE |
| h1 | -0.0080 | [-0.0356, 0.0309] | NOISE |
| h2 | -0.0009 | [-0.0526, 0.0467] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4435
- unanimity: 0.5540
- delta_dir_align_all: 0.3054
- coherence_primary: 0.5050

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
| class_prior | direction/mcc | beats | does not beat | beats |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | does not beat | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
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

- n_trades: 164
- total_return: -0.3306
- sharpe_net: -95.5612
- sharpe_gross: 7.4551
- sortino: -110.3010
- max_drawdown: 0.3309
- hit_rate: 0.0976
- profit_factor: 0.0524
- avg_hold_bars: 8.3171
- exposure: 0.1885
- turnover: 269.3588
- fees_paid: 2693.7561
- costs_paid: 3501.8830
- gross_pnl: 195.7284
- net_pnl: -3306.1546
