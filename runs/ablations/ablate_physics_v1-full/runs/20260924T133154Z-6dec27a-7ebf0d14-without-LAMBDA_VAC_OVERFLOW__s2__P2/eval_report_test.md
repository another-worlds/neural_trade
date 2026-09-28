# Evaluation report - test split - run `20260924T133154Z-6dec27a-7ebf0d14-without-LAMBDA_VAC_OVERFLOW__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0462 | -0.0069 | -0.0116 |
| direction AUC | 0.4712 | 0.4944 | 0.5000 |
| direction ECE | 0.0470 | 0.0311 | 0.0510 |
| Gaussian MCC | -0.0402 | -0.0546 | -0.0340 |
| Gaussian AUC | 0.4699 | 0.4712 | 0.4743 |
| delta EV | -0.0051 | -0.0091 | -0.0063 |
| delta corr | -0.0768 | -0.0651 | -0.0633 |
| skill vs zero | -0.0061 | -0.0120 | -0.0084 |
| CRPS ($) | 105.2334 | 127.3084 | 145.0328 |
| PIT KS | 0.0304 | 0.0373 | 0.0387 |
| var/err2 Spearman | 0.2814 | 0.2615 | 0.2529 |
| coverage 90% | 0.9046 | 0.9075 | 0.9121 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0230 | [-0.0688, 0.0184] | NOISE |
| h1 | -0.0056 | [-0.0488, 0.0302] | NOISE |
| h2 | 0.0076 | [-0.0430, 0.0505] | NOISE |

## Coherence across horizons

- mag_order_full: 0.2771
- unanimity: 0.3093
- delta_dir_align_all: 0.2605
- coherence_primary: 0.5557

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
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | beats | beats |
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

- n_trades: 178
- total_return: -0.3469
- sharpe_net: -95.9569
- sharpe_gross: 10.2552
- sortino: -111.8630
- max_drawdown: 0.3481
- hit_rate: 0.1124
- profit_factor: 0.0491
- avg_hold_bars: 10.8371
- exposure: 0.2666
- turnover: 288.9532
- fees_paid: 2889.5381
- costs_paid: 3756.3996
- gross_pnl: 287.5652
- net_pnl: -3468.8343
