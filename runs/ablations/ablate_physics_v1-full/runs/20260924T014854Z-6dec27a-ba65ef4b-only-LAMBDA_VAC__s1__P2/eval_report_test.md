# Evaluation report - test split - run `20260924T014854Z-6dec27a-ba65ef4b-only-LAMBDA_VAC__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0026 | -0.0255 | 0.0096 |
| direction AUC | 0.5037 | 0.4863 | 0.4976 |
| direction ECE | 0.0532 | 0.0419 | 0.0351 |
| Gaussian MCC | 0.0228 | -0.0326 | -0.0397 |
| Gaussian AUC | 0.5119 | 0.4930 | 0.4714 |
| delta EV | 0.0022 | -0.0002 | -0.0005 |
| delta corr | 0.0501 | 0.0162 | -0.0085 |
| skill vs zero | 0.0032 | 0.0011 | 0.0002 |
| CRPS ($) | 105.7321 | 126.9064 | 145.1942 |
| PIT KS | 0.0447 | 0.0256 | 0.0360 |
| var/err2 Spearman | 0.2534 | 0.2478 | 0.2381 |
| coverage 90% | 0.9055 | 0.9102 | 0.9122 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0040 | [-0.0573, 0.0495] | NOISE |
| h1 | -0.0078 | [-0.0488, 0.0320] | NOISE |
| h2 | -0.0157 | [-0.0600, 0.0333] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0402
- unanimity: 0.4754
- delta_dir_align_all: 0.0549
- coherence_primary: 0.4831

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | beats |
| zero_delta | delta/mae | beats | beats | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | does not beat |
| zero_delta | delta/skill_vs_zero | beats | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | beats | does not beat | beats |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | does not beat | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 212
- total_return: -0.4323
- sharpe_net: -117.8762
- sharpe_gross: -4.5997
- sortino: -133.9291
- max_drawdown: 0.4323
- hit_rate: 0.0566
- profit_factor: 0.0147
- avg_hold_bars: 7.8160
- exposure: 0.2290
- turnover: 322.1792
- fees_paid: 3221.7808
- costs_paid: 4188.3151
- gross_pnl: -134.6513
- net_pnl: -4322.9663
