# Evaluation report - test split - run `20260924T130411Z-6dec27a-10461644-without-LAMBDA_VAC_OVERFLOW__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0508 | -0.0111 | 0.0240 |
| direction AUC | 0.4710 | 0.5053 | 0.5154 |
| direction ECE | 0.0735 | 0.0653 | 0.0449 |
| Gaussian MCC | 0.0307 | 0.0414 | 0.0331 |
| Gaussian AUC | 0.5299 | 0.5218 | 0.5291 |
| delta EV | 0.0176 | 0.0211 | 0.0226 |
| delta corr | 0.1506 | 0.1555 | 0.1698 |
| skill vs zero | 0.0168 | 0.0198 | 0.0221 |
| CRPS ($) | 108.5547 | 132.9916 | 155.8557 |
| PIT KS | 0.0437 | 0.0478 | 0.0357 |
| var/err2 Spearman | 0.4454 | 0.4135 | 0.4112 |
| coverage 90% | 0.9233 | 0.9250 | 0.9276 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0176 | [-0.0559, 0.0248] | NOISE |
| h1 | 0.0125 | [-0.0243, 0.0541] | NOISE |
| h2 | -0.0063 | [-0.0566, 0.0416] | NOISE |

## Coherence across horizons

- mag_order_full: 0.5221
- unanimity: 0.4099
- delta_dir_align_all: 0.2644
- coherence_primary: 0.5999

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
| class_prior | direction/auc | does not beat | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | beats |
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | does not beat | beats | beats |

## Backtest (costs included)

- n_trades: 318
- total_return: -0.5532
- sharpe_net: -127.7913
- sharpe_gross: 4.8844
- sortino: -150.7476
- max_drawdown: 0.5532
- hit_rate: 0.1352
- profit_factor: 0.0763
- avg_hold_bars: 9.7484
- exposure: 0.4284
- turnover: 436.8016
- fees_paid: 4367.9429
- costs_paid: 5678.3257
- gross_pnl: 146.7097
- net_pnl: -5531.6160
