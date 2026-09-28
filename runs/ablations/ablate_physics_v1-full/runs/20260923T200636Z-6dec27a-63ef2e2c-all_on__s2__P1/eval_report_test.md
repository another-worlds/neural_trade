# Evaluation report - test split - run `20260923T200636Z-6dec27a-63ef2e2c-all_on__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0079 | -0.0453 | 0.0452 |
| direction AUC | 0.4904 | 0.4623 | 0.5316 |
| direction ECE | 0.0620 | 0.0923 | 0.0558 |
| Gaussian MCC | 0.0302 | 0.0498 | 0.0269 |
| Gaussian AUC | 0.5375 | 0.5569 | 0.5425 |
| delta EV | 0.0211 | 0.0208 | 0.0259 |
| delta corr | 0.1471 | 0.1650 | 0.1837 |
| skill vs zero | 0.0136 | 0.0180 | 0.0215 |
| CRPS ($) | 109.1048 | 133.2334 | 156.1045 |
| PIT KS | 0.0506 | 0.0430 | 0.0548 |
| var/err2 Spearman | 0.4261 | 0.4145 | 0.4186 |
| coverage 90% | 0.9261 | 0.9299 | 0.9256 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0268 | [-0.0690, 0.0151] | NOISE |
| h1 | -0.0403 | [-0.0809, 0.0067] | NOISE |
| h2 | -0.0010 | [-0.0466, 0.0476] | NOISE |

## Coherence across horizons

- mag_order_full: 0.1581
- unanimity: 0.5399
- delta_dir_align_all: 0.2514
- coherence_primary: 0.5363

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
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | beats |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | does not beat | beats |

## Backtest (costs included)

- n_trades: 279
- total_return: -0.5248
- sharpe_net: -125.7043
- sharpe_gross: -2.9124
- sortino: -146.8498
- max_drawdown: 0.5248
- hit_rate: 0.0896
- profit_factor: 0.0528
- avg_hold_bars: 11.3047
- exposure: 0.4359
- turnover: 396.7652
- fees_paid: 3967.4216
- costs_paid: 5157.6480
- gross_pnl: -90.0573
- net_pnl: -5247.7054
