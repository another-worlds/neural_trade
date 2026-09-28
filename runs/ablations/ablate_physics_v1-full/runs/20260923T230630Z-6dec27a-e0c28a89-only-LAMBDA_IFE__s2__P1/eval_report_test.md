# Evaluation report - test split - run `20260923T230630Z-6dec27a-e0c28a89-only-LAMBDA_IFE__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0047 | -0.0049 | 0.0353 |
| direction AUC | 0.4902 | 0.4979 | 0.5312 |
| direction ECE | 0.0634 | 0.0793 | 0.0609 |
| Gaussian MCC | 0.0592 | 0.0730 | 0.0769 |
| Gaussian AUC | 0.5373 | 0.5367 | 0.5489 |
| delta EV | 0.0028 | 0.0029 | 0.0036 |
| delta corr | 0.0533 | 0.0576 | 0.0860 |
| skill vs zero | -0.0000 | 0.0016 | 0.0023 |
| CRPS ($) | 109.5098 | 134.3390 | 157.8093 |
| PIT KS | 0.0454 | 0.0528 | 0.0439 |
| var/err2 Spearman | 0.4189 | 0.3914 | 0.3895 |
| coverage 90% | 0.9247 | 0.9272 | 0.9258 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0199 | [-0.0802, 0.0378] | NOISE |
| h1 | -0.0178 | [-0.0621, 0.0313] | NOISE |
| h2 | 0.0204 | [-0.0287, 0.0718] | NOISE |

## Coherence across horizons

- mag_order_full: 0.1464
- unanimity: 0.5242
- delta_dir_align_all: 0.3353
- coherence_primary: 0.5983

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | beats | beats |
| zero_delta | delta/mae | does not beat | beats | beats |
| zero_delta | delta/ev | beats | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
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
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | beats |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 238
- total_return: -0.4565
- sharpe_net: -101.1147
- sharpe_gross: 3.1031
- sortino: -121.6186
- max_drawdown: 0.4565
- hit_rate: 0.1261
- profit_factor: 0.0836
- avg_hold_bars: 11.0168
- exposure: 0.3624
- turnover: 359.0409
- fees_paid: 3589.9541
- costs_paid: 4666.9404
- gross_pnl: 102.1361
- net_pnl: -4564.8043
