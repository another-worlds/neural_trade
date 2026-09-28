# Evaluation report - test split - run `20260923T222133Z-6dec27a-be7548f9-only-LAMBDA_HD__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0411 | -0.0169 | -0.0094 |
| direction AUC | 0.4785 | 0.5137 | 0.4929 |
| direction ECE | 0.0645 | 0.0814 | 0.0784 |
| Gaussian MCC | -0.0332 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.4870 | 0.5000 | 0.5000 |
| delta EV | -0.0026 | 0.0000 | 0.0000 |
| delta corr | 0.0141 | 0.0000 | 0.0000 |
| skill vs zero | -0.0055 | 0.0000 | 0.0000 |
| CRPS ($) | 111.3788 | 135.9466 | 159.1376 |
| PIT KS | 0.0631 | 0.0666 | 0.0577 |
| var/err2 Spearman | 0.3439 | 0.3475 | 0.3574 |
| coverage 90% | 0.9168 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0062 | [-0.0382, 0.0434] | NOISE |
| h1 | 0.0188 | [-0.0290, 0.0689] | NOISE |
| h2 | -0.0163 | [-0.0715, 0.0391] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.3622
- delta_dir_align_all: 0.0106
- coherence_primary: 0.0782

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | beats | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | does not beat |
| logreg_lags | direction/auc | does not beat | beats | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | beats |

## Backtest (costs included)

- n_trades: 309
- total_return: -0.5769
- sharpe_net: -142.4931
- sharpe_gross: -13.2739
- sortino: -160.7329
- max_drawdown: 0.5773
- hit_rate: 0.0518
- profit_factor: 0.0216
- avg_hold_bars: 8.6990
- exposure: 0.3716
- turnover: 412.8612
- fees_paid: 4128.3771
- costs_paid: 5366.8902
- gross_pnl: -402.1321
- net_pnl: -5769.0224
