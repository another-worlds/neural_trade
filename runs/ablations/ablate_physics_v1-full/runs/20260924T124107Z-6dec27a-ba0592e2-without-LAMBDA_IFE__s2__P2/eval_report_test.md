# Evaluation report - test split - run `20260924T124107Z-6dec27a-ba0592e2-without-LAMBDA_IFE__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0269 | -0.0017 | 0.0209 |
| direction AUC | 0.4950 | 0.4962 | 0.5171 |
| direction ECE | 0.0470 | 0.0266 | 0.0713 |
| Gaussian MCC | -0.0324 | -0.0760 | 0.0000 |
| Gaussian AUC | 0.4671 | 0.4635 | 0.5000 |
| delta EV | -0.0128 | -0.0008 | 0.0000 |
| delta corr | -0.0593 | -0.0522 | 0.0000 |
| skill vs zero | -0.0145 | -0.0007 | 0.0000 |
| CRPS ($) | 105.9817 | 127.0349 | 144.8752 |
| PIT KS | 0.0330 | 0.0184 | 0.0241 |
| var/err2 Spearman | 0.2635 | 0.2475 | 0.2360 |
| coverage 90% | 0.9056 | 0.9060 | 0.9124 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0040 | [-0.0583, 0.0449] | NOISE |
| h1 | 0.0008 | [-0.0346, 0.0357] | NOISE |
| h2 | 0.0318 | [-0.0200, 0.0780] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.4547
- delta_dir_align_all: 0.3582
- coherence_primary: 0.4924

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | beats |
| mean_delta | delta/mae | does not beat | does not beat | beats |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | beats | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
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

- n_trades: 206
- total_return: -0.3794
- sharpe_net: -101.0072
- sharpe_gross: 15.6813
- sortino: -117.9062
- max_drawdown: 0.3794
- hit_rate: 0.1262
- profit_factor: 0.0641
- avg_hold_bars: 8.7282
- exposure: 0.2485
- turnover: 327.5183
- fees_paid: 3275.3797
- costs_paid: 4257.9936
- gross_pnl: 463.7751
- net_pnl: -3794.2185
