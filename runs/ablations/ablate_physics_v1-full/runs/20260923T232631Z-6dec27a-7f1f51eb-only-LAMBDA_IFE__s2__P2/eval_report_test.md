# Evaluation report - test split - run `20260923T232631Z-6dec27a-7f1f51eb-only-LAMBDA_IFE__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0191 | -0.0804 | -0.0054 |
| direction AUC | 0.4741 | 0.4509 | 0.4958 |
| direction ECE | 0.0292 | 0.0633 | 0.0481 |
| Gaussian MCC | -0.0167 | -0.0405 | -0.0281 |
| Gaussian AUC | 0.4784 | 0.4751 | 0.4790 |
| delta EV | -0.0029 | -0.0008 | -0.0154 |
| delta corr | 0.0099 | 0.0290 | -0.0102 |
| skill vs zero | -0.0036 | -0.0012 | -0.0165 |
| CRPS ($) | 105.8854 | 127.4008 | 146.4132 |
| PIT KS | 0.0268 | 0.0242 | 0.0287 |
| var/err2 Spearman | 0.2457 | 0.2255 | 0.2239 |
| coverage 90% | 0.9004 | 0.9091 | 0.9106 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0303 | [-0.0890, 0.0360] | NOISE |
| h1 | -0.0122 | [-0.0508, 0.0276] | NOISE |
| h2 | 0.0167 | [-0.0475, 0.0694] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4407
- unanimity: 0.3337
- delta_dir_align_all: 0.2884
- coherence_primary: 0.5822

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | does not beat | beats |
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

- n_trades: 221
- total_return: -0.4672
- sharpe_net: -127.6960
- sharpe_gross: -13.9993
- sortino: -144.4389
- max_drawdown: 0.4686
- hit_rate: 0.0679
- profit_factor: 0.0463
- avg_hold_bars: 12.7964
- exposure: 0.3910
- turnover: 327.3651
- fees_paid: 3273.4055
- costs_paid: 4255.4272
- gross_pnl: -416.2579
- net_pnl: -4671.6851
