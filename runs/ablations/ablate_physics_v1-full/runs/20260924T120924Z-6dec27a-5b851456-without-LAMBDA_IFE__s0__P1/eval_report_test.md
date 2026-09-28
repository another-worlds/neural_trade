# Evaluation report - test split - run `20260924T120924Z-6dec27a-5b851456-without-LAMBDA_IFE__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0063 | -0.0128 | -0.0114 |
| direction AUC | 0.4923 | 0.5029 | 0.5007 |
| direction ECE | 0.0906 | 0.0602 | 0.0851 |
| Gaussian MCC | 0.0375 | 0.0369 | 0.0418 |
| Gaussian AUC | 0.5244 | 0.5261 | 0.5312 |
| delta EV | -0.0003 | 0.0003 | 0.0061 |
| delta corr | 0.0744 | 0.0559 | 0.0794 |
| skill vs zero | -0.0100 | -0.0044 | 0.0029 |
| CRPS ($) | 110.2693 | 134.4945 | 157.4578 |
| PIT KS | 0.0608 | 0.0441 | 0.0420 |
| var/err2 Spearman | 0.4286 | 0.4131 | 0.4208 |
| coverage 90% | 0.9225 | 0.9263 | 0.9276 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0253 | [-0.0770, 0.0264] | NOISE |
| h1 | 0.0161 | [-0.0228, 0.0682] | NOISE |
| h2 | -0.0083 | [-0.0647, 0.0481] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0470
- unanimity: 0.6444
- delta_dir_align_all: 0.4295
- coherence_primary: 0.7598

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | beats |
| zero_delta | delta/mae | does not beat | does not beat | beats |
| zero_delta | delta/ev | does not beat | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| mean_delta | delta/rmse | does not beat | does not beat | beats |
| mean_delta | delta/mae | does not beat | does not beat | beats |
| mean_delta | delta/ev | does not beat | beats | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | does not beat |
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 294
- total_return: -0.5389
- sharpe_net: -123.4975
- sharpe_gross: -2.7765
- sortino: -144.6242
- max_drawdown: 0.5389
- hit_rate: 0.1020
- profit_factor: 0.0525
- avg_hold_bars: 8.2313
- exposure: 0.3344
- turnover: 407.3153
- fees_paid: 4072.9561
- costs_paid: 5294.8430
- gross_pnl: -93.9268
- net_pnl: -5388.7698
