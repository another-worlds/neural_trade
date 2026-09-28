# Evaluation report - test split - run `20260924T022538Z-6dec27a-5d19e662-without-LAMBDA_T_PERP__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0261 | -0.0224 | -0.0535 |
| direction AUC | 0.4810 | 0.4993 | 0.4643 |
| direction ECE | 0.0401 | 0.0317 | 0.0329 |
| Gaussian MCC | 0.0240 | -0.0275 | -0.0008 |
| Gaussian AUC | 0.4957 | 0.4810 | 0.4856 |
| delta EV | 0.0057 | -0.0033 | -0.0176 |
| delta corr | 0.0758 | 0.0103 | -0.0164 |
| skill vs zero | 0.0041 | -0.0045 | -0.0196 |
| CRPS ($) | 105.1434 | 126.9977 | 146.0912 |
| PIT KS | 0.0376 | 0.0310 | 0.0387 |
| var/err2 Spearman | 0.2562 | 0.2477 | 0.2418 |
| coverage 90% | 0.9040 | 0.9069 | 0.9096 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0278 | [-0.0745, 0.0235] | NOISE |
| h1 | 0.0263 | [-0.0107, 0.0623] | NOISE |
| h2 | -0.0058 | [-0.0561, 0.0413] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4169
- unanimity: 0.4063
- delta_dir_align_all: 0.2374
- coherence_primary: 0.5234

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | does not beat | does not beat |
| zero_delta | delta/mae | beats | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | does not beat |
| zero_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
| mean_delta | delta/rmse | beats | does not beat | does not beat |
| mean_delta | delta/mae | beats | does not beat | does not beat |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
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

- n_trades: 312
- total_return: -0.5728
- sharpe_net: -152.4520
- sharpe_gross: -9.7731
- sortino: -170.2745
- max_drawdown: 0.5734
- hit_rate: 0.0609
- profit_factor: 0.0309
- avg_hold_bars: 10.3718
- exposure: 0.4473
- turnover: 417.9251
- fees_paid: 4179.3079
- costs_paid: 5433.1002
- gross_pnl: -294.6650
- net_pnl: -5727.7652
