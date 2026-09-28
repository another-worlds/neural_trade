# Evaluation report - test split - run `20260924T123416Z-6dec27a-e5e02f0f-without-LAMBDA_IFE__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0135 | -0.0253 | -0.0419 |
| direction AUC | 0.4839 | 0.4829 | 0.4674 |
| direction ECE | 0.0297 | 0.0380 | 0.0310 |
| Gaussian MCC | -0.0212 | -0.0236 | -0.0293 |
| Gaussian AUC | 0.4844 | 0.4742 | 0.4812 |
| delta EV | -0.0051 | -0.0084 | -0.0079 |
| delta corr | 0.0090 | -0.0224 | -0.0124 |
| skill vs zero | -0.0053 | -0.0086 | -0.0077 |
| CRPS ($) | 105.8979 | 127.6430 | 145.7132 |
| PIT KS | 0.0431 | 0.0370 | 0.0420 |
| var/err2 Spearman | 0.2468 | 0.2318 | 0.2238 |
| coverage 90% | 0.9041 | 0.9078 | 0.9150 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0280 | [-0.0766, 0.0189] | NOISE |
| h1 | 0.0063 | [-0.0353, 0.0443] | NOISE |
| h2 | -0.0259 | [-0.0686, 0.0245] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3697
- unanimity: 0.3541
- delta_dir_align_all: 0.2416
- coherence_primary: 0.6128

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
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

- n_trades: 290
- total_return: -0.5255
- sharpe_net: -143.8717
- sharpe_gross: 2.7728
- sortino: -161.0564
- max_drawdown: 0.5259
- hit_rate: 0.0724
- profit_factor: 0.0326
- avg_hold_bars: 9.8448
- exposure: 0.3947
- turnover: 410.0918
- fees_paid: 4100.8304
- costs_paid: 5331.0795
- gross_pnl: 75.5871
- net_pnl: -5255.4924
