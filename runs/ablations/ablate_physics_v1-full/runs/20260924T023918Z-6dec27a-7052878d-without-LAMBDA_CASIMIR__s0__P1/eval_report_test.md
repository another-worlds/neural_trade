# Evaluation report - test split - run `20260924T023918Z-6dec27a-7052878d-without-LAMBDA_CASIMIR__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0524 | -0.0213 | -0.0092 |
| direction AUC | 0.4785 | 0.4919 | 0.4883 |
| direction ECE | 0.0897 | 0.0551 | 0.0754 |
| Gaussian MCC | 0.0217 | 0.0292 | 0.0637 |
| Gaussian AUC | 0.5095 | 0.5317 | 0.5515 |
| delta EV | 0.0003 | 0.0075 | 0.0157 |
| delta corr | 0.0864 | 0.1064 | 0.1284 |
| skill vs zero | 0.0002 | 0.0063 | 0.0151 |
| CRPS ($) | 109.6873 | 134.2495 | 156.5683 |
| PIT KS | 0.0356 | 0.0396 | 0.0386 |
| var/err2 Spearman | 0.4192 | 0.3984 | 0.4076 |
| coverage 90% | 0.9232 | 0.9272 | 0.9270 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0107 | [-0.0541, 0.0329] | NOISE |
| h1 | -0.0184 | [-0.0636, 0.0244] | NOISE |
| h2 | -0.0196 | [-0.0721, 0.0334] | NOISE |

## Coherence across horizons

- mag_order_full: 0.8278
- unanimity: 0.5348
- delta_dir_align_all: 0.2322
- coherence_primary: 0.6294

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
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
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

- n_trades: 246
- total_return: -0.4709
- sharpe_net: -109.6300
- sharpe_gross: -0.6193
- sortino: -130.6488
- max_drawdown: 0.4711
- hit_rate: 0.1301
- profit_factor: 0.0629
- avg_hold_bars: 8.5488
- exposure: 0.2906
- turnover: 360.3010
- fees_paid: 3602.8420
- costs_paid: 4683.6946
- gross_pnl: -25.6950
- net_pnl: -4709.3896
