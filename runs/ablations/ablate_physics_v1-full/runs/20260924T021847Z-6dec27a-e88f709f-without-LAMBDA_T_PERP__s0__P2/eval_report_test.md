# Evaluation report - test split - run `20260924T021847Z-6dec27a-e88f709f-without-LAMBDA_T_PERP__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0273 | 0.0007 | 0.0260 |
| direction AUC | 0.5094 | 0.4897 | 0.5152 |
| direction ECE | 0.0357 | 0.0188 | 0.0322 |
| Gaussian MCC | 0.0066 | -0.0016 | -0.0059 |
| Gaussian AUC | 0.4959 | 0.4988 | 0.4992 |
| delta EV | -0.0008 | -0.0137 | -0.0163 |
| delta corr | 0.0029 | -0.0272 | -0.0348 |
| skill vs zero | -0.0012 | -0.0147 | -0.0180 |
| CRPS ($) | 106.0304 | 128.3675 | 146.7748 |
| PIT KS | 0.0552 | 0.0538 | 0.0573 |
| var/err2 Spearman | 0.2541 | 0.2509 | 0.2488 |
| coverage 90% | 0.9017 | 0.9098 | 0.9145 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0127 | [-0.0474, 0.0269] | NOISE |
| h1 | -0.0399 | [-0.0827, -0.0024] | INVERTED |
| h2 | 0.0099 | [-0.0270, 0.0476] | NOISE |

## Coherence across horizons

- mag_order_full: 0.6117
- unanimity: 0.3776
- delta_dir_align_all: 0.2370
- coherence_primary: 0.6493

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
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | beats | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | beats | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 218
- total_return: -0.4308
- sharpe_net: -115.1865
- sharpe_gross: 0.7769
- sortino: -131.3396
- max_drawdown: 0.4313
- hit_rate: 0.0872
- profit_factor: 0.0445
- avg_hold_bars: 8.4404
- exposure: 0.2543
- turnover: 332.8821
- fees_paid: 3328.9133
- costs_paid: 4327.5873
- gross_pnl: 19.4276
- net_pnl: -4308.1597
