# Evaluation report - test split - run `20260924T134305Z-6dec27a-23e1d47c-without-LAMBDA_VAC__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0174 | 0.0089 | -0.0282 |
| direction AUC | 0.4872 | 0.4985 | 0.4916 |
| direction ECE | 0.0865 | 0.0640 | 0.0909 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 109.6840 | 134.1464 | 157.4308 |
| PIT KS | 0.0357 | 0.0351 | 0.0365 |
| var/err2 Spearman | 0.4203 | 0.4155 | 0.4220 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0385 | [-0.0935, 0.0164] | NOISE |
| h1 | -0.0329 | [-0.0702, 0.0171] | NOISE |
| h2 | -0.0106 | [-0.0710, 0.0507] | NOISE |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.6900
- delta_dir_align_all: 0.0164
- coherence_primary: 0.1410

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | beats | does not beat |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | beats | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | does not beat | beats | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | does not beat |
| logreg_lags | direction/auc | does not beat | beats | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | does not beat |

## Backtest (costs included)

- n_trades: 266
- total_return: -0.5009
- sharpe_net: -122.1856
- sharpe_gross: -1.5801
- sortino: -140.1693
- max_drawdown: 0.5011
- hit_rate: 0.0827
- profit_factor: 0.0408
- avg_hold_bars: 8.5376
- exposure: 0.3138
- turnover: 381.2545
- fees_paid: 3812.4639
- costs_paid: 4956.2030
- gross_pnl: -52.7958
- net_pnl: -5008.9989
