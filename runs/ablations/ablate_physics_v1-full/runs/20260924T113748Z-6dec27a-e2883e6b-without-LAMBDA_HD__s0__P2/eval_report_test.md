# Evaluation report - test split - run `20260924T113748Z-6dec27a-e2883e6b-without-LAMBDA_HD__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0399 | 0.0375 | 0.0650 |
| direction AUC | 0.5207 | 0.5234 | 0.5322 |
| direction ECE | 0.0227 | 0.0203 | 0.0278 |
| Gaussian MCC | -0.0564 | 0.0258 | 0.0428 |
| Gaussian AUC | 0.4835 | 0.5058 | 0.5112 |
| delta EV | -0.0005 | -0.0010 | -0.0036 |
| delta corr | -0.0245 | -0.0114 | -0.0092 |
| skill vs zero | -0.0002 | -0.0007 | -0.0036 |
| CRPS ($) | 106.2343 | 128.0024 | 146.5757 |
| PIT KS | 0.0521 | 0.0470 | 0.0566 |
| var/err2 Spearman | 0.2472 | 0.2191 | 0.1977 |
| coverage 90% | 0.9037 | 0.9091 | 0.9122 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0007 | [-0.0412, 0.0460] | NOISE |
| h1 | 0.0017 | [-0.0440, 0.0423] | NOISE |
| h2 | -0.0071 | [-0.0415, 0.0281] | NOISE |

## Coherence across horizons

- mag_order_full: 0.6917
- unanimity: 0.5188
- delta_dir_align_all: 0.1569
- coherence_primary: 0.4602

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | beats | does not beat | does not beat |
| mean_delta | delta/mae | beats | beats | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | beats | beats |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | beats |
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | does not beat | beats |

## Backtest (costs included)

- n_trades: 319
- total_return: -0.5508
- sharpe_net: -141.1816
- sharpe_gross: 6.9098
- sortino: -159.5548
- max_drawdown: 0.5511
- hit_rate: 0.0815
- profit_factor: 0.0369
- avg_hold_bars: 10.2915
- exposure: 0.4538
- turnover: 440.2775
- fees_paid: 4402.7989
- costs_paid: 5723.6385
- gross_pnl: 215.9228
- net_pnl: -5507.7157
