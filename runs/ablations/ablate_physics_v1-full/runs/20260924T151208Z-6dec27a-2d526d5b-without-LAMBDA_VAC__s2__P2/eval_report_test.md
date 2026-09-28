# Evaluation report - test split - run `20260924T151208Z-6dec27a-2d526d5b-without-LAMBDA_VAC__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0014 | -0.0334 | -0.0165 |
| direction AUC | 0.4943 | 0.4811 | 0.4906 |
| direction ECE | 0.0128 | 0.0366 | 0.0397 |
| Gaussian MCC | 0.0089 | 0.0034 | 0.0013 |
| Gaussian AUC | 0.4929 | 0.4994 | 0.5002 |
| delta EV | -0.0163 | -0.0043 | -0.0112 |
| delta corr | -0.0052 | 0.0357 | 0.0272 |
| skill vs zero | -0.0157 | -0.0057 | -0.0131 |
| CRPS ($) | 106.1044 | 127.4128 | 145.7364 |
| PIT KS | 0.0315 | 0.0428 | 0.0436 |
| var/err2 Spearman | 0.2676 | 0.2515 | 0.2467 |
| coverage 90% | 0.9017 | 0.9064 | 0.9113 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0156 | [-0.0578, 0.0221] | NOISE |
| h1 | -0.0127 | [-0.0503, 0.0317] | NOISE |
| h2 | -0.0164 | [-0.0588, 0.0361] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4414
- unanimity: 0.3836
- delta_dir_align_all: 0.2980
- coherence_primary: 0.6454

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | beats | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | beats | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 286
- total_return: -0.5383
- sharpe_net: -144.8012
- sharpe_gross: -8.9858
- sortino: -162.6815
- max_drawdown: 0.5396
- hit_rate: 0.0629
- profit_factor: 0.0323
- avg_hold_bars: 12.8601
- exposure: 0.5084
- turnover: 393.4874
- fees_paid: 3935.1387
- costs_paid: 5115.6804
- gross_pnl: -267.7867
- net_pnl: -5383.4670
