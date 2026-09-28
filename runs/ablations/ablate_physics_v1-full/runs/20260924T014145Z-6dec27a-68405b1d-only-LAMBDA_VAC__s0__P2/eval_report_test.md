# Evaluation report - test split - run `20260924T014145Z-6dec27a-68405b1d-only-LAMBDA_VAC__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0311 | 0.0077 | 0.0397 |
| direction AUC | 0.5210 | 0.5048 | 0.5204 |
| direction ECE | 0.0387 | 0.0180 | 0.0316 |
| Gaussian MCC | 0.0494 | 0.0491 | 0.0330 |
| Gaussian AUC | 0.5042 | 0.5177 | 0.5189 |
| delta EV | -0.0076 | -0.0180 | -0.0355 |
| delta corr | 0.0339 | 0.0226 | 0.0238 |
| skill vs zero | -0.0128 | -0.0239 | -0.0479 |
| CRPS ($) | 106.7298 | 129.0723 | 148.7644 |
| PIT KS | 0.0667 | 0.0624 | 0.0719 |
| var/err2 Spearman | 0.2610 | 0.2540 | 0.2391 |
| coverage 90% | 0.9045 | 0.9062 | 0.9038 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0016 | [-0.0435, 0.0376] | NOISE |
| h1 | 0.0093 | [-0.0400, 0.0487] | NOISE |
| h2 | 0.0143 | [-0.0346, 0.0613] | NOISE |

## Coherence across horizons

- mag_order_full: 0.5376
- unanimity: 0.4522
- delta_dir_align_all: 0.3514
- coherence_primary: 0.6182

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | beats | does not beat |
| class_prior | direction/ece_pos | does not beat | beats | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | does not beat |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | does not beat | does not beat | does not beat |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | beats | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 207
- total_return: -0.4061
- sharpe_net: -107.6354
- sharpe_gross: 4.5931
- sortino: -124.4829
- max_drawdown: 0.4062
- hit_rate: 0.0773
- profit_factor: 0.0358
- avg_hold_bars: 8.6280
- exposure: 0.2468
- turnover: 322.5738
- fees_paid: 3225.8389
- costs_paid: 4193.5905
- gross_pnl: 132.4641
- net_pnl: -4061.1264
