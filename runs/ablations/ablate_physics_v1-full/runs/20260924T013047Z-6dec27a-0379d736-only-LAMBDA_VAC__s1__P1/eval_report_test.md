# Evaluation report - test split - run `20260924T013047Z-6dec27a-0379d736-only-LAMBDA_VAC__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0063 | 0.0024 | -0.0368 |
| direction AUC | 0.5071 | 0.4837 | 0.4752 |
| direction ECE | 0.0548 | 0.0768 | 0.0701 |
| Gaussian MCC | -0.0389 | 0.0000 | 0.0235 |
| Gaussian AUC | 0.4971 | 0.5000 | 0.5237 |
| delta EV | 0.0012 | 0.0000 | 0.0009 |
| delta corr | 0.0392 | 0.0000 | 0.0306 |
| skill vs zero | -0.0000 | 0.0000 | 0.0008 |
| CRPS ($) | 110.2703 | 135.8041 | 159.0375 |
| PIT KS | 0.0449 | 0.0422 | 0.0444 |
| var/err2 Spearman | 0.3713 | 0.3392 | 0.3350 |
| coverage 90% | 0.9227 | 0.9279 | 0.9239 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0093 | [-0.0556, 0.0353] | NOISE |
| h1 | -0.0518 | [-0.1038, -0.0064] | INVERTED |
| h2 | -0.0384 | [-0.0845, 0.0066] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.6617
- delta_dir_align_all: 0.0210
- coherence_primary: 0.1143

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | beats |
| zero_delta | delta/mae | does not beat | does not beat | beats |
| zero_delta | delta/ev | beats | does not beat | beats |
| zero_delta | delta/corr | beats | does not beat | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | beats | does not beat | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | beats | beats | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | beats | beats | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | does not beat |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | does not beat |

## Backtest (costs included)

- n_trades: 352
- total_return: -0.5916
- sharpe_net: -139.5936
- sharpe_gross: 2.4002
- sortino: -161.2234
- max_drawdown: 0.5916
- hit_rate: 0.1051
- profit_factor: 0.0569
- avg_hold_bars: 9.0597
- exposure: 0.4407
- turnover: 460.3016
- fees_paid: 4602.9948
- costs_paid: 5983.8933
- gross_pnl: 67.5752
- net_pnl: -5916.3181
