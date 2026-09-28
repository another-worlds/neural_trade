# Evaluation report - test split - run `20260924T015539Z-6dec27a-9269d5cd-only-LAMBDA_VAC__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0039 | -0.0647 | -0.0240 |
| direction AUC | 0.5042 | 0.4637 | 0.4939 |
| direction ECE | 0.0137 | 0.0508 | 0.0383 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 105.2230 | 126.8673 | 144.6142 |
| PIT KS | 0.0268 | 0.0232 | 0.0309 |
| var/err2 Spearman | 0.2719 | 0.2616 | 0.2576 |
| coverage 90% | 0.9027 | 0.9059 | 0.9124 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0055 | [-0.0354, 0.0473] | NOISE |
| h1 | -0.0228 | [-0.0641, 0.0166] | NOISE |
| h2 | 0.0298 | [-0.0115, 0.0724] | NOISE |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.2453
- delta_dir_align_all: 0.0773
- coherence_primary: 0.4279

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
| mean_delta | delta/corr | beats | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | beats | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | beats | does not beat | does not beat |
| class_prior | direction/acc | beats | does not beat | beats |
| class_prior | direction/bal_acc | beats | does not beat | does not beat |
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

- n_trades: 260
- total_return: -0.4970
- sharpe_net: -136.7523
- sharpe_gross: -3.9984
- sortino: -153.7265
- max_drawdown: 0.4976
- hit_rate: 0.0769
- profit_factor: 0.0397
- avg_hold_bars: 9.1923
- exposure: 0.3303
- turnover: 373.7120
- fees_paid: 3737.2767
- costs_paid: 4858.4597
- gross_pnl: -111.6747
- net_pnl: -4970.1343
