# Evaluation report - test split - run `20260923T222516Z-6dec27a-ebcdcc31-only-LAMBDA_HD__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0165 | -0.0295 | -0.0211 |
| direction AUC | 0.4940 | 0.4709 | 0.4703 |
| direction ECE | 0.0599 | 0.0995 | 0.0763 |
| Gaussian MCC | 0.0117 | 0.0149 | 0.0114 |
| Gaussian AUC | 0.5024 | 0.5164 | 0.5200 |
| delta EV | 0.0004 | 0.0007 | 0.0011 |
| delta corr | 0.0312 | 0.0284 | 0.0333 |
| skill vs zero | -0.0050 | -0.0001 | -0.0007 |
| CRPS ($) | 110.5812 | 135.4181 | 158.8300 |
| PIT KS | 0.0557 | 0.0514 | 0.0581 |
| var/err2 Spearman | 0.4031 | 0.4177 | 0.4112 |
| coverage 90% | 0.9254 | 0.9277 | 0.9254 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0042 | [-0.0492, 0.0484] | NOISE |
| h1 | -0.0546 | [-0.1020, -0.0029] | INVERTED |
| h2 | -0.0400 | [-0.0980, 0.0174] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0488
- unanimity: 0.6346
- delta_dir_align_all: 0.5803
- coherence_primary: 0.8704

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | beats | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | beats | beats | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | does not beat |

## Backtest (costs included)

- n_trades: 289
- total_return: -0.5211
- sharpe_net: -122.5806
- sharpe_gross: 5.3183
- sortino: -143.3427
- max_drawdown: 0.5211
- hit_rate: 0.1384
- profit_factor: 0.0613
- avg_hold_bars: 11.6159
- exposure: 0.4639
- turnover: 412.6335
- fees_paid: 4126.0931
- costs_paid: 5363.9210
- gross_pnl: 152.6686
- net_pnl: -5211.2524
