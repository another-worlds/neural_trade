# Evaluation report - test split - run `20260923T214544Z-6dec27a-9abeb709-only-LAMBDA_CASIMIR__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0257 | -0.0035 | 0.0073 |
| direction AUC | 0.5088 | 0.5009 | 0.5164 |
| direction ECE | 0.0546 | 0.0578 | 0.0781 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5000 |
| delta EV | 0.0000 | 0.0000 | 0.0000 |
| delta corr | 0.0000 | 0.0000 | 0.0000 |
| skill vs zero | 0.0000 | 0.0000 | 0.0000 |
| CRPS ($) | 111.0186 | 136.9906 | 159.7688 |
| PIT KS | 0.0587 | 0.0861 | 0.0608 |
| var/err2 Spearman | 0.3596 | 0.3038 | 0.3500 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0205 | [-0.0650, 0.0248] | NOISE |
| h1 | -0.0071 | [-0.0544, 0.0395] | NOISE |
| h2 | 0.0152 | [-0.0361, 0.0661] | NOISE |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.4313
- delta_dir_align_all: 0.0525
- coherence_primary: 0.2166

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
| class_prior | direction/mcc | beats | does not beat | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | beats |
| logreg_lags | direction/auc | beats | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 258
- total_return: -0.4964
- sharpe_net: -114.7169
- sharpe_gross: -3.0196
- sortino: -132.9901
- max_drawdown: 0.4964
- hit_rate: 0.0814
- profit_factor: 0.0371
- avg_hold_bars: 9.3295
- exposure: 0.3326
- turnover: 373.8720
- fees_paid: 3738.2092
- costs_paid: 4859.6719
- gross_pnl: -104.5883
- net_pnl: -4964.2603
