# Evaluation report - test split - run `20260923T212005Z-6dec27a-bc3f4dd8-only-LAMBDA_T_PERP__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0147 | -0.0309 | 0.0462 |
| direction AUC | 0.4875 | 0.4777 | 0.5261 |
| direction ECE | 0.0490 | 0.0808 | 0.0478 |
| Gaussian MCC | 0.0255 | 0.0444 | 0.0297 |
| Gaussian AUC | 0.5308 | 0.5396 | 0.5262 |
| delta EV | 0.0016 | -0.0056 | -0.0029 |
| delta corr | 0.0790 | 0.0437 | 0.0473 |
| skill vs zero | 0.0015 | -0.0038 | -0.0010 |
| CRPS ($) | 109.5021 | 134.2253 | 158.6991 |
| PIT KS | 0.0372 | 0.0328 | 0.0399 |
| var/err2 Spearman | 0.4288 | 0.4096 | 0.3858 |
| coverage 90% | 0.9239 | 0.9266 | 0.9259 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0064 | [-0.0509, 0.0393] | NOISE |
| h1 | -0.0344 | [-0.0772, 0.0029] | NOISE |
| h2 | 0.0096 | [-0.0351, 0.0560] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4035
- unanimity: 0.4707
- delta_dir_align_all: 0.2148
- coherence_primary: 0.4583

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | does not beat | does not beat |
| zero_delta | delta/mae | beats | beats | beats |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
| mean_delta | delta/rmse | beats | does not beat | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | does not beat | beats |
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 332
- total_return: -0.5601
- sharpe_net: -129.9428
- sharpe_gross: 10.9515
- sortino: -151.0602
- max_drawdown: 0.5601
- hit_rate: 0.1054
- profit_factor: 0.0603
- avg_hold_bars: 10.9157
- exposure: 0.5008
- turnover: 455.4221
- fees_paid: 4553.7394
- costs_paid: 5919.8612
- gross_pnl: 318.4494
- net_pnl: -5601.4118
