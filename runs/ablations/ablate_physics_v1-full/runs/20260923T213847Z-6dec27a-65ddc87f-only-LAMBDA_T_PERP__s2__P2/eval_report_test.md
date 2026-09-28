# Evaluation report - test split - run `20260923T213847Z-6dec27a-65ddc87f-only-LAMBDA_T_PERP__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0445 | 0.0194 | 0.0140 |
| direction AUC | 0.5149 | 0.4958 | 0.5188 |
| direction ECE | 0.0420 | 0.0328 | 0.0565 |
| Gaussian MCC | -0.0348 | -0.0266 | 0.0000 |
| Gaussian AUC | 0.4700 | 0.4685 | 0.5000 |
| delta EV | -0.0055 | -0.0015 | 0.0000 |
| delta corr | -0.0472 | -0.0424 | 0.0000 |
| skill vs zero | -0.0057 | -0.0015 | 0.0000 |
| CRPS ($) | 105.8285 | 127.0469 | 145.0496 |
| PIT KS | 0.0224 | 0.0196 | 0.0315 |
| var/err2 Spearman | 0.2538 | 0.2400 | 0.2148 |
| coverage 90% | 0.9046 | 0.9067 | 0.9124 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0070 | [-0.0469, 0.0478] | NOISE |
| h1 | -0.0455 | [-0.0773, -0.0106] | INVERTED |
| h2 | 0.0301 | [-0.0172, 0.0733] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.6226
- delta_dir_align_all: 0.3232
- coherence_primary: 0.5281

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | beats |
| mean_delta | delta/mae | does not beat | does not beat | beats |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 230
- total_return: -0.4485
- sharpe_net: -127.0573
- sharpe_gross: -0.3655
- sortino: -142.9276
- max_drawdown: 0.4485
- hit_rate: 0.0783
- profit_factor: 0.0261
- avg_hold_bars: 7.7870
- exposure: 0.2475
- turnover: 344.0549
- fees_paid: 3440.7704
- costs_paid: 4473.0016
- gross_pnl: -12.4400
- net_pnl: -4485.4416
