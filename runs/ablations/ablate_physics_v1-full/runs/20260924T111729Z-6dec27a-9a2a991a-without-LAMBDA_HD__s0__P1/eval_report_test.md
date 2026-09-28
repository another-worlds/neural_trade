# Evaluation report - test split - run `20260924T111729Z-6dec27a-9a2a991a-without-LAMBDA_HD__s0__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0404 | -0.0008 | -0.0135 |
| direction AUC | 0.4798 | 0.5074 | 0.4912 |
| direction ECE | 0.0826 | 0.0571 | 0.0778 |
| Gaussian MCC | 0.0485 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5223 | 0.5000 | 0.5000 |
| delta EV | 0.0024 | 0.0000 | 0.0000 |
| delta corr | 0.0633 | 0.0000 | 0.0000 |
| skill vs zero | 0.0001 | 0.0000 | 0.0000 |
| CRPS ($) | 109.9761 | 134.4917 | 158.2277 |
| PIT KS | 0.0603 | 0.0461 | 0.0563 |
| var/err2 Spearman | 0.3893 | 0.3781 | 0.3981 |
| coverage 90% | 0.9232 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0311 | [-0.0689, 0.0099] | NOISE |
| h1 | -0.0192 | [-0.0652, 0.0251] | NOISE |
| h2 | -0.0100 | [-0.0555, 0.0351] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.4954
- delta_dir_align_all: 0.0376
- coherence_primary: 0.2172

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | beats | does not beat | does not beat |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | beats | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | does not beat |
| logreg_lags | direction/auc | does not beat | beats | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | does not beat |

## Backtest (costs included)

- n_trades: 263
- total_return: -0.5156
- sharpe_net: -121.0911
- sharpe_gross: -9.1216
- sortino: -141.5245
- max_drawdown: 0.5156
- hit_rate: 0.0951
- profit_factor: 0.0342
- avg_hold_bars: 8.4905
- exposure: 0.3086
- turnover: 373.9340
- fees_paid: 3739.7178
- costs_paid: 4861.6331
- gross_pnl: -294.0732
- net_pnl: -5155.7063
