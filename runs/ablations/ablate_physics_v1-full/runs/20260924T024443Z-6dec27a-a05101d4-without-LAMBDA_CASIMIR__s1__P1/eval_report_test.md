# Evaluation report - test split - run `20260924T024443Z-6dec27a-a05101d4-without-LAMBDA_CASIMIR__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0140 | -0.0283 | -0.0249 |
| direction AUC | 0.5031 | 0.4816 | 0.4802 |
| direction ECE | 0.0591 | 0.0941 | 0.0687 |
| Gaussian MCC | 0.0145 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.5194 | 0.5000 | 0.5000 |
| delta EV | 0.0014 | 0.0000 | 0.0000 |
| delta corr | 0.0434 | 0.0000 | 0.0000 |
| skill vs zero | -0.0025 | 0.0000 | 0.0000 |
| CRPS ($) | 109.9277 | 134.8079 | 158.1113 |
| PIT KS | 0.0476 | 0.0435 | 0.0404 |
| var/err2 Spearman | 0.4182 | 0.4228 | 0.4026 |
| coverage 90% | 0.9236 | 0.9279 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0058 | [-0.0568, 0.0438] | NOISE |
| h1 | -0.0213 | [-0.0697, 0.0253] | NOISE |
| h2 | -0.0250 | [-0.0765, 0.0225] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.6278
- delta_dir_align_all: 0.0428
- coherence_primary: 0.1191

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | beats | beats |
| class_prior | direction/mcc | beats | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | beats | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | does not beat |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | does not beat |

## Backtest (costs included)

- n_trades: 323
- total_return: -0.5404
- sharpe_net: -128.1387
- sharpe_gross: 15.0416
- sortino: -149.8674
- max_drawdown: 0.5404
- hit_rate: 0.1176
- profit_factor: 0.0581
- avg_hold_bars: 9.2322
- exposure: 0.4121
- turnover: 448.8594
- fees_paid: 4488.4777
- costs_paid: 5835.0210
- gross_pnl: 431.3709
- net_pnl: -5403.6500
