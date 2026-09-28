# Evaluation report - test split - run `20260923T221438Z-6dec27a-2b342340-only-LAMBDA_CASIMIR__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0377 | -0.0534 | 0.0020 |
| direction AUC | 0.4783 | 0.4718 | 0.5025 |
| direction ECE | 0.0353 | 0.0520 | 0.0335 |
| Gaussian MCC | -0.0734 | -0.0508 | -0.0733 |
| Gaussian AUC | 0.4620 | 0.4638 | 0.4682 |
| delta EV | -0.0072 | -0.0026 | -0.0030 |
| delta corr | -0.0591 | -0.0522 | -0.0463 |
| skill vs zero | -0.0078 | -0.0037 | -0.0044 |
| CRPS ($) | 105.6169 | 127.1280 | 145.0996 |
| PIT KS | 0.0242 | 0.0254 | 0.0318 |
| var/err2 Spearman | 0.2670 | 0.2430 | 0.2362 |
| coverage 90% | 0.9046 | 0.9067 | 0.9113 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0232 | [-0.0635, 0.0203] | NOISE |
| h1 | 0.0060 | [-0.0283, 0.0435] | NOISE |
| h2 | 0.0007 | [-0.0387, 0.0361] | NOISE |

## Coherence across horizons

- mag_order_full: 0.2043
- unanimity: 0.3197
- delta_dir_align_all: 0.2600
- coherence_primary: 0.5804

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
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
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 158
- total_return: -0.3163
- sharpe_net: -87.4588
- sharpe_gross: 9.7099
- sortino: -102.4418
- max_drawdown: 0.3168
- hit_rate: 0.1329
- profit_factor: 0.0566
- avg_hold_bars: 11.6646
- exposure: 0.2547
- turnover: 264.8327
- fees_paid: 2648.4427
- costs_paid: 3442.9755
- gross_pnl: 280.0493
- net_pnl: -3162.9263
