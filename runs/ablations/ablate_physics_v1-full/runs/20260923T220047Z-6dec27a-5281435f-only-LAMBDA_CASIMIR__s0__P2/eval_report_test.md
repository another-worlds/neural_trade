# Evaluation report - test split - run `20260923T220047Z-6dec27a-5281435f-only-LAMBDA_CASIMIR__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0319 | 0.0107 | 0.0290 |
| direction AUC | 0.5054 | 0.5143 | 0.5189 |
| direction ECE | 0.0401 | 0.0139 | 0.0255 |
| Gaussian MCC | -0.0184 | 0.0000 | 0.0079 |
| Gaussian AUC | 0.4772 | 0.5000 | 0.5079 |
| delta EV | -0.0030 | 0.0000 | -0.0012 |
| delta corr | 0.0261 | 0.0000 | 0.0174 |
| skill vs zero | -0.0044 | 0.0000 | -0.0009 |
| CRPS ($) | 106.7689 | 128.5702 | 146.1402 |
| PIT KS | 0.0648 | 0.0583 | 0.0537 |
| var/err2 Spearman | 0.2579 | 0.2458 | 0.2294 |
| coverage 90% | 0.9038 | 0.9059 | 0.9135 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0332 | [-0.0726, 0.0068] | NOISE |
| h1 | 0.0141 | [-0.0174, 0.0480] | NOISE |
| h2 | 0.0078 | [-0.0302, 0.0478] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.4031
- delta_dir_align_all: 0.2051
- coherence_primary: 0.4637

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | beats |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | beats | does not beat |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | does not beat | beats |
| mean_delta | delta/skill_vs_zero | does not beat | beats | does not beat |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | beats | beats |
| class_prior | direction/ece_pos | does not beat | beats | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | beats | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 213
- total_return: -0.3943
- sharpe_net: -110.5475
- sharpe_gross: 16.7063
- sortino: -126.6972
- max_drawdown: 0.3949
- hit_rate: 0.0845
- profit_factor: 0.0371
- avg_hold_bars: 8.6385
- exposure: 0.2543
- turnover: 338.6124
- fees_paid: 3386.0715
- costs_paid: 4401.8930
- gross_pnl: 458.8482
- net_pnl: -3943.0448
