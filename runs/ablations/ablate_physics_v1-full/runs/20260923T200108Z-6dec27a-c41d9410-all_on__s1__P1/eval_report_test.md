# Evaluation report - test split - run `20260923T200108Z-6dec27a-c41d9410-all_on__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0091 | -0.0284 | -0.0471 |
| direction AUC | 0.5033 | 0.4740 | 0.4632 |
| direction ECE | 0.0642 | 0.0935 | 0.0763 |
| Gaussian MCC | 0.0031 | 0.0515 | 0.0570 |
| Gaussian AUC | 0.5061 | 0.5344 | 0.5402 |
| delta EV | 0.0025 | 0.0027 | 0.0039 |
| delta corr | 0.0495 | 0.0519 | 0.0819 |
| skill vs zero | -0.0006 | 0.0016 | 0.0031 |
| CRPS ($) | 109.8339 | 134.2286 | 157.9256 |
| PIT KS | 0.0480 | 0.0518 | 0.0539 |
| var/err2 Spearman | 0.4196 | 0.4128 | 0.4088 |
| coverage 90% | 0.9239 | 0.9262 | 0.9255 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0129 | [-0.0305, 0.0581] | NOISE |
| h1 | -0.0466 | [-0.0885, 0.0035] | NOISE |
| h2 | -0.0462 | [-0.0941, -0.0041] | INVERTED |

## Coherence across horizons

- mag_order_full: 0.0915
- unanimity: 0.6827
- delta_dir_align_all: 0.5554
- coherence_primary: 0.7776

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | beats | beats |
| zero_delta | delta/mae | does not beat | beats | beats |
| zero_delta | delta/ev | beats | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | beats | beats | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | does not beat | does not beat |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | does not beat |

## Backtest (costs included)

- n_trades: 306
- total_return: -0.5306
- sharpe_net: -122.6168
- sharpe_gross: 10.3638
- sortino: -143.3940
- max_drawdown: 0.5315
- hit_rate: 0.1373
- profit_factor: 0.0672
- avg_hold_bars: 10.9967
- exposure: 0.4650
- turnover: 431.4604
- fees_paid: 4314.2691
- costs_paid: 5608.5498
- gross_pnl: 302.4883
- net_pnl: -5306.0615
