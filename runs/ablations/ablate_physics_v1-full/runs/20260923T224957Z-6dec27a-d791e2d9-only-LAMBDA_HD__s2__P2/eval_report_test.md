# Evaluation report - test split - run `20260923T224957Z-6dec27a-d791e2d9-only-LAMBDA_HD__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0453 | -0.0351 | -0.0014 |
| direction AUC | 0.4680 | 0.4744 | 0.4999 |
| direction ECE | 0.0404 | 0.0447 | 0.0341 |
| Gaussian MCC | -0.0147 | 0.0095 | 0.0044 |
| Gaussian AUC | 0.4843 | 0.4891 | 0.4882 |
| delta EV | -0.0146 | -0.0040 | -0.0069 |
| delta corr | -0.0410 | -0.0486 | -0.0495 |
| skill vs zero | -0.0185 | -0.0056 | -0.0095 |
| CRPS ($) | 106.1049 | 127.1336 | 145.2453 |
| PIT KS | 0.0412 | 0.0309 | 0.0409 |
| var/err2 Spearman | 0.2724 | 0.2518 | 0.2426 |
| coverage 90% | 0.9016 | 0.9078 | 0.9131 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0340 | [-0.0764, 0.0077] | NOISE |
| h1 | -0.0223 | [-0.0584, 0.0085] | NOISE |
| h2 | 0.0163 | [-0.0276, 0.0568] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0836
- unanimity: 0.2622
- delta_dir_align_all: 0.1942
- coherence_primary: 0.6303

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
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
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

- n_trades: 187
- total_return: -0.3945
- sharpe_net: -111.8265
- sharpe_gross: -4.4490
- sortino: -126.7266
- max_drawdown: 0.3952
- hit_rate: 0.1016
- profit_factor: 0.0424
- avg_hold_bars: 11.2246
- exposure: 0.2901
- turnover: 293.4796
- fees_paid: 2934.8679
- costs_paid: 3815.3283
- gross_pnl: -129.2808
- net_pnl: -3944.6091
