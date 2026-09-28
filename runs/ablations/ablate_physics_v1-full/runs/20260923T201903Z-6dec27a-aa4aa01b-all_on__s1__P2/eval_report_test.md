# Evaluation report - test split - run `20260923T201903Z-6dec27a-aa4aa01b-all_on__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0072 | 0.0004 | -0.0494 |
| direction AUC | 0.5014 | 0.4949 | 0.4661 |
| direction ECE | 0.0273 | 0.0343 | 0.0408 |
| Gaussian MCC | 0.0016 | 0.0015 | -0.0220 |
| Gaussian AUC | 0.4884 | 0.4844 | 0.4828 |
| delta EV | -0.0079 | -0.0091 | -0.0162 |
| delta corr | 0.0229 | -0.0055 | -0.0213 |
| skill vs zero | -0.0090 | -0.0090 | -0.0162 |
| CRPS ($) | 105.9306 | 127.3187 | 145.9690 |
| PIT KS | 0.0505 | 0.0374 | 0.0432 |
| var/err2 Spearman | 0.2494 | 0.2545 | 0.2539 |
| coverage 90% | 0.9038 | 0.9093 | 0.9109 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0103 | [-0.0564, 0.0358] | NOISE |
| h1 | -0.0217 | [-0.0569, 0.0136] | NOISE |
| h2 | -0.0118 | [-0.0547, 0.0268] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3318
- unanimity: 0.4089
- delta_dir_align_all: 0.2468
- coherence_primary: 0.5862

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | beats | beats | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | does not beat |
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

- n_trades: 216
- total_return: -0.4445
- sharpe_net: -123.4148
- sharpe_gross: -6.6242
- sortino: -138.5923
- max_drawdown: 0.4453
- hit_rate: 0.0556
- profit_factor: 0.0213
- avg_hold_bars: 9.0463
- exposure: 0.2702
- turnover: 327.3698
- fees_paid: 3273.9241
- costs_paid: 4256.1013
- gross_pnl: -188.9716
- net_pnl: -4445.0730
