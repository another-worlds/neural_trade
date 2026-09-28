# Evaluation report - test split - run `20260924T025022Z-6dec27a-e98dc3b1-without-LAMBDA_CASIMIR__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0102 | -0.0200 | 0.0246 |
| direction AUC | 0.4821 | 0.4966 | 0.5201 |
| direction ECE | 0.0559 | 0.0794 | 0.0632 |
| Gaussian MCC | 0.0331 | 0.0197 | 0.0005 |
| Gaussian AUC | 0.5138 | 0.5193 | 0.5093 |
| delta EV | -0.0021 | -0.0077 | -0.0006 |
| delta corr | -0.0014 | -0.0046 | 0.0108 |
| skill vs zero | -0.0038 | -0.0109 | -0.0023 |
| CRPS ($) | 109.0825 | 134.0649 | 157.5016 |
| PIT KS | 0.0362 | 0.0411 | 0.0462 |
| var/err2 Spearman | 0.4500 | 0.4303 | 0.4102 |
| coverage 90% | 0.9237 | 0.9268 | 0.9261 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0344 | [-0.0973, 0.0177] | NOISE |
| h1 | -0.0214 | [-0.0645, 0.0258] | NOISE |
| h2 | -0.0109 | [-0.0583, 0.0437] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3817
- unanimity: 0.5390
- delta_dir_align_all: 0.3825
- coherence_primary: 0.7384

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | does not beat | beats |
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
| logreg_lags | direction/mcc | beats | beats | beats |
| logreg_lags | direction/auc | does not beat | beats | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 314
- total_return: -0.5556
- sharpe_net: -132.5278
- sharpe_gross: 2.7060
- sortino: -153.3670
- max_drawdown: 0.5556
- hit_rate: 0.0860
- profit_factor: 0.0456
- avg_hold_bars: 11.5732
- exposure: 0.5022
- turnover: 433.2423
- fees_paid: 4332.1532
- costs_paid: 5631.7992
- gross_pnl: 76.0509
- net_pnl: -5555.7483
