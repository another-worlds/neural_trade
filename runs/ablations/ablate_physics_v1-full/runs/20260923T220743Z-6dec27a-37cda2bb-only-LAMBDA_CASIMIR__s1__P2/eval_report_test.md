# Evaluation report - test split - run `20260923T220743Z-6dec27a-37cda2bb-only-LAMBDA_CASIMIR__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0220 | -0.0379 | -0.0757 |
| direction AUC | 0.4858 | 0.4746 | 0.4550 |
| direction ECE | 0.0378 | 0.0421 | 0.0512 |
| Gaussian MCC | -0.0765 | -0.0579 | -0.0475 |
| Gaussian AUC | 0.4553 | 0.4606 | 0.4629 |
| delta EV | -0.0091 | -0.0108 | -0.0154 |
| delta corr | -0.0399 | -0.0610 | -0.0577 |
| skill vs zero | -0.0086 | -0.0105 | -0.0171 |
| CRPS ($) | 105.8047 | 127.5032 | 146.0670 |
| PIT KS | 0.0226 | 0.0170 | 0.0273 |
| var/err2 Spearman | 0.2500 | 0.2371 | 0.2288 |
| coverage 90% | 0.9024 | 0.9063 | 0.9109 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0082 | [-0.0537, 0.0384] | NOISE |
| h1 | -0.0241 | [-0.0661, 0.0155] | NOISE |
| h2 | -0.0229 | [-0.0810, 0.0311] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4121
- unanimity: 0.3665
- delta_dir_align_all: 0.1903
- coherence_primary: 0.5153

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
| class_prior | direction/acc | does not beat | does not beat | does not beat |
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

- n_trades: 207
- total_return: -0.4369
- sharpe_net: -124.7766
- sharpe_gross: -9.1783
- sortino: -139.7038
- max_drawdown: 0.4369
- hit_rate: 0.0725
- profit_factor: 0.0469
- avg_hold_bars: 9.3237
- exposure: 0.2669
- turnover: 316.9139
- fees_paid: 3168.9723
- costs_paid: 4119.6640
- gross_pnl: -249.4414
- net_pnl: -4369.1054
