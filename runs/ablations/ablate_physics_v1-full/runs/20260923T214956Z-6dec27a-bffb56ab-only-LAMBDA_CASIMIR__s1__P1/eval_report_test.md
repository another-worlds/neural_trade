# Evaluation report - test split - run `20260923T214956Z-6dec27a-bffb56ab-only-LAMBDA_CASIMIR__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0011 | -0.0195 | 0.0270 |
| direction AUC | 0.5013 | 0.4799 | 0.5152 |
| direction ECE | 0.0537 | 0.0856 | 0.0642 |
| Gaussian MCC | -0.0064 | 0.0237 | 0.0176 |
| Gaussian AUC | 0.4928 | 0.5084 | 0.5159 |
| delta EV | -0.0127 | -0.0228 | -0.0181 |
| delta corr | 0.0008 | 0.0160 | 0.0158 |
| skill vs zero | -0.0160 | -0.0200 | -0.0151 |
| CRPS ($) | 110.7844 | 136.2989 | 159.6088 |
| PIT KS | 0.0549 | 0.0366 | 0.0404 |
| var/err2 Spearman | 0.4178 | 0.4025 | 0.4003 |
| coverage 90% | 0.9186 | 0.9223 | 0.9218 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0194 | [-0.0301, 0.0634] | NOISE |
| h1 | -0.0319 | [-0.0808, 0.0136] | NOISE |
| h2 | -0.0141 | [-0.0651, 0.0308] | NOISE |

## Coherence across horizons

- mag_order_full: 0.2344
- unanimity: 0.6052
- delta_dir_align_all: 0.1973
- coherence_primary: 0.4019

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | does not beat |
| mean_delta | delta/mae | does not beat | does not beat | does not beat |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| class_prior | direction/mcc | does not beat | does not beat | beats |
| class_prior | direction/auc | beats | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | beats |
| logreg_lags | direction/auc | beats | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 354
- total_return: -0.6023
- sharpe_net: -146.1846
- sharpe_gross: 1.9180
- sortino: -167.2575
- max_drawdown: 0.6023
- hit_rate: 0.0706
- profit_factor: 0.0485
- avg_hold_bars: 9.1525
- exposure: 0.4478
- turnover: 467.3960
- fees_paid: 4673.5900
- costs_paid: 6075.6671
- gross_pnl: 52.8641
- net_pnl: -6022.8029
