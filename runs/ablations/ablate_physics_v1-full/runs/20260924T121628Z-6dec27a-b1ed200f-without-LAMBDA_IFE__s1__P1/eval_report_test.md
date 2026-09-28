# Evaluation report - test split - run `20260924T121628Z-6dec27a-b1ed200f-without-LAMBDA_IFE__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0353 | 0.0057 | -0.0161 |
| direction AUC | 0.4890 | 0.4859 | 0.4888 |
| direction ECE | 0.0654 | 0.0918 | 0.0749 |
| Gaussian MCC | 0.0110 | -0.0134 | 0.0000 |
| Gaussian AUC | 0.5198 | 0.5045 | 0.5000 |
| delta EV | -0.0007 | -0.0003 | 0.0000 |
| delta corr | 0.0132 | -0.0011 | 0.0000 |
| skill vs zero | -0.0060 | -0.0017 | 0.0000 |
| CRPS ($) | 111.3484 | 136.6688 | 159.4908 |
| PIT KS | 0.0557 | 0.0456 | 0.0465 |
| var/err2 Spearman | 0.4041 | 0.4081 | 0.4059 |
| coverage 90% | 0.9237 | 0.9269 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0209 | [-0.0302, 0.0690] | NOISE |
| h1 | -0.0357 | [-0.0874, 0.0176] | NOISE |
| h2 | -0.0124 | [-0.0771, 0.0418] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.6429
- delta_dir_align_all: 0.0155
- coherence_primary: 0.9156

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | beats | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | does not beat | beats |
| mean_delta | delta/mae | does not beat | does not beat | beats |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | beats | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | does not beat | beats |
| class_prior | direction/mcc | does not beat | beats | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | does not beat | beats | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | does not beat | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | beats |

## Backtest (costs included)

- n_trades: 322
- total_return: -0.5610
- sharpe_net: -131.0752
- sharpe_gross: 3.8752
- sortino: -152.4764
- max_drawdown: 0.5620
- hit_rate: 0.1149
- profit_factor: 0.0532
- avg_hold_bars: 11.7143
- exposure: 0.5213
- turnover: 440.0158
- fees_paid: 4399.9006
- costs_paid: 5719.8708
- gross_pnl: 109.4062
- net_pnl: -5610.4646
