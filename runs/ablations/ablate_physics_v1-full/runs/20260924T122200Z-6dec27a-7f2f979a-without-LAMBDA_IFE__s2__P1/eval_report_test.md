# Evaluation report - test split - run `20260924T122200Z-6dec27a-7f2f979a-without-LAMBDA_IFE__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0453 | -0.0149 | -0.0021 |
| direction AUC | 0.4632 | 0.4852 | 0.5025 |
| direction ECE | 0.0723 | 0.0783 | 0.0498 |
| Gaussian MCC | 0.0383 | 0.0207 | 0.0279 |
| Gaussian AUC | 0.5261 | 0.5153 | 0.5129 |
| delta EV | 0.0177 | 0.0165 | 0.0203 |
| delta corr | 0.1351 | 0.1292 | 0.1456 |
| skill vs zero | 0.0119 | 0.0127 | 0.0169 |
| CRPS ($) | 108.8207 | 133.3750 | 156.8885 |
| PIT KS | 0.0457 | 0.0445 | 0.0468 |
| var/err2 Spearman | 0.4440 | 0.4221 | 0.4282 |
| coverage 90% | 0.9248 | 0.9302 | 0.9284 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0378 | [-0.0885, 0.0104] | NOISE |
| h1 | -0.0229 | [-0.0652, 0.0226] | NOISE |
| h2 | -0.0188 | [-0.0665, 0.0274] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3553
- unanimity: 0.5239
- delta_dir_align_all: 0.3630
- coherence_primary: 0.6766

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | beats |
| zero_delta | delta/mae | beats | beats | beats |
| zero_delta | delta/ev | beats | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | beats | beats | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | beats | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | beats | beats |

## Backtest (costs included)

- n_trades: 308
- total_return: -0.5501
- sharpe_net: -132.1156
- sharpe_gross: 0.8964
- sortino: -152.8453
- max_drawdown: 0.5501
- hit_rate: 0.0877
- profit_factor: 0.0408
- avg_hold_bars: 11.0097
- exposure: 0.4686
- turnover: 424.9092
- fees_paid: 4248.5690
- costs_paid: 5523.1397
- gross_pnl: 22.6392
- net_pnl: -5500.5005
