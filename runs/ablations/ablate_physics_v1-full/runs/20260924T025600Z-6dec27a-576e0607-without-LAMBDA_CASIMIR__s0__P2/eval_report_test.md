# Evaluation report - test split - run `20260924T025600Z-6dec27a-576e0607-without-LAMBDA_CASIMIR__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0228 | 0.0268 | 0.0528 |
| direction AUC | 0.5241 | 0.5172 | 0.5273 |
| direction ECE | 0.0329 | 0.0157 | 0.0196 |
| Gaussian MCC | 0.0353 | -0.0161 | 0.0286 |
| Gaussian AUC | 0.5167 | 0.4932 | 0.5119 |
| delta EV | -0.0048 | -0.0247 | -0.0320 |
| delta corr | 0.0082 | -0.0246 | -0.0195 |
| skill vs zero | -0.0089 | -0.0352 | -0.0585 |
| CRPS ($) | 106.0294 | 129.1253 | 148.4325 |
| PIT KS | 0.0551 | 0.0566 | 0.0820 |
| var/err2 Spearman | 0.2509 | 0.2392 | 0.2529 |
| coverage 90% | 0.9052 | 0.9057 | 0.9081 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0242 | [-0.0192, 0.0647] | NOISE |
| h1 | -0.0031 | [-0.0448, 0.0352] | NOISE |
| h2 | 0.0115 | [-0.0291, 0.0564] | NOISE |

## Coherence across horizons

- mag_order_full: 0.5402
- unanimity: 0.4941
- delta_dir_align_all: 0.2955
- coherence_primary: 0.5949

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
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | beats | beats |
| class_prior | direction/ece_pos | does not beat | beats | beats |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | does not beat |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | does not beat |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | beats | beats |
| logreg_lags | direction/acc | does not beat | does not beat | beats |
| logreg_lags | direction/bal_acc | does not beat | does not beat | beats |

## Backtest (costs included)

- n_trades: 222
- total_return: -0.4190
- sharpe_net: -109.5085
- sharpe_gross: 7.9498
- sortino: -126.9255
- max_drawdown: 0.4193
- hit_rate: 0.0991
- profit_factor: 0.0486
- avg_hold_bars: 9.8964
- exposure: 0.3036
- turnover: 340.7712
- fees_paid: 3407.7996
- costs_paid: 4430.1394
- gross_pnl: 239.9237
- net_pnl: -4190.2158
