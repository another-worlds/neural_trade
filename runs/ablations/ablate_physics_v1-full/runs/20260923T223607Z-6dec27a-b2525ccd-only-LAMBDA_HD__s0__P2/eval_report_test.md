# Evaluation report - test split - run `20260923T223607Z-6dec27a-b2525ccd-only-LAMBDA_HD__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0204 | -0.0401 | 0.0307 |
| direction AUC | 0.5218 | 0.4739 | 0.5270 |
| direction ECE | 0.0292 | 0.0332 | 0.0309 |
| Gaussian MCC | 0.0064 | 0.0051 | 0.0122 |
| Gaussian AUC | 0.5029 | 0.5033 | 0.5009 |
| delta EV | -0.0035 | -0.0085 | -0.0068 |
| delta corr | -0.0059 | -0.0163 | -0.0045 |
| skill vs zero | -0.0075 | -0.0094 | -0.0076 |
| CRPS ($) | 106.1660 | 127.9291 | 145.8908 |
| PIT KS | 0.0657 | 0.0484 | 0.0504 |
| var/err2 Spearman | 0.2645 | 0.2526 | 0.2275 |
| coverage 90% | 0.9055 | 0.9103 | 0.9125 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0283 | [-0.0131, 0.0732] | NOISE |
| h1 | -0.0218 | [-0.0698, 0.0178] | NOISE |
| h2 | 0.0293 | [-0.0130, 0.0708] | NOISE |

## Coherence across horizons

- mag_order_full: 0.3489
- unanimity: 0.3606
- delta_dir_align_all: 0.2403
- coherence_primary: 0.6683

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
| class_prior | direction/mcc | beats | does not beat | beats |
| class_prior | direction/auc | beats | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | beats |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | does not beat | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
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

- n_trades: 255
- total_return: -0.4747
- sharpe_net: -123.5538
- sharpe_gross: 6.4502
- sortino: -140.8571
- max_drawdown: 0.4747
- hit_rate: 0.0627
- profit_factor: 0.0226
- avg_hold_bars: 8.7529
- exposure: 0.3085
- turnover: 379.7685
- fees_paid: 3797.7601
- costs_paid: 4937.0881
- gross_pnl: 189.9513
- net_pnl: -4747.1367
