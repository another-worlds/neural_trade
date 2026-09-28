# Evaluation report - test split - run `20260924T021321Z-6dec27a-87535633-without-LAMBDA_T_PERP__s2__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0221 | -0.0298 | -0.0011 |
| direction AUC | 0.4874 | 0.4816 | 0.5127 |
| direction ECE | 0.0679 | 0.0868 | 0.0560 |
| Gaussian MCC | 0.0120 | 0.0636 | 0.0000 |
| Gaussian AUC | 0.5092 | 0.5331 | 0.5000 |
| delta EV | -0.0009 | 0.0013 | 0.0000 |
| delta corr | 0.0227 | 0.0367 | 0.0000 |
| skill vs zero | -0.0030 | -0.0006 | 0.0000 |
| CRPS ($) | 109.4681 | 133.7151 | 157.3743 |
| PIT KS | 0.0421 | 0.0451 | 0.0432 |
| var/err2 Spearman | 0.4403 | 0.4200 | 0.3908 |
| coverage 90% | 0.9237 | 0.9270 | 0.9250 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0293 | [-0.0868, 0.0252] | NOISE |
| h1 | -0.0304 | [-0.0764, 0.0132] | NOISE |
| h2 | -0.0180 | [-0.0625, 0.0273] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.6206
- delta_dir_align_all: 0.1122
- coherence_primary: 0.7595

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | beats | does not beat |
| zero_delta | delta/corr | beats | beats | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | does not beat | beats | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | beats | beats |
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
| logreg_lags | direction/mcc | does not beat | does not beat | beats |
| logreg_lags | direction/auc | does not beat | does not beat | beats |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | beats |

## Backtest (costs included)

- n_trades: 323
- total_return: -0.5610
- sharpe_net: -130.9160
- sharpe_gross: 6.2262
- sortino: -151.6909
- max_drawdown: 0.5610
- hit_rate: 0.1084
- profit_factor: 0.0444
- avg_hold_bars: 10.5975
- exposure: 0.4731
- turnover: 445.7763
- fees_paid: 4457.5544
- costs_paid: 5794.8208
- gross_pnl: 185.2263
- net_pnl: -5609.5945
