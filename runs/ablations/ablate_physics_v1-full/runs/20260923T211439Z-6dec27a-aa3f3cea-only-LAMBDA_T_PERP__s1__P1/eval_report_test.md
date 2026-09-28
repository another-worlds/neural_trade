# Evaluation report - test split - run `20260923T211439Z-6dec27a-aa3f3cea-only-LAMBDA_T_PERP__s1__P1`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0018 | -0.0129 | -0.0186 |
| direction AUC | 0.5035 | 0.4876 | 0.4881 |
| direction ECE | 0.0583 | 0.0891 | 0.0742 |
| Gaussian MCC | 0.0574 | 0.0672 | 0.0373 |
| Gaussian AUC | 0.5232 | 0.5299 | 0.5301 |
| delta EV | 0.0054 | 0.0050 | 0.0071 |
| delta corr | 0.0767 | 0.1195 | 0.1389 |
| skill vs zero | 0.0012 | 0.0035 | 0.0052 |
| CRPS ($) | 110.3265 | 134.9573 | 158.0335 |
| PIT KS | 0.0448 | 0.0438 | 0.0495 |
| var/err2 Spearman | 0.3904 | 0.3990 | 0.3914 |
| coverage 90% | 0.9229 | 0.9279 | 0.9268 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0080 | [-0.0559, 0.0351] | NOISE |
| h1 | -0.0310 | [-0.0780, 0.0170] | NOISE |
| h2 | -0.0206 | [-0.0710, 0.0400] | NOISE |

## Coherence across horizons

- mag_order_full: 0.2580
- unanimity: 0.6736
- delta_dir_align_all: 0.5231
- coherence_primary: 0.8281

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | beats |
| zero_delta | delta/mae | does not beat | beats | beats |
| zero_delta | delta/ev | beats | beats | beats |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | beats | beats |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | beats | beats | beats |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | beats | does not beat | does not beat |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | does not beat | does not beat | does not beat |
| class_prior | direction/bal_acc | beats | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | beats | beats | does not beat |
| logreg_lags | direction/auc | beats | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | beats | beats | does not beat |

## Backtest (costs included)

- n_trades: 344
- total_return: -0.5820
- sharpe_net: -136.9976
- sharpe_gross: 4.3249
- sortino: -159.0935
- max_drawdown: 0.5827
- hit_rate: 0.1047
- profit_factor: 0.0493
- avg_hold_bars: 10.9273
- exposure: 0.5195
- turnover: 457.0701
- fees_paid: 4570.6206
- costs_paid: 5941.8068
- gross_pnl: 122.1686
- net_pnl: -5819.6382
