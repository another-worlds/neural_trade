# Evaluation report - test split - run `20260924T030337Z-6dec27a-e499a12e-without-LAMBDA_CASIMIR__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0494 | -0.0038 | -0.0346 |
| direction AUC | 0.4704 | 0.4926 | 0.4747 |
| direction ECE | 0.0586 | 0.0290 | 0.0219 |
| Gaussian MCC | -0.0060 | -0.0565 | -0.0424 |
| Gaussian AUC | 0.4738 | 0.4668 | 0.4686 |
| delta EV | -0.0017 | -0.0023 | -0.0069 |
| delta corr | 0.0161 | -0.0041 | -0.0391 |
| skill vs zero | -0.0038 | -0.0039 | -0.0079 |
| CRPS ($) | 105.7619 | 127.2427 | 145.4835 |
| PIT KS | 0.0437 | 0.0321 | 0.0351 |
| var/err2 Spearman | 0.2429 | 0.2331 | 0.2245 |
| coverage 90% | 0.9016 | 0.9073 | 0.9132 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0320 | [-0.0840, 0.0096] | NOISE |
| h1 | -0.0234 | [-0.0662, 0.0115] | NOISE |
| h2 | -0.0159 | [-0.0612, 0.0314] | NOISE |

## Coherence across horizons

- mag_order_full: 0.2461
- unanimity: 0.3449
- delta_dir_align_all: 0.2572
- coherence_primary: 0.5498

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
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | beats |
| class_prior | direction/acc | does not beat | beats | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | beats |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 193
- total_return: -0.4209
- sharpe_net: -117.2370
- sharpe_gross: -11.4003
- sortino: -133.5776
- max_drawdown: 0.4209
- hit_rate: 0.0881
- profit_factor: 0.0483
- avg_hold_bars: 9.1399
- exposure: 0.2438
- turnover: 298.2074
- fees_paid: 2981.9522
- costs_paid: 3876.5378
- gross_pnl: -332.5764
- net_pnl: -4209.1142
