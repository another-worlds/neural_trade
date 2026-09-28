# Evaluation report - test split - run `20260924T145858Z-6dec27a-82ba74cf-without-LAMBDA_VAC__s1__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0073 | -0.0412 | -0.0223 |
| direction AUC | 0.4970 | 0.4777 | 0.4793 |
| direction ECE | 0.0279 | 0.0402 | 0.0212 |
| Gaussian MCC | 0.0286 | -0.0431 | -0.0346 |
| Gaussian AUC | 0.5139 | 0.4676 | 0.4914 |
| delta EV | 0.0060 | -0.0005 | -0.0104 |
| delta corr | 0.0783 | 0.0331 | 0.0101 |
| skill vs zero | 0.0058 | 0.0001 | -0.0089 |
| CRPS ($) | 105.2831 | 127.0759 | 145.6387 |
| PIT KS | 0.0449 | 0.0324 | 0.0356 |
| var/err2 Spearman | 0.2513 | 0.2419 | 0.2382 |
| coverage 90% | 0.9031 | 0.9049 | 0.9103 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0047 | [-0.0454, 0.0427] | NOISE |
| h1 | 0.0007 | [-0.0417, 0.0382] | NOISE |
| h2 | -0.0169 | [-0.0644, 0.0249] | NOISE |

## Coherence across horizons

- mag_order_full: 0.4692
- unanimity: 0.4088
- delta_dir_align_all: 0.2761
- coherence_primary: 0.6618

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | does not beat |
| zero_delta | delta/mae | beats | does not beat | does not beat |
| zero_delta | delta/ev | beats | does not beat | does not beat |
| zero_delta | delta/corr | beats | beats | beats |
| zero_delta | delta/skill_vs_zero | beats | beats | does not beat |
| mean_delta | delta/rmse | beats | beats | does not beat |
| mean_delta | delta/mae | beats | does not beat | does not beat |
| mean_delta | delta/ev | beats | does not beat | does not beat |
| mean_delta | delta/corr | beats | beats | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | does not beat |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | beats |
| class_prior | direction/acc | beats | does not beat | beats |
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

- n_trades: 294
- total_return: -0.5433
- sharpe_net: -145.1292
- sharpe_gross: -4.4257
- sortino: -161.6294
- max_drawdown: 0.5440
- hit_rate: 0.0714
- profit_factor: 0.0256
- avg_hold_bars: 10.4286
- exposure: 0.4239
- turnover: 407.5604
- fees_paid: 4075.6950
- costs_paid: 5298.4035
- gross_pnl: -134.8139
- net_pnl: -5433.2174
