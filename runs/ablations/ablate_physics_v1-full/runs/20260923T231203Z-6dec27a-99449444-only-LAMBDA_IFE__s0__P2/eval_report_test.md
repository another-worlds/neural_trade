# Evaluation report - test split - run `20260923T231203Z-6dec27a-99449444-only-LAMBDA_IFE__s0__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0118 | 0.0024 | 0.0196 |
| direction AUC | 0.5139 | 0.5049 | 0.5043 |
| direction ECE | 0.0290 | 0.0189 | 0.0260 |
| Gaussian MCC | -0.0249 | -0.0161 | -0.0054 |
| Gaussian AUC | 0.4991 | 0.4917 | 0.4966 |
| delta EV | -0.0045 | -0.0037 | -0.0060 |
| delta corr | -0.0070 | -0.0245 | -0.0195 |
| skill vs zero | -0.0047 | -0.0036 | -0.0060 |
| CRPS ($) | 106.5786 | 129.1798 | 146.8459 |
| PIT KS | 0.0619 | 0.0610 | 0.0569 |
| var/err2 Spearman | 0.2678 | 0.2455 | 0.2317 |
| coverage 90% | 0.9028 | 0.9073 | 0.9149 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0310 | [-0.0136, 0.0717] | NOISE |
| h1 | 0.0126 | [-0.0310, 0.0498] | NOISE |
| h2 | -0.0177 | [-0.0549, 0.0185] | NOISE |

## Coherence across horizons

- mag_order_full: 0.2709
- unanimity: 0.4024
- delta_dir_align_all: 0.2816
- coherence_primary: 0.5391

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
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | beats | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | does not beat | does not beat | does not beat |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | beats | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |

## Backtest (costs included)

- n_trades: 214
- total_return: -0.4312
- sharpe_net: -115.2474
- sharpe_gross: -2.1736
- sortino: -131.0496
- max_drawdown: 0.4323
- hit_rate: 0.0701
- profit_factor: 0.0268
- avg_hold_bars: 8.2383
- exposure: 0.2436
- turnover: 326.6590
- fees_paid: 3266.7911
- costs_paid: 4246.8284
- gross_pnl: -65.4354
- net_pnl: -4312.2638
