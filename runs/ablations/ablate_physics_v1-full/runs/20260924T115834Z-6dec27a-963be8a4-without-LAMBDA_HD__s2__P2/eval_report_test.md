# Evaluation report - test split - run `20260924T115834Z-6dec27a-963be8a4-without-LAMBDA_HD__s2__P2`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0267 | 0.0033 | -0.0483 |
| direction AUC | 0.4646 | 0.4978 | 0.4786 |
| direction ECE | 0.0272 | 0.0183 | 0.0651 |
| Gaussian MCC | -0.0434 | -0.0132 | 0.0009 |
| Gaussian AUC | 0.4722 | 0.4798 | 0.4839 |
| delta EV | -0.0138 | -0.0126 | -0.0137 |
| delta corr | -0.0588 | -0.0478 | -0.0467 |
| skill vs zero | -0.0168 | -0.0208 | -0.0240 |
| CRPS ($) | 106.1886 | 127.9525 | 146.4333 |
| PIT KS | 0.0306 | 0.0446 | 0.0498 |
| var/err2 Spearman | 0.2681 | 0.2548 | 0.2407 |
| coverage 90% | 0.9024 | 0.9074 | 0.9110 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0695 | [-0.1224, -0.0116] | INVERTED |
| h1 | -0.0173 | [-0.0650, 0.0309] | NOISE |
| h2 | 0.0181 | [-0.0373, 0.0874] | NOISE |

## Coherence across horizons

- mag_order_full: 0.5887
- unanimity: 0.2611
- delta_dir_align_all: 0.2428
- coherence_primary: 0.5761

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
| class_prior | direction/mcc | does not beat | beats | does not beat |
| class_prior | direction/auc | does not beat | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | beats | does not beat |
| class_prior | direction/acc | does not beat | beats | does not beat |
| class_prior | direction/bal_acc | does not beat | beats | does not beat |
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

- n_trades: 265
- total_return: -0.5057
- sharpe_net: -136.8896
- sharpe_gross: -3.5600
- sortino: -153.4785
- max_drawdown: 0.5070
- hit_rate: 0.0642
- profit_factor: 0.0364
- avg_hold_bars: 12.8075
- exposure: 0.4692
- turnover: 380.9483
- fees_paid: 3809.6352
- costs_paid: 4952.5258
- gross_pnl: -104.4422
- net_pnl: -5056.9679
