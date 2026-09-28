# Evaluation report - test split - run `20260924T043447Z-b5d9b70-dirty-af67ee43`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0110 | 0.0088 | 0.0132 |
| direction AUC | 0.5024 | 0.5172 | 0.5154 |
| direction ECE | 0.0511 | 0.0507 | 0.0287 |
| Gaussian MCC | -0.0053 | 0.0000 | 0.0000 |
| Gaussian AUC | 0.4957 | 0.5000 | 0.5000 |
| delta EV | -0.0184 | 0.0000 | 0.0000 |
| delta corr | -0.0440 | 0.0000 | 0.0000 |
| skill vs zero | -0.0202 | 0.0000 | 0.0000 |
| CRPS ($) | 107.3995 | 128.6199 | 146.2984 |
| PIT KS | 0.0579 | 0.0577 | 0.0520 |
| var/err2 Spearman | 0.2086 | 0.1672 | 0.1891 |
| coverage 90% | 0.9037 | 0.9059 | 0.9124 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | -0.0015 | [-0.0503, 0.0426] | NOISE |
| h1 | 0.0303 | [-0.0205, 0.0788] | NOISE |
| h2 | 0.0255 | [-0.0136, 0.0626] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0000
- unanimity: 0.2244
- delta_dir_align_all: 0.1260
- coherence_primary: 0.6989

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | does not beat |
| zero_delta | delta/corr | does not beat | does not beat | does not beat |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | does not beat | beats | beats |
| mean_delta | delta/mae | does not beat | beats | beats |
| mean_delta | delta/ev | does not beat | does not beat | does not beat |
| mean_delta | delta/corr | does not beat | does not beat | does not beat |
| mean_delta | delta/skill_vs_zero | does not beat | beats | beats |
| class_prior | direction/mcc | beats | beats | beats |
| class_prior | direction/auc | beats | beats | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | beats | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | beats | beats | beats |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | does not beat | does not beat | does not beat |
| logreg_lags | direction/acc | does not beat | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |
