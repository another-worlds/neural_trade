# Evaluation report - test split - run `20260924T114819Z-e23fd9f-dirty-af67ee43`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | -0.0098 | -0.0130 | -0.0055 |
| direction AUC | 0.4988 | 0.4997 | 0.5073 |
| direction ECE | 0.0276 | 0.0379 | 0.0240 |
| Gaussian MCC | 0.0000 | 0.0000 | 0.0478 |
| Gaussian AUC | 0.5000 | 0.5000 | 0.5187 |
| delta EV | 0.0000 | 0.0000 | 0.0004 |
| delta corr | 0.0000 | 0.0000 | 0.0192 |
| skill vs zero | 0.0000 | 0.0000 | -0.0001 |
| CRPS ($) | 105.8291 | 127.5036 | 145.1798 |
| PIT KS | 0.0392 | 0.0384 | 0.0438 |
| var/err2 Spearman | 0.2414 | 0.2391 | 0.2315 |
| coverage 90% | 0.9027 | 0.9059 | 0.9122 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0063 | [-0.0328, 0.0395] | NOISE |
| h1 | 0.0152 | [-0.0252, 0.0567] | NOISE |
| h2 | 0.0071 | [-0.0391, 0.0574] | NOISE |

## Coherence across horizons

- mag_order_full: 1.0000
- unanimity: 0.2872
- delta_dir_align_all: 0.1617
- coherence_primary: 0.6064

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | does not beat | does not beat | does not beat |
| zero_delta | delta/mae | does not beat | does not beat | does not beat |
| zero_delta | delta/ev | does not beat | does not beat | beats |
| zero_delta | delta/corr | does not beat | does not beat | beats |
| zero_delta | delta/skill_vs_zero | does not beat | does not beat | does not beat |
| mean_delta | delta/rmse | beats | beats | beats |
| mean_delta | delta/mae | beats | beats | beats |
| mean_delta | delta/ev | does not beat | does not beat | beats |
| mean_delta | delta/corr | beats | does not beat | beats |
| mean_delta | delta/skill_vs_zero | beats | beats | beats |
| class_prior | direction/mcc | does not beat | does not beat | does not beat |
| class_prior | direction/auc | does not beat | does not beat | beats |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | does not beat | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | does not beat | does not beat | does not beat |
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
