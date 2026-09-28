# Evaluation report - test split - run `20260924T131735Z-609d19e-dirty-af67ee43`

n = 7236 samples; direction metrics exclude moves within 5 bps (neutral mask); n_eff counts non-overlapping outcomes.

| metric | h0 | h1 | h2 |
|---|---|---|---|
| n_eff | 723 | 482 | 361 |
| direction MCC | 0.0289 | -0.0165 | 0.0057 |
| direction AUC | 0.5152 | 0.4966 | 0.4979 |
| direction ECE | 0.0138 | 0.0472 | 0.0259 |
| Gaussian MCC | 0.0382 | 0.0040 | -0.0062 |
| Gaussian AUC | 0.5177 | 0.4991 | 0.4987 |
| delta EV | 0.0024 | 0.0005 | -0.0074 |
| delta corr | 0.0530 | 0.0303 | -0.0185 |
| skill vs zero | 0.0017 | 0.0003 | -0.0088 |
| CRPS ($) | 105.6368 | 127.4940 | 145.7098 |
| PIT KS | 0.0459 | 0.0422 | 0.0476 |
| var/err2 Spearman | 0.2407 | 0.2224 | 0.2238 |
| coverage 90% | 0.9041 | 0.9081 | 0.9136 |

## Confidence gap (accuracy of the more confident half minus the less confident half)

| horizon | gap | 95% CI | verdict |
|---|---|---|---|
| h0 | 0.0074 | [-0.0291, 0.0490] | NOISE |
| h1 | 0.0244 | [-0.0078, 0.0626] | NOISE |
| h2 | -0.0284 | [-0.0722, 0.0220] | NOISE |

## Coherence across horizons

- mag_order_full: 0.0721
- unanimity: 0.2790
- delta_dir_align_all: 0.2275
- coherence_primary: 0.6353

## Against baselines (fit on the training block)

| baseline | metric | h0 | h1 | h2 |
|---|---|---|---|---|
| zero_delta | delta/rmse | beats | beats | does not beat |
| zero_delta | delta/mae | beats | beats | does not beat |
| zero_delta | delta/ev | beats | beats | does not beat |
| zero_delta | delta/corr | beats | beats | does not beat |
| zero_delta | delta/skill_vs_zero | beats | beats | does not beat |
| mean_delta | delta/rmse | beats | beats | does not beat |
| mean_delta | delta/mae | beats | beats | does not beat |
| mean_delta | delta/ev | beats | beats | does not beat |
| mean_delta | delta/corr | beats | beats | does not beat |
| mean_delta | delta/skill_vs_zero | beats | beats | does not beat |
| class_prior | direction/mcc | beats | does not beat | beats |
| class_prior | direction/auc | beats | does not beat | does not beat |
| class_prior | direction/brier | does not beat | does not beat | does not beat |
| class_prior | direction/ece_pos | beats | does not beat | does not beat |
| class_prior | direction/acc | beats | beats | beats |
| class_prior | direction/bal_acc | beats | does not beat | beats |
| const_var | variance/crps | beats | beats | beats |
| const_var | variance/nll | beats | beats | beats |
| const_var | variance/pit_ks | beats | beats | beats |
| const_var | variance/corr_var_err2_spearman | beats | beats | beats |
| logreg_lags | direction/mcc | does not beat | does not beat | does not beat |
| logreg_lags | direction/auc | does not beat | does not beat | does not beat |
| logreg_lags | direction/brier | does not beat | does not beat | does not beat |
| logreg_lags | direction/ece_pos | beats | does not beat | does not beat |
| logreg_lags | direction/acc | beats | does not beat | does not beat |
| logreg_lags | direction/bal_acc | does not beat | does not beat | does not beat |
