# capacity_v1_linear_indicators_vs_control

Does the linear_indicators arm differ from today's gru_attention in h1 direction AUC (primary), on 5 judgement folds no earlier choice used? Non-inferiority margin and minimum effect are both 0.01 AUC.


**A** = `capacity_v1`, **B** = `capacity_v1` · metric `h1/direction/auc` (higher_better, diff) · minimum effect 0.01 · judgement folds [-96, -95, -94, -93, -92] (min_folds 5) · alpha 0.05

spec hash `a0602bf66c3e` · registered 2026-10-06T00:00:00Z (effective 2026-10-06T12:50:57+05:00, source git_commit_time)

**Verdict: inconclusive** (mean: -0.02618, 95% CI [-0.05507, 0.002705], n = 5 folds, 5 pairs)

| fold | seeds | mean diff |
|---|---|---|
| -96 | [0] | -0.02647 |
| -95 | [0] | -0.0621 |
| -94 | [0] | -0.03024 |
| -93 | [0] | -0.01098 |
| -92 | [0] | -0.001128 |

| seed | fold | A | B | diff |
|---|---|---|---|---|
| 0 | -96 | 0.4829 | 0.5094 | -0.02647 |
| 0 | -95 | 0.4968 | 0.5588 | -0.0621 |
| 0 | -94 | 0.527 | 0.5572 | -0.03024 |
| 0 | -93 | 0.5054 | 0.5164 | -0.01098 |
| 0 | -92 | 0.5307 | 0.5318 | -0.001128 |

## Guard-rails

| metric | verdict | estimate | CI |
|---|---|---|---|
| h1/variance/crpss | undecided | -0.004935 | [-0.0186, 0.008731] |
| h0/variance/crpss | undecided | -0.01053 | [-0.02426, 0.003205] |
| h2/variance/crpss | undecided | -0.01046 | [-0.02492, 0.003997] |
| h0/direction/auc | pass | 0.03196 | [-0.00387, 0.06779] |
| h2/direction/auc | undecided | -0.006894 | [-0.02756, 0.01377] |
| h1/direction/brier | undecided | -0.005381 | [-0.01608, 0.005323] |

Non-inferiority (margin 0.01): **undecided**