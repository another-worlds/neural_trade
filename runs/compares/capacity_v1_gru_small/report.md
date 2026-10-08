# capacity_v1_gru_small_vs_control

Does the gru_small arm differ from today's gru_attention in h1 direction AUC (primary), on 5 judgement folds no earlier choice used? Non-inferiority margin and minimum effect are both 0.01 AUC.


**A** = `capacity_v1`, **B** = `capacity_v1` · metric `h1/direction/auc` (higher_better, diff) · minimum effect 0.01 · judgement folds [-96, -95, -94, -93, -92] (min_folds 5) · alpha 0.05

spec hash `a5b33e917fdb` · registered 2026-10-06T00:00:00Z (effective 2026-10-06T12:50:57+05:00, source git_commit_time)

**Verdict: inconclusive** (mean: -0.01214, 95% CI [-0.02415, -0.0001314], n = 5 folds, 5 pairs)

| fold | seeds | mean diff |
|---|---|---|
| -96 | [0] | -0.006446 |
| -95 | [0] | -0.02105 |
| -94 | [0] | -0.008524 |
| -93 | [0] | -0.001177 |
| -92 | [0] | -0.02349 |

| seed | fold | A | B | diff |
|---|---|---|---|---|
| 0 | -96 | 0.503 | 0.5094 | -0.006446 |
| 0 | -95 | 0.5378 | 0.5588 | -0.02105 |
| 0 | -94 | 0.5487 | 0.5572 | -0.008524 |
| 0 | -93 | 0.5152 | 0.5164 | -0.001177 |
| 0 | -92 | 0.5083 | 0.5318 | -0.02349 |

## Guard-rails

| metric | verdict | estimate | CI |
|---|---|---|---|
| h1/variance/crpss | undecided | 0.001121 | [-0.008809, 0.01105] |
| h0/variance/crpss | undecided | 0.0007617 | [-0.01667, 0.01819] |
| h2/variance/crpss | undecided | 7.375e-05 | [-0.01386, 0.01401] |
| h0/direction/auc | pass | 0.01622 | [-0.007835, 0.04027] |
| h2/direction/auc | pass | 0.003374 | [-0.008602, 0.01535] |
| h1/direction/brier | undecided | -0.006063 | [-0.01738, 0.005257] |

Non-inferiority (margin 0.01): **undecided**