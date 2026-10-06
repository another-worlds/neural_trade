# loss_prune_v1_ece0_vs_control

Non-inferiority: does turning off the soft-ECE calibration term (A_losses.md section 7: an improper L1 kink that does not vanish at calibration and dominated the gradient direction in NT-037's CPU probe) cost more than 0.005 of h1 variance CRPSS against today's defaults, on 5 judgement folds no earlier choice used?


**A** = `loss_prune_v1`, **B** = `loss_prune_v1` · metric `h1/variance/crpss` (higher_better, diff) · minimum effect 0.005 · judgement folds [-39, -38, -37, -36, -35] (min_folds 5) · alpha 0.05

spec hash `283274d4318c` · registered 2026-10-01T00:00:00Z (effective 2026-10-01T03:55:40+05:00, source git_commit_time)

**Verdict: inconclusive** (mean: 0.002453, 95% CI [-0.001037, 0.005942], n = 5 folds, 5 pairs)

| fold | seeds | mean diff |
|---|---|---|
| -39 | [0] | 0.003366 |
| -38 | [0] | -0.002446 |
| -37 | [0] | 0.003323 |
| -36 | [0] | 0.003248 |
| -35 | [0] | 0.004772 |

| seed | fold | A | B | diff |
|---|---|---|---|---|
| 0 | -39 | 0.01159 | 0.008227 | 0.003366 |
| 0 | -38 | 0.01833 | 0.02078 | -0.002446 |
| 0 | -37 | 0.04294 | 0.03962 | 0.003323 |
| 0 | -36 | 0.01897 | 0.01572 | 0.003248 |
| 0 | -35 | 0.04328 | 0.03851 | 0.004772 |

## Guard-rails

| metric | verdict | estimate | CI |
|---|---|---|---|
| h0/variance/crpss | pass | 0.00156 | [-0.001305, 0.004425] |
| h2/variance/crpss | pass | 0.002754 | [-0.002864, 0.008372] |

Non-inferiority (margin 0.005): **pass**