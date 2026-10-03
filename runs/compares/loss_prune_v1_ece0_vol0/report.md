# loss_prune_v1_ece0_vol0_vs_control

Non-inferiority: does also turning off the volatility penalty (A_losses.md section 8: minimised at std(mu) = std(y), which opposes every proper score at R^2 ~ 0 and pushes the price head toward noise-level spread) on top of soft ECE off cost more than 0.005 of h1 variance CRPSS against today's defaults, on the same 5 judgement folds?


**A** = `loss_prune_v1`, **B** = `loss_prune_v1` · metric `h1/variance/crpss` (higher_better, diff) · minimum effect 0.005 · judgement folds [-39, -38, -37, -36, -35] (min_folds 5) · alpha 0.05

spec hash `eb47e789da74` · registered 2026-10-01T00:00:00Z (effective 2026-10-01T03:55:40+05:00, source git_commit_time)

**Verdict: inconclusive** (mean: 0.002661, 95% CI [-0.002927, 0.008249], n = 5 folds, 5 pairs)

| fold | seeds | mean diff |
|---|---|---|
| -39 | [0] | 0.002256 |
| -38 | [0] | -0.001345 |
| -37 | [0] | -0.0007315 |
| -36 | [0] | 0.009942 |
| -35 | [0] | 0.003184 |

| seed | fold | A | B | diff |
|---|---|---|---|---|
| 0 | -39 | 0.01048 | 0.008227 | 0.002256 |
| 0 | -38 | 0.01943 | 0.02078 | -0.001345 |
| 0 | -37 | 0.03889 | 0.03962 | -0.0007315 |
| 0 | -36 | 0.02566 | 0.01572 | 0.009942 |
| 0 | -35 | 0.0417 | 0.03851 | 0.003184 |

## Guard-rails

| metric | verdict | estimate | CI |
|---|---|---|---|
| h0/variance/crpss | pass | 0.0006548 | [-0.004193, 0.005503] |
| h2/variance/crpss | pass | 0.002798 | [-0.002951, 0.008547] |

Non-inferiority (margin 0.005): **pass**