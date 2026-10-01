# loss_prune_v1_ece0_vol0_vs_control

Non-inferiority: does also turning off the volatility penalty (A_losses.md section 8: minimised at std(mu) = std(y), which opposes every proper score at R^2 ~ 0 and pushes the price head toward noise-level spread) on top of soft ECE off cost more than 0.005 of h1 variance CRPSS against today's defaults, on the same 5 judgement folds?


**A** = `loss_prune_v1`, **B** = `loss_prune_v1` · metric `h1/variance/crpss` (higher_better, diff) · minimum effect 0.005 · judgement folds [-39, -38, -37, -36, -35] (min_folds 5) · alpha 0.05

spec hash `eb47e789da74` · registered 2026-10-01T00:00:00Z (effective 2026-10-01T03:55:40+05:00, source git_commit_time)

**Refused: only 4 judgement fold(s) had a usable pair ([-39, -38, -37, -36]), need >= 5 (D-046)**

1 pair(s) excluded:

- seed 0, fold -35: side A: no matching (seed, fold) run