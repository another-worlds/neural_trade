# T06: an ensemble of the direction models with the best AUC

Lead, 2026-10-13. Committed BEFORE any result.

**Owner request.** Looking at deck slide 5 (the left panel): "вижу вот слева хорошие AUC. Составь из них ансамбль и проверь его точность".

## Members: the left panel of slide 5
Every member scores the same validation blocks.

| Group | Members | Source |
|---|---|---|
| Regressions | sklearn logistic tb7 (L0) | the lab, `runs/tactical/rebuild/` ladder, 11.5-day blocks |
| Lab networks | rich-43 linear (L5); + residual MLP (L6); + GRU on the window (L7) | the same ladder |
| New networks (H36 / H44) | the regression on 1 y; PatchTST-lite (1 y); the PatchTST 3-seed ensemble; PatchTST + unfrozen regression 3-seed ensemble | `runs/tactical/newnet/`, 1-3 y training |

- Members whose saved validation predictions exist are used as saved.
- A member without them is re-run with its frozen code and configuration, on CPU or within about 1 GPU-hour in total. The GPU is shared with the outlook session, at most 2 heavy processes.
- A member that cannot be reproduced is dropped and listed.

## Ensembles (fixed here)
- **E1 (primary):** the equal-weight mean of the members' P(up), per horizon.
- **E2:** the equal-weight mean of the members' logits.
- **E3 (stacking):** a logistic regression on the members' logits. It is fitted ONLY on the last 15% of each slice's TRAINING block (the inner split the new network used), never on the validation block.

## Where it is judged
- **Main:** the 24 lab slices (the blocks behind slide 5), paired by slice.
- **Confirmation:** the 6 held-out FINAL slices (`cache.py --final`). They were looked at once before, by H44.
- **Honest present:** the monthly walk-forward 2024-07..2025-07-20 (next-open labels, as H59), for the members that can be run in walk-forward on CPU (L0, L5, L6) and their E1. This shows what the ensemble is worth in today's market.

## Metrics
- mean-of-3 direction AUC;
- log loss against the constant;
- the top-10% hit rate and gross bps, with the threshold from the training block only;
- pairwise correlation of the members' logits.

## Pre-registered criterion
**Success** iff E1 beats the best single member, the sklearn regression L0, by **>= +0.005 mean-of-3 AUC with the 95% CI over the 24 slices above 0**, AND the confirmation on the 6 FINAL slices has the same sign. The 0.005 is the new network's criterion 1 (PLAN 10). E2 and E3 are reported; they are not the verdict.

## Never
- No bar after 2025-07-20.
- No choice on validation, FINAL or walk-forward data.
- The members' configurations are frozen as logged.

## Outputs
`runs/tactical/tasks/T06_direction_ensemble/`: code, `results.json`, `tables.txt`, `tests.log`, `run.log`. The report text goes to the lead. A slide or report section follows.
