# Tactical session journal (D-062)

Rules: [docs/TACTICAL.md](../../docs/TACTICAL.md). Every row is tactical, exploratory: one seed, dev folds
only, with its noise level. Earlier work this continues from: `runs/experiments/micro_loop_v1/LOG.md`.

## Handoff

_Rewritten at the end of every tactical session._ 2026-10-06: round 1 (3 direction-head switches) is closed with no effect. GPU hours used today: 0. Next: the owner names the goal, or the session starts from where the micro loop
stopped (H4b: re-run the LOOKBACK 240 cell with more patience; direction is the bottleneck at every horizon).

## Hypotheses

| # | Date | Hypothesis | Method | Cost | Result | Evidence |
|---|---|---|---|---|---|---|
| H5 | 2026-10-06 | R1: the direction head is spoiled by the high-capacity path; constrain it | hc_baseline (defaults, 100 trials) vs 3 switches (cc85a5c), same 5 slices x 20 seeds, screen layout, paired by (slice, seed), CI over slices | ~300 trials, ~27 s each | **No effect.** skip_only -0.012 [-0.101, +0.076]; shrink1 -0.010 [-0.037, +0.017] (83 of 100 trials: shard 2 segfaulted 3 times at 16/34; accepted as final by the owner); dropout 0.5 -0.015 [-0.029, -0.0001]. None beats the baseline (0.522); logreg_lags 0.554 is within noise of it. | runs/tactical/screens/hc_*, hc_logreg.json, configs/tactical/ |

## For the MVP lead

Findings worth adopting, with evidence; each becomes an MVP backlog item and a paired test (D-025) before
any default changes.

- Measurement, not a model finding: one slice's direction AUC has an error of about 0.06-0.12 (n_eff 20-40), the 5-slice mean about 0.03; seed sd 0.04-0.05 inside a slice. A 0.02-0.03 effect needs far more independent slices (30-50) or longer validation blocks. logreg_lags on the same blocks is 0.554 (0.42-0.67 per slice): not evidence of skill.
