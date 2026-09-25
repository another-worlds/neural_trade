# Roadmap

Milestones toward the end goal in [VISION.md](VISION.md). Each milestone has exit criteria that can
be checked; its work items are in [BACKLOG.md](BACKLOG.md). The lead updates a milestone's status
when it moves (see the `/handoff` skill). A milestone that ends in a clear negative result is
**closed with that result**, recorded, and the owner decides what follows. It is not reopened by
tuning.

## Done: the remediation (2026-09-22 to 2026-09-25)

The model trains, the package is installable, and the measurement stack can be trusted. Plan:
[archive/REMEDIATION_PLAN_2026-09.md](archive/REMEDIATION_PLAN_2026-09.md); evidence:
[runs/gates/REPORT.md](../runs/gates/REPORT.md).

- M1 (a gradient step is possible), M2 (every loss term bounded), M4 (metrics and calibration
  trusted: conformal coverage 0.90-0.91 at a 0.90 target): **pass**.
- Phases B (package, typed config, nine strict registries, serving) and C (evaluation protocol
  with baselines, run tracking, honest backtest engine, ablation harness): **done**.
- Full physics-term ablation (84 runs): **done**, verdicts inconclusive
  ([report](../runs/ablations/ablate_physics_v1-full/report.md)).
- Rich notebooks executed on real runs, the served-epoch fix, noise-aware statistics: **done**.
- **Not met:** M3 (direction skill, price-head EV), and "CI green".

## R1: a pipeline that stays trustworthy (in progress)

Nothing downstream means anything if CI is red or a baseline is wrong.

- Items: NT-001 (CI green), NT-002 (random null ignores position size), NT-010 (every cited number
  links to a tracked run), NT-011 (done).
- **Exit:** the `ci` workflow is green on the branch head; no open P0; the numbers in STATUS and
  README each link to a committed report or run summary.

## R2: direction skill and price heads (M3)

The central research question. Can the network beat the trailing-returns logistic regression
(AUC about 0.52-0.56 on fold -1)?

- Items: NT-003 (direction: M3 direction clauses), NT-004 (price heads: served EV > 0).
- **Exit (pass):** on fold -1, averaged over at least 3 seeds with the current trainer, test AUC
  h1 > 0.52 on all rows and at least the `logreg_lags` baseline, best-epoch val MCC h1 > 0.02,
  Gaussian-readout MCC > 0, and served EV(delta) h1 > 0 with beta_h1 > 0.
- **Exit (negative):** each item's pre-registered variants are exhausted without passing. The
  report says so, and the owner decides the next direction (see VISION: a negative answer is
  valid).

## R3: an edge that survives costs

- Items: NT-005 (cost-aware trading). Depends on NT-002 and the owner's answer on NT-007.
- **Exit (pass):** a strategy whose net return on fold -1's test block, over at least 3 seeds,
  beats buy-and-hold and ranks above the 95th percentile of the size-matched random null after
  26 bps round-trip costs; checked for look-ahead.
- **Exit (negative):** the edge per trade stays below the cost for every pre-registered variant.

## R4: the physics terms, judged on the current trainer

- Items: NT-006 (re-run the ablation grid after the served-epoch fix). Needs the owner's approval
  for about 7-10 GPU hours (84 runs at 5-7 min each), and disk (NT-009).
- **Exit:** a regenerated `report.md` with a verdict per term under the unchanged pre-registered
  criteria. A term that is HARMFUL or NEUTRAL in both modes is proposed to the owner for removal
  (D-003 says the terms stay until the evidence says they cannot help).

## R5: merge and publish

- Items: NT-008 (merge `remediation/plan` into `master`; after NT-001), NT-017 (licence).
- **Exit:** the owner has merged, and the nightly workflow runs from `master`.

## Continuous: tooling, evaluation depth, polish

Taken between milestone items, highest priority first: NT-012 (training logging), NT-013 and
NT-014 (evaluation report), NT-015 (interval toolkit), NT-016 (backtest data), NT-018 (notebook
UX), NT-019 to NT-023 (figure polish, batched per module).

## Order

R1 first. Then R2 (GPU, experimenter) with the continuous items in parallel (CPU, implementer).
R4 when the owner approves the GPU time, never at the same time as another GPU job. R3 after
R2 has a verdict, unless the owner decides otherwise. R5 whenever the owner is ready, after R1.
