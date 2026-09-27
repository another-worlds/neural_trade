# Roadmap

Milestones toward [VISION.md](VISION.md) "The MVP", foundations first (D-021). Items and their
acceptance criteria are in [BACKLOG.md](BACKLOG.md); the lead updates a milestone's status when it
moves (`/handoff`). A milestone that ends in a clear negative result is **closed with that result**,
recorded, and the owner decides what follows; it is not reopened by tuning. Names: MVP-1 to MVP-6
here; M1 to M4 are the remediation gates in [runs/gates/REPORT.md](../runs/gates/REPORT.md).

## Done: the remediation (2026-09-22 to 2026-09-25)

The model trains, the package installs, the measurement stack can be trusted
([plan](archive/REMEDIATION_PLAN_2026-09.md)).

- Gates M1 (a gradient step), M2 (every loss term bounded), M4 (conformal coverage 0.90-0.91 at a 0.90
  target): **pass**. M3 (direction skill, price-head EV): **not met**; it continues in R2.
- Package, typed config, nine strict registries, serving, evaluation protocol with baselines, honest
  backtest engine, ablation harness, notebooks executed on real runs, the served-epoch fix (D-011),
  noise-aware statistics (D-012): **done**. Physics ablation v1 (84 runs): no term reaches VALUE,
  family inconclusive ([report](../runs/ablations/ablate_physics_v1-full/report.md)).

## R1: a trustworthy pipeline (in progress, first)

- Items (nothing downstream counts while CI is red or a baseline is wrong): NT-001 (CI green),
  NT-002 (random null ignores position size), NT-010 (every cited number links to a tracked run),
  NT-025 (5 MB notebook limit enforced); NT-011 done.
- **Exit:** the `ci` workflow is green on the branch head; no open P0; the numbers in STATUS and
  README each link to a committed report or run summary.

## MVP-1: structure and the experiment engine (VISION MVP point 1; D-023, D-029)

- Items: NT-026 (experiment engine: scenario and sweep spec, resumable runner, one run store with an
  index, one scorer; supersedes NT-024), NT-027 (layering: no circular imports, one metrics and
  statistics module, figures only draw), NT-028 (stale removal under D-029), NT-029 (config metadata
  for the panel and search spaces, generated config reference).
- **Exit:** NT-026 to NT-029 done; VISION point 1 holds on the merged head, with notebooks 00-05
  executed and checked on it (D-013, D-028).

## MVP-2: control panel and model comparison (point 2, "The yardstick"; D-020, D-023 to D-025)

- Items: NT-030 (sweeps: quick and Optuna, `neural-trade sweep`), NT-031 (leaderboard by dev-fold net
  Sharpe after costs), NT-032 (paired comparator), NT-033 (manual-search baselines: frozen twin and
  classic TA rules), NT-034 (control-panel notebook 06), NT-035 (GPU measurements: concurrent-runs
  throughput, deterministic-mode speed; experimenter).
- **Exit:** NT-030 to NT-035 done; VISION point 2 holds: a real sweep on the reference setup, with both
  manual-search baselines, appears on the leaderboard in notebook 06 and from the CLI.

## MVP-3: gradient stability (point 3; D-026)

- Items: NT-036 (stability invariants in CI, strict mode), NT-037 (per-run gradient health, at most 2%
  of `sec_per_step`; absorbs NT-012), NT-038 (stability harness and config guard), NT-039
  (pre-registered A/B: gradient-based loss weighting against today's value calibration; experimenter).
- **Exit:** NT-036 to NT-038 done, and the harness passes its pre-registered thresholds on the
  reference setup; NT-039 closed with the paired comparator's verdict (a negative verdict closes it).

## MVP-4: generality (point 4; D-022)

- Items: NT-040 (annualisation uses the bar size; P0 before any other bar size runs), NT-041 (dataset
  spec, wall-clock window and horizons, 7-day training block, walk-forward folds over the long
  history), NT-042 (variable number of horizons).
- **Exit:** NT-040 to NT-042 done (N=3 reproduces today's numbers in the golden run; the harness passes
  for N=2 and N=4); VISION point 4 holds. Only the reference setup is tested (VISION "Not in the MVP").

## MVP-5: visual comprehension (point 5; D-014, D-027, D-028)

- Items: NT-043 (learned indicators on price against the textbook defaults, notebook 07; may start in
  parallel with MVP-1), NT-044 (guides for the owner and reviewers, README landing page,
  ARCHITECTURE), NT-045 (notebook overlap: each figure has one home).
- **Exit:** NT-043 to NT-045 done; VISION point 5 holds: every notebook executed on real runs, checked
  and rendered, and every changed figure looked at (D-013, D-014).

## MVP-6: indicator catalogue (VISION "Also in the MVP"; D-027)

- Items: NT-046 (indicators package and registry): a placeholder, not picked until the owner's
  indicator Q&A (held right after the 2026-09-28 overhaul) sets its scope. **Exit:** its acceptance.

## Research tracks R2-R4: scenarios run through the engine

The former research milestones, now engine scenarios (D-021): pre-registered A/B studies within the
research budget (OPERATING_MODEL), choices on dev folds, one verdict by the paired comparator (D-025).
R2 and R3 depend on NT-026, NT-031 and NT-032; R4 on NT-026 and NT-032.

- **R2 direction and price heads:** NT-003 (the M3 direction clauses), NT-004 (served price EV > 0).
  **Exit:** each item's acceptance: a pass, or its complete negative report.
- **R3 edge after costs:** NT-005 (cost-aware trading; also needs NT-002 and the owner's NT-007).
  **Exit:** NT-005's acceptance (a), or its negative-result report (b).
- **R4 physics terms:** NT-006 (ablation re-run on the current trainer, pre-registered again under
  D-025; needs the owner's GPU approval, asked 2026-09-25). **Exit:** a verdict per term; a term that
  does not help is proposed to the owner for removal (D-003). The v1 grid stays the record under its
  own criteria.

## R5: merge and publish

- Items: NT-008 (owner: merge `remediation/plan` into `master`, after NT-001), NT-017 (licence; P3,
  not needed for the MVP audience, D-019).
- **Exit:** the owner has merged, and the nightly workflow runs from `master`.

## Continuous: evaluation depth and polish

Outside the milestones, taken when no milestone item of the same priority is actionable: NT-013,
NT-014 (evaluation report), NT-015 (interval toolkit; after NT-027), NT-016 (backtest data), NT-018
(notebook UX), NT-019 to NT-023 (figure polish, batched per module).

## Order

Pick rule: OPERATING_MODEL "Picking the next item" (priority, then the earliest open milestone here).

1. R1; then MVP-1 to MVP-5 in order. NT-043 may run in parallel with MVP-1 (a second implementer,
   disjoint files).
2. MVP-6 after MVP-1; the indicator Q&A fixes its exact position.
3. R2-R4 after MVP-2 (they need the engine, the leaderboard and the comparator): experimenter GPU work
   beside the implementer's CPU items, never two GPU jobs at once.
4. R5 whenever the owner is ready, after R1.

Priority comes first, so a milestone's P2 items (NT-025 in R1; NT-027, NT-028 in MVP-1) may be picked
after later milestones' P1 items; a milestone closes only when all its items are done. The move to D:
(D-030) ends the 2026-09-28 session; NT-009 closes when the move is verified. The MVP is done when
VISION "The MVP" holds on the reference setup; MVP-1 to MVP-6 map to its points.
