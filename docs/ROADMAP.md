# Roadmap

Milestones toward [VISION.md](VISION.md) "The MVP", foundations first (D-021). Items and their
acceptance criteria are in [BACKLOG.md](BACKLOG.md); the lead updates a milestone's status when it
moves (`/handoff`). A milestone that ends in a clear negative result is **closed with that result**,
recorded, and the owner decides what follows; it is not reopened by tuning. Names: MVP-1 to MVP-6
here; M1 to M4 are the remediation gates in [runs/gates/REPORT.md](../runs/gates/REPORT.md). The
sections below are in pick order ("Order").

## Done: the remediation (2026-09-22 to 2026-09-25)

Gates M1 (a gradient step), M2 (every loss term bounded) and M4 (conformal coverage 0.90-0.91 at a
0.90 target) pass; M3 (direction skill, price-head EV) is not met and continues in R2. Package, typed
config, nine strict registries, serving, evaluation with baselines, honest backtest, ablation harness,
notebooks on real runs, D-011, D-012: done. Physics ablation v1 (84 runs): no term reaches VALUE,
family inconclusive ([report](../runs/ablations/ablate_physics_v1-full/report.md);
[plan](archive/REMEDIATION_PLAN_2026-09.md)).

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

## MVP-6: indicator catalogue (VISION "Also in the MVP"; D-027, D-031)

- Items (scope: [qa/2026-09-28-indicators.md](qa/2026-09-28-indicators.md)): NT-046 (indicators
  package and registry with today's four families: one entry per family, 3 instances each, the
  per-window shift with an off switch; the golden run shows no number changed), NT-047 (OHLCV input
  and the new families: range / volatility, volume, trend / channels; all learnable, all on by
  default), NT-048 (the discovered-indicators report: a self-contained interactive HTML report per
  run, the same figures in notebook 07).
- **The window (D-032):** the owner committed to removing the fixed input window (research record:
  [research/2026-09-28-window-free/](research/2026-09-28-window-free/README.md)). NT-053 (the lead's
  second research round) writes the path, the gates and the A/B specifications; the owner approves
  the plan before its first implementation item is picked. NT-047 waits for NT-053 (the input path
  and the indicator forms may change). No period ceiling (owner). NT-054 (fixed costs and launches)
  is window-independent and sits in Continuous.
- **Exit:** NT-046 to NT-048 and NT-053 done; VISION "Also in the MVP" holds with the design of D-031.

## MVP-2: control panel and model comparison (point 2, "The yardstick"; D-020, D-023 to D-025)

- Items: NT-030 (sweeps: quick and Optuna, `neural-trade sweep`), NT-031 (leaderboard by dev-fold net
  Sharpe after costs), NT-032 (paired comparator), NT-033 (manual-search baselines: frozen twin and
  classic TA rules), NT-034 (control-panel notebook 06), NT-035 (GPU measurements: concurrent-runs
  throughput, deterministic-mode speed; experimenter), NT-050 (the first real Optuna sweep on the
  reference setup for the learned model, the frozen twin and the TA rules, then the paired verdicts;
  experimenter).
- **Exit:** NT-030 to NT-035 and NT-050 done; VISION point 2 holds: NT-050's sweep appears on the
  leaderboard in notebook 06 and from the CLI, and its verdicts learned-vs-frozen and
  learned-vs-TA-rules are recorded (a negative verdict is a valid outcome).

## MVP-3: gradient stability (point 3; D-026)

- Items: NT-036 (stability invariants in CI, strict mode), NT-037 (per-run gradient health, at most 2%
  of `sec_per_step`; absorbs NT-012), NT-038 (stability harness and config guard), NT-039
  (pre-registered A/B: gradient-based loss weighting against today's value calibration; experimenter),
  NT-051 (the first harness run on the reference setup; experimenter).
- **Exit:** NT-036 to NT-039 and NT-051 done: the harness passes its pre-registered thresholds on the
  reference setup (NT-051); NT-039 has the comparator's verdict (a negative one closes it).

## MVP-4: generality (point 4; D-022)

- Items: NT-040 (annualisation uses the bar size; P0 before any other bar size runs), NT-041 (dataset
  spec, wall-clock window and horizons, 7-day training block, walk-forward folds over the long
  history), NT-042 (variable number of horizons), NT-052 (harness runs for N = 2 and N = 4 horizons;
  experimenter).
- **Exit:** NT-040 to NT-042 and NT-052 done: N = 3 reproduces today's numbers in the golden run
  (NT-042), and the harness passes for N = 2 and N = 4 (NT-052); VISION point 4 holds. Only the
  reference setup is tested (VISION "Not in the MVP").

## MVP-5: visual comprehension (point 5; D-014, D-027, D-028)

- Items: NT-043 (learned indicators on price against the textbook defaults, notebook 07; may run in
  parallel from the start), NT-044 (guides for the owner and reviewers, README landing page,
  ARCHITECTURE), NT-045 (notebook overlap: each figure has one home).
- **Exit:** NT-043 to NT-045 done; VISION point 5 holds: every notebook executed on real runs, checked
  and rendered, and every changed figure looked at (D-013, D-014).

## Research tracks R2-R4: scenarios run through the engine

Pre-registered A/B studies (OPERATING_MODEL "Sweeps and pre-registered studies"), choices on dev folds, one verdict by
the paired comparator (D-021, D-025). R2 and R3 need NT-026, NT-031, NT-032; R4 needs NT-026, NT-032.

- **R2 direction and price heads:** NT-003 (the M3 direction clauses), NT-004 (served price EV > 0).
  **Exit:** each item's acceptance: a pass, or its complete negative report.
- **R3 edge after costs:** NT-005 (cost-aware trading; also needs NT-002 and the owner's NT-007).
  **Exit:** NT-005's acceptance (a), or its negative-result report (b).
- **R4 physics terms:** NT-006 (ablation re-run on the current trainer, pre-registered again under
  D-025; needs the owner's GPU approval, asked 2026-09-25). **Exit:** a verdict per term; a term that
  does not help is proposed to the owner for removal (D-003); the v1 grid stays the record.

## R5: merge and publish

- Items: NT-008 (owner: merge `remediation/plan` into `master`, after NT-001), NT-017 (licence: left
  open by the owner; P3 because the MVP audience does not need one, lead's reading, D-019).
- **Exit:** the owner has merged, and the nightly workflow runs from `master`.

## Continuous: evaluation depth and polish

Outside the milestones, taken when no milestone item of the same priority is actionable: NT-013,
NT-014 (evaluation report), NT-015 (interval toolkit; after NT-027), NT-016 (backtest data), NT-018
(notebook UX), NT-019 to NT-023 (figure polish, batched per module), NT-049 (training silently
warm-starts from weights in the working directory), NT-054 (per-run fixed costs and GPU launches;
after NT-027 and NT-046).

## Order

Pick rule 3(b) of OPERATING_MODEL "Picking the next item" uses this one total order:
**R1, MVP-1, MVP-6, MVP-2, MVP-3, MVP-4, MVP-5, then R2-R4 (research tracks); R5 whenever the owner
is ready, after R1.** MVP-6 right after MVP-1 is the lead's reading: the indicator catalogue is part of
the structure, and the indicator Q&A did not place it. NT-043 (MVP-5) may run in parallel from the
start (a second implementer, disjoint files). Implementer and experimenter slots are picked separately
by the same order (OPERATING_MODEL); never two GPU jobs at once.

A milestone closes only when all its items are done. The move to D: (D-030) is done (2026-09-28);
NT-009 closes when the owner confirms the D: copy. The MVP is done when VISION "The MVP" holds on the
reference setup; MVP-1 to MVP-6 map to its points.
