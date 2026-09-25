# Status

_Rewritten at the end of every session by the `/handoff` skill. Last update: 2026-09-25._

## Where things stand

- **Branch** `remediation/plan` (about 107 commits ahead of `master`, which is untouched at
  7002a71). Worktrees `../neural_trade_gates` and `../neural_trade_ablation` (detached at 6dec27a)
  hold the gate and ablation runs; leave them.
- **CI is red** since fe4ba85 (the `unit` job; lint passes; last green 609d19e); latest check:
  ed0ed9b, run 36124565636, lint success, unit failure (same step). Cause, reproduced
  locally: CI pins plotly 5.24.1, which writes figure arrays as JSON lists, so two figure
  size-budget tests fail there (`test_viz_delta.py::test_no_empty_panel_and_size_budget_on_a_full_size_block`,
  `test_viz_trading.py::test_size_budget_x0_dx_and_float32`); the local env has plotly 6.7.0.
  Other pins drift too (pandas, scikit-learn). This is NT-001, the next item.
- **Tests locally:** 697 fast pass (5-6 min), 13 slow pass; ruff clean; coverage gates pass.
- **Disk:** C: 5.2 GB free (Docker WSL image, NT-009), D: 106 GB free.
- **Works:** the model trains (M1, M2); the variance heads beat constant variance (CRPS DM z
  6.2 / 4.0 / 4.0) and the conformal 90% intervals cover 0.90-0.91 (M4); nine strict registries;
  serving API; purged split with baselines and noise tests in every report; honest backtest
  engine; notebooks executed on the latest real run and saved with outputs.
- **Does not work:**
  - Direction skill (M3): latest run test AUC 0.486 / 0.478 / 0.504; at h1 significantly worse
    than the trailing-returns logistic regression (0.523, boot z -2.31).
  - Price heads: served-delta betas 0.12 / 0 / 0.07; raw heads worse than predicting zero.
  - Trading: best strategy `calibrated_quantile` +2.3% gross, -29.9% net after 26 bps round trips
    (buy-and-hold +4.1%).
  - Physics terms: no term shows VALUE (84-run ablation); the family's variance gain is withdrawn
    by the h1 AUC guard-rail. The grid and gate runs predate the served-epoch fix (D-011).
  - Evidence: `runs/gates/REPORT.md`, `runs/ablations/ablate_physics_v1-full/report.md`, the saved
    outputs of `notebooks/01`-`05`; the latest run `runs/20260924T182915Z-1aeff1c-dirty-af67ee43`
    is local only (NT-010).

## Done in the last session (2026-09-24 / 25)

- Notebook figures rebuilt and reviewed in two rounds (144 confirmed findings fixed); notebooks
  executed on real runs (c96db59 ... be93193).
- Bugs found by real runs: served weights were the last epoch, not the best (53c0df2); zero served
  delta treated as a prediction (1863d05, 9a0d070, d1b3019); ablation report hid a guard-rail
  breach (1aeff1c); `Predictor.predict(calibrated=False)` ignored the flag (609d19e).
- Project set up for autonomous multi-session work: `CLAUDE.md`, `docs/` (VISION, OPERATING_MODEL,
  ROADMAP, BACKLOG, STATUS, DECISIONS, RUNBOOK), `.claude/agents/` (implementer, qa,
  experimenter), `.claude/skills/` (`/next`, `/handoff`), `.claude/settings.json` (allowlist),
  notebook tooling moved into `scripts/notebooks/` (542c8c8), nbstripout trap removed (NT-011).
- The setup was checked by a QA pass (cold-start simulations of a new session, an implementer, a
  QA agent and an experimenter, plus fact checks); 32 confirmed findings fixed. New items from it:
  NT-024 (multi-seed gate tooling, needed before NT-003 / NT-004) and NT-025 (notebook size limit).

## Waiting for the owner

All asked 2026-09-25. Nothing below blocks NT-001, NT-002, NT-010 or NT-024.

1. **Vision:** the end goal in `docs/VISION.md` was inferred. Confirm or edit it.
2. **Pushing (D-017):** sessions push `remediation/plan` after each item and at handoff, and
   implementers push `nt-*` branches for CI (never `master`, never `--force`). This overrides the
   ask-before-push rule of your global `~/.claude/CLAUDE.md` for this project. Keep or revoke.
3. **NT-007, which delta the strategies read.** `magnitude_coherent` and the delta-based
   strategies read the served (beta-shrunk) delta; with beta_h1 = 0 it is mostly an artefact.
   Options: keep served / switch to raw heads / raw heads scaled by beta where beta > 0.
   Recommendation: raw heads for the coherence check, served delta for sizing. Changes trading
   behaviour, so it is yours.
4. **NT-006, re-run the physics ablation on the current trainer:** about 7-10 GPU hours.
   Recommendation: yes, after NT-001, on a night the GPU is free.
5. **NT-009, disk:** C: has 5 GB free because of the Docker WSL image (~116 GB). Only you can
   free it. `Bitcoin_BTCUSDT.csv` (291 MB, repo root, unreferenced) is also yours to keep or delete.
6. **NT-017, licence:** none chosen.
7. **NT-008, merge into `master`:** when you are ready, after CI is green (this also lets the
   nightly workflow run).

## Next

In the pick order of OPERATING_MODEL:

1. **NT-001** CI green (P0, implementer): failing test names visible in CI; align
   `requirements-ci.txt` with the tested env or make the size budgets version-independent;
   confirm on the pushed heads.
2. **NT-002** random null ignores position size (P0, implementer).
3. **NT-010** tracked evidence for cited numbers (P1), **NT-024** multi-seed gate tooling (P1,
   implementer), then **NT-003** direction skill (P1, experimenter; GPU).
