# Status

_Rewritten at the end of every session by the `/handoff` skill. Last update: 2026-09-28 (mid-session)._

## Where things stand

- **The vision changed** (owner, 2026-09-25 / 28). `neural_trade` is an indicator-learning predictor
  of financial time series, a substitute for manual indicator search, judged by a searched
  leaderboard on dev-fold net Sharpe after costs. BTC/USDT 1-minute is the reference setup, not the
  purpose. See [VISION.md](VISION.md) (2c58370), the Q&A record
  [qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md) and D-019 to D-030. Plan: R1, then
  MVP-1 to MVP-6, with R2-R4 run as engine scenarios ([ROADMAP.md](ROADMAP.md)).
- **Branch** `remediation/plan` (about 110 commits ahead of `master`, untouched at 7002a71).
  Worktrees `../neural_trade_gates` and `../neural_trade_ablation` (detached at 6dec27a) hold the
  gate and ablation runs; leave them. All still under `C:/Users/Step/Documents/`; the project moves
  to `D:/neural_trade` at the end of this session (D-030), the C: copy deleted only after the owner
  confirms the D: copy.
- **CI is red** since fe4ba85 (the `unit` job; lint passes; last green 609d19e); latest check:
  ed0ed9b, run 36124565636, lint success, unit failure (same step). Cause, reproduced
  locally: CI pins plotly 5.24.1, which writes figure arrays as JSON lists, so two figure
  size-budget tests fail there (`test_viz_delta.py::test_no_empty_panel_and_size_budget_on_a_full_size_block`,
  `test_viz_trading.py::test_size_budget_x0_dx_and_float32`); the local env has plotly 6.7.0.
  Other pins drift too (pandas, scikit-learn). This is NT-001.
- **Tests locally:** 697 fast pass (5-6 min), 13 slow pass; ruff clean; coverage gates pass.
- **Disk (2026-09-28):** C: 9.1 GB free (Docker WSL image), D: 103 GB free.
- **Works (reference setup):** the model trains (remediation gates M1, M2); the variance heads beat
  constant variance (CRPS DM z 6.2 / 4.0 / 4.0) and the conformal 90% intervals cover 0.90-0.91
  (gate M4); nine strict registries; serving API; purged split with baselines and noise tests in
  every report; honest backtest engine; notebooks executed on the latest real run and saved with
  outputs.
- **Does not work (reference setup):**
  - Direction skill (gate M3): latest run test AUC 0.486 / 0.478 / 0.504; at h1 significantly
    worse than the trailing-returns logistic regression (0.523, boot z -2.31).
  - Price heads: served-delta betas 0.12 / 0 / 0.07; raw heads worse than predicting zero.
  - Trading: best strategy `calibrated_quantile` +2.3% gross, -29.9% net after 26 bps round trips
    (buy-and-hold +4.1%).
  - Physics terms: no term shows VALUE (84-run ablation); the family's variance gain is withdrawn
    by the h1 AUC guard-rail. The grid and gate runs predate the served-epoch fix (D-011).
  - Evidence: `runs/gates/REPORT.md`, `runs/ablations/ablate_physics_v1-full/report.md`, the saved
    outputs of `notebooks/01`-`05`; the latest run `runs/20260924T182915Z-1aeff1c-dirty-af67ee43`
    is local only (NT-010).

## Done this session (2026-09-25 / 28)

- Owner Q&A on the vision and the MVP (two owner statements, rounds 1 to 9), recorded in `docs/qa/`
  (b7b8eaf). VISION.md rewritten and approved by the owner as written (2c58370).
- Docs rewritten to the new vision: DECISIONS D-019 to D-030; ROADMAP; BACKLOG NT-026 to NT-046
  (from the Q&A and a code survey, 2026-09-28); OPERATING_MODEL, CLAUDE.md, RUNBOOK, skills and
  agents (sweeps D-024, verdicts D-025, notebooks D-028, deletions D-029, Q&A records in
  `docs/qa/`); README intro, package description and docstring.
- Existing items re-scoped: NT-024 dropped (superseded by NT-026); NT-012 dropped (absorbed into
  NT-037); NT-003 to NT-006 run as engine scenarios; NT-009 in progress (the move to D:); NT-017 P3.

## Waiting for the owner

Nothing below blocks the "Next" list. Items 2-5 were asked 2026-09-25.

1. **Indicator Q&A** (D-027): the design of the indicator catalogue and registry, and where
   indicator combinations come from. Held right after this overhaul, this session, before the move
   to D:. NT-046 is not picked until its answers are recorded in `docs/qa/`.
2. **Pushing (D-017):** sessions push `remediation/plan` after each item and at handoff, and
   implementers push `nt-*` branches for CI (never `master`, never `--force`). This overrides the
   ask-before-push rule of your global `~/.claude/CLAUDE.md` for this project. Keep or revoke.
3. **NT-007, which delta the strategies read** (changes trading behaviour). Today the served,
   beta-shrunk delta, mostly an artefact with beta_h1 = 0. Options: keep served / raw heads / raw
   heads scaled by beta where beta > 0. Recommendation: raw heads for the coherence check, served
   delta for sizing.
4. **NT-006, GPU time for the physics re-run.** It now runs after the engine and the paired
   comparator (NT-026, NT-032), pre-registered again under D-025; its SPEC states the GPU budget
   (the 2026-09-25 estimate was about 7-10 GPU hours). Needs your approval.
5. **NT-008, merge into `master`:** when you are ready, after CI is green (this also lets the
   nightly workflow run).

No longer waiting: the vision (answered 2026-09-28, 2c58370); NT-009 disk (resolved by the move to
D:, D-030; `Bitcoin_BTCUSDT.csv` stays as the 2017-2025 walk-forward file); NT-017 licence (P3, not
needed for the MVP audience, D-019).

## Next

This session (Q&A rounds 6 and 9): the indicator Q&A, then the move to `D:/neural_trade` (D-030),
verified by the tests and `check.py` on D:. NT-009 is done when the move is verified.

Then R1 first (D-021):

1. **NT-001** CI green (P0, implementer): align `requirements-ci.txt` with the tested env or make
   the size budgets version-independent; confirm on the pushed head.
2. **NT-002** random null ignores position size (P0, implementer).
3. **NT-010** tracked evidence for cited numbers (P1), **NT-025** `check.py` enforces the 5 MB
   notebook limit (P2).
4. MVP-1: **NT-026** experiment engine, then **NT-029** config metadata (both P1, implementer).
   **NT-043** learned indicators on price (MVP-5) may run in parallel with a second implementer
   (disjoint files).
