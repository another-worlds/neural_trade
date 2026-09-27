# Status

_Rewritten at the end of every session by the `/handoff` skill. Last update: 2026-09-28 (end of session)._

## Where things stand

- **The vision changed** (owner, 2026-09-25 / 28): an indicator-learning predictor, judged by a
  searched leaderboard on dev-fold net Sharpe after costs; BTC/USDT 1-minute is the reference setup,
  not the purpose. [VISION.md](VISION.md) (2c58370), [qa/](qa/), D-019 to D-031,
  [ROADMAP.md](ROADMAP.md).
- **Indicator Q&A done** (2026-09-28): [qa/2026-09-28-indicators.md](qa/2026-09-28-indicators.md),
  D-031, items NT-046 to NT-048 (MVP-6). **Window research done** (`docs/research/2026-09-28-window-free/`):
  the owner committed to removing the fixed input window; NT-053 writes the plan (D-032); NT-047 waits.
- **Branch** `remediation/plan` (about 110 commits ahead of `master`, untouched at 7002a71).
  Worktrees `../neural_trade_gates` and `../neural_trade_ablation` (detached at 6dec27a) hold the
  gate and ablation runs; leave them. **The working copy is `D:/neural_trade`** (moved and verified
  2026-09-28, D-030: 697 fast tests pass in 3:28 on D:, notebook checks clean, worktrees repaired,
  editable install re-pointed). The C: copy (`C:/Users/Step/Documents/neural_trade` and its two
  worktrees) is untouched: after the owner confirms the D: copy works, the lead deletes it only on
  the owner's explicit go-ahead.
- **CI is red** since fe4ba85 (the `unit` job; lint passes; last green 609d19e); latest check:
  71a0fd2, run 36349235067, lint success, unit failure (the same step); 97ad06b, run 36359654472,
  lint success, unit still running at handoff (the next session checks it first). Cause, reproduced
  locally: CI pins plotly 5.24.1 (local 6.7.0), so two figure size-budget tests fail there (names in
  NT-001); other pins drift too.
- **Tests locally:** 697 fast pass (3:28 on D:, 2026-09-28), 13 slow pass; ruff clean; coverage gates pass.
- **Disk (2026-09-28):** C: 9.1 GB free (Docker WSL image), D: 103 GB free.
- **Works (reference setup):** the model trains (remediation gates M1, M2); the variance heads beat
  constant variance (CRPS DM z 6.2 / 4.0 / 4.0) and the conformal 90% intervals cover 0.90-0.91
  (gate M4); nine strict registries; serving API; purged split with baselines and noise tests in
  every report; honest backtest engine; notebooks executed on the latest real run.
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
- **Trap (NT-049):** an existing `MODEL_PATH` weights file makes training load it and skip (unless `force`).
- **For information:** Casimir and IFE sum their two horizon pairs today, so NT-042 scales the per-pair mean (D-022).

## Done in the 2026-09-25 / 28 session

- Owner Q&A on the vision and the MVP ([qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md),
  b7b8eaf); VISION rewritten and approved as written (2c58370). Indicator Q&A recorded (D-031).
- Docs rewritten to the new vision (04f4296), reviewed, findings resolved: DECISIONS D-019 to D-031;
  ROADMAP with one total pick order; BACKLOG NT-026 to NT-054 (milestone-exit items P1); NT-024 and
  NT-012 dropped (into NT-026, NT-037); OPERATING_MODEL, CLAUDE.md, RUNBOOK, skills, agents, README.
- Window research (five investigations, CPU benchmarks, adversarial review) in
  `docs/research/2026-09-28-window-free/`; the owner's answers recorded (D-032, NT-053, NT-054).

## Waiting for the owner

Nothing below blocks the first items of "Next"; item 1 gates NT-047. Details in BACKLOG and DECISIONS.

1. **Window-free plan** (D-032): NT-053 writes it; you approve it before its first implementation
   item is picked. Research record: `docs/research/2026-09-28-window-free/`.
2. **The move is done:** reopen VS Code at `D:/neural_trade`, confirm it works, then say whether the
   lead may delete the C: copy (repo and both worktrees).
3. **Pushing (D-017, 2026-09-25):** keep or revoke the rule that sessions push without asking.
4. **NT-007** which delta the strategies read (asked 2026-09-25; recommendation: raw heads for the
   coherence check, served delta for sizing).
5. **NT-006** GPU time for the physics re-run, about 7-10 GPU-hours, an estimate (asked 2026-09-25).
6. **NT-008** merge into `master` when you are ready, after CI is green (asked 2026-09-25).
7. **NT-017** licence: left open by you (Round 9, 2026-09-28); P3 (asked 2026-09-25).

No longer waiting: the vision (2c58370); the indicator catalogue (D-031); the window research's
four questions (Round D, D-032); NT-009 disk (resolved by
the move to D:, D-030; `Bitcoin_BTCUSDT.csv` stays as the 2017-2025 walk-forward file).

## Next

Work from `D:/neural_trade` only (CLAUDE.md start step 2). NT-009 closes when the owner confirms the
D: copy; the C: copy is deleted only on the owner's go-ahead.

Then by ROADMAP "Order":

1. R1: **NT-001** CI green (P0), **NT-002** random null ignores position size (P0), **NT-010**
   tracked evidence for cited numbers (P1), **NT-025** `check.py` enforces the 5 MB limit (P1).
2. MVP-1: **NT-026** experiment engine, **NT-027** layering, **NT-028** stale removal, **NT-029** config
   metadata (all P1, implementer; table order).
3. **NT-043** learned indicators on price (MVP-5) may run in parallel (a second implementer) next to
   NT-026 or NT-029, not next to NT-027 or NT-028 (shared visualization/ and build.py). The experimenter slot's first item is NT-035, once NT-026 is done.
4. Then MVP-6 (NT-046; NT-053, the window-free plan, is the lead's and runs beside it; then NT-047,
   NT-048), MVP-2 to MVP-5, R2-R4.
