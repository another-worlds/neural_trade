# Tactical session (owner, D-062)

A second local Claude session, separate from the MVP work, for quick tactical experiments with the
network. This file is authoritative for that session. CLAUDE.md, OPERATING_MODEL and DECISIONS apply
to it unless this file says otherwise. The MVP session follows only the "GPU" section and "Findings
for the MVP".

## Where it works

- **Worktree** `D:/nt/nt_tactical`, **branch** `nt-tactical`. The owner opens that folder as the
  workspace (`.claude/` loads from the working directory). At start the expected branch is
  `nt-tactical`, not `remediation/plan`.
- **Journal and handoff:** `runs/tactical/LOG.md`, one row per hypothesis (method, cost, result,
  evidence path), as `runs/experiments/micro_loop_v1/LOG.md` does. Scenario and sweep configs:
  `configs/tactical/`. One-off analysis scripts: next to their results under `runs/tactical/<id>/`.
- **Git:** commits only to `nt-tactical` (and short implementer branches `nt-tactical-<topic>` off
  it), stages files by path, pushes `nt-tactical` without asking (D-017 covers `nt-*`). It never
  commits to `remediation/plan` and never edits STATUS, BACKLOG, ROADMAP or DECISIONS (the MVP lead's).
  At session start, when its tree is clean, it merges `origin/remediation/plan` into `nt-tactical`
  (a merge, never a rebase: D-004).

## GPU: shared, in parallel (owner, D-063; supersedes D-062's MVP priority)

- The tactical session uses the GPU **in parallel** with the MVP session: it does not wait for MVP
  jobs and the MVP does not wait for it. The owner's other project (Docker/WSL) is still never touched:
  if the RUNBOOK check shows that project on the GPU, wait.
- **Every tactical run trains on an ultra-short block and takes at most 2 minutes wall-clock**
  (screen layout, NT-088; the micro layout's ~4-minute cells are too long). A run that hits 2 minutes
  is stopped and recorded as over-limit, not extended.
- No lock file and no waiting flag (D-062's are withdrawn).
- **Never idle (owner, 2026-10-10: "почему стоит????").** The queue must not run dry while the owner is away: when the
  last running task finishes, the lead immediately starts the next steps it recommended (and reports them), instead
  of waiting for the owner to pick. Questions go alongside the running work, never in place of it.
- **Heavy variants (persistent-tape ones: INDICATOR_GRAD_SOURCE direction, geometry + indgrad): at most 2 processes
  on the GPU at once** (H43: three such processes on the 12 GB card coincided with two driver resets).

## Rigour and budget (exploratory)

- **Data: the static excerpt by default (owner, 2026-10-08: "the whole file only for giant runs"):** tactical runs and
  tests read `D:/nt/neural_trade/Bitcoin_BTCUSDT_tactical.csv` (1,741,241 of 4,598,198 rows, 112 MB,
  sha256 `e2f4ccfcd973b0c6...`; git-ignored). It holds, unchanged, the 95 days before each of the 12 tactical
  slices (6 climb + 6 held-out final, runs/tactical/hc4/make_hc4.py) and the file's last 70 days (the protected span,
  so DATA_END_PROTECTED_DAYS refuses the same dates). Checked: a long-block run gets bit-identical windows from it
  (runs/tactical/data/excerpt_check.json), loading is 2-2.5x faster. Rebuild: `python runs/tactical/data/make_excerpt.py`
  (manifest: runs/tactical/data/excerpt_manifest.json). A run on other DATA_END values (e.g. the 40-slice design) needs
  the full `Bitcoin_BTCUSDT.csv`, as do giant runs (360-day, the long-history studies).
  Tactical specs also set `DATA_TAIL_TRIM: true` (windows built only from the tail the run keeps; bit-identical arrays,
  tests/test_data_tail_trim.py): on a long-block slice 24.5 -> 5.0 s and 2.1 GB -> 0.8 GB peak RAM.
- **Standard screening design (owner, 2026-10-07):** 6 slices x 2 seeds = 12 trials per variant, paired with the
  base on the same (slice, seed); accepted tolerances: error of the effect up to 0.05, false wins up to 10%
  (measured: mean error 0.014, worst 5% 0.043, false 'significant' 7%). It sees only large jumps (about +0.05 AUC);
  a survivor is confirmed on the 40 climb slices, then once on the 10 final slices (runs/tactical/make_hc2.py).
- **Hill-climb metric (owner, 2026-10-07):** the aggregate of `runs/tactical/hc4_metric.py`: price, direction and
  confidence groups at 1/3 each, every paired difference in seed-noise units, verdict by the 95% interval over slices
  with no group below -0.5. Block: 1 day if the epochs check (runs/tactical/epochs_1d/SPEC.md) passes, else 7 days.
- **Load guard (owner, 2026-10-07: no critical CPU/RAM load, real time; then "everything lags" -> "more carefully"):**
  runs start only through the guard scripts (`runs/tactical/hc4/run_guarded_v*.sh`): our processes at BELOW-NORMAL
  priority (owner, 2026-10-08: idle starved a run's data preparation from ~1 to ~19 minutes on a busy machine), TensorFlow grows
  GPU memory as needed (TF_FORCE_GPU_ALLOW_GROWTH); start only with >= 16 GB RAM and >= 5 GB GPU memory free, GPU use <
  40%, CPU < 95%; stop and re-queue below 8 GB RAM or 1.5 GB GPU memory free, or at CPU/GPU >= 98% for a minute; checks
  every 10 s into resources.csv and the dashboard. The GPU also draws the owner's desktop: three processes (v2) filled it
  (100%, 11.9 of 12.3 GB) and the machine lagged. Never edit a guard script while it runs (bash reads it as it goes).
- **One dashboard tab per run (owner, 2026-10-07):** every new experiment gets an entry in `EXPERIMENTS` of
  `runs/tactical/dashboard.py` when it is launched (progress, its rules from the SPEC, all 9 outputs, verdict), so the
  owner follows progress in `runs/tactical/dashboard.html` without asking; the running experiment's tab opens by default.
- **All 9 outputs, always (owner, 2026-10-07):** every run measures the delta, direction and variance heads of every
  horizon and saves its predictions (screen `run.save_predictions`), never the direction AUC alone.
- Screen layout (NT-088), one seed, no SPEC and no pre-registration. Every journal row says
  "tactical, exploratory", the seeds and folds it used, and its noise level (D-012).
- **Budget: none** (owner, D-063): GPU time is not capped or counted against a limit; the journal still
  records the GPU minutes each round used.
- **Choices never use test data** (D-020): select on dev folds only; the long file's protected span
  (`DATA_END_PROTECTED_DAYS`) stays protected.
- **No default changes from here.** A finding becomes a default only through an MVP backlog item and
  a paired test (D-025, D-046).
- Standing decisions still hold: no new data source (D-050), trading costs 0 (D-044), TF 2.10 (D-001),
  the physics terms stay (D-003).
- **Goal** (owner, D-063): escape the strategic trap (no direction skill: AUC 0.50-0.53, below
  `logreg_lags`) by optimisation: search for a tactical breakthrough in how the network computes
  (architecture, losses, training, inputs from the existing data). The measure is direction skill,
  with `logreg_lags` on the same blocks as the first bar to clear. **Risk management** (drawdown,
  sizing, stops, the trading side of the owner's >60% / <5% target) is a separate, independent
  branch of work, not this session's.

## Roles: as in the MVP

- The tactical lead plans hypotheses, writes their criteria, judges results and keeps the journal.
  Code goes to the `implementer` (branch `nt-tactical-<topic>`, merged into `nt-tactical` by the
  tactical lead); GPU runs go to the `experimenter`, in a worktree pinned to an `nt-tactical` commit.
  Models and effort as D-061; no pinging (D-042).
- **QA by risk** (D-060): exploratory code that stays on `nt-tactical` needs no QA agent, only tier 0-1
  tests (OPERATING_MODEL "Test tiers"). A full suite only when no other pytest run is going on the
  machine (D-048: never two full suites at once).
- New code that changes numbers sits behind a Config switch whose default is today's behaviour, so a
  later merge into `remediation/plan` keeps the golden run.

## Findings for the MVP

A finding worth adopting goes into the journal's "For the MVP lead" section with its evidence, and,
when it needs code, as a pull request from `nt-tactical` (or a narrow branch off it) into
`remediation/plan`. The MVP lead QA's and merges it like a remote session's PR (D-033) and files the
paired test as a backlog item. The MVP lead reads that section between items.

## Session start and end

- **Start:** this file and the journal's "Handoff" section; `git fetch origin`, `git status -sb`;
  merge `origin/remediation/plan` if the tree is clean; check the GPU lock.
- **End:** rewrite the journal's "Handoff" section (state, next hypotheses, GPU hours used today),
  commit by path, push `nt-tactical`. Not the MVP `/handoff`: it would rewrite STATUS.
