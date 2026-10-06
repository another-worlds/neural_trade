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

## GPU: the MVP has priority

- **Lock file** `D:/nt/gpu.lock` (outside the repo). Every GPU job of either session (experimenter
  runs, the notebook routine) writes one line before it starts: `<session> <job> <start UTC>
  <expected end UTC>`, and deletes the file when the job ends. The RUNBOOK free check still applies.
- **The tactical session starts a GPU job only when** there is no lock, the RUNBOOK check says the GPU
  is free, and `D:/nt/gpu.mvp_waiting` does not exist.
- **One tactical launch takes at most 30 minutes** wall-clock. Longer work is split into launches,
  so an MVP job waits at most one launch.
- **The MVP side:** when it wants the GPU while a tactical lock holds it, it creates
  `D:/nt/gpu.mvp_waiting` (one line: what it waits to run), waits for the lock to go (a background
  wait or the tracker: D-042, no pinging), and deletes the waiting file when its own job starts.
- **Stale lock:** a lock whose expected end passed more than 30 minutes ago, with the GPU idle by the
  RUNBOOK check, may be removed; the remover records it in its journal or STATUS.

## Rigour and budget (exploratory)

- Screen layout (NT-088) and micro layout (D-041), one seed, no SPEC and no pre-registration. Every
  journal row says "tactical, exploratory", the seeds and folds it used, and its noise level (D-012).
- **Budget:** up to about 3 GPU-hours per day without asking; more goes to the owner. The journal
  keeps the day's running total.
- **Choices never use test data** (D-020): select on dev folds only; the long file's protected span
  (`DATA_END_PROTECTED_DAYS`) stays protected.
- **No default changes from here.** A finding becomes a default only through an MVP backlog item and
  a paired test (D-025, D-046).
- Standing decisions still hold: no new data source (D-050), trading costs 0 (D-044), TF 2.10 (D-001),
  the physics terms stay (D-003).
- **Goal** unless the owner names another for the session: D-041's (predictive power and PnL; the
  owner's target is a stable hit rate above 60% with drawdown below 5%), with `logreg_lags` as the
  first bar to clear. Start from where `runs/experiments/micro_loop_v1/LOG.md` stopped.

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
