# neural_trade: instructions for every Claude session

This file is loaded at the start of every session in this folder, with the files it imports at the
bottom (STATUS, VISION, OPERATING_MODEL, DECISIONS). Together they are everything a new session
needs. **OPERATING_MODEL is authoritative for roles, the work loop, limits, escalation and the
definition of done**; this file adds the project facts and does not repeat it.

## The project in brief

A neural network that predicts financial time series from technical indicators whose parameters and
combinations it learns by gradient descent: a substitute for manual indicator search (D-019). A run
delivers the discovered indicators and the predictions on them (price change, P(up), variance per
horizon), judged by dev-fold net Sharpe after costs against manual-search baselines (D-020). Ticker,
bar size, window and horizons are configuration; BTC/USDT 1-minute (60-minute window, 10/15/20-minute
horizons) is the reference setup, the only one tested in the MVP (D-022). Package: `src/neural_trade/`.

## Start of every session

1. The state and the next item are in `docs/STATUS.md` (imported below).
2. `git fetch origin`, `git status -sb`, `git log --oneline -5`. Expected: branch
   `remediation/plan`, a clean tree, not behind `origin`. **Run directories** (NT-010, RUNBOOK "Run
   directories in git"): their light files are tracked; weights and other heavy files are ignored and
   exist only on this machine: never delete, move or stash them. Untracked light files under `runs/`
   are a run whose evidence is not committed yet: commit them by path if this session made the run
   (`$PY scripts/check_run_evidence.py --list-untracked` for cited runs;
   `git ls-files --others --exclude-standard -- runs/<dir>` for one nothing cites yet), otherwise treat them as below. Never use
   `git clean`, `git stash -u`, `git add -A`, `git add .` or `git add runs`; stage files by explicit
   path. Any other untracked or modified file may belong to another session: do not discard or
   commit it; record it in STATUS and ask the owner. If behind origin: fast-forward only when the tree is clean
   (no modified or staged tracked files); otherwise ask. If this checkout is
   C:/Users/Step/Documents/neural_trade and D:/nt/neural_trade exists, do not work here: the working
   copy is D: (D-030); tell the owner. **If this checkout is `D:/nt/nt_tactical` (branch
   `nt-tactical`), this is the tactical session (D-062): [docs/TACTICAL.md](docs/TACTICAL.md) replaces
   steps 1, 3 and 4 and STATUS "Next".**
3. Check CI on the pushed head and list the open pull requests (docs/RUNBOOK.md "CI"). A PR from
   a remote session into `remediation/plan` is QA'd and merged like an implementer branch (D-033).
4. Then work: the item the owner names, otherwise the `/next` skill. **Keep going** through the
   backlog without asking to continue (D-017); between items, fetch `origin` and check the open PRs
   again (D-033); stop only under OPERATING_MODEL "Stop instead of guessing". End with the
   `/handoff` skill.

## Where things live

| What | Where |
|---|---|
| Why and the end goal (owner-owned) | [docs/VISION.md](docs/VISION.md) (imported) |
| Roles, loop, limits, escalation, definition of done | [docs/OPERATING_MODEL.md](docs/OPERATING_MODEL.md) (imported) |
| Settled decisions (do not re-litigate) | [docs/DECISIONS.md](docs/DECISIONS.md) (imported) |
| Current state, handoff, questions for the owner | [docs/STATUS.md](docs/STATUS.md) (imported) |
| Owner Q&A records: read before asking the owner anything | [docs/qa/](docs/qa/) |
| Milestones and exit criteria | [docs/ROADMAP.md](docs/ROADMAP.md) |
| All open work with acceptance criteria (NT-xxx) | [docs/BACKLOG.md](docs/BACKLOG.md) |
| How to run everything; machine traps | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Subagent roles | `.claude/agents/implementer.md`, `qa.md`, `experimenter.md` |
| Skills | `/next` (one backlog item end to end), `/handoff` (end of session) |
| Notebook workflow | [scripts/notebooks/README.md](scripts/notebooks/README.md) |
| Evidence | `runs/gates/REPORT.md`, `runs/experiments/*/REPORT.md`, `runs/ablations/*/report.md` |

## Project rules (beyond OPERATING_MODEL)

Pointers only; the rule lives where the pointer says.

- **Owner decisions:** DECISIONS D-001 to D-004 (TF 2.10 / Keras 2, registries, physics terms, no
  history rewrite) and D-019 onward (the vision, the yardstick, the MVP plan).
- **Autonomy and pushing** (D-017): push `remediation/plan` and `nt-*` without asking; never
  `master`, never `--force`. This overrides the global "ask before pushing".
- **Models and effort** (owner, D-061): OPERATING_MODEL "Models and task tracking". Lead and plans Opus 5.5
  high (`.claude/settings.json`); research through `/research` workflows; roles pinned in `.claude/agents/`
  (implementer and experimenter Sonnet 5.5 medium, qa Opus 5.5 medium, qa-deep Opus 5.5 high, tracker Haiku);
  delegate only to these roles (a hook denies generic agents). Open this folder, not `D:/nt`, as the workspace:
  `.claude/` loads only from the session's working directory.
- **No pinging** (owner, 2026-09-29, D-042): the lead never polls or checks running agents, runs or suites
  itself and never answers an interim "still running" notification; it waits for the completion notice.
  Any waiting that needs active polling goes to the Haiku 4.5 `tracker`.
- **Remote sessions** (owner, D-033): a remote (cloud) session runs review sweeps and takes no
  backlog items; its PRs into `remediation/plan` are QA'd and merged by the lead.
- **Tactical session** (owner, D-062): a separate local session for exploratory experiments on branch
  `nt-tactical` ([docs/TACTICAL.md](docs/TACTICAL.md)). It shares the GPU in parallel with the MVP,
  runs of at most 2 minutes on ultra-short blocks, no GPU budget (D-063). The MVP lead reads its
  journal's "For the MVP lead" section between items and handles its PRs like a remote session's.
- **Speed** (owner, D-018): the per-step training path must not get slower (definition of done).
- **Evidence** (D-012, D-020, VISION "Principles"): no choice uses test-block numbers, the notebooks'
  and the leaderboard's test columns included.
- **Sweeps and the GPU:** OPERATING_MODEL "Sweeps and pre-registered studies"; RUNBOOK "GPU rules".
- **Tiny first** (owner, D-048): maths, stability and architecture checks on the 6-hour screen layout first; test-run
  discipline and `-n 8` (OPERATING_MODEL "Tiny first").
- **Notebooks** (D-013, D-028): OPERATING_MODEL work loop step 6; never add an nbstripout filter or
  hook (NT-011).
- **Figures** (D-014, D-022, D-027): `src/neural_trade/visualization/theme.py`; no simplified tier.
- **Deleting** (owner, D-029): OPERATING_MODEL "Deletion" (under Limits) and "Escalate to the owner".
- **Packages:** the `nt` env: OPERATING_MODEL "Escalate to the owner". CI pins and
  `requirements-ci.txt` are test infrastructure; an item may change them (TF stays 2.10.x).

## Environment essentials (Windows 10, RTX 4070 Ti)

- Python: `C:/Users/Step/miniforge3/envs/nt/python` (conda env `nt`, Python 3.10, TF 2.10 GPU). Do
  not use `conda run`. `import neural_trade` before `tensorflow` (it puts the CUDA DLLs on PATH).
- Tests and CPU scripts: `CUDA_VISIBLE_DEVICES=-1`; never for real training or notebook 01. The
  console is cp1251: `PYTHONIOENCODING=utf-8`.
- `ptxas.exe ... CreateProcess failed` log lines are harmless.
- One GPU job at a time (sweeps: OPERATING_MODEL); the owner's other project also uses this GPU
  (Docker/WSL): never touch it. Is the GPU free: RUNBOOK "GPU rules".
- Disk C: is nearly full (the other project's Docker image): scratch, renders and worktrees go to D:/nt/.
- The working copy is `D:/nt/neural_trade` (moved 2026-09-28, D-030; into D:/nt/ by 2026-10-06, D-058). The old C: copy is ignored (D-038):
  never delete it, never ask about it. (Formerly: deleted only on
  the owner's go-ahead (OPERATING_MODEL "Escalate to the owner"); never work in it.
- Bash heredocs with nested quotes break easily here: write scripts with the Write tool.
- The editable install imports the main checkout's `src/`: in a worktree set `PYTHONPATH=<worktree>/src`
  for ad-hoc scripts (pytest and `scripts/notebooks/*` do it themselves).

## Commands (details and more in docs/RUNBOOK.md)

```bash
PY=C:/Users/Step/miniforge3/envs/nt/python
$PY scripts/test_changed.py --run                                              # tier 1: tests of the changed modules (D-060)
CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m "not slow" -n 8   # fast suite, ~3.5 min (pytest-xdist, D-048)
CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -q -p no:cacheprovider -m slow -n 8   # slow suite; never two full suites at once (D-048)
$PY -m ruff check src tests scripts
$PY scripts/notebooks/build.py            # regenerate notebooks/ from the generator
$PY scripts/notebooks/execute.py          # execute in place (01 trains ~5 min on the GPU)
$PY scripts/notebooks/check.py            # errors / stderr / empty panels / unexecuted cells / over 5 MB
$PY scripts/notebooks/render.py           # PNGs of every saved figure, then LOOK at them
CUDA_VISIBLE_DEVICES=-1 $PY scripts/test_inventory.py   # regenerate TESTING_DOCUMENTATION.md
```

## Talking to the owner

Short and factual; the owner writes English and Russian (the docs are in English, D-019). Lead with
what changed and what they can open to see it. Every claim with its evidence. Say plainly when
something failed or was not checked. Ask a question once (check `docs/qa/` and DECISIONS first),
with options and a recommendation, record it in STATUS with the date, and afterwards only say how
many questions are still open.

## Keep these instructions current

When the owner states a rule, preference, correction or Q&A answer, record it in the same session,
where OPERATING_MODEL "Keeping the instructions current" says (project rules in this file, decisions
in DECISIONS, Q&A records in `docs/qa/`). neural_trade rules live only in this repo
(`~/.claude/CLAUDE.md` is for cross-project preferences).

---

@docs/STATUS.md

@docs/VISION.md

@docs/OPERATING_MODEL.md

@docs/DECISIONS.md
