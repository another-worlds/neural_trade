# Status

_Rewritten at the end of every session by the `/handoff` skill. Last update: 2026-10-06 (end of session)._

## Where things stand

- **Branch** `remediation/plan` (master untouched at 7002a71; D-054). Working copy `D:/nt/neural_trade` (D-058).
  CI: green on 9b9999d (run 37413510675, the last code merge); the handoff commit's run is listed below.
- **Done milestones:** R1, MVP-1. MVP items 10 of 26 done (MVP-6 4/7, MVP-2 2/7, MVP-3 2/5, MVP-4 1/4, MVP-5 1/3).
- **Model quality (unchanged):** no direction skill (AUC 0.50-0.53, below a 3-lag logistic regression); the
  variance heads and conformal coverage are the only edge. The owner's goal (stable >60% hit, drawdown <5%) is
  not reached; D-050: no new data source.
- **Defaults changed this session:** `LAMBDA_SOFT_ECE` 0 and `LAMBDA_VOL` 0 (under calibration vol is the 0.1
  floor, the arm NT-099 tested; D-057, D-058); strategies read coherence from the raw heads (D-051; not yet on
  the Predictor path, NT-119). `DETERMINISTIC_GRU` exists, default off (NT-114).
- **Suites on 9b9999d:** fast 1924 passed (3:00, `-n 8`), ruff clean. Golden fixture `tests/fixtures/golden_nt117.npz`
  (455 arrays) is the record for the new defaults.

## Done this session (2026-10-06)

- **Takeover reconciled** (D-058): the 2026-10-01 stall left no handoff; another session worked 2026-10-03/04 on
  `nt-099` without 59 commits. Editable install re-pointed to `D:\nt` (owner-approved), 28 worktrees repaired,
  uncommitted NT-114 and NT-031 work rescued, `nt-099` merged (7a20a4d), BACKLOG table synced.
- **Items done:** NT-099 (QA PASS on 7a20a4d; REPORT erratum: the vol arm trained at 0.1), NT-115 (PASS; (4)
  moved to NT-119), NT-117 (FAIL then re-QA PASS on 9aad6c5), NT-112 and NT-113 (lead-verified, D-060).
  NT-114 code merged (feab7d8, QA PASS on 928d6a1; a CPU-cost claim withdrawn as load noise).
- **New items:** NT-118 (vol gate in calibration + A/B 0 vs 0.1), NT-119 (raw heads on the Predictor path),
  NT-120 (NT-060 kit seeding and gate), NT-121 (fast-suite durations).
- **Way of working** (owner, `docs/qa/2026-10-06-takeover-and-process.md`): D-059 slow suite once per merge batch;
  D-060 QA by risk, test tiers, `scripts/test_changed.py`, CI only on `remediation/plan`/`master` and not for
  docs-only pushes; D-061 one pinned model/effort standard (OPERATING_MODEL "Models and task tracking"): roles
  pinned in `.claude/agents/` (new `qa-deep`), a hook that denies generic agents and logs every agent call,
  `/research` runs research as a workflow.

## In progress

- **NT-114:** the experimenter's GPU check (3 separate-process runs, seed 777, deterministic + DETERMINISTIC_GRU,
  bit-equal val_loss, GPU sec_per_step; < 0.5 GPU-hour). NT-074 closes with it.
- **NT-031:** leaderboard WIP (nt-031 7ef905d), never reviewed: implementer next.
- **NT-048, NT-060** (from the takeover): partly built on nt-048 (= merged), not QA'd; NT-048's `auc` field is a
  hit-rate drop; NT-060's kit fixes are NT-120.
- **Code merged, A/B open:** NT-097, NT-100, NT-104, NT-105, NT-106.
- **PR #15** (remote review sweep, 2026-09-28): not triaged.

## Waiting for the owner

1. **Commit the settings part of D-061?** It is in the working copy since 09:49 (the owner's edit; exactly the
   asked JSON) and not committed: the lead commits it on the owner's yes. Original request (asked 2026-10-06; the permission classifier did not let the lead write
   `.claude/settings.json`): add `"model": "claude-opus-5-5"`, `"effortLevel": "high"` and a `PreToolUse` hook,
   matcher `Agent`, command `C:/Users/Step/miniforge3/envs/nt/python "$CLAUDE_PROJECT_DIR/.claude/hooks/agent_guard.py"`,
   timeout 15. Then start sessions with `D:/nt/neural_trade` as the working folder (a session in `D:/nt` loads
   none of the project's `.claude/`).

Not questions: a stuck `python.exe` (PID 9652, this session's) the classifier did not let the lead stop;
untracked `runs/nt_l2_run.log` and `runs/screens_smoke_console.log` from the takeover session, left in place.

## Next

Start of session: CLAUDE.md steps; confirm D-061 is live (the session model is Opus 5.5 high; a test
`general-purpose` agent call is denied by the hook; `.claude/agent_ledger.jsonl` gets a line).

1. **Experimenter slot:** NT-114's GPU check (closes NT-074), then NT-075 (speed), then the open A/Bs (NT-097,
   100, 104, 105, 106; each <= 3 GPU-hours, pre-registered, >= 5 judgement folds); NT-006 when the GPU is free.
2. **Implementer slot (ROADMAP order):** MVP-6: NT-048 finish (+ the `auc` fix) and QA, NT-120, then NT-060's
   GPU rerun and NT-061 (TF32). MVP-2: NT-031 (from the WIP), NT-030, NT-033, NT-034, then NT-050 (the first
   overnight Optuna sweep; budget recorded here before launch).
3. **Lead:** PR #15 triage (D-033). P2 bugs NT-118, NT-119 when an implementer is free.

## Agent ledger (D-061)

Per agent: role, model, effort, tokens, minutes, outcome. 2026-10-06 (before D-061 roles were not pinned; effort
was the session's):

| agent | model | tokens | minutes | outcome |
|---|---|---|---|---|
| audit of the takeover (general-purpose) | Opus | 169k | 6 | found the vol arm at 0.1 |
| QA of the merge 7a20a4d | Opus | 118k | 30 | NT-117 FAIL (2 stale tests), NT-099/115 PASS |
| implementer NT-114 | Sonnet | 86k | 25 | done; fixed a default-graph layer rename |
| re-QA NT-117 + QA NT-114 | Opus | 108k | 44 | PASS; withdrew a load-noise CPU claim |
| tracker: CI (x2) | Haiku | 42k each | 14-19 | red 7fbf6c1, green 9aad6c5 |
| research workflow: model/effort scheme (9 agents) | Opus | 625k | 7 | D-061 |
| docs check: settings keys (claude-code-guide) | session | 97k | 2 | partly wrong (project `model` key); corrected from the docs |
