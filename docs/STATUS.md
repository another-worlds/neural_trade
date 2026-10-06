# Status

_Rewritten at the end of every session by the `/handoff` skill. Last update: 2026-10-06 (interim, mid-session:
the takeover reconciliation; the session's `/handoff` replaces it)._

## What happened since 2026-10-01

- **2026-10-01, about 12:36:** the lead's session stopped at a spend limit with no handoff. `remediation/plan`
  was at `9b215dd` (CI green). The STATUS of that day was never written; the BACKLOG entries were current.
- **2026-10-03/04:** another session ("another-world") worked on branch `nt-099`, forked at `f7d4a41`, without
  the 59 commits of 2026-10-01, and did not merge. It recorded the owner quiz (D-050 to D-056), finished the
  NT-099 cells (D-049), wrote D-057, NT-115 and NT-117, and began NT-048 and NT-060.
- **Folders moved:** everything under `D:/` moved into `D:/nt/` (who and when not recorded). The editable install
  and 28 worktree links broke.
- **2026-10-06 (this session):** the owner approved a reconciliation plan. Done so far:
  - editable install re-pointed to `D:\nt\neural_trade\src`; `git worktree repair` on all 28 worktrees (D-058);
  - uncommitted work rescued and pushed: NT-114 (`nt-114` b2cdffc) and NT-031 (`nt-031` 7ef905d);
  - `nt-099` merged into `remediation/plan` (`7a20a4d`, pushed); conflicts were docs only;
  - an independent QA audit of the takeover: the NT-099 statistics hold, but the `ece0_vol0` arm trained at
    `LAMBDA_VOL` 0.1 (the calibration floor), not 0. REPORT erratum, config text and D-058 correct it; the
    shipped default 0 stays (it reproduces the tested arm when calibration runs). NT-118 to NT-120 filed;
  - NT-099, NT-115 and NT-117 reopened until QA passes on the merged head (their QA claim had no record);
  - the BACKLOG table was synced with the item status lines (9 rows were stale).

## Where things stand

- **Branch** `remediation/plan` (master untouched at 7002a71; D-054: no merge yet). Working copy
  `D:/nt/neural_trade` (D-058).
- **Done milestones:** R1, MVP-1. **MVP items:** 10 of 26 done (MVP-6 4/7, MVP-2 2/7, MVP-3 2/5, MVP-4 1/4,
  MVP-5 1/3).
- **Model quality (unchanged):** no direction skill (AUC 0.50-0.53, below a 3-lag logistic regression); the
  variance heads and conformal coverage are the only edge. The owner's goal (stable >60% hit, drawdown <5%) is
  not reached; D-050: no new data source.
- **Defaults changed by the merge:** `LAMBDA_SOFT_ECE` 0 and `LAMBDA_VOL` 0 (effectively the 0.1 floor under
  calibration); strategies read coherence from the raw heads (D-051; not yet on the Predictor path, NT-119).

## In progress

- **QA of the merged head** (Opus): fast, slow and stability suites, golden run, NT-099/115/117 criteria.
- **NT-114** (implementer, `nt-114`): deterministic GRU path, from the rescued WIP; then the GPU check
  (experimenter), which also closes NT-074.
- **NT-112, NT-113:** code pushed (26d8819, 9ffc1d5); QA after the merged-head QA (one full suite at a time).
- **NT-031:** leaderboard WIP (7ef905d), not reviewed.
- **NT-048, NT-060** (from the takeover): partly built, not QA'd; NT-048's `auc` field is a hit-rate drop.
- **Code merged, A/B open:** NT-097, NT-100, NT-104, NT-105, NT-106.
- **PR #15** (remote review sweep, 2026-09-28): not triaged.

## Waiting for the owner

None. Leftovers for the owner, not questions: a stuck `python.exe` (PID 9652, from this session) the permission
classifier did not let the lead stop; untracked `runs/nt_l2_run.log` and `runs/screens_smoke_console.log`
(from the takeover session, left in place).

## Next

1. Merged-head QA verdict; then NT-099, NT-115 and NT-117 to done (or repair).
2. QA and merge NT-112, NT-113; NT-114 to QA, then its GPU check; NT-031 to an implementer.
3. PR #15 triage (D-033).
4. ROADMAP order: MVP-6 (NT-048, NT-060 via NT-120, NT-061), MVP-2 (NT-030, NT-033, NT-034, then NT-050, the
   first overnight Optuna sweep), MVP-3, MVP-4, MVP-5. Experimenter slot: NT-075, then the A/Bs of NT-097, 100,
   104, 105, 106 (each <= 3 GPU-hours), NT-006 when the GPU is free (D-052).
