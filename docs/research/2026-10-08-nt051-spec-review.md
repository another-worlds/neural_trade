# NT-051 SPEC: three qa-deep reviews (2026-10-08)

The SPEC is `runs/experiments/stability_ref_v1/SPEC.md` on branch `nt-051-spec` (draft 1 87ab07d, "SPEC" 9e640f5,
"SPEC repair 1" 7a1d241, "SPEC repair 2" e29f481). All four commits change only that file; `src/` and `configs/`
equal remediation/plan e8a1eac. Thresholds v2 sha256 34a122b28861c13622165aed81fdb1e9405eea91fe4823d0a754d070cbade2cb.
No GPU ran; no result of the study exists. Review 1 (draft 1) and review 2 (9e640f5) and review 3 (e29f481) all
returned FAIL; two repair rounds are used (OPERATING_MODEL "Repair"), so NT-051 is `blocked` for this session with
the exact edits below. The edits are mechanical (wording and one rule), so the next session applies them itself and
asks for one narrow review.

## What holds after review 3

The K / reserve rule and its single 9,720 s basis (simulated independently: with actual times equal to the estimates
the total never exceeds 9,720 s and all 15 cases run exactly when `G_pilot + 56 x t_cell <= 9720`); the timing
sources (`runner.py:372` console line, `wall_s` in result.json, also for probe re-runs); the branch mechanics
(detached run worktree, outputs under `D:/nt/neural_trade/runs/...`, `nt-051-run` dropped); `timeout` ends python
and its child processes in Git Bash (GNU coreutils 8.32; rc 124, `-s KILL` 137), no `taskkill` needed.

## Required edits (P1 blocks any GPU time)

P1
1. Section 6, K and reserve: a probe re-run of a `horizons_5_60_240` cell retrains a 5x cell, so it weighs 5 x
   `t_rerun`. Add: "A re-run weighs like its cell: in the horizons launch (and a retry of one of its cells) each
   re-run counts 5 x t_rerun; that launch passes `--max-probe-reruns min(floor(K/5), 1)` until a re-run has been
   timed, then `floor(K/5)`; a timed horizons re-run updates `t_rerun = max(t_rerun, its time / 5)`." Add "a horizons
   re-run" to the residual-breach list. (Evidence: t_cell 100 s, pilot fails, all three horizons seeds fail: total
   10,184 s > 9,720 at estimates.)
2. Sections 5 and 6 (a), the timeout command: GNU coreutils `/usr/bin/timeout` in Git Bash only (Windows
   `timeout.exe` in PowerShell or cmd runs nothing; `timeout 3 s ...` fails with rc 127); whole seconds
   `floor(10800 - G_used)`, no space before a unit; the detached line
   `nohup env PYTHONPATH=D:/nt/nt_exp_stability_ref_v1/src /usr/bin/timeout <N> neural-trade stability ... > <log> 2>&1 &`;
   rc 124 (137 with `-s KILL`) means the timeout ended the launch. Delete the stray `s` in 6 (a).
3. Section 6 (a), a killed launch: `_judge` writes `stability_verdict.json` (kind primary) into every primary cell
   right after the primary phase (stability.py ~981), before any probe re-run, so the SPEC's "a killed launch has no
   verdicts" is wrong and, as written, turns FAILs (the only cells that get re-runs) into "not run (cap)". Rule:
   "If the kill falls after the primary phase (every primary cell of the launch has `stability_verdict.json` with
   `kind: primary`), those primary verdicts are the case's verdicts (the first run's verdict counts) and a failed cell
   without a finished re-run has the blame '- (killed in the probe phase)'. A kill during the primary phase makes the
   case 'not run (cap)'; no per-cell file of it is used."

P2
4. Section 5: logs `logs/<case>_<n>.log` (n = launch number; the pilot is `pilot.log`), redirect `> <log> 2>&1` (the
   CLI logs to stderr, cli.py:700), sampler files `samples_<case>_<n>.csv`: the earlier names are overwritten by the
   second control launch and by retries.
5. Section 5, Not a verdict: the pilot's retry passes `--max-probe-reruns 1` (t_cell does not exist yet).
6. Section 8.7: "every (case, seed) of a case not reported 'not run (cap)' has exactly one primary verdict".
7. Section 1, evidence commits: `git commit -m "<msg>" -- <paths>` (pathspec only, never `git add` then
   `git commit`), never while a merge or rebase is in progress in the main checkout.
8. Section 1: before the first launch the lead merges `nt-051-spec` into `remediation/plan`; the run worktree stays
   pinned at the SPEC commit.

P3
9. Section 6 scope rule: the 57 condition is sufficient; the exact one is `G_used + 56 x t_cell <= 9720`; the gate
   decides.
10. Section 6, K bullet: "before every launch after the pilot"; when not all remaining cells fit, K = 0 and no failing
    cell is re-run.
11. Line ~185: "(section 7)" should be "(section 6)". 12. Line 3: "replaces drafts 3 (7a1d241), 2 (9e640f5) and 1
    (87ab07d)". 13. Section 8.1: add 87ab07d's message ("NT-051: SPEC draft ..."). 14. Section 6 timing: name the
    source of the launch wall time of a detached launch (`date +%s` before and after, in the log); a retry launch uses
    "its weighted cells (primary or retry)". 15. Section 6 residual breach: drop the citation of the scratch file
    `D:/nt/nt_qa/nt051r4_budget_sim.py` (outside the repo; it does not model horizons re-runs).

## Code findings from the reviews (backlog)

NT-206 (a passing re-run still blames a term), NT-207 (residues), and one more: the harness judges a launch only
after all its cells, and writes the case's `verdicts.json` and REPORT only then; writing each verdict as its cell
finishes would make the budget kill rule cheap (filed as NT-208).

## Evidence

QA scratch (not in the repo): `D:/nt/nt_qa/nt051r5_qa_sim.py` (independent budget simulator),
`nt051r4_timeout_test.sh`, `nt051r4_child.py`, `nt051r3_checks.py`, `nt051r3_budget_sim*.py`, `nt051r4_budget_sim.py`.
