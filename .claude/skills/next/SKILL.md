---
name: next
description: Take the next neural_trade backlog item and run it through the work loop (pick, specify, implement, QA, integrate, record), then continue with the following item. Use when the owner says "next", "continue", "keep going", or starts a session without a specific request.
---

Run the work loop of `docs/OPERATING_MODEL.md` (authoritative; this is the checklist).

1. **Session start**, if not done in this session: CLAUDE.md "Start of every session" (fetch,
   status as defined there, CI on the pushed head). Act on owner answers recorded since last time.
2. **Pick** by OPERATING_MODEL "Picking the next item". Set it to `in-progress` in `docs/BACKLOG.md`.
3. **Specify:** objective criteria checkable at QA time, the files that may change, the relevant
   DECISIONS entries. Tighten the item in the backlog if needed.
4. **Delegate:** code → the `implementer` agent (branch `nt-<id>`, its own worktree; parallel
   implementers need disjoint files). GPU experiments → the `experimenter` agent (code the experiment
   needs is an implementer item first). Give each agent the item ID, the criteria, the files, the
   branch.
5. **Verify:** the `qa` agent with the same criteria and the commit sha (it makes its own worktree).
6. **Repair** at most twice per item per session; then `blocked` with the evidence.
7. **Integrate:** merge into `remediation/plan`; fast suite + ruff (slow suite when the definition
   of done requires it); the notebook routine once when figures or notebooks changed
   (OPERATING_MODEL step 6); push; check CI on the pushed head (RUNBOOK "CI").
8. **Record:** backlog item → `done` with the evidence (commit, run id, tests, QA verdict); new
   findings → new backlog items (triaged); decisions → `docs/DECISIONS.md`; `docs/STATUS.md` updated;
   commit by path and push.

Tell the owner in two or three lines: the item, the verdict with evidence, what is next. Then take
the next item in the same turn (D-017) unless a stop condition holds (OPERATING_MODEL "Stop
instead of guessing"); in that case run `/handoff`.
