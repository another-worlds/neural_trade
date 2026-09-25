---
name: next
description: Take the next neural_trade backlog item and run it through the work loop (pick, specify, implement, QA, integrate, record). Use when the owner says "next", "continue", "keep going", or starts a session without a specific request.
---

Run one pass of the work loop in `docs/OPERATING_MODEL.md` for the next backlog item.

1. Session start (if not done yet in this session): `git status -sb` and `git log --oneline -5`.
   You must be on the working branch (`remediation/plan`), clean and in sync with origin. Read
   `docs/STATUS.md` "Waiting for the owner" and act on any answers the owner gave.
2. **Pick** from `docs/BACKLOG.md`: the highest-priority `todo` item whose `depends_on` items are
   done and whose role is not `owner`. If the owner named an item, take that one. Set it to
   `in-progress` in the backlog.
3. **Specify:** make sure its acceptance criteria are objective and name the files that may be
   touched. Tighten the item in the backlog if needed.
4. **Delegate:** code → the `implementer` agent (a worktree per implementer when several run in
   parallel, disjoint files). GPU runs / experiments → the `experimenter` agent. Give each agent
   the item ID, the acceptance criteria, the files and the branch name.
5. **Verify:** the `qa` agent with the same acceptance criteria and the commit sha.
6. **Repair** at most twice per item in this session. After that, set the item to `blocked`
   with the evidence and take the next one.
7. **Integrate:** merge into the working branch; run `pytest -q -p no:cacheprovider -m "not slow"`
   and `ruff check src tests scripts`; the slow suite and the notebook routine
   (`scripts/notebooks/README.md`) when the definition of done requires them.
8. **Record:** backlog item → `done` with the evidence (commit, run id, tests); new findings →
   new backlog items (triaged, not fixed now); decisions → `docs/DECISIONS.md`; `docs/STATUS.md`
   updated. Commit and push the working branch.

Then report to the owner in a few lines: the item, the verdict with evidence, what is next, and
any question waiting for them. If time remains and no owner decision blocks it, take the next item.
