# Operating model

How work gets done on this project across many sessions without the owner repeating
instructions and without agents looping or over-managing. Every session follows it.

## Roles

Each role owns different things. A role does not do another role's job.

| Role | Who | Owns | Must not |
|---|---|---|---|
| **Owner** | the human | the vision, the `owner-decision` backlog items, merges into `master`, spending decisions (disk, new packages, long GPU jobs) | - |
| **Lead** | the main Claude session | choosing the next item, writing its acceptance criteria, delegating, integrating, the verdict "done", and the planning docs (STATUS, ROADMAP, BACKLOG, DECISIONS) | implement large changes itself; mark an item done without a QA pass; reopen a recorded decision |
| **Implementer** | subagent `.claude/agents/implementer.md` | code, tests and code docs for ONE backlog item, inside the files the lead assigned | edit planning docs; widen scope; fix unrelated things it notices (it lists them instead) |
| **QA** | subagent `.claude/agents/qa.md` | an independent verdict per acceptance criterion, with evidence | edit source or tests; pass anything it did not check itself |
| **Experimenter** | subagent `.claude/agents/experimenter.md` | GPU runs and experiments: the pre-registered spec, running, reporting against the spec | choose anything on test data; change criteria after seeing results; run two GPU jobs at once |

The lead may do small edits itself (a one-line fix, a doc update, a merge conflict). Anything
that needs tests or touches more than one module goes to an implementer.

## The work loop (one backlog item)

1. **Pick.** Take the highest-priority `todo` item in [BACKLOG.md](BACKLOG.md) whose
   dependencies are met and whose role is not `owner`. Mark it `in-progress` in the backlog.
2. **Specify.** Make sure the item has objective acceptance criteria (a test, a number against a
   threshold, a file that exists, a figure panel that shows something). Write them into the item
   if they are missing. Name the files the implementer may touch.
3. **Implement.** Delegate to the implementer (or the experimenter for runs). Parallel
   implementers need disjoint files, each in its own git worktree.
4. **Verify.** Delegate to QA with the acceptance criteria. QA runs the checks itself.
5. **Repair.** If QA fails an item, send the implementer QA's evidence. **At most two repair
   rounds per item per session.** After that the item goes to `blocked`, with the evidence in the
   backlog and in STATUS. Then move on.
6. **Integrate.** Merge, run the full fast suite and ruff, push the working branch, and check that
   CI is green on the pushed head.
7. **Record.** Set the item to `done` with its evidence (commit, run id, test names). Update
   STATUS. Write any decision into DECISIONS. Put new findings into the backlog, not into the
   current item.

## Limits that stop loops and over-management

- **One item in progress per implementer.** No new item starts while the current one is
  unverified.
- **Findings are triaged, not chased.** A QA or review finding outside the item's acceptance
  criteria becomes a backlog item. It is P0 only if a number is wrong or behaviour is broken;
  polish is P3 and is batched per module.
- **No review of the review.** One QA pass per repair round. A second opinion only for P0 items
  or when QA and the implementer disagree on evidence.
- **Research budget.** A research item states its hypothesis and at most three variants in its
  spec. A negative or inconclusive result closes the item with a record. Further ideas go into
  new items; the current item does not grow.
- **Decisions are not re-litigated.** An entry in [DECISIONS.md](DECISIONS.md) changes only
  through a new entry that cites new evidence. Owner decisions change only with the owner.
- **Stop instead of guessing.** If the next item needs an owner decision, record the question
  in STATUS ("Waiting for the owner") and take the next item that does not need one. If no item
  is left, end the session with a handoff.

## Escalate to the owner (ask; do not act)

- Every backlog item of type `owner-decision`.
- Changing default trading behaviour (strategy defaults, what a signal means), even if it
  looks like a fix.
- Anything that touches `master`, rewrites history or force-pushes.
- Deleting runs, data or anything the session did not create; freeing disk space outside our
  own files (the Docker WSL image on C: belongs to another project).
- Installing or upgrading packages, changing the conda env.
- A GPU job longer than about 3 hours, or any job while the GPU is busy with another project.

## Definition of done

An item is done when all of these hold, and the evidence is in the backlog entry:

- its acceptance criteria are met, and QA says PASS with evidence;
- the fast suite passes (`-m "not slow"`), ruff is clean, and the slow suite passes if training,
  serving or notebooks were touched;
- if figures, notebooks or anything they display changed: the notebooks were rebuilt from
  `scripts/notebooks/build.py`, executed in place on the real defaults, checked, and every
  changed figure was rendered and looked at (see [RUNBOOK.md](RUNBOOK.md));
- docs that describe the changed behaviour are updated (README, docstrings, RUNBOOK);
- it is committed and pushed on the working branch, and the `ci` workflow is green on the pushed
  head (check it: [RUNBOOK.md](RUNBOOK.md), "CI"). Green locally is not enough: CI pins other versions.

## Session protocol

**Start.**
1. Read [STATUS.md](STATUS.md) (it is imported into CLAUDE.md, so it is already in context).
2. `git status` and `git log --oneline -5`: check that you are on the working branch
   (`remediation/plan`) with a clean tree, and that it matches `origin`.
3. Answer any owner questions and act on owner decisions recorded since the last session.
4. Take the next item (the work loop above).

**End (always, even when interrupted).**
1. Rewrite STATUS.md: what was done (with commits and run ids), what is in progress, what is
   blocked and why, the questions waiting for the owner, and the next item to take.
2. Update BACKLOG.md statuses and DECISIONS.md.
3. Commit and push the working branch.

## Keeping the instructions current

When the owner states a rule, a preference or a correction, record it **in the same session**,
in the place future sessions read:

- a project rule or way of working → `CLAUDE.md` (and the agent file of the role it concerns);
- a decision with its reason → `docs/DECISIONS.md`;
- a change of direction → `docs/VISION.md` (owner only) and `docs/ROADMAP.md`;
- a fact about this machine only (paths, disk, GPU sharing) → the machine-local memory, and
  `docs/RUNBOOK.md` if other machines could hit it too.

The owner should never have to say the same thing twice.
