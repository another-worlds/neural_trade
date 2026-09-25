# Operating model

How work gets done across many sessions without the owner repeating instructions and without
agents looping or over-managing. Every session follows it. It is authoritative for roles, the loop,
limits, escalation and the definition of done; CLAUDE.md adds project facts.

## Roles

Each role owns different things. A role does not do another role's job.

| Role | Who | Owns | Must not |
|---|---|---|---|
| **Owner** | the human | the vision, the `owner-decision` backlog items, merges into `master`, spending decisions (disk, the local env, GPU time over the limit) | - |
| **Lead** | the main Claude session | choosing the next item, its acceptance criteria, delegating, integrating, **executing the notebooks** (the routine in `scripts/notebooks/README.md`; notebook 01's ~5-minute training is the one GPU job the lead runs itself), the verdict "done", and the planning docs (STATUS, ROADMAP, BACKLOG, DECISIONS) | implement changes that need tests or touch more than one module; set an item `done` without a QA PASS; reopen a recorded decision |
| **Implementer** | subagent `.claude/agents/implementer.md` | code, tests and code docs for ONE item, in the files the lead assigned, on its own branch `nt-<id>` | edit planning docs; widen scope; run GPU jobs |
| **QA** | subagent `.claude/agents/qa.md` | an independent PASS / FAIL per acceptance criterion, with evidence it produced itself, in its own worktree | edit the repo; touch another checkout's HEAD; pass anything it did not check |
| **Experimenter** | subagent `.claude/agents/experimenter.md` | GPU experiments: the pre-registered SPEC, running it in a pinned worktree, the REPORT against the SPEC | edit `src/` or `tests/` (code an experiment needs is an implementer item first); choose anything on test data; change criteria after results; run two GPU jobs at once |

The lead may make small edits itself (a one-line fix, a doc update, a merge conflict).

## Picking the next item

In this order:

1. the item the owner named in this session;
2. otherwise the item STATUS "Next" names first;
3. otherwise a `todo` item whose `depends on` items are `done` and whose role is not `owner`, by
   (a) priority (P0 first), then (b) the earliest open milestone in ROADMAP "Order", then (c) table
   order in BACKLOG.

While a GPU job runs, the lead may take a CPU item in parallel; never two GPU items.

## The work loop (one backlog item)

1. **Pick** (above). Set the item to `in-progress`.
2. **Specify.** Acceptance criteria must be objective and checkable **at QA time from the delegated
   role's output** (a test, a number against a threshold, a file, a figure panel). STATUS, DECISIONS
   and ROADMAP updates are the lead's step 7, never criteria. Name the files that may change. Read
   the DECISIONS entries for the item's area.
3. **Implement.** Code → the implementer, on branch `nt-<id>` in its own worktree. Experiments → the
   experimenter; if the experiment needs code, open (or take) an implementer item for it first.
4. **Verify.** QA with the acceptance criteria and the commit sha. QA works in its own worktree.
5. **Repair.** If QA fails the item, send the implementer QA's evidence. **At most two repair rounds
   per item per session**, then the item goes to `blocked` with the evidence.
6. **Integrate.** Merge into `remediation/plan`; run the fast suite and ruff. If the definition of
   done's notebook clause applies, run the notebook routine **once** on the merged head (several
   figure items integrated together share one execution): GPU free (RUNBOOK), `execute.py` for the
   affected notebooks (01 only when the model, training, evaluation or 01's own figures changed),
   `check.py`, `render.py`, look at every changed figure, commit the notebooks with outputs. If this
   changed anything QA did not see, QA checks the executed notebooks (one call). Push, then check CI.
7. **Record.** Set the item to `done` with its evidence (commit, run id, tests, QA verdict). Update
   STATUS; write any decision into DECISIONS; put new findings into the backlog, not into this item.

Then take the next item in the same turn (D-017).

## Limits that stop loops and over-management

- **One item in progress per implementer.**
- **Findings are triaged, not chased.** A finding outside the item's criteria becomes a backlog item
  (P0 only if a number is wrong or behaviour broken). While verifying a P3 item, only P0-P2 findings
  are filed. The lead may set an item to `dropped` (with the reason and the date) when it is not
  worth doing; never a P0 or an owner-decision item.
- **No review of the review.** One QA pass per round; a second opinion only for P0 items or when QA
  and the implementer disagree on evidence.
- **QA is for items, not for bookkeeping.** Edits to STATUS / BACKLOG / ROADMAP / DECISIONS and
  recorded owner answers need no QA. A lead edit that closes an item stays `in-progress` until QA
  passes it; several such small items may share one QA call.
- **Blocked items.** A `blocked` item returns to `todo` only with a new approach or new evidence
  written into the item. If it blocks a second time, it goes to the owner (STATUS "Waiting for the
  owner").
- **Research budget.** A research item states its hypothesis and at most three variants in its SPEC.
  A negative or inconclusive result closes the item with a record; further ideas become new items.
  An inconclusive ablation is re-run at most once before the owner decides.
- **Decisions are not re-litigated.** A DECISIONS entry changes only through a new entry that cites
  new evidence. Owner decisions change only with the owner.
- **Stop instead of guessing.** If the next item needs an owner decision, record the question in
  STATUS and take the next item that does not. Stop the run only when no actionable item is left,
  or the owner said stop; then hand off.

## Escalate to the owner (ask; do not act)

- Every backlog item of type `owner-decision`.
- Changing default trading behaviour (strategy defaults, what a signal means).
- Anything that touches `master`, rewrites history or force-pushes.
- Deleting runs, data or anything the session did not create; freeing disk outside our own files
  (the Docker WSL image on C: belongs to another project).
- Installing, upgrading or removing packages in the local `nt` env, or anything else on the owner's
  machine. (`requirements-ci.txt` and CI pins are test infrastructure, not this.)
- GPU time over about **3 hours for one backlog item** (the sum over all runs in its SPEC: every
  variant, fold and seed, plus the judgement; splitting launches does not reset it), or any GPU job
  while the GPU is busy (RUNBOOK). The notebook routine's ~5-minute training does not count.

Ask each question once (options, recommendation, date in STATUS); afterwards only report how many
questions are open.

## Definition of done

An item is `done` when all of these hold, with the evidence in the backlog entry:

- its acceptance criteria are met, and QA says PASS with evidence;
- the fast suite passes, ruff is clean, and the slow suite passes if training, serving or notebooks
  were touched; if tests were added, `TESTING_DOCUMENTATION.md` is regenerated;
- if it touches code that runs every training step: `sec_per_step` in a real run's `status.json` is
  not worse than before (within noise), or the owner accepted the cost (D-018);
- if figures, notebooks or anything they display changed: the notebook routine ran on the real
  defaults (work loop step 6) and every changed figure was looked at;
- docs that describe the changed behaviour are updated (README, docstrings, RUNBOOK);
- it is committed and pushed on the working branch, and the `ci` run for the pushed head is green,
  with two exceptions: (1) it is red only in a job that was already red on the previously pushed
  head and an open P0 item tracks that failure (cite both run ids; once CI writes failure
  annotations, compare the failing test names); (2) CI has not finished 30 minutes after the push:
  the item stays `in-progress` and the next session checks it first.

## Session protocol

**Start:** CLAUDE.md "Start of every session".

**End (always, even when interrupted):** the `/handoff` skill: STATUS rewritten, BACKLOG and
DECISIONS updated, committed, pushed, CI checked.

## Keeping the instructions current

When the owner states a rule, a preference or a correction, record it **in the same session**:

- a project rule or way of working → `CLAUDE.md` (and the agent file of the role it concerns);
- a decision with its reason → `docs/DECISIONS.md`;
- a change of direction → `docs/VISION.md` (owner only) and `docs/ROADMAP.md`;
- a change to the loop → this file and the `/next` and `/handoff` skills, in the same commit;
- a fact about this machine only → the machine-local memory, and `docs/RUNBOOK.md` if other
  machines could hit it too.

The owner should never have to say the same thing twice.
