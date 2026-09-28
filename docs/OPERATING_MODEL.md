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
| **Remote session** | a cloud Claude session the owner starts (D-033) | review sweeps: fixes and findings outside the backlog items, as PRs into `remediation/plan` | take backlog items; merge its own PRs; touch `master` |
| **Experimenter** | subagent `.claude/agents/experimenter.md` | GPU work: sweeps (their stated budget, the run, the leaderboard) and pre-registered studies (the SPEC, running it in a pinned worktree, the REPORT against the SPEC); see "Sweeps and pre-registered studies" | edit `src/` or `tests/` (code an experiment needs is an implementer item first); choose anything on test data; change criteria after results; run two GPU jobs at once (a sweep's parallel trials only as "Sweeps and pre-registered studies" allows) |

The lead may make small edits itself (a one-line fix, a doc update, a merge conflict).

## Picking the next item

In this order:

1. the item the owner named in this session;
2. otherwise the item STATUS "Next" names first;
3. otherwise a `todo` item whose `depends on` items are `done` and whose role is not `owner`, by
   (a) priority (P0 first), then (b) the earliest open milestone in ROADMAP "Order", then (c) table
   order in BACKLOG.

An item that waits on an owner answer (today NT-047, on the window research of
`docs/qa/2026-09-28-indicators.md` Round B) is not picked until the answer is recorded in `docs/qa/`
and DECISIONS and the item's criteria are written from it.

Implementer and experimenter slots are picked separately, by the same order: while implementers work
on CPU items, the experimenter takes the first actionable experimenter item (GPU rules permitting).
While a GPU job runs (a sweep included), the lead may take a CPU item in parallel; never two GPU
items. Two implementers may work at once only on disjoint files (for example NT-043 next to NT-026 or NT-029, not next to NT-027 or NT-028).

## The work loop (one backlog item)

1. **Pick** (above). Set the item to `in-progress`.
2. **Specify.** Acceptance criteria must be objective and checkable **at QA time from the delegated
   role's output** (a test, a number against a threshold, a file, a figure panel). STATUS, DECISIONS
   and ROADMAP updates are the lead's step 7, never criteria. Name the files that may change. Read
   the DECISIONS entries for the item's area. If the item changes what a notebook shows, the
   notebook update through `scripts/notebooks/build.py` is part of the item (D-028).
3. **Implement.** Code → the implementer, on branch `nt-<id>` in its own worktree. Experiments → the
   experimenter, as a sweep or a pre-registered study ("Sweeps and pre-registered studies"); if the
   experiment needs code, open (or take) an implementer item for it first. Once NT-026 exists, every
   sweep and scenario runs through the experiment engine; the frozen set (D-023) stays runnable as
   history, and nothing new builds on it.
4. **Verify.** QA with the acceptance criteria and the commit sha. QA works in its own worktree.
5. **Repair.** If QA fails the item, send the implementer QA's evidence. **At most two repair rounds
   per item per session**, then the item goes to `blocked` with the evidence.
6. **Integrate.** Merge into `remediation/plan`; run the fast suite and ruff (and the other suites
   the definition of done asks for). If the definition of done's notebook clause applies, run the
   notebook routine **once** on the merged head (several figure items integrated together share one
   execution): GPU free (RUNBOOK), `execute.py` for the affected notebooks, new ones included (01
   only when the model, training, evaluation or 01's own figures changed), `check.py`, `render.py`,
   look at every changed figure, commit the notebooks with outputs and the new run's light files
   (NT-010). If this changed anything QA did
   not see, QA checks the executed notebooks (one call). Push, then check CI.
7. **Record.** Set the item to `done` with its evidence (commit, run id, tests, QA verdict). Update
   STATUS; write any decision into DECISIONS; put new findings into the backlog, not into this item.

Then take the next item in the same turn (D-017). Before each pick, fetch `origin` and check the
open pull requests: a PR from a remote session is verified by QA (against the item's criteria, or
the PR's own claims plus the definition of done's general clauses when it is not a backlog item) and
integrated as in step 6 (D-033). A PR that would change a recorded decision, default trading
behaviour or `master` goes to the owner.

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
- **Deletion (owner, D-029).** Code, docs, config fields and repo files are deleted only when they
  are both stale and without any effect on the current system. The item shows the evidence of both
  for each deletion: no production, notebook, script, doc or test use, and not re-created by a
  default. Anything with any effect stays. Runs, data and remote branches are escalated (below).
- **Decisions are not re-litigated.** A DECISIONS entry changes only through a new entry that cites
  new evidence. Owner decisions change only with the owner.
- **Stop instead of guessing.** If the next item needs an owner decision, record the question in
  STATUS and take the next item that does not. Stop the run only when no actionable item is left,
  or the owner said stop; then hand off.

## Sweeps and pre-registered studies

GPU work is one of two kinds, with different limits. An item says which kind it is. The sweep rules
live here only; RUNBOOK, the agent files and the backlog point here.

- **Sweeps are exploratory** (owner, D-020, D-023, D-024).
  - A **quick sweep** takes about 5 minutes wall-clock in total and needs no spec. Its results are
    labelled quick.
  - An **Optuna sweep** commits its scenario / sweep spec with the stated GPU budget before the
    first trial: measured `sec_per_step` x steps x trials x folds, plus the top-5 x 3-seed re-runs
    (NT-030's formula). The lead records the budget in STATUS. No QA.
  - **Cap** (lead's reading of D-024; the owner: "optuna is measured, but can allow for
    overnight"): one launch runs at most one night, about 12 GPU-hours (NT-030's `--max-hours`
    defaults to 12). A larger budget goes to the owner (STATUS "Waiting for the owner") before it
    starts.
  - A sweep runs only while the owner's other project leaves the GPU idle. Parallel trials only as
    RUNBOOK "GPU rules" and NT-030 (4) define them (batches of N, N from NT-035's recorded
    throughput result; until then N = 1).
  - One seed per trial per dev fold; the top 5 are re-run with 3 seeds and ranked by the seed mean.
  - Sweeps rank on the dev folds only; the test-fold columns are shown and never rank (D-020).
    `RESAMPLE_MINUTES` is not tunable until NT-040 is done (NT-029, NT-030).
  - A sweep picks a winner; it is not a verdict that A beats B.
- **Pre-registered studies** (A/B comparisons, ablations, stability-harness runs, research items).
  The SPEC states the hypothesis, at most three variants, the metrics, the minimum effect and the
  guard-rails before any GPU time, and the GPU limit below applies. "A beats B" is judged only by
  the paired comparator (D-025, NT-032): a paired test over (seed, fold) pairs on the same blocks
  plus the pre-registered minimum effect, guard-rails by the same test.
  - **Verdict folds** (lead's reading of D-025): a verdict's pairs are (seed, fold) over judgement
    folds that no choice used; the SPEC names them before any GPU time (fold -1 x at least 5 seeds
    today; more held-out folds from the long history once NT-041 exists); at least 5 pairs.
  - **Exception:** a physics-term ablation (NT-006) may carry one condition per term plus the
    family (D-003) instead of at most three variants; its GPU time still needs the owner.
  - The v1 physics ablation stays the record under its own criteria (D-003, D-025). A negative or
    inconclusive result closes the item with a record; further ideas become new items. An
    inconclusive ablation is re-run at most once before the owner decides.

## Escalate to the owner (ask; do not act)

- Every backlog item of type `owner-decision`.
- Changing default trading behaviour (strategy defaults, what a signal means).
- Anything that touches `master`, rewrites history or force-pushes.
- Deleting runs, data, remote branches, or untracked files the session did not create; freeing disk
  outside our own files (the Docker WSL image on C: belongs to another project). The C: copy after
  the move (D-030): after the owner confirms the D: copy works, the lead deletes the C: copy (repo
  and both worktrees) only on the owner's explicit go-ahead. Repo code and docs follow the deletion
  rule above.
- Installing, upgrading or removing packages in the local `nt` env, or anything else on the owner's
  machine. Already approved: adding optuna (D-023; the lead installs it pinned, after a dry run, as
  RUNBOOK says); re-pointing the editable install to D: (D-030). (`requirements-ci.txt` and CI pins
  are test infrastructure, not this.)
- GPU time over about **3 hours for one pre-registered study or other backlog item** (the sum over
  all runs in its SPEC: every variant, fold and seed, plus the judgement; splitting launches does not
  reset it), or any GPU job while the GPU is busy (RUNBOOK). A sweep has its own cap instead: a
  budget over one night (about 12 GPU-hours) goes to the owner (above). The notebook routine's
  ~5-minute training does not count.

Before asking, read the owner Q&A records in `docs/qa/` and DECISIONS: a question answered there is
never asked again. Ask each new question once (options, recommendation, date in STATUS); afterwards
only report how many questions are open. The answers go, in the same session, into a Q&A record
`docs/qa/<date>-<topic>.md` (the owner's words verbatim where they go beyond the offered options)
and the decisions they settle into DECISIONS.

## Definition of done

An item is `done` when all of these hold, with the evidence in the backlog entry:

- its acceptance criteria are met, and QA says PASS with evidence;
- the fast suite passes, ruff is clean, and the slow suite passes if training, serving or notebooks
  were touched; if tests were added, `TESTING_DOCUMENTATION.md` is regenerated;
- if loss, model, indicator or train-step code changed, and once NT-036 has added the `stability`
  pytest marker: `-m stability` passes (strict mode, masks off);
- if it touches code that runs every training step: `sec_per_step` in a real run's `status.json` is
  not worse than before (within noise), or the owner accepted the cost (D-018);
- if it moves or restructures code without meaning to change numbers: `scripts/golden_run.py verify`
  passes against a record made before the change (D-023);
- if it deletes anything: the D-029 evidence for each deletion is in the backlog entry;
- if figures, notebooks or anything they display changed: the notebooks were updated through
  `scripts/notebooks/build.py` in the same item (D-028: 00-05 keep their numbers and roles, a new
  view gets a new notebook, each figure has one home), the notebook routine ran on the real
  defaults (work loop step 6), and every changed figure was looked at (D-014 on every figure);
- docs that describe the changed behaviour are updated (README, docstrings, RUNBOOK);
- it is committed and pushed on the working branch, and the `ci` run for the pushed head is green,
  with two exceptions: (1) it is red only in a job that was already red on the previously pushed
  head and an open P0 item tracks that failure (cite both run ids; once CI writes failure
  annotations, compare the failing test names); (2) CI has not finished 30 minutes after the push:
  the item stays `in-progress` and the next session checks it first.

## Session protocol

**Start:** CLAUDE.md "Start of every session".

**End (always, even when interrupted):** the `/handoff` skill: STATUS rewritten, BACKLOG and
DECISIONS updated, owner Q&A answers recorded in `docs/qa/`, committed, pushed, CI checked.

## Keeping the instructions current

When the owner states a rule, a preference or a correction, record it **in the same session**:

- a project rule or way of working → `CLAUDE.md` (and the agent file of the role it concerns);
- a decision with its reason → `docs/DECISIONS.md`;
- answers in an owner Q&A → a record in `docs/qa/<date>-<topic>.md`, plus DECISIONS entries that
  cite it;
- a change of direction → `docs/VISION.md` (owner only) and `docs/ROADMAP.md`;
- a change to the loop → this file and the `/next` and `/handoff` skills, in the same commit;
- a fact about this machine only → the machine-local memory, and `docs/RUNBOOK.md` if other
  machines could hit it too.

The owner should never have to say the same thing twice.
