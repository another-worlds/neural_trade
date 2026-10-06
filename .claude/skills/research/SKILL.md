---
name: research
description: Run a neural_trade research question as a multi-agent workflow (the owner's standard for research, D-061) - gather from several angles, design or analyse, adversarially verify, synthesise. Use when the owner types /research <question>, or says "ultracode" for a research question. Not for implementation, QA or routine lookups.
---

The owner invoked this skill, so this turn runs a **Workflow** (D-061: research = ultracode-style
orchestration). The question is the skill's argument; if it is empty, ask the owner for it once.

1. **Scope it first, inline** (a few reads, no agents): restate the question, name the decision or
   number it feeds, the done-criterion, and the sources (repo files, runs, docs, web). Fix the
   acceptance criteria before any agent runs (owner's rule: criteria before experiments).
2. **Budget**: state an agent cap (default under 10, the session's workflow size guideline) and a
   token ceiling in the reply before launching; above about 1M projected tokens, ask the owner first.
   No GPU jobs inside a research workflow (GPU work is the experimenter's, under its own caps).
3. **Author the workflow** (load the `workflow-authoring` skill if its reference is not in context):
   - *Gather*: 3-4 agents in parallel, each a different angle (project docs and runs, git/BACKLOG
     evidence, official docs, external literature or practice); structured findings with sources and
     confidence.
   - *Analyse or design*: 2-3 independent agents from different angles when there is a choice to make.
   - *Verify*: an adversarial agent (`effort: 'xhigh'`) that tries to refute the key claims and the
     numbers; anything refuted is corrected or dropped.
   - *Synthesise*: one agent with a completeness critic; it lists what is still unverified.
   Agents inherit the lead's model (Opus 5.5) and effort (high) unless a stage sets `model` / `effort`;
   use `effort: 'low'` for mechanical stages. Agents are read-only: no edits, commits or GPU runs.
4. **Report** to the owner: the answer first, then the evidence, what was refuted, what is unverified,
   and the cost (agents, tokens, minutes from the completion notice). Record the result where the
   project keeps it (docs/research/<date>-<topic>/ or the item's BACKLOG entry) and any decision in
   DECISIONS; a decision the owner must take goes to STATUS "Waiting for the owner".
