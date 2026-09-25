---
name: handoff
description: End a neural_trade session cleanly so the next session can continue without the owner repeating anything - rewrite docs/STATUS.md, update the backlog and decisions, record new owner rules, commit and push. Use at the end of every session, when the owner says "wrap up", "handoff", "stop here", or before the context runs out.
---

Write the handoff. The next session starts from these files and nothing else.

1. **Owner rules:** did the owner state a rule, preference or correction in this session? Record
   each one where future sessions read it (see "Keeping the instructions current" in
   `docs/OPERATING_MODEL.md`): `CLAUDE.md`, the relevant `.claude/agents/*.md`,
   `docs/DECISIONS.md`, `docs/VISION.md` (owner direction only), machine facts in the local memory.
2. **Backlog** (`docs/BACKLOG.md`): every item touched gets its status (`done` with evidence:
   commit, run id, tests; `blocked` with the reason and the evidence; `in-progress` with where it
   stands). New findings become items with priority, role and acceptance criteria.
3. **Decisions** (`docs/DECISIONS.md`): append any decision made, with context and evidence.
4. **Roadmap** (`docs/ROADMAP.md`): update milestone status if one moved.
5. **Status** (`docs/STATUS.md`): rewrite it (do not append a diary). Keep it under about 80 lines:
   - the date, the branch and the last commit;
   - the state in a few lines (what works, what does not, headline numbers with run ids);
   - done this session (with commits / run ids);
   - in progress / blocked (with the reason);
   - **Waiting for the owner:** the questions, each with the options and a recommendation;
   - **Next:** the next backlog item(s) to take.
6. Run `git status`: no stray files (scratch output belongs outside the repo). Commit
   ("Handoff: <date> <one-line summary>") and push the working branch.
7. Tell the owner in a few lines what was done, what is waiting for them, and what comes next.
