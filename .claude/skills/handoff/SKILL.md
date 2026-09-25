---
name: handoff
description: End a neural_trade session cleanly so the next session can continue without the owner repeating anything - record new owner rules, update the backlog, decisions and roadmap, rewrite docs/STATUS.md, commit, push and check CI. Use at the end of every session, when the owner says "wrap up", "handoff", "stop here", or before the context runs out.
---

Write the handoff. The next session starts from these files and nothing else.

1. **Owner rules:** did the owner state a rule, preference or correction in this session? Record each
   one where future sessions read it (OPERATING_MODEL "Keeping the instructions current").
2. **Backlog** (`docs/BACKLOG.md`): every item touched gets its status (`done` with evidence: commit,
   run id, tests, QA verdict; `blocked` with the reason and evidence; `in-progress` with where it
   stands; `dropped` with the reason). New findings become items with priority, role and criteria.
3. **Decisions** (`docs/DECISIONS.md`): append any decision made, with context and evidence.
4. **Roadmap** (`docs/ROADMAP.md`): update a milestone's status if it moved.
5. **Status** (`docs/STATUS.md`): rewrite it (not a diary), under about 80 lines:
   - the date, the branch, the last commit, and CI on the pushed head (run id and result);
   - the state in a few lines (what works, what does not, headline numbers with run ids);
   - done this session (commits, run ids);
   - in progress / blocked (with the reason); long jobs still running (how to check them);
   - **Waiting for the owner:** new questions with options, a recommendation and the date asked;
     questions already asked stay as one line each;
   - **Next:** the next item(s), in the pick order of OPERATING_MODEL.
6. **Commit and push:** `git status` must show no modified tracked files you did not mean to commit;
   untracked `runs/` directories stay as they are (CLAUDE.md start step 2). Stage by path, commit
   ("Handoff: <date> <one-line summary>"), `git push origin remediation/plan`, then check CI on the
   pushed head (RUNBOOK "CI") and put the result in STATUS if it changed.
7. Tell the owner in a few lines: what was done, how many questions wait for them (with any new one
   spelled out), and what comes next.
