# Status

_Rewritten at the end of every session by the `/handoff` skill. Last update: 2026-09-29 (end of session)._

## Where things stand

- **Branch** `remediation/plan` (master untouched at 7002a71). Working copy `D:/neural_trade`; the old
  C: copy is ignored (D-038). CI on the pushed head: see the last line of this file.
- **R1 and MVP-1 are done** (ROADMAP). The experiment engine exists (`neural-trade scenario run|plan|reindex`,
  NT-026); packages have no mutual imports and one statistics module (NT-027); stale code removed under
  D-029 (NT-028); every Config field has units, ranges, tunable and deprecated flags and a generated
  reference (NT-029). Notebooks 00-05 and 07 executed on the MVP-1 head, check clean.
- **Numbers (reference setup, test block, run 20260929T081632Z-426de4f-dirty-aba344d6, served epoch 19):**
  direction AUC h0/h1/h2 0.506 / 0.500 / 0.507 (no skill; the trailing-returns baseline has more, NT-003);
  variance CRPSS against constant variance 0.020 / 0.014 / 0.009 and conformal coverage 0.904 / 0.909 /
  0.912 at a 0.90 target (the model's only edge); trading (calibrated_quantile) -33.8% net after costs on
  156 trades, buy-and-hold +4.4% (notebook 02 of that run).
- **GPU facts (NT-035, runs/experiments/gpu_measurements_v1/REPORT.md):** at most 3 concurrent training
  processes (4 crashed); op determinism is free but same-seed runs differ at epoch 0 (NT-074, P1).
- **sec_per_step 0.1066 on the MVP-1 head against 0.0984** on 2026-09-24 (one run each): possibly
  contention, possibly a regression; NT-075 measures it (D-018).
- **The window-free plan is approved** (D-037; docs/research/2026-09-29-window-free-plan/): NT-059 (CPU
  benchmark kit) done; NT-060/061 in MVP-6; the series-engine build-up NT-064 to NT-068 and the A/Bs
  NT-069 to NT-073 are research track R6, after the MVP (D-039). NT-047 is built in window mode.

## Done this session (2026-09-28 / 29)

NT-001 (CI green again; remote PR #14), NT-002 (size-matched random null), NT-010 (every cited run's light
files tracked, 1,060 files; clean tree), NT-025 (5 MB notebook limit), NT-026, NT-027, NT-028 (one repair
round), NT-029, NT-043 (notebook 07, learned indicators on price), NT-053 (the window-free plan: three CPU
investigations, an adversarial review, a revision, approval), NT-059 (benchmark kit), NT-009 (closed by
D-038). Evidence in each BACKLOG entry. Decisions D-033 to D-039 (remote PRs, purge rule, models and the
tracker agent, the plan, the C: copy and pushing, R6 placement). About 21 hours were lost to two spend-limit
stalls; each agent resumed without loss.

## In progress

- **NT-035** (experimenter): results and REPORT committed (426de4f, 7ee2916; 0.47 GPU-hours of a 3-hour
  cap). Open: QA of the REPORT against the SPEC (the SPEC was not QA-checked before GPU time, a deviation
  the experimenter reported). The pinned worktree `D:/nt_exp_gpu_measurements_v1` still holds the run copies
  with weights (untracked, heavy): remove it with `git worktree remove --force` only after QA (it is this
  session's own scratch; the evidence is committed in the main checkout).
- **PR #15** (remote review sweep, docs only: docs/research/2026-09-28-cpu-review-sweep/, 155 findings, 33
  proposed items CPU-01..33 whose ids would collide with NT-058+, evidence for 30 items). **Open, not yet
  integrated:** the next session triages it (D-033: QA its claims, merge the record, add the accepted
  items with fresh ids NT-076+, file the additions into the existing items).

## Waiting for the owner

1. **NT-007** which delta the strategies read (asked 2026-09-25; recommendation: raw heads for the
   coherence check, served delta for sizing).
2. **NT-006** GPU time for the physics re-run, about 7-10 GPU-hours (asked 2026-09-25).
3. **NT-008** merge into `master` when you are ready (asked 2026-09-25).
4. **NT-017** licence (left open; P3).
5. **Inference latency** (asked 2026-09-29): file a P3 item to measure `Predictor.predict` on the GPU
   (single window and batched, a few GPU-minutes)? Recommendation: yes, as P3.

## Next

Start of session: CLAUDE.md steps (fetch, status, CI, open PRs). Then, in order:

1. **PR #15 triage** (lead; D-033).
2. **NT-035 QA** (one QA call), then NT-035 done.
3. **Implementer slot:** NT-074 (P1, same-seed runs differ) and MVP-6: NT-046 (indicator registry), then
   NT-047 (new families, window mode; after NT-046: shared files), NT-048 (HTML report); MVP-2 items that
   are disjoint from them may run beside (NT-032 comparator, NT-031 leaderboard, NT-033 baselines, NT-030
   sweeps with cell locking; NT-034 panel).
4. **Experimenter slot (one GPU job at a time):** NT-075 (speed check, < 1 GPU-hour), NT-060 (the kit on
   the GPU, < 0.5 GPU-hour), then NT-050 (the first Optuna sweep, overnight, after NT-030 to NT-033).
5. Use the `tracker` agent for CI and suite waiting (D-036); poll the GitHub API at most every 3 minutes
   (60 requests/hour shared machine-wide, RUNBOOK "CI").

MVP estimate without spend-limit stalls: about 4-6 days of continuous work (an estimate; GPU queue about
20 GPU-hours, mostly NT-050).

CI on the pushed head: green on d00c73e (run 36542822585, the handoff commit); this one-line update follows it.
