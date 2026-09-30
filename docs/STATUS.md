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

## Strategy study (owner request 2026-09-29, branch nt-005-strategy-study)

NT-005 done with a clear negative: of 12 researched strategy architectures (and 8 model-free EWMA twins), none
makes money after the 26 bps round trip on the dev folds; the recorded winner is always_flat ("do not trade this
model"); the incumbent default calibrated_quantile is the worst of 20. Evidence: runs/experiments/strategy_study_v1/REPORT.md.
New engine pieces: stored predictions and `scenario rescore` (NT-076), an exposure backtest mode and five
variance-driven strategies (NT-077). Follow-ups: NT-078 (EWMA/HAR variance baselines: the EWMA twin beat the
model's sigma), NT-079, NT-080, NT-081.

## Session 2026-09-29 (afternoon/evening): long history, micro loop, new inputs

- **NT-005 done** (strategy study: clear negative; recorded winner always_flat). **NT-076, NT-077, NT-082,
  NT-046 done** (stored predictions + rescore; exposure mode + 5 strategies; SHUFFLE_BUFFER + notebook 08;
  Indicators registry, golden bit-for-bit).
- **360-day run** (D-040; runs/scenarios/long_360d/RESULT.md): 19 min on the GPU; direction at a logistic
  baseline's level; zero-cost re-score shows a real but tiny timing signal (+17% gross in 32 days, ~0.9 bps per
  trade vs 26 bps cost).
- **Micro loop** (D-041, NT-085; runs/experiments/micro_loop_v1/LOG.md): 10 hypotheses (selectivity, holds,
  confidence buckets, horizons 10 min-5 h, windows 60/240, 10 vs 360 days, OHLCV + 14 families): no variant beats
  logreg_lags (AUC 0.51-0.53). Owner's /goal (stable >60% hit, drawdown <5%) not reached.
- **In progress:** NT-047 repair round 1 (QA FAIL: legacy bundles, notebooks 01/04/07, Grappler leak, soft-extremum
  scale); then QA (Opus) and a re-run of the 3 ohlcv14 duel cells. P&L-target research (owner's point 3) running
  (docs/research/2026-09-29-pnl-target/). NT-035 done (re-QA PASS).
- **Rules added:** D-042 (no pinging; Haiku tracker polls), D-043 (model split).
- **Owner question 7 (asked 2026-09-29):** NT-047's default input. D-031 wants all families on by default; the new
  default is 1.64x slower per CPU step and 1.63x per GPU step (0.1735 vs 0.1066 s; D-018 needs the owner); NT-047 passed QA on 72d3838 and waits only on this answer and showed no directional gain in the duel (final re-run on the fixed code 2026-09-30: seed-mean AUC - logreg_lags -0.006 / -0.005 / -0.016, close-only -0.004 / -0.010 / -0.007).
  Recommendation: keep the close-only input and the four families as the default; the new families stay available
  by config until an A/B shows value.

## Owner goal (/goal 2026-09-29): stable >60% hit, drawdown < 5% - where it stands (2026-09-30)

Not reached. 17 lines of attack, all negative, each with committed runs (runs/experiments/micro_loop_v1/LOG.md):
selectivity, holds, confidence buckets, horizons 10 min-5 h, windows 60/240, 10 vs 360 days of training, OHLCV +
14 indicator families (NT-047), cost-sensitive labels (E1), a net-P&L objective (E2, NT-087), triple-barrier labels
(E3 model-free bar), a 960-trial maths/hyperparameter/loss screen (NT-088/NT-092) and its level 2. The network never
beats a 3-lag logistic regression (and is significantly below it at 1 h, z -3.3..-3.5); that baseline's own AUC is
<= 0.53. Research verdict (docs/research/2026-09-29-pnl-target/): the target needs a different information source.
**Next step waits on owner question 8** (a new data source, taker-buy volume first); without it, the lead closes
the micro loop with its report and returns to the MVP backlog.
Done today: NT-035, NT-046, NT-076, NT-077, NT-082, NT-083, NT-087, NT-088, NT-092 (all QA PASS). NT-047 passed QA
and waits on question 7. Open follow-ups: NT-078, NT-079, NT-080, NT-081, NT-084, NT-086, NT-089-NT-091, NT-093.

8. **New data source** (asked 2026-09-30): add Binance taker-buy volume (1-minute klines include it; the local file
   does not), then basis and funding? A new source is outside the MVP (VISION). Recommendation: yes, starting with a
   CPU-only logistic check on 2024-2025 (does order flow lift AUC above 0.53?) before any model work.

## Level-1 screen campaign (2026-09-30; approved plan docs/research/2026-09-29-screen-plan.md)

Specs configs/screens/campaign_l1/ (A hyperparameters 264, B loss weights 328, C physics 288, D loss choice 64,
E maths 16 = 960 trials; rules and slices fixed in the specs before launch). **GPU budget (estimate, stated
before launch):** ~12 s per trial (8 epochs x ~1.4 s at batch 64 + ~1 s scoring, graph reuse per structural
group) = ~3.2 h in one process, ~1.2-1.5 h with 3 shards (NT-035: N = 3 allowed); cap 4 GPU-hours. Screens
maths and stability only; quality comes from level 2 on the survivors.

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
