# Status

_Rewritten at the end of every session by the `/handoff` skill. Last update: 2026-10-07 (end of session)._

## Where things stand

- **Branch** `remediation/plan` (master untouched at 7002a71; D-054), working copy `D:/nt/neural_trade`. Last code head
  83a3907 (merge of nt-187). **CI:** green on c24c1ee (run 37527638543, carries every merge up to NT-179); 83a3907
  is run 37536754652 (check it first; the handoff commit itself is docs-only and not run).
- **Done milestones:** R1, MVP-1. **MVP items 16 of 26 done:** MVP-2 6/7 (NT-050 left), MVP-3 3/5 (NT-039, NT-051),
  MVP-4 1/4 (NT-041, NT-042, NT-052), MVP-5 1/3 (NT-044, NT-045), MVP-6 5/7 (NT-060, NT-061).
- **Model quality (unchanged):** no direction skill. capacity_v1 (NT-104, 5 judgement folds): h1 AUC gru_attention
  0.535, gru_small 0.523, linear_indicators 0.509 against `logreg_lags` 0.552 on the same blocks; verdicts
  inconclusive, default stays gru_attention. The calibrated P(up) is worse than a constant 0.5 out of sample for
  every arm (h1 Brier above 0.25: NT-176). The owner's goal (>60% hit, drawdown <5%) is not reached; D-050.
- **Suites on the pushed code:** fast 2309 + 15 passed (`-n 4`, the 3 failures were "paging file too small"
  and pass alone), slow 39 passed (batch 1d27a7e, two memory-only failures passed alone), ruff clean. Golden record
  `tests/fixtures/golden_nt117.npz` 455/455 on every item that could move it.
- **Machine:** the disk D: is FULL (0.1-0.2 GB free of 639; our project is 11 GB, the rest is the owner's data), and
  the tactical session (`nt-tactical`, D-062/063) runs `screen` shards in parallel (about 20 GB RAM). Both caused
  "paging file too small", OOM and "no space left" failures this session. Heavy suites need `-n 4` or serial.

## Done this session (2026-10-06/07)

- **Merged with QA PASS:** NT-030 (sweeps; 3 QA rounds), NT-031 (leaderboard; 3), NT-033 (frozen twin and TA rules),
  NT-034 (control-panel notebook 06; 3), NT-038 (stability harness and config guard; 2), NT-048, NT-104 (A/B),
  NT-114/074 (deterministic GRU on the GPU), NT-118 (code), NT-119, NT-122, NT-123, NT-125, NT-141, NT-160 (P0, CI),
  NT-182, NT-183 (CI red 8bbf8b7), NT-185, NT-187. NT-075 closed inconclusive. Lead-verified: NT-167, NT-177, NT-179.
- **PR #15** (remote review sweep) merged as a research record and triaged: 37 items filed (NT-122 to NT-158), 20 open
  items got notes. Notebook 06 executed on the real store (no sweep yet), notebook 07 on run 20261003T225052Z-91fa363.
- **Decisions** D-064 (stability thresholds v1 frozen, v2 for NT-051), D-065 (engine identity and concurrency rules).
- **Process:** repair rounds beyond the first were escalated (Opus implementer, qa-deep); the lead lost time to
  memory/disk flakes. About 55 agent calls (implementer ~25, QA Opus ~22, qa-deep 6, experimenter 2, Explore 3).

## In progress / blocked

- **NT-124 blocked** on the owner (question 1): code on `nt-124` (f47e519), not merged.
- **NT-118 (3):** the A/B `CALIB_VOL_ZERO_TO_FLOOR=false` against the default is open (experimenter, needs a SPEC).
  Code merged, A/B open: NT-097, 100, 105, 106. NT-060 in progress (NT-120 first); NT-006 approved (D-052), needs a SPEC.
- **No sweep, no GPU job running.** Heavy run files live only in worktrees `D:/nt/nt_wt_104ab` (capacity_v1 weights),
  `nt_wt_075`, `nt_wt_114`: never delete them. Untracked `runs/nt_l2_run.log`, `runs/screens_smoke_console.log`
  (an earlier session's) stay.

## Waiting for the owner (6 open)

1. **NT-124 / NT-174 (2026-10-07):** the fixed temperature fit sends every horizon to T = 1000 on the reference run
   (P(up) about 0.5; calibrated_quantile becomes degenerate: a trading default). Options: (a) merge and make "no
   direction signal" an explicit reported state, strategies refuse or stay flat (recommended); (b) clamp T (say 10);
   (c) a constant P(up) from the training base rate; (d) skip the direction heads.
2. **Disk D: (2026-10-07):** free 20-30 GB (the Recycle Bin of D: holds 17 GB) or allow moving our worktrees/scratch
   to another volume (C: has 4.9 GB). Until then no new agent worktrees, NT-050/051/173 cannot run.
3. **Commit `.claude/settings.json` (D-061)?** the owner's edit, in the working copy since 2026-10-06.
4. **NT-143:** a `permissions.deny` list for the git commands CLAUDE.md forbids (needs the owner to write the file).
5. **NT-144:** security updates of the `nt` env (TF 2.10.1 and 12 packages).
6. **NT-050 budget heads-up:** two launches (learned and frozen twin: 11.89 h upper bound, 5.4 h expected, each) plus
   CPU rule sweeps; each is under the 12 h cap (D-024), recorded here before launch. Say if one night is wanted.

## Next

1. **Experimenter slot (needs disk, question 2):** NT-173 (re-measure the parallel record on the D-047 default,
   a few GPU-minutes), NT-051's SPEC (v2 thresholds on the REFERENCE profile, the fuzz_jumps expectation, expected
   n_eff per case: D-064), then NT-051; NT-050 after NT-182/183/185 (done) and NT-173; then NT-039, NT-060/061.
2. **Implementer slot:** NT-120 then NT-060's GPU rerun and NT-061; NT-041 then NT-042 (MVP-4); NT-044, NT-045; P2s
   NT-168, NT-171, NT-172, NT-176 (CPU study), NT-180 (served-delta n/a), NT-190; P3 batches NT-181, 184, 186, 188, 189.
3. **Lead:** first the CI check above, the owner questions, `git fetch` and the tactical journal's "For the MVP
   lead" (nothing to adopt yet). The notebook routine for 06 repeats when a real sweep exists.
