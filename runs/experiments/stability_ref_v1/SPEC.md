# SPEC: first stability-harness run on the reference setup (NT-051)

Status: amended draft 4 (2026-10-08, SPEC repair 2 after the QA FAIL of draft 3, 7a1d241 "SPEC repair 1"); replaces draft 2
(9e640f5 "SPEC", QA FAIL, never ran) and draft 1 (87ab07d, QA FAIL, never ran). Pre-registered
before any GPU time; it does not change after results exist. This is a pre-registered study of the harness (D-026), not an
"A beats B" verdict: no variant, no paired comparison, no judgement fold; each case is judged by the harness against the
pre-registered thresholds (the same rule for every case). It is bound by the 3-GPU-hour limit (OPERATING_MODEL "Sweeps and
pre-registered studies"); more goes to the owner. Setup of every number below: BTC/USDT 1-minute, the bundled
`binance_btcusdt_1min_ccxt.csv`, the harness `reference` profile, fold -2.

## 1. Pinned code and thresholds

- **Code:** `origin/remediation/plan` at `e8a1eac733ea2df8a445578f1942e70c93298536` (2026-10-08; it carries NT-191, merged as
  844c8fd: `--probe`, `--max-probe-reruns`, `--retry-non-verdict`, NOT A VERDICT, dry-run n_eff, exit codes 0/1/2/64). The
  branch `nt-051-spec` merged that head (merge commits 5d9cd7d and c6995cc; the second brings only `docs/BACKLOG.md` of
  0fd680f) and changes only this SPEC file, so `src/` and `configs/` equal e8a1eac's. The run worktree is made detached at the
  SPEC commit (the last commit of this file, its sha is in the REPORT):
  `git worktree add --detach D:/nt/nt_exp_stability_ref_v1 <spec-commit-sha>`, `PYTHONPATH=D:/nt/nt_exp_stability_ref_v1/src`,
  cwd the worktree (the bundled CSV path is relative). Its HEAD never moves during the study. Before the first launch the
  experimenter records the output of `git diff --stat e8a1eac <run worktree HEAD = the SPEC commit> -- src configs` (must be
  empty). The code must not change between the first and the last launch; if NT-190, NT-192 or any `src/` change merges
  meanwhile, this run does not take it (a new SPEC would).
- **Where the evidence goes and who commits it.** There is no run branch (draft 3's `nt-051-run` is dropped). Every output
  of the study (the harness store, PILOT.md, the logs, the GPU-free and sampler files, the REPORT) is written to absolute paths
  under the main checkout `D:/nt/neural_trade/runs/...`. The main checkout stays on `remediation/plan` and must not switch
  branch during the study. The experimenter commits these light files there by explicit path (CLAUDE.md "Run directories in
  git"; never `git add -A`, `.` or `runs`): PILOT.md before the next launch, the REPORT with the cited runs' light files
  (`scripts/check_run_evidence.py --list-untracked`) at the end. Those commits touch only `runs/`; they do not change the run
  worktree.
- **Thresholds:** `configs/stability_thresholds_v2.yaml`, sha256
  `34a122b28861c13622165aed81fdb1e9405eea91fe4823d0a754d070cbade2cb` (D-064; checked 2026-10-08 on this branch). v1 (0b706aa2...)
  stays frozen and is NOT used. Every launch passes `--thresholds v2`; the sha256 printed in each harness REPORT.md and in
  `verdicts.json` must equal the value above, otherwise that report is void (and re-run, which counts against the budget).
- **Profile:** `reference` only (`PROFILES["reference"]`: N_FOLDS 2, MAX_SEQUENCE_COUNT 4500, EPOCHS 3, strict mode
  `STRICT_LOSS_MASKS`, the D-047 default model otherwise). The `tiny` profile judges no variance check under v2 (n_eff 11/7/5,
  D-064) and is not used for a verdict.
- **Data, fold, seeds:** the bundled CSV (the harness rewrites a CSV per data case; each cell's `meta.json`
  `dataset.sha256` records it), fold -2 (`PROFILE_FOLDS`), seeds 0, 1, 2.
- **Probe mode:** `--probe failed` (the reference default): every cell runs with the per-term probe OFF; a cell that fails a
  verdict check is re-run ONCE with the probe on at PROBE_EVERY 1 for attribution only. Verdicts are identical with the probe
  on or off (NT-191), and the FIRST run's verdict is the one that counts.
- **`DETERMINISTIC_GRU`: off** (the Config default; the harness has no switch for it and this SPEC adds no code). Consequence,
  stated now: GPU runs are not bit-reproducible and the cuDNN GRU path differs between two runs in about the 7th digit
  (RUNBOOK). A probe re-run is therefore a new training run, not a replay: it may not reproduce the first run's failure. If a
  re-run passes where the first run failed, the REPORT records "failure not reproduced on the re-run", keeps the first run's
  FAIL as the verdict, and gives no blamed term (the re-run's probe sample is then of a passing run and is not reported as a
  blame). **Known harness behaviour, overridden here:** at e8a1eac `_attribute` (`stability.py` ~987-1000) names the passing
  re-run's probe sample as the blame of the first run's FAIL, in REPORT.md (cases table, failures list), the re-run table and
  `verdicts.json`, even when the re-run passes (verified 2026-10-08 with the test fake `HarnessFake({"control": "probe_fixes"})`:
  first FAIL, re-run PASS, blamed `crps`, "probe sample, one batch"). The item REPORT overrides it: "failure not reproduced on the
  re-run; no blame", quoting the harness text beside it. NT-035/NT-114 measured the deterministic mode's GPU cost differently and this study is not a comparison study, so
  the mode stays off.

## 2. Disk and store (facts checked 2026-10-08)

- Disk D: has about 13.5 GB free (measured 2026-10-08, 13.46 GB; the disk is near full and the rest is the owner's data).
  C: has no room and is not used.
- Store: `--store D:/nt/neural_trade/runs`; reports under `runs/stability/<harness id>/` (REPORT.md, `verdicts.json`,
  `failing_regions.json`), cells under `runs/scenarios/...` as the engine writes them, logs and the GPU-free check files under
  `D:/nt/neural_trade/runs/experiments/stability_ref_v1/`. Each cell writes `weights.h5` (about 2.9 MB, git-ignored,
  `.gitignore:19`); a cell directory is about 3.4 MB; the 8 data CSVs are 3.2-4.8 MB each, 31 MB in total.
- Size: draft 1's "about 1.2 MB of light files per cell" was not measured and is withdrawn. The pilot records the pilot
  cell's directory size (`du -sk`) and the data CSV size. Each transformed data CSV is 3.2-4.8 MB (8 data cases, 31 MB in
  total; 1 per case launch), written under
  `runs/stability/<id>/data/`, which is git-ignored (`.gitignore:59`, checked with `git check-ignore -v`); its sha256 is in
  the cell's `meta.json`. Before every launch the experimenter requires at least 5 GB free on D: (RUNBOOK "GPU rules"; the study itself needs about
  0.2 GB: 45 cells x 3.4 MB plus 31 MB of data, so 5 GB is the RUNBOOK margin on a near-full disk) and records the free
  space; below that the launch does not start and the lead is told (no run or data is deleted to make room, D-029).
- The light files of the runs the REPORT cites are committed by path (`scripts/check_run_evidence.py --list-untracked`); the
  data CSVs and weights are not.

## 3. Cases, seeds, expected n_eff

Cases: the 15 runnable cases of `neural-trade stability --profile reference --thresholds v2 --seeds 0,1,2 --dry-run`
(re-checked 2026-10-08 on this branch: 45 cells, and every cell's planner `n_eff` equals the v2 table's `expected_n_eff`,
`n_eff_matches_expected: true` for all 45), 3 seeds each.

| case | kind | steps | expected n_eff h0/h1/h2 | variance checks judged under v2 gates (100 for 9a, 30 for 9b) |
|---|---|---|---|---|
| control | sanity | 6 | 150 / 100 / 75 | 9a (excess over baseline) on h0, h1; 9b (CRPS ratio, scaled NLL) on h0, h1, h2 |
| scale_x0.1, scale_x10, vol_x0.1, vol_x10 | scale, volatility | 6 | 150 / 100 / 75 | as control |
| fuzz_constant (100-bar block), fuzz_jumps, fuzz_large_price (x1e4), fuzz_small_price (x1e-4) | extreme input | 6 | 150 / 100 / 75 | as control |
| fault_nan_input, fault_nan_term, fault_nan_gradient | fault | 6 | 150 / 100 / 75 (not scored: the run must stop) | none: detection is judged |
| slow_periods_proxy_lr5, slow_periods_proxy_lr1 | configuration | 6 | 150 / 100 / 75 | as control |
| horizons_5_60_240 | configuration | 24 | 450 / 37 / 9 | 9a and 9b on h0 (450); 9b only on h1 (37); h2 (9) none |

Each cell's recorded `<h>/n_eff` is compared with this table in the REPORT; a difference is reported (the gates then judge
other horizons than stated), not accepted silently.

**N = 2 and N = 4 horizon cases belong to NT-052, not to this item.** The only multi-horizon case here is the existing
`horizons_5_60_240` wide-span case (N = 3).

**Long-memory cases are out of scope** (`long_memory_1440_lr5/lr1`, `long_memory_10080_lr5/lr1`, `long_memory_scale_norm`:
`runnable=False`; the window model caps learned periods at LOOKBACK, the per-channel-scale variant has no Config switch).
The REPORT lists them as "not run, with the reason"; the follow-up is research track R6. The `slow_periods_proxy_lr5/lr1`
cases stand in and are not evidence about 1,440 or 10,080-bar periods.

What the verdict is about (first paragraph of the REPORT): the reference profile trains 360 windows (1,800 for the wide-horizon case) for 3 epochs (6 steps,
24 for the wide-horizon case, which uses N_FOLDS 3 and MAX_SEQUENCE_COUNT 9000, `stability.py` ~265) with 2 folds otherwise. It tests the harness invariants (finite losses and gradients, attribution of
faults, variance-head sanity, coverage) on the D-047 default model, not the behaviour of a long training run.

## 4. Expected outcomes, stated before the run

The harness's verdict is the harness's verdict; this section fixes only how an outcome is classified in the REPORT. Reference
for the variance checks (v2 file, 288 horizon-scores of 96 stored runs): a healthy head's **scaled NLL is 1.14-2.88** (the
constant baseline's 1.42-3.45); the limit is 8.0, the baseline-degenerate bound 8.0; NLL minus the baseline's is -1.91 to +0.49
(limit 2.0); the CRPS ratio over the baseline 0.919-1.006 (limit 1.5). These ranges come from long trainings: the CPU
reference control cell (730f87c) has an h0 CRPS ratio of 1.132, outside 0.919-1.006 and still below the limit 1.5; a 6-step cell
is not expected to sit inside the long-training range. (Draft 1 quoted 1.4-2.0, which is the tiny profile's
range; 1.14-2.88 is the one to compare with.)

1. **Expected PASS** (a FAIL is a real finding): control, scale_x0.1, scale_x10, vol_x0.1, vol_x10, fuzz_large_price,
   fuzz_small_price, fuzz_constant (flat training windows 21 of 360 = 6%, NT-187), and the three fault cases (each must stop
   the run with `UnstableTrainingError`; `fault_nan_term` must name `crps_loss`; `fault_nan_gradient` has no term to name, the
   documented limit of NT-038).
2. **Expected, low confidence: PASS** for horizons_5_60_240 and the two slow_periods_proxy cases. They exist to find failing
   regions; a FAIL is a valid outcome and becomes a region for `Config.validate` and a backlog item, not a harness error.
3. **fuzz_jumps: the outcome cannot be predicted with confidence, and a by-design rule is fixed here before any run.**
   - Why: on the tiny profile the case already scores scaled NLL 7.19 / 6.80 / 6.27 against the limit 8 and its constant
     baseline is about 6.9 (NT-190). The spikes (x4, x0.25) and the x2 level jump inflate every sigma against the test block;
     the baseline is equally inflated. On reference n_eff is 150/100/75, so the checks ARE judged. Possible outcomes: (a) PASS;
     (b) the baseline's scaled NLL exceeds 8 on a horizon, the over-baseline checks (`variance_nll_over_const` and `variance_crps_over_const`, not evaluated by 9c) report "not evaluated" there and only
     the absolute scaled NLL (9d) is judged; (c) FAIL on `variance_nll` (9d, limit 8.0). My forecast, an estimate: (b) or (c)
     about as likely as (a).
   - How the two numbers are computed and where they are read: for a cell, in its `result.json` under `scores`
     (`verdicts.json` holds only each check's worst value, so read `result.json`), per horizon h: `head_scaled(h) = scores["<h>/variance/nll"] -
     ln(scores["<h>/delta/rmse_zero"])` and `base_scaled(h) = scores["baseline/const_var/<h>/variance/nll"] -
     ln(scores["<h>/delta/rmse_zero"])`. This is exactly the harness's own formula (`_variance_checks_v2`, `nll - math.log(rms)`),
     and the `variance_nll` check's value is the worst `head_scaled(h)` over the horizons with n_eff >= 30.
   - **FAIL by design of the case (recorded as such, not as a defect of the model)** when ALL hold, on the same seed: (i) the
     only failed check of the cell is `variance_nll` (9d); (ii) on the horizon the check names as worst, `base_scaled(h) > 5.0`
     (the data's inflation: 1.45x the largest stored baseline, 3.45, and 1.7x the largest healthy head, 2.88); (iii) `head_scaled(h) - base_scaled(h) <= 0.5` on that horizon (the head is no worse than the baseline by
     more than the largest stored excess, +0.49); (iv) `control` passes on the same seed; (v) losses and gradients are finite
     (no `UnstableTrainingError`, `nonfinite_step_rate` 0). Condition (ii) is implied by 9d failing together with (iii) and is kept as an explicit check. The numbers 5.0 and 0.5 are
     fixed by this SPEC from the stored ranges above, before any run.
   - **MVP-3 consequence:** a "FAIL by design" is still a FAIL: MVP-3's exit ("every setup passes the harness") is not met for
     `fuzz_jumps`, and the REPORT files one backlog item (case design, or a v3 thresholds file per NT-190 (4)). A case still
     NOT A VERDICT after its retry also leaves the exit unmet.
   - **Real finding** when any other check fails on a fuzz_jumps cell (non-finite or exploding loss, non-finite gradient
     steps, a clipped share above its limit, coverage below its limit, an `UnstableTrainingError`), or when 9d fails and (ii) or
     (iii) does not hold. Either way the verdict is stored as the harness gave it; thresholds are not changed after results
     (D-064; a v3 file is the owner's question, NT-190 (4)).
4. **Infrastructure failure (NT-199).** A FAIL whose only failed check is `run_completed` with an error other than
   `UnstableTrainingError` (for example `PermissionError [Errno 13]` or `AtomicReplaceError`) is labelled "infrastructure failure
   suspected (NT-199)". The verdict stays FAIL (it is the harness's verdict) and the REPORT files a backlog item.
5. **Any other failure** is a real finding, recorded with the blamed loss term (as a probe sample, section 6) and the failing
   region written for `Config.validate` (NT-038 (5)). Data and fault cases have no failing region by design (RUNBOOK). Failing regions go only to the REPORT and
   `failing_regions.json`; they are NOT copied into `configs/` during the study (that would break section 8.6).
6. **Known blind spots** (so that a PASS is read correctly): v2 judges variance on the h2 horizon of the default cases only by the
   9b checks (n_eff 75); an h2-only sigma error of about x0.3 is not judged (NT-190 (2)); a NaN or inf variance head under v2 is
   not pinned by a test (NT-190 (3)); the n_eff gates 100 and 30 are extrapolations below the stored minimum 135.
7. **Not a finding in this SPEC:** the per-term `term_gradient_share` check. It is report-only in v2, and NT-192 shows the probe
   cadence makes it unreliable (the probe fires once per 6-step cell and its epoch mean divides the shares by the number of
   epochs, so a share cannot exceed 1/3 on 14 of the 15 cases). Draft 1's finding "a term gradient share above its limit" is
   dropped; no verdict and no finding of this study rests on it.

## 5. Execution

- **One GPU job at a time.** One launch per case (3 seeds inside a launch): `neural-trade stability --profile reference
  --thresholds v2 --cases <case> --seeds 0,1,2 --probe failed --max-probe-reruns <the section 6 value> --store D:/nt/neural_trade/runs`
  run under `timeout <10800 - G_used>` (section 6), console to `D:/nt/neural_trade/runs/experiments/stability_ref_v1/logs/<case>.log`. `run_harness` refuses an existing report directory, so a
  launch is never reused; every launch is a new harness id.
- **GPU-free rule, before every launch (and before the pilot):** run `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits`
  ten times, 1 s apart. The GPU is free only if **every** sample is below **2000 MB**. Otherwise do not start: wait and re-check
  every 60 s for about 2 hours at most, then stop and report to the lead for parking. Reason: the RUNBOOK "Determinism" bullet (~line 828, next to NT-035's
  "Concurrent training" bullet at ~line 827: "A GPU-free check can read a median sm of about 40% from the desktop alone: judge by
  fb (memory)") says the desktop alone reads a median sm of about 40%, so the sm rule of RUNBOOK "GPU rules" (~line 186, median sm above 30%) is not applied; `utilization.gpu` is logged in the
  samples for information only. The tactical session's `screen`
  shards (D-063, at most 2 minutes each) hold about 10 GB, and the owner's other project (Docker/WSL) is never touched; a tactical
  run on the GPU is waited out. (2026-10-08 the card showed 4,873 MiB in use, so a launch would have waited.) The check, its
  UTC time, its samples and its verdict are appended to `D:/nt/neural_trade/runs/experiments/stability_ref_v1/gpu_free_checks.log` (committed with
  the REPORT); each launch also writes a 1 s `nvidia-smi` sampler file `D:/nt/neural_trade/runs/experiments/stability_ref_v1/samples_<case>.csv`. The disk check of section 2 is made
  at the same time.
- **Pilot (before anything else; item 10 of the QA review).** One reference-profile cell: `neural-trade stability --profile reference --thresholds v2 --store
  D:/nt/neural_trade/runs --cases control --seeds 0 --probe failed --max-probe-reruns 1`, run under `timeout 10800` (a passing run is then probe-off, as with
  `off`, and a failing pilot still gets its one blame re-run), its own harness id. If the pilot is NOT A VERDICT it is retried
  (below; that retry is one cell and is exempt from the section 6 gate) and `t_cell` is taken only from a completed cell. It
  records: wall time of the launch and of the cell (`time` and `status.json`), the time of a probe re-run if there was one (its
  "done in X s" console line, section 6), the cell's
  `sec_per_step` from its `status.json`, the steps (6), the cell's directory size, the data CSV size (control writes none), the free
  VRAM check lines, and the cell's verdict. The numbers go to `D:/nt/neural_trade/runs/experiments/stability_ref_v1/PILOT.md`, committed BEFORE the
  next launch by the experimenter, by explicit path, on `remediation/plan` in the main checkout (section 1; the SPEC itself does
  not change). The pilot cell IS control seed 0: its verdict counts, and the later control launch
  runs `--seeds 1,2`; the REPORT combines the two launches for the case (PASS only if all three seeds pass; it lists both ids).
  All GPU numbers in draft 1 were CPU numbers or estimates and are withdrawn (section 7).
- **Order** (the priority order; the scope rule below stops at the budget): control (pilot = seed 0, then seeds 1,2), fuzz_jumps,
  fault_nan_term, fault_nan_input, fault_nan_gradient, fuzz_constant, scale_x0.1, scale_x10, vol_x0.1, vol_x10, fuzz_large_price,
  fuzz_small_price, horizons_5_60_240, slow_periods_proxy_lr5, slow_periods_proxy_lr1. (A control and the fault cases come
  first because they decide whether the harness itself works; fuzz_jumps because its outcome is the open question.)
- **Not a verdict:** a cell that ends as NOT A VERDICT (resource error such as out of memory, or a Windows lock error WinError 5,
  32, 33) is re-run once: `neural-trade stability --retry-non-verdict <harness id> --store D:/nt/neural_trade/runs --max-probe-reruns <the section 6 value>`
  (under `timeout <10800 - G_used>`) with the SAME thresholds file, probe mode and profile (the command refuses a different
  thresholds file). Without `--store` it looks in the worktree's own `runs/`; without the cap it defaults to 10. A retry launch
  is subject to the section 6 scope rule, with projected = `t_cell` x its non-verdict cells, weighted as in `K` (a
  `horizons_5_60_240` cell counts 5, every other cell 1); the pilot's retry is exempt (above). The case verdict then uses the re-run (both ids listed). A cell
  still NOT A VERDICT after that is reported in a "NOT A VERDICT" list and **not counted** as pass or fail; the case is reported
  as NOT A VERDICT. The experimenter reads the verdicts from `<store>/stability/<id>/verdicts.json` (`case_status`: PASS | FAIL |
  NOT A VERDICT, and each verdict's `non_verdict`/`passed`), not from the exit code: exit 1 means some verdict failed even if
  non-verdict cells are also left, exit 2 means only non-verdict cells are left, exit 64 means the command was refused and nothing
  ran (not a result; fix the arguments, no GPU time was used). A verdict FAIL is never re-run (except the one probe re-run for
  attribution).

## 6. GPU-time budget, scope rule, probe re-run cap

- **Stated limit:** 3 GPU-hours for the whole item (every launch, the pilot, the retries and the probe re-runs together;
  splitting launches does not reset it). More goes to the owner.
- **Measured numbers on hand are CPU numbers** and are not the GPU time: 58 s per reference cell with the probe off and 777 s
  with it on (at PROBE_EVERY 5; CPU, QA review 2026-10-07, RUNBOOK "Stability harness"); about 650 s of the 777 s is host-side graph
  tracing that a GPU run also pays. Nothing is measured on the GPU for this profile. Draft 1's estimates (2.7 / 3.9 / 5.1
  hours) are withdrawn.
- **Pilot first.** From the pilot: `t_cell` = the pilot's whole-launch wall time (launch overhead included) for one probe-off
  reference cell on the GPU (measured, s); if the pilot cell was re-run with the probe, `t_cell` = the launch wall time minus the
  re-run's time.
- **Timing sources and updates.** Each probe re-run's time is its "done in X s" console line: the engine runner prints
  `[scenario <name>] <cell> <status> in X s` for every cell it runs, the probe re-run included (`runner.py` ~372; the status is
  `done` or `failed`; the same number is `wall_s` in the re-run cell's `result.json`). After each launch:
  `t_cell = max(t_cell, (launch wall time - its probe re-run times) / its weighted primary cells)` (weights as in `W` below), and
  `t_rerun = max(t_rerun, every timed re-run)`. Neither ever decreases. Until a re-run has been timed, `t_rerun = 971 s`
  (1.25 x 777: the CPU probe-on time at PROBE_EVERY 5 times the 1.25x cost of PROBE_EVERY 1 measured on the tiny profile, CPU
  236.7 s against 189.9 s; a CPU stand-in, labelled so).
- **Probe re-run cap `K` and reserve.** Before every launch: `W` = the weighted primary cells still to run, this launch included
  (a `horizons_5_60_240` cell counts 5, every other cell 1; a retry counts its cells with the same weights).
  `K = max(0, floor((9720 - G_used - t_cell x W) / t_rerun))`; the launch passes `--max-probe-reruns min(K, 1)` until a re-run
  has been timed, then `K`; `reserve = K x t_rerun`. Re-runs already done are inside `G_used` and are not subtracted again.
  Failed cells beyond a launch's cap are listed in the REPORT as "not re-run (cap)" with no blame. (The CLI default 10 assumes
  777 s per re-run and is never used.)
- **Scope rule (time-based).** Let `G_used` = the GPU wall time spent so far (pilot, retries and re-runs included, launch start
  to end). Before each case launch the experimenter computes `projected = t_cell x` the launch's weighted cells: `3 x t_cell`
  for a case (a fault case is no longer than `t_cell`, so this is an upper bound; control's second launch `2 x t_cell`), and for
  `horizons_5_60_240`, 24 steps and 1,800 training windows against the 360 of the other cases, `3 x 5 x t_cell` as an upper
  bound, 5 = 1,800/360 (launching `--seeds 0` first is not used). A case launch starts only if
  `G_used + projected + reserve <= 9720 s` (10,800 s less a 10% margin for launch overhead and contention by tactical runs,
  D-063). The gate and `K` use the same 9,720 s, so with the actual times equal to the estimates the gate always holds: if
  `G_used + 57 x t_cell <= 9720 s` after the pilot (57 = 42 cells plus the 3 horizons cells x 5; the pilot cell is counted
  again, a one-cell margin), all 15 cases run. If not, cases run in the order of section 5 until the next one does not fit;
  **the cases not reached are reported "not run (cap)"** and the MVP-3 exit ("every setup passes the harness") is then NOT met;
  the rest is a request to the owner with the measured numbers. A case is never started and stopped halfway for the budget by
  choice (a launch is whole); the only exception is the timeout below.
- **Residual breach.** The gate uses estimates, so actual times above them (contention, a slower re-run than `t_rerun`, a
  horizons cell above 5 x `t_cell`) can still carry the total past 10,800 s (simulation of these rules: `D:/nt/nt_qa/nt051r4_budget_sim.py`, scratch, not in the repo).
  Therefore: (a) the pilot runs under `timeout 10800` and every later launch under `timeout <10800 - G_used> s`; a launch killed
  by the timeout is reported "not run (cap)" for its case, because the harness judges only after all of a launch's cells have run
  (`stability.py` ~1117: `_judge(_run_spec(...))`), so a killed launch has no verdicts; its cells' directories stay (never
  deleted, D-029) and its time counts in `G_used`; no further launch starts. (b) If the total still passes 10,800 s (the timeout
  overshoots by the kill time), section 8.5 is not met, the REPORT says so, no further launch starts, and the lead tells the
  owner (OPERATING_MODEL: GPU time over about 3 hours for one item goes to the owner).
- **Expected size, an estimate until the pilot:** on CPU the 45 probe-off cells would take 45 x 58 s about 0.7 h; the GPU number
  may be lower or higher (cell time is mostly fixed cost: windows, calibration, scoring of 1500 windows, backtest, report).
  If the pilot gives `t_cell` of about 60 s the whole set plus about 6 re-runs (floor((9720 - 57 x 60)/971)) fits; this is not
  claimed here.

## 7. Verdicts and the REPORT

No "A beats B" test. Verdict per case, from the harness against v2 and sections 4-5: **PASS** (all three seeds pass; for a
fault case, all three stop with `UnstableTrainingError`, and for `fault_nan_term` name `crps_loss`), **FAIL** (any seed fails a
check, or `fuzz_jumps` classed "FAIL by design" in section 4 and labelled so), **NOT A VERDICT**, or **not run (cap)**.

`D:/nt/neural_trade/runs/experiments/stability_ref_v1/REPORT.md` (committed with the small summary files and each run's light files, by path)
contains, in this order:
1. the first paragraph of section 3 ("what the verdict is about");
2. verdicts per case (the harness ids, each seed's run id, the verdict, the failing checks, the n_eff recorded against the
   expected table, the combined control case);
3. the probe re-run table (first run id, re-run id, first and re-run verdict, "failure not reproduced" where so, and the blamed
   term, labelled **"probe sample, one batch"** (the largest probe share of the first epoch whose shares sum to 1 on the
   re-run); never from `term_gradient_share`; where the re-run passed, "failure not reproduced on the re-run; no blame", with the harness's
   own blame text quoted beside it (section 1); a failure with no re-run has "-" and the reason);
4. the fuzz_jumps classification of section 4 with `head_scaled` and `base_scaled` per horizon per seed;
5. the list "not re-run (cap)";
6. the list NOT A VERDICT (with the retry's ids);
7. the list "not run (cap)" and the out-of-scope long-memory cases with their reason;
8. the failing regions written for `Config.validate` (configuration cases only);
9. thresholds sha256 and code sha (must match section 1), the GPU-free check lines, the disk free before each launch;
10. the pilot's numbers and the measured wall time per cell against the section 6 rule, the probe re-run wall times, and the GPU
    time used against the 3-hour limit, with the total;
11. what the result means for the backlog (each failure a new item; NT-052; MVP-3's exit is unmet for any FAIL, "FAIL by design" included, and for any case still NOT A
    VERDICT or not run).

## 8. Guard-rails and acceptance criteria (checkable at QA time)

- No `src/` or `tests/` edit; thresholds file unchanged (sha256 as in section 1); no run directory deleted or overwritten (each
  launch a new harness id and report directory); no choice made on any result; criteria not changed after results.
- A FAIL is a valid outcome: it is recorded, each fix becomes a new backlog item, and the item closes with the REPORT.
- Acceptance (NT-051 criteria (1)-(4)):
  1. This SPEC is committed alone (each of its commits changes only this file: "SPEC", "SPEC repair 1", "SPEC repair 2", and any
     later "SPEC repair <n>"), before any GPU time, with the code sha and the thresholds sha256 above and a GPU-time basis within
     3 hours or the owner's approval requested (here: pilot first, then the scope rule of section 6). The SPEC's last commit
     before the first launch is the pre-registered text; the earlier commits 87ab07d (draft 1) and 9e640f5 (draft 2) were
     reviewed and never ran (nor did 7a1d241, draft 3). No commit to SPEC.md after the first launch's start time.
  2. For every launch `gpu_free_checks.log` has a line before it, with ten samples all below 2000 MB and the verdict "free".
  3. Every `REPORT.md` of a harness id shows the sha256 `34a122b2...ade2cb`, the cases with PASS / FAIL / NOT A VERDICT, and every
     run id; the REPORT of this item has the items 1-11 of section 7.
  4. A failed case names a blamed term as a probe sample, or says why none, and the failing region where the case is a
     configuration case.
  5. The sum of the wall times of all launches (pilot, retries and re-runs included), recorded in the REPORT, is at most
     10800 s, or the owner's approval is cited.
  6. At the first and the last launch the run worktree's HEAD (printed in the REPORT) is the SPEC commit or a descendant that
     changes only `runs/experiments/stability_ref_v1/`, and `git diff --stat e8a1eac HEAD -- src configs` is empty. (The worktree is
     detached at the SPEC commit and is not expected to move; the evidence commits go to the main checkout, section 1.)
  7. Across all harness ids of the study, every (case, seed) has exactly one primary verdict, except NOT A VERDICT retries.
