# SPEC (draft): first stability-harness run on the reference setup (NT-051)

Status: DRAFT for the lead's QA of the SPEC. Pre-registered before any GPU time; it does not change after results
exist. Written 2026-10-07 on `remediation/plan` 730f87c (the head where `neural-trade stability --thresholds v2` and
the v2 file exist). This is a pre-registered study of the harness (D-026), not an "A beats B" verdict: there is no
variant, no paired comparison and no judgement fold; each case is judged by the harness against the pre-registered
thresholds (the same rule for every case). It is bound by the 3-GPU-hour limit (OPERATING_MODEL).

## Pinned code and thresholds

- Thresholds: `configs/stability_thresholds_v2.yaml`, sha256
  `34a122b28861c13622165aed81fdb1e9405eea91fe4823d0a754d070cbade2cb` (committed alone in ad3f590, unchanged since;
  D-064). v1 (0b706aa2...) stays frozen and is NOT used. Every launch passes `--thresholds v2`, and the sha256 in each
  harness REPORT.md must equal the value above; a mismatch voids that report.
- Profile: `reference` only (the screen-size layout of `PROFILES["reference"]`: N_FOLDS 2, MAX_SEQUENCE_COUNT 4500,
  EPOCHS 3, per-term probe on every 5 steps, VAL/CAL fraction 0.1, strict mode `STRICT_LOSS_MASKS`, the D-047
  default model otherwise). Under v2 the `tiny` profile judges no variance check (n_eff 11/7/5), so it is not used for a
  verdict (D-064).
- Code: a worktree `git worktree add --detach D:/nt/nt_exp_stability_ref_v1 <sha>` of the head the lead names when the
  SPEC passes QA (at least 730f87c; if NT-190 merges first, that head), `PYTHONPATH=D:/nt/nt_exp_stability_ref_v1/src`,
  cwd the worktree (the bundled CSV path is relative). The sha is recorded in the REPORT. The harness code must not
  change between the first and the last launch.
- Data: the bundled `binance_btcusdt_1min_ccxt.csv` (the harness rewrites a CSV per data case; run meta records the
  fingerprint). Fold: -2 (the harness's `PROFILE_FOLDS`), seeds 0, 1, 2.
- Outputs: `--store D:/nt/neural_trade/runs` (reports under `runs/stability/<id>/`), a per-case folder
  `runs/experiments/stability_ref_v1/` for the logs. Disk D: has about 30 GB free (96% full): about 1.2 MB of light
  files per cell, no weights (`save_artifacts` is off in the harness).

## Cases, seeds, expected n_eff

The runnable cases are the 15 of `neural-trade stability --profile reference --dry-run` (checked 2026-10-07: 15 cells
at one seed), each with 3 seeds (the thresholds file's `seeds: 3`): 45 cells in all.

| case | kind | expected n_eff h0/h1/h2 | variance checks judged |
|---|---|---|---|
| control | sanity | 150 / 100 / 75 | 9a (excess over baseline) on h0, h1; 9b (CRPS ratio, scaled NLL) on h0, h1, h2 |
| scale_x0.1, scale_x10 | scale | 150 / 100 / 75 | as control |
| vol_x0.1, vol_x10 | volatility | 150 / 100 / 75 | as control |
| fuzz_constant (100-bar block) | extreme input | 150 / 100 / 75 | as control |
| fuzz_jumps | extreme input | 150 / 100 / 75 | as control (see the expectation below) |
| fuzz_large_price (x1e4), fuzz_small_price (x1e-4) | extreme input | 150 / 100 / 75 | as control |
| fault_nan_input, fault_nan_term, fault_nan_gradient | fault | not scored (the run must stop) | none: detection is judged |
| horizons_5_60_240 | configuration | 450 / 37 / 9 | 9a and 9b on h0 (450); 9b only on h1 (37); h2 (9) none |
| slow_periods_proxy_lr5, slow_periods_proxy_lr1 | configuration | 150 / 100 / 75 | as control |

Source: the v2 file's `expected_n_eff` table (training windows 360 and scored windows 1500 for the default cases;
1800 and 2250 for the wide-horizon case, which uses MAX_SEQUENCE_COUNT 9000 and N_FOLDS 3). Each cell's recorded
`<h>/n_eff` is compared with this table in the REPORT; a difference is reported (the gates then judge different horizons
than stated here), not silently accepted.

**Long-memory cases are out of scope.** `long_memory_1440_lr5/lr1`, `long_memory_10080_lr5/lr1` and
`long_memory_scale_norm` are defined in the harness with `runnable=False` ("GPU, NT-051" in the code, but the window model
caps the learned periods at LOOKBACK and 1,440-bar periods need a 1,440-bar window or the series engine of research track
R6; the per-channel-scale variant has no Config switch, B_model_indicators.md item 5). They cannot be run on the D-047
default by any launch, so the REPORT lists them as "not run, with the reason" and the lead files the follow-up (R6). The
CPU `slow_periods_proxy_lr5/lr1` cases (periods starting at 50) stand in for the long-memory family and the REPORT says
so; it is not evidence about 1,440 or 10,080-bar periods.

What the verdict is about: the screen-size reference profile trains 360 windows for 3 epochs, 6 steps, with 2 folds. It
tests the harness's invariants (finite losses and gradients, attribution of faults, variance-head sanity, coverage) on
the D-047 default model, not the behaviour of a long training run; the REPORT says so in its first paragraph.

## Expected outcomes, stated before the run

The harness's verdict is the harness's verdict; this section only fixes how an outcome is classified in the REPORT.

1. **Expected PASS** (a FAIL is a real finding): control, scale_x0.1, scale_x10, vol_x0.1, vol_x10, fuzz_large_price,
   fuzz_small_price, fuzz_constant (flat training windows 21 of 360 = 6%, NT-187), and the three fault cases (each must
   stop the run with `UnstableTrainingError`; `fault_nan_term` must name `crps_loss`; `fault_nan_gradient` has no term to
   name, the documented limit of NT-038). Healthy scaled NLL is 1.4-2.0 against the limit 8 (NT-187's 288 horizon-scores).
2. **Expected, low confidence: PASS** for horizons_5_60_240 and the two slow_periods_proxy cases. They exist to find
   failing regions; a FAIL is recorded as a valid outcome and becomes a region for `Config.validate` and a backlog item,
   not an error of the harness.
3. **fuzz_jumps: the outcome cannot be predicted with confidence.** On the tiny profile the case already scores scaled NLL
   7.19 / 6.80 / 6.27 against the limit 8 and its constant baseline is about 6.9, near the "absurd baseline" bound 8 (NT-190,
   QA of NT-187). On reference n_eff is 150/100/75, so the checks ARE judged. The spikes (x4, x0.25) and the x2 level jump
   inflate every sigma against the test block; the constant baseline is equally inflated. Three outcomes are possible:
   (a) PASS; (b) the baseline's scaled NLL passes the bound 8, the over-baseline checks (9a, 9c) report "not evaluated"
   and only the absolute scaled NLL (9d) is judged, which then passes or fails; (c) FAIL on `max_variance_nll_scaled` 8.0.
   My forecast, as an estimate: (b) or (c) is about as likely as (a).
   - **By-design FAIL (recorded as "FAIL, by design of the case", not as a defect of the model)** when ALL hold: the only
     failed check is `max_variance_nll_scaled` (9d); the same seed's constant baseline is also above 5 in scaled units
     (so the inflation is the data's, not the head's); `control` passes on the same seed; losses and gradients are finite
     (no `UnstableTrainingError`, no non-finite steps).
   - **Real finding** when ANY other check fails on fuzz_jumps (a non-finite or exploding loss, non-finite gradient
     steps, a term gradient share above its limit, a clipped share above its limit, coverage below 0.25, an
     `UnstableTrainingError`), or when 9d fails while the baseline's scaled NLL is below 5 (the head is worse than the
     data's own inflation explains).
   - Either way the verdict is stored as the harness gave it. The thresholds are not changed after results (D-064): a v3 file
     is the owner's question (NT-190 (4)).
4. **Any other failure** is a real finding and is recorded with the blamed loss term and the failing region (NT-038 (5)).
5. **Known blind spots** (stated so that a PASS is read correctly): v2 does not judge variance on the h2 horizon of the
   default reference cases except by the 9b checks (n_eff 75); an h2-only sigma error of about x0.3 is not judged (NT-190
   (2)); a NaN or inf variance head under v2 is not pinned by a test (NT-190 (3)); the n_eff gates 100 and 30 are extrapolations
   below the stored minimum 135.

## Execution

- **One launch per case**, 3 seeds inside a launch (`neural-trade stability --profile reference --thresholds v2
  --cases <case> --seeds 0,1,2 --store D:/nt/neural_trade/runs`): 15 launches, 15 reports. The harness has no per-cell
  hook, and `run_harness` refuses an existing report directory, so a per-cell GPU-free check is not possible without an
  `src/` change; the per-launch check is the closest. (If the lead prefers a per-cell check, it is an implementer item
  first; the launches are 3 cells of a few minutes.)
- **GPU-free check before every launch** (RUNBOOK "GPU rules": `nvidia-smi dmon -s um -c 10`; busy if fb above 2000 MB
  or median sm above 30%; the desktop alone shows median sm about 40% and the check's fb is the decisive signal, NT-035;
  as measured 2026-10-07 the card already shows fb about 10,300 MB from another process, so a launch waits). A busy GPU:
  wait and re-check every 60 s, for about 2 hours at most; then stop and report to the lead for parking. A tactical run
  (D-063, at most 2 minutes) on the GPU: wait for it to end. The check, its UTC time, its samples and its verdict are
  appended to `runs/experiments/stability_ref_v1/gpu_free_checks.log` (committed with the report). Each launch also
  writes a 1 s nvidia-smi sampler file for the launch (`samples_<case>.csv`).
- **Order** (the core first, see the budget): control, fuzz_jumps (the pilot: 6 cells, which also measure the real
  per-cell wall time), then fuzz_constant, horizons_5_60_240, scale_x0.1, scale_x10, vol_x0.1, vol_x10, fuzz_large_price,
  fuzz_small_price, slow_periods_proxy_lr5, slow_periods_proxy_lr1, fault_nan_input, fault_nan_term, fault_nan_gradient.
- One GPU job at a time. A cell the engine marks failed for a reason that is not a verdict (crash, out of memory) is
  re-run once with `--retry-failed`; a verdict FAIL is never re-run.
- Logs: each launch's console to `runs/experiments/stability_ref_v1/logs/<case>.log`.

## GPU-time estimate (an estimate, labelled; nothing measured on this profile yet)

- Step time: 0.1735 s/step (D-047, measured with the probe off) x 6 steps (360 windows, batch 256, 3 epochs) is about
  1 s a cell; the horizons case (1800 windows, 8 steps x 3) about 4 s; the per-term probe (about 10% of the step time)
  adds nothing visible. So the cell time is the per-cell fixed cost: the data windows, calibration, scoring of 1500
  windows, the backtest and the eval report.
- Fixed cost per scored cell: from `runs/scenarios/capacity_v1/` meta (the micro layout: wall 280-612 s per cell,
  median 444 s; train median 370 s, score median 80 s over its 15 finished cells) the lead's range is **250-490 s**;
  this profile's training is far shorter than those cells', so the true value is probably at the low end or below (not
  measured). A fault cell stops at the fault (no scoring): taken as 60 s (an estimate).
- **Full set** (45 cells: 36 scored cells + 9 fault cells): 36 x (250 / 370 / 490) s + 9 x 60 s =
  **2.7 / 3.9 / 5.1 GPU-hours**. Above the 3-hour limit at the middle and the high end: the full set needs the owner
  unless the pilot shows it is not (the pilot's measured wall time replaces these numbers; a re-estimate does not remove
  the limit by itself: the owner is asked with the measured number).
- **Proposed cut (the core, fits under 3 hours at every point of the range): 23 cells.**
  - 3 seeds on the cases the review named: control, fuzz_jumps, fuzz_constant, horizons_5_60_240 (12 cells);
  - 1 seed (seed 0) on scale_x0.1, scale_x10, vol_x0.1, vol_x10, fuzz_large_price, fuzz_small_price,
    slow_periods_proxy_lr5, slow_periods_proxy_lr1 (8 cells);
  - 1 seed (seed 0) on the three fault cases (3 cells, 60 s each).
  - 20 scored cells x (250 / 370 / 490) s + 3 x 60 s = **1.5 / 2.1 / 2.8 GPU-hours**.
  - A case run with fewer than 3 seeds is reported as "k of 3 seeds, cut", not as a pre-registered PASS; the MVP-3 exit
    ("every setup passes the harness", 3 seeds) is NOT met by the core alone.
- **Optional rest (needs the owner, only if the core is clean or the owner wants the full 3-seed record)**: seeds 1 and 2
  of the 8 single-seed scored cases (16 cells) and of the 3 fault cases (6 cells): 16 x (250 / 370 / 490) s + 6 x 60 s =
  **1.2 / 1.8 / 2.4 GPU-hours** (total with the core: the full-set range above).
- If the core's cumulative GPU time passes 3 hours before it finishes, the run stops there and reports (the limit counts
  every launch together).

## Guard-rails

- No `src/` or `tests/` edits; thresholds not changed; no choice made on any result; no run directory deleted or
  overwritten (a new `--store` report directory per launch, a new harness id).
- A FAIL is a valid outcome. Each fix is a new backlog item; the REPORT names, per failed case, the failing check, the
  blamed loss term (from the per-term probe) and the failing region written for `Config.validate`.
- The REPORT lists every run id, the thresholds sha256 and the code sha, the GPU-free check lines, the measured per-cell wall
  time against the estimate above, and the GPU time used against the stated budget.
