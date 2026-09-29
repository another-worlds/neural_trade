# SPEC: GPU measurements (NT-035) — concurrent-runs throughput and deterministic-mode speed

Status: pre-registered before any GPU time. Does not change after results exist (OPERATING_MODEL).
This is an infra measurement, not an "A beats B" quality verdict (no leaderboard ranking, no model
comparison): it measures wall-clock speed and reproducibility of the training process itself, so
D-025's judgement-fold rule (fold -1 x >=5 seeds, held out from choices) does not apply here. It
still follows the pre-registered-study format the backlog item names (`runs/experiments/<name>/`,
SPEC before GPU time, REPORT against it), per the experimenter agent file.

## Pinned code

- Spec commit (this file, on `remediation/plan`): `8f35053be2d185c5270a0135c93b1b17f2145f50`
- Worktree: `D:/nt_exp_gpu_measurements_v1` (detached at the spec commit sha above)
- Every job launches with `PYTHONPATH=D:/nt_exp_gpu_measurements_v1/src`, cwd
  `D:/nt_exp_gpu_measurements_v1`, so `CSV_PATH` (relative, `binance_btcusdt_1min_ccxt.csv`)
  resolves inside the pinned tree.

## Hypothesis / questions (NT-035 acceptance criteria (1) and (4))

1. Does running several training processes at once on this GPU raise total training throughput
   (steps/s summed across processes), and what is the largest N (1-4) the pre-registered rule below
   picks?
2. What does the opt-in deterministic mode (`seed_everything(seed, deterministic=True)`, which calls
   `tf.config.experimental.enable_op_determinism()`) cost in `sec_per_step`, relative to the same
   config without it, given that `import neural_trade` already sets `TF_DETERMINISTIC_OPS=1` for
   every run (so that env var alone is not the "without" arm)?
3. Do two same-seed deterministic-mode GPU runs give identical val_loss per epoch?
4. Does any op used by this model raise for lack of a deterministic GPU kernel under
   `enable_op_determinism()`?

## Fixed setup (identical across every condition; the reference setup)

- `base_config: configs/default.yaml` (BTC/USDT 1-minute, `binance_btcusdt_1min_ccxt.csv`,
  LOOKBACK 60, HORIZON_STEPS [10, 15, 20], BATCH_SIZE 256).
- `--set FOLD_INDEX=-3` (a dev fold; picked only because its train block gives a convenient,
  measured batches/epoch below — no quality is judged on it, so it is not a "choice fold" or
  "judgement fold" in the D-025 sense).
- Measured on this setup (CPU, `DataProcessor.prepare_datasets`, 2026-09-29): 15,741 train
  sequences -> **62 batches (steps) per epoch** at BATCH_SIZE 256.
- `EPOCHS = 3` (fixed) -> **186 steps per process** (fixed), identical for every N, every repeat and
  both determinism arms.
- `--no-calibrate` (skip the loss-weight warm-up/sampling pass before epoch 1) and `--no-baselines`
  (skip the trailing-return baseline fit): both are fixed-cost, per-process overheads unrelated to
  the per-step training loop; turning them off isolates `sec_per_step` and is applied identically to
  every condition, so it cannot bias one condition against another. Post-hoc calibration
  (`fit_calibration=True`) and `save_artifacts=True` stay on (small, fixed, identical costs) because
  `neural-trade train` does not expose switches for them.
- `sec_per_step` is read from each run's `status.json` (`telemetry/epoch_logger.py`): the *last*
  completed epoch's `epoch_seconds / steps`, so it already excludes the first epoch's one-off graph
  build / XLA compile cost. `elapsed_seconds` (status.json) is wall time from the start of `fit()`
  to the last status write, covering all 3 epochs; used for the batch's total steps/s.
- TF32 matmul: `tf.config.experimental.tensor_float_32_execution_enabled()` is `True` by default in
  this TF 2.10 env (verified 2026-09-29, CPU import). Recorded, not changed, for both parts.
- Op determinism hint: `TF_DETERMINISTIC_OPS` is set to `"1"` by `import neural_trade`
  (`setdefault`, `src/neural_trade/__init__.py:26`) for every process **except** the determinism
  test's "without" arm, which sets it to `"0"` in the child process's environment *before* Python
  starts (so `setdefault` leaves it at `"0"`).

## Part A: concurrent-runs throughput (criterion 1)

- **Conditions:** N = 1, 2, 3, 4 concurrent `neural-trade train` processes, same fixed config/epochs
  above, launched together (Python `subprocess.Popen`, no `--parallel` flag exists yet — NT-030 is
  what will read this item's result). Each process writes into its own run directory (`--runs-dir
  runs/experiments/gpu_measurements_v1/throughput`, `--name n<N>_r<repeat>_p<idx>`) with a unique
  `--seed` (a running counter, seed value irrelevant to a speed measurement) — this avoids the
  engine's cell-store locking problem entirely by not using the scenario store for this measurement
  (facts given with this item: "the engine has no cross-process cell locking yet, so give each
  concurrent process its own scenario name or store"; here every process gets its own run directory
  and no shared index).
- **Repeats:** >= 3 repeats per N (3 used). One "repeat" = one batch of N processes launched
  together and waited on to completion.
- **Metrics per repeat:** total steps/s = (N x 186) / wall_seconds_of_the_batch; per-process
  `sec_per_step` from each process's `status.json`; peak `fb` (MB) and `sm` (%) sampled by
  `nvidia-smi dmon -s um -c <n>` running for the duration of the batch.
- **Pre-registered rule that picks N:** rank N = 1..4 by the **mean total steps/s over its 3
  repeats**. Pick the largest N whose mean total steps/s is at least 1.15x the mean total steps/s of
  N=1 (a >=15% real gain over running one at a time) AND whose mean per-process `sec_per_step` has
  not grown past 2x the N=1 mean (guards against a false throughput "win" that is actually N slow,
  contended processes each doing useless work at a worse rate than the wall clock suggests). If no
  N > 1 clears both bars, the picked N is 1. Ties broken by the smaller N (simplicity, less GPU
  memory).

## Part B: deterministic-mode speed and reproducibility (criterion 4)

- **Conditions:** `det_on` (before `train_and_evaluate`, the harness calls
  `seed_everything(SEED, deterministic=True)` itself, since neither `neural-trade train` nor
  `train_and_evaluate` exposes the `deterministic=` kwarg; `train_and_evaluate`'s own internal
  `seed_everything(cfg.SEED)` call re-seeds afterwards but cannot un-call
  `enable_op_determinism()`, which is a one-way process-global switch in TF 2.10) vs. `det_off`
  (`TF_DETERMINISTIC_OPS=0` set before the child process starts; no `enable_op_determinism()` call;
  this is the true "without" arm, since the hint alone already makes
  `_pywrap_determinism.is_enabled()` report `True` per the window-free research finding cited with
  this item).
- Both arms use the **same fixed `SEED = 777`**, so that `det_on`'s 3 repeats test reproducibility
  (criterion 3) directly against each other, and `det_off`'s 3 repeats show the same-seed run-to-run
  spread without determinism.
- **Repeats:** 3 per arm (>= 3, as the item asks), sequential (one GPU job at a time; this is not a
  sweep, so RUNBOOK's parallel-trial allowance does not apply).
- **Metrics:** `sec_per_step` (status.json, last epoch) per run, mean and spread per arm; the ratio
  det_on / det_off; whether the 3 `det_on` runs' `training_log.csv` `val_loss` column (all 3 epochs)
  are bit-identical; every stderr line naming an op without a deterministic GPU kernel (grep for
  "eterministic" and "OpKernel" in each run's captured stderr), collected across both arms.
- **Verdict rule:** report the measured ratio and reproducibility as findings (no pass/fail
  threshold is pre-registered here — NT-035's job is to measure and report, not to gate a decision;
  the consuming decision, "use deterministic mode for comparison studies", is D-025's, already made,
  conditional on this speed test existing).

## GPU-time estimate (before any GPU time; NT-030's formula)

Baseline `sec_per_step` (measured, same reference setup, real run
`runs/20260924T182915Z-1aeff1c-dirty-af67ee43/status.json`): **0.09840 s/step**.

- Total steps: Part A = 30 processes x 186 steps = 5,580 steps. Part B = 6 processes x 186 steps =
  1,116 steps. Total = 6,696 steps.
- Conservative sequential-equivalent bound (assumes, worst case, that contention makes every step
  4x slower than baseline, and adds a flat 25 s/process fixed overhead for interpreter/CUDA
  start-up, data load and the post-hoc calibration/eval pass that stay on):
  `6,696 x 0.0984 x 4 + 36 x 25 s = 2,635 s + 900 s = 3,535 s ≈ 58.9 minutes ≈ 1.0 GPU-hour.`
- This is an upper bound: Part A's N=2..4 batches run their processes concurrently (wall time per
  batch, not summed per process), so the real elapsed GPU time will be lower. **Estimate: about 1.0
  GPU-hour, well under the 3-hour cap;** no owner sign-off needed.

## Guard-rails

- Before every batch/run: the RUNBOOK GPU-free check (`nvidia-smi dmon -s um -c 10`); if busy (fb >
  2000 MB or median sm > 30%), do not start, record it, re-check later (RUNBOOK "GPU rules").
- Never run two GPU jobs outside this item's own approved concurrency (Part A's own N processes;
  Part B strictly sequential) at the same time as anything else.
- No run directory, study or data file is deleted; every run gets a new, unique directory.
- Weights (`weights.h5`, `artifacts/`) stay untracked (heavy); only light files
  (`status.json`, `config.yaml`, `meta.json`, `training_log.csv`, `eval_report_*`, `env.json`) are
  committed for the runs the report cites, via `scripts/check_run_evidence.py --list-untracked`.

## Result file for NT-030 `--parallel`

`runs/experiments/gpu_measurements_v1/parallel_n.json`: `{"allowed_n": <int>, "utilization": {"<N>":
{"mean_sm_pct": <float>, "mean_fb_mb": <float>}, ...}, "spec": "runs/experiments/gpu_measurements_v1/SPEC.md",
"measured_utc": "<iso>"}`. NT-030 (4) reads `allowed_n` to gate `--parallel` above 1, and the
per-N utilisation numbers as the level past which the sweep must stop launching new batches (someone
else is on the GPU).

## Harness scripts (committed alongside this SPEC, not in `src/` or `tests/`)

- `harness_concurrent.py`: launches N `neural-trade train` subprocesses per repeat via the CLI
  (no source change), waits, samples `nvidia-smi dmon`, writes one JSON summary per repeat.
- `harness_train_once.py`: calls `train_and_evaluate` and, for `det_on`, `seed_everything(seed,
  deterministic=True)` directly (the public API `neural-trade train` does not expose a determinism
  switch); used only for Part B.
