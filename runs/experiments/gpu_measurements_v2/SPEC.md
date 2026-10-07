# SPEC: GPU parallel-trials record on the D-047 default (NT-173)

Status: pre-registered before any GPU time. Does not change after results exist. An infra measurement,
not an "A beats B" verdict (no model comparison, no quality number), so D-025's judgement-fold rule does
not apply. Re-measures Part A of `runs/experiments/gpu_measurements_v1/` (NT-035, 2026-09-29), which was
taken on the pre-D-047 close-only model (0.1066 s/step; D-047 default: 0.1735 s/step, 1.63x).

## Pinned code

- Branch `nt-173` off `remediation/plan` 730f87c; worktree `D:/nt/nt_wt_173`. This SPEC's own commit sha is
  recorded in the REPORT (the SPEC is committed before any run).
- Every job: `PYTHONPATH=D:/nt/nt_wt_173/src`, cwd `D:/nt/nt_wt_173`, so the relative `CSV_PATH` resolves there.
- Outputs: `runs/experiments/gpu_measurements_v2/` in the worktree, committed on `nt-173` (light files only).
- Harness: v1's `harness_concurrent.py` copied unchanged into this folder (it launches N
  `neural-trade train` processes, one run directory each, no shared store). A launcher wrapper
  (`run_v2.sh`) adds the GPU-free check, a per-run 1 s nvidia-smi sampler to a file, and the crash-safe stop.

## Questions

1. What are a single D-047 process's peak `fb` and median/peak `sm`?
2. For N = 2, 3 (and 4 only under the rule below) concurrent processes: per-process `sec_per_step`,
   aggregate throughput against N = 1, peak `fb`, peak `sm`, crashes.
3. Which N does the decision rule allow, and what is the record the sweep reads?

## Fixed setup (every condition)

- `configs/default.yaml` as of 730f87c: BTC/USDT 1-minute `binance_btcusdt_1min_ccxt.csv`, the D-047
  default input layout (OHLCV, all 14 indicator families, 3 instances each), LOOKBACK 60,
  HORIZON_STEPS [10, 15, 20], BATCH_SIZE 256, MAX_SEQUENCE_COUNT 53280 (the default; fold -3 gives about
  62 steps per epoch, as in v1; re-checked from the first N = 1 run's `status.json` steps), `FOLD_INDEX=-3`,
  `--epochs 3`, `--no-calibrate --no-baselines` (isolate the step loop, as v1), seeds `9001..` per v1.
- `sec_per_step` = last epoch's `epoch_seconds / steps` from each run's `status.json`.
- 3 repeats of N = 1, 2, 3 (9 batches; N = 4 below). A repeat is one batch of N simultaneous processes.

## Procedure

1. Before EVERY launch: RUNBOOK GPU-free check (`nvidia-smi dmon -s um -c 10`), recorded with a UTC
   timestamp in `gpu_free_checks.log`. Free = median sm <= 30 and max fb <= 2000 MB. The desktop alone
   shows median sm about 40% in free checks (NT-035's note) and about 1000-1100 MB fb; the stated sm
   baseline is measured by the first check, and busy is judged by fb. If the GPU is busy (fb above 2000 MB
   or a tactical run, D-063, still on it), wait and re-check every 60 s for up to 50 minutes, then stop
   and report; never run alongside another trainer (it ruins a throughput measurement). The owner's other
   project (Docker/WSL) is never touched.
2. During each batch a sampler writes `nvidia-smi --query-gpu=utilization.gpu,memory.used -l 1` to
   `samples_n<N>_r<R>.csv`. Peak fb, mean and peak sm come from it. A batch that overlaps a foreign trainer
   (fb above the desktop baseline plus the expected own-process footprint cannot be told per process under
   WDDM, so: fb at the batch start above 2000 MB) is discarded and repeated.
3. N = 1 (x3), N = 2 (x3), N = 3 (x3), in that order, one batch at a time.
4. N = 4 (x1, a crash-safe launcher: each child's exit code recorded, a crash is a result, not an abort)
   only if N = 3's peak fb leaves at least 2 GB headroom on the 12282 MB card (peak fb <= 10282 MB).
   Otherwise N = 4 is recorded as not attempted, with the number.
5. The last 4 (or fewer) results are written into `parallel_n.json`.

## Metrics

- per-process `sec_per_step` (mean over processes and repeats), aggregate throughput =
  sum over processes of 1/`sec_per_step`, ratio to N = 1's, the same for wall time per batch;
- peak and mean `fb`, mean and peak `sm`, crash count (non-zero exit code), each averaged over repeats
  (peak fb: the maximum over repeats is also reported and is the one the rule uses).

## Decision rule (fixed now)

`allowed_n` = the largest N in {1, 2, 3, 4} such that ALL hold:
(a) aggregate throughput of N exceeds N = 1's by at least 15% (mean over repeats);
(b) peak fb (maximum over repeats) stays at least 1.5 GB (1536 MB) below the card's total (12282 MB),
    i.e. <= 10746 MB;
(c) no process crashed in any repeat of N.
If none above 1 qualifies, `allowed_n` = 1. If a larger N fails but a smaller one qualifies, the smaller wins.
The sweep adds its own 1024 MB margin (`WATCH_FB_MARGIN_MB`) to the recorded `peak_fb_mb`, so `utilization[N]`
holds the measured peak, not peak + 1024 (the lead's brief says "peak + 1024": the code applies it; recording
the sum would count it twice).

## Output record

`runs/experiments/gpu_measurements_v2/parallel_n.json`, the keys sweep.py reads: `allowed_n`,
`utilization` {N: mean_sm_pct, peak_sm_pct, mean_fb_mb, peak_fb_mb}, `setup` (a text that contains
`OHLCV`, `LOOKBACK <n>`, `BATCH_SIZE <n>`, `HORIZON_STEPS [..]`, so `record_setup_warnings` stays silent),
`measured_utc`, plus `refused_n`, `table`, `desktop_baseline`. The v1 file is not touched. The one-line
change to read v2 (not made here, no `src/` edit): `DEFAULT_PARALLEL_RECORD` in
`src/neural_trade/experiments/sweep.py:89` and the `--parallel-record` default in `src/neural_trade/cli.py:577`.

## GPU-time estimate

Per process about 3 x 62 steps x 0.1735 s = 32 s of stepping plus startup and data windows, about 50 s
when alone; v1 batches took 40 to 100 s. 9 batches at about 1.5 min on average plus the N = 4 batch plus
GPU-free checks (10 s each): about 20 minutes of wall time, about 0.35 GPU-hours counting every concurrent
process as GPU time. Under the 0.5 GPU-hour budget.

## Guard-rails

No `src/` or `tests/` edits; a new run directory per batch (no overwrite); no run deleted.
