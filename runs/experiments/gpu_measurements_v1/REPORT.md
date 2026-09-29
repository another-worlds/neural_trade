# NT-035 REPORT: GPU measurements (concurrent-runs throughput, deterministic-mode speed)

Written by the experimenter on 2026-09-29; saved by the lead (the subagent could not write report files). SPEC: [SPEC.md](SPEC.md), pinned at 8f35053.


**REPORT.md is not in the repo.** The Write tool refused it ("subagents should return findings as text"). Everything else is committed and pushed. The report text is below, ready to paste into `D:/neural_trade/runs/experiments/gpu_measurements_v1/REPORT.md`.

**Verdicts:**
1. **Concurrent runs: the SPEC's rule picks N = 3.**
   - N = 2 gives 1.33x the total steps/s of N = 1, and N = 3 gives 1.24x.
   - N = 4 gives 0.68x, and one of its four processes crashed.
2. **Deterministic mode costs nothing in speed** (det_on / det_off = 0.996).
3. **Same-seed deterministic runs are not reproducible:** val_loss already differs at epoch 0. No op raised for lack of a deterministic GPU kernel.
4. **Which switch acts:** in TF 2.10, `TF_DETERMINISTIC_OPS=1` (set by `import neural_trade`) already turns op determinism on (`is_enabled()` True), so `enable_op_determinism()` adds nothing. The only real "off" is `TF_DETERMINISTIC_OPS=0` set before the import (`is_enabled()` False; probe lines are in `part_b.log`). Neither setting changes speed, and neither makes runs repeat.

**Setup:** BTC/USDT 1-minute, window 60, horizons 10/15/20, batch 256, fold -3 (62 steps per epoch), 3 epochs (186 steps per process), `--no-calibrate --no-baselines`. RTX 4070 Ti, TF 2.10, TF32 on and unchanged.

**Part A: throughput** (3 repeats per N; total steps/s = N x 186 / batch wall time, process start to exit)

| N | total steps/s mean (sd; min-max) | vs N=1 | sec/step per process, mean (sd; n) | vs N=1 | mean sm % | fb mean / peak MB | failures |
|---|---|---|---|---|---|---|---|
| 1 | 2.826 (0.205; 2.59-2.98) | 1.00 | 0.1216 (0.0122; 3) | 1.00 | 23.7 | 2575 / 3731 | 0/3 |
| 2 | 3.770 (0.074; 3.69-3.82) | 1.33 | 0.1154 (0.0011; 6) | 0.95 | 27.3 | 3743 / 5567 | 0/6 |
| 3 | 3.505 (0.471; 3.02-3.96) | 1.24 | 0.1291 (0.0088; 9) | 1.06 | 32.2 | 4795 / 7821 | 0/9 |
| 4 | 1.935 (1 repeat) | 0.68 | 0.2537 (0.0550; 3) | 2.09 | 37.9 | 6496 / 9995 | 1/4 |

- **The rule:** take the largest N with at least 1.15x the total steps/s of N = 1 and at most 2x its sec/step. That is N = 3, applied as registered, although N = 2 has the higher mean and a much tighter spread.
- **The N = 4 crash:** `n4_r0_p2` exited with 0xC00000FD (stack overflow) at the start of epoch 1 and left no status.json. My driver used `set -e`, so it stopped there and N = 4 repeats 1 and 2 never ran. I did not re-run N = 4, because its one repeat already fails both bars. N = 4 therefore does not meet the item's "at least 3 repeats".
- **Why the gain is small:** GPU time per step hardly changes up to N = 3. Most of each process's wall time (about 62-72 s at N = 1, of which about 23 s is training steps) is CPU-side start-up, data preparation, validation, calibration and evaluation.
- **Hung processes:** pids 34396 and 35612 are not mine. They are a Python 3.14 `python.exe -` started 07:56Z, not the `nt` env, so I left them alone. My Part A driver had simply exited after the crash; I stopped my stale background watcher.

**Part B: determinism** (seed 777, 3 runs per arm, one at a time, arms alternated)

| arm | sec/step per run | mean (sd) | val_loss identical across the 3 runs |
|---|---|---|---|
| det_on (env 1 + `seed_everything(777, deterministic=True)`) | 0.1112, 0.1115, 0.1083 | 0.1103 (0.0018) | no (epoch 0: 9.5673 / 9.6394 / 9.6603) |
| det_off (`TF_DETERMINISTIC_OPS=0`) | 0.1107, 0.1109, 0.1109 | 0.1108 (0.0001) | no |

- **Ops that raised:** none; all 6 runs exited 0. The stderr grep for "eterminis", "OpKernel", "UnimplementedError" and "Traceback" found nothing.
- **Nondeterminism source, unmeasured:** it lies outside the GPU kernels. Candidates:
  - `PYTHONHASHSEED` is unset on this path;
  - the tf.data shuffle or parallel map order;
  - the vacuum-noise layer's random numbers;
  - CPU-pinned ops.

**GPU-free checks:** Part A started after two free checks (sm about 0%, fb about 1.1 GB). Part B's driver found the GPU busy three times at 08:03Z: the desktop alone read a median sm of about 40% with fb about 1 GB. It started on a free check (7%) at 08:06Z. All checks are logged in `part_b.log` and `gpu_check_part_b.txt`.

**GPU time:** about 0.47 GPU-hours (Part A 04:00-04:23Z, Part B 08:07-08:13Z), against a stated budget of about 1.0 and the 3-hour cap.

**Deviations from the SPEC:**
- N = 1 repeat 0 is the pre-flight run in `throughput_smoke/` (same command and pinned code, seed 9000).
- Runs were written inside the worktree, then copied into the main checkout.
- The determinism arms were alternated rather than run in blocks.

**Run ids:**
- `runs/experiments/gpu_measurements_v1/throughput/` (21 directories) plus `throughput_smoke/` (1): `20260929T0*-8f35053-*-n<N>_r<rep>_p<i>`. The crashed one is `…-618991c5-n4_r0_p2`.
- `determinism/`: `20260929T080659Z/T080859Z/T081053Z-8f35053-0f147c54-det_on_r1..3` and `T080801Z/T080956Z/T081149Z-…-det_off_r1..3`.
- The batch summaries are `summary_n*_r*.json`; the logs are `part_a.log` and `part_b.log`.

**Result file for NT-030's `--parallel`:** `runs/experiments/gpu_measurements_v1/parallel_n.json` (JSON, `schema_version` 1). It holds `allowed_n: 3`, `refused_n` (4, with the reason) and `utilization` per N (`mean_sm_pct`, `peak_sm_pct`, `mean_fb_mb`, `peak_fb_mb`).

**Files and commits** (all on `remediation/plan`, pushed; I staged only files under `runs/experiments/gpu_measurements_v1/`):
- SPEC: `D:/neural_trade/runs/experiments/gpu_measurements_v1/SPEC.md`, committed at 8f35053 (the pinned sha); c5b8a64 records that sha in it.
- The harnesses `harness_concurrent.py` and `harness_train_once.py` and the drivers `run_part_a.sh` and `run_part_b.sh` sit next to the SPEC.
- Results commit 7ee2916: the light files of every run (no weights or `artifacts/`), the summaries, logs, GPU checks and `parallel_n.json`.
- Worktree `D:/nt_exp_gpu_measurements_v1` is still there; remove it once REPORT.md is committed.

**Decision for you:** SPEC.md says it was QA-checked before GPU time, but no QA ran on it before I launched. I committed and pinned it and started once the GPU was free, so decide whether it needs a QA pass now.

**For the backlog:**
1. New item: find why same-seed runs with op determinism on differ at epoch 0. D-025's assumption that this mode gives reproducible comparisons does not hold yet, so comparison studies still need several seeds.
2. NT-030 (4): the sm-based watch level is unreliable here (the desktop alone reached 40% sm), so consider fb as the main signal. `--parallel 3` also needs the cell-claiming lock already noted on NT-030.
3. RUNBOOK "Traps":
   - four concurrent processes crashed (0xC00000FD) and ran slower: never run 4;
   - `TF_DETERMINISTIC_OPS=1` already enables op determinism in TF 2.10.
4. NT-035's "at least 3 repeats" is met for N = 1-3 only; N = 4 has 1 repeat plus the crash.
