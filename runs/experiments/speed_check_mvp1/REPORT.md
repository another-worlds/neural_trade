# NT-075 speed check: 1aeff1c (A) against 426de4f (B, the MVP-1 head)

Experimenter run on 2026-10-06. The lead saved this report from the experimenter's hand-back, because the
subagent could not write `*.md` files.

- **Pre-registration:** the SPEC was committed before any run, in `SPEC.md` (aeb7c90).
- **Analysis:** `analyse.py` produced `analysis.txt`.

## Setup

- **Code:** each side ran from its own detached worktree, with `PYTHONPATH` set to that side's `src`. Every run id
  carries its sha.
- **Command:** `python -m neural_trade.cli train --config configs/default.yaml --csv <bundled 30-day file> --no-baselines`.
  - Defaults: 20 epochs, seed 42, calibration on.
  - All runs completed 20 epochs; none stopped early.
  - The two default.yaml files differ only in a comment.
- **Order:** interleaved A B A B A B. The SPEC rule then gave "inconclusive", so it allowed one more A B pair:
  8 runs in all, n = 4 per side.
- **Checks during the runs:** a GPU-free check before each run (`*.gpucheck.txt`), and host CPU sampled with
  psutil (`*.cpu.csv`).
- **Decision rule (SPEC):** the primary metric is sec_per_step, as a ratio of medians r = B / A.
  - r <= 1.05: no regression.
  - r > 1.05 and the ranges are disjoint: a regression.
  - r > 1.05 and the ranges overlap: inconclusive, and one more pair runs.

## Results

| run | sec_per_step | median epoch s (epochs >= 1) | elapsed s | host CPU mean / max % |
|---|---|---|---|---|
| A r1 | 0.1997 | 12.82 | 307.9 | 41.0 / 75.5 |
| B r1 | 0.1209 | 16.77 | 359.2 | 60.9 / 86.0 |
| A r2 | 0.1348 | 14.93 | 324.5 | 54.4 / 81.3 |
| B r2 | 0.1541 | 14.02 | 324.6 | 54.3 / 85.8 |
| A r3 | 0.1026 | 12.85 | 324.7 | 50.1 / 83.8 |
| B r3 | 0.1418 | 15.44 | 330.1 | 55.9 / 84.0 |
| A r4 | 0.1146 | 14.84 | 322.2 | 51.8 / 79.3 |
| B r4 | 0.1113 | 13.30 | 295.9 | 32.4 / 50.5 |

- **After 3 runs per side:** the sec_per_step ratio was 1.051 and the epoch-time ratio 1.202, with overlapping
  ranges. That is inconclusive, so the extra pair ran.
- **sec_per_step, n = 4 per side:**
  - A: median 0.1247, range 0.1026-0.1997.
  - B: median 0.1313, range 0.1113-0.1541.
  - Ratio 1.053; the ranges overlap.
- **Median epoch time, n = 4 per side:**
  - A: 13.85 s (12.82-14.93).
  - B: 14.73 s (13.30-16.77).
  - Ratio 1.064; the ranges overlap.

## Verdict (per SPEC): INCONCLUSIVE

- **The gap is small next to the noise.** The medians sit 5-6% apart, with B slower, which is the same direction
  as the notebooks' +8%. Run-to-run noise is far larger: A alone ranges over 2x.
- **No call either way.** A regression can be neither claimed nor excluded at the 5% level, and the SPEC allows no
  further runs.
- **The primary metric is one epoch.** `status.json` sec_per_step, the SPEC's primary metric, is the LAST epoch's
  step time only (telemetry/epoch_logger.py:127, :136): one noisy epoch out of 20. A r1's last epoch is 0.1997,
  against a median of 0.1075 over its epochs >= 1. QA's sensitivity check used each run's median over epochs >= 1
  instead: A 0.1160, B 0.1235, r 1.065, ranges overlap. The verdict is still inconclusive. (Added after QA,
  2026-10-06; NT-163.)
- **Likely source of the noise.** The host's CPU was 32-61% busy during every run, from other agents' CPU work and
  the training process itself. The input pipeline and launch overhead are CPU-side.
- **One borderline run.** B r1 (60.9%) sits at the SPEC's 60% repeat threshold. It was not repeated, because the
  sampled utilisation includes the training process.

## Follow-up (lead)

If the +5% still matters under D-018, measure on a quiet machine with GPU-side per-step timing. That means a
fixed-step micro-benchmark of the training step, taking the median of many steps, not the whole-run
sec_per_step. Bisect 1aeff1c..426de4f on the step path only if that micro-benchmark shows a clear gap.

## Budget and evidence

- **GPU time:** 8 runs, about 45 minutes (0.75 GPU-hours), against a budget of 1 GPU-hour.
- **Evidence:** in this directory: `launch.py`, `analyse.py`, `analysis.txt`, `progress.txt`, `*.cpu.csv`,
  `*.gpucheck.txt`, stdout and stderr. The 8 run directories are under `runs/`, with suffixes `-A_1aeff1c_r<k>`
  and `-B_426de4f_r<k>`.
