# GPU benchmark: old against new code, batch 256 to 2048 (2026-09-29)

For D-040 (one 360-day training run). An experimenter measured; the lead checked the numbers against the raw
JSONs and wrote the estimates.

## Setup

- **Code versions:**
  - old = `master` 7002a71, the single-file layout (`model.py`);
  - new = `remediation/plan` f39593a.
- **Hardware:** RTX 4070 Ti, one job at a time, 3 epochs each.
- **Data:** the bundled 30-day file. The cost of one training step does not depend on the dataset size; only the
  number of steps does.
- **"Steady"** means epochs 2 and 3. Epoch 1 includes graph tracing.
- **Evidence:**
  - [results/](results/): per-run JSON and nvidia-smi dmon traces;
  - [scripts/](scripts/): the drivers. Paths inside them point to the experimenter's scratch folder `D:/nt_bench_runs`.
- **GPU time:** about 30 minutes.

## Measured

| code | batch | steps/epoch | steady s/step | train s/epoch | val s/epoch | peak GPU memory (TF) | median GPU sm % |
|---|---|---|---|---|---|---|---|
| new | 256 | 119 | 0.104 | 12.4 | 0.73 | 0.8 GB | 44 |
| new | 512 | 60 | 0.134 | 8.1 | 0.45 | 1.5 GB | 58 |
| new | 1024 | 30 | 0.198 | 6.0 | 0.32 | 2.9 GB | 70 |
| new | **2048** | 15 | 0.325 | **4.9** | 0.28 | 5.4 GB | **80** |
| old | 256 | 118 | 2.114 | 248.8 | 11.0 | 0.7 GB | 15 |
| old | 576 (its default) | 53 | 1.907 | 101.5 | 4.8 | 1.3 GB | 19 |

**Speed per step:**

- **At the same batch size (256) and the same step count:** the new code is **20.4× faster** (0.104 against 2.114 s
  per step). The experimenter's summary said "8x"; that was an arithmetic slip, and the raw numbers give 20×.
- **Each at its best setting:** the new code at 2048 is **20.7× faster** than the old code at its default 576
  (4.9 against 101.5 s per epoch).

**Caveat on the GPU-free check:** the check before the runs read a median sm of 43%, which is over RUNBOOK's 30%
"busy" line. The experimenter launched anyway, and gave these reasons:

- no compute process was listed;
- the Docker container list was empty;
- the GPU sat at idle clocks.

If that load was real, it inflates mainly the old code's noisy numbers (its two steady epochs at batch 256 differ by
45%) and the new code's batch-256 figure. The new code's 0.104 s per step at 256 matches the 0.103-0.107 measured
in earlier runs.

## Why the old code is slow (file:line in master 7002a71)

1. **Sequential moving averages.** `math_helpers.py:227-258` `ewma_sequence` is a `tf.scan(parallel_iterations=1)`
   over 59 time steps. `LearnableIndicators.call` calls it 24 times per forward pass (`model.py:1071-1158`), which
   makes about 1,400 sequential GPU kernel steps per batch, plus the backward pass. The new code computes the
   same 24 averages as two batched einsums (`utils/math.py:181-241`, `learnable_indicators.py:105-122`). This is
   the main difference, and it is why the old code's time per step barely changes with batch size: it waits on
   sequential latency, not compute.
2. **Metrics on every step.** The old `train_step` computes the direction metrics (twice), the trend margins and
   29 loss scalars on every step (`model.py:1934-2040`). The new code does this every `TRAIN_METRICS_EVERY` = 10
   steps (D-010).
3. **A Python loop after every step** clips the indicator periods with `var.assign` (`model.py:1917-1932`).
4. **Eager loss-weight calibration:** 512 batches in eager mode with a `float()` per batch (`model.py:750-776`).
   This makes the old code's 2.5-3 minutes of setup.
5. **A trap for a long run:** the old code keeps only the newest `MAX_SEQUENCE_COUNT = 2,880` windows by default
   (`model.py:48`). Unless that is overridden, it trains on 2 days whatever the file size.

## Why batch 2048 is the right maximum

- **A step costs far less than 8× more for 8× the batch** (0.104 to 0.325 s). So the time per window keeps
  falling up to 2048.
- **2048 is `Config.validate`'s upper limit.**
- **The GPU is close to saturated there:** 80% median sm, against 44% at 256, where the step is kernel-launch-bound
  (D-010).
- **Going higher would gain little:** from 1024 to 2048 the time per window fell only 18%. It would also need a
  Config change and about 11 GB of the 12 GB card.
- **Quality at 2048 was not measured.** With 3 epochs the benchmark cannot say whether batch 2048 at the same
  learning rate learns as well. The long run's logs will show its learning curve.

## Estimates for the 360-day run (lead's extrapolation, not measured)

The run trains on 518,428 windows and validates on 43,200 each epoch. Train time is the step count times the
steady time per step; validation time scales from the table.

| code | batch | steps/epoch | s/epoch | 20 epochs | 40 epochs |
|---|---|---|---|---|---|
| new | **2048** | 254 | about 87 | about 29 min | **about 58 min** |
| new | 1024 | 507 | about 105 | about 35 min | about 70 min |
| new | 256 | 2,026 | about 220 | about 73 min | about 2.4 h |
| old | 576 | 901 | about 1,760 (29 min) | about 9.8 h | about 19.5 h |
| old | 256 | 2,026 | about 4,350 (73 min) | about 24 h | about 48 h |

**Fixed costs:**

- data load and windowing of the 4.6M-bar file: 31 s, measured on CPU, 4.1 GB RAM;
- tracing: about 20 s;
- the loss-weight calibration pass and the scoring of the 32-day out-of-sample block: a few minutes, estimated.

**What is not in the table:** the cost of the full shuffle (NT-082 measures it). EarlyStopping may end the run
before the last epoch.

**The chosen setup:** batch 2048, 40 epochs (about 10,000 optimiser updates, about 4× today's 7-day run), LR 1e-3
unchanged, full shuffle. This is **about 1 hour on the new code, against about 19.5 hours on the old code at its
default batch for the same epochs.**
