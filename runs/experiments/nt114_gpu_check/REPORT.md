# NT-114 GPU check: DETERMINISTIC_GRU across separate processes (closes NT-074)

Experimenter run 2026-10-06; this report was saved by the lead from the experimenter's hand-back, because the
subagent could not write `*.md` files. The lead re-read the three runs' `metrics.jsonl` (val_loss below).

## Setup

- Code: f9b60eb (`remediation/plan`, DETERMINISTIC_GRU merged in feab7d8); every process imported
  `D:/nt/nt_wt_114/src`.
- Data and config: BTC/USDT 1-minute bundled file, `configs/default.yaml` at f9b60eb (OHLCV input and 14 families,
  D-047), window 60, horizons 10/15/20, batch 256, fold -3, 3 epochs.
- Seeding: seed 777 with `seed_everything(777, deterministic=True)` and `--override DETERMINISTIC_GRU=true`.
- Runs: 3 separate processes, run one at a time, with a GPU-free check before each (`*.gpucheck.txt`).
- Reference: 1 run with `DETERMINISTIC_GRU=false` (cuDNN). One run only, so it is a reference, not a sample.
- Harness: `harness_train_once.py` (copied from the NT-074 check) and `run_all.sh`.

## (a) val_loss per epoch is bit-equal across the 3 processes: PASS

| epoch | val_loss (r1 = r2 = r3) | float hex |
|---|---|---|
| 0 | 8.898392677307129 | 0x1.1cbfa20000000p+3 |
| 1 | 8.846050262451172 | 0x1.1b12d80000000p+3 |
| 2 | 8.74549388885498 | 0x1.17db160000000p+3 |

- Training loss per epoch is also bit-equal in all three runs: 9.924745559692383, 9.434490203857422,
  9.226221084594727.
- The cuDNN reference differs, as expected: 8.891057014465332, 8.883086204528809, 8.688572883605957.
- No traceback appears in any stderr.

## (b) GPU cost: reported, not gated (D-018, an opt-in path)

| run | DETERMINISTIC_GRU | sec_per_step | epoch 1 / 2 wall (s) | elapsed (s) |
|---|---|---|---|---|
| det_gru_r1 | on | 0.2966 | 18.7 / 18.4 | 100.3 |
| det_gru_r2 | on | 0.3171 | 20.8 / 19.7 | 153.0 |
| det_gru_r3 | on | 0.3372 | 20.2 / 21.0 | 102.0 |
| det_cudnn_ref | off | 0.2106 | 13.0 / 13.1 | 92.4 |

- **On:** median sec_per_step 0.3171 (range 0.2966-0.3372).
- **Off:** 0.2106, one run with no spread.
- **Ratio:** about 1.5x on sec_per_step (1.41-1.60) and on steady epoch time (about 19.8 s against 13.05 s). The
  earlier 1.2-1.4x estimate was low. A 3-epoch run's sec_per_step includes the first epoch's graph build, so treat
  it as an upper bound.
- **Unexplained:** r2's elapsed time was 153 s against about 101 s, although its epoch times are normal; the extra
  time is outside the epochs.

## Meaning

- **NT-114 (4), GPU part: passes.**
- **NT-074 (2):** met on the GPU.
- **D-025 deterministic comparison studies:** possible with DETERMINISTIC_GRU on, at about 1.5x the step cost.
  Budget them with that factor.
- **Not checked:** more than 3 epochs, calibration on, other folds, and the interaction with
  SEEDED_STOCHASTIC_LAYERS.

## Budget and run ids

GPU time was about 12 minutes (0.2 GPU-hours), against a budget of 0.5 GPU-hours.

Run ids (under `runs/`):
- 20261006T061501Z-f9b60eb-995467d5-det_gru_r1
- 20261006T061705Z-f9b60eb-995467d5-det_gru_r2
- 20261006T062002Z-f9b60eb-995467d5-det_gru_r3
- 20261006T062222Z-f9b60eb-b8e98292-det_cudnn_ref
