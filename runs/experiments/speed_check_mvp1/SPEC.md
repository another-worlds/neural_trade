# NT-075 SPEC: did sec_per_step regress between 1aeff1c and the MVP-1 head 426de4f?

Pre-registered 2026-10-06 by the experimenter, before any GPU run of this item. Study kind: pre-registered
measurement (not a sweep, not an A/B quality verdict: no folds/seeds inference applies; the unit is the run).

## Question and hypothesis

The notebook run on 426de4f (`runs/20260929T081632Z-426de4f-dirty-aba344d6`) logged `sec_per_step` 0.1066, the
earlier notebook run on 1aeff1c (`runs/20260924T182915Z-1aeff1c-dirty-af67ee43`) 0.0984 (+8%, one run each).
H0: the per-step training path of 426de4f is not slower than 1aeff1c beyond run-to-run noise (D-018).
H1: it is slower.

This is NOT a measurement of today's head: D-047 changed the default input (OHLCV + 14 families, 1.63x step).

## Sides (code pinned, each in its own detached worktree)

- A (old): `git worktree` D:/nt/nt_exp_speed_1aeff1c at 1aeff1c, `PYTHONPATH=D:/nt/nt_exp_speed_1aeff1c/src`.
- B (new, MVP-1 head): D:/nt/nt_exp_speed_426de4f at 426de4f, `PYTHONPATH=D:/nt/nt_exp_speed_426de4f/src`.
- Each process prints `neural_trade.__file__` (a `import neural_trade` check) into its log to prove the import.

## Setup (identical on both sides)

- Command (run from the side's worktree root, so its own `configs/default.yaml` is used; the two files differ
  only in one comment line, HUBER_DELTA):
  `python -m neural_trade.cli train --config configs/default.yaml --csv D:/nt/neural_trade/binance_btcusdt_1min_ccxt.csv
  --runs-dir D:/nt/neural_trade/runs/experiments/speed_check_mvp1/runs --name <side>_r<k> --no-baselines`
  Defaults otherwise: BTC/USDT 1-minute, window 60, horizons 10/15/20, EPOCHS 20 (early stopping may stop
  sooner), SEED 42, default fold, batch 256, calibration on, as the notebook runs did. Same data file for both.
- Not deterministic mode (the path under test is the ordinary one).
- Interleaved order A B A B A B (n = 3 per side); more pairs only if the rule below says inconclusive and the
  budget allows, appended in the same alternation.
- Before each launch: `nvidia-smi dmon -s um -c 10` (GPU free per RUNBOOK: fb <= 2000 MB, median sm <= 30%) and
  the free-disk check. A launcher records host CPU utilisation (psutil, every 5 s, all cores) per run into
  `<run>.cpu.csv`; the REPORT states its mean and max per run and notes any overlap with other agents' CPU work.

## Metrics

1. Primary: `sec_per_step` from each run's `status.json` (the trainer's own figure, includes the step only).
2. Secondary: per-run median epoch time over epochs >= 1 (the first epoch carries graph building), taken from the
   differences of the `time` stamps of consecutive lines in `metrics.jsonl`.
3. Per side: median over its runs, and the spread as (max - min) / median. Ratio r = median(B) / median(A).
4. Reference only (not part of the rule): the two single notebook runs above.

## Decision rule (fixed now)

Applied to the primary metric; the secondary must agree in direction for REGRESSION, else the verdict is INCONCLUSIVE.

- **NO REGRESSION within noise:** r <= 1.05.
- **REGRESSION:** r > 1.05 and the ranges are disjoint (min of B runs > max of A runs). Its size is reported as r - 1
  with both ranges.
- **INCONCLUSIVE:** r > 1.05 but the ranges overlap, or the secondary metric disagrees. Then one more interleaved
  pair (A B) is run once if the budget allows (n = 4 per side) and the same rule is applied; if still not
  decided the REPORT says inconclusive and states the observed r and spreads.
- A run that fails or stops with a CPU-load overlap above 60% mean utilisation by another process is reported
  and repeated once in its slot (noted, both original and repeat kept).

On REGRESSION the REPORT proposes an implementer item (bisect 1aeff1c..426de4f on the step path, with a profiler);
the lead files it.

## Guard-rails

- One GPU job at a time; no GPU job while the GPU check reads busy; never touch the owner's other project.
- No edits to src/ or tests/. Run dirs are written under the main checkout's `runs/experiments/speed_check_mvp1/runs/`
  (new names only; nothing deleted).

## GPU-time estimate

Notebook runs took 280 s (A) and 340 s (B) of training; with start-up, calibration and report about 6-7 min per run.
6 runs = about 40 min = 0.67 GPU-hours; one optional extra pair (2 runs, +0.2 h) fits; if more are needed the experimenter stops and reports. Budget cap
1 GPU-hour (the 3-hour item cap is not approached).

## Code pin

Spec commit sha: recorded in the REPORT (the SPEC is committed before any run; git log of this path).
