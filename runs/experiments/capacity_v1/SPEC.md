# SPEC: capacity_v1 (NT-104 acceptance (3))

Pre-registered A/B/C on model capacity. Written under docs/OPERATING_MODEL.md "Sweeps and
pre-registered studies" and "Tiny first", DECISIONS D-025, D-044, D-046, D-047, D-048, D-057/D-058,
and docs/RUNBOOK.md "Paired comparator". Template: `runs/experiments/loss_prune_v1/SPEC.md`.
Files: `configs/scenarios/capacity_v1.yaml`, `configs/compares/capacity_v1_gru_small.yaml`,
`configs/compares/capacity_v1_linear_indicators.yaml`. Timing cells: `configs/scenarios/capacity_v1_timing.yaml`
(scores discarded). Written before any scored cell exists; it does not change after results.

## Hypothesis

`docs/research/2026-09-30-math-report/B_model_indicators.md` 1.1 and 7 item 2: today's `gru_attention`
has 316,751 parameters (44.6% in one attention block) and never beats a 3-lag logistic regression on
direction (LOG.md L2: significantly below it at 1 h). The direction head already holds a logistic skip
(`DIRECTION_SKIP`).

- **H1 (gru_small):** a smaller recurrent readout (indicators -> GRU(32) -> same heads, 77,138
  parameters) is **not worse** than today's model on h1 direction AUC, and does not degrade the variance
  heads (CRPSS), direction Brier or coverage. A "beats" outcome is possible but not predicted.
- **H2 (linear_indicators):** a linear readout on pooled indicator features (7,610 parameters) is also
  not worse. Its pooled+last-bar features pass through a per-sample `LayerNormalization` (QA note of
  NT-104), so the readout is linear in normalised features but not strictly linear in the raw indicators.

If a smaller arm is not worse, the extra capacity of today's model buys nothing measurable on this setup
and the cheaper model is a candidate default (a follow-up item; this SPEC changes no default).
A negative or inconclusive result is a valid outcome.

## Conditions (3 arms, at most 3)

One engine scenario `capacity_v1`, three variants:

| arm | override | parameters (LOOKBACK 60) | role |
|---|---|---|---|
| `control` | none (`MODEL_NAME gru_attention`) | 316,751 | baseline (B in both verdicts) |
| `gru_small` | `MODEL_NAME: gru_small` | 77,138 | A of verdict 1 |
| `linear_indicators` | `MODEL_NAME: linear_indicators` | 7,610 | A of verdict 2 |

**All arms use today's shipped defaults otherwise**: `LAMBDA_SOFT_ECE 0` and `LAMBDA_VOL 0` (D-057,
D-058), `DIRECTION_SKIP true`, OHLCV input with 14 families (D-047), calibration pass on, zero trading
costs (D-044). **`DIRECTION_DEEP_ZERO_INIT` stays at today's default (false) for every arm**: the same
setting for all arms, so the arms differ only in architecture; the zero-init is a separate switch for a
separate A/B. Per arm and cell, `skip_share`, `tower_share` and `corr_skip_tower` (the NT-110 covariance
decomposition, `eval_report_dev.json` `health.direction_skip_share`, validation block) are reported.
`tower_share` outside [0, 1] is possible (a covariance share; the timing cell of gru_small read skip 1.09,
tower -0.09 at h1) and is reported as it is.

## Layout: the micro layout, with fresh judgement folds

Same design principle as loss_prune_v1 (D-041, D-048), with folds that no earlier choice used (D-034,
D-046). loss_prune_v1's folds -39..-35 (OOS 2023-12-08 .. 2024-03-02) informed a default (D-057), so they
are not reused. Layout (verified with `neural_trade.experiments.dataset.data_layout`, no training):
`N_FOLDS: 100`, `MAX_SEQUENCE_COUNT: 1500000` (the newest 1.5M sequences of `Bitcoin_BTCUSDT.csv`; window
starts 2022-11-22), `VAL_FRACTION = CAL_FRACTION = 0.02`, `BATCH_SIZE 1024` (2048 OOMs on the default
attention, loss_prune_v1 amendment (c)), `SHUFFLE_BUFFER 0`. 96 usable folds; FOLD_INDEX -1 stays the
untouched test fold and is not requested.

**Judgement folds: -96, -95, -94, -93, -92** (the 5 oldest usable folds). No choice used them.

| FOLD_INDEX | train sequences | train range | OOS range |
|---|---|---|---|
| -96 | 14,064 | 2022-11-22 .. 2022-12-02 | 2023-01-13 .. 2023-01-23 |
| -95 | 28,915 | 2022-11-22 .. 2022-12-12 | 2023-01-23 .. 2023-02-02 |
| -94 | 43,766 | 2022-11-22 .. 2022-12-22 | 2023-02-02 .. 2023-02-13 |
| -93 | 58,617 | 2022-11-22 .. 2023-01-02 | 2023-02-13 .. 2023-02-23 |
| -92 | 73,468 | 2022-11-22 .. 2023-01-12 | 2023-02-23 .. 2023-03-05 |

OOS blocks are 14,851 sequences (about 10 days) each, non-overlapping, all earlier than every block any
earlier choice scored (loss_prune_v1: 2023-12 onward; the micro loop and `long_360d_stab`: 2024-05 onward).
Every arm sees identical blocks. No choice (epoch cap, variant, threshold) is made on these folds'
results; the epoch cap below was set from timing alone.

**Seeds: 0** (1 seed per fold). 5 folds x 1 seed x 3 arms = **15 cells**; each verdict pairs one arm
against `control` over the same 5 folds = 5 pairs (>= the floor, D-046: the fold is the unit of
inference; D-046's simulation found 5 folds x 1 seed the safest documented layout). `EPOCHS: 14` (cap;
see GPU time; `EARLY 6` and `PATIENCE 3` unchanged). Today's default is 20; loss_prune_v1 cells stopped at
11-12 epochs, so the cap binds mainly for the slow-converging small arms; the cap is the same for every
arm and is reported with the served epoch of every cell.

## Metrics, minimum effects, decision rule

Paired comparator (NT-032, `neural-trade compare`), estimator mean over the 5 per-fold differences
(arm minus control), paired t interval, alpha 0.05 two-sided; no multiplicity correction across the two
verdicts (as loss_prune_v1; the lead may apply one by hand when reading).

- **Primary metric (both verdicts): `h1/direction/auc`** (the middle horizon, 15 minutes; out-of-sample
  direction AUC of the calibrated P(up)). Minimum practical effect **0.01 AUC**; non-inferiority margin
  **0.01 AUC** (about a third of the whole edge the model has ever shown, 0.50-0.53). The comparator spec's
  `min_effect` and `non_inferiority_margin` are both 0.01.
- **Guard-rails (paired, same CI, non-inferiority; a breach or undecided result blocks "adopt"):**
  `h1/variance/crpss`, `h0/variance/crpss`, `h2/variance/crpss` (margin 0.005 each, as loss_prune_v1);
  `h0/direction/auc`, `h2/direction/auc` (margin 0.01); `h1/direction/brier` (lower is better, margin 0.002;
  a proper score standing in for BCE in the engine's scores).
- **Secondary, reported and not gating:** direction BCE per horizon (computed by a small script from each
  cell's `predictions_oos.npz`: calibrated P(up) against the sign of the realised price change `y`; mean per
  arm and paired difference per fold, same 5 folds); direction AUC minus the `logreg_lags` baseline row;
  CRPSS and AUC at every horizon; `skip_share` / `tower_share` / `corr_skip_tower`; the training
  `sec_per_step`, the served epoch and epochs run per cell; parameter count.
- **Absolute guard-rails per cell (checked directly, not paired):** `coverage90` in **[0.85, 0.95]** at every
  horizon; `nonfinite_grad_steps == 0`; the mean pre-clip gradient norm per group and the share of clipped
  steps reported for every arm (descriptive, not gating, as loss_prune_v1 and NT-102).

**Verdict per arm vs control** (stated now, applied once, on the 5 judgement folds):

1. **Arm better**: comparator verdict "A beats B" (CI lower bound >= +0.01 AUC at h1) and every paired
   guard-rail passes.
2. **Not worse, cheaper (adoptable)**: non-inferiority on h1 AUC passes (CI lower bound > -0.01), every
   paired guard-rail passes, absolute guard-rails hold, and the arm is not "better". Reading: capacity above
   this arm buys nothing measurable here.
3. **Capacity helps (control kept)**: "control beats arm" (CI upper bound <= -0.01 AUC at h1), or a
   non-inferiority "breach".
4. **Inconclusive**: anything else, including a guard-rail that is undecided. Recorded as inconclusive; an
   inconclusive study is re-run at most once before the owner decides (OPERATING_MODEL).

With 5 folds the paired t interval is wide (a fold-to-fold SD of 0.01 AUC gives a half-width of about 0.012),
so "inconclusive" is a realistic outcome and is reported as such, not tuned away.

## GPU-time estimate (measured, counts toward the budget)

Timing cells (`capacity_v1_timing`, 2 epochs on fold -92, run 2026-10-06 from commit 590829b, scores
discarded): wall, train, score seconds and `sec_per_step` from `result.json`:

| arm | wall_s | train_s | score_s | sec_per_step | per epoch (72 steps) | fixed per cell (train_s minus epochs + score_s) |
|---|---|---|---|---|---|---|
| control | 462 | 303 | 159 | 0.464 | 33 s | about 395 s |
| gru_small | 539 | 348 | 191 | 0.344 | 25 s | about 489 s |
| linear_indicators | 299 | 142 | 156 | 0.308 | 22 s | about 254 s |

The timing runs themselves used 1,300 s = **0.36 GPU-hours** of the budget. Fixed cost per cell (data
windows, calibration pass, scoring) dominates; it is taken at the fold -92 value for every fold
(conservative: smaller folds should cost less). Steps per epoch summed over the five folds: 14 + 29 + 43
+ 58 + 72 = **216**. Worst case (every cell runs all 14 epochs), per arm = 5 x fixed + 14 x 216 x
sec_per_step:

- control: 1,975 + 1,404 = 3,379 s
- gru_small: 2,445 + 1,040 = 3,485 s
- linear_indicators: 1,270 + 931 = 2,201 s

Study worst case **9,065 s = 2.52 GPU-hours**; with the timing cells **2.88 GPU-hours** against the
**3-hour** cap. Expected (about 12 epochs, smaller folds' fixed cost lower): about 2.2-2.5 GPU-hours
including timing. `sec_per_step` was measured on the real arms this time (the loss_prune_v1 lesson), but
only on the largest fold; per-cell cost is re-measured from `result.json` after each cell. **Stop rule:**
the run stops at 3.0 GPU-hours counting the timing cells, whatever cells are missing, and a partial
design is reported as such. Parallel cells: none (N = 1).

## Guard-rails on the design itself

- `min_folds: 5`, never lower (D-046); exactly 5 here. `pairs_planned: 5` in both comparator specs, so a
  comparison with another pair count is refused.
- `registered_utc` of both comparator specs is superseded by their git commit time; no scored cell may
  start before that commit. The scenario requests only the five folds above; no other fold ever trains.
- One GPU job at a time; RUNBOOK GPU-free check before every launch; never concurrently with the lead's
  notebook run or the owner's other project.
- All launches from the worktree `D:/nt/nt_wt_104ab` (detached at the SPEC commit, `PYTHONPATH` its `src`),
  store `D:/nt/nt_wt_104ab/runs`; the SPEC commit sha is recorded in REPORT.md.

## What adopting an arm means

Nothing changes by this SPEC. If an arm lands in verdict 2 ("not worse, cheaper"), the lead may open an
implementer item to change `MODEL_NAME`, which re-records the golden run, would be a default change needing
its own QA, and for the physics terms interacts with NT-006; the owner is informed because a smaller default
model changes what every notebook shows. A "capacity helps" or inconclusive verdict closes NT-104 (3) with
the record.
