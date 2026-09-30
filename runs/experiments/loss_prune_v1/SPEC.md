# SPEC: loss_prune_v1 (NT-099)

Pre-registered A/B/C, step 1 only (this SPEC and the engine/comparator specs; no GPU time spent
yet). Written under docs/OPERATING_MODEL.md "Sweeps and pre-registered studies" and "Tiny first"
(D-048), docs/DECISIONS.md D-025, D-044, D-046, D-047, D-048, and docs/RUNBOOK.md "Paired
comparator". Files: `configs/scenarios/loss_prune_v1.yaml`,
`configs/compares/loss_prune_v1_ece0.yaml`, `configs/compares/loss_prune_v1_ece0_vol0.yaml`.

## Hypothesis

`docs/research/2026-09-30-math-report/A_losses.md` sections 7-8 (recommendations 1 and 3):

- **Soft ECE** (section 7) is not a proper score: a calibrated predictor scores ~0 in the
  population, so adding it should not move the population optimum. In finite batches its `|.|` kink
  gives an O(1), non-vanishing bias-gradient on the direction heads (an L1 penalty on a noisy batch
  statistic), and a CPU probe on the served model's weights (three batches, `D:/nt_math_scratch/probe.py`)
  found it carried 97-99% cosine alignment with the total gradient and 3.6-11% of the loss value.
  **Prediction:** turning it off (`LAMBDA_SOFT_ECE: 0`) should not cost the model's proper scores
  (CRPSS, NLL) anything real, since it was never supposed to help them.
- **Volatility penalty** (section 8) is minimised at `std(mu) = std(y)`, which is *larger* than the
  conditional mean's true spread whenever `R^2` is near 0 (measured: h1 prediction std 0.16-0.21
  against a target std of ~1.1-1.17). It actively opposes log-cosh, NLL and CRPS, which all want a
  shrunk `mu`. **Prediction:** turning it off in addition to soft ECE should not cost CRPSS either,
  and may relax the fight over the price head.

**Caveat (not this study's gate, flagged for the lead before step 2):** NT-102's amendment
(docs/BACKLOG.md, 2026-10-01) found the pre-clip main-group gradient norm is currently ~900 against
`GRAD_CLIP_NORM` 20 on the new OHLCV+14-family default (D-047) — every step clips heavily under
today's defaults. NT-102 asks that the clip be re-measured/re-chosen on the screen layout *before*
the loss-pruning A/Bs run, and NT-098 (the gradient-share probe this study's own rationale rests on)
has not run yet either. Both are `todo`. This SPEC does not depend on their outcome to be written
(the hypothesis and design stand on their own), but the lead should confirm whether to wait for
NT-098/NT-102 before authorising step 2's GPU time, or to run loss_prune_v1 as originally scheduled
and let NT-102's clip choice apply to a later iteration if it changes.

## Conditions (3, at most 3 allowed by OPERATING_MODEL)

One engine scenario, `loss_prune_v1`, three variants (`configs/scenarios/loss_prune_v1.yaml`):

| variant | LAMBDA_SOFT_ECE | LAMBDA_VOL | role |
|---|---|---|---|
| `control` | 1.0 (default) | 1.0 (default) | baseline (B in both verdicts) |
| `ece0` | 0.0 | 1.0 (default) | A of verdict 1 |
| `ece0_vol0` | 0.0 | 0.0 | A of verdict 2 |

Everything else at the reference setup's defaults (`configs/default.yaml`): `HORIZON_STEPS [10, 15,
20]` bars (h0/h1/h2), `LOOKBACK 60`, `calibrated_quantile` strategy, zero trading costs (D-044). The
calibration pass (`run.calibrate: true`) stays on: `training/lambda_calibration.py` treats a
zero-weight term as inactive and leaves it at 0 rather than rescaling it back up (`active = lambda >
0.0`, confirmed at `lambda_calibration.py:174-175` for ECE and the same pattern for every other
term) — so `ece0` and `ece0_vol0` genuinely train with those terms off, not just started at 0.

## Layout: the micro layout, with pre-2024-06 judgement folds

"The micro layout" here means small (day-scale) per-fold training blocks on `Bitcoin_BTCUSDT.csv`
(D-041), the same design principle as `configs/scenarios/micro_l2.yaml` — **not** that file's exact
window. That window (`MAX_SEQUENCE_COUNT: 129600`, the newest ~90 days) and every other layout this
project has scored on the long file so far (the micro-loop hypotheses H1-L2, `long_360d_stab`'s
folds -3/-2) draw only from the newest ~90-490 days of `Bitcoin_BTCUSDT.csv` (2024-05-28 onward at
the very earliest: `long_360d_stab`'s fold -2 train starts 2024-06-01, per its own header). D-034/
D-046 require this study's judgement folds to be ones "no choice used" — so they must sit entirely
outside that touched span, not just at a different `FOLD_INDEX` under a different layout (fold
indices are per-spec; what matters is the calendar dates).

**Design:** `N_FOLDS: 40`, `MAX_SEQUENCE_COUNT: 1000000` (the newest 1,000,000 sequences, i.e. the
window starts 2023-11-04, well before any prior choice's earliest date), `VAL_FRACTION =
CAL_FRACTION = 0.02`. TimeSeriesSplit is expanding-window (verified against
`neural_trade.experiments.dataset.data_layout` directly, no training, on this exact override set;
script not committed, output below): position 0 (the earliest fold, most negative `FOLD_INDEX`) has
the smallest training block and the oldest out-of-sample block; `FOLD_INDEX -1` is always the latest
(largest train, newest OOS) and stays untouched as `role: test` (D-020) regardless.

Judgement folds: **-39, -38, -37, -36, -35** (the 5 oldest usable folds under this layout). Their
training blocks are 5.9-73.7 days (all substantially smaller than `long_360d_stab`'s 328-360 days,
i.e. "micro" scale, fast); their out-of-sample blocks are 16.9 days each, non-overlapping, and land
**2023-12-08 through 2024-03-19** — more than two months before `long_360d_stab`'s earliest touched
date (2024-06-01) and long before the micro-loop's newest-90-days window. `FOLD_INDEX -1` (test,
OOS 2025-09-13..2025-09-29) and every fold from -34 to -1 are **not** requested in this scenario's
`folds:` list, so they never train or score here: this study only ever touches -39..-35.

| FOLD_INDEX | train days | train range | OOS range (16.9 d) |
|---|---|---|---|
| -39 | 5.9 | 2023-11-04 .. 2023-11-10 | 2023-12-08 .. 2023-12-25 |
| -38 | 22.9 | 2023-11-04 .. 2023-11-27 | 2023-12-25 .. 2024-01-11 |
| -37 | 39.8 | 2023-11-04 .. 2023-12-14 | 2024-01-11 .. 2024-01-28 |
| -36 | 56.8 | 2023-11-04 .. 2023-12-31 | 2024-01-28 .. 2024-02-14 |
| -35 | 73.7 | 2023-11-04 .. 2024-01-17 | 2024-02-14 .. 2024-03-02 |

(`n_usable_folds` at this N_FOLDS/MAX_SEQUENCE_COUNT/fractions is 39; FOLD_INDEX -1 alone is `test`,
-2 through -39 are `dev` in this spec's own layout — none of which is the same fold this project's
earlier choices actually scored, since those used a different override set entirely.)

Seeds: **0, 1** (2 seeds per fold; the budget comfortably allows 2, see below). 5 folds x 2 seeds x
3 variants = 30 cells; each verdict pairs `control` against one other variant over the same 5 folds
x 2 seeds = 10 pairs (>= the 5-pair floor, D-046's unit of inference is still the fold: 5 distinct
judgement folds, each fold's 2 seeds averaged first by the comparator's `_fold_rows`).

## Metrics, minimum effect and thresholds

- **Primary (both verdicts):** `h1/variance/crpss` (CRPSS against the constant-variance baseline,
  the middle horizon), judged as **non-inferiority**: `non_inferiority_margin: 0.005` in the
  comparator spec (pass iff the paired CI's lower bound is above -0.005, i.e. the pruned variant is
  not worse than control by more than 0.005 CRPSS, `comparator.non_inferiority_verdict`). `min_effect:
  0.005` is also set so the ordinary "beats"/"loses" verdict is reported alongside for context, but
  the study's pass/fail call is the non-inferiority verdict, not the beats verdict. Stated per horizon
  (h1), not averaged: the engine's `result.json` scores are per-horizon keys
  (`h0/variance/crpss`, `h1/...`, `h2/...`); there is no engine-level mean-over-horizons metric to
  compare on directly.
- **Guard-rails (paired, via the comparator's `guard_rails`, same 0.005 margin):** `h0/variance/crpss`
  and `h2/variance/crpss` must also not degrade by more than 0.005 (same non-inferiority logic,
  `gv = "pass" if gest["ci_lo"] > -g.max_degradation else ...`). A study that passes on h1 but
  breaches h0 or h2 is not adopted without the lead's review.
- **Secondary (reported, not gating):** direction BCE (`h*/direction/bce` if present in scores, else
  read from each cell's `eval_report_dev.json`) and direction AUC against `logreg_lags`
  (`evaluation/baselines.py`'s built-in baseline row for every scored run) for both configurations, on
  the same 5 folds — computed and shown per horizon in REPORT.md the way
  `runs/experiments/micro_loop_v1/LOG.md` reports it (seed-mean AUC minus logreg_lags), not run
  through the paired comparator (AUC here is exploratory context, not a go/no-go criterion for this
  item; the loss terms under test are calibration/variance terms, not the direction loss).
- **Guard-rails (absolute, per cell, checked directly in REPORT.md, not paired):**
  - conformal coverage `coverage90` in **[0.85, 0.95]** at every horizon, for every cell of every
    configuration (target 0.90, D-008);
  - `nonfinite_grad_steps == 0` for every cell (`training/custom_model.py`'s `TrainMetrics`);
  - clipped share: `grad_clip_steps_main / total steps` for every cell **once NT-037 lands**; until
    then, the mean pre-clip global gradient norm per group from `metrics.jsonl`, reported (not
    gated) for both configurations. NT-102's amendment already expects heavy clipping under today's
    defaults on the new OHLCV input (pre-clip norm ~900 vs `GRAD_CLIP_NORM` 20): this study reports
    the number so NT-102 has real data from the pruned loss too, but does not fail the item on
    clipping alone until NT-102 sets a rule.
- **Minimum practical effect:** 0.005 CRPSS (chosen to match `long_360d_stab`'s CRPSS magnitudes,
  0.009-0.024, and NT-098's own guard-rail scale; about half of one horizon's measured CRPSS on the
  reference setup, so it would catch a material loss without demanding bit-level equality).

## GPU-time estimate

`sec_per_step` **0.1735 s** (D-047's measured figure for the OHLCV + 14-family default on the GPU;
the STATUS-recorded number, not a fresh timing run for this SPEC — no GPU time has been spent on
this item yet, and re-measuring it was not worth a separate GPU touch when a recent same-default
number already exists). Steps, from the fold layout above at `BATCH_SIZE 2048`, `EPOCHS 20`
(`ceil(train_n / 2048)` per fold, summed over the 5 judgement folds):

| fold | train_n | steps/epoch |
|---|---|---|
| -39 | 8,550 | 5 |
| -38 | 32,940 | 17 |
| -37 | 57,330 | 28 |
| -36 | 81,720 | 40 |
| -35 | 106,110 | 52 |
| **sum** | | **142** |

Total steps = 142 steps/epoch x 20 epochs x 3 variants x 2 seeds = **17,040 steps** (worst case, no
early stopping; `EARLY: 6` will usually stop sooner). At 0.1735 s/step: **2,956 s = 0.82 GPU-hours**.
Adding ~15% for the calibration pre-pass, evaluation and per-cell overhead: **~0.94 GPU-hours**,
against the **3-hour** cap (OPERATING_MODEL). Well inside budget; no owner escalation needed for the
GPU time itself.

## Guard-rails on the design itself

- `min_folds: 5`, never lower (D-046); this design uses exactly 5.
- `pairs_planned: 10` in both comparator specs (5 folds x 2 seeds), so a comparison run with any
  other pair count is refused rather than silently accepted (no peeking).
- `registered_utc` in both comparator specs is superseded by the file's git commit time once
  committed and unedited (`comparator.py`'s `from_yaml`): no GPU run may start before that commit.
- The scenario spec requests only `folds: [-39, -38, -37, -36, -35]`; no cell for any other
  `FOLD_INDEX` is trained under this spec, so there is no way to peek at a closer-to-present fold
  and relabel it later.

## What adopting a variant means (acceptance criterion 4)

If a verdict's non-inferiority check passes (and its guard-rails pass), the lead records a DECISIONS
entry citing this SPEC, the run ids and the verdict, and changes the shipped default
(`LAMBDA_SOFT_ECE: 0`, and additionally `LAMBDA_VOL: 0` if `ece0_vol0` also passes) in
`core/config.py` / `configs/default.yaml` as a follow-up implementer item — this SPEC does not
change any default by itself. A negative or inconclusive result on either verdict is recorded as-is
(VISION "Principles"); it does not retry with a new margin or a new fold set under this item.

## Commit

Committed on branch `nt-099` before any GPU time: see the handback for this call's commit sha.
