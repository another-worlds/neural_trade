# REPORT: loss_prune_v1 (NT-099)

Against `runs/experiments/loss_prune_v1/SPEC.md` (commit `70ca348`, pinned worktree
`D:/nt_exp_loss_prune_v1`). GPU work: `configs/scenarios/loss_prune_v1.yaml`, scored with
`configs/compares/loss_prune_v1_ece0.yaml` and `configs/compares/loss_prune_v1_ece0_vol0.yaml`.

## Verdicts (summary)

| verdict | result | caveat |
|---|---|---|
| **1. ece0 (LAMBDA_SOFT_ECE 0) vs control** | **non-inferiority: PASS** (margin 0.005; ordinary "beats" verdict: inconclusive); both guard-rails (h0, h2 crpss) PASS | **Provisional** — one of its 5 pairs (fold -39) trained through a 27-second window of contaminated code on disk (incident below) and must be confirmed by a re-run before this verdict is final |
| **2. ece0_vol0 (LAMBDA_SOFT_ECE 0, LAMBDA_VOL 0) vs control** | **refused by the comparator**: only 4 of 5 judgement folds have a pair (D-046 needs >= 5) | Incomplete, not wrong — fold -35 was not run (3-GPU-hour cap, below) |

The item is **not closed**. Both gaps (the f-39 re-run, the missing f-35 cell) are small
(estimated 30-40 GPU-minutes total, "Remaining cost" below) and need the owner's go-ahead since the
study is already over its stated 3-hour budget.

## What ran

15 cells were requested (5 judgement folds x 3 variants x 1 seed, per the SPEC's amendment (c));
**14 finished, 1 was stopped before it logged a training step** (`ece0_vol0__f-35__s0`, below), plus
**2 earlier incomplete attempts kept as evidence** (D-029, never deleted): the original `BATCH_SIZE
2048` design's `control__f-39__s0` (OOM) and its orphaned `control__f-39__s1` (killed alongside it,
no result ever attempted). All run directories: `runs/scenarios/loss_prune_v1/`.

| run id | cell | status | wall_s | sec_per_step | note |
|---|---|---|---|---|---|
| `20260930T225007Z-fb840fd-7d108cb2-control__f-39__s0` | control f-39 s0 | failed | 137.5 | - | BATCH_SIZE 2048 OOM (pre-fix design); no data used |
| `20260930T225225Z-fb840fd-c48fa3b3-control__f-39__s1` | control f-39 s1 | incomplete | - | - | killed with the above before it trained; no data used |
| `20260930T225637Z-fb840fd-6fc821bd-control__f-39__s0` | control f-39 s0 | done | 542.4 | 1.065 | |
| `20260930T230540Z-fb840fd-9bd6073c-control__f-38__s0` | control f-38 s0 | done | 641.9 | 0.425 | |
| `20260930T231622Z-fb840fd-3ee2ce56-control__f-37__s0` | control f-37 s0 | done | 549.2 | 0.443 | |
| `20260930T232532Z-fb840fd-f2ddc1ec-control__f-36__s0` | control f-36 s0 | done | 1029.7 | 0.563 | |
| `20260930T234243Z-fb840fd-f1fd90cf-control__f-35__s0` | control f-35 s0 | done | 1539.7 | 0.544 | |
| `20261001T000824Z-fb840fd-d0074ed7-ece0__f-39__s0` | ece0 f-39 s0 | done | 499.7 | 0.783 | **suspect — see incident** |
| `20261001T001645Z-fb840fd-68e19787-ece0__f-38__s0` | ece0 f-38 s0 | done | 398.5 | 0.408 | |
| `20261001T002324Z-fb840fd-8c5e4460-ece0__f-37__s0` | ece0 f-37 s0 | done | 604.6 | 0.409 | |
| `20261001T003329Z-fb840fd-1858c1d2-ece0__f-36__s0` | ece0 f-36 s0 | done | 619.4 | 0.377 | |
| `20261001T004350Z-fb840fd-622d7752-ece0__f-35__s0` | ece0 f-35 s0 | done | 1385.0 | 0.503 | |
| `20261001T010656Z-fb840fd-cdbbbb84-ece0_vol0__f-39__s0` | ece0_vol0 f-39 s0 | done | 652.9 | 0.958 | |
| `20261001T011750Z-fb840fd-07072865-ece0_vol0__f-38__s0` | ece0_vol0 f-38 s0 | done | 936.9 | 0.507 | |
| `20261001T013328Z-fb840fd-8d0838b5-ece0_vol0__f-37__s0` | ece0_vol0 f-37 s0 | done | 757.3 | 0.446 | |
| `20261001T014606Z-fb840fd-d0c338cf-ece0_vol0__f-36__s0` | ece0_vol0 f-36 s0 | done | 771.9 | 0.431 | |
| `20261001T015859Z-fb840fd-063f015e-ece0_vol0__f-35__s0` | ece0_vol0 f-35 s0 | incomplete | ~120 (stopped) | - | stopped before its first logged epoch, 3-hour cap (below) |

## GPU time used against the budget — and why the estimate was wrong

**Actual: ~3 h 16 min** of wall-clock scenario-run time (the lead's figure, 03:56-07:12 local,
matching the sum of the 14 completed cells' `wall_s` + the failed/killed cells, ~11,187 s = 3.11 h of
pure training time plus inter-cell data-loading/scoring overhead) — **over the SPEC's revised
estimate of ~1.78 GPU-hours**, and well over OPERATING_MODEL's 3-hour cap for a pre-registered study.
Stopped per the lead's explicit instruction once the cap was crossed.

**Why the estimate was off:** the SPEC's `sec_per_step` (both the original 0.1735 s and the revised
~0.33 s interpolation) significantly undershot the measured values. Actual `sec_per_step` from each
cell's own `status.json`, by fold (smaller fold = fewer steps/epoch = more per-epoch fixed overhead
amortized over fewer steps):

| fold | steps/epoch (BATCH_SIZE 1024) | control | ece0 | ece0_vol0 |
|---|---|---|---|---|
| -39 | 9 | 1.065 | 0.783 | 0.958 |
| -38 | 33 | 0.425 | 0.408 | 0.507 |
| -37 | 56 | 0.443 | 0.409 | 0.446 |
| -36 | 80 | 0.563 | 0.377 | 0.431 |
| -35 | 104 | 0.544 | 0.503 | (not run) |

Two things the SPEC's interpolation missed: (1) `status.json`'s `sec_per_step` evidently includes
per-epoch fixed costs (data pipeline re-iteration, validation pass, the lambda-calibration pass is
once per cell not per epoch) that do **not** shrink with more steps/epoch, so the smallest folds
(-39, 9 steps/epoch) show `sec_per_step` 2-3x the larger folds' — the SPEC's steps-only formula
(`steps x sec_per_step`) does not model this fixed cost; (2) `gpu_batch_bench_v1`'s 256->1024
batch-scaling ratio (1.90x) was measured on the pre-OHLCV architecture and evidently does not
transfer to the OHLCV+14-family model's attention-heavy forward pass. The real, measured
`sec_per_step` here (0.38-1.07 s) is 1.1x-2.4x the SPEC's 0.33 s interpolation, which — compounded
over 14-15 cells instead of being caught at cell 1 — produced the overrun. **Lesson for future specs
(flagged for the backlog, not acted on here):** interpolating `sec_per_step` from a different input
default's batch-scaling curve is unreliable for this architecture; a short real timing cell (one
cheap fold) before committing to a multi-cell budget would have caught this in ~9 minutes instead of
3+ hours.

## Incident: NT-037 merge contamination of `ece0__f-39__s0`

The lead merged `origin/nt-037` into this main checkout (where this study's scenario-run process was
launched from, with `PYTHONPATH` pointing at the pinned worktree `D:/nt_exp_loss_prune_v1/src`) by
mistake at 05:11:24 local, reverted to `70ca348` at 05:11:51 (27 seconds). Cell
`20261001T000824Z-fb840fd-d0074ed7-ece0__f-39__s0` was mid-training at that moment
(`period_init 05:10:58`, `weights 05:12:12`): TensorFlow's autograph tracing reads function source
from disk, so some part of its graph may have traced against `nt-037`'s code rather than `70ca348`'s.
**This cell is flagged suspect and is used in verdict 1 as-is for now** (its fold-mean diff for h1
CRPSS is +0.00337, in line with the other folds' diffs, so it does not look like an outlier by eye —
see `runs/compares/loss_prune_v1_ece0/result.json`'s `fold_rows`), but per the lead's instruction the
re-run (into a separate store, same spec, code `70ca348`) is **not done in this call**. The original
cell is kept in place (D-029); once the re-run exists, verdict 1 should be recomputed with the re-run
replacing the suspect cell, and this REPORT updated with both side by side.

## Verdict 1: ece0 vs control (`runs/compares/loss_prune_v1_ece0/result.json`)

5 of 5 judgement folds paired (seed 0 only, per the amended SPEC). Primary metric `h1/variance/crpss`:

| fold | control h1 crpss | ece0 h1 crpss | diff (ece0 - control) |
|---|---|---|---|
| -39 | 0.00823 | 0.01159 | +0.00337 |
| -38 | 0.02078 | 0.01833 | -0.00245 |
| -37 | 0.03962 | 0.04294 | +0.00332 |
| -36 | 0.01572 | 0.01897 | +0.00325 |
| -35 | 0.03851 | 0.04328 | +0.00477 |

Paired mean diff **+0.00245** (CI [-0.00104, +0.00594], estimator `mean`, alpha 0.05). Ordinary
beats/loses verdict (`min_effect` 0.005): **inconclusive** (the CI straddles neither +-0.005).
**Non-inferiority (margin 0.005): PASS** — the CI's lower bound (-0.00104) is above -0.005, so ece0
is not worse than control by more than the pre-registered margin; it may even be marginally better
(point estimate positive on 4 of 5 folds), consistent with the hypothesis that soft ECE's gradient
noise was not buying the proper scores anything. Guard-rails: `h0/variance/crpss` PASS (estimate
+0.00156, CI [-0.00130, +0.00442]); `h2/variance/crpss` PASS (estimate +0.00275, CI [-0.00286,
+0.00837]).

## Verdict 2: ece0_vol0 vs control (`runs/compares/loss_prune_v1_ece0_vol0/result.json`)

**Refused**: `"only 4 judgement fold(s) had a usable pair ([-39, -38, -37, -36]), need >= 5 (D-046)"`.
Fold -35's pair is incomplete on the `ece0_vol0` side (`excluded_pairs`: `"side A: no matching (seed,
fold) run"`). The 4 completed pairs' h1 CRPSS diffs (ece0_vol0 - control): f-39 +0.00226, f-38
-0.00135, f-37 -0.00073, f-36 +0.00994 — no sign pattern that suggests a large effect either way, but
this is descriptive only; **no verdict can be drawn with 4 folds** (D-046's floor is 5, not advisory).

## Absolute guard-rails (per cell, not paired)

- **Conformal coverage `coverage90`:** every one of the 14 completed cells is inside **[0.85, 0.95]**
  at every horizon (range observed: 0.879-0.910; per-variant horizon means 0.897-0.900). **PASS** for
  all three configurations.
- **`nonfinite_grad_steps`:** **0** for every completed cell, every epoch. **PASS**.
- **Clipped share / gradient norm (descriptive, NT-037/NT-102, not gated):** NT-037's `grad_clip_steps`
  metric does not exist yet, so this reports the epoch-level pre-clip `grad_global_norm` instead
  (`training/custom_model.py`'s `grad_global_norm`, computed before `clip_by_global_norm`):

  | variant | min (any cell) | mean (per-cell mean, range) | max (any cell) |
  |---|---|---|---|
  | control | 7.5 | 12.8-24.3 | 86.1 |
  | ece0 | 2.5 | 10.5-19.7 | 71.4 |
  | ece0_vol0 (4 cells) | 2.0 | 10.7-21.7 | 117.6 |

  `GRAD_CLIP_NORM` is 20: every variant's max exceeds it in most cells (clipping does occur in most
  epochs), but **none of these per-cell maxima approaches NT-102's reported ~900** figure for the
  OHLCV+14-family default. This study's folds are small (5.9-73.7 day training blocks, 9-104
  steps/epoch) and may simply not reach the regime NT-102 measured; it is reported here as NT-102's
  own amendment asked, with no claim about why the two numbers differ. No clear difference in the
  pruned variants' gradient-norm distribution against control is visible at this cell count.

## Secondary: direction AUC against `logreg_lags`, and val BCE (`val_dir_loss`)

Reported, not gating (the loss terms under test are calibration/variance terms). Seed-0, per-horizon
mean of (model AUC - logreg_lags AUC) over the available folds:

| variant (n folds) | h0 | h1 | h2 |
|---|---|---|---|
| control (5) | -0.0240 | -0.0160 | -0.0245 |
| ece0 (5) | -0.0292 | -0.0074 | -0.0238 |
| ece0_vol0 (4) | -0.0376 | -0.0108 | -0.0234 |

All three configurations sit **below** `logreg_lags` at every horizon, consistent with every earlier
finding in `runs/experiments/micro_loop_v1/LOG.md` on this reference setup (the network has not beaten
the linear baseline on any configuration tried to date); pruning soft ECE / vol does not change this
picture on 4-5 folds. Mean validation direction loss (`val_dir_loss`, the trained BCE, last served
epoch) was broadly similar across variants (control 0.70-0.86, ece0 0.69-0.99, ece0_vol0 0.86-0.98
across h0-h2); no large or consistent shift.

## Remaining cost to complete (for the owner's go-ahead)

Both gaps are small, estimated from this run's own measured per-fold times:

- **`ece0_vol0` fold -35** (the missing cell): comparable folds took 1,385-1,540 s (ece0/control
  f-35) -> **~23-26 GPU-minutes**.
- **`ece0` fold -39 re-run** (the suspect cell, into a separate store per the lead's instruction): the
  original cell took 500 s -> **~8-9 GPU-minutes**.

**Total: ~30-35 GPU-minutes** to close both gaps. This pushes the item's total GPU time to roughly
3.6-3.7 hours against the SPEC's 3-hour cap (OPERATING_MODEL): over the limit, so it needs the
owner's go-ahead before either runs (the lead is asking).

## What this means for the backlog

- **NT-099 stays open**, not `done`: verdict 1 is provisional (pending the f-39 re-run) and verdict 2
  is incomplete (pending the fold -35 cell). No default changes (`LAMBDA_SOFT_ECE`, `LAMBDA_VOL`) are
  made from this REPORT; SPEC.md "What adopting a variant means" still applies once both gaps close.
- **Follow-up for NT-030/NT-050 and future specs:** do not interpolate `sec_per_step` across batch
  sizes or input defaults from a different architecture's benchmark; measure a cheap real cell first.
  Worth a line in RUNBOOK "GPU rules" or NT-030's budget formula — not actioned here, flagged for the
  lead.
- **NT-102 gets real (if small) OHLCV-default gradient-norm data** from three fully-pruned-loss
  configurations at small training-block sizes (table above), alongside its own screen-layout
  measurement.
