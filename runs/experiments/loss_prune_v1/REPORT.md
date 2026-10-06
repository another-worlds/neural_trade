# REPORT: loss_prune_v1 (NT-099)

Against `runs/experiments/loss_prune_v1/SPEC.md` (commit `70ca348`, pinned worktree
`D:/nt_exp_loss_prune_v1`). GPU work: `configs/scenarios/loss_prune_v1.yaml`, scored with
`configs/compares/loss_prune_v1_ece0.yaml` and `configs/compares/loss_prune_v1_ece0_vol0.yaml`.

The two cells D-049 allowed were trained on 2026-10-03 from this checkout at `2573116`. That commit
is docs-only on top of `fde4f30`; `git diff 70ca348 2573116 -- src configs` is empty, so the stamp
is not a code change relative to the SPEC pin. The compare yaml files were not edited. Verdict 1
scores a separate store by setting `CompareSpec.root` in memory (`root` is part of the spec hash, so
the rerun registration hash is `725dbd712256` rather than the main store's `283274d4318c`).

## Verdicts (summary)

| verdict | result | what is in it |
|---|---|---|
| **1. ece0 (LAMBDA_SOFT_ECE 0) vs control** | **non-inferiority: PASS** (margin 0.005; ordinary "beats" verdict: inconclusive); both guard-rails (h0, h2 crpss) PASS | **Final.** Fold -39 is the re-run `20261003T093120Z-2573116-d0074ed7-ece0__f-39__s0`. The suspect cell is kept and is not in this verdict. |
| **2. ece0_vol0 (LAMBDA_SOFT_ECE 0, LAMBDA_VOL 0) vs control** | **non-inferiority: PASS** (margin 0.005; ordinary "beats" verdict: inconclusive); both guard-rails (h0, h2 crpss) PASS | **Final.** Fold -35 is `20261003T091952Z-2573116-063f015e-ece0_vol0__f-35__s0`, scored on the main store. |

Both variants meet the SPEC's adoption rule (non-inferiority and the paired guard-rails). This
report does not change `LAMBDA_SOFT_ECE` or `LAMBDA_VOL`. The SPEC assigns that default change to a
lead DECISIONS entry and a follow-up implementer item. QA has not passed this item, so NT-099 is
not done.

## Erratum (2026-10-06, lead): the `ece0_vol0` arm trained at LAMBDA_VOL 0.1, not 0

The SPEC (Conditions) says the calibration pass leaves a 0 weight at 0. That holds for soft ECE
(`ece_active` gate) but not for vol: `training/lambda_calibration.py` at `70ca348` rescales vol with
no gate, and the clamp lifts 0 to `CALIB_LAMBDA_MIN` 0.1. All five `ece0_vol0` cells record
`"lambda_vol": 0.1` in `metrics.jsonl`; the control cells record 1.44-1.60. Verdict 2 therefore
compares vol at the 0.1 floor against vol at about 1.5, not vol off against on. The statistics and
the verdict stand for that comparison; only the label "vol off" was wrong. The shipped default
`LAMBDA_VOL: 0` (NT-117) reproduces the tested arm only when calibration runs; a true 0 is untested.
D-058 records this; NT-118 adds the gate and the A/B of 0 against the floor. The SPEC is
pre-registered and is not edited.

## What ran

15 cells were requested (5 judgement folds x 3 variants x 1 seed, per the SPEC's amendment (c)).
The first campaign finished 14 and stopped the 15th before it logged a training step. On 2026-10-03
the missing cell was trained into a new directory on the main store, and the suspect `ece0` fold -39
was re-run into `runs/loss_prune_v1_ece0_f39_rerun/` (light-file copies of the other done cells live
there so the comparator could pair them). Those copies are the same light files as the originals,
not a second training; they are committed with this report so a cited run id has no untracked light
file beside it (NT-010). Weights and the other heavy artefacts stay untracked. The incomplete
directory and the suspect cell were not copied. The batch-2048 OOM cell and the seed-1
orphan are still on disk (D-029). Nothing was deleted.

| run id | cell | status | wall_s | sec_per_step | note |
|---|---|---|---|---|---|
| `20260930T225007Z-fb840fd-7d108cb2-control__f-39__s0` | control f-39 s0 | failed | 137.5 | - | BATCH_SIZE 2048 OOM (pre-fix design); no data used |
| `20260930T225225Z-fb840fd-c48fa3b3-control__f-39__s1` | control f-39 s1 | incomplete | - | - | killed with the above before it trained; no data used |
| `20260930T225637Z-fb840fd-6fc821bd-control__f-39__s0` | control f-39 s0 | done | 542.4 | 1.065 | |
| `20260930T230540Z-fb840fd-9bd6073c-control__f-38__s0` | control f-38 s0 | done | 641.9 | 0.425 | |
| `20260930T231622Z-fb840fd-3ee2ce56-control__f-37__s0` | control f-37 s0 | done | 549.2 | 0.443 | |
| `20260930T232532Z-fb840fd-f2ddc1ec-control__f-36__s0` | control f-36 s0 | done | 1029.7 | 0.563 | |
| `20260930T234243Z-fb840fd-f1fd90cf-control__f-35__s0` | control f-35 s0 | done | 1539.7 | 0.544 | |
| `20261001T000824Z-fb840fd-d0074ed7-ece0__f-39__s0` | ece0 f-39 s0 | done | 499.7 | 0.783 | **suspect — kept, not in verdict 1** |
| `20261001T001645Z-fb840fd-68e19787-ece0__f-38__s0` | ece0 f-38 s0 | done | 398.5 | 0.408 | |
| `20261001T002324Z-fb840fd-8c5e4460-ece0__f-37__s0` | ece0 f-37 s0 | done | 604.6 | 0.409 | |
| `20261001T003329Z-fb840fd-1858c1d2-ece0__f-36__s0` | ece0 f-36 s0 | done | 619.4 | 0.377 | |
| `20261001T004350Z-fb840fd-622d7752-ece0__f-35__s0` | ece0 f-35 s0 | done | 1385.0 | 0.503 | |
| `20261001T010656Z-fb840fd-cdbbbb84-ece0_vol0__f-39__s0` | ece0_vol0 f-39 s0 | done | 652.9 | 0.958 | |
| `20261001T011750Z-fb840fd-07072865-ece0_vol0__f-38__s0` | ece0_vol0 f-38 s0 | done | 936.9 | 0.507 | |
| `20261001T013328Z-fb840fd-8d0838b5-ece0_vol0__f-37__s0` | ece0_vol0 f-37 s0 | done | 757.3 | 0.446 | |
| `20261001T014606Z-fb840fd-d0c338cf-ece0_vol0__f-36__s0` | ece0_vol0 f-36 s0 | done | 771.9 | 0.431 | |
| `20261001T015859Z-fb840fd-063f015e-ece0_vol0__f-35__s0` | ece0_vol0 f-35 s0 | incomplete | ~120 (stopped) | - | stopped before its first logged epoch; kept |
| `20261003T091952Z-2573116-063f015e-ece0_vol0__f-35__s0` | ece0_vol0 f-35 s0 | done | 587.9 | 0.347 | the missing cell; main store; 12 epochs, served epoch 6 |
| `20261003T093120Z-2573116-d0074ed7-ece0__f-39__s0` | ece0 f-39 s0 | done | 210.1 | 0.642 | re-run; store `runs/loss_prune_v1_ece0_f39_rerun/`; 11 epochs, served epoch 5 |

Directories for every row except the re-run are `runs/scenarios/loss_prune_v1/`. The re-run is
`runs/loss_prune_v1_ece0_f39_rerun/scenarios/loss_prune_v1/20261003T093120Z-2573116-d0074ed7-ece0__f-39__s0`.

## GPU time used against the budget — and why the estimate was wrong

**First campaign: ~3 h 16 min** of wall-clock scenario-run time (the lead's figure, 03:56-07:12 local,
matching the sum of the 14 completed cells' `wall_s` + the failed/killed cells, ~11,187 s = 3.11 h of
pure training time plus inter-cell data-loading/scoring overhead) — **over the SPEC's revised
estimate of ~1.78 GPU-hours**, and well over OPERATING_MODEL's 3-hour cap for a pre-registered study.
Stopped per the lead's explicit instruction once the cap was crossed.

**D-049 addition (2026-10-03), measured, not estimated:** the missing `ece0_vol0` fold -35 took
587.9 s of `wall_s` (~9.8 min) and the `ece0` fold -39 re-run took 210.1 s (~3.5 min). Together
798 s, about 13 minutes, against the 30-35 minutes estimated from the first campaign's slowest
matching cells. The study's GPU time is about **3 h 29 min**.

**Why the first estimate was off:** the SPEC's `sec_per_step` (both the original 0.1735 s and the revised
~0.33 s interpolation) significantly undershot the measured values. Actual `sec_per_step` from each
cell's own `status.json`, by fold (smaller fold = fewer steps/epoch = more per-epoch fixed overhead
amortized over fewer steps):

| fold | steps/epoch (BATCH_SIZE 1024) | control | ece0 | ece0_vol0 |
|---|---|---|---|---|
| -39 | 9 | 1.065 | 0.783 (suspect); 0.642 (re-run) | 0.958 |
| -38 | 33 | 0.425 | 0.408 | 0.507 |
| -37 | 56 | 0.443 | 0.409 | 0.446 |
| -36 | 80 | 0.563 | 0.377 | 0.431 |
| -35 | 104 | 0.544 | 0.503 | 0.347 (2026-10-03) |

Two things the SPEC's interpolation missed: (1) `status.json`'s `sec_per_step` evidently includes
per-epoch fixed costs (data pipeline re-iteration, validation pass, the lambda-calibration pass is
once per cell not per epoch) that do **not** shrink with more steps/epoch, so the smallest folds
(-39, 9 steps/epoch) show `sec_per_step` 2-3x the larger folds' — the SPEC's steps-only formula
(`steps x sec_per_step`) does not model this fixed cost; (2) `gpu_batch_bench_v1`'s 256->1024
batch-scaling ratio (1.90x) was measured on the pre-OHLCV architecture and evidently does not
transfer to the OHLCV+14-family model's attention-heavy forward pass. The real, measured
`sec_per_step` here (0.35-1.07 s) is 1.1x-3.2x the SPEC's 0.33 s interpolation, which — compounded
over 14-15 cells instead of being caught at cell 1 — produced the overrun. The two later cells landed
at or under the 0.33-0.64 s band, so the overrun was the first campaign's, not these two. **Lesson for
future specs (flagged for the backlog, not acted on here):** interpolating `sec_per_step` from a
different input default's batch-scaling curve is unreliable for this architecture; a short real
timing cell (one cheap fold) before committing to a multi-cell budget would have caught this in ~9
minutes instead of 3+ hours.

## Incident: NT-037 merge contamination of `ece0__f-39__s0`

The lead merged `origin/nt-037` into this main checkout (where this study's scenario-run process was
launched from, with `PYTHONPATH` pointing at the pinned worktree `D:/nt_exp_loss_prune_v1/src`) by
mistake at 05:11:24 local, reverted to `70ca348` at 05:11:51 (27 seconds). Cell
`20261001T000824Z-fb840fd-d0074ed7-ece0__f-39__s0` was mid-training at that moment
(`period_init 05:10:58`, `weights 05:12:12`): TensorFlow's autograph tracing reads function source
from disk, so some part of its graph may have traced against `nt-037`'s code rather than `70ca348`'s.
The original cell is kept (D-029). The provisional verdict that used it is still at
`runs/compares/loss_prune_v1_ece0/result.json` (mean diff +0.00245, fold -39 diff +0.00337).

**Resolution (2026-10-03).** The re-run used the same scenario spec and code equivalent to `70ca348`,
in a store that does not contain the suspect directory, so `duplicate_policy: refuse` did not see two
done runs. Verdict 1 below uses the re-run only. Side by side, seed 0, fold -39, against the same
control cell `20260930T225637Z-fb840fd-6fc821bd-control__f-39__s0` (h1 crpss 0.008227):

| | suspect `20261001T000824Z-...` | re-run `20261003T093120Z-...` |
|---|---|---|
| h1 crpss (diff vs control) | 0.011593 (+0.003366) | 0.012110 (+0.003883) |
| h0 / h2 crpss | 0.005203 / 0.014639 | 0.004928 / 0.016446 |
| direction AUC h0 / h1 / h2 | 0.4633 / 0.5272 / 0.4881 | 0.4632 / 0.5254 / 0.4881 |
| coverage90 h0 / h1 / h2 | 0.8932 / 0.8948 / 0.8889 | 0.8932 / 0.8948 / 0.8889 |
| nonfinite_grad_steps | 0 | 0 |
| grad_global_norm min / mean / max | 2.47 / 19.74 / 71.44 | 2.63 / 17.55 / 53.07 |
| wall_s / sec_per_step | 499.7 / 0.783 | 210.1 / 0.642 |

The fold diff moves by about 0.0005. The re-run is not an outlier against the suspect cell, and the
suspect cell was not an outlier against the other folds. Verdict 1 uses the re-run anyway, because
the point of the re-run was to drop the contaminated trace, not because the two numbers disagreed.

## Verdict 1: ece0 vs control

Final scores: `runs/loss_prune_v1_ece0_f39_rerun/compares/loss_prune_v1_ece0/result.json`
(registration sidecar `runs/loss_prune_v1_ece0_f39_rerun/compares/loss_prune_v1_ece0_vs_control/registration.json`).
The yaml on disk still says the main store; `root` was overridden in memory after
`CompareSpec.from_yaml` confirmed the file matches HEAD (effective registration
`2026-10-01T03:55:40+05:00`, source `git_commit_time`). Five pairs, nothing excluded. Fold -39's A
run is the re-run. Primary metric `h1/variance/crpss`:

| fold | control h1 crpss | ece0 h1 crpss | diff (ece0 - control) |
|---|---|---|---|
| -39 | 0.00823 | 0.01211 | +0.00388 |
| -38 | 0.02078 | 0.01833 | -0.00245 |
| -37 | 0.03962 | 0.04294 | +0.00332 |
| -36 | 0.01572 | 0.01897 | +0.00325 |
| -35 | 0.03851 | 0.04328 | +0.00477 |

Paired mean diff **+0.002556** (CI [-0.000997, +0.006109], estimator `mean`, alpha 0.05). Ordinary
beats/loses verdict (`min_effect` 0.005): **inconclusive**. The interval is not entirely above
+0.005 (its upper end, +0.006109, does cross that line) and not entirely below -0.005.
**Non-inferiority (margin 0.005): PASS** — the CI's lower bound (-0.000997) is above -0.005.
Guard-rails: `h0/variance/crpss` PASS (estimate +0.001505, CI [-0.001439, +0.004449]);
`h2/variance/crpss` PASS (estimate +0.003116, CI [-0.002593, +0.008824]).

## Verdict 2: ece0_vol0 vs control (`runs/compares/loss_prune_v1_ece0_vol0/result.json`)

The earlier 4-fold refusal is replaced by this 5-fold result. The new fold -35 cell is on the main
store, so this call did not need a separate root. Five pairs, nothing excluded.

| fold | control h1 crpss | ece0_vol0 h1 crpss | diff (ece0_vol0 - control) |
|---|---|---|---|
| -39 | 0.00823 | 0.01048 | +0.00226 |
| -38 | 0.02078 | 0.01943 | -0.00135 |
| -37 | 0.03962 | 0.03889 | -0.00073 |
| -36 | 0.01572 | 0.02566 | +0.00994 |
| -35 | 0.03851 | 0.04170 | +0.00318 |

Paired mean diff **+0.002661** (CI [-0.002927, +0.008249], estimator `mean`, alpha 0.05). Ordinary
beats/loses verdict: **inconclusive** (fold -36's +0.00994 widens the interval; the lower bound does
not clear +0.005). **Non-inferiority (margin 0.005): PASS** (lower bound -0.002927 is above -0.005).
Guard-rails: `h0/variance/crpss` PASS (estimate +0.000655, CI [-0.004193, +0.005503]);
`h2/variance/crpss` PASS (estimate +0.002798, CI [-0.002951, +0.008547]).

## Absolute guard-rails (per cell, not paired)

Checked on the 15 done cells that enter the two verdicts (ece0 fold -39 is the re-run, not the
suspect). The kept suspect cell is inside the same bands and is not part of these ranges.

- **Conformal coverage `coverage90`:** every one of those 15 cells is inside **[0.85, 0.95]** at
  every horizon (range observed: 0.879-0.911). **PASS** for all three configurations. The suspect
  cell's coverage90 is 0.893 / 0.895 / 0.889.
- **`nonfinite_grad_steps`:** **0** for every completed cell, every epoch, including the suspect cell
  and both new cells. **PASS**.
- **Clipped share / gradient norm (descriptive, NT-037/NT-102, not gated):** NT-037's `grad_clip_steps`
  metric does not exist yet, so this reports the epoch-level pre-clip `grad_global_norm` instead
  (`training/custom_model.py`'s `grad_global_norm`, computed before `clip_by_global_norm`). The ece0
  row uses the re-run in place of the suspect cell:

  | variant | min (any cell) | mean (per-cell mean, range) | max (any cell) |
  |---|---|---|---|
  | control | 7.5 | 12.8-24.3 | 86.1 |
  | ece0 (re-run on fold -39) | 2.5 | 10.5-17.6 | 61.7 |
  | ece0_vol0 (5 cells) | 2.0 | 10.7-21.7 | 117.6 |

  `GRAD_CLIP_NORM` is 20: every variant's max exceeds it in most cells (clipping does occur in most
  epochs), but **none of these per-cell maxima approaches NT-102's reported ~900** figure for the
  OHLCV+14-family default. This study's folds are small (5.9-73.7 day training blocks, 9-104
  steps/epoch) and may simply not reach the regime NT-102 measured; it is reported here as NT-102's
  own amendment asked, with no claim about why the two numbers differ. No clear difference in the
  pruned variants' gradient-norm distribution against control is visible at this cell count. The
  suspect cell's own max was 71.4, which the re-run (max 53.1) does not repeat; the ece0 row's max
  of 61.7 is fold -37.

## Secondary: direction AUC against `logreg_lags`, and val BCE (`val_dir_loss`)

Reported, not gating (the loss terms under test are calibration/variance terms). Seed-0, per-horizon
mean of (model AUC - logreg_lags AUC) over the five folds in each verdict (ece0 fold -39 is the
re-run):

| variant (n folds) | h0 | h1 | h2 |
|---|---|---|---|
| control (5) | -0.0240 | -0.0160 | -0.0245 |
| ece0 (5, re-run on -39) | -0.0292 | -0.0077 | -0.0238 |
| ece0_vol0 (5) | -0.0333 | -0.0114 | -0.0234 |

All three configurations sit **below** `logreg_lags` at every horizon, consistent with every earlier
finding in `runs/experiments/micro_loop_v1/LOG.md` on this reference setup (the network has not beaten
the linear baseline on any configuration tried to date); pruning soft ECE / vol does not change this
picture. Served-epoch validation direction loss (`val_dir_loss_h*`, the trained BCE at
`status.json`'s `weights_epoch`) stays in a similar band across variants: control 0.69-0.87, ece0
0.69-1.10, ece0_vol0 0.69-1.06 across h0-h2. No large or consistent shift.

## What this means for the backlog

- **NT-099 stays open**, not `done`. Both verdicts are final and both non-inferiority checks pass,
  with both paired guard-rails. No default (`LAMBDA_SOFT_ECE`, `LAMBDA_VOL`) is edited in the commit
  that records these cells. SPEC.md "What adopting a variant means" still applies: the lead records a
  DECISIONS entry citing this SPEC, the run ids and these verdicts, and a follow-up implementer item
  changes the shipped default to `LAMBDA_SOFT_ECE: 0` and `LAMBDA_VOL: 0`. This branch's backlog
  stops at NT-109; later ids already exist on `remediation/plan`, so that item is not numbered here.
  QA has not reviewed the report.
- **Follow-up for NT-030/NT-050 and future specs:** do not interpolate `sec_per_step` across batch
  sizes or input defaults from a different architecture's benchmark; measure a cheap real cell first.
  Worth a line in RUNBOOK "GPU rules" or NT-030's budget formula — not actioned here, flagged for the
  lead.
- **NT-102 gets real (if small) OHLCV-default gradient-norm data** from three loss configurations at
  small training-block sizes (table above), alongside its own screen-layout measurement. The new
  cells do not approach the ~900 figure either.
