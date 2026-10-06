# NT-104 (3) capacity A/B (capacity_v1): both arms INCONCLUSIVE; the default stays gru_attention

Experimenter run, 2026-10-06. The lead saved this report from the experimenter's hand-back, because the
subagent could not write `*.md` files. The SPEC (`SPEC.md`, commit 0557e20) was committed before any scored cell.
Code pinned at 590829b `src/`; no `src/` change.

## Design (as pre-registered)

- **Arms:** control `gru_attention` (316,751 params), `gru_small` (77,138), `linear_indicators` (7,610).
- **Defaults:** today's defaults otherwise (D-057, D-058). `DIRECTION_DEEP_ZERO_INIT` is off for all arms.
- **Layout:** micro layout, `N_FOLDS` 100, newest 1.5M sequences, `VAL_FRACTION` = `CAL_FRACTION` = 0.02,
  `BATCH_SIZE` 1024, `EPOCHS` cap 14, seed 0. loss_prune_v1's folds were not reused, because they informed a
  default (D-057).
- **Judgement folds -96..-92:** out-of-sample blocks 2023-01-13 to 2023-03-05, earlier than anything any choice
  used. 5 pairs per verdict (D-046).
- **Primary metric:** h1 direction AUC. Minimum effect 0.01; non-inferiority margin 0.01.
- **Guard-rails (paired):** CRPSS at h0, h1, h2 (margin 0.005); AUC at h0 and h2 (0.01); h1 Brier (0.002, lower is
  better).
- **Comparator:** NT-032: mean of the 5 per-fold differences, paired t interval, alpha 0.05, no multiplicity
  correction.
- **Decision rule:** (1) arm better: it beats control on h1 AUC and every paired guard-rail passes;
  (2) not worse and cheaper: non-inferior on h1 AUC and every guard-rail passes; (3) capacity helps: control beats
  the arm, or non-inferiority is breached; (4) inconclusive: anything else.

## Verdicts

| arm vs control | h1 direction AUC, arm minus control | verdict |
|---|---|---|
| gru_small | -0.0121, 95% CI [-0.0241, -0.0001], negative on 5/5 folds | **Inconclusive** (rule 4): the CI upper bound is not below -0.01 (so not "capacity helps") and its lower bound is below -0.01 (so not "not worse"). |
| linear_indicators | -0.0262, 95% CI [-0.0551, +0.0027], negative on 5/5 folds | **Inconclusive** (rule 4). |

Nothing supports changing `MODEL_NAME`. The SPEC allows an inconclusive study one re-run before the owner decides.

### Paired guard-rails (arm minus control)

| metric (margin) | gru_small | linear_indicators |
|---|---|---|
| h1 CRPSS (0.005) | +0.0011 [-0.0088, +0.0111] undecided | -0.0049 [-0.0186, +0.0087] undecided |
| h0 CRPSS (0.005) | +0.0008 [-0.0167, +0.0182] undecided | -0.0105 [-0.0243, +0.0032] undecided |
| h2 CRPSS (0.005) | +0.0001 [-0.0139, +0.0140] undecided | -0.0105 [-0.0249, +0.0040] undecided |
| h0 AUC (0.01) | +0.0162 [-0.0078, +0.0403] pass | +0.0320 [-0.0039, +0.0678] pass |
| h2 AUC (0.01) | +0.0034 [-0.0086, +0.0154] pass | -0.0069 [-0.0276, +0.0138] undecided |
| h1 Brier (0.002, sign flipped: positive is better) | -0.0061 [-0.0174, +0.0053] undecided | -0.0054 [-0.0161, +0.0053] undecided |

With 5 folds every interval is wider than its margin, so "undecided" is expected.

## Per-arm numbers (mean over 5 folds, fold SD in brackets, seed 0)

| metric | control | gru_small | linear_indicators |
|---|---|---|---|
| h0 AUC | 0.4970 (0.023) | 0.5132 (0.017) | 0.5290 (0.010) |
| **h1 AUC** | **0.5347 (0.023)** | **0.5226 (0.020)** | **0.5085 (0.020)** |
| h2 AUC | 0.5227 (0.018) | 0.5261 (0.013) | 0.5158 (0.019) |
| h0 / h1 / h2 CRPSS | 0.0226 / 0.0187 / 0.0208 | 0.0234 / 0.0198 / 0.0209 | 0.0121 / 0.0137 / 0.0104 |
| h0 BCE | 0.7679 (0.041) | 0.7233 (0.022) | 0.7246 (0.038) |
| h1 BCE | 0.7059 (0.008) | 0.7187 (0.023) | 0.7175 (0.024) |
| h2 BCE | 0.7257 (0.012) | 0.7209 (0.027) | 0.7177 (0.026) |
| h1 Brier | 0.2550 | 0.2610 | 0.2603 |

- The `logreg_lags` baseline has h1 AUC 0.5523 on the same blocks, above every arm.
- **Secondary paired BCE differences (arm minus control; descriptive, post hoc, not gated, no multiplicity
  control):** h0: gru_small -0.0446 [-0.0702, -0.0190], linear -0.0433 [-0.0632, -0.0235]; h1: +0.0128
  [-0.0142, +0.0399] and +0.0116 [-0.0161, +0.0393]; h2: -0.0048 [-0.0271, +0.0175] and -0.0080
  [-0.0282, +0.0122]. Only the h0 interval excludes 0; control's h0 AUC is below chance (0.497), so part of
  its h0 BCE gap is over-confident h0 probabilities.
- **Absolute guard-rails:** coverage90 passes in all 45 cell-horizons (0.892 to 0.935, band [0.85, 0.95]);
  `nonfinite_grad_steps` is 0 in all 15 cells.
- **Pre-clip gradient norm** (mean over epochs, descriptive, `GRAD_CLIP_NORM` 20): control 22.7, gru_small 9.8,
  linear 3.6. Clipped main steps per cell: control 17-98, gru_small 7-18, linear 4-5.

### Skip and tower share of the direction-logit variance (NT-110 covariance shares; skip + tower = 1; mean over 5 folds)

| arm | h0 skip / tower | h1 skip / tower | h2 skip / tower |
|---|---|---|---|
| control | 0.784 / 0.216 | 0.140 / 0.860 | 0.318 / 0.682 |
| gru_small | 0.514 / 0.486 | 0.644 / 0.356 | 0.641 / 0.359 |
| linear_indicators | 0.912 / 0.088 | 0.631 / 0.369 | 0.660 / 0.340 |

Per-cell values are in `summary.json`; `corr_skip_tower` runs from -0.01 to -0.49, all negative.

## How to read the verdict

1. **Epoch cap confound.** Control stopped early in 3 of 5 cells (11 epochs, best epoch 5). Both small arms ran
   all 14 epochs in every cell, with the best epoch at or near the cap (linear 11-14, gru_small 8-14). The cap came
   from the GPU budget and is the same for every arm, but it probably under-trains the small arms. The negative h1
   estimates are therefore not a clean statement about capacity. A re-run at `EPOCHS` 20 would cost about
   +1 GPU-hour for the small arms.
2. **Power.** 5 folds x 1 seed. Control's h1 AUC fold SD is 0.023, twice the 0.01 minimum effect, so this design
   cannot decide at 0.01. More folds shrink the interval; more seeds do not (D-046).
3. **Retried cell.** `gru_small__f-93__s0` trained fully and then the scorer died with a host `MemoryError`
   (5.14 GiB array; run `20261006T085757Z-61014d0-78fa44ab-gru_small__f-93__s0`, kept and not scored, D-029).
   The same cell (same SPEC, code and seed) was retried with `--retry-failed`:
   `20261006T105223Z-61014d0-78fa44ab-gru_small__f-93__s0`. No criterion changed.
4. **Fixed per-cell cost** dominates GPU time: about 250-490 s of data windows, calibration and scoring per cell,
   against 22-33 s per epoch.

## Budget and evidence

- **GPU time:** 2.35 GPU-hours against the 3 h cap (1.99 h for the 16 cells, 15 done plus 1 failed; 0.36 h for the
  timing cells). The SPEC worst case was 2.88 h. One GPU job at a time, with a GPU-free check before every cell.
- **Commits (branch nt-104-ab):** SPEC, configs and timing cells 0557e20 (before any scored cell); cell driver
  61014d0; results c084048; remaining light files de8ee95.
- **Evidence:** `SPEC.md`, `summary.md`, `summary.json`, `summarize.py`; `configs/scenarios/capacity_v1.yaml`;
  `configs/compares/capacity_v1_*.yaml`; `runs/compares/capacity_v1_*/result.json` and `report.md`. Heavy files
  (npz, weights) are machine-local in the experimenter's worktree `D:/nt/nt_wt_104ab`: do not remove it without
  keeping them. Timing cells (discarded, 2 epochs on fold -92): `20261006T072756Z-590829b-211b6c0c-control__f-92__s0`,
  `20261006T073538Z-590829b-e6f31e0d-gru_small__f-92__s0`, `20261006T074438Z-590829b-61336ed5-linear_indicators__f-92__s0`.

## Follow-ups

- NT-104 (3) closes as done with an inconclusive verdict. The default `MODEL_NAME` stays `gru_attention`.
- If a decision is wanted, one allowed re-run: more judgement folds (up to 96 positions exist in this layout),
  `EPOCHS` 20, within 3 GPU-hours, with a new SPEC.
- Skip against tower is a hypothesis for NT-103 and NT-106, not a result: control reads h1 mostly from the deep
  tower (86%), the small arms mostly from the linear skip.
- Future SPEC budgets should again use a timing cell and a measured `sec_per_step` per arm; this SPEC overestimated
  by about 20%.
