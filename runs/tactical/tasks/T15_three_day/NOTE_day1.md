# T15 Day 1 note: stability of the compact module (written BEFORE any Day 1 run)

SPEC.md (6659bc3) is unchanged. Executor: experimenter sub-agent for the lead, 2026-10-18. CPU only, below-normal priority, BLAS 4 threads / GBM 4 threads, GPU 0.
Harness `T14_multimodal/core` unchanged, run on a COPY of the cache (`D:/nt/nt_t15_cache`: anchors, base, bars, desc of both sources); no CSV is read.
Frozen artefacts (M7c, M9h6, `frozen_P*.json`, every `T14_multimodal/*/results.json`) are only read.

## Views (all row-wise from the UTC timestamp and the M6 keys; same M7 row set `ok`: harness ok anchors with all 12 keys and finite labels)
`T15_h{K}` for K in {4, 6, 8, 12}: sin_j, cos_j (j = 1..K, hour = integer UTC hour of `ta`), `wkend`, `wkend * sin_j`, `wkend * cos_j`, plus the 4 stretch terms
(|str20|, |str60|, str20^2, str60^2). This is the M9h6_prof_str recipe with K varied. For K = 12 `sin12` and `wkend*sin12` are identically 0 at integer hours and are
omitted (so K = 12 has 23 + 1 + 23 + 4 = 51 columns, the same span as the 23 hour dummies of M9_prof_str). Column counts: K=4: 21, K=6: 29, K=8: 37, K=12: 51.
References, stored and not re-run: `M9_prof_str` (the full 24-slot profile with weekend interaction, 51 columns), `M9h6_prof_str`, `M9h3_prof_str`, `M7_gbm`, `M7c_gbm_resid`.
Reproduction check: `T15_h6` must reproduce the stored `M9h6_prof_str` monthly series (max abs difference < 1e-9); reported either way.
Evaluation: `core/evaluate.py` unchanged (Look 1 2024-07..2025-07, Look 2 2025-11..2026-09, n_perm 5, n_boot 10000), per view and look as in T14.

## Readout (descriptive, no choice is made on it)
Per view: the A increment over V3 and the B relative Brier gain over H3, per look and pooled over the 24 months (12 + 11... the months the harness scores), with the harness
flags and controls; recovery ratios R_A, R_B against `M7_gbm` pooled (T14 M9 definition: pooled mean of the monthly series over the M7_gbm pooled mean).

## Sub-sample checks on Binance (Look 2, 11 months 2025-11..2026-09), from the stored monthly series of each view (no refit; the walk-forward months are as scored)
(a) drop Nov and Dec 2025 (9 months); (b) drop the first 3 months, Nov, Dec, Jan (8 months); (c) 4 rolling windows of 4 months with 0-based start months 0, 2, 5, 7:
[2025-11..2026-02], [2026-01..2026-04], [2026-04..2026-07], [2026-06..2026-09] (starts evenly spread over the 8 possible starts, covering every month; fixed now).
Each: mean and 95% two-sided t-interval of the monthly series, for A and B.

## Pre-registered rule (fixed here, from SPEC)
A harmonic count (a view `T15_h{K}`, and the stored h6 and the full 24-slot profile as comparators) is **stable** if its A increment has a monthly t-interval whose lower
bound is above 0 in at least 3 of the 4 rolling windows (each window has 4 months, t quantile with 3 df = 3.182). The same count for B is reported as "B-stable",
descriptive only. Windows with a month the view lacks use the months it has. Nothing in Day 1 selects a module: Day 2 runs on the frozen modules M9h6 and M7c whatever
Day 1 shows; Day 3's recommendation uses Day 2's controls and the M9 freeze rule.

## Guard-rails
Stop everything if free RAM < 8 GB or free D: < 20 GB (checked before every job), wall-clock cap 72 h; progress.md is rewritten after every job.
