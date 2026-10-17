# T14 M9: a plain profile module (ridge A, logistic B) that should recover M7 (SPEC, committed BEFORE anything runs)

Tactical sprint T14, candidate M9, 2026-10-17, CPU only. Parent spec `../SPEC.md` and `../M8_module/SPEC_M8.md` unchanged. Harness `../core` unchanged.
Branch `nt-tactical-t14-m9`. Builds on the M8 finding: M7's gain sits in the UTC-hour + weekend group (A 62%, B 92% share) and, for A, the stretch keys (47%);
S/R distances and the last-3-hours log RV add nothing; eight cheap terms recovered 51% of A and none of B because B needs the hour/weekend SHAPE.

## Hypothesis
A full 24-hour profile with a weekend interaction (plus |stretch| terms) in the plain ridge (A) and logistic (B) recovers most of M7_gbm's gain.
Note (stated now): V3 already contains 96 UTC 15-minute slot dummies, so for A the weekday hour profile is already in the base; the new content of the
profile views for A is the weekend flag and weekend x hour. H3 has only 4 hour harmonics, so for B the hour dummies are new too.

## Setup (shared)
- Rows: the M7 row set `ok` (harness ok anchors with ALL 12 M6 keys finite and finite labels), the same rows in every view (11,872 on Look 1, 8,346 on
  Look 2 as in M7_gbm), so gains are comparable. Evaluation: `evaluate.py --view <name>` per view, Look 1 (2024-07..2025-07) and Look 2 (2025-11..2026-09),
  n_perm 5, n_boot 10000. Never read: 2025-07-21..2025-09-30, anything after 2026-09-30 23:59, the original Bitcoin_BTCUSDT.csv, any October-2026 target.
  Data only through the harness caches. Nothing is chosen on a look except by the freeze rule below, which is written here and only picks among the
  pre-registered views.
- All columns are causal: functions of the anchor timestamp `ta` (UTC) and of the M6 keys at the anchor (which use bars strictly before `ta`).
  Row-wise: a row's columns depend on no other row.

## Pre-registered views (column order fixed in `m9_lib.py`)
| view | columns | n |
|---|---|---|
| `M9_prof` | hour dummies h1..h23 (UTC hour of `ta`, hour 0 dropped); `wkend` (ta.dayofweek >= 5); `wkend * h1 .. wkend * h23` | 47 |
| `M9_prof_str` | M9_prof + `abs_str20`, `abs_str60`, `sq_str20`, `sq_str60` (the M6 keys str20, str60) | 51 |
| `M9_prof_str_vp` | M9_prof_str + `vph_c = vp_h - 0.5`, `vpd_c = vp_d - 0.5`, `vph_c*hsin`, `vph_c*hcos`, `vpd_c*hsin`, `vpd_c*hcos` | 57 |
| `M9_gbm7` | M7's LightGBM scheme (`m7_lib.monthly_view`, same parameters, 365-day pool, both-source pool, monthly walk-forward) restricted to the 7 keys hsin, hcos, wkend, str20, str60, vp_h, vp_d; output columns as M7_gbm (`gbm_dA_hat, gbm_hit_hat, gbm_dA_rank, gbm_hit_rank`) | 4 |

C2b (`--c2-view`): for `M9_gbm7` the stored `M7_kernel` (as M7_gbm was run); the three linear views have no C2b, the harness C1
(permuted within month, 5 seeds) and C2a (S3 duplicated) are their controls.

## References (stored, not re-run)
`M7_gbm` (nonlinear reference), `M7c_gbm_resid`, `M8_lin`, and the harness controls C1 / C2a. Also shown for context: `M8x_lin_wk`, `M8x_abl_hour`, `M8x_abl_wkend` (post hoc of M8).

## Readout
- Per view, target and look: the harness numbers (A: increment over V3 with the monthly t-interval; B: relative Brier improvement over H3 with the day-block CI),
  controls, harness flags.
- Pooled gain G_T(view) = mean over the 24 pooled months (12 of Look 1 plus 12 of Look 2, those the harness scores) of the monthly gain series
  (A `increment_series.view`; B `monthly_rel.view`), as `analyse_m8.py` does. Recovery ratio **R_T = G_T(view) / G_T(M7_gbm)**, T in {A, B}; also per look
  (R_T,look). Also the paired difference M7_gbm minus view over the 24 months (t-interval), for context.
- **Pass of a view on a target** = the harness flags `pass` on BOTH looks: A increment >= +0.005 with CI above 0 and above C1, C2a (and C2b where present);
  B relative Brier >= 1% with day-block CI above 0 and above C1, C2a (and C2b where present). **A view passes** if it passes either target (as in the parent SPEC).
  "Recovers M7" if min(R_A, R_B) >= 0.75 (pooled); "partly" if 0.40 <= min < 0.75; "no" if < 0.40.
- No multiplicity correction (descriptive; the parent Holm was over the six candidates and M7 already carries it). The numbers are not used for any choice other
  than the freeze rule.

## Post hoc improvement (labelled `M9x_*`, run only if the best pre-registered linear view has min(R_A, R_B) < 0.75)
The menu is fixed now (at most these three views; each built on `M9_prof_str_vp`; nothing else is tried):
- `M9x_B`: + 7 three-hour blocks (UTC hours 0-2, 3-5, ..., 21-23, the first block dropped) x (`vph_c`, `vpd_c`) = 14 columns, plus `wkend * vph_c`, `wkend * vpd_c`.
- `M9x_A`: + `wkend * abs_str20`, `wkend * abs_str60`, `wkend * sq_str20`, `wkend * sq_str60`, and the 7 three-hour blocks x `abs_str20` = 11 columns.
- `M9x_AB`: both blocks.
They are reported next to the pre-registered result, flagged post hoc, and are NOT eligible for the freeze unless no pre-registered view passes (then the same rule
below, applied to the post hoc views, picks one and the artefact says "post hoc").

## Freeze rule (fixed now)
Among the pre-registered linear views M9_prof, M9_prof_str, M9_prof_str_vp (the GBM7 is a diagnostic, not a module), take those that PASS (above). Freeze the one with
the highest min(R_A, R_B) (pooled); ties (difference < 0.005) to the fewer columns. If none passes, apply the same rule to the post hoc views; if none passes there,
nothing is frozen and the report says so. The "frozen" artefact has no weights: it stores the column recipe (`m9_lib.py`, sha256 of the file and of the recipe
function), the pool window (label-known anchors to 2026-09-30: `ta + 60 min <= 2026-10-01`, 365 days back, ok rows with finite keys: first / last anchor, row count,
sha256 of the anchor times), the harness' own fixed ridge (alpha 1, standardised) and logistic (C 1, standardised) settings, and a final fit on that pool used for
the figure. V3 / H3 and the view's coefficients are refitted monthly by the unchanged harness. `score_P9.py` is modelled on `../M8_module/score_P.py`: withheld with
< 3 complete months; Pocock alpha 0.0294 (one-sided) at the first 3 and the first 6 complete months; criterion A increment >= +0.005 with one-sided monthly t p < 0.0294;
B relative Brier >= 1% over H3 with one-sided day-block p < 0.0294; both above C1 and C2a; pass = either target. It builds the view columns from the new anchor
times and keys by the recipe (no history file needed), tested on synthetic data and a replay on already-used 2026 months; never run on real post-2026-09 data.

## Figure (`fig_m9.png`, `fig_m9.json`, theme.py colours)
(i) the estimated 24-hour profiles, weekday against weekend, of the A target (next-hour change in log RV) and of P(hit) (B) from the final fit of the frozen
view (fitted values of the final ridge / logistic on the pool, mean by UTC hour and weekday / weekend, with the realised means as dots), annotated at the US
open (13:30 UTC in summer, 14:30 in winter: the 13 and 14 UTC hours are marked); (ii) a recovery bar chart: M7_gbm, M7c, M8_lin, the M9 views, per target, per look, as
the gain with its CI and R printed.

## Tests (seconds, `test_m9.py`)
Columns depend only on the timestamp and past keys (mutating keys of rows from the anchor on leaves earlier rows identical; a row depends on no other row); the
hour and weekend dummies are exact on known timestamps (a Saturday, a Sunday, hours 0 and 23, the interaction columns); the three view definitions are nested
and the column names are exactly the lists above; the GBM7 uses exactly the 7 listed keys; score_P9 withholds with 2 complete months, runs the right looks,
Pocock level 0.0294; the frozen recipe hash is reproduced.

## Budget
CPU only, below-normal priority, 4 threads. Estimate: GBM7 about 1 min, each harness evaluation 10-60 s; the whole item a few minutes.
