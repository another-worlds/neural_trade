# T14 M9 addendum 1 (written AFTER the pre-registered and M9x results; committed BEFORE the views below were built or run)

SPEC_M9.md is unchanged. What the pre-registered views showed (stored in `tables_m9.txt` at the commit of this addendum): on Look 1 the dummy-profile views recover M7
(R_A 1.09-1.58, R_B 1.11-1.18 on Look 1) but on Look 2 they fail, and the cause is in the monthly series: the harness' Look 2 starts training at 2025-10-02, so
Nov 2025 trains on 384 rows and Dec 2025 on 1,104, and the 47-57 extra columns blow up (A increment -0.136 and B -13.8% in Nov 2025; the later months are positive).
M7's GBM is immune because its pool uses both source files. So the failure is a parameter-count problem at small n, not a missing profile shape. The SPEC's M9x menu
(more columns) makes it worse, as it should. This is post hoc and labelled so.

## Post hoc views `M9h*` (low-dimension profile; same rows, same harness, same controls)
Harmonics replace the 23 hour dummies: sin_k, cos_k = sin / cos(2 pi k hour / 24), k = 1..K (hour = integer UTC hour of `ta`), `wkend`, `wk_sin_k`, `wk_cos_k`.
| view | columns | n |
|---|---|---|
| `M9h3_prof_str` | K = 3 (6) + wkend + wk x 6 (6) + the 4 stretch terms of M9_prof_str | 17 |
| `M9h3_prof_str_vp` | + the 6 vp columns of M9_prof_str_vp | 23 |
| `M9h6_prof_str` | K = 6 (12) + wkend + wk x 12 (12) + the 4 stretch terms | 29 |
| `M9h6_prof_str_vp` | + the 6 vp columns | 35 |
(M9_prof alone is not repeated: the stretch terms carry A.) Nothing else is tried. Recovery ratios, flags and R labels exactly as in SPEC_M9.

## Sensitivity (descriptive, no flag, no choice)
Look 2 with the months that have fewer than 1,500 training rows removed (Nov and Dec 2025 in the harness): the pooled and per-look gains and R of every linear
view, from the stored monthly series, with the t-interval over the remaining months. It says what the views do once the sample is not tiny; it is NOT a pass.

## Freeze (amends nothing pre-registered)
The pre-registered rule is applied first and, because no pre-registered view passed, then to the post hoc views. The M9h* views join the post hoc pool (M9x_B, M9x_A,
M9x_AB, M9h*). The freeze artefact carries the label "post hoc" and the prospective test is the real judge. If no view in the pool passes, nothing is frozen.
