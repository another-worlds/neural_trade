# T15 Day 3 note: summary and recommendation (written BEFORE the Day 3 computation; Day 2 is committed at bbcfb59b)

Day 3 runs nothing on the harness. It reads `day1/results_day1.json`, `day2/results_day2.json` and the stored T14 results, and writes `day3/summary_table.txt|json|md`,
`day3/recommendation.md`, `day3/prospective_bets_update.md` (a proposed replacement for the morning report's prospective-bets rows; the report itself, the dashboard,
LOG*.md, PLAN.md, TACTICAL.md and the frozen files are NOT edited, by the lead's instruction for this run).

## The table (one row per variant)
Variants: M7_gbm, M7c_gbm_resid, M9h6 (= M9h6_prof_str), the other harmonic counts T15_h4, T15_h8, T15_h12 and the full 24-slot profile M9_prof_str.
Columns: number of columns; pooled-24 gain A and B with t-CI; R_A, R_B, min R (against M7_gbm, pooled) and the per-look ratios; Day 1 A-stable (>= 3 of 4 windows), windows count; Day 2:
C-H, C-W survival per target, C-P and C-S survivals with the untestable ones listed, the per-target and module Day 2 pass; eligibility; verdict. Variants without Day 2 runs are marked
"controls not run" (Day 2 ran on the three modules fixed in NOTE_day2 only; nothing was added after results).

## Recommendation rule (as in NOTE_day2, unchanged)
Eligible = frozen artefacts (M7c, M9h6) that pass Day 2. Recommended = highest min(R_A, R_B) (pooled-24 mean gain over M7_gbm's), ties (< 0.005) to fewer columns, via `m9_lib.choose_freeze`.
No eligible module: recommend none. The alternative reading (min over target x look) is shown. Descriptive additions that do not change the verdict: for each module the count of testable controls
passed out of all defined, and the list of controls that were untestable (fewer than 5 months), so the strength of a pass is visible.
