# T15: three unattended days of CPU experiments on the M7/M9 finding

Lead, 2026-10-18, committed BEFORE any run. Owner away for three days; the machine should do useful, resource-heavy work.

## Why this work
M7 (LightGBM on 12 regime keys) passed both targets on both looks. M8 showed the gain sits in the UTC hour x weekend profile and |stretch|. M9h6 is a compact linear recovery, labelled post hoc. Open question: does the compact module hold up when the samples are tilted, and does the gain vanish when the hour is shuffled?

## Limits (hard stops)
- CPU only, below-normal priority, at most 6 threads.
- GPU: 0 (shared with another session at 100%). If it frees up, nothing is started on it.
- Stop the whole run if free RAM < 8 GB or free disk on D: < 20 GB.
- Wall-clock cap 72 h. Progress file rewritten after every job.
- Data: the cut file only up to 2025-07-20 23:59; the Binance file only up to 2026-09-30 23:59. Never read 2025-07-21..2025-09-30. No October-2026 target.
- Nothing is changed in the frozen artefacts (M7c, M9h6). Nothing is chosen on the looks.

## Day 1: stability of the compact module (M9)
- Sub-sample checks on Binance: drop Nov and Dec 2025; drop the first 3 months; rolling 4-month windows.
- Harmonics sweep: 4, 6, 8, 12 hour harmonics plus weekend, versus the full 24-slot profile. Report each one's pooled and per-look gain over V3 and H3.
- Pre-registered rule (fixed here): a harmonic count is "stable" if its A increment has a monthly t-interval above 0 in at least 3 of 4 rolling windows.

## Day 2: falsification and controls
- Hour shuffle: permute the UTC hour across anchors within each month. The M9 gain must disappear; if it does not, the gain is not about the hour.
- Weekend shuffle: same test for the weekend flag.
- Pool sensitivity: Bitstamp-only pool versus Binance-only pool versus both, for M7 and M9h6.
- Start-date sensitivity: training start shifted by 6, 12, 18 months where history allows.
- Pre-registered rule: the gain survives a control only if its paired CI over months excludes 0.

## Day 3: summary and recommendation
- One table: each variant, its controls, its stability and its verdict.
- Recommend at most one module for the prospective freeze, chosen by the rule written in the SPEC for M9 (min recovery ratio across looks, ties to fewer columns). If no module passes Day 2's controls, recommend none and say so.
- Update the morning report's prospective-bets table; do not change the frozen files.

## Outputs
`runs/tactical/tasks/T15_three_day/`: `progress.md` (rewritten after each job), per-day results, tables, logs. Nothing is pushed until the final summary.
