# Recent market: is there any direction signal left? REPORT (2026-10-11), against SPEC.md

Protocol, picks and rules were fixed in SPEC.md (commit f31aba9) and the picks in selection.json (commit 8195477) before the look
months were computed. Full selection tables: `selection_table.txt` (84 configurations, 42 selection months 2021-01..2024-06).
Look: `look_table.txt`, `look.json`. Rows: `sel_*.jsonl`, `look_*.jsonl`. Tests: `test_recent.py` (3 passed, 1.3 s).

## Answer

- **Absolute signal: yes, small.** On the 15 look months (2024-07..2025-09) every pick has mean monthly AUC above 0.5 with a 95% interval
  excluding 0.5: 1-minute 0.530-0.534 (intervals about [0.524, 0.540]); 1-hour 0.542-0.548 (about [0.53, 0.565]).
- **Gain from "everything we built": 1-minute small, 1-hour none.** Paired AUC difference against the tb7-logistic baseline of the same
  training length (95% t-interval over 15 months; Bonferroni 98.3% over 3 looks in brackets):
  - 1m `all:l1:365` +0.0042 [+0.0012, +0.0072], success (98.3%: [+0.0004, +0.0080], also above 0)
  - 1m `all:hgb:365` +0.0050 [+0.0002, +0.0098], success at 95%, not at 98.3% ([-0.0011, +0.0111])
  - 1m `gR:l1:365` +0.0017 [-0.0019, +0.0053], no
  - 1h `rich:logreg:365` -0.0002, `all:hgb:365` -0.0054, `rich:l1:365` +0.0002: all intervals contain 0, no success.
  The 1m gain is a few thousandths of AUC, on months that overlap in their training windows (optimistic intervals).
- **Is it tradable? Not on this evidence.** Honest top-10% (month's own confidence threshold) on the look months: 1m hit 0.547-0.548 and
  +0.65..+0.95 gross bps per 15-minute trade (baseline AUC 0.529); 1h hit 0.55-0.567, +3.1..+4.4 bps per hour. Costs are 0 by D-044, but
  these sizes are the order of a taker fee, and drawdown was not measured.
- **Decay is visible within the study:** tb7-logreg 1m AUC 0.5475 on the selection months (2021-24) and 0.5286 on the look months; the best
  1m pick on the selection months (all:hgb:365, 0.5518) is 0.5336 on the look.

## Selection months, what carried the gain (see selection_table.txt)

- 1m, T=365: all-features hgb +0.0044 [+0.0023,+0.0064], all l1 +0.0032 [+0.0014,+0.0051], gR l1 +0.0017 [0.0000,+0.0034]; rich, ctx, hgb of
  single sets about 0. T=60 is worse for every set than T=365 (more features need more rows).
- novel N1-N7 alone: -0.009..-0.020 against tb7 (adds nothing alone); har (volatility + hour/day of week only) AUC 0.505-0.508 (no direction).
- 1h, T=365: rich logreg +0.0064 [+0.0023,+0.0106], rich l1 +0.0051, all hgb +0.0061 [-0.0001,+0.0122]; the look did not confirm them.
- ctx on the 1h cell is bad (0.512-0.523): its scales are in bars, so a "1440-bar" scale is 60 days.

## Deviations and caveats

- Purge stricter than decay_lib (a training label must CLOSE by the month start); the 1m tb7-logreg selection mean (0.547 at T=60) matches
  the earlier decay runs' level (0.54-0.58 for 2021-23).
- Picks were taken per cell (3 + 3 looks); 6 looks in total, so one success at 95% in three of six is weak evidence on its own.
- L1 is slow on 284 features (all:l1:365: 17 min for 42 months); features clipped at +-8 after standardisation for every set.
