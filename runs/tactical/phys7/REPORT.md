# phys7 report: physics terms and the variance head (tactical, shrunk; SPEC.md, pinned code 3702d1b)

Setup: BTC/USDT 1-minute, static excerpt, hc4 long-block layout (126,000 windows, fold -2, batch 256), horizons 10/15/20 bars,
4 epochs, calibration off, 6 climb slices x 1 seed x 4 cells = 24 runs. Metrics on the fold -2 OOS block (12,600 windows per
slice); latest data end 2024-11-11, cutoff 2025-07-20 asserted in the analysis; the test fold is never scored. The log-variance
head does not exist as a switch, so the 2nd axis is `PRICE_HEAD` none / on (SPEC explains: with `none` only t_perp and vacuum
overflow are alive).

## Verdict (pre-registered rule, SPEC "Verdict rule")

* **Primary (PRICE_HEAD none; t_perp + vacuum overflow): HELPS, small.** Physics on minus off, mean-of-horizons Spearman
  +0.0196, 95% interval [+0.0019, +0.0373], 6 of 6 slices up (each horizon positive; h2 interval above 0, h0 and h1 just
  include 0). Rule needs interval > 0 and mean >= +0.010: met. QLIKE guard (not worse than +0.05): +0.043 [-0.25, +0.33], met, but
  QLIKE shows no gain (2 of 6 slices better). The lower bound is 0.002, the effect is about half the pre-stated minimum
  detectable effect (0.037) and no multiplicity correction is applied: read it as a weak positive on the ranking axis, not as a
  firm result.
* **Secondary (PRICE_HEAD on; all five default-on terms): NO DETECTABLE EFFECT.** Spearman +0.0375 [-0.017, +0.092], 4 of 6
  slices up; QLIKE -0.002 [-0.051, +0.047]. The point estimate is larger but the spread is 3 times wider.
* **Distance to HAR+tod** (same data, same scaling rule): net minus HAR Spearman, physics on -0.050 [-0.091, -0.009] (none) and
  -0.055 [-0.093, -0.017] (price head); physics off -0.070 [-0.116, -0.024] and -0.093 [-0.172, -0.013]. Physics closes roughly
  a quarter to a half of the gap but the net stays below HAR on ranking in every cell. QLIKE after re-scaling: net 2.57 (none,
  on), 2.53 (none, off), 2.70 (price, both) against HAR 2.39: net worse by 0.14 to 0.31, intervals wide (cells with physics on
  include 0).

## Numbers (mean over 6 slices)

| cell | net Spearman | HAR+tod Spearman | net QLIKE (B) | HAR QLIKE |
|---|---|---|---|---|
| none, physics on | 0.2336 | 0.2840 | 2.573 | 2.394 |
| none, physics off | 0.2140 | 0.2840 | 2.530 | 2.394 |
| price head, physics on | 0.2290 | 0.2840 | 2.702 | 2.394 |
| price head, physics off | 0.1915 | 0.2840 | 2.704 | 2.394 |

Full tables including per-horizon intervals: `report_tables.txt`, `report_tables.json`; per-run scores
`scores_phys7_<cell>.json` (trial keys = `runs/tactical/screens/phys7_<cell>/results.shard-*.jsonl` and untracked
`preds/<key>.npz`). Run ids: screens `phys7_none_on`, `phys7_none_off`, `phys7_price_on`, `phys7_price_off`, 6 trials each.
No run failed, no constant variance output, no non-finite step.

## Precision and caveats

1 seed, 4 epochs, 6 slices. The pre-run noise estimate (sd 0.035 of a single-run paired difference, MDE 0.037) was larger
than what these runs showed (sd 0.017 in the primary pair), so the primary interval is tighter than planned but still has a
lower bound of 0.002. HAR is fitted on the same data segment (about 26,000 to 120,000 bars before the OOS block, limited by
the excerpt), and the scale c for both is fitted on the first half of the OOS block (the cal block is not saved by the screen), so
QLIKE is on the second half only. 4-epoch nets are under-trained; a converged net could differ. Slice 2018-11-16 is the worst
slice for the net in every cell (Spearman 0.11 to 0.13 against HAR 0.24).

## GPU time

Wall clock of the study 2.08 h (18:14 to 20:19, two shard processes at a time, below-normal priority, 4000 MB cap); summed
process time 4.09 h (mean 610 / 571 / 709 / 563 s per run; the SPEC's serial estimate of 1.5 h was too low because two
processes share the GPU). Within the 3 h stop rule counted as wall time, over it counted as process time: stated for the lead.

## For the backlog

* If a firm answer is wanted: 3 seeds and 8 epochs on the primary pair only (about 12 more runs, about 2 h wall) would settle
  whether +0.02 is real. Not started.
* A log-variance head switch (not in `Config`) would be an implementer item before the original 2 x 2 can be run.
* The physics terms are not a cost to the variance head in these runs (no cell where physics off is better); the owner alone
  decides D-003.
