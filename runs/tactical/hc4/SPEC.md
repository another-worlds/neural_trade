# Hill-climb on the 7-day block (tactical; fixed before round 1)

Owner, 2026-10-07: "now let's form the hill-climb", after choosing the aggregated metric ("3 groups equally, by
noise") and "1 day if it catches up" (it did not: journal H11), so the block is 7 days.

**Metric and verdict.** `runs/tactical/hc4_metric.py`: price, direction and confidence groups at 1/3 each, every paired
difference in seed-noise units, composite per slice, 95% t-interval over slices. BETTER = interval above 0, no group
below -0.5, and the anti-collapse guard (direction AUC and the variance~error ranking each >= -0.5). WORSE = interval
below 0. Otherwise NO DIFFERENCE.

**Design.** 7-day training block (10,080 windows), batch 256, 8 epochs, all 9 outputs, predictions saved. The owner's
tactical design: 6 slices x 2 seeds = 12 runs per variant, 3 processes at once (about 7 minutes a run, about 30 minutes
a variant). Climb slices = the candidate check's C2 slices (2017-05-02, 2018-11-16, 2020-03-30, 2021-10-13, 2023-04-29,
2024-11-11); held-out slices = climb[4::7][:6] of make_hc2.py, used only once per winner. The test fold is never used.

**Rounds.** Round 0 base = the default network (the C2 default runs, `cand_c2_base`). Each round tries one change per
variant against the current base. The best BETTER variant becomes the next base (changes stack); a round with no
BETTER ends the Config part of the climb and the next round needs new code (an implementer item). A winner of the
whole climb is checked once on the held-out slices with the same rule.

**Round 1 variants** (one change each, existing Config keys only):
| name | change | why |
|---|---|---|
| calval | run.calibrate on, CALIB_MODE value | the production path calibrates loss weights; screen did not |
| calgrad | run.calibrate on, CALIB_MODE gradient | NT-101's gradient-norm balancing |
| ep12 | 12 epochs instead of 8 | the 7-day net may be under-trained |
| look120 | LOOKBACK 120 instead of 60 | a longer input window at a scale that can learn |
| nophys | the six physics terms at 0 | D-003 keeps them; this only measures them, it changes no default |
| bs1024 | batch 1024 | fewer, larger steps; checks the collapse guard at 7 days |

**Limits.** Runs exceed the 2-minute rule (7-day block, owner's choice of block). No default changes from here.
