# Candidate check: the 0.805 AUC trial (tactical, fixed before any run)

Owner, 2026-10-07: "take 1 candidate, the one with the highest score, 0.8 or more, and check it; formalise the
requirements". Record: [docs/qa/2026-10-06-tactical-session.md](../../../docs/qa/2026-10-06-tactical-session.md) round 3.

## The candidate

The highest single score of all tactical runs: `hc2_skiponly_c4`, configuration `DIRECTION_HEAD_MODE: skip_only`
(the direction logit comes only from the linear skip over lag features), slice `DATA_END 2022-04-19T07:00`, seed 2:
**h1 direction AUC 0.805** (h0 0.592, h2 0.666; mean 0.688). On the same block and seed the base scored 0.710 (mean).

Known before the check (not a result of it):
- That validation block is Saturday 2022-04-16 11:01-18:30 UTC (Easter weekend, US markets shut, range 0.52%).
  The model-free rule "fade the last 10 bars" scores AUC 0.756 on it (`runs/tactical/slice_2022_04_19.json`).
- Over 40 slices x 3 seeds, skip_only vs base is +0.0075, 95% CI [-0.007, +0.022]: no effect (LOG H6).
- One head on one block has about 14-24 independent points: pure noise reaches 0.8 in about 1% of values.

## Hypothesis

H: the 0.805 is a property of the configuration, not of one lucky (block, seed). It must (C1) repeat on the same
block with new seeds and (C2) carry over to new blocks at the 7-day scale.

## Checks and pass rules

All 9 outputs are measured in every run (owner: "always use the heads"): per horizon the delta head (corr, skill vs
zero), the direction head (AUC, Brier, log loss, hit rate) and the variance head (CRPSS vs a constant variance,
NLL, 90% coverage and width, Spearman variance vs squared error). Predictions are saved.

| check | design | pass rule (all parts) |
|---|---|---|
| **C1: same block, new seeds** | screen layout (360 train / 450 val), slice 2022-04-19T07:00, seeds 3-12 (10 new), skip_only and base | (a) skip_only mean h1 AUC over the 10 seeds >= 0.75; (b) paired diff vs base (mean of h0-h2) has its 95% t-interval over seeds above 0; (c) skip_only mean of h0-h2 >= 0.756, the fade-10-bars rule on the same block |
| **C2: new blocks, 7-day scale** | 7-day training block (10,080 windows; MAX_SEQUENCE_COUNT 126,000, batch 256, 8 epochs), the owner's tactical design 6 slices x 2 seeds, skip_only vs default, paired by (slice, seed) | (a) 95% t-interval over the 6 per-slice mean diffs of direction AUC (mean of h0-h2) above 0; (b) guard-rails: mean CRPSS (h0-h2) and mean 90% coverage of skip_only not lower than default by more than 0.05 |

C2 slices (6 of the 40 climb slices, evenly spread; the first climb slice, 2017-03-01, is skipped because a 7-day
layout needs about 87 days of history before DATA_END): climb[1::7] = 2017-05-02T12:00, 2018-11-16T02:00, 2020-03-30T04:00, 2021-10-13T18:00, 2023-04-29T08:00, 2024-11-11T23:00, fixed with this file.
The 10 final slices stay untouched.

## Verdicts

- C1 and C2 pass: a real candidate. Next: the 40-slice climb design at screen scale, then the 10 final slices, then
  a note in the journal's "For the MVP lead" section (no default changes from the tactical session).
- C1 passes, C2 fails: the configuration is good on that kind of block (a quiet holiday range) only; recorded as a
  regime finding.
- C1 fails: the 0.805 was one lucky seed on a lucky block; closed.

## Budget and limits

C1: 20 screen trials (about 20 minutes). C2: 24 seven-day runs at 7.5-9 minutes each, 3 at a time (about 70
minutes). The 7-day runs exceed the 2-minute rule; the owner allowed 7-day runs for this point on 2026-10-07.
