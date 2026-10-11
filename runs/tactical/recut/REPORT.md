# Tactical recut under a hard data cutoff (H66)

Branch `nt-tactical-recut` (from `nt-tactical` 3702d1b), 2026-10-11. Owner: "пересчитать".

**Cutoff.** No bar after **2025-07-20 23:59** is read anywhere (features, labels, exits, training, scoring). Held-out "looks" are
**2024-07-01 .. 2025-07-20** (13 months, the last one partial: 20 days). Every configuration, selection and protocol is the frozen one;
only the data end changed. All CPU except H57 (GPU, cap 4000 MB, 1 process).

## How the cutoff is enforced

- `runs/tactical/common/cutoff.py`: `CUTOFF`, `CSV` (a physical copy of `Bitcoin_BTCUSDT.csv` whose rows stop at the cutoff, built once to
  `D:/nt/nt_recut_data/`, 4,495,960 rows, last bar 2025-07-20 23:59), `assert_cut`, and `read_csv` (refuses any other file, checks the bound).
  The recomputed loaders point at `C.CSV`, so a later bar does not exist for them. `common/lowrun.py` runs a job at below-normal priority.
- `runs/tactical/recut/patch_cutoff.py` is the patcher used (loaders, chdir to the recut checkout, cache dirs under `D:/nt/nt_recut_cache/`,
  look months to 2025-07). Caches were rebuilt from the cut file (recent, flow, decay, hourly bars, fwd); no old cache is read.
- `runs/tactical/common/test_cutoff.py` (29 tests, pass): the cut file ends at the cutoff; `assert_cut` and `read_csv` refuse later bars; each of
  the 16 recomputed scripts names no raw CSV; each loader (decay_lib 1m and 1h, nolib, volnet_lib, flowlib, recent feats, callib, geom2
  meta and meta_nextopen, hourly) returns a last timestamp <= cutoff (and >= 2025-07-20, so none stops early). With the 35 existing tactical
  tests of the touched modules: 64 passed (`pytest -n 4`).
- Old outputs that were overwritten in place are kept under `runs/tactical/recut/old/`; the pre-recut tree is also branch `nt-tactical`.

**Selection stages are not rerun, and why that is exact.** Every item's selection or dev stage uses only data up to 2024-06. The rebuilt
feature caches are bit-identical to the old ones on every anchor up to 2024-06-30 (`recut/cache_equal_recent.json`, `cache_equal_flow.json`:
tb7, rich, ctx, gR, novel, har, flow, t and r all equal); H62's cache is bit-identical (`recut/check_h62.json`); H57's HAR and ridge rows for
2024-06 equal the stored selection run to the last digit (the nets differ by GPU non-determinism only). So `selection.json` of recent / volnet /
flow and the hourly dev ranking are what the cut data gives. H38 and H40 part 2 reran their dev stage anyway (identical: chosen filter
`trend:hi`, dev excess 26.7355 as logged).

**One deliberate change in method (flagged).** With 13 look months the t-quantile tables in `hourly.py` (`TQ` had no 12-df entry and fell
back to 2.0) and `holdout_min*.py` (`2.14 if 15 months else 2.0`) would understate the intervals; I added 2.18 (t, 12 df) for 13 months.
All other intervals already use `scipy.stats.t`.

## Result table

Old = as logged in LOG.md (to 2025-09-29). New = recomputed. Verdict = the item's own pass/fail statement.

| Item | Old result (LOG) | Recomputed (cutoff) | Verdict old -> new |
|---|---|---|---|
| **H38** hourly barrier, final look, dev winner (rich, hgb, barrier, mag) | n 2,013, hit 53.8%, +9.83 bps, null95 5.69, monthly CI [-11.7,+26.6] (15 mo) | n 1,767, hit 53.9%, **+12.69** bps, null95 6.09, CI [-12.0,+31.7] (13 mo) | FAIL -> FAIL (same) |
| H38 #2 (tb7 logreg sign) | +2.1 bps, null95 5.4 | +0.47, null95 5.83, CI [-15.4,+14.6] | same (fail) |
| H38 #3 (tb7 logreg barrier) | +7.7, null95 6.3, CI [-6.9,+21.5] | +6.46, null95 6.03, CI [-10.1,+22.7] | same (fail) |
| H38 net of costs | "at 10 bps per round trip every one loses" | winner +7.7 at 5 bps and +2.7 at 10 bps (gross +12.7); #2 -4.5 / -9.5; #3 +1.5 / -3.5 | verdict same; the 10-bps sentence is no longer true for the winner (CI still crosses 0) |
| **H40 part 2** regime filter (dev choice: high trend strength) | filtered +15.8 bps vs null95 8.2, CI [-7.6,+33.2]; unfiltered +9.8 | filtered +19.5 vs null95 8.0, CI [-7.1,+38.8], n 1,023; unfiltered +12.7 (null95 7.4), CI [-12.0,+31.7] | FAIL -> FAIL (same); same filter chosen |
| **H48** minute barrier held-out (close entry) | meta +0.54 bps (null95 0.29), monthly +0.54 [+0.16,+0.91], 13/15 months, hit 52.3% | meta +0.54 (null95 0.30), monthly +0.56 [+0.12,+0.99], 11/13 months, hit 52.2%; primary +0.72 [+0.13,+1.37] | PASS -> PASS (same) |
| **H60** H48 with next-open fill | meta +0.30 (null95 0.25), monthly [-0.20,+0.79]; primary +0.28 | meta +0.21 (null95 0.29), monthly [-0.34,+0.79], 9/13 months; primary monthly +0.23 [-0.49,+0.96] | FAIL -> FAIL (same; meta now below its null95 too) |
| **H51** decay, yearly AUC (close entry) | 0.568, 0.590, 0.599, 0.590, 0.549, 0.544, 0.554, 0.531, 0.520; corr -0.67 | 0.568, 0.590, 0.599, 0.590, 0.549, 0.544, 0.554, 0.531, 0.521 (2025 = Jan to Jul 20); corr -0.655 | same |
| **H52** decay map (25 cells; selection 2021-01..2024-06, one look) | every cell 0.58-0.65 (2018-20) -> 0.50-0.54 (2025); 6 best cells 0.557-0.568 -> 0.497-0.538 on the look, baseline 0.523 | 2025 cells 0.494-0.542 (1h/3 0.542, 1m/1 0.539); 6 best 0.557-0.568 -> 0.477-0.540 (1m/1 0.530, 1h/1 0.540), baseline 1m/15 0.524 | same (one 2025 cell, 1h/3 at 0.542 +-0.021, is a hair above the logged "none above 0.54") |
| **H53** vol decay (HAR Spearman) | 0.286 (2018) ... 0.393 (2024), 0.399 (2025); corr +0.41 | 0.286 ... 0.393 (2024), 0.409 (2025); corr +0.412 | same (it grew) |
| **H54** excursion asymmetry | asym AUC 0.58-0.61 (2017-20) -> 0.51-0.52, below the sign's | 0.577-0.606 -> 0.513-0.522 (2021-25; 2025: 0.513); sign 0.518-0.554 | same |
| **H56** 1m all-features L1 vs tb7 logreg (close entry) | +0.0042 [+0.0012,+0.0072]; passes 95% and Bonferroni 98.3% | **+0.0025 [-0.0005,+0.0055]**; fails both | **PASS -> FAIL (changed)** |
| H56 1m all-features boosting (close) | +0.0050 [+0.0002,+0.0098]; 95% only | +0.0042 [+0.00003,+0.0083]; 95% only, marginal (lower bound +0.00003) | PASS(95%) -> PASS(95%), marginal |
| H56 1m gR:l1 (close) | +0.0017, fail | +0.0012 [-0.0029,+0.0054], fail | same |
| H56 1h picks (close) | nothing beats tb7 (look AUC ~0.547) | rich logreg +0.0011, rich L1 +0.0008, all hgb -0.0021, all CIs cross 0 (look AUC 0.546-0.549) | same |
| H56 absolute look numbers | 1m AUC 0.529-0.534, top-10% +0.7..+1.0 bps per 15 min; 1h +3..+4 bps per hour | 1m AUC 0.531-0.534, top-10% +0.6..+1.0 bps; 1h +3.4..+4.5 bps per hour | same |
| **H57** volnet look (ens vs HAR, paired, 95% CI over months) | 15 min: Sp +0.0061 [+0.0047,+0.0074], QLIKE [-0.68,+2.17]; 60: Sp +0.0059, QLIKE [-0.13,+0.51]; **240: Sp +0.0089 [+0.0020,+0.0158], QLIKE +0.0263 [+0.0042,+0.0483]** | run 1: 15: Sp +0.0058 [+0.0043,+0.0074], QLIKE [-0.81,+2.52]; 60: Sp +0.0048 [+0.0026,+0.0070], QLIKE [-0.16,+0.59]; **240: Sp +0.0072 [+0.0001,+0.0144], QLIKE +0.0244 [-0.0038,+0.0526]**. Run 2 (same code and data, GPU non-determinism): 240: Sp [-0.0009,+0.0132], QLIKE [-0.0053,+0.0497]; 15 and 60 as run 1 | 15, 60: FAIL -> FAIL (same). **240: PASS -> FAIL (changed)**, in both runs |
| **H58** equity tables (goal hit > 60%, DD < 5%) | none meets it; closest 1h rich L1: hit 56.7%, +44.5% cum, DD 10.6%, positive through 2 bps; hourly barrier +20.6% at 5 bps, DD 20-25%; minute strategies negative at 1 bps; at most 3/15 months above 60% | none meets it; 1h rich L1: hit 56.1%, +37.3% cum, DD 10.6%, positive through 2 bps (-9.0% at 5); hourly barrier hit 54.1%, +47.1% at 0 and +24.8% at 5 bps, DD 20.0-25.1%; 1m L1 and H48 negative at 1 bps; at most 2/13 months above 60% (3/13 for next-open H38) | same |
| **H61** next-open re-measure | H56 1m boosting fails (+0.0047 [-0.0014,+0.0109]); 1m L1 still passes at 95% (+0.0040 [+0.0004,+0.0076]); H38 +9.8 -> +5.4 bps (below null95 6.3), equity edge +9.0 -> +2.3 bps, DD 28.6%; H48 fails; 1h picks unchanged; "the only surviving effect is a +0.004 AUC sliver at 1 minute" | H56 1m boosting +0.0038 [-0.0021,+0.0098] fails; **1m L1 +0.0023 [-0.0016,+0.0063] fails**; H38 +12.7 -> **+6.95 bps, above null95 6.01** (monthly CI [-19.1,+28.9] still crosses 0), equity edge +10.5 -> +2.0 bps, DD 28.6% (same); H48 fails; 1h rich L1 hit 56.4%, +3.6 bps, DD 9.2%, positive through 2 bps | **changed:** no 1m pick passes with an honest fill, so the "surviving +0.004 AUC sliver" is gone; H38's "below null95 after honest fills" becomes "above null95, CI over 0" |
| **H62** frozen hourly candidate (stops 2024-06) | frozen rich logreg 730 d, 4 h, top 10%; by-year table | verified clean: `fwdlib.load_bars` stops at 2024-07-01 and the cache rebuilt from the cut file is bit-identical (t, anch, open, close, tb7, rich); choice, grid and by-year table unchanged | clean (same) |
| **H63** calendar look | hour 01 UTC +2.6 bps, hit 49.7%, monthly CI [-106,+261] | hour 01 UTC (same Holm pick on selection, +6.71 [2.7,10.7]) +3.91 bps, hit 51.3%, 384 trades, monthly CI [-91,+322] (13 mo) | FAIL -> FAIL (same) |
| **H64** flow look | 1m logistic tb7+flow - tb7 = -0.0003, 7/15 months positive; flow alone 0.536-0.542; 2025 flow-alone 0.515-0.529 | -0.0003 [-0.0019,+0.0014], 7/13 positive; 2025 (Jan to Jul 20) flow-alone 0.515-0.531 | same |

H58 detail: `equity/summary.txt`, `nextopen/h58_side_by_side.txt` (zero cost to 5 bps, null bands, monthly hits) and the PNGs beside them. Old tables: `recut/old/`.
One catch in the logged H58 sentence "every zero-cost cumulative return lies at or below the random-side null's 95th percentile": the old
table itself has three of four above it (H48 +42.8 vs 32.0, 1h L1 +44.5 vs 29.6, 1m L1 +37.2 vs 34.5). Recut: H48 +35.9 vs 30.4 and 1h L1 +37.3 vs 28.4
above; 1m L1 +30.8 vs 35.0 and the hourly barrier +47.1 vs 58.7 below. That does not change "none meets the goal".

## What changed in substance

1. **H56 and H61's last survivor.** With the protected two months removed, the small 1m all-feature L1 edge over tb7 is +0.0025 and its interval
   touches 0. The L1 pick no longer passes at 95% under either fill. The boosting pick passes at 95% with the close entry (lower bound +0.00003) and
   fails with the next-open fill. So the recut leaves **no** direction effect that survives an honest fill and a 95% interval.
2. **H57 at 240 minutes.** The only horizon that passed both tests no longer does (Spearman lower bound +0.0001 or -0.0009, QLIKE lower bound
   -0.004 or -0.005, by GPU run). All three horizons now fail; the ensemble's +0.005..+0.007 Spearman gain stays.
3. **H38 with next-open fill** now beats its null95 (+6.95 vs 6.01); the old text said below. It still does not clear zero over months, and the
   equity edge is +2.0 bps per trade.
4. All other verdicts are unchanged. H48's pass stands (+0.54 bps, tiny); H60's fail stands.

## Could not recompute, or limits

- H40 part 1 and every journal item not named were not touched.
- Selection stages of H56, H57, H64 and the hourly dev stage were reused or rerun on data entirely before 2024-07 (evidence above). H57's nets are
  GPU-nondeterministic: its old look cannot be reproduced bit for bit, so two independent looks on the cut data are reported.
- H56 L1: `wf._fit` was seeded after the old 1m look was stored (the journal notes run-to-run solver noise); the recomputed numbers use the seeded code.
- The 13th look month (2025-07) has 20 days; 2025-08 and 2025-09 are gone. Interval widths use 12 degrees of freedom instead of 14.
- H62's grid was not rerun (the cache is identical, so the grid is). H52 has no look table beyond the Q4 lines in `recut/decay/report.log`.
- Off-repo: caches in `D:/nt/nt_recut_cache/`, the cut CSV in `D:/nt/nt_recut_data/` (both git-ignored or outside the repo).

## Where things are

`hourly` -> `recut/hourly/final.jsonl`; `regimes/results_recut.jsonl`, `part2_recut.log`; `geom2/holdout_min_recut.json`, `holdout_min_nextopen_recut.json`;
`recut/lab/decay*.json`, `recut/decay/` (H52 q1-q3, report.log); `recent/look.json`, `look_table.txt`, `look_*.jsonl`; `nextopen/h56_close`, `h56_nextopen`,
`h56_side_by_side.txt`, `h58_side_by_side.txt`, `*_nextopen.json`; `volnet/look_recut.json`, `look_tables_recut.txt`, `look_tables_recut_rep.txt`;
`equity/*.json`, `summary.txt`, PNGs; `calendar/results_recut.*`; `flow/look_recut_*.jsonl`, `look_table_recut.txt`; `recut/check_h62.json`,
`recut/cache_equal_*.json`. The chain scripts `recut/run_*.sh` give the order of every run.
