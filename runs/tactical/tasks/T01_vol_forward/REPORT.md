# T01 REPORT: forward test of the next-hour volatility-change forecast (H72)

Run on branch `nt-tactical-t01` from 34b40f4 (SPEC committed before any number). CPU only, no new packages. Run time 9 s for the whole
study (data load 7 s); `run.log`, `tables.txt`, `results.json`, `tests.log` (6 passed in about 5 s) in this folder.
Code: `t01_lib.py` (loaders with bound asserts, features/targets copied from `gaps/study3_vov.build`, frozen fit, monthly scoring),
`run_t01.py`, `test_t01.py`. Training data `D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv` (loaded 2024-06-01..2025-07-20 23:59), forward
data `D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv` (loaded 2025-10-01..2026-10-02; months 2025-10..2026-09 scored). No bar
2025-07-21..2025-09-30 is read (asserted in the loader, tested). The forward file has no gap (0 non-1-minute steps).

## Verdict: YES (pre-registered criterion met), arm F, S3, target A

| criterion (SPEC) | value | met |
|---|---|---|
| 1. mean monthly skill vs `ar` >= 0.02 | +0.1419 | yes |
| 2. 95% t-CI lower bound > 0 | [+0.1250, +0.1588] | yes |
| 3. mean monthly R2 vs `rw` > 0 | +0.1947 [+0.1763, +0.2130] | yes |

S3 skill vs ar is positive in 12 of 12 months (min +0.0684 in 2026-08, max +0.1716 in 2026-02). The secondary S1 (source-robust
features) also passes: skill +0.1295 [+0.1140, +0.1450], R2 vs rw +0.1831, 12 of 12 months positive. So the pass is not an artifact
of the volume/range extras shifting between feeds.

## Key numbers (target A; mean over scored months, 95% t-CI over months)

| arm / model | skill vs ar | R2 vs rw | Spearman |
|---|---|---|---|
| H72 look (2024-07..2025-07), S3 | +0.1633 [+0.1466, +0.1801] | +0.2047 | n/a here |
| **F, S3 (primary)** | **+0.1419 [+0.1250, +0.1588]** | **+0.1947 [+0.1763, +0.2130]** | **+0.4475 [+0.4323, +0.4626]** |
| F, S2 | +0.1381 [+0.1198, +0.1565] | +0.1911 | +0.4459 |
| **F, S1** | **+0.1295 [+0.1140, +0.1450]** | +0.1831 [+0.1680, +0.1983] | +0.4240 [+0.4084, +0.4396] |
| F, ar (baseline) | 0 | +0.0615 | +0.2378 |
| F, arlvl (descriptive baseline) | +0.0336 [+0.0122, +0.0549] | +0.0931 | +0.3381 |
| R (refit), S3 | +0.1778 [+0.1633, +0.1922] | +0.2278 [+0.2135, +0.2421] | +0.4812 [+0.4662, +0.4963] |
| R, S2 | +0.1775 [+0.1627, +0.1924] | +0.2276 | +0.4813 |
| R, S1 | +0.1271 [+0.1085, +0.1456] | +0.1802 | +0.4317 |

The frozen forward skill (+0.142) is 0.021 below the look (+0.163), about 13% smaller; the intervals overlap only barely
([+0.125, +0.159] against [+0.147, +0.180]), so there is a modest loss out of sample and out of source, not a collapse. Arm R (refit on the
forward file itself, 11 months 2025-11..2026-09, expanding window from 6,138 rows in November to 68,351 in September) is higher,
+0.178, above the look: the model fitted on the new feed's own scale does better than the frozen cross-source fit; the S1/S2/S3
ordering is the same. Arm R is secondary and not a verdict. Its S1 (+0.127) is almost equal to arm F's S1 (+0.130), consistent with
the extras (volume/range) being the part that shifts between feeds, while the log-RV features transfer.

Target B (4-hour vol-of-vol, descriptive only; it failed its threshold in H72): arm F S3 skill vs ar +0.0413 [+0.0313, +0.0514], R2 vs
rw +0.5015, Spearman +0.2129; arm R S3 +0.0548 [+0.0348, +0.0748]. In H72 the look value was +0.0111 [+0.0022, +0.0199], so B is
now above 0.02 with a positive interval on the forward months. B was not pre-registered for a verdict here and no claim is made.

Reading note: the Spearman of `rw` for target A is undefined (a constant prediction); the table shows it as +nan/0. R2 and skill of
`rw` vs `ar` are -0.066 (the AR baseline beats the random walk).

## Per-month counts and tables
Per-month MSE, R2 vs rw, Spearman, skill vs ar, n_te and dropped rows for both arms and every model: `tables.txt` (rows) and
`results.json`. Scored rows per month are 5,760-6,377 (stride 7). Dropped rows (a zero-RV window): 0 in 2025-10, 2025-11, 2026-02, 2026-03, 2026-06 and a few (for example 10 in 2025-12) in the other months (see
`tables.txt` for each month). Training rows for arm F: 74,652 (A and B), first row 2024-07-21 00:01.

## Deviations from the SPEC (none changes a criterion)
1. The last 240 bars of the training file are never anchors, because `volnet_lib.Data` builds anchors only up to n - max(horizons) (H72's
   own construction; it also yields H72's training rows). So the frozen fit's last training label ends 2025-07-20 20:56 (A) and 23:56
   (B), not at 23:59; this is within "at or before the cutoff".
2. Forward warm-up is realised as `first = 1,620` (1,440 + 180 bars) for the forward Data: the first anchor is 2025-10-02 03:00. Anchors in
   October therefore start on the 2nd; October's row count (6,146) is still near the other months.
3. Arm R's first month (2025-11) trains on about 29 days (6,138 rows), fewer than H72's 365-day windows; reported as is.
4. `volnet_lib` inserts `D:/nt/nt_recut/runs/tactical/common` on its path to import `cutoff.py` (the data file it names is the cut copy
   and the module only checks the cut file exists); the study's own loaders do not use it. The original main file is never touched.
5. The 95% interval and "months positive" are over months with a t-interval as in H72; a sign-flip or block check was not added (not in the SPEC).

## Tests (`test_t01.py`, 6 passed)
bars of the forbidden gap are refused/never returned; real files respect their bounds; frozen training labels end by the cutoff and
start no earlier than 365 days; forward features and the AR regressor at an anchor are unchanged when every bar from the anchor on is
scrambled (and the target does change, so the test can fail); the warm-up; the frozen arm's coefficients are bit-identical after
scoring several months.

## Limits
Cross-source: a pass means H72's signal holds on a different feed (Binance spot) and on unseen months, which is the stronger claim; a
drop of 0.02 may be source or time and cannot be separated here. One look, one model family (ridge), 12 months with strong
overlap in features across neighbours (CI over months treats them as independent, as H72 did). Describes predictability of realised
volatility change, not a trading edge.
