# T15 Day 2 note: falsification and controls (written BEFORE any Day 2 run; Day 1 is committed at 124cbc5f)

SPEC.md unchanged. Day 1 correction: Look 1 has 13 scored months (2024-07..2025-07) and Look 2 has 11 (2025-11..2026-09); "pooled" always means these 24 months
(the Day 1 note wrote "12 + 11" in one place, a typo). Same machine rules as Day 1 (CPU, below-normal, BLAS 4 / GBM 4 threads, guard before every job, progress.md).
Harness `core/` unchanged and imported; the T15 driver only (i) builds view matrices in memory, (ii) calls `evaluate.evaluate_look`, (iii) for the pool control rebinds
`evaluate.train_mask` inside its own process to restrict the training rows to a source. Nothing in T14 is written.

## Modules under test
- **M9h6** = the frozen compact module `M9h6_prof_str` (29 columns; rebuilt as `T15_h6`, which Day 1 showed reproduces the stored monthly series to 0.0).
- **M7c** = the frozen `M7c_gbm_resid` (LightGBM, 12 keys, fitted to the V3 / H3 walk-forward residual; `m7_lib.monthly_view`, parameters unchanged).
- **M7_gbm** = the pre-registered pooled GBM (12 keys -> labels), run as a reference and a diagnostic: it is NOT a frozen artefact and is not eligible for Day 3's recommendation.
Each rebuilt module with no control applied must reproduce its stored monthly series (max abs difference < 1e-9); reported as the reproduction check.

## Controls (all through the harness: Look 1 and Look 2 months as in T14, n_perm 5, n_boot 10000)
**C-H hour shuffle.** The UTC hour is permuted across the `ok` anchors inside each UTC calendar month (pool rows and test rows alike; weekend flag and every other key unchanged),
10 seeds (3000..3009). M9h6: the harmonic and weekend-by-harmonic columns are recomputed from the permuted hour (the stretch terms and `wkend` stay). M7c / M7_gbm: the key
columns `hsin`, `hcos` are permuted jointly (one permutation, so a row keeps a consistent hour), the GBM is refitted walk-forward on the permuted keys.
**C-W weekend shuffle.** Same with the weekend flag permuted across the `ok` anchors of each month (hour unchanged; key column `wkend`; for M9h6 the `wkend` column and the
weekend-by-harmonic columns), 10 seeds (4000..4009).
**C-P pool sensitivity** (the pool of the model that is fitted to produce the view's prediction; Look 2 months only, because Look 1 has a single source, Bitstamp):
 M7c / M7_gbm: the GBM pool of month m is restricted to Bitstamp rows (`cut`), Binance rows (`fwd`) or both (default), same 365-day / label-ended / MIN_POOL 5000 rules; a month
 the pool cannot serve is not scored. M9h6: the view has no model of its own, so the pool is the training set of the harness' ridge (V3 and V3 + view alike): `cut` = Bitstamp rows of the
 365 days before the month, `fwd` = Binance rows from 2025-10-02 (the T14 default; must equal the stored Look 2 result), `both` = both sources; run on the concatenation of the two
 base tables with `train_mask` restricted to the pool, test rows are the month's Binance rows.
**C-S start-date sensitivity.** Training start shifted by +6, +12, +18 months from the look's default start (Look 1: 2023-07-01; Look 2: 2025-10-02) wherever any month is still scored
 (Look 2 only +6 can; the harness skips a month with fewer than 300 training rows, the GBM with fewer than 5000). For M7c / M7_gbm the GBM pool start moves too; the residual targets of
 M7c are held at their default definition. The V3 / H3 baselines are refitted from the same start inside the harness, so each increment is paired by month.

## Pre-registered rules (fixed here)
Series: the monthly gain series of the harness (A: increment of skill over V3; B: relative Brier improvement over H3 per month, `monthly_rel`). All CIs are 95% two-sided t-intervals over
months. A target T in {A, B} of a module **survives** a control as follows:
- C-H, C-W: the paired monthly difference real minus the mean over the 10 seeds of the shuffled series (months matched by name, pooled 24 months) has a CI whose lower bound is above 0.
 (Reported too: the shuffled gain itself, per seed pooled means, and the share of the real gain the shuffled view keeps.)
- C-P, per pool variant: the variant's increment series on the Look 2 months it scores has a CI above 0. A variant with fewer than 5 scored months is "not testable" (neither survives nor fails).
- C-S, per shift: the increment series pooled over the months scored under the shift (Look 1 and Look 2 together) has a CI above 0; fewer than 5 months: "not testable".
A module-target **passes Day 2** iff it survives C-H, C-W and every testable C-P and C-S variant. A module **passes Day 2** iff A or B passes (the parent SPEC's "either target"). No threshold on the
size of the gain is added (the T14 minimum effects +0.005 / +1% are shown, not applied). Paired differences against the default pool (variant minus default) are reported as description.
No multiplicity correction. Nothing is chosen on a look except by Day 3's pre-registered rule, which reads this file's outcome.

## Day 3 rule (restated so that it is fixed before any result)
Eligible: the frozen artefacts M7c and M9h6 that pass Day 2. The recommended module is the eligible one with the highest min(R_A, R_B) (R = pooled-24 mean gain over M7_gbm's pooled gain, as in M9), ties
(difference < 0.005) to the fewer columns (M7c has 2, M9h6 29). If none passes: recommend none. The alternative reading "min over the four target x look ratios" is computed and shown; if the two readings
differ in the choice, both are shown and the primary (M9's min(R_A, R_B) pooled) decides.
