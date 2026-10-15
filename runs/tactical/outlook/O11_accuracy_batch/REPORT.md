# O11 report: accuracy gains on four untested promising hypotheses

SPEC `21bb2c1` + addendum (prospective criteria). Parts a-c: e476ee2 (tests 81c82cf); part d: 82bae56, 15d0f7c, e8f29a8 (choice alone),
fbd079e. CPU only. Written by the outlook lead from the implementers' text; the primary lines were checked against `part_*/tables.txt`.

## Verdicts (Holm over the four primaries)
| part | claim | J | effect | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|
| a | geometry + EWMA bank improve the forecast of the volatility CHANGE | 25 months | skill vs AR +0.0447 [+0.0361, +0.0533], 25/25 | 1.4e-10 | 5.4e-10 | **pass** |
| b | monthly refit beats the frozen fit (L1, H = 15, forward year) | 11 months | Spearman +0.0046 [+0.0024, +0.0069], 10/11 | 9.8e-4 | 2.0e-3 | **pass** (small) |
| c | quantile regression with geometry gives better tails | 24 months | pinball +0.431 bps [+0.339, +0.522], better in 0/24 | - | - | **fail** (worse, and the coverage band is missed) |
| d | 15-minute geometry or a 90-day window improves direction AUC | 25 months | -0.0015 [-0.0072, +0.0042], 12/25 | 0.597 | 0.597 | **fail** |

## Reading
- **a (the main gain).** The volatility-change forecast improves by +0.045 skill over T01's S3 base, in every one of 25 months, and in 39 of 42 selection months.
  - **Where it comes from:** the EWMA bank adds +0.0325 and geometry adds +0.0122 [+0.0072, +0.0172] over the bank (20/25 months). The bank is a plain denser HAR, so most of the gain is not geometry.
  - **Not pre-registered:** the split above.
  - **Where it does not help:** the 4-hour vol-of-vol target gets worse (-0.0157, 4/25 months).
  - **Check:** the S3 base reproduces T01's published forward skill (+0.1430 against +0.1419).
- **b.** A refit helps in the same direction in every cell: H = 15 and 60, S3 and L1, and the sensitivity arm (+0.0039). QLIKE is lower for the refit everywhere. The effect is about half a Spearman point per month. It is real but small.
- **c.** The monthly empirical recalibration of O7 is hard to beat. A quantile regression with only ln sigma_hat is already worse (+0.194 bps, 23/24 months), and the geometry features add overfitting (hit rates 0.063 and 0.937 against the target 0.05 and 0.95).
- **d.** The +0.0155 AUC gain from G15 on the selection span (38/42 months) was a property of that span. On J it is gone. The 90-day window is significantly worse. The base direction model keeps AUC about 0.525 on J.

## What this adds up to
The accuracy of the forecast of SIZE and of CHANGE in volatility can still be improved (a, b), mostly by richer HAR-type memory, with a small geometry part.
The accuracy of DIRECTION and of the TAILS could not be improved (c, d). This is the same split as in O1-O10: volatility is where the signal is.

## Prospective (frozen, scored once, first 3 complete months after 2026-10)
- **a:** `part_a/frozen_P_a.json`, sha256 e2c7ac48c8b2c5b6fb650af1e717fde279ef7efbc1f754d34f8eae829f89e09e. Criterion: mean skill gain >= +0.010, 3/3 months positive.
- **b:** `part_b/frozen_P_b.json`, sha256 7219a4285326e2a46971bf82d8b6df160b150a8e675c3f8e3770c038c81af117. Criterion: mean gain >= +0.003, 3/3 months positive.
- The scoring scripts refuse today and were never run on data. The criteria are ratified in the SPEC addendum.

## Caveats and deviations
- **J is a second look.** B and C were read by O1, O3, T01 and O8-O10 for other claims.
- **Part a (not formed on J):** the candidate is O5's L1 feature set. T01's S3 is the base.
- **Part b window ambiguity:**
  - "latest 365 days available" is ambiguous. The primary `refit_cal` is the 365 calendar days before the month, which gives about 59k rows in the early months against 74.6k for the frozen fit.
  - The sensitivity arm `refit_avail` moves the window back by the 72-day gap, and its result is the same in sign.
- **Part c:**
  - the base uses the previous month's fit for sigma, so the candidate's sigma is fresher; the base with a refreshed sigma was not run;
  - J has 24 months because 2025-10 has no previous month for the calibration;
  - "G1 top-15" is chosen by the largest standardised coefficient of that month's L1 ridge.
- **Part d:**
  - scoring rows are the strategy-valid decisions with a non-zero return;
  - the selection table's t is against 0.5 (an earlier version printed it against 0, before the choice commit; the choice did not change);
  - the gated subset of d1 is larger than the base's, so its hit rate is not like-for-like.
- **Forward warm-up:** the first anchor of the forward file is 2025-10-05, so 2025-10 is partial.

## Files
`part_a|b|c|d/` (code, `results.jsonl`, `tables.txt`, freezes and scorers for a and b), `summary.txt`, `tests_vol.log`, `part_d/tests_dir.log`, `part_d/choice_d.json`.
