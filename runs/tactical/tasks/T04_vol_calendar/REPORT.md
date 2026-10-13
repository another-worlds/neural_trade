# T04 report: the 15-minute calendar for the next-hour volatility change

The lead wrote this from the implementer's hand-back (2026-10-13).
- SPEC: [SPEC.md](SPEC.md), commit 44441a5.
- Code, results and the freeze: dc83031.
- Tests: 9 passed in 4.5 s.
- Run times: Part A 10 s, freeze about 1 min.

## Part A (retrospective, 2017-09..2018-12, 16 months): FAIL

Paired monthly skill, measured against `ar`:

| a - b | Mean | 95% CI | Months positive |
|---|---|---|---|
| **S3C - S3 (criterion A)** | **-0.0031** | [-0.0107, +0.0044] | **7/16** |
| S3 - HAR levels | +0.0186 | [+0.0102, +0.0269] | 13/16 |
| S3 - ar | +0.1837 | [+0.1693, +0.1982] | 16/16 |
| C (slots + ar regressor) - ar | -0.0038 | [-0.0162, +0.0087] | 8/16 |

- **The calendar result does not replicate.** The +0.076 found on the forward months is absent in 2017-18, and the slots alone have no skill. Some of the gap is the 96 extra columns overfitting the short early training spans (14-105 days). From 2018-07 on, S3C is slightly above S3 in 8 of 12 months. But the slot-only model is flat, so the calendar pattern looks like a feature of the recent market, not a constant one.
- **S3's gain beyond HAR levels replicates** on this earlier, untouched history: +0.019 [+0.010, +0.027]. H83 found +0.030 on the forward months.

## Part B (prospective): frozen, not scored

Models S3, S3C, C, HAR levels, `ar` and `rw` were fitted once on the Binance file, 2025-10-02..2026-09-30. That is 74,474 rows, with 346 dropped by the feature-RV floor.

| Freeze artefact | Value |
|---|---|
| File | `frozen_B.json` |
| Internal sha256 | `61749ad6d52d4faaf06066e39439c28110a6db2520d0e568c1322dd1cc5e80db` |
| sha256 of the file bytes | `e77eec63...e284b48` |

`score_B.py` checks the hash and scores with no refit. It gives a verdict only once at least 3 complete months from 2026-10 exist. **To run it in a later session:** download the new months into the forward file, then `python score_B.py`.

## Readings

- The SPEC's "paired skill of S3C over S3" was read as the difference of the monthly skills against `ar`. 1 - MSE_S3C/MSE_S3 gives the same verdict: -0.0037 [-0.0125, +0.0051], 7/16 months.
- HAR levels: a ridge on the four log-RV levels, with no hour terms.
- All 225 drops come from the feature-RV floor, not from zero-RV targets.
