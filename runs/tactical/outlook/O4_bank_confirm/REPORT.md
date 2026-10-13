# O4 report: confirm the EWMA-bank volatility forecast on 2017-09 .. 2018-12

SPEC: `SPEC.md` (d8712e6). Code and results: dea26d6 (`o4_lib.py` on read-only `o3_lib` at 5e4ab42, `run_o4.py`,
`make_frozen_B.py`, `score_B.py`, `test_o4.py`: 7 passed). CPU: Part A 51 s, freeze 13 s. Written by the outlook lead from the
implementer's results (its Write tool refused the file); key numbers checked against `tablesA.txt`.

## Verdict A: not confirmed
At H = 60, for BANK - R0 over the 16 months, criterion A has three parts:
1. **Not met.** The mean Spearman gain must be >= +0.005 with a lower bound > 0. It was -0.0043 [-0.0266, +0.0181].
2. **Met.** At least 11 of 16 months must be positive. 14/16 were.
3. **Met.** The QLIKE difference must be <= 0. It was -0.0007 [-0.0110, +0.0095].

| H | R0 Spearman / QLIKE | BANK Spearman / QLIKE | BANK - R0 Spearman, months positive | BANK - R0 QLIKE | Holm (15/240) |
|---|---|---|---|---|---|
| 15 | 0.6143 / 0.3968 | 0.6191 / 0.3888 | +0.0047 [-0.0057, +0.0152], 15/16 | -0.0080 [-0.0183, +0.0023] | adj p 0.686 |
| 60 | 0.6447 / 0.2629 | 0.6404 / 0.2622 | -0.0043 [-0.0266, +0.0181], 14/16 | -0.0007 [-0.0110, +0.0095] | primary |
| 240 | 0.6172 / 0.2104 | 0.5966 / 0.2164 | -0.0206 [-0.0654, +0.0242], 7/16 | +0.0060 [-0.0136, +0.0256] | adj p 0.686 |

Per-month gain at H = 60 (Spearman / QLIKE, training days):

| Month | Days | Gain | Month | Days | Gain |
|---|---|---|---|---|---|
| 2017-09 | 10.7 | **-0.1603 / +0.0682** | 2018-05 | 252.7 | +0.0152 / -0.0029 |
| 2017-10 | 40.7 | +0.0067 / -0.0023 | 2018-06 | 283.7 | +0.0078 / -0.0034 |
| 2017-11 | 71.7 | +0.0052 / -0.0066 | 2018-07 | 313.7 | +0.0021 / +0.0020 |
| 2017-12 | 101.7 | +0.0064 / -0.0033 | 2018-08 | 344.7 | +0.0094 / -0.0078 |
| 2018-01 | 132.7 | +0.0030 / -0.0024 | 2018-09 | 365 | +0.0099 / -0.0034 |
| 2018-02 | 163.7 | +0.0023 / -0.0025 | 2018-10 | 365 | +0.0185 / -0.0216 |
| 2018-03 | 191.7 | +0.0025 / -0.0027 | 2018-11 | 365 | +0.0023 / -0.0153 |
| 2018-04 | 222.7 | +0.0050 / -0.0019 | 2018-12 | 365 | -0.0042 / -0.0056 |

## What the failure is, and the lead's own error
- **One month sinks the mean.** In 2017-09 the 52-feature BANK ridge trained on 10.7 days (2,199 rows) and lost 0.160 Spearman (0.334 at H = 240).
- **The SPEC allowed it.** The short-history rule was copied from T04 (a 3-regressor target, where 10 days suffice) without asking whether 52 features can be fit on 10 days. That was a design error of the lead, made before the results; it is not repaired after them.
- **Post hoc only (not a verdict).** Without 2017-09:
  - H = 60 is +0.0061 [+0.0030, +0.0093], 14/15, QLIKE -0.0053 [-0.0086, -0.0021];
  - H = 15 is +0.0095, 15/15;
  - H = 240 is +0.0003, 7/15.

  That is the same pattern as O3 (a gain at 15 and 60 minutes, none at 240), but dropping a month after seeing it is a choice made on this data.
- **The verdict stays "not confirmed".** The clean test of BANK is now Part B (a full year of training, unseen months).

## Part B (frozen, not scored)
- **Artefact:** `frozen_B.json`, sha256 `83dd3ad75f78bc74d8271d13dee87b44be39956825ae760f7158f8fe13a1e4f7`.
- **Fit:** once per H on Binance bars 2025-10-01 .. 2026-09-30 23:59. The first anchor is 2025-10-05 06:24 (73,825 rows). No bar after 2026-09-30 was read.
- **Scoring:** `score_B.py` (not run) scores the first 3 complete months from 2026-10. It refuses fewer months and any earlier month. Criterion B is criterion A at H = 60 with 3 of 3 months positive.

## Deviations
1. Part A loads bars to 2019-01-01 03:59, for the label tail only.
2. The freeze trains from the first warm-up anchor (2025-10-05).
3. `score_B.py` scores only the first 3 complete months; the SPEC was silent on this.
4. A harmless divide-by-zero warning comes from rows that the ok mask excludes.
5. Extra files: `summaryA.json`, `freeze.log`, `frozen_B.sha256`.
