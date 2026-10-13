# O6 report: does hourly-bar geometry (G60) help at H = 240?

SPEC: `SPEC.md` (62764b2). Code and results: 28f724f. CPU only, 2,155 s. Numbers: `tables.txt`; rows: `results.jsonl` (711); tests: `tests.log` (8 passed; run on the code committed in 28f724f). Written by the outlook lead from the implementer's text (its Write tool refused the file); the criterion line checked against `tables.txt`. A negative result: lead-verified, no QA agent (D-060).

## Verdict: **fail**
The criterion compares L2 (L1 + G60) against L1 at H = 240, over 24 judgement months (2019-01..2020-12):

| | measured | needed | met |
|---|---|---|---|
| Spearman gain | -0.0143 [-0.0231, -0.0055] | >= +0.003, lower bound > 0 | no |
| months positive | 8/24 | >= 16 | no |
| QLIKE difference | -0.0007 [-0.0094, +0.0079] | <= 0 | on the mean only |

**Sanity gate:** L2sh - L1 = -0.0009 [-0.0013, -0.0006], inside +-0.003. So the verdict is "fail", not "broken".

| H | L1 Spearman / QLIKE | L2 Spearman / QLIKE | L2sh Spearman / QLIKE |
|---|---|---|---|
| 15 | 0.6664 / 0.3407 | 0.6654 / 0.3377 | 0.6654 / 0.3414 |
| 60 | 0.6814 / 0.2888 | 0.6774 / 0.2851 | 0.6803 / 0.2894 |
| 240 | 0.6280 / 0.2511 | 0.6137 / 0.2504 | 0.6270 / 0.2515 |

## Secondary results (no verdict)
- **Other horizons (L2 - L1, Holm):**
  - H = 15: -0.0010 (6/24);
  - H = 60: -0.0039 [-0.0068, -0.0010] (8/24), Holm p 0.021.
- **Spent spans (descriptive):** the sign is negative on every span and every horizon. At H = 240:
  - 2021-01..2024-06: -0.0085 [-0.0123, -0.0048];
  - 2024-07..2025-07: -0.0049 [-0.0089, -0.0008].

## Reading
**The geometry ladder stops at the 15-minute rung.** Hourly geometry hurts:
- at H = 240 by 0.014 Spearman;
- at H = 60 by 0.004.

The frozen-fit coefficients are large and have opposite signs on collinear hourly EMA slopes (ema20.slope3 -1.14, ema10.slope10 +0.60, ema40.slope3 +0.56, ...). That is a fit to non-stationary structure, not a readable signal.

The shuffled control costs only 0.0009. So the loss comes from the real hourly features, not from the number of columns. This is the lead's interpretation and was not tested.

Practical consequence: the volatility model stays at G1 + G15 (O5's L1). H = 240 needs something other than geometry.

## Freeze
- **File:** `frozen_B.json`, sha256 `20d43f968293df67e4cbaf377da74975ba89f2d66d72c881c6fa1d962a3ac1a4`.
- **Fit:** L1 and L2 on Binance 2025-10-05..2026-09-30.
- **Scoring:** `score_B.py` is not run yet. It is mostly moot after this fail, but stays for completeness.

## Deviations
1. The frame is rebuilt from the whole cut file up to 2025-07-20 (cache at D:/nt/nt_scratch/o6/). L1 equals O5's L1 to 1e-9 on 2019-03 and 2020-06 at all horizons (tested).
2. L2sh uses seeded permutations of the G60 block only, with train and test permuted independently.
3. `o6_lib` removes O5's folder from `sys.path` after importing, to avoid a name clash between the two `score_B.py` files.
4. Windows are defined by bar position. The file has one non-1-minute step, as in O5.
