# O5 report: does indicator geometry add over the best volatility base?

SPEC: `SPEC.md` (eba7ddf). Code and results: 0ab6466. CPU only (about 17 min). Numbers: `tables.txt`; rows: `results.jsonl` (576);
tests: `tests.log` (6 passed). Written by the outlook lead from the implementer's text (its Write tool refused the file). The
criterion lines were checked against `tables.txt` lines 42-45.

## Base choice (selection months 2021-01..2024-06, H = 60)
| model | mean Spearman | mean QLIKE |
|---|---|---|
| S3 ridge | 0.7305 | 0.3324 |
| **L0 = BANK ridge** | **0.7342** | **0.3255** |
| N0 = HGB on S3 + bank | 0.7241 | 0.3430 |

B* = L0. The candidate is L1 = L0 + G (G1 + G15, 328 columns).

## Verdict: **geometry adds over the best base**
The criterion is L1 against L0 at H = 60, over 30 judgement months (2018-07..2020-12). It has three parts, all met:
1. The Spearman gain is +0.0056 [+0.0024, +0.0088]. It needs to be >= +0.003 with the lower bound above 0. **Met.**
2. 23 of 30 months are positive. It needs at least 20. **Met.**
3. The QLIKE difference is -0.0116 [-0.0150, -0.0082], better in 27 of 30 months. It needs to be <= 0. **Met.**

## Judgement span, H = 60
| S3 | L0 | L1 | N0 | N1 |
|---|---|---|---|---|
| 0.6565 \| 0.3267 | 0.6619 \| 0.3195 | 0.6675 \| 0.3079 | 0.6329 \| 0.3547 | 0.6416 \| 0.3434 |

| pair | Spearman | QLIKE |
|---|---|---|
| L1 - L0 (candidate) | +0.0056 [+0.0024, +0.0088], 23/30 | -0.0116 [-0.0150, -0.0082] |
| N1 - N0 | +0.0087 [+0.0036, +0.0138], 24/30 | -0.0113 [-0.0156, -0.0070] |
| L0 - S3 (BANK vs S3) | +0.0054 [+0.0030, +0.0077], 25/30 | -0.0072 [-0.0098, -0.0046] |

Notes on the judgement-span results:
- **By year:**

  | year | mean gain | 95% interval | months positive |
  |---|---|---|---|
  | 2018 | +0.0061 | [-0.0029, +0.0150] | 4/6 |
  | 2019 | +0.0041 | [-0.0024, +0.0105] | 9/12 |
  | 2020 | +0.0069 | [+0.0024, +0.0113] | 10/12 |

- **Weak stretch:** 2019-11..2020-02 were four negative months in a row.
- **Other horizons, L1 - L0:**
  - H = 15: +0.0122 [+0.0097, +0.0147], 30/30;
  - H = 240: +0.0004, 14/30 (Holm p 0.87).
- **BANK at full training length:** BANK beats S3 here (+0.0054 at H = 60, +0.0062 at H = 15). That answers O4's open question descriptively; it is not O4's verdict.
- **Boosted base:** HGB is worse than the ridge at every horizon on this span. The forward-span advantage of HGB seen in O1 does not appear here.

## Part B (frozen, not scored)
- **Artefact:** `frozen_B.json`, sha256 `79f7f25d725bc5638738bdb6d94ad1a96b3cd9870e7b99c0befb2d89b96fff24`.
- **Fit:** L0 and L1 were fitted once on Binance bars 2025-10-01..2026-09-30 (first anchor 2025-10-05 06:24).
- **Scoring:** `score_B.py` scores the first 3 complete months from 2026-10, once. The criterion is the same with 3/3 months positive.

## Caveats
- **Partial touch:** H47's 24 short lab slices touched these years, and H53 used them descriptively. Both are stated in the SPEC.
- **Margins:** the mean gain clears its bar by 0.0026, and the months count clears its bar by 3.
- **Shorter history:** 2018-07 trained on 313.7 days and 2018-08 on 344.7 days. The SPEC said at least 318 days, which was the lead's miscount of the warm-up.
- **No refit:** the earlier spans O1 and O3 used the same model definitions with no refit. Nothing was tuned between studies.

## Deviations
1. **First month history:** 2018-07 has 313.7 training days, not the SPEC's 318 (warm-up on top of the 2017-08-17 file start).
2. **First anchor:** `first = 6145` so the anchors share O3's stride-7 grid.
3. **Selection fits:** only S3, L0 and N0 at H = 60.
4. **N0 test:** N0 is tested through the shared HGB code path on S3 columns against O1's H0, to 1e-9.
5. **`score_B.py` months rule:** it requires 3 of 3 months positive, following O4.
6. **HGB freeze:** the HGB state is also frozen (base64 pickle), though it is not needed.
7. **Extra files:** `choice.json`, `summary.json`, `runtime.json`, `freeze.log`, `run_stdout.txt`.
