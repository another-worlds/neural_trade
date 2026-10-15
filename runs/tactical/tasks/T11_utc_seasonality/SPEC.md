# T11: correct clocks: a UTC helper, and volatility seasonality in UTC and New York time

Lead, 2026-10-15. Committed BEFORE any result.

- **Owner:** "запускай". The task is hypothesis 1.1 of
  [research/2026-10-15-accuracy-hypotheses](../../research/2026-10-15-accuracy-hypotheses/README.md).
- **The bug:** the Bitstamp file's hours are file time (UTC+4/+5 until the 2024-09-13 splice, H91). Several tactical
  libraries read the hour from file time (`t04_lib.py`, `volnet/volnet_lib.py`).

## Part 1: the helper (`runs/tactical/common/timefix.py`)

| Function | What it does |
|---|---|
| `to_utc(ts, source)` | Converts file timestamps to UTC: minus 4 h in US daylight saving (2nd Sunday of March to 1st Sunday of November, by date), minus 5 h otherwise, 0 from 2024-09-13 08:00 file time on. The Binance file is already UTC. |
| `seam_mask(ts, pad_min)` | Marks every DST-change day and ±2 h around the splice, padded by `pad_min` minutes. |
| `ny_time(ts_utc)` | Returns New York local time (DST-aware, the same date rule). |

**Tests:**
- known seams: 2018-03-11 08:00 file time (109 bps jump) and 2024-09-13 08:01;
- the splice offset;
- a Binance 2025-06 cross-check (lag 0 after the splice, using the downloaded `BTCUSDT_1m_2025-06.csv`);
- one known pre-splice offset (+4 in summer, +5 in winter), consistent with H91's 300/240-minute best lags.

It must reproduce `T09_regression_anatomy/lib.py:utc_hour` exactly.

## Part 2: the study

- **Target A** (next-hour change in log RV) as T01 / gaps study 3, stride 7, the same drops.
- **Secondary:** |r| over 15 minutes (Spearman, QLIKE), as T03.

**Variants** (ridge alpha 1, standardised):

| ID | Model | Calendar terms |
|---|---|---|
| V0 | S3 as T01 | hour sin/cos from FILE time |
| V1 | S3 | hour sin/cos from UTC |
| V2 (primary candidate) | S3 + a New York basis, 13 terms | NY time-of-day harmonics, 2 pairs (4); US cash open window 09:30-10:30 NY (1); US close window 15:30-16:30 NY (1); weekend in UTC (1); quarter-hour mark, minute of hour in {0, 15, 30, 45} ± 1 min (1); NY hour harmonics × weekend (4); minute-of-hour sin/cos (2) |
| V3 | S3 + 96 UTC 15-minute slots | the T04-B variant, for information only |

- Seam anchors are excluded everywhere: 0.57% of anchors, per T09.
- No external calendar (NYSE holidays, macro releases). That waits for the owner's D-050 answer.

## Periods

| Period | Data | Role |
|---|---|---|
| 2019-01..2024-06 | cut file, monthly walk-forward, 365 days | Selection, exploratory. H72 selected on the same months. One of V1, V2 or V3 is chosen by mean monthly skill over V0, and written to `selection.json`, committed before anything else. |
| 2017-09..2018-12 | cut file | Report only; it is partly seen (T04-A). |
| Binance 2025-11..2026-09 | forward file | Report only. These months were seen by H83 and showed the 96-slot effect. |
| From 2026-10-02 | forward file | **The verdict, prospective.** V0, V1, the chosen variant and V3 are fitted ONCE on the forward file 2025-10-02..2026-09-30 and frozen with a sha256 (`frozen_B.json`). `score_B.py` scores the months from 2026-10-02 (T01 read the first hours of 2026-10-01). Looks at 3 and 6 complete months, Pocock alpha 0.0294 for each look. |

## Pre-registered criterion (prospective, target A)

- The chosen variant minus V0 in monthly skill is ≥ +0.01.
- The Pocock CI is above 0.
- At least 2 of 3 months positive (or 5 of 6) are required.

**Secondary:** |r| at 15 minutes, Spearman ≥ +0.005 and QLIKE not worse. Holm over the two targets.

## Never
- No bar 2025-07-21..2025-09-30.
- Nothing chosen on the forward months or on 2017-18.
- The main file is not modified.

## Outputs
`runs/tactical/tasks/T11_utc_seasonality/`, plus `runs/tactical/common/timefix.py` with tests: code, `selection.json`,
`results.json`, `tables.txt`, `frozen_B.json`, `score_B.py`, and figures in the tactical_report style (volatility
seasonality by UTC hour, NY hour and minute of hour, before and after 2021-11). The report text goes to the lead.
