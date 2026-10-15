# T11 report: correct clocks, and volatility seasonality in UTC and New York time

The lead wrote this from the implementer's hand-back (2026-10-15).
- **SPEC:** [SPEC.md](SPEC.md), commit 0a55dac.
- **Commits:** helper, selection and `selection.json` in a3ebba2, before the looks; report-only looks, freeze and figures in 85992c1.
- **Tests:** 20 passed.

## Part 1: the helper, `runs/tactical/common/timefix.py`

- **`to_utc`**
  - Reproduces T09's `utc_hour` exactly, around every DST change and the splice.
  - Offsets: +4 h in summer and +5 h in winter until 2024-09-13 08:00, then 0.
  - Against Binance 2025-06 the 1-minute returns correlate 0.921 at lag 0, and about 0 at the other lags.
- **`ny_time`** matches zoneinfo on 4,000 random instants.
- **`seam_mask`** marks the DST and splice seams. The test checks the 109 bps jump on 2018-03-11 and the 51 bps jump at the splice.
- **The intraday peak in UTC is at 13-15 h,** the US open. In the old file time it showed at 17-20 h, late and smeared.

All later tactical work that uses time of day should go through this helper.

## Part 2: selection (2019-01..2024-06, next-hour change in log RV, skill vs AR)

| Variant | Gain over V0 [95% CI] | Months positive |
|---|---|---|
| V1, S3 with UTC hour | +0.0004 [+0.0000, +0.0007] | 44/66 |
| V2, S3 + New York basis (14 terms) | +0.0098 [+0.0077, +0.0119] | 57/66 |
| **V3, S3 + 96 UTC 15-minute slots (chosen by the rule)** | **+0.0199 [+0.0165, +0.0233]** | 61/66 |

- **V1 adds almost nothing.** The phase-free harmonics absorb a constant 4-5 h shift, as the research verifier predicted.
- **The new calendar terms add real skill.** V2 and V3 contain them.
- **The choice was V3, by the rule fixed in advance:** the highest mean gain over V0.

## Report only (nothing chosen from these)

| Period | V2 − V0 | V3 − V0 |
|---|---|---|
| Binance 2025-11..2026-09, monthly refit | +0.047 [+0.033, +0.061], 11/11 | +0.056 [+0.034, +0.078], 10/11 |
| Cut file 2017-09..2018-12 | -0.002 [-0.009, +0.006] | -0.003 [-0.011, +0.005] |

**Seasonality figure (fig1).**
- Before 2021-11 the intraday volatility curve is nearly flat.
- After 2021-11 it peaks at the US open (UTC 14, NY 9-10), with spikes at :00, :15, :30 and :45. Weekends peak on Sunday evening in NY.

**Reading.** The calendar effect is a feature of the recent market (from about 2021-22). It fits the Kraken study's dated
shift of November 2021. That explains why T04-A (2017-18) found nothing, while H83 and this look found a lot on 2025-26.

## Verdict: prospective, not yet

- **What is frozen:** V0, V1 and V3 (plus ar and rw), fitted on Binance 2025-10-02..2026-09-30.
  - `frozen_B.json` sha256 `6d62faaf34b1cb930748d4988fb227f99a4b5ba693d62586e66372a173f08e32`.
  - It is tied to the sha of `selection.json`.
- **How it is scored:** `score_B.py` scores from 2026-10-02 at 3 and 6 complete months, with Pocock alpha 0.0294. The first look is possible after about 2027-01-01.
- **Criterion:** V3 − V0 ≥ +0.01 with the CI above 0, and at least 2 of 3 (or 5 of 6) months positive. The secondary is |r| over 15 minutes, with Holm.
- **V1 is identical to V0 on the forward file** (UTC either way), so V1 cannot pass there.

## Deviations

- **V2 has 14 terms, not the SPEC's 13:** the SPEC's own list sums to 14.
- **V2 and V3 sit on T01's S3,** which keeps its hour terms in file time. On the forward file that makes no difference.
- **"Monthly skill" means skill vs AR.**
- **Holm and Pocock details are the implementer's interpretation.** Review them before the first look.
- **The seam pad is 61 minutes, so it does not cover S3's 1,440-bar windows.** A pad of 1,500 minutes would be cleaner.
- **`score_B` treats a month as complete once its last 240 bars exist.**
